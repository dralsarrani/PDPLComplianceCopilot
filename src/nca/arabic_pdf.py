"""
Read the Arabic ECC-2:2024 PDF (data/nca/ECC-2-2024---NCA.pdf) into the same 108 controls.

The Arabic PDF stores its letters as shaped glyphs in visual order (right to left on the
page), and some two-letter ligatures (ما, ني, ير, ين ...) have no Unicode value: most PDF
tools drop them ("ضمان" comes out as "ض ن"). PyMuPDF gives each of those glyphs a
placeholder character of its own, so:
    1. every glyph is mapped back to its letters (NFKC for the shaped forms, LIGATURES
       for the placeholders),
    2. each printed line is read right to left, numbers and Latin words left to right,
    3. the table is cut into cells by its horizontal borders, as in build_ecc.py: the
       right column holds the control number (or الهدف / الضوابط), the rest is the text.
The controls come out in the same order as the English PDF, and build_ecc.py checks that
every Arabic control number matches its English one.
"""
import re
import unicodedata

import pymupdf

# placeholder glyph -> the letters it stands for (found by reading the PDF next to its text)
LIGATURES = {"˻": "ني", "˴": "ما", "˼": "ير", "˾": "ين", "ɬ": "بم", "˺": "نى", "˽": "يم",
             "ˮ": "لمي", "ɧ": "ئم", "˹": "نم", "ɵ": "ثر"}
# three words the PDF itself splits with a space
SPLIT_WORDS = {"وفًق ا": "وفقًا", "الأطرا ف": "الأطراف", "ا لإسناد": "الإسناد"}
LEFT_TO_RIGHT = re.compile(r"[٠-٩]+|[A-Za-z0-9][A-Za-z0-9 .,:/\-–_&()'’]*[A-Za-z0-9]|[A-Za-z0-9]")
MIRROR = str.maketrans("()[]“”", ")(][”“")
NUMBER = re.compile(r"^[٠-٩]+(-[٠-٩]+)+$")
WESTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
WORD_GAP = 2.0  # points between two glyphs that count as a space


def line_text(chars):
    """[(x0, x1, c)] of one printed line -> the text in reading order."""
    units, previous = [], None
    for x0, x1, c in sorted(chars):
        if previous is not None and x0 - previous > WORD_GAP and c != " " and units and units[-1] != " ":
            units.append(" ")
        units.append(LIGATURES.get(c, unicodedata.normalize("NFKC", c)))
        previous = x1
    units.reverse()  # right to left
    # a glyph that stands for several letters keeps their order: hide it while runs are flipped
    multi = iter([u for u in units if len(u) > 1])
    text = "".join("\x00" if len(u) > 1 else u for u in units)
    out, at = [], 0
    for match in LEFT_TO_RIGHT.finditer(text):  # numbers and Latin words read left to right
        out += [text[at:match.start()].translate(MIRROR), match.group(0)[::-1]]
        at = match.end()
    out.append(text[at:].translate(MIRROR))
    text = re.sub("\x00", lambda _: next(multi), "".join(out))
    return re.sub(r"\s+", " ", text.replace("", "")).strip()


def clean(text):
    text = re.sub(r"(?<=\S) (ني|ة)(?=[\s.,،:؛)]|$)", r"\1", text)  # a word ending split off
    text = re.sub(r"\s+([.,،:؛ً-ْ])", r"\1", text)      # space before punctuation or a vowel mark
    text = text.replace("اًء", "اءً")
    for wrong, right in SPLIT_WORDS.items():
        text = text.replace(wrong, right)
    return re.sub(r"\s+", " ", text).strip()


def page_lines(page):
    chars = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                for ch in span["chars"]:
                    x0, y0, x1, y1 = ch["bbox"]
                    chars.append((x0, x1, (y0 + y1) / 2, ch["c"]))
    lines = []
    for x0, x1, y, c in sorted(chars, key=lambda t: t[2]):
        if lines and abs(y - lines[-1]["y"]) < 3:
            lines[-1]["chars"].append((x0, x1, c))
        else:
            lines.append({"y": y, "chars": [(x0, x1, c)]})
    return lines


def page_cells(page):
    """[(label, text)] for each control-level cell of the page's table, top to bottom.
    Rows are cut at the borders of the right (label) column; sub-control rows have no
    border there, so they stay inside their control."""
    borders = [r[1] for d in page.get_drawings() for r in d["items"]
               if r[0] == "re" and r[1].height < 2 and r[1].width > 1]
    if not borders:
        return []
    right = max(r.x1 for r in borders)
    label_borders = [r for r in borders if abs(r.x1 - right) < 2 and r.width < 70]
    if not label_borders:
        return []
    label_x = min(r.x0 for r in label_borders) - 1
    ys = sorted({round(r.y0) for r in label_borders})
    cells = {}
    for line in page_lines(page):
        if not ys[0] - 25 < line["y"] < ys[-1] + 2:  # page header, footer (a subdomain title row can sit above the first border)
            continue
        label = line_text([c for c in line["chars"] if c[0] >= label_x])
        text = line_text([c for c in line["chars"] if c[0] < label_x])
        if line["y"] < ys[0] - 2 and not label:  # a main domain heading above the table
            continue
        cell = cells.setdefault(sum(1 for y in ys if y <= line["y"]), {"label": [], "text": []})
        if label:
            cell["label"].append(label)
        if text:
            if NUMBER.match(text) and cell["text"]:  # a sub-control number centred on its two lines
                cell["text"][-1] = f"{text} {cell['text'][-1]}"
            else:
                cell["text"].append(text)
    return [("".join(c["label"]).replace(" ", ""), clean(" ".join(c["text"]))) for _, c in sorted(cells.items())]


def parse_arabic(pdf_path):
    """[{id, subdomain, objective, text}] in the order of the PDF, ids as '2-10-3'."""
    controls, subdomain, objective, inside = [], "", "", None
    with pymupdf.open(pdf_path) as pdf:
        for page in pdf:
            cells = page_cells(page)
            if controls and not any(NUMBER.match(label) for label, _ in cells):
                break  # the appendices after the last control
            for label, text in cells:
                if NUMBER.match(label) and label.count("-") == 2:
                    controls.append({"id": label.translate(WESTERN), "subdomain": subdomain,
                                     "objective": objective, "text": text})
                    inside = "control"
                elif NUMBER.match(label) and label.count("-") == 1:
                    subdomain, objective, inside = f"{label.translate(WESTERN)} {text}", "", None
                elif label == "الهدف":
                    objective, inside = text, "objective"
                elif label == "الضوابط":
                    inside = None
                elif not label and text and inside == "control":  # continued from the page before
                    controls[-1]["text"] += " " + text
                elif not label and text and inside == "objective":
                    objective += " " + text
    for c in controls:
        c["objective"] = clean(c["objective"])
    return controls
