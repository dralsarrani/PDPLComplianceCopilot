"""
Show where the search loses the gold articles, one line per question.

For every answerable question in data/evaluation/gold_qa.json it prints the
rank of each gold article at two stages:
    hybrid  = BM25 + embeddings, same language (before the reranker; "-" = not in the top 50)
    final   = after the reranker, the order the app uses (top 20 candidates)
A gold article with hybrid rank above 20 never reaches the reranker, so the
reranker cannot fix it. The summary at the end counts both kinds of miss.

Run from the project root:
    python -m retrieval.diagnose_misses
"""
import json
from pathlib import Path

from retrieval import retriever

GOLD_PATH = Path(__file__).resolve().parent.parent / "data" / "evaluation" / "gold_qa.json"


def rank_of(ids, aid):
    return ids.index(aid) + 1 if aid in ids else None


def main():
    questions = [q for q in json.loads(GOLD_PATH.read_text(encoding="utf-8"))["questions"]
                 if q["gold_article_ids"]]
    not_candidate, reranked_out, found = 0, 0, 0
    for q in questions:
        hybrid = [a["record_id"] for a in retriever.retrieve(q["question"], k=50, rerank=False,
                                                              candidates=50, language=q["language"])]
        final = [a["record_id"] for a in retriever.retrieve(q["question"], k=20, language=q["language"])]
        parts = []
        for aid in q["gold_article_ids"]:
            h, f = rank_of(hybrid, aid), rank_of(final, aid)
            parts.append(f"{aid} hybrid {h or '-'} final {f or '-'}")
            if f and f <= 5:
                found += 1
            elif not h or h > retriever.CANDIDATES:
                not_candidate += 1
            else:
                reranked_out += 1
        mark = "OK  " if all(rank_of(final, aid) and rank_of(final, aid) <= 5 for aid in q["gold_article_ids"]) else "MISS"
        print(f"{mark} {q['id']} [{q['language']}] {q['question'][:70]}")
        print(f"       {' | '.join(parts)}   top5: {', '.join(final[:5])}")
    print()
    print(f"Gold articles in the top 5: {found}")
    print(f"Missed because hybrid search did not put them in the top {retriever.CANDIDATES}: {not_candidate}")
    print(f"Missed because the reranker pushed them below 5: {reranked_out}")


if __name__ == "__main__":
    main()
