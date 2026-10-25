// مُصان's orb: sits on the edge of the question box of the open page.
// Its eyes follow the pointer, follow the caret while the user types, scan from
// side to side while the app is answering or analysing, and smile at the result.
// Runs in a 1-pixel iframe, so it draws in the app page (window.parent.document).
const doc = window.parent.document;
const win = window.parent;

if (doc.musanOrb) doc.musanOrb.stop();  // a rerun loaded this script again

const css = doc.createElement("style");
css.textContent = `
[data-testid="stMain"] { position: relative; }
#musan-orb { position: absolute; z-index: 50; width: 64px; height: 64px; pointer-events: none;
  filter: drop-shadow(0 6px 10px rgba(11, 61, 44, .25)); transition: opacity .2s; }
#musan-orb svg { display: block; width: 100%; height: 100%; overflow: visible; }
#musan-orb .ball { transform-box: fill-box; transform-origin: 50% 100%; }
#musan-orb .eyes { transition: transform .1s ease-out; }
#musan-orb .eye { fill: #FFFFFF; transform-box: fill-box; transform-origin: center; transition: transform .15s; }
#musan-orb.blink .eye { animation: musan-blink .2s ease-in-out; }
#musan-orb.hop .ball { animation: musan-hop .18s ease-out; }
#musan-orb.happy .eye { transform: scaleY(.45); }
@media (max-width: 640px) { #musan-orb { display: none; } }  /* no room beside the box on a phone */
@keyframes musan-blink { 50% { transform: scaleY(.1); } }
@keyframes musan-hop { 50% { transform: scale(1.04, .95); } }
@media (prefers-reduced-motion: reduce) {
  #musan-orb.blink .eye, #musan-orb.hop .ball { animation: none; }
  #musan-orb .eyes, #musan-orb .eye { transition: none; } }`;
doc.head.appendChild(css);

const orb = doc.createElement("div");
orb.id = "musan-orb";
orb.setAttribute("aria-hidden", "true");
orb.innerHTML = `<svg viewBox="0 0 100 100">
  <defs><radialGradient id="musan-orb-fill" cx=".35" cy=".3" r=".8">
    <stop offset="0" stop-color="#23855A"/><stop offset="1" stop-color="#0B3D2C"/></radialGradient></defs>
  <g class="ball"><circle cx="50" cy="50" r="46" fill="url(#musan-orb-fill)" stroke="#FFFFFF" stroke-opacity=".5" stroke-width="3"/>
    <ellipse cx="36" cy="28" rx="12" ry="6" fill="#FFFFFF" opacity=".12" transform="rotate(-25 36 28)"/>
    <g class="eyes"><ellipse class="eye" cx="40" cy="46" rx="5.5" ry="8.5"/><ellipse class="eye" cx="60" cy="46" rx="5.5" ry="8.5"/></g>
  </g></svg>`;
// inside the part of the page that scrolls, so it moves with the box with no delay
const scroller = () => doc.querySelector('[data-testid="stMain"]') || doc.body;
scroller().appendChild(orb);
const eyes = orb.querySelector(".eyes");

let mode = "follow", pointer = null, box = null, field = null;

function visibleBox() {  // the question box of the page on screen
  for (const el of doc.querySelectorAll('[class*="st-key-askbox"]')) {
    const r = el.getBoundingClientRect();
    if (r.width > 0 && r.height > 0) return el;
  }
  return null;
}

function place() {  // on the top edge of the box, at its end (left in Arabic, right in English)
  const main = scroller();
  if (!orb.isConnected || orb.parentNode !== main) main.appendChild(orb);  // a rerun rebuilt the page
  box = visibleBox();
  field = box ? box.querySelector("input[type=text], input:not([type])") : null;
  if (!box) { orb.style.opacity = "0"; return; }
  const r = box.getBoundingClientRect(), m = main.getBoundingClientRect();
  const rtl = getComputedStyle(box).direction === "rtl";
  orb.style.opacity = "1";
  orb.style.top = `${r.top - m.top + main.scrollTop - 58}px`;
  orb.style.left = `${(rtl ? r.left + 18 : r.right - 82) - m.left + main.scrollLeft}px`;
}

function look(x, y) {
  const b = orb.getBoundingClientRect();
  const dx = x - (b.left + b.width / 2), dy = y - (b.top + b.height * .46), d = Math.hypot(dx, dy) || 1;
  const k = Math.min(1, d / (b.width * 1.2)), reach = 15;
  eyes.style.transform = `translate(${(dx / d * reach * k).toFixed(1)}px, ${(dy / d * reach * .8 * k).toFixed(1)}px)`;
}

// where the caret sits in the field, measured with a hidden copy of the text before it
const meter = doc.createElement("span");
meter.style.cssText = "position:fixed;visibility:hidden;white-space:pre;left:-9999px;top:0";
doc.body.appendChild(meter);
function lookAtCaret() {
  if (mode !== "typing" || !field) return;
  const cs = win.getComputedStyle(field), r = field.getBoundingClientRect();
  meter.style.font = cs.font;
  meter.textContent = field.value.slice(0, field.selectionStart ?? field.value.length);
  const rtl = cs.direction === "rtl";
  const x = rtl ? Math.max(r.left + 8, r.right - parseFloat(cs.paddingRight) - meter.offsetWidth)
                : Math.min(r.right - 8, r.left + parseFloat(cs.paddingLeft) + meter.offsetWidth);
  look(x, r.top + r.height / 2);
}

function bounce(name) { orb.classList.remove(name); void orb.offsetWidth; orb.classList.add(name); }

const on = {
  move: (e) => { pointer = [e.clientX, e.clientY]; if (mode === "follow") look(e.clientX, e.clientY); },
  focus: (e) => { if (e.target === field && mode !== "reading") { mode = "typing"; lookAtCaret(); } },
  blur: (e) => { if (e.target === field && mode === "typing") { mode = "follow"; if (pointer) look(...pointer); } },
  input: (e) => { if (e.target === field) { lookAtCaret(); bounce("hop"); } },
  caret: (e) => { if (e.target === field) lookAtCaret(); },
};
doc.addEventListener("pointermove", on.move);
doc.addEventListener("focusin", on.focus);
doc.addEventListener("focusout", on.blur);
doc.addEventListener("input", on.input);
["keyup", "click", "select"].forEach((t) => doc.addEventListener(t, on.caret));

// reading: while Streamlit runs the script for more than a moment (an answer or an analysis)
let scan = null, readTimer = null;
function startReading() {
  mode = "reading";
  const t0 = performance.now();
  scan = setInterval(() => {
    eyes.style.transform = `translate(${(Math.sin((performance.now() - t0) / 260) * 12).toFixed(1)}px, 5px)`;
  }, 30);
}
function stopReading() {
  clearInterval(scan); scan = null;
  const result = doc.querySelector(".answer-card, .verdict");
  if (result) {
    const r = result.getBoundingClientRect();
    look(r.left + r.width / 2, r.top + 20);
    bounce("happy");
    setTimeout(() => orb.classList.remove("happy"), 900);
  }
  mode = "answer";
  setTimeout(() => { if (mode === "answer") { mode = "follow"; if (pointer) look(...pointer); } }, 2500);
}
const app = doc.querySelector('[data-testid="stApp"]');
const watcher = new MutationObserver(() => {
  const running = app.getAttribute("data-test-script-state") === "running";
  if (running && !scan && !readTimer) readTimer = setTimeout(() => { readTimer = null; startReading(); }, 400);
  if (!running) {
    clearTimeout(readTimer); readTimer = null;
    if (scan) stopReading();
  }
});
if (app) watcher.observe(app, { attributes: true, attributeFilter: ["data-test-script-state"] });

let frame;
(function follow() { place(); frame = win.requestAnimationFrame(follow); })();

let blinker;
(function blinkLoop() {
  blinker = setTimeout(() => { if (mode !== "answer") bounce("blink"); blinkLoop(); }, 2500 + Math.random() * 3000);
})();

doc.musanOrb = {
  stop() {
    win.cancelAnimationFrame(frame); clearTimeout(blinker); clearTimeout(readTimer); clearInterval(scan);
    watcher.disconnect();
    doc.removeEventListener("pointermove", on.move);
    doc.removeEventListener("focusin", on.focus);
    doc.removeEventListener("focusout", on.blur);
    doc.removeEventListener("input", on.input);
    ["keyup", "click", "select"].forEach((t) => doc.removeEventListener(t, on.caret));
    orb.remove(); css.remove(); meter.remove();
  },
};
