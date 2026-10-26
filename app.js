"use strict";

// ---------------------------------------------------------------------------
// Layout (design coordinates; the canvas is scaled to fit the page)
// ---------------------------------------------------------------------------
const W = 760, H = 840;
const N = 28;                         // the network's input is N x N
const SPAN = 660, LEFT = (W - SPAN) / 2;
const PAD = { x: LEFT, y: 520, size: 294 };   // where you draw, no grid
const INPUT = { x: W - LEFT - 294, y: 520, cell: 294 / N };  // what the network gets: one cell per input neuron
const PEN_WIDTH = 2.4;                // on-screen pen, in input cells
const OUT = { y: 58, size: 50 };      // one box per digit
const L2 = { y: 242, count: 16, cells: 14, cell: 2.5 };  // second-layer maps, 14x14 each
const L1 = { y: 360, count: 8, cells: 28, cell: 2.25 };  // first-layer maps, 28x28 each
const FILTER = { gap: 8, cell: 5 };   // 5x5 filter drawn under each first-layer map
const PANEL = { x: 36, y: 138, w: W - 72, h: 340 };
const ZOOM = { w: 200, map: 140, pad: 16 };   // pop-up with the hovered map, enlarged

const COLORS = {
  ink: "#111111",
  boxLine: "#111111",
  panel: "#e4e4e4",
  cellEmpty: [236, 236, 236],
  cellInk: [17, 17, 17],
  positive: [52, 199, 140],           // green: pushes the next layer up
  negative: [220, 64, 200],           // magenta: pushes the next layer down
  label: "#111111",
  muted: "#9a9a9a",
  answer: "#1fa463",                  // outline around the selected answer
};

// Map values are unbounded (ReLU); these are typical strong values on MNIST digits,
// used to squash them into 0..1 shades.
const MAP_SCALE = { l1: 1.5, l2: 3 };

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const canvas = document.getElementById("net");
const ctx = canvas.getContext("2d");
const strokes = [];                   // pen path in grid cells, redrawn cleanly for the network
let cnn = null;                       // the network, see cnn.js
let state = null;                     // last forward pass
let drawing = false;
let dirty = true;
let hover = null;                     // { layer: 1 | 2, index } of the map under the pointer

// ---------------------------------------------------------------------------
// Network
// ---------------------------------------------------------------------------
function forward() {
  const input = renderStrokes(strokes, N);  // resized and centred like the MNIST digits
  const result = cnnForward(cnn, input, N);
  const empty = strokes.length === 0;
  // the fullest box wins; its fill shows how sure the network is
  const answer = empty ? -1 : result.probs.indexOf(Math.max(...result.probs));
  return { input, ...result, answer, empty };
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------
function rowX(count, size, i) {
  const gap = (SPAN - count * size) / (count - 1);
  return LEFT + i * (size + gap);
}
const nodeCenterX = (count, size, i) => rowX(count, size, i) + size / 2;
const mapSize = (layer) => layer.cells * layer.cell;

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------
function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
}

function fillBox(x, y, size, fill, radius, lineWidth) {
  ctx.fillStyle = "#ffffff";
  roundRect(x, y, size, size, radius);
  ctx.fill();
  const pad = lineWidth + 2;
  const inner = size - 2 * pad;
  const h = inner * Math.min(1, Math.max(0, fill));
  if (h > 0.5) {
    ctx.fillStyle = COLORS.ink;
    roundRect(x + pad, y + pad + inner - h, inner, h, Math.max(1, radius - pad));
    ctx.fill();
  }
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = COLORS.boxLine;
  roundRect(x, y, size, size, radius);
  ctx.stroke();
}

// contrib[j * from.count + i]: how much source i pushed target j
function drawConnections(contrib, from, to, floor) {
  const lines = [];
  let maxAbs = floor;                 // floor keeps weak inputs looking weak
  for (let j = 0; j < to.count; j++) {
    for (let i = 0; i < from.count; i++) {
      const c = contrib[j * from.count + i];
      if (c !== 0) lines.push([i, j, c]);
      maxAbs = Math.max(maxAbs, Math.abs(c));
    }
  }
  for (const l of lines) l[3] = Math.pow(Math.abs(l[2]) / maxAbs, 0.7);
  lines.sort((a, b) => a[3] - b[3]);  // strongest on top
  for (const [i, j, c, s] of lines) {
    if (s < 0.08) continue;
    const [r, g, b] = c > 0 ? COLORS.positive : COLORS.negative;
    ctx.strokeStyle = `rgba(${r},${g},${b},${0.15 + 0.75 * s})`;
    ctx.lineWidth = 0.6 + 1.8 * s;
    ctx.beginPath();
    ctx.moveTo(nodeCenterX(from.count, from.size, i), from.y);
    ctx.lineTo(nodeCenterX(to.count, to.size, j), to.y);
    ctx.stroke();
  }
}

// one feature map: dark where the filter found its pattern
function drawMap(maps, index, x0, y0, cell, scale) {
  const { size, values } = maps, offset = index * size * size;
  const [r0, g0, b0] = COLORS.cellEmpty, [r1, g1, b1] = COLORS.cellInk;
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const v = Math.tanh(values[offset + r * size + c] / scale);
      ctx.fillStyle = `rgb(${r0 + (r1 - r0) * v},${g0 + (g1 - g0) * v},${b0 + (b1 - b0) * v})`;
      // a hair of overlap hides seams between fractional cells
      ctx.fillRect(x0 + c * cell, y0 + r * cell, cell + 0.3, cell + 0.3);
    }
  }
  ctx.strokeStyle = COLORS.ink;
  ctx.lineWidth = 1.5;
  ctx.strokeRect(x0 - 0.75, y0 - 0.75, size * cell + 1.5, size * cell + 1.5);
}

// a first-layer filter: green where it wants ink, magenta where it wants the pixel empty
function drawFilter(index, x0, y0, cell) {
  const { k, W: weights } = cnn.conv1, w = weights.slice(index * k * k, (index + 1) * k * k);
  const maxAbs = Math.max(...w.map(Math.abs));
  for (let r = 0; r < k; r++) {
    for (let c = 0; c < k; c++) {
      const v = w[r * k + c] / maxAbs, t = Math.pow(Math.abs(v), 0.7);
      const [cr, cg, cb] = v > 0 ? COLORS.positive : COLORS.negative;
      ctx.fillStyle = `rgb(${255 + (cr - 255) * t},${255 + (cg - 255) * t},${255 + (cb - 255) * t})`;
      ctx.fillRect(x0 + c * cell, y0 + r * cell, cell, cell);
    }
  }
  ctx.strokeStyle = COLORS.panel;
  ctx.lineWidth = 1;
  ctx.strokeRect(x0 - 0.5, y0 - 0.5, k * cell + 1, k * cell + 1);
}

function render() {
  const { input, maps1, maps2, contrib1, contrib2, probs, answer, empty } = state;
  ctx.clearRect(0, 0, W, H);

  // panel behind the hidden layers
  ctx.strokeStyle = COLORS.panel;
  ctx.lineWidth = 1.5;
  roundRect(PANEL.x, PANEL.y, PANEL.w, PANEL.h, 10);
  ctx.stroke();

  const s1 = mapSize(L1), s2 = mapSize(L2);
  if (!empty) {
    drawConnections(contrib1,
      { count: L1.count, size: s1, y: L1.y },
      { count: L2.count, size: s2, y: L2.y + s2 }, 1);
    drawConnections(contrib2,
      { count: L2.count, size: s2, y: L2.y },
      { count: 10, size: OUT.size, y: OUT.y + OUT.size }, 0.25);
  }

  // hidden layers: first-layer maps with their filters, then second-layer maps
  for (let i = 0; i < L1.count; i++) {
    const x = rowX(L1.count, s1, i);
    drawMap(maps1, i, x, L1.y, L1.cell, MAP_SCALE.l1);
    const k = cnn.conv1.k * FILTER.cell;
    drawFilter(i, x + (s1 - k) / 2, L1.y + s1 + FILTER.gap, FILTER.cell);
  }
  for (let i = 0; i < L2.count; i++) drawMap(maps2, i, rowX(L2.count, s2, i), L2.y, L2.cell, MAP_SCALE.l2);
  if (hover) {
    const layer = hover.layer === 1 ? L1 : L2, size = mapSize(layer);
    ctx.strokeStyle = COLORS.ink;
    ctx.lineWidth = 2;
    roundRect(rowX(layer.count, size, hover.index) - 5, layer.y - 5, size + 10, size + 10, 6);
    ctx.stroke();
  }

  // layer captions, in the panel's left margin
  ctx.font = "600 11px Inter, system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  for (const [text, y] of [["Layer 2: 16 filters of filters", L2.y - 12], ["Layer 1: 8 stroke filters", L1.y - 12]]) {
    ctx.fillStyle = "rgba(255,255,255,0.9)";  // keeps the caption readable over the lines
    ctx.fillRect(PANEL.x + 8, y - 11, ctx.measureText(text).width + 8, 15);
    ctx.fillStyle = COLORS.label;
    ctx.fillText(text, PANEL.x + 12, y);
  }

  // outputs + labels
  ctx.font = "600 34px Caveat, 'Comic Sans MS', cursive";
  ctx.textAlign = "center";
  for (let i = 0; i < 10; i++) {
    const x = rowX(10, OUT.size, i), cx = x + OUT.size / 2;
    fillBox(x, OUT.y, OUT.size, empty ? 0 : probs[i], 8, 2.5);
    if (i === answer) {
      ctx.strokeStyle = COLORS.answer;
      ctx.lineWidth = 4;
      roundRect(x - 5, OUT.y - 5, OUT.size + 10, OUT.size + 10, 12);
      ctx.stroke();
    }
    ctx.fillStyle = i === answer ? COLORS.answer : COLORS.label;
    ctx.fillText(String(i), cx, OUT.y - 14);
  }

  // how sure the network is, under its answer
  if (answer >= 0) {
    const text = `${Math.round(probs[answer] * 100)}%`, cx = rowX(10, OUT.size, answer) + OUT.size / 2;
    ctx.font = "600 13px Inter, system-ui, sans-serif";
    const w = ctx.measureText(text).width + 8;
    ctx.fillStyle = "rgba(255,255,255,0.9)";  // readable over the lines
    ctx.fillRect(cx - w / 2, OUT.y + OUT.size + 9, w, 18);
    ctx.fillStyle = COLORS.answer;
    ctx.fillText(text, cx, OUT.y + OUT.size + 23);
  }

  // your drawing, and the same-size 28x28 input it becomes
  ctx.font = "600 12px Inter, system-ui, sans-serif";
  ctx.fillStyle = COLORS.label;
  ctx.textAlign = "left";
  ctx.fillText("You draw", PAD.x, PAD.y - 10);
  ctx.fillText(`Network sees: ${N} × ${N} input neurons`, INPUT.x, INPUT.y - 10);
  drawPad();
  drawArrow(PAD.x + PAD.size + 14, INPUT.x - 14, PAD.y + PAD.size / 2);
  drawPixels(input, INPUT.x, INPUT.y, INPUT.cell, 1);

  if (hover) drawZoom(hover);
}

function drawPad() {
  ctx.fillStyle = "#ffffff";
  roundRect(PAD.x, PAD.y, PAD.size, PAD.size, 8);
  ctx.fill();
  ctx.strokeStyle = COLORS.panel;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  const k = PAD.size / N;
  ctx.save();
  roundRect(PAD.x, PAD.y, PAD.size, PAD.size, 8);
  ctx.clip();
  ctx.strokeStyle = ctx.fillStyle = COLORS.ink;
  ctx.lineWidth = PEN_WIDTH * k;
  ctx.lineCap = ctx.lineJoin = "round";
  for (const path of strokes) {
    if (path.length === 1) {          // a tap: zero-length lines don't draw, so draw the dot
      ctx.beginPath();
      ctx.arc(PAD.x + path[0].gx * k, PAD.y + path[0].gy * k, ctx.lineWidth / 2, 0, 2 * Math.PI);
      ctx.fill();
      continue;
    }
    ctx.beginPath();
    path.forEach((p, i) => ctx[i ? "lineTo" : "moveTo"](PAD.x + p.gx * k, PAD.y + p.gy * k));
    ctx.stroke();
  }
  ctx.restore();
}

function drawArrow(x0, x1, y) {
  ctx.strokeStyle = ctx.fillStyle = COLORS.muted;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x0, y);
  ctx.lineTo(x1 - 8, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x1, y);
  ctx.lineTo(x1 - 10, y - 6);
  ctx.lineTo(x1 - 10, y + 6);
  ctx.fill();
}

// pop-up under the hovered map: the map enlarged, with what its shades mean
function drawZoom({ layer, index }) {
  const maps = layer === 1 ? state.maps1 : state.maps2;
  const spec = layer === 1 ? L1 : L2, size = mapSize(spec);
  const lines = layer === 1 ? 3 : 1;
  const h = ZOOM.pad * 2 + 22 + ZOOM.map + lines * 16;
  const x = Math.min(W - 8 - ZOOM.w, Math.max(8, rowX(spec.count, size, index) + size / 2 - ZOOM.w / 2));
  const y = L1.y + mapSize(L1) + FILTER.gap + 5 * FILTER.cell + 12;
  ctx.save();
  ctx.shadowColor = "rgba(0,0,0,0.18)";
  ctx.shadowBlur = 16;
  ctx.shadowOffsetY = 4;
  ctx.fillStyle = "#ffffff";
  roundRect(x, y, ZOOM.w, h, 10);
  ctx.fill();
  ctx.restore();
  ctx.strokeStyle = COLORS.ink;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  const tx = x + ZOOM.pad, mx = x + (ZOOM.w - ZOOM.map) / 2, my = y + ZOOM.pad + 22;
  ctx.font = "600 12px Inter, system-ui, sans-serif";
  ctx.fillStyle = COLORS.label;
  ctx.textAlign = "left";
  ctx.fillText(`Layer ${layer}, map ${index + 1}`, tx, my - 10);
  drawMap(maps, index, mx, my, ZOOM.map / maps.size, layer === 1 ? MAP_SCALE.l1 : MAP_SCALE.l2);
  ctx.font = "400 12px Inter, system-ui, sans-serif";
  ctx.fillStyle = COLORS.label;
  ctx.fillText("dark: its pattern is here", tx, my + ZOOM.map + 18);
  if (layer === 1) {
    ctx.fillStyle = `rgb(${COLORS.positive})`;
    ctx.fillText("filter green: wants ink", tx, my + ZOOM.map + 34);
    ctx.fillStyle = `rgb(${COLORS.negative})`;
    ctx.fillText("filter magenta: wants empty", tx, my + ZOOM.map + 50);
  }
}

function drawPixels(values, x0, y0, cell, gap) {
  const [r0, g0, b0] = COLORS.cellEmpty, [r1, g1, b1] = COLORS.cellInk;
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const v = values[r * N + c];
      ctx.fillStyle = `rgb(${r0 + (r1 - r0) * v},${g0 + (g1 - g0) * v},${b0 + (b1 - b0) * v})`;
      ctx.fillRect(x0 + c * cell + gap / 2, y0 + r * cell + gap / 2, cell - gap, cell - gap);
    }
  }
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------
function toDesign(evt) {
  const rect = canvas.getBoundingClientRect();
  return { x: (evt.clientX - rect.left) * (W / rect.width), y: (evt.clientY - rect.top) * (H / rect.height) };
}

// pointer position on the pad, in input cells
function toGrid(evt) {
  const { x, y } = toDesign(evt);
  const k = N / PAD.size;
  return { gx: (x - PAD.x) * k, gy: (y - PAD.y) * k };
}

// the feature map under the pointer, or null; a first-layer map's filter counts as part of it
function mapAt(evt) {
  const { x, y } = toDesign(evt);
  for (const [n, layer] of [[1, L1], [2, L2]]) {
    const size = mapSize(layer);
    const bottom = layer.y + size + (n === 1 ? FILTER.gap + 5 * FILTER.cell : 0);
    if (y < layer.y || y > bottom) continue;
    for (let i = 0; i < layer.count; i++) {
      const bx = rowX(layer.count, size, i);
      if (x >= bx && x <= bx + size) return { layer: n, index: i };
    }
  }
  return null;
}

function setHover(h) {
  if (h?.layer === hover?.layer && h?.index === hover?.index) return;
  hover = h;
  dirty = true;
}

function insidePad({ gx, gy }) {
  return gx >= 0 && gx < N && gy >= 0 && gy < N;
}

function strokeTo(p) {
  const path = strokes[strokes.length - 1];
  const prev = path[path.length - 1];
  const pt = { gx: Math.min(N, Math.max(0, p.gx)), gy: Math.min(N, Math.max(0, p.gy)) };
  if (!prev || Math.hypot(pt.gx - prev.gx, pt.gy - prev.gy) >= 0.2) path.push(pt);
  dirty = true;
}

canvas.addEventListener("pointerdown", (e) => {
  const p = toGrid(e);
  if (!insidePad(p)) {
    setHover(mapAt(e));               // a tap enlarges the map on touch screens
    return;
  }
  drawing = true;
  strokes.push([]);
  canvas.setPointerCapture(e.pointerId);
  strokeTo(p);
});
canvas.addEventListener("pointermove", (e) => {
  if (!drawing) {
    setHover(mapAt(e));
    return;
  }
  // coalesced events give smooth strokes on fast pen/mouse movement
  const events = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
  for (const ev of events.length ? events : [e]) strokeTo(toGrid(ev));
});
const stop = () => { drawing = false; };
canvas.addEventListener("pointerup", stop);
canvas.addEventListener("pointercancel", stop);
canvas.addEventListener("pointerleave", (e) => { if (e.pointerType === "mouse") setHover(null); });

function clearDrawing() {
  strokes.length = 0;
  dirty = true;
}

document.getElementById("clear").addEventListener("click", clearDrawing);
document.addEventListener("keydown", (e) => {
  if (e.key === "c" || e.key === "C" || e.key === "Escape") clearDrawing();
});

// ---------------------------------------------------------------------------
// Sizing + main loop
// ---------------------------------------------------------------------------
function resize() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(canvas.width / W, 0, 0, canvas.height / H, 0, 0);
  dirty = true;
}

function loop() {
  if (dirty && cnn) {
    state = forward();
    render();
    dirty = false;
  }
  requestAnimationFrame(loop);
}

async function start() {
  const model = await fetch("model/cnn.json").then((r) => r.json());
  for (const layer of [model.conv1, model.conv2, model.fc]) {
    layer.W = Float32Array.from(layer.W);
    layer.b = Float32Array.from(layer.b);
  }
  cnn = model;
  resize();
  window.addEventListener("resize", resize);
  // redraw once the handwritten label font has arrived
  document.fonts?.load("600 34px Caveat").then(() => { dirty = true; }, () => {});
  loop();
}

start();
