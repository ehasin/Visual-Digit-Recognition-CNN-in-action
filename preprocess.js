"use strict";

// MNIST digits were scanned at high resolution, shrunk to fit a 20x20 box and
// centered by their center of mass in the 28x28 frame, which left thin strokes
// with soft edges. Redrawing the pen path the same way makes a drawing look like
// the training data, whatever size or position it was drawn at and however thick
// the on-screen brush is.
const FIT_BOX = 20;
const STROKE_WIDTH = 2.8;             // output pixels
const EDGE_SOFTNESS = 1;              // width of the grey edge, in output pixels
const MIN_EXTENT = 2;                 // in cells; keeps a dot from being blown up to fill the box

// strokes: arrays of {gx, gy} points in grid cells; returns an n x n image
function renderStrokes(strokes, n) {
  const pts = strokes.flat();
  if (!pts.length) return new Float32Array(n * n);

  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const p of pts) {
    x0 = Math.min(x0, p.gx); x1 = Math.max(x1, p.gx);
    y0 = Math.min(y0, p.gy); y1 = Math.max(y1, p.gy);
  }
  const scale = (FIT_BOX - STROKE_WIDTH) / Math.max(x1 - x0, y1 - y0, MIN_EXTENT);
  const midX = (x0 + x1) / 2, midY = (y0 + y1) / 2;

  // segments in output pixels, bounding box centered
  const segs = [];
  for (const s of strokes) {
    for (let k = 0; k < s.length; k++) {
      const a = s[k], b = s[Math.min(k + 1, s.length - 1)];
      segs.push([(a.gx - midX) * scale + n / 2, (a.gy - midY) * scale + n / 2,
                 (b.gx - midX) * scale + n / 2, (b.gy - midY) * scale + n / 2]);
    }
  }

  // draw once to find the center of mass, then again shifted onto the center
  const first = rasterize(segs, n, 0, 0);
  let mass = 0, mx = 0, my = 0;
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const v = first[r * n + c];
      mass += v; mx += v * (c + 0.5); my += v * (r + 0.5);
    }
  }
  return rasterize(segs, n, n / 2 - mx / mass, n / 2 - my / mass);
}

function rasterize(segs, n, dx, dy) {
  // distance from each pixel center to the nearest segment; only pixels within
  // reach of a segment can get ink, so each segment visits just its neighbourhood
  const reach = STROKE_WIDTH / 2 + EDGE_SOFTNESS / 2;
  const dist = new Float32Array(n * n).fill(Infinity);
  for (const [ax, ay, bx, by] of segs) {
    const s = [ax + dx, ay + dy, bx + dx, by + dy];
    const c0 = Math.max(0, Math.floor(Math.min(s[0], s[2]) - reach)), c1 = Math.min(n - 1, Math.ceil(Math.max(s[0], s[2]) + reach));
    const r0 = Math.max(0, Math.floor(Math.min(s[1], s[3]) - reach)), r1 = Math.min(n - 1, Math.ceil(Math.max(s[1], s[3]) + reach));
    for (let r = r0; r <= r1; r++) {
      for (let c = c0; c <= c1; c++) {
        const d = distToSegment(c + 0.5, r + 0.5, s);
        if (d < dist[r * n + c]) dist[r * n + c] = d;
      }
    }
  }
  const out = new Float32Array(n * n);
  for (let i = 0; i < n * n; i++) {
    const v = (STROKE_WIDTH / 2 - dist[i]) / EDGE_SOFTNESS + 0.5;
    out[i] = Math.min(1, Math.max(0, v));
  }
  return out;
}

function distToSegment(px, py, [ax, ay, bx, by]) {
  const vx = bx - ax, vy = by - ay;
  const len2 = vx * vx + vy * vy;
  const t = len2 ? Math.min(1, Math.max(0, ((px - ax) * vx + (py - ay) * vy) / len2)) : 0;
  return Math.hypot(px - ax - t * vx, py - ay - t * vy);
}

if (typeof module !== "undefined") module.exports = { renderStrokes };
