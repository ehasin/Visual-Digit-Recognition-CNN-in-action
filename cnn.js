"use strict";

// A small convolutional network: instead of one weight per pixel, each filter is a
// 5x5 pattern slid across the whole image, so a stroke is recognised wherever it is.
// Two rounds of filter + shrink, then one dense layer picks the digit.

// "same" 5x5 convolution with ReLU over a size x size image with layer.in channels.
// Also returns how much each input channel added to each output map (out x in),
// summed over the positions where the map lit up.
function convRelu(layer, input, size) {
  const { out, k, W, b } = layer, cin = layer.in, pad = (k - 1) >> 1, area = size * size;
  const result = new Float32Array(out * area);
  const contrib = new Float32Array(out * cin);
  const part = new Float32Array(cin);
  for (let o = 0; o < out; o++) {
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        let s = b[o];
        for (let c = 0; c < cin; c++) {
          let p = 0;
          for (let ky = 0; ky < k; ky++) {
            const iy = y + ky - pad;
            if (iy < 0 || iy >= size) continue;
            for (let kx = 0; kx < k; kx++) {
              const ix = x + kx - pad;
              if (ix < 0 || ix >= size) continue;
              p += W[((o * cin + c) * k + ky) * k + kx] * input[c * area + iy * size + ix];
            }
          }
          part[c] = p;
          s += p;
        }
        if (s > 0) for (let c = 0; c < cin; c++) contrib[o * cin + c] += part[c];
        result[o * area + y * size + x] = Math.max(0, s);
      }
    }
  }
  return { values: result, size, contrib };
}

// keeps the strongest value in each step x step square; leftover edge rows are dropped
function maxPool(input, channels, size, step) {
  const outSize = Math.floor(size / step), area = size * size;
  const result = new Float32Array(channels * outSize * outSize);
  for (let c = 0; c < channels; c++) {
    for (let y = 0; y < outSize; y++) {
      for (let x = 0; x < outSize; x++) {
        let m = -Infinity;
        for (let dy = 0; dy < step; dy++) {
          for (let dx = 0; dx < step; dx++) m = Math.max(m, input[c * area + (y * step + dy) * size + x * step + dx]);
        }
        result[(c * outSize + y) * outSize + x] = m;
      }
    }
  }
  return { values: result, size: outSize };
}

// input: 28x28 values 0..1 (ink = 1).
// Returns the feature maps of both layers, the softmax probabilities for the 10 digits,
// and how much each map pushed each map or digit above it (contrib1: 16 x 8, contrib2: 10 x 16).
function cnnForward(model, input, n) {
  const c1 = convRelu(model.conv1, input, n);
  const p1 = maxPool(c1.values, model.conv1.out, n, model.pools[0]);
  const c2 = convRelu(model.conv2, p1.values, p1.size);
  const p2 = maxPool(c2.values, model.conv2.out, p1.size, model.pools[1]);
  const { fc } = model, maps = model.conv2.out, cells = fc.in / maps;
  const z = new Float32Array(fc.out), contrib2 = new Float32Array(fc.out * maps);
  for (let j = 0; j < fc.out; j++) {
    let s = fc.b[j];
    for (let i = 0; i < fc.in; i++) {
      const v = fc.W[j * fc.in + i] * p2.values[i];
      s += v;
      contrib2[j * maps + Math.floor(i / cells)] += v;
    }
    z[j] = s;
  }
  const top = Math.max(...z), e = Array.from(z, (v) => Math.exp(v - top)), sum = e.reduce((a, v) => a + v, 0);
  return { maps1: c1, maps2: c2, contrib1: c2.contrib, contrib2, probs: e.map((v) => v / sum) };
}

if (typeof module !== "undefined") module.exports = { cnnForward };
