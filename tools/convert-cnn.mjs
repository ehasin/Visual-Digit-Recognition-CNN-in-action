// Converts the ONNX Model Zoo MNIST CNN (https://github.com/onnx/models, mnist-8.onnx, MIT)
// into the compact format the page loads. Reads the protobuf directly, so no ONNX tooling is needed.
// Usage: node tools/convert-cnn.mjs <path/to/mnist-8.onnx>
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";

const buf = readFileSync(process.argv[2]);
const round = (v) => Number(v.toPrecision(5));

// protobuf wire format: yields [field number, wire type, value or [start, end] of a nested message]
function* fields(start = 0, end = buf.length) {
  let p = start;
  const varint = () => {
    let r = 0, mul = 1, x;
    do { x = buf[p++]; r += (x & 127) * mul; mul *= 128; } while (x & 128);
    return r;
  };
  while (p < end) {
    const key = varint(), field = Math.floor(key / 8), type = key & 7;
    if (type === 0) yield [field, type, varint()];
    else if (type === 2) { const n = varint(); yield [field, type, [p, p + n]]; p += n; }
    else if (type === 5) { yield [field, type, buf.readFloatLE(p)]; p += 4; }
    else if (type === 1) p += 8;
    else throw new Error(`unsupported wire type ${type}`);
  }
}

// ModelProto.graph = 7, GraphProto.initializer = 5, TensorProto: dims = 1, float_data = 4, name = 8
const tensors = {};
for (const [f, , graph] of fields()) {
  if (f !== 7) continue;
  for (const [g, , tensor] of fields(...graph)) {
    if (g !== 5) continue;
    const t = { dims: [], data: [] };
    for (const [h, type, v] of fields(...tensor)) {
      if (h === 1) t.dims.push(v);
      if (h === 8) t.name = buf.toString("utf8", ...v);
      if (h === 4 && type === 2) for (let p = v[0]; p < v[1]; p += 4) t.data.push(buf.readFloatLE(p));
      if (h === 4 && type === 5) t.data.push(v);
    }
    tensors[t.name] = t;
  }
}

const conv = (w, b) => ({
  out: tensors[w].dims[0], in: tensors[w].dims[1], k: tensors[w].dims[2],
  W: tensors[w].data.map(round), // out x in x k x k
  b: tensors[b].data.map(round),
});

// The dense layer is stored as [16, 4, 4, 10] (inputs x outputs); flip it to outputs x inputs.
const fcIn = tensors.Parameter193.data.length / 10;
const fcW = [];
for (let j = 0; j < 10; j++) for (let i = 0; i < fcIn; i++) fcW.push(round(tensors.Parameter193.data[i * 10 + j]));

const model = {
  source: "https://github.com/onnx/models/tree/main/validated/vision/classification/mnist (mnist-8.onnx)",
  license: "MIT (ONNX Model Zoo), trained by Microsoft with the CNTK 103D tutorial",
  layout: "conv5x5 same + relu + maxpool 2, conv5x5 same + relu + maxpool 3, dense; outputs are scores before softmax",
  conv1: conv("Parameter5", "Parameter6"),
  conv2: conv("Parameter87", "Parameter88"),
  pools: [2, 3],
  fc: { in: fcIn, out: 10, W: fcW, b: tensors.Parameter194.data.map(round) },
};

mkdirSync("model", { recursive: true });
writeFileSync("model/cnn.json", JSON.stringify(model));
console.log(`wrote model/cnn.json: conv ${model.conv1.out}, conv ${model.conv2.out}, dense ${fcIn}->10`);
