// Reproducible, frame-by-frame capture of the app's own renderer and model.
// No model outputs are scripted or interpolated.
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { spawn } = require('child_process');
const { createCanvas } = require(process.env.NAPI_CANVAS || '@napi-rs/canvas');  // NAPI_CANVAS: optional path to a local copy
const root = path.resolve(__dirname, '..');
const out = path.join(root, 'assets');
fs.mkdirSync(out, { recursive: true });
const assets = path.join(root, 'assets');
fs.mkdirSync(assets, { recursive: true });
const net = createCanvas(760, 846);
net.addEventListener = () => {};
const sandbox = { Math, Float32Array, document: { getElementById: id => id === 'net' ? net : { addEventListener() {} }, addEventListener() {} }, window: {}, requestAnimationFrame() {} };
vm.createContext(sandbox);
for (const file of ['preprocess.js', 'cnn.js', 'app.js']) {
  let src = fs.readFileSync(path.join(root, file), 'utf8');
  if (file === 'app.js') src = src.replace(/start\(\);\s*$/, '');
  vm.runInContext(src.replaceAll('650', '600'), sandbox);
}
sandbox.model = JSON.parse(fs.readFileSync(path.join(root, 'model/cnn.json')));
vm.runInContext('cnn = model;', sandbox);
const movie = createCanvas(1440, 1080), g = movie.getContext('2d');
const fps = 30, seconds = 14;
const encoder = spawn('ffmpeg', ['-y', '-hide_banner', '-loglevel', 'error', '-f', 'image2pipe', '-vcodec', 'png', '-framerate', String(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path.join(assets, 'digit-evolution-1-4-9.mp4')]);
encoder.stderr.on('data', x => process.stderr.write(x));
const completed = new Promise((resolve, reject) => { encoder.on('error', reject); encoder.on('close', c => c === 0 ? resolve() : reject(new Error('ffmpeg exit ' + c))); });
function text(s, x, y, size = 25, color = '#16150f', weight = 500) { g.font = `${weight} ${size}px "Segoe UI", sans-serif`; g.fillStyle = color; g.fillText(s, x, y); }
function lines(ss, y, size = 28, color = '#16150f') { ss.forEach((s, i) => text(s, 860, y + i * (size + 12), size, color)); }
const clamp = x => Math.min(1, Math.max(0, x));
function partial(points, amount) {
  if (amount <= 0) return [];
  const lengths = points.slice(1).map((p, i) => Math.hypot(p[0]-points[i][0], p[1]-points[i][1]));
  let remaining = lengths.reduce((a,b)=>a+b,0)*clamp(amount), result = [points[0]];
  for(let i=0;i<lengths.length;i++) {
    const a=points[i], b=points[i+1], u=Math.min(1,remaining/lengths[i]);
    result.push([a[0]+(b[0]-a[0])*u,a[1]+(b[1]-a[1])*u]);
    remaining-=lengths[i]; if(remaining<=0) break;
  }
  return result.map(([gx,gy])=>({gx,gy}));
}
const paths = [ [[18,5],[18,24]], [[8,5],[8,15],[20,15]], [[8,5],[9,4],[11,3.3],[14,3],[16.5,3.5],[18,5]] ];
const log = [];
(async () => {
 for (let frame=0;frame<fps*seconds;frame++) {
  const t=frame/fps;
  const amounts=[clamp((t-.3)/1.3),clamp((t-3)/2),clamp((t-7)/4.3)];
  sandbox.frameStrokes=paths.map((p,i)=>partial(p,amounts[i])).filter(p=>p.length);
  const probs=vm.runInContext('strokes.length=0; strokes.push(...frameStrokes); state=forward(); render(); Array.from(state.probs)', sandbox);
  const winner=probs.indexOf(Math.max(...probs));
  log.push({time:t,probs});
  g.fillStyle='#fbfaf6';g.fillRect(0,0,1440,1080);
  text('ONE DRAWING. THREE PREDICTIONS.',50,60,20,'#6f6b61',600);
  text('Watch a neural network change its mind',50,112,43,'#16150f',600);
  g.drawImage(net,30,153);
  g.strokeStyle='#e2ded2';g.lineWidth=2;g.beginPath();g.moveTo(820,174);g.lineTo(820,970);g.stroke();
  const stage=t<3?0:t<7?1:2;
  text(['01 / DRAW A 1','02 / ADD TWO LINES','03 / CLOSE THE TOP'][stage],860,205,20,'#1a8c57',700);
  const captions = t<3 ? ['A single vertical stroke.', 'The model sees a 1.'] : t<7 ? ['Add a left stroke and crossbar.', 'New features favor a 4.'] : t<11.4 ? ['Slowly close the upper loop.', 'Watch 4 and 9 compete.'] : ['The loop completes a 9.', 'Same weights. New prediction.'];
  lines(captions,264,27);
  text('LIVE MODEL CONFIDENCE',860,395,18,'#6f6b61',600);
  for(const [i,d] of [1,4,9].entries()) {
    const y=440+i*76;
    text(String(d),860,y+6,30,d===winner?'#1a8c57':'#6f6b61',700);
    g.fillStyle='#e8e5db';g.fillRect(905,y-17,365,22);
    g.fillStyle=d===winner?'#1fa463':'#a9a496';g.fillRect(905,y-17,365*probs[d],22);
    const pct = probs[d]*100;
    text(pct>99.9?'>99.9%':pct<.1?'<0.1%':pct.toFixed(1)+'%',1290,y+2,20,'#16150f',600);
  }
  g.fillStyle='#f0eee5';g.beginPath();g.roundRect(848,680,548,205,16);g.fill();
  text('WHAT THE COLORS SHOW',870,717,18,'#6f6b61',600);
  text('Green connections push scores up.',870,760,23,'#14804a');
  text('Magenta connections push them down.',870,798,23,'#a62e98');
  text('Their contributions change as you draw.',870,850,22);
  text('Trained weights stay fixed throughout.',860,935,23,'#6f6b61');
  text('hasin.eu/projects/cnn-digit-recognition-in-action/',50,1035,22,'#6f6b61');
  text('1  →  4  →  9',1162,1035,25,'#1a8c57',600);
  g.fillStyle='#1fa463';g.fillRect(0,1074,1440*(frame+1)/(fps*seconds),6);
  if([75,180,270,390].includes(frame)) fs.writeFileSync(path.join(out,`preview-${frame}.png`),movie.toBuffer('image/png'));
  if(!encoder.stdin.write(movie.toBuffer('image/png'))) await new Promise(resolve=>encoder.stdin.once('drain',resolve));
  if(frame%60===0) console.log(`Rendered ${frame}/${fps*seconds}`);
 }
 encoder.stdin.end(); await completed;
 fs.writeFileSync(path.join(out,'confidence-timeline.json'),JSON.stringify(log));
 console.log('Saved 14-second MP4.');
})().catch(e=>{console.error(e);encoder.kill();process.exitCode=1;});
