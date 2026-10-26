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
const movie = createCanvas(1080, 1920), g = movie.getContext('2d');
const fps = 30, seconds = 14;
const encoder = spawn('ffmpeg', ['-y', '-hide_banner', '-loglevel', 'error', '-f', 'image2pipe', '-vcodec', 'png', '-framerate', String(fps), '-i', 'pipe:0', '-i', path.join(assets, 'digit-evolution-background.wav'), '-c:a', 'aac', '-b:a', '192k', '-shortest', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path.join(assets, 'digit-evolution-1-4-9-vertical.mp4')]);
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
  g.fillStyle='#fbfaf6';g.fillRect(0,0,1080,1920);
  g.drawImage(net,0,170,1080,846*1080/760);
  const captions = t<2.7 ? ['Start with a 1.', 'Watch the confidence above.'] : t<6.2 ? ['Add a stroke and a crossbar.', 'Now the network sees a 4.'] : t<8.8 ? ['Slowly close the top.', 'The internal signals change.'] : t<11.4 ? ['Watch 4 and 9 compete', 'as the loop closes.'] : ['Now it sees a 9.', 'Same trained weights. New prediction.'];
  g.font='600 44px "Segoe UI", sans-serif';g.textAlign='center';
  const widths=captions.map(s=>g.measureText(s).width);
  const boxWidth=Math.max(...widths)+64;
  g.fillStyle='rgba(0,0,0,0.82)';g.beginPath();g.roundRect((1080-boxWidth)/2,1530,boxWidth,142,14);g.fill();
  g.fillStyle='#ffffff';captions.forEach((s,i)=>g.fillText(s,540,1586+i*57));
  g.textAlign='left';
  if([75,180,270,390].includes(frame)) fs.writeFileSync(path.join(out,`vertical-preview-${frame}.png`),movie.toBuffer('image/png'));
  if(!encoder.stdin.write(movie.toBuffer('image/png'))) await new Promise(resolve=>encoder.stdin.once('drain',resolve));
  if(frame%60===0) console.log(`Rendered ${frame}/${fps*seconds}`);
 }
 encoder.stdin.end(); await completed;
 fs.writeFileSync(path.join(out,'vertical-confidence-timeline.json'),JSON.stringify(log));
 console.log('Saved 14-second MP4.');
})().catch(e=>{console.error(e);encoder.kill();process.exitCode=1;});
