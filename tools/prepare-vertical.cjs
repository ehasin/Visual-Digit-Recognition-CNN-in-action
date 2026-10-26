const fs = require('fs');
const path = require('path');
const root = path.resolve(__dirname, '..');
let src = fs.readFileSync(path.join(__dirname, 'render-gtm-video.cjs'), 'utf8');
src = src.replace("const movie = createCanvas(1440, 1080)", "const movie = createCanvas(1080, 1920)");
src = src.replace("'-an',", "'-i', path.join(assets, 'digit-evolution-background.wav'), '-c:a', 'aac', '-b:a', '192k', '-shortest',");
src = src.replace("'digit-evolution-1-4-9.mp4'", "'digit-evolution-1-4-9-vertical.mp4'");
const start = src.indexOf("  g.fillStyle='#fbfaf6';");
const end = src.indexOf("  if(!encoder.stdin.write", start);
src = src.slice(0,start) + `  g.fillStyle='#fbfaf6';g.fillRect(0,0,1080,1920);
  g.drawImage(net,0,170,1080,846*1080/760);
  const captions = t<2.7 ? ['Start with a 1.', 'Watch the confidence above.'] : t<6.2 ? ['Add a stroke and a crossbar.', 'Now the network sees a 4.'] : t<8.8 ? ['Slowly close the top.', 'The internal signals change.'] : t<11.4 ? ['Watch 4 and 9 compete', 'as the loop closes.'] : ['Now it sees a 9.', 'Same trained weights. New prediction.'];
  g.font='600 44px "Segoe UI", sans-serif';g.textAlign='center';
  const widths=captions.map(s=>g.measureText(s).width);
  const boxWidth=Math.max(...widths)+64;
  g.fillStyle='rgba(0,0,0,0.82)';g.beginPath();g.roundRect((1080-boxWidth)/2,1530,boxWidth,142,14);g.fill();
  g.fillStyle='#ffffff';captions.forEach((s,i)=>g.fillText(s,540,1586+i*57));
  g.textAlign='left';
  if([75,180,270,390].includes(frame)) fs.writeFileSync(path.join(out,\x60vertical-preview-\x24{frame}.png\x60),movie.toBuffer('image/png'));
` + src.slice(end);
src=src.replace("'confidence-timeline.json'", "'vertical-confidence-timeline.json'");
fs.writeFileSync(path.join(__dirname,'render-gtm-video-vertical.cjs'),src);
// Original ambient music: soft chords, a quiet pulse, and a sparse bell motif.
const rate=48000, seconds=14, samples=rate*seconds;
const wav=Buffer.alloc(44+samples*4);
wav.write('RIFF');wav.writeUInt32LE(wav.length-8,4);wav.write('WAVEfmt ',8);wav.writeUInt32LE(16,16);wav.writeUInt16LE(1,20);wav.writeUInt16LE(2,22);wav.writeUInt32LE(rate,24);wav.writeUInt32LE(rate*4,28);wav.writeUInt16LE(4,32);wav.writeUInt16LE(16,34);wav.write('data',36);wav.writeUInt32LE(samples*4,40);
const chords=[[130.813,164.814,195.998,246.942],[110,130.813,164.814,195.998],[87.307,130.813,174.614,220],[97.999,146.832,195.998,246.942]];
for(let i=0;i<samples;i++){
 const t=i/rate, fade=Math.min(1,t/0.8,(seconds-t)/1.4); let l=0,r=0;
 for(let c=0;c<4;c++){
 const local=t-c*3.5;if(local<0||local>4.5)continue;
 const env=Math.min(1,local/0.55)*Math.max(0,Math.min(1,(4.5-local)/1));
 chords[c].forEach((f,j)=>{l+=0.027*env*Math.sin(2*Math.PI*f*t);r+=0.027*env*Math.sin(2*Math.PI*(f+.22)*t);});
 }
 const beat=t%0.75, k=Math.floor(t/0.75), freq=[523.251,659.255,783.991,987.767,783.991,659.255][k%6];
 const bell=0.045*Math.exp(-beat*7)*Math.sin(2*Math.PI*freq*beat);
 const pulse=0.018*Math.exp(-beat*26)*Math.sin(2*Math.PI*65.406*beat);
 wav.writeInt16LE(Math.round((l+bell+pulse)*fade*32767),44+i*4);wav.writeInt16LE(Math.round((r+bell+pulse)*fade*32767),46+i*4);
}
fs.writeFileSync(path.join(root,'assets/digit-evolution-background.wav'),wav);
