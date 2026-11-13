// Renders animation.html into demo.mp4 and demo.gif (the clip at the top of the README).
//
//   npm install --no-save playwright-core     (uses the Chrome already installed)
//   node docs/demo/render.cjs
//
// Needs Google Chrome and ffmpeg. Pass a comma-separated list of seconds to save
// still frames instead, for a quick look:  node docs/demo/render.cjs 1.5,8,12

const { chromium } = require('playwright-core');
const { execFileSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const FPS = 30;
const here = __dirname;

(async () => {
  const browser = await chromium.launch({ channel: 'chrome' });
  const page = await browser.newPage({ viewport: { width: 960, height: 560 } });
  await page.goto('file://' + path.join(here, 'animation.html') + '?capture=1&scale=2');
  const frame = async (t, file) => {
    const url = await page.evaluate(s => { draw(s); return document.getElementById('c').toDataURL('image/png'); }, t);
    fs.writeFileSync(file, Buffer.from(url.split(',')[1], 'base64'));
  };

  if (process.argv[2]) {
    const out = process.argv[3] || here;
    for (const t of process.argv[2].split(',').map(Number)) await frame(t, path.join(out, `still-${t}.png`));
    await browser.close();
    return;
  }

  const duration = await page.evaluate("DURATION");
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'eyetracker-demo-'));
  const total = Math.round(duration * FPS);
  for (let i = 0; i < total; i++) await frame(i / FPS, path.join(tmp, `f${String(i).padStart(4, '0')}.png`));
  await browser.close();

  const input = ['-y', '-loglevel', 'error', '-framerate', String(FPS), '-i', path.join(tmp, 'f%04d.png')];
  execFileSync('ffmpeg', [...input, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
    '-movflags', '+faststart', path.join(here, 'demo.mp4')]);
  execFileSync('ffmpeg', [...input, '-filter_complex',
    'fps=15,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160:stats_mode=diff[p];' +
    '[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle', path.join(here, 'demo.gif')]);
  fs.rmSync(tmp, { recursive: true });
  console.log(`Rendered ${total} frames to demo.mp4 and demo.gif`);
})();
