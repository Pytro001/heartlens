// node output/record.mjs [out.mp4]  films the whole ?film=1 timeline frame by frame (needs `npm run dev` on port 5301)
import { launch } from './browser.mjs'
import { spawn } from 'node:child_process'
const dir = new URL('.', import.meta.url).pathname
const OUT = process.argv[2] || dir + 'heartlens-screen.mp4', FPS = 30
const b = await launch(['--use-angle=metal', '--ignore-gpu-blocklist', '--autoplay-policy=no-user-gesture-required'])
const p = await b.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 })
p.on('pageerror', e => console.log('ERR', e.message))
await p.goto('http://localhost:5301/?film=1')
await p.waitForFunction(() => window.__film?.ready, null, { timeout: 60000 })
await p.waitForTimeout(1500)
const chs = await p.evaluate(() => window.__film.chapters)
const total = chs.reduce((a, c) => a + c.length, 0)
console.log('chapters', chs.map(c => `${c.name} ${c.length}s`).join(', '), 'total', total, 's')
// warm up every chapter off camera so shaders and layouts are ready before the first real frame
for (let i = 0; i < chs.length; i++) await p.evaluate(async i => { await window.__film.goto(i); await window.__film.step(1 / 30, 6) }, i)
await p.evaluate(() => window.__film.goto(0))
const n = Math.round(total * FPS)
const ff = spawn('/opt/homebrew/bin/ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-vcodec', 'mjpeg', '-i', 'pipe:0',
  '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', OUT])
const t0 = Date.now()
for (let i = 0; i < n; i++) {
  if (i) await p.evaluate(() => window.__film.step(1 / 30))
  const jpg = await p.screenshot({ type: 'jpeg', quality: 92 })
  if (!ff.stdin.write(jpg)) await new Promise(r => ff.stdin.once('drain', r))
  if (i % 300 === 0) console.log('frame', i, '/', n, ((Date.now() - t0) / 1000).toFixed(0) + 's')
}
ff.stdin.end(); await new Promise(r => ff.on('close', r))
console.log('done', OUT, (n / FPS).toFixed(1), 's')
await b.close()
