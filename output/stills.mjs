// node stills.mjs "0:0.5,1:0.6" [prefix]  -> output/stills/<prefix><ch>-<u>.jpg, stepping the film clock from the chapter start
import { launch } from './browser.mjs'
const list = (process.argv[2] || '0:0.6,1:0.7,2:0.5,3:0.4,3:0.75,4:0.3,5:0.5,6:0.5,7:0.6').split(',').map(s => s.split(':').map(Number))
const prefix = process.argv[3] || 's-'
const dir = new URL('./stills/', import.meta.url).pathname
const b = await launch(['--use-angle=metal', '--ignore-gpu-blocklist'])
const p = await b.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 })
p.on('pageerror', e => console.log('ERR', e.message))
p.on('console', m => { if (m.type() === 'error') console.log(m.type(), m.text().slice(0, 300)) })
await p.goto('http://localhost:5301/?film=1')
await p.waitForFunction(() => window.__film?.ready, null, { timeout: 60000 })
const chs = await p.evaluate(() => window.__film.chapters)
for (const [c, u] of list) {
  const n = Math.max(1, Math.round(u * chs[c].length * 30))
  await p.evaluate(async ([c, n]) => { await window.__film.goto(c); await window.__film.step(1 / 30, n) }, [c, n])
  await p.screenshot({ path: `${dir}${prefix}${c}-${u}.jpg`, type: 'jpeg', quality: 85 })
  console.log('shot', c, u, chs[c].name)
}
await b.close()
