// node shot.mjs "<url query>" out.jpg [js to eval before shot]
import { launch } from './browser.mjs'
const [q = '', out = 'shot.jpg', js = ''] = process.argv.slice(2)
const b = await launch(['--use-angle=metal', '--ignore-gpu-blocklist'])
const p = await b.newPage({ viewport: { width: 1920, height: 1080 } })
p.on('pageerror', e => console.log('ERR', e.message))
p.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') console.log(m.type(), m.text().slice(0, 300)) })
await p.goto('http://localhost:5301/' + q)
await p.waitForFunction(() => !document.getElementById('loading'), null, { timeout: 60000 })
await p.waitForTimeout(1500)
if (js) console.log(await p.evaluate(js))
await p.waitForTimeout(1800)
await p.screenshot({ path: out, type: 'jpeg', quality: 85 })
await b.close()
