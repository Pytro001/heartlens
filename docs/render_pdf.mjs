// node docs/render_pdf.mjs  renders docs/HeartLens_Report.html to docs/HeartLens_Report.pdf with Playwright
import { launch } from '../output/browser.mjs'
const dir = new URL('.', import.meta.url).pathname
const b = await launch([])
const p = await b.newPage()
await p.goto('file://' + dir + 'HeartLens_Report.html')
await p.waitForLoadState('networkidle')
await p.pdf({ path: dir + 'HeartLens_Report.pdf', format: 'A4', printBackground: true, preferCSSPageSize: true })
await b.close()
console.log('wrote', dir + 'HeartLens_Report.pdf')
