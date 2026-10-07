// Runs the four exported ONNX models through onnxruntime-web (WASM, same runtime as the page)
// and checks that calibrated probabilities match the Python export for all 303 patients,
// and that a full what-if batch stays well under 200 ms.
import * as ort from 'onnxruntime-web'
import { readFileSync } from 'node:fs'
const pub = new URL('../public/', import.meta.url)
const meta = JSON.parse(readFileSync(new URL('data/meta.json', pub)))
const patients = JSON.parse(readFileSync(new URL('data/patients.json', pub)))
ort.env.wasm.numThreads = 1
const nf = meta.features.length
const cal = (c, p) => {
  if (c.type === 'platt') { const q = Math.min(Math.max(p, 1e-6), 1 - 1e-6); return 1 / (1 + Math.exp(-(c.a * Math.log(q / (1 - q)) + c.b))) }
  const { x, y } = c; if (p <= x[0]) return y[0]; if (p >= x.at(-1)) return y.at(-1)
  let i = 1; while (x[i] < p) i++; return y[i - 1] + (p - x[i - 1]) / (x[i] - x[i - 1]) * (y[i] - y[i - 1])
}
const flat = new Float32Array(patients.length * nf)
patients.forEach((p, i) => flat.set(p.x, i * nf))
let fail = 0
const sessions = {}
for (const t of meta.targets) {
  const s = await ort.InferenceSession.create(readFileSync(new URL(t.onnx, pub)))
  sessions[t.key] = s
  const out = await s.run({ input: new ort.Tensor('float32', flat, [patients.length, nf]) })
  const name = s.outputNames.find(n => out[n].dims.length === 2 && out[n].dims[1] === 2)
  const d = out[name].data
  let err = 0
  patients.forEach((p, i) => { err = Math.max(err, Math.abs(cal(t.calibration, d[i * 2 + 1]) - p.pred[t.key])) })
  const ok = err < 1e-3
  if (!ok) fail++
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${t.key} ${t.model_label}: max |browser - python| = ${err.toExponential(2)}`)
}
// latency of one what-if batch: edited row + 9 single feature reversals, all four models
const rows = new Float32Array(10 * nf); for (let r = 0; r < 10; r++) rows.set(patients[0].x, r * nf)
const times = []
for (let k = 0; k < 30; k++) {
  const t0 = performance.now()
  await Promise.all(Object.values(sessions).map(s => s.run({ input: new ort.Tensor('float32', rows, [10, nf]) })))
  times.push(performance.now() - t0)
}
times.sort((a, b) => a - b)
const med = times[15]
console.log(`${med < 200 ? 'ok  ' : 'FAIL'} what-if batch (10 rows x 4 models) median ${med.toFixed(1)} ms`)
if (med >= 200) fail++
process.exit(fail ? 1 : 0)
