// Browser inference: four ONNX models (one per target) run in onnxruntime-web (WASM, single thread),
// followed by the calibration map exported from Python (Platt or isotonic).
import * as ort from 'onnxruntime-web/wasm'
import type { Calibration, Meta, Probs, TargetKey } from './types'
import { TARGETS } from './types'

ort.env.wasm.numThreads = 1

const sessions = new Map<TargetKey, ort.InferenceSession>()
let meta: Meta

export async function loadModels(m: Meta) {
  meta = m
  await Promise.all(m.targets.map(async t => {
    const buf = await (await fetch(t.onnx)).arrayBuffer()
    sessions.set(t.key, await ort.InferenceSession.create(new Uint8Array(buf), { executionProviders: ['wasm'] }))
  }))
}

export function calibrate(c: Calibration, p: number): number {
  if (c.type === 'platt') {
    const q = Math.min(Math.max(p, 1e-6), 1 - 1e-6)
    return 1 / (1 + Math.exp(-(c.a * Math.log(q / (1 - q)) + c.b)))
  }
  const { x, y } = c
  if (p <= x[0]) return y[0]
  if (p >= x[x.length - 1]) return y[y.length - 1]
  let lo = 0, hi = x.length - 1
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (x[mid] <= p) lo = mid; else hi = mid }
  const f = x[hi] === x[lo] ? 0 : (p - x[lo]) / (x[hi] - x[lo])
  return y[lo] + f * (y[hi] - y[lo])
}

async function rawProbs(key: TargetKey, rows: Float32Array, n: number): Promise<Float32Array> {
  const s = sessions.get(key)!
  const nf = meta.features.length
  const out = await s.run({ input: new ort.Tensor('float32', rows, [n, nf]) })
  for (const name of s.outputNames) {
    const t = out[name]
    if (t.dims.length === 2 && t.dims[1] === 2) {
      const d = t.data as Float32Array
      const r = new Float32Array(n)
      for (let i = 0; i < n; i++) r[i] = d[i * 2 + 1]
      return r
    }
  }
  throw new Error('no probability output for ' + key)
}

/** Calibrated probabilities for a batch of rows. Returns one Probs per row. */
export async function predictBatch(rows: number[][]): Promise<Probs[]> {
  const nf = meta.features.length
  const flat = new Float32Array(rows.length * nf)
  rows.forEach((r, i) => flat.set(r, i * nf))
  const res: Probs[] = rows.map(() => ({ Cath: 0, LAD: 0, LCX: 0, RCA: 0 }))
  await Promise.all(TARGETS.map(async k => {
    const p = await rawProbs(k, flat, rows.length)
    const cal = meta.targets.find(t => t.key === k)!.calibration
    for (let i = 0; i < rows.length; i++) res[i][k] = calibrate(cal, p[i])
  }))
  return res
}

export interface WhatIf {
  probs: Probs
  /** single feature reversal: p(edited) minus p(edited with only this feature set back to the patient value) */
  approx: Record<TargetKey, { key: string; value: number }[]>
  ms: number
}

/** Runs the edited row plus one row per edited feature reverted to baseline, in one batch. */
export async function predictWhatIf(base: number[], edited: number[]): Promise<WhatIf> {
  const t0 = performance.now()
  const changed = edited.map((v, i) => (Math.abs(v - base[i]) > 1e-9 ? i : -1)).filter(i => i >= 0)
  const rows = [edited, ...changed.map(i => { const r = edited.slice(); r[i] = base[i]; return r })]
  const out = await predictBatch(rows)
  const approx = {} as WhatIf['approx']
  for (const k of TARGETS) {
    approx[k] = changed.map((fi, j) => ({ key: meta.features[fi].key, value: out[0][k] - out[j + 1][k] }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
  }
  return { probs: out[0], approx, ms: performance.now() - t0 }
}
