// ?film=1 : deterministic, frame stepped demo film. The recorder (output/record.mjs) calls
// window.__film.goto(i) and window.__film.step(1/30) and screenshots every frame.
import type { App } from './main'
import { ease } from './main'
import { POSES, type Pose, type VesselKey } from './heart'

export const FILM_PATIENT = 25
const lerp = (a: number, b: number, u: number) => a + (b - a) * u
const pc = (a: App, k: 'Cath' | 'LAD' | 'LCX' | 'RCA') => (a.probs ? Math.round(a.probs[k] * 100) + '%' : '')
const pb = (a: App, k: 'Cath' | 'LAD' | 'LCX' | 'RCA') => (a.baseProbs ? Math.round(a.baseProbs[k] * 100) + '%' : '')
const mixPose = (a: Pose, b: Pose, u: number): Pose => {
  let daz = b.az - a.az
  while (daz > Math.PI) daz -= 2 * Math.PI
  while (daz < -Math.PI) daz += 2 * Math.PI
  return { az: a.az + daz * u, el: lerp(a.el, b.el, u), dist: lerp(a.dist, b.dist, u), tx: lerp(a.tx, b.tx, u), ty: lerp(a.ty, b.ty, u), tz: lerp(a.tz, b.tz, u) }
}
const seg = (t: number, a: number, b: number) => ease(Math.min(1, Math.max(0, (t - a) / (b - a))))

type Pt = { x: number; y: number }
type Target = string | ((app: App) => Pt)
interface Ev { at: number; run: (app: App) => unknown }
interface Tw { a: number; b: number; run: (app: App, u: number) => unknown }
interface Chapter {
  name: string; length: number
  captions: [number, number, string | ((app: App) => string)][]
  events: Ev[]; tweens: Tw[]
  cam: (t: number, app: App) => Pose
  cursor: [number, Target | null][] // time, target (null hides)
  clicks: number[]
}

function el(sel: string): Pt {
  const r = document.querySelector(sel)!.getBoundingClientRect()
  return { x: r.left + r.width / 2, y: r.top + r.height / 2 }
}
function thumb(key: string, v: number): Pt {
  const inp = document.querySelector(`.ctl[data-key="${key}"] input`) as HTMLInputElement
  const r = inp.getBoundingClientRect()
  const u = (v - +inp.min) / (+inp.max - +inp.min)
  return { x: r.left + 8 + u * (r.width - 16), y: r.top + r.height / 2 }
}
function label(k: VesselKey): Pt { return el(`.vlabel[data-k="${k}"]`) }
function orbit(base: Pose, t: number, rate: number): Pose { return { ...base, az: base.az + t * rate } }

export function buildChapters(app: App): Chapter[] {
  const P = FILM_PATIENT
  const pat = () => app.patients[P - 1]
  const fi = (k: string) => app.fIndex.get(k)!
  const fbs0 = () => pat().x[fi('FBS')]
  const FBS_TO = 90
  return [
    {
      name: 'A heart, not a number', length: 20,
      captions: [[1.5, 9, 'A risk score is a number'], [10, 19, 'HeartLens shows where the risk is']],
      events: [{ at: 0, run: a => { a.setView('patient'); a.select(null); return a.setPatient(null) } }],
      tweens: [], cursor: [[0, null]], clicks: [],
      cam: t => mixPose({ ...POSES.wide, az: -1.2 }, POSES.hero, seg(t, 0, 20)),
    },
    {
      name: 'Pick a patient', length: 34,
      captions: [[1, 8.5, `Patient ${P}, one of 303 real patients`],
        [9.5, 15.5, a => `Left anterior descending: ${pc(a, 'LAD')} probability of stenosis`],
        [16.5, 22.5, a => `Left circumflex: ${pc(a, 'LCX')}`],
        [23.5, 29.5, a => `Right coronary: ${pc(a, 'RCA')}`], [30, 33.8, 'Green is low. Red is high.']],
      events: [
        { at: 4, run: a => a.setPatient(P) },
        { at: 0.01, run: () => { (document.querySelector('#pick') as HTMLSelectElement).focus() } },
      ],
      tweens: [],
      cursor: [[0, '#pick'], [3.2, '#pick'], [6, null]],
      clicks: [3.4],
      cam: t => t < 15 ? mixPose(POSES.hero, POSES.front, seg(t, 4, 10)) : t < 22 ? mixPose(POSES.front, POSES.LCX, seg(t, 15, 19))
        : t < 29 ? mixPose(POSES.LCX, POSES.RCA, seg(t, 22, 26)) : mixPose(POSES.RCA, POSES.front, seg(t, 29, 33)),
    },
    {
      name: 'Why this artery', length: 40,
      captions: [[1, 8, 'Click the LAD'], [9, 19, 'SHAP: what pushed this prediction'], [20, 30, 'Red raises risk. Green lowers it.'], [31, 39.5, 'Compared with all 303 patients']],
      events: [{ at: 2.6, run: a => a.select('LAD') }],
      tweens: [{ a: 31, b: 34, run: (_a, u) => { document.querySelector('#right')!.scrollTop = 0 * u } }],
      cursor: [[0, '#pick'], [2.3, () => label('LAD')], [5, () => label('LAD')], [7.5, '#detail .wf'], [30, '#detail .pct'], [39, '#detail .pct']],
      clicks: [2.5],
      cam: t => orbit(mixPose(POSES.front, POSES.LAD, seg(t, 2.6, 6)), Math.max(0, t - 6), 0.004),
    },
    {
      name: 'What if', length: 44,
      captions: [[1, 10, 'What if blood sugar were controlled?'], [11, 21.5, a => `LAD ${pb(a, 'LAD')} to ${pc(a, 'LAD')}. Sugar is not a big driver here.`],
        [23, 30, 'Now remove typical chest pain'], [30.5, 37, a => `LAD ${pb(a, 'LAD')} to ${pc(a, 'LAD')}. The artery cools.`], [38.5, 43.5, 'Four models rerun in under a millisecond']],
      events: [
        { at: 22.5, run: a => a.setFeature('Typical Chest Pain', 0) },
        { at: 38, run: a => a.resetEdits() },
      ],
      tweens: [{ a: 3, b: 10, run: (a, u) => a.setFeature('FBS', Math.round(lerp(fbs0(), FBS_TO, u))) }],
      cursor: [[0, '#detail .pct'], [2.2, () => thumb('FBS', fbs0())], [3, () => thumb('FBS', fbs0())],
        ...Array.from({ length: 15 }, (_, i) => [3 + (i + 1) * 0.5, () => thumb('FBS', lerp(fbs0(), FBS_TO, ease(Math.min(1, (i + 1) / 14))))] as [number, Target]),
        [12, () => thumb('FBS', FBS_TO)], [14, '#detail .wf'], [20.5, '.tog[data-key="Typical Chest Pain"]'], [23, '.tog[data-key="Typical Chest Pain"]'], [26, '#vlist'], [36.5, '#reset'], [39, '#reset'], [41, null]],
      clicks: [22.4, 37.9],
      cam: t => orbit(POSES.LAD, t, 0.003),
    },
    {
      name: 'How good is it', length: 50,
      captions: [[1.5, 11, '5 x 5 repeated cross-validation'], [12, 23, 'Tuned inside each fold. No target leakage.'], [24, 36, 'Three model families compared per target'], [37, 49, 'Calibrated: predicted matches observed']],
      events: [{ at: 1.2, run: a => a.setView('evidence') }],
      tweens: [{ a: 27, b: 35, run: (_a, u) => { document.querySelector('#overlay')!.scrollTop = 900 * ease(u) } }],
      cursor: [[0, null], [0.3, '#tab-evidence'], [1.2, '#tab-evidence'], [4, '#overlay table.m tbody tr:first-child td:nth-child(2)'], [12, '#overlay table.m tbody tr:nth-child(2) td:nth-child(2)'], [22, null]],
      clicks: [1.1],
      cam: t => orbit(POSES.LAD, 44 + t, 0.003),
    },
    {
      name: 'How it works', length: 36,
      captions: [[1.5, 12, 'Python trains. The browser predicts.'], [13, 24, 'ONNX models run on any laptop, no GPU'], [25, 35, 'No server. No data leaves the page.']],
      events: [{ at: 1.2, run: a => a.setView('arch') }],
      tweens: [],
      cursor: [[0, null], [0.3, '#tab-arch'], [1.2, '#tab-arch'], [3, null]],
      clicks: [1.1],
      cam: t => orbit(POSES.LAD, 94 + t, 0.003),
    },
    {
      name: 'Limitations', length: 30,
      captions: [[1.5, 13, '303 patients. One centre. No external validation.'], [14, 29, 'Decision support research only. Not a medical device.']],
      events: [{ at: 1.2, run: a => a.setView('limits') }],
      tweens: [],
      cursor: [[0, null], [0.3, '#tab-limits'], [1.2, '#tab-limits'], [3, null]],
      clicks: [1.1],
      cam: t => orbit(POSES.LAD, 130 + t, 0.003),
    },
    {
      name: 'Risk you can see', length: 26,
      captions: [[3, 22, 'HeartLens. Risk you can see.']],
      events: [{ at: 1.2, run: a => { a.setView('patient'); a.select(null) } }],
      tweens: [],
      cursor: [[0, null], [0.3, '#tab-patient'], [1.2, '#tab-patient'], [3, null]],
      clicks: [1.1],
      cam: t => orbit(mixPose(POSES.LAD, POSES.hero, seg(t, 1.2, 7)), Math.max(0, t - 1.2), 0.012),
    },
  ]
}

export function installFilm(app: App) {
  document.body.classList.add('film')
  app.heart.scripted = true
  app.heart.renderer.setPixelRatio(1)
  const chapters = buildChapters(app)
  const cursor = document.getElementById('cursor')!
  cursor.innerHTML = `<svg viewBox="0 0 24 24"><path d="M4 2 L4 19 L8.5 15 L11.5 22 L14.5 20.8 L11.6 14 L18 14 Z" fill="#fff" stroke="#111" stroke-width="1.3" stroke-linejoin="round"/></svg><span class="ring"></span>`
  const ring = cursor.querySelector('.ring') as HTMLElement
  const cap = document.getElementById('caption')!
  let ci = 0, t = 0
  let curPos: Pt = { x: 960, y: 540 }

  const resolve = (tg: Target | null): Pt | null => {
    if (tg == null) return null
    if (typeof tg === 'function') return tg(app)
    if (typeof tg === 'string') { if (!tg) return null; return document.querySelector(tg) ? el(tg) : null }
    return null
  }
  const cursorAt = (c: Chapter, time: number): { p: Pt | null; vis: number } => {
    const ks = c.cursor.filter(k => typeof k[1] !== 'object' || k[1] === null)
    let prev = ks[0]
    for (const k of ks) { if (k[0] <= time) prev = k; else break }
    const next = ks.find(k => k[0] > time)
    const a = resolve(prev?.[1] ?? null)
    if (!a) return { p: null, vis: 0 }
    if (!next) return { p: a, vis: 1 }
    const b = resolve(next[1])
    if (!b) {
      const fade = Math.max(0, 1 - (time - prev[0]) / 0.6)
      return { p: a, vis: next[0] - time < 0.6 ? Math.max(0, (next[0] - time) / 0.6) : fade > 0 && prev[0] === 0 ? 1 : 1 }
    }
    const u = ease(Math.min(1, (time - prev[0]) / Math.max(0.001, next[0] - prev[0])))
    return { p: { x: lerp(a.x, b.x, u), y: lerp(a.y, b.y, u) }, vis: 1 }
  }

  const applyStatic = async (c: Chapter, time: number) => {
    app.heart.applyPose(c.cam(time, app))
    let op = 0, text = ''
    for (const [a, b, s] of c.captions) if (time >= a - 0.01 && time <= b) { text = typeof s === 'function' ? s(app) : s; op = Math.min(1, (time - a) / 0.45, (b - time) / 0.45) }
    cap.innerHTML = text ? `<span>${text}</span>` : ''
    cap.style.opacity = String(Math.max(0, op))
    const cu = cursorAt(c, time)
    if (cu.p && cu.vis > 0) {
      curPos = cu.p
      cursor.style.display = 'block'
      cursor.style.opacity = String(cu.vis)
      cursor.style.transform = `translate(${curPos.x - 4}px, ${curPos.y - 2}px)`
    } else cursor.style.display = 'none'
    let r = 0
    for (const k of c.clicks) if (time >= k && time < k + 0.5) r = (time - k) / 0.5
    ring.style.opacity = r ? String(1 - r) : '0'
    ring.style.transform = `scale(${0.4 + r * 1.2})`
  }

  const runRange = async (c: Chapter, t0: number, t1: number) => {
    // events and tween updates are applied in time order, so seeking replays a chapter exactly
    const acts: [number, () => unknown][] = []
    for (const e of c.events) if (e.at > t0 && e.at <= t1) acts.push([e.at, () => e.run(app)])
    for (const w of c.tweens) if (t1 >= w.a && t0 < w.b) {
      const at = Math.min(t1, w.b)
      acts.push([at, () => w.run(app, Math.min(1, Math.max(0, (at - w.a) / (w.b - w.a))))])
    }
    acts.sort((x, y) => x[0] - y[0])
    for (const [, f] of acts) { await f(); await app.settle() }
  }

  const goto = async (i: number) => {
    app.setView('patient'); app.select(null); await app.setPatient(null)
    document.querySelector('#right')!.scrollTop = 0
    for (let j = 0; j < i; j++) await runRange(chapters[j], -1, chapters[j].length)
    ci = i; t = 0
    await runRange(chapters[i], -1, 0)
    // let the colour lerps settle so a chapter start looks like the end of the previous one
    for (let k = 0; k < (i ? 60 : 1); k++) app.heart.update(1 / 30)
    await applyStatic(chapters[i], 0)
    app.frame(0)
  }

  const step = async (dt = 1 / 30, n = 1) => {
    for (let k = 0; k < n; k++) {
      const c = chapters[ci]
      const t0 = t
      t += dt
      if (t > c.length && ci < chapters.length - 1) { ci++; t = t - c.length; await runRange(chapters[ci], -1, t) }
      else await runRange(c, t0, t)
      await applyStatic(chapters[ci], t)
      app.frame(dt)
    }
  }

  ;(window as unknown as { __film: unknown }).__film = {
    ready: true,
    chapters: chapters.map(c => ({ name: c.name, length: c.length })),
    goto, step,
    state: () => ({ chapter: ci, t, probs: app.probs }),
  }
  goto(0)
}
