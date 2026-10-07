import './style.css'
import { HeartView, POSES, riskCss, type VesselKey } from './heart'
import { loadModels, predictBatch, predictWhatIf, type WhatIf } from './model'
import { renderArch, renderEvidence, renderLimits } from './panels'
import { TARGETS, VESSELS, type Meta, type Patient, type Probs, type TargetKey } from './types'
import { installFilm } from './film'

const $ = <T extends HTMLElement = HTMLElement>(s: string, root: ParentNode = document) => root.querySelector(s) as T
const pct = (p: number) => `${Math.round(p * 100)}%`
const esc = (s: string) => s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]!))

export const KEY_SLIDERS = ['Age', 'BP', 'FBS', 'LDL', 'EF-TTE', 'BMI']
export const KEY_TOGGLES = ['Typical Chest Pain', 'DM', 'Tinversion']
const TARGET_NAME: Record<TargetKey, string> = { Cath: 'CAD', LAD: 'LAD', LCX: 'LCX', RCA: 'RCA' }

export class App {
  meta!: Meta
  patients!: Patient[]
  heart!: HeartView
  cohort = {} as Record<TargetKey, number[]>
  pid: number | null = null
  base: number[] = []
  edited: number[] = []
  baseProbs: Probs | null = null
  probs: Probs | null = null
  whatIf: WhatIf | null = null
  selected: VesselKey | null = null
  view = 'patient'
  private running: Promise<void> | null = null
  private dirty = false
  fIndex = new Map<string, number>()

  async init() {
    const base = document.baseURI
    const [meta, patients] = await Promise.all([
      fetch(new URL('data/meta.json', base)).then(r => r.json()),
      fetch(new URL('data/patients.json', base)).then(r => r.json()),
    ])
    this.meta = meta; this.patients = patients
    meta.features.forEach((f: { key: string }, i: number) => this.fIndex.set(f.key, i))
    for (const t of TARGETS) this.cohort[t] = patients.map((p: Patient) => p.pred[t]).sort((a: number, b: number) => a - b)
    this.heart = new HeartView($('#gl'))
    this.heart.onPick = k => this.select(k)
    await Promise.all([this.heart.load(new URL('anatomy/heart.glb', base).href), loadModels(meta)])
    this.buildLeft(); this.buildRight(); this.buildLabels()
    document.querySelectorAll<HTMLButtonElement>('nav button').forEach(b => b.addEventListener('click', () => this.setView(b.dataset.view!)))
    $('nav').addEventListener('keydown', e => {
      const tabs = [...document.querySelectorAll<HTMLButtonElement>('nav button')]
      const i = tabs.indexOf(document.activeElement as HTMLButtonElement)
      if (i < 0) return
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        const j = (i + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length
        tabs[j].focus(); this.setView(tabs[j].dataset.view!); e.preventDefault()
      }
    })
    window.addEventListener('keydown', e => {
      if (e.key === 'Escape') { if (this.view !== 'patient') this.setView('patient'); else this.select(null) }
    })
    $('#loading').remove()
  }

  // ---------- state ----------
  async setPatient(id: number | null) {
    this.pid = id
    if (id == null) {
      this.base = []; this.edited = []; this.probs = this.baseProbs = null; this.whatIf = null
      this.heart.setRisk(null); this.renderLeft(); this.renderRight(); return
    }
    const p = this.patients[id - 1]
    this.base = p.x.slice(); this.edited = p.x.slice()
    const [pr] = await predictBatch([this.base])
    this.baseProbs = pr; this.probs = pr; this.whatIf = null
    this.heart.setRisk(pr)
    this.renderLeft(); this.renderRight()
  }

  setFeature(key: string, value: number) {
    const i = this.fIndex.get(key)!
    if (this.edited[i] === value) return this.settle()
    this.edited[i] = value
    this.syncControl(key)
    return this.infer()
  }

  resetEdits() { this.edited = this.base.slice(); this.renderLeft(); return this.infer() }

  /** coalescing inference: at most one run in flight, the latest edit always wins */
  infer(): Promise<void> {
    this.dirty = true
    if (!this.running) {
      this.running = (async () => {
        while (this.dirty) {
          this.dirty = false
          const w = await predictWhatIf(this.base, this.edited.slice())
          this.whatIf = w; this.probs = w.probs
          this.heart.setRisk(w.probs)
          this.renderRight()
          $('#perf').textContent = `In-browser inference ${w.ms.toFixed(1)} ms (4 ONNX models, WASM)`
        }
        this.running = null
      })()
    }
    return this.running
  }
  async settle() { while (this.running) await this.running }
  get isEdited() { return this.edited.some((v, i) => Math.abs(v - this.base[i]) > 1e-9) }

  select(k: VesselKey | null) {
    this.selected = k
    this.heart.select(k)
    if (!this.heart.scripted && k) { this.heart.lastInteract = this.heart.time; this.flyTo(k) }
    this.renderRight()
  }
  private fly: { from: typeof POSES.hero; to: typeof POSES.hero; t: number } | null = null
  flyTo(k: string) { this.fly = { from: { ...this.heart.pose, ...this.currentPose() }, to: POSES[k], t: 0 } }
  currentPose() {
    const c = this.heart.camera.position, t = this.heart.controls.target
    const d = c.clone().sub(t)
    const dist = d.length()
    return { az: Math.atan2(d.x, d.z), el: Math.asin(d.y / dist), dist, tx: t.x, ty: t.y, tz: t.z }
  }

  setView(v: string) {
    this.view = v
    document.querySelectorAll<HTMLButtonElement>('nav button').forEach(b => b.setAttribute('aria-selected', String(b.dataset.view === v)))
    const ov = $('#overlay')
    if (v === 'patient') { ov.hidden = true; ov.innerHTML = ''; return }
    ov.hidden = false
    ov.innerHTML = v === 'evidence' ? renderEvidence(this.meta) : v === 'arch' ? renderArch(this.meta) : renderLimits(this.meta)
    ov.scrollTop = 0
  }

  percentile(t: TargetKey, p: number) {
    const a = this.cohort[t]
    let n = 0
    for (const v of a) if (v < p) n++
    return Math.round((n / a.length) * 100)
  }

  // ---------- left panel ----------
  buildLeft() {
    const L = $('#left')
    const opts = this.patients.map(p => {
      const age = p.x[this.fIndex.get('Age')!], male = p.x[this.fIndex.get('Sex')!] === 1
      return `<option value="${p.id}">Patient ${p.id}, ${age} y, ${male ? 'M' : 'F'}</option>`
    }).join('')
    L.innerHTML = `
      <div class="card">
        <h2>Patient</h2>
        <div class="row">
          <button class="btn" id="prev" aria-label="Previous patient">&#8249;</button>
          <select class="btn" id="pick" aria-label="Choose one of 303 patients"><option value="">Choose a patient</option>${opts}</select>
          <button class="btn" id="next" aria-label="Next patient">&#8250;</button>
        </div>
        <div id="summary"></div>
      </div>
      <div class="card" id="whatif">
        <div class="row"><h2 style="margin:0">What if</h2><span class="sp"></span><button class="btn small" id="reset">Reset to record</button></div>
        <div id="sliders"></div>
        <div id="toggles" style="margin-top:6px"></div>
        <details id="all"><summary>All 55 features</summary><div class="allf" id="allf"></div></details>
        <p class="note">Yellow ticks mark the patient's recorded value. Every change reruns all four models in your browser.</p>
      </div>`
    $('#pick').addEventListener('change', e => { const v = (e.target as HTMLSelectElement).value; this.setPatient(v ? +v : null) })
    $('#prev').addEventListener('click', () => this.setPatient(Math.max(1, (this.pid ?? 2) - 1)))
    $('#next').addEventListener('click', () => this.setPatient(Math.min(303, (this.pid ?? 0) + 1)))
    $('#reset').addEventListener('click', () => this.resetEdits())
    const S = $('#sliders')
    for (const k of KEY_SLIDERS) {
      const f = this.meta.features[this.fIndex.get(k)!]
      const id = 'sl-' + k.replace(/\W/g, '')
      S.insertAdjacentHTML('beforeend', `<div class="ctl" data-key="${k}"><div class="row"><label for="${id}">${f.label}</label><span class="sp"></span><output></output></div>
        <div class="rng"><input type="range" id="${id}" min="${f.min}" max="${f.max}" step="${f.step}" aria-describedby="${id}-b"><i class="tick"></i></div>
        <div class="base" id="${id}-b"></div></div>`)
      const inp = $(`#${id}`) as HTMLInputElement
      inp.addEventListener('input', () => { if (this.pid) this.setFeature(k, +inp.value) })
    }
    const T = $('#toggles')
    for (const k of KEY_TOGGLES) {
      const f = this.meta.features[this.fIndex.get(k)!]
      T.insertAdjacentHTML('beforeend', `<div class="tog" role="switch" tabindex="0" aria-checked="false" data-key="${k}"><span class="sw"></span><span>${f.label}</span></div>`)
    }
    T.querySelectorAll<HTMLElement>('.tog').forEach(el => {
      const flip = () => { if (!this.pid) return; const i = this.fIndex.get(el.dataset.key!)!; this.setFeature(el.dataset.key!, this.edited[i] ? 0 : 1) }
      el.addEventListener('click', flip)
      el.addEventListener('keydown', e => { if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); flip() } })
    })
    const A = $('#allf')
    this.meta.features.forEach((f, i) => {
      const id = 'f-' + i
      const unit = f.unit ? ` <span class="dim">${esc(f.unit)}</span>` : ''
      const inp = f.kind === 'bin' ? `<input type="checkbox" id="${id}" data-key="${f.key}">`
        : `<input type="number" id="${id}" data-key="${f.key}" min="${f.min}" max="${f.max}" step="${f.step}">`
      A.insertAdjacentHTML('beforeend', `<label for="${id}">${esc(f.label)}${unit}</label>${inp}`)
    })
    A.querySelectorAll<HTMLInputElement>('input').forEach(inp => inp.addEventListener('change', () => {
      if (!this.pid) return
      const v = inp.type === 'checkbox' ? (inp.checked ? 1 : 0) : +inp.value
      if (Number.isFinite(v)) this.setFeature(inp.dataset.key!, v)
    }))
    this.renderLeft()
  }

  syncControl(key: string) {
    const i = this.fIndex.get(key)!
    const v = this.edited[i], b = this.base[i]
    const f = this.meta.features[i]
    const ctl = document.querySelector<HTMLElement>(`.ctl[data-key="${key}"]`)
    if (ctl) {
      const inp = $('input', ctl) as HTMLInputElement
      if (document.activeElement !== inp || inp.value !== String(v)) inp.value = String(v)
      $('output', ctl).textContent = this.pid ? `${fmt(v)} ${f.unit}` : ''
      const tick = $('.tick', ctl)
      tick.style.left = `calc(${((b - f.min) / (f.max - f.min)) * 100}% - 1px)`
      tick.style.display = this.pid ? '' : 'none'
      $('.base', ctl).textContent = this.pid ? `Recorded ${fmt(b)} ${f.unit}` : ''
      ctl.classList.toggle('changed', !!this.pid && Math.abs(v - b) > 1e-9)
      inp.disabled = !this.pid
    }
    const tog = document.querySelector<HTMLElement>(`.tog[data-key="${key}"]`)
    if (tog) {
      tog.setAttribute('aria-checked', String(!!v))
      tog.classList.toggle('changed', !!this.pid && v !== b)
      tog.setAttribute('aria-disabled', String(!this.pid))
    }
    const all = document.querySelector<HTMLInputElement>(`#allf input[data-key="${CSS.escape(key)}"]`)
    if (all) { if (all.type === 'checkbox') all.checked = !!v; else all.value = String(v); all.disabled = !this.pid }
  }

  renderLeft() {
    ($('#pick') as HTMLSelectElement).value = this.pid ? String(this.pid) : ''
    for (const f of this.meta.features) this.syncControl(f.key)
    const S = $('#summary')
    if (!this.pid) { S.innerHTML = `<p class="note">Pick any of the 303 patients from the UCI cohort. All inference runs locally in this page.</p>`; return }
    const p = this.patients[this.pid - 1]
    const g = (k: string) => this.base[this.fIndex.get(k)!]
    const flags = [['DM', 'Diabetes'], ['HTN', 'Hypertension'], ['Current Smoker', 'Smoker'], ['Typical Chest Pain', 'Typical chest pain'], ['Tinversion', 'T inversion'], ['DLP', 'Dyslipidaemia'], ['FH', 'Family history']]
    S.innerHTML = `<div class="chips"><span class="chip">${g('Age')} years</span><span class="chip">${g('Sex') ? 'Male' : 'Female'}</span><span class="chip">BMI ${fmt(g('BMI'))}</span><span class="chip">EF ${g('EF-TTE')}%</span>
      ${flags.map(([k, l]) => g(k) ? `<span class="chip on">${l}</span>` : '').join('')}</div>
      <div class="truth" aria-label="Angiography result in the dataset">${TARGETS.map(t => `<div><b>${t === 'Cath' ? 'Cath' : t}</b>${p.truth[t] ? (t === 'Cath' ? 'CAD' : 'Stenotic') : 'Normal'}</div>`).join('')}</div>
      <p class="note">Bottom row: the patient's angiography result from the dataset, shown for comparison.</p>`
  }

  // ---------- right panel ----------
  buildRight() {
    $('#right').innerHTML = `
      <div class="card" id="cad"></div>
      <div class="card"><h2>Coronary arteries</h2><div id="vlist">${VESSELS.map(k => `<button class="vrow" data-k="${k}" aria-pressed="false"></button>`).join('')}</div></div>
      <div class="card" id="detail"></div>`
    document.querySelectorAll<HTMLButtonElement>('.vrow').forEach(b => b.addEventListener('click', () => this.select(this.selected === b.dataset.k ? null : b.dataset.k as VesselKey)))
    this.renderRight()
  }

  renderRight() {
    const P = this.probs, B = this.baseProbs
    const cad = $('#cad')
    if (!P || !this.pid) {
      cad.innerHTML = `<h2>Coronary artery disease</h2><div class="big dim">--</div><p class="note">Choose a patient to paint the arteries.</p>`
    } else {
      const d = B && this.isEdited ? P.Cath - B.Cath : 0
      const pc = this.percentile('Cath', P.Cath)
      cad.innerHTML = `<div class="row"><h2 style="margin:0">Coronary artery disease</h2>${this.isEdited ? '<span class="badge">what-if</span>' : ''}</div>
        <div class="row" style="margin-top:8px"><div class="big" style="color:${riskCss(P.Cath)}">${pct(P.Cath)}</div>${deltaHtml(d)}</div>
        <div class="muted small" style="margin-top:6px">Calibrated probability of CAD on angiography</div>
        <div class="pct"><div class="row small"><span>Higher than <b>${pc}%</b> of the 303 patients</span></div>${this.histo('Cath', P.Cath)}</div>
        ${this.isEdited ? '' : `<div class="note">Cross-validated estimate for this record: ${pct(this.patients[this.pid - 1].oof.Cath)} (the model never saw this patient when making it).</div>`}`
    }
    document.querySelectorAll<HTMLButtonElement>('.vrow').forEach(b => {
      const k = b.dataset.k as TargetKey
      const t = this.meta.targets.find(x => x.key === k)!
      b.setAttribute('aria-pressed', String(this.selected === k))
      if (!P) { b.innerHTML = `<b>${k}</b><span class="bar"><i style="width:0"></i></span><span class="v dim">--</span>`; b.setAttribute('aria-label', `${t.label}, no patient`); return }
      const d = B && this.isEdited ? P[k] - B[k] : 0
      b.innerHTML = `<b>${k}</b><span class="bar"><i style="width:${P[k] * 100}%;background:${riskCss(P[k])}"></i></span><span class="v" style="color:${riskCss(P[k])}">${pct(P[k])}${Math.abs(d) >= 0.005 ? `<small class="${d > 0 ? 'delta up' : 'delta down'}" style="margin:0">${d > 0 ? '+' : ''}${Math.round(d * 100)} pts</small>` : `<small class="dim">${{ LAD: 'anterior desc.', LCX: 'circumflex', RCA: 'right coronary' }[k as 'LAD']}</small>`}</span>`
      b.setAttribute('aria-label', `${t.label}: ${pct(P[k])} predicted probability of stenosis`)
    })
    this.renderDetail()
  }

  histo(t: TargetKey, p: number) {
    const bins = new Array(20).fill(0)
    for (const v of this.cohort[t]) bins[Math.min(19, Math.floor(v * 20))]++
    const mx = Math.max(...bins)
    const bars = bins.map((n, i) => `<rect x="${i * 5 + 0.4}" y="${26 - (n / mx) * 22}" width="4.2" height="${(n / mx) * 22}" fill="${i / 20 <= p ? '#3a4356' : '#262c39'}"/>`).join('')
    return `<div class="pctbar"><svg viewBox="0 0 100 26" preserveAspectRatio="none" aria-hidden="true">${bars}<rect x="${p * 100 - 0.4}" y="0" width="0.8" height="26" fill="${riskCss(p)}"/></svg></div>`
  }

  renderDetail() {
    const D = $('#detail')
    const k = this.selected
    if (!k || !this.probs || !this.pid) {
      D.innerHTML = `<h2>Why</h2><p class="note" style="margin:0">${this.pid ? 'Click an artery on the heart, or pick one from the list above, to see what drives its prediction.' : 'No patient selected.'}</p>`
      return
    }
    const t = this.meta.targets.find(x => x.key === k)!
    const P = this.probs[k]
    const cv = this.meta.cv[k][t.model][this.meta.selection[k].cal]
    const p = this.patients[this.pid - 1]
    const shap = p.shap[k]
    const items = shap.map((v, i) => ({ i, v })).sort((a, b) => Math.abs(b.v) - Math.abs(a.v))
    const top = items.slice(0, 8)
    const rest = items.slice(8).reduce((s, x) => s + x.v, 0)
    const lines = top.map(x => ({ name: this.meta.features[x.i].label, val: fmtFeat(this.meta.features[x.i], this.base[x.i]), v: x.v }))
    lines.push({ name: `${items.length - 8} other features`, val: '', v: rest })
    const shapHtml = waterfall(lines)
    const pc = this.percentile(k, P)
    const tip = `Approximate contribution, computed in your browser: for each edited feature, the model is rerun with only that feature set back to the recorded value. The bar is the change in calibrated probability. It is a single feature perturbation, not a SHAP value, and the bars do not have to add up to the total change when features interact.`
    let whatIfHtml = ''
    if (this.isEdited && this.whatIf && this.baseProbs) {
      const a = this.whatIf.approx[k].map(x => {
        const f = this.meta.features[this.fIndex.get(x.key)!]
        const i = this.fIndex.get(x.key)!
        return { name: f.label, val: `${fmtFeat(f, this.base[i])} → ${fmtFeat(f, this.edited[i])}`, v: x.value }
      })
      whatIfHtml = `<div style="margin-top:14px"><div class="row"><h2 style="margin:0">What-if change</h2><span class="badge">approximate contribution</span>
        <button class="info" aria-label="What approximate contribution means">i<span class="tip" role="tooltip">${tip}</span></button></div>
        <div class="small muted" style="margin-top:6px">Record ${pct(this.baseProbs[k])} to what-if <b style="color:${riskCss(P)}">${pct(P)}</b></div>
        ${waterfall(a, true)}</div>`
    }
    D.innerHTML = `<div class="row"><h3 style="color:${riskCss(P)}">${t.label} (${k})</h3><span class="sp"></span><span class="big" style="font-size:30px;color:${riskCss(P)}">${pct(P)}</span></div>
      <div class="small muted" style="margin-top:4px">Probability of stenosis. ${t.model_label}, ${this.meta.selection[k].cal === 'iso' ? 'isotonic' : 'Platt'} calibration. CV ROC AUC ${cv.roc_auc.mean.toFixed(2)} (${cv.roc_auc.lo.toFixed(2)} to ${cv.roc_auc.hi.toFixed(2)}).</div>
      <div class="pct small">Higher than <b>${pc}%</b> of the cohort for ${TARGET_NAME[k]}${this.histo(k, P)}</div>
      ${whatIfHtml}
      <div style="margin-top:14px"><div class="row"><h2 style="margin:0">Top drivers ${this.isEdited ? 'for the recorded values' : ''}</h2>
        <button class="info" aria-label="About these SHAP values">i<span class="tip" role="tooltip">SHAP values computed in Python for this dataset record (permutation explainer on the calibrated model, 100 patient background). Each bar is how many percentage points the feature moved this prediction away from the cohort average of ${pct(t.shap_base)}. Red raises risk, green lowers it.</span></button></div>
        <div class="small muted" style="margin-top:6px">Cohort average ${pct(t.shap_base)} to this record ${pct(p.pred[k])}</div>
        ${shapHtml}</div>`
  }

  // ---------- labels ----------
  buildLabels() {
    $('#labels').innerHTML = VESSELS.map(k => `<div class="vlabel" data-k="${k}" role="button" tabindex="-1"><b>${k}</b><em></em><small></small></div>`).join('')
    document.querySelectorAll<HTMLElement>('.vlabel').forEach(el => el.addEventListener('click', () => this.select(el.dataset.k as VesselKey)))
  }

  updateLabels() {
    const anchors = this.heart.anchors()
    const c = this.heart.centerScreen()
    const svg: string[] = []
    const show = this.heart.activeShown
    const W = $('#stage').clientWidth
    // labels sit outside the heart silhouette, left or right of it, de-overlapped vertically
    const placed = anchors.map(a => ({ a, side: a.x < c.x ? -1 : 1, y: a.y }))
    for (const side of [-1, 1]) {
      const g = placed.filter(p => p.side === side).sort((p, q) => p.y - q.y)
      for (let i = 1; i < g.length; i++) if (g[i].y - g[i - 1].y < 58) g[i].y = g[i - 1].y + 58
    }
    for (const { a, side, y } of placed) {
      const el = document.querySelector<HTMLElement>(`.vlabel[data-k="${a.key}"]`)!
      const P = this.probs?.[a.key]
      const col = P != null ? riskCss(this.heart.vessels.get(a.key)!.shown) : '#9aa3b2'
      const vis = Math.max(0.35, Math.min(1, 0.6 + a.facing))
      const bw = el.offsetWidth || 180
      let lx = c.x + side * (c.r + 70 + bw / 2)
      lx = Math.min(W - bw / 2 - 12, Math.max(bw / 2 + 12, lx))
      el.style.left = lx + 'px'; el.style.top = y + 'px'
      el.style.opacity = String(vis)
      el.style.color = col
      el.classList.toggle('sel', this.selected === a.key)
      $('em', el).textContent = P != null && show > 0.02 ? pct(this.heart.vessels.get(a.key)!.shown) : ''
      $('small', el).textContent = { LAD: 'anterior descending', LCX: 'circumflex', RCA: 'right coronary' }[a.key]
      const ex = lx - side * bw / 2
      const mx = ex - side * 28
      svg.push(`<path d="M${a.x},${a.y} L${mx},${y} L${ex},${y}" fill="none" stroke="${col}" stroke-opacity="${0.8 * vis}" stroke-width="1.4"/><circle cx="${a.x}" cy="${a.y}" r="3.5" fill="${col}" fill-opacity="${vis}" stroke="#0b0e14" stroke-width="1.2"/>`)
    }
    $('#leaders').innerHTML = svg.join('')
  }

  frame(dt: number) {
    if (this.fly) {
      this.fly.t = Math.min(1, this.fly.t + dt / 1.4)
      const e = ease(this.fly.t)
      const a = this.fly.from, b = this.fly.to
      let daz = b.az - a.az
      while (daz > Math.PI) daz -= 2 * Math.PI
      while (daz < -Math.PI) daz += 2 * Math.PI
      this.heart.applyPose({ az: a.az + daz * e, el: a.el + (b.el - a.el) * e, dist: a.dist + (b.dist - a.dist) * e, tx: a.tx + (b.tx - a.tx) * e, ty: a.ty + (b.ty - a.ty) * e, tz: a.tz + (b.tz - a.tz) * e })
      if (this.fly.t >= 1) this.fly = null
    }
    this.heart.update(dt)
    this.heart.render()
    this.updateLabels()
  }
}

export const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)
function fmt(v: number) { return Number.isInteger(v) ? String(v) : v.toFixed(1) }
function fmtFeat(f: { kind: string; unit: string }, v: number) {
  if (f.kind === 'bin') return v ? 'yes' : 'no'
  return `${fmt(v)}${f.unit && f.kind === 'num' ? ' ' + f.unit : ''}`
}
function deltaHtml(d: number) {
  if (Math.abs(d) < 0.005) return ''
  return `<span class="delta ${d > 0 ? 'up' : 'down'}">${d > 0 ? '+' : ''}${Math.round(d * 100)} pts vs record</span>`
}
function waterfall(lines: { name: string; val: string; v: number }[], approx = false) {
  const mx = Math.max(0.02, ...lines.map(l => Math.abs(l.v)))
  return `<div class="wf" role="list">${lines.map(l => {
    const w = (Math.abs(l.v) / mx) * 50
    const col = l.v > 0 ? 'var(--up)' : 'var(--down)'
    const style = l.v >= 0 ? `left:50%;width:${w}%` : `left:${50 - w}%;width:${w}%`
    const sign = l.v > 0 ? '+' : l.v < 0 ? '' : ''
    return `<div class="ln" role="listitem" aria-label="${esc(l.name)} ${esc(l.val)}: ${approx ? 'approximate ' : ''}${sign}${(l.v * 100).toFixed(1)} points"><span class="nm">${esc(l.name)} ${l.val ? `<i>${esc(l.val)}</i>` : ''}</span><span class="trk"><span style="${style};background:${col}"></span></span><span class="val" style="color:${col}">${sign}${(l.v * 100).toFixed(1)}</span></div>`
  }).join('')}</div>`
}

const app = new App()
;(window as unknown as { app: App }).app = app
const film = new URLSearchParams(location.search).has('film')
app.init().then(() => {
  if (film) { installFilm(app); return }
  const q = new URLSearchParams(location.search).get('patient')
  if (q) app.setPatient(+q)
  let last = performance.now()
  const loop = (now: number) => {
    const dt = Math.min(0.05, (now - last) / 1000); last = now
    app.frame(dt)
    requestAnimationFrame(loop)
  }
  requestAnimationFrame(loop)
}).catch(err => { $('#loading').textContent = 'Failed to load: ' + err.message; console.error(err) })
