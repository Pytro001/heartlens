import type { Meta, MetricName, Stat, TargetKey } from './types'
import { TARGETS } from './types'

const f2 = (s: Stat) => `${s.mean.toFixed(2)}<small>${s.lo.toFixed(2)} to ${s.hi.toFixed(2)}</small>`
const NAME: Record<TargetKey, string> = { Cath: 'CAD (any vessel)', LAD: 'LAD stenosis', LCX: 'LCX stenosis', RCA: 'RCA stenosis' }
const MODELS = [['logreg', 'Logistic regression'], ['rf', 'Random forest'], ['xgb', 'XGBoost']]

export function renderEvidence(m: Meta): string {
  const cols: [MetricName, string][] = [['roc_auc', 'ROC AUC'], ['pr_auc', 'PR AUC'], ['accuracy', 'Accuracy'], ['precision', 'Precision'], ['recall', 'Recall'], ['f1', 'F1'], ['brier', 'Brier']]
  const rows = TARGETS.map(t => {
    const sel = m.selection[t]
    const r = m.cv[t][sel.kind][sel.cal]
    const mod = MODELS.find(x => x[0] === sel.kind)![1]
    return `<tr><td><b>${NAME[t]}</b><div class="dim small">${mod}, ${sel.cal === 'iso' ? 'isotonic' : 'Platt'}</div></td>${cols.map(([k]) => `<td>${f2(r[k])}</td>`).join('')}</tr>`
  }).join('')
  const cmp = TARGETS.map(t => {
    const best = m.selection[t].kind
    return `<tr><td><b>${NAME[t]}</b></td>${MODELS.map(([k]) => `<td class="${k === best ? 'win' : ''}">${f2(m.cv[t][k].platt.roc_auc)}</td>`).join('')}<td>${(m.targets.find(x => x.key === t)!.prevalence * 100).toFixed(0)}%</td></tr>`
  }).join('')
  const calib = TARGETS.map(t => `<div class="card plot"><h2>${NAME[t]}</h2>${reliability(m, t)}</div>`).join('')
  const conf = TARGETS.map(t => `<div class="card"><h2>${NAME[t]}</h2>${confusion(m.confusion[t], t)}</div>`).join('')
  return `<h1>How good is it</h1>
    <p class="lead">Repeated stratified cross-validation, 5 folds x 5 repeats, on all ${m.dataset.patients} patients. Hyperparameters are tuned by an inner 3 fold grid search inside every training fold. Calibration is fitted inside the training fold only. Numbers are means over 25 test folds; small text is the ${m.protocol.interval}. Threshold 0.5. Every target is predicted without any of the other three target columns.</p>
    <div class="card"><h2>Selected pipeline per target, cross-validated</h2>
      <table class="m"><thead><tr><th>Target</th>${cols.map(c => `<th>${c[1]}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div>
    <div class="grid2">
      <div class="card"><h2>Model comparison, ROC AUC</h2><table class="m"><thead><tr><th>Target</th>${MODELS.map(x => `<th>${x[1]}</th>`).join('')}<th>Positive</th></tr></thead><tbody>${cmp}</tbody></table>
        <p class="note">Winner per target picked by mean CV ROC AUC, then Platt or isotonic by Brier score. Grids: logistic regression C in {0.03, 0.1, 0.3, 1}; random forest depth {none, 6} x min leaf {1, 3}; XGBoost depth {2, 3} x min child weight {1, 3}.</p></div>
      <div class="card"><h2>Confusion matrices, mean counts per repeat</h2><div class="grid2" style="gap:10px">${conf}</div></div>
    </div>
    <h2 class="muted" style="font-size:12px;letter-spacing:0.9px;text-transform:uppercase;margin:6px 0 10px">Calibration, pooled out-of-fold predictions</h2>
    <div class="grid4">${calib}</div>
    <h2 class="muted" style="font-size:12px;letter-spacing:0.9px;text-transform:uppercase;margin:14px 0 10px">Global SHAP, all 303 patients</h2>
    <div class="grid2">${TARGETS.map(t => `<div class="card"><img src="figures/shap_${t.toLowerCase()}.png" alt="SHAP beeswarm for ${NAME[t]}: top 10 features by mean absolute SHAP value" style="width:100%;display:block" loading="lazy"></div>`).join('')}</div>
    <p class="note">Each dot is one patient. Position: how much the feature moved that patient's calibrated probability. Colour: the feature value, red high, blue low.</p>`
}

function reliability(m: Meta, t: TargetKey) {
  const W = 260, H = 220, pad = 28
  const X = (v: number) => pad + v * (W - pad - 8), Y = (v: number) => H - pad - v * (H - pad - 8)
  const path = (pts: { pred: number; obs: number }[]) => pts.map((p, i) => `${i ? 'L' : 'M'}${X(p.pred).toFixed(1)},${Y(p.obs).toFixed(1)}`).join('')
  const c = m.calibration_curves[t]
  const grid = [0, 0.25, 0.5, 0.75, 1].map(v => `<line x1="${X(v)}" y1="${Y(0)}" x2="${X(v)}" y2="${Y(1)}" stroke="#ffffff10"/><line x1="${X(0)}" y1="${Y(v)}" x2="${X(1)}" y2="${Y(v)}" stroke="#ffffff10"/><text x="${X(v)}" y="${H - 10}" fill="#5f697a" font-size="10" text-anchor="middle">${v}</text><text x="${pad - 6}" y="${Y(v) + 3}" fill="#5f697a" font-size="10" text-anchor="end">${v}</text>`).join('')
  const dots = c.calibrated.map(p => `<circle cx="${X(p.pred)}" cy="${Y(p.obs)}" r="${2 + Math.sqrt(p.n) / 6}" fill="#ff8a6b"/>`).join('')
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Reliability diagram for ${t}">${grid}
    <line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="#ffffff40" stroke-dasharray="3 4"/>
    <path d="${path(c.raw)}" fill="none" stroke="#7d8597" stroke-width="1.5" stroke-dasharray="4 3"/>
    <path d="${path(c.calibrated)}" fill="none" stroke="#ff8a6b" stroke-width="2.2"/>${dots}</svg>
    <div class="note" style="margin-top:2px">x predicted, y observed. Orange calibrated, grey dashed uncalibrated.</div>`
}

function confusion(cm: number[][], t: TargetKey) {
  const pos = t === 'Cath' ? 'CAD' : 'Sten.'
  const tot = cm.flat().reduce((a, b) => a + b, 0)
  const cell = (v: number, ok: boolean) => `<div style="background:${ok ? 'rgba(53,208,160,' : 'rgba(255,107,94,'}${0.08 + (v / tot) * 0.9})">${v.toFixed(1)}</div>`
  return `<div class="cm"><div></div><div class="h">pred normal</div><div class="h">pred ${pos}</div>
    <div class="h" style="text-align:left">true normal</div>${cell(cm[0][0], true)}${cell(cm[0][1], false)}
    <div class="h" style="text-align:left">true ${pos}</div>${cell(cm[1][0], false)}${cell(cm[1][1], true)}</div>`
}

export function renderArch(m: Meta): string {
  const box = (x: number, y: number, w: number, h: number, title: string, lines: string[], col: string) =>
    `<g><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="14" fill="#121722" stroke="${col}" stroke-opacity="0.55"/>
     <text x="${x + 18}" y="${y + 30}" fill="${col}" font-size="15" font-weight="650">${title}</text>
     ${lines.map((l, i) => `<text x="${x + 18}" y="${y + 56 + i * 21}" fill="#c8cfda" font-size="13">${l}</text>`).join('')}</g>`
  const arrow = (x1: number, y1: number, x2: number, y2: number) => `<path d="M${x1},${y1} L${x2},${y2}" stroke="#5f697a" stroke-width="2" marker-end="url(#ah)"/>`
  const sizes = m.targets.map(t => `${t.key} ${t.model_label}`).join(', ')
  return `<h1>How it works</h1>
    <p class="lead">Training happens once in Python. Everything the page needs is exported as files, so the app is static: no server, no GPU, no data leaves the browser.</p>
    <div class="card"><svg viewBox="0 0 1400 330" style="width:100%" role="img" aria-label="Architecture diagram">
      <defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#5f697a"/></marker></defs>
      ${box(20, 60, 300, 160, 'Data', ['UCI Z-Alizadeh Sani extension', `${m.dataset.patients} patients, ${m.dataset.features} features`, 'Targets: CAD, LAD, LCX, RCA', 'Licence CC BY 4.0'], '#7cc4ff')}
      ${box(380, 20, 340, 290, 'Python pipeline', ['Leakage rule: drop all targets', '  (unit tested)', '5 x 5 repeated stratified CV', 'Nested 3 fold grid tuning', 'Logistic reg. / Random forest /', '  XGBoost per target', 'Platt or isotonic calibration', 'Permutation SHAP, 303 patients', 'Metrics with 95% intervals', `make train: ${Math.round(m.train_seconds / 60)} min on 4 cores`], '#ff8a6b')}
      ${box(780, 30, 270, 240, 'Exported files', ['4 ONNX models', '  (skl2onnx, onnxmltools)', 'Calibration maps (JSON)', 'SHAP values, 303 x 55 x 4', 'CV metrics, calibration', '  curves, confusion matrices', 'Heart mesh (GLB)', '  BodyParts3D, decimated'], '#f2b84b')}
      ${box(1110, 50, 270, 220, 'Browser', ['onnxruntime-web (WASM)', '4 models per edit, batched', 'Calibration in TypeScript', 'Three.js heart, vertex', '  colour risk ramp per artery', 'SHAP waterfall + what-if', 'Static site, any laptop'], '#35d0a0')}
      ${arrow(320, 150, 375, 150)}${arrow(720, 150, 775, 150)}${arrow(1050, 150, 1105, 150)}
    </svg></div>
    <div class="grid2"><div class="card"><h2>Selected models</h2><p class="small" style="margin:0;line-height:1.6">${sizes}.</p><p class="note">ONNX outputs match scikit-learn and XGBoost to within ${Math.max(...m.targets.map(t => t.onnx_max_abs_err)).toExponential(1)} on all 303 patients.</p></div>
    <div class="card"><h2>Explanations</h2><p class="small" style="margin:0;line-height:1.6">Dataset records: SHAP values from Python, in percentage points of the calibrated probability. Edited what-if rows: an approximate contribution computed in the browser by setting each edited feature back to the recorded value, one at a time. The UI labels the two differently.</p></div></div>`
}

export function renderLimits(m: Meta): string {
  return `<h1>Limitations</h1><p class="lead">HeartLens is a research prototype for decision support. It is not a medical device and must not be used to diagnose or treat anyone.</p>
  <div class="card"><ul class="limits">
    <li><b>Small, single centre cohort.</b> ${m.dataset.patients} patients from one hospital in Tehran, Iran (Z-Alizadeh Sani dataset). Results may not transfer to other populations, devices or referral patterns.</li>
    <li><b>No external validation.</b> All metrics are internal, from repeated cross-validation on the same 303 patients. Intervals show fold to fold spread, not population uncertainty.</li>
    <li><b>Referral population.</b> Everyone in the dataset was sent for angiography, so CAD prevalence is ${(m.targets[0].prevalence * 100).toFixed(0)}%. Probabilities are calibrated to that population, not to the general public.</li>
    <li><b>Vessel labels, not lesion locations.</b> The model predicts whether each artery is stenotic. The whole artery is coloured; the colour does not show where along the vessel a narrowing is.</li>
    <li><b>Correlation, not causation.</b> What-if sliders show how the model responds, not what would happen to the patient if a value changed. Features like BMI and weight are linked in reality but edited independently here.</li>
    <li><b>Dataset patients are in the training set.</b> Their live probabilities are in-sample; the cross-validated estimate shown under the CAD score is the honest out-of-sample number.</li>
    <li><b>Anatomy is generic.</b> The heart is one reference model (BodyParts3D), not the patient's own anatomy.</li>
  </ul></div>`
}
