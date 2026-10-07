"""Build docs/figures/*.png, web/public/figures/*.png and docs/HeartLens_Report.{md,html}
from the artifacts written by `make train`. PDF rendering: docs/render_pdf.mjs (Playwright)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.data import FEATURE_META, TARGETS, encode, load_raw  # noqa: E402

ART = ROOT / "artifacts"
DOCS = ROOT / "docs"
FIG = DOCS / "figures"
WEBFIG = ROOT / "web" / "public" / "figures"
NAME = {"Cath": "CAD (any vessel)", "LAD": "LAD stenosis", "LCX": "LCX stenosis", "RCA": "RCA stenosis"}
MODELS = [("logreg", "Logistic regression"), ("rf", "Random forest"), ("xgb", "XGBoost")]
COL = {"logreg": "#4c78a8", "rf": "#59a14f", "xgb": "#e15759"}


def f2(s):
    return f"{s['mean']:.2f} ({s['lo']:.2f} to {s['hi']:.2f})"


def fig_cv(meta):
    fig, ax = plt.subplots(figsize=(7.2, 2.9), dpi=200)
    for j, t in enumerate(TARGETS):
        for k, (m, lab) in enumerate(MODELS):
            s = meta["cv"][t][m]["platt"]["roc_auc"]
            x = j + (k - 1) * 0.22
            sel = meta["selection"][t]["kind"] == m
            ax.errorbar(x, s["mean"], yerr=[[s["mean"] - s["lo"]], [s["hi"] - s["mean"]]], fmt="o", color=COL[m],
                        ms=7 if sel else 4.5, mfc=COL[m] if sel else "white", capsize=2.5, lw=1.2,
                        label=lab if j == 0 else None)
    ax.axhline(0.5, color="#999", lw=0.8, ls=":")
    ax.set_xticks(range(4), [NAME[t] for t in TARGETS], fontsize=8.5)
    ax.set_ylabel("ROC AUC", fontsize=9)
    ax.set_ylim(0.45, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="lower left")
    ax.set_title("Repeated 5x5 CV, mean and 95% fold interval (filled marker = selected)", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "cv_auc.png")
    plt.close(fig)


def fig_calibration(meta):
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.1), dpi=200, sharey=True)
    for ax, t in zip(axs, TARGETS):
        c = meta["calibration_curves"][t]
        ax.plot([0, 1], [0, 1], ":", color="#999", lw=0.8)
        r = c["raw"]
        ax.plot([p["pred"] for p in r], [p["obs"] for p in r], "--", color="#888", lw=1, label="uncalibrated")
        k = c["calibrated"]
        ax.plot([p["pred"] for p in k], [p["obs"] for p in k], "o-", color="#e15759", lw=1.4, ms=3, label="calibrated")
        ax.set_title(NAME[t], fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("predicted", fontsize=7.5)
    axs[0].set_ylabel("observed", fontsize=7.5)
    axs[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png")
    plt.close(fig)


def fig_confusion(meta):
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 1.9), dpi=200)
    for ax, t in zip(axs, TARGETS):
        cm = np.array(meta["confusion"][t])
        ax.imshow(cm, cmap="Blues")
        for a in range(2):
            for b in range(2):
                ax.text(b, a, f"{cm[a, b]:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if cm[a, b] > cm.max() * 0.6 else "black")
        pos = "CAD" if t == "Cath" else "sten."
        ax.set_xticks([0, 1], ["normal", pos], fontsize=7)
        ax.set_yticks([0, 1], ["normal", pos], fontsize=7)
        ax.set_xlabel("predicted", fontsize=7)
        ax.set_title(NAME[t], fontsize=8)
    axs[0].set_ylabel("true", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "confusion.png")
    plt.close(fig)


def fig_beeswarm(meta, dark=False):
    import shap
    enc = encode(load_raw())
    feats = [f["key"] for f in meta["features"]]
    X = enc[feats].to_numpy(float)
    out = []
    for t in TARGETS:
        vals = np.load(ART / f"shap_{t}.npy")
        ex = shap.Explanation(values=vals, data=X, feature_names=[FEATURE_META[k][0] for k in feats])
        style = "dark_background" if dark else "default"
        with plt.style.context(style):
            plt.figure(dpi=170)
            shap.plots.beeswarm(ex, max_display=10, show=False, plot_size=(5.2, 3.4), color_bar_label="feature value")
            ax = plt.gca()
            ax.set_title(f"{NAME[t]}: SHAP, probability points", fontsize=10)
            ax.set_xlabel("SHAP value (change in calibrated probability)", fontsize=8.5)
            ax.tick_params(labelsize=8)
            if dark:
                plt.gcf().patch.set_alpha(0)
                ax.set_facecolor("none")
                for lbl in ax.get_yticklabels() + ax.get_xticklabels():
                    lbl.set_color("#c8cfda")
                for a in plt.gcf().axes:
                    a.tick_params(colors="#c8cfda")
                    a.yaxis.label.set_color("#c8cfda")
                    for tx in a.texts:
                        tx.set_color("#c8cfda")
            p = (WEBFIG if dark else FIG) / f"shap_{t.lower()}.png"
            plt.tight_layout()
            plt.savefig(p, transparent=dark)
            plt.close("all")
        out.append(p)
    return out


def write_report(meta):
    sel = meta["selection"]
    tb = ["| Target | Model, calibration | ROC AUC | PR AUC | Accuracy | Precision | Recall | F1 | Brier |",
          "|---|---|---|---|---|---|---|---|---|"]
    for t in TARGETS:
        s = meta["cv"][t][sel[t]["kind"]][sel[t]["cal"]]
        lab = dict(MODELS)[sel[t]["kind"]] + (", Platt" if sel[t]["cal"] == "platt" else ", isotonic")
        tb.append(f"| {NAME[t]} | {lab} | " + " | ".join(f2(s[m]) for m in
                  ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "brier")) + " |")
    cmp = ["| Target | Positive rate | Logistic regression | Random forest | XGBoost |", "|---|---|---|---|---|"]
    for t in TARGETS:
        prev = next(x for x in meta["targets"] if x["key"] == t)["prevalence"]
        cells = []
        for m, _ in MODELS:
            v = f2(meta["cv"][t][m]["platt"]["roc_auc"])
            cells.append(f"**{v}**" if sel[t]["kind"] == m else v)
        cmp.append(f"| {NAME[t]} | {prev * 100:.0f}% | " + " | ".join(cells) + " |")
    cal = ["| Target | Brier, uncalibrated | Brier, Platt | Brier, isotonic |", "|---|---|---|---|"]
    for t in TARGETS:
        c = meta["cv"][t][sel[t]["kind"]]
        cal.append(f"| {NAME[t]} | {c['raw']['brier']['mean']:.3f} | {c['platt']['brier']['mean']:.3f} | {c['iso']['brier']['mean']:.3f} |")
    top = []
    for t in meta["targets"]:
        names = ", ".join(FEATURE_META[g["key"]][0].lower() for g in t["global_importance"][:5])
        top.append(f"- **{NAME[t['key']]}**: {names}.")
    err = max(t["onnx_max_abs_err"] for t in meta["targets"])
    md = f"""# HeartLens: coronary risk on a 3D heart

**Konstantin Saifoulline.** Multimodal AI Hackathon 2026, Track A (Cardiovascular Risk Visualization and Prediction). Code: [REPO]. MIT licence.

> Decision support research prototype. Not a medical device. All results are internal cross-validation on 303 patients from one centre.

## 1. Problem

A cardiovascular risk score is a single number. Clinicians and patients have to translate it into anatomy themselves. HeartLens predicts coronary artery disease (CAD) and stenosis of each of the three main coronary arteries (left anterior descending, LAD; left circumflex, LCX; right coronary, RCA) from routine clinical data, and paints each prediction onto the matching artery of an interactive 3D heart, with a per patient explanation of what drove it. The whole model runs inside the browser, so the app is a static page with no server and no GPU.

## 2. Data

UCI Machine Learning Repository, *extension of Z-Alizadeh Sani dataset* (id 411, DOI 10.24432/C5461K, CC BY 4.0). {meta['dataset']['patients']} patients who underwent angiography at one centre in Tehran, Iran. 55 features after encoding: demographics, history, symptoms, examination, ECG, laboratory and echocardiography. Targets: Cath (CAD vs normal, {next(x for x in meta['targets'] if x['key']=='Cath')['prevalence']*100:.0f}% positive) and LAD, LCX, RCA stenotic vs normal ({next(x for x in meta['targets'] if x['key']=='LAD')['prevalence']*100:.0f}%, {next(x for x in meta['targets'] if x['key']=='LCX')['prevalence']*100:.0f}%, {next(x for x in meta['targets'] if x['key']=='RCA')['prevalence']*100:.0f}%). No missing values. Encoding: yes/no to 1/0, sex to male = 1, bundle branch block to two indicators, valvular disease as ordinal 0 to 3; *Exertional CP* is constant and dropped.

**Leakage rule.** The four targets are strongly linked (Cath is positive when any vessel is stenotic). When one target is predicted, all four target columns are removed from the features. `tests/test_leakage.py` asserts this for every target, checks that no feature column equals any label, and checks the exported model metadata.

## 3. Method

- **Outer loop**: repeated stratified 5 fold cross-validation, 5 repeats (25 test folds), seed 42.
- **Inner loop** (inside each outer training fold only): 3 fold stratified grid search on ROC AUC. Logistic regression (standardised, L2, C in 0.03, 0.1, 0.3, 1), random forest (300 trees, depth none or 6, min leaf 1 or 3), XGBoost (200 trees, learning rate 0.05, depth 2 or 3, min child weight 1 or 3).
- **Calibration**: inside each outer training fold, 5 fold out-of-fold probabilities of the tuned model are used to fit both Platt scaling and isotonic regression; both are evaluated on the outer test fold.
- **Selection**: per target, the model family with the highest mean CV ROC AUC, then the calibration with the lower mean Brier score.
- **Metrics**: ROC AUC, PR AUC, accuracy, precision, recall, F1 (threshold 0.5) and Brier score per test fold. We report the mean over 25 folds and the 2.5 to 97.5 percentile interval of the fold estimates.
- **Final models** are refit on all 303 patients with the same tuning and calibration recipe and exported to ONNX (skl2onnx, onnxmltools). ONNX and Python outputs agree to {err:.1e} on all patients; a Node test runs the same files in onnxruntime-web and checks the calibrated outputs.
- **Explanations**: permutation SHAP on the full calibrated pipeline (100 patient background), so values are in percentage points of the displayed probability and add up exactly to it. Precomputed for all 303 patients and 4 targets.
- `make train` takes {meta['train_seconds']/60:.1f} minutes on 4 CPU cores.

## 4. Results

**Table 1.** Selected pipeline per target, repeated 5x5 CV, mean (95% fold interval).

{chr(10).join(tb)}

**Table 2.** Model comparison, CV ROC AUC (selected in bold).

{chr(10).join(cmp)}

![CV ROC AUC](figures/cv_auc.png)

**Table 3.** Calibration, mean Brier score of the selected model family.

{chr(10).join(cal)}

![Calibration](figures/calibration.png)

![Confusion matrices](figures/confusion.png)

*Confusion matrices: mean counts per repeat at threshold 0.5, pooled out-of-fold predictions.*

CAD is predicted well (ROC AUC around 0.93). LAD is moderately predictable. LCX and RCA are the hardest targets: their AUCs around 0.72 to 0.74 are clearly better than chance but recall at 0.5 is low, so a negative vessel prediction should not be read as reassurance. We show this rather than tune thresholds to make the table look better.

## 5. Interpretability and the app

Most important features by mean absolute SHAP value:

{chr(10).join(top)}

![SHAP LAD](figures/shap_lad.png)

Typical chest pain dominates CAD and LAD, which matches clinical intuition; regional wall motion abnormality and ejection fraction (echo findings of prior ischemia) rank high for LAD.

![HeartLens app](figures/app.jpg)

The app (Vite, TypeScript, Three.js, onnxruntime-web) loads a BodyParts3D heart (heart wall, aorta, veins and the LAD, LCX and RCA trees, CC BY-SA 2.1 JP). Each artery is coloured on a green to red ramp by its calibrated probability with a soft glow, and the overall CAD probability subtly tints the muscle. Clicking an artery opens its card: probability, model and CV AUC, cohort percentile among all 303 patients and the SHAP waterfall for the patient. What-if sliders and toggles (age, blood pressure, fasting blood sugar, LDL, ejection fraction, BMI, typical chest pain, diabetes, T inversion) and an editable form for all 55 features rerun all four models on every change, well under 200 ms (median under 1 ms for a full batch in our test). For edited rows the app shows an *approximate contribution*: each edited feature is set back to the recorded value one at a time and the change in probability is shown. The UI labels this as approximate and explains it in a tooltip; it is not a SHAP value.

## 6. Limitations

- 303 patients, one centre in Iran, a referral population (71% CAD). No external validation; intervals reflect fold to fold spread only.
- Vessel labels say whether an artery is stenotic, not where. The whole vessel is coloured.
- What-if results show model behaviour, not causal effects. Linked features (weight, BMI) are edited independently.
- Dataset patients are in the final training set, so their live probabilities are in-sample; the app also shows the out-of-fold estimate.
- The heart is a generic reference anatomy, not the patient's.
- Not a medical device; not for diagnosis or treatment.

**Attribution.** Z-Alizadeh Sani extension dataset: R. Alizadehsani, M. Roshanzamir, Z. Sani, UCI Machine Learning Repository, CC BY 4.0. BodyParts3D, Copyright 2008 Life Science Integrated Database Center, CC BY-SA 2.1 Japan (via the K. M. Moerman GitHub mirror).
"""
    (DOCS / "HeartLens_Report.md").write_text(md)
    import markdown
    body = markdown.markdown(md, extensions=["tables"])
    css = """@page { size: A4; margin: 13mm 14mm 13mm 14mm }
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 9.6pt; line-height: 1.42; color: #1b1f27 }
h1 { font-size: 19pt; margin: 0 0 4pt; color: #b3262d } h2 { font-size: 12.5pt; margin: 12pt 0 4pt; border-bottom: 1px solid #ddd; padding-bottom: 2pt }
blockquote { margin: 6pt 0; padding: 5pt 9pt; background: #fff6e5; border-left: 3px solid #e3a21a; color: #6b4b00 }
table { border-collapse: collapse; width: 100%; font-size: 8pt; margin: 4pt 0 6pt } th, td { border-bottom: 1px solid #e3e3e3; padding: 3pt 4pt; text-align: left }
th { background: #f4f5f7 } img { display: block; max-width: 100%; margin: 4pt auto } img[alt="SHAP LAD"] { max-width: 66% } img[alt="HeartLens app"] { max-width: 92%; border: 1px solid #ddd }
ul { padding-left: 15pt; margin: 3pt 0 } li { margin: 1.5pt 0 } p { margin: 4pt 0 } code { font-size: 8.5pt; background: #f2f2f2; padding: 0 2pt }"""
    (DOCS / "HeartLens_Report.html").write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>HeartLens report</title><style>{css}</style></head><body>{body}</body></html>")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ART / "meta.json").read_text())
    fig_cv(meta); fig_calibration(meta); fig_confusion(meta)
    fig_beeswarm(meta, dark=False)
    if "--web" in sys.argv:
        WEBFIG.mkdir(parents=True, exist_ok=True)
        fig_beeswarm(meta, dark=True)
    write_report(meta)
    print("report written:", DOCS / "HeartLens_Report.md")


if __name__ == "__main__":
    main()
