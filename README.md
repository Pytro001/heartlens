# HeartLens

**See coronary risk where it lives.** HeartLens predicts coronary artery disease and stenosis of each main coronary artery (LAD, LCX, RCA) from routine clinical data, then paints every prediction onto the matching artery of a real 3D heart (BodyParts3D anatomy) that you can orbit and click. Each artery card explains its prediction with SHAP values, compares the patient with the 303 patient cohort, and what-if sliders rerun four ONNX models inside the browser on every change. The models are validated with repeated 5x5 cross-validation, nested tuning, calibration and a unit tested leakage rule. Everything runs as a static page: no server, no GPU, no data leaves the browser.

> **Decision support research prototype. Not a medical device.** 303 patients from one centre, no external validation. Do not use for diagnosis or treatment.

![HeartLens](docs/figures/app.jpg)

Entry for the Multimodal AI Hackathon 2026, Track A (Cardiovascular Risk Visualization and Prediction), by Konstantin Saifoulline. Report (4 pages): [docs/HeartLens_Report.pdf](docs/HeartLens_Report.pdf).

## Results (repeated stratified 5x5 CV, mean and 95% fold interval)

| Target | Selected model | ROC AUC | PR AUC | Accuracy | Precision | Recall | F1 | Brier |
|---|---|---|---|---|---|---|---|---|
| CAD (any vessel) | Logistic regression, Platt | 0.93 (0.88 to 0.99) | 0.97 (0.94 to 0.99) | 0.87 (0.79 to 0.96) | 0.91 (0.83 to 0.98) | 0.91 (0.85 to 1.00) | 0.91 (0.86 to 0.97) | 0.09 (0.05 to 0.14) |
| LAD stenosis | Random forest, Platt | 0.85 (0.78 to 0.93) | 0.89 (0.83 to 0.96) | 0.79 (0.72 to 0.86) | 0.80 (0.73 to 0.88) | 0.87 (0.76 to 0.95) | 0.83 (0.77 to 0.88) | 0.15 (0.11 to 0.19) |
| LCX stenosis | XGBoost, Platt | 0.74 (0.63 to 0.84) | 0.63 (0.49 to 0.77) | 0.69 (0.61 to 0.75) | 0.65 (0.52 to 0.80) | 0.47 (0.34 to 0.62) | 0.54 (0.43 to 0.65) | 0.20 (0.17 to 0.24) |
| RCA stenosis | Logistic regression, Platt | 0.72 (0.60 to 0.79) | 0.62 (0.49 to 0.73) | 0.67 (0.55 to 0.74) | 0.60 (0.36 to 0.80) | 0.39 (0.20 to 0.58) | 0.46 (0.27 to 0.60) | 0.20 (0.18 to 0.24) |

Model comparison, ROC AUC (selected in bold):

| Target | Logistic regression | Random forest | XGBoost |
|---|---|---|---|
| CAD (any vessel) | **0.93 (0.88 to 0.99)** | 0.92 (0.85 to 0.98) | 0.92 (0.83 to 0.98) |
| LAD stenosis | 0.83 (0.74 to 0.94) | **0.85 (0.78 to 0.93)** | 0.83 (0.74 to 0.92) |
| LCX stenosis | 0.67 (0.56 to 0.79) | 0.73 (0.63 to 0.82) | **0.74 (0.63 to 0.84)** |
| RCA stenosis | **0.72 (0.60 to 0.79)** | 0.71 (0.60 to 0.79) | 0.71 (0.57 to 0.80) |

Threshold 0.5. The interval is the 2.5 to 97.5 percentile of the 25 test fold estimates. LCX and RCA are hard to predict from clinical data alone and have low recall at 0.5; the app and report say so.

## Prerequisites

- macOS or Linux, Python 3.12 via [uv](https://docs.astral.sh/uv/), Node 20 or newer with npm.
- macOS only: XGBoost needs an OpenMP runtime. The Makefile points `DYLD_LIBRARY_PATH` at the `libomp.dylib` that ships inside the scikit-learn wheel, so `brew install libomp` is not required. If you run Python by hand, either use `make` or install libomp.

## Setup and run

```bash
make setup        # uv venv (.venv, Python 3.12) + pip install -r requirements.txt + npm install in web/
make test         # leakage tests (pytest) + browser runtime parity and latency test (node)
make train        # 5x5 CV, model selection, calibration, ONNX + SHAP export (about 6 minutes on 4 cores)
make report       # figures + docs/HeartLens_Report.md/.html/.pdf (PDF via Playwright)
make demo         # Vite dev server on http://localhost:5301
make build        # static site in web/dist (about 18 MB, base './', works on GitHub Pages)
```

The trained models, SHAP values, metrics and the heart mesh are committed under `web/public/`, so `cd web && npm install && npm run dev` works without running Python. Useful URLs: `http://localhost:5301/?patient=25` opens a patient directly; `?film=1` is the deterministic film mode used by the recorder.

To deploy on GitHub Pages: `make build`, then publish `web/dist` (for example with the `peaceiris/actions-gh-pages` action or by pushing `dist` to a `gh-pages` branch).

## Architecture

```
 UCI Z-Alizadeh Sani extension (303 x 55, CC BY 4.0)           BodyParts3D heart + coronary STLs (CC BY-SA 2.1 JP)
        |                                                               |
        v                                                               v
 ml/data.py   encode + leakage rule (drop LAD, LCX, RCA, Cath)   ml/build_anatomy.py  decimate, recentre -> heart.glb
        |            tested by tests/test_leakage.py
        v
 ml/train.py  RepeatedStratifiedKFold 5x5
              |- inner 3 fold grid search (LogReg, RF, XGBoost)
              |- Platt vs isotonic on inner out-of-fold predictions
              |- pick model by ROC AUC, calibration by Brier
              |- refit on all 303, export ONNX (skl2onnx, onnxmltools), parity check
              |- permutation SHAP on the calibrated pipeline, 303 patients x 4 targets
        |
        v
 web/public/models/*.onnx   web/public/data/meta.json (features, calibration maps, CV metrics, curves)
 web/public/data/patients.json (features, labels, predictions, out-of-fold estimates, SHAP)
        |
        v
 web/ (Vite + TypeScript, no framework)
   model.ts   onnxruntime-web (WASM, 1 thread): edited row + one row per edited feature, 4 models, then calibration
   heart.ts   Three.js: GLB anatomy, vertex colour risk ramp + flow pulse + fresnel glow per artery, picking, labels
   main.ts    patient picker, what-if sliders and toggles, all 55 feature form, cards, SHAP waterfall, percentiles
   panels.ts  model evidence (tables, calibration, confusion), architecture, limitations
   film.ts    ?film=1 deterministic director (window.__film) for output/record.mjs
```

## How the explanations work

- **Dataset records**: SHAP values computed in Python with a permutation explainer on the *calibrated* pipeline and a 100 patient background. Units are percentage points of the displayed probability; for each patient they add up exactly to the prediction minus the cohort average.
- **Edited (what-if) rows**: the app shows the new probabilities from the ONNX models and an **approximate contribution** for each edited feature: the model is rerun with only that feature set back to the recorded value, and the bar is the difference. This is a single feature perturbation, not a SHAP value, and the bars need not sum to the total change when features interact. The UI labels it "approximate contribution" with a tooltip saying exactly this.
- **Cohort percentile**: share of the 303 patients whose predicted probability (final model) is lower.
- Dataset patients are in the final training set, so their live probabilities are in-sample. The CAD card also shows the cross-validated estimate for that record.

## Tests

- `tests/test_leakage.py` (pytest): dataset shape, no target column in the features of any target, no feature identical to any label, feature metadata complete, exported metadata free of target names.
- `web/scripts/test-inference.mjs` (node): loads the four ONNX files in onnxruntime-web, checks calibrated outputs against the Python export for all 303 patients (max difference 5e-5, which is the 4 decimal rounding of the export) and that a 10 row x 4 model what-if batch takes well under 200 ms (median 0.2 to 0.5 ms here).

## Film

`output/heartlens-screen.mp4` (1920x1080, 30 fps, 4:40, no audio) is rendered frame by frame by `output/record.mjs` through Playwright from `?film=1`. The film is scripted: the cursor, clicks, slider drags and camera moves are driven by `web/src/film.ts`, but every number on screen comes from the real models running in the page. Stills: `output/stills/`. The global SHAP beeswarms at the bottom of the Model evidence page were added after the film was rendered and are not shown in it. The camera script for the voice over is `DEMO_SCRIPT.md`.

## What is new

Everything in this repository was written during the hackathon (October 2026): data pipeline, models, tests, export, web app, 3D scene, film tooling and report. Pre-existing inputs are the public dataset, the BodyParts3D meshes and the open source libraries listed below.

## Limitations

- 303 patients from one centre in Tehran, Iran; a referral population with 71% CAD. No external validation. Intervals describe fold to fold spread, not population uncertainty.
- LCX and RCA predictions are weak (ROC AUC about 0.72 to 0.74, recall about 0.4 to 0.5 at threshold 0.5).
- Vessel labels say whether an artery is stenotic, not where. The whole artery is coloured.
- What-if sliders show model behaviour, not causal effects; linked features (weight and BMI) are edited independently.
- Dataset patients' live probabilities are in-sample (out-of-fold estimate shown alongside).
- Generic reference anatomy, not patient specific.
- Not a medical device, not for clinical use.
- Assumption: the brief said UCI id 412; the per vessel labels live in UCI id 411 ("extention of Z-Alizadeh sani dataset"), which is what is used. See `data/README.md`.

## Licence and attribution

Code: MIT, see `LICENSE`.

- Data: R. Alizadehsani, M. Roshanzamir, Z. Sani, *Extension of Z-Alizadeh Sani dataset*, UCI Machine Learning Repository, DOI 10.24432/C5461K, CC BY 4.0.
- Anatomy: BodyParts3D, Copyright 2008 Life Science Integrated Database Center, licensed under CC BY-SA 2.1 Japan; STL files from the GitHub mirror by Kevin M. Moerman. The decimated `heart.glb` is a derivative shared under the same licence.
- Libraries: scikit-learn, XGBoost, SHAP, pandas, NumPy, skl2onnx, onnxmltools, ONNX Runtime and onnxruntime-web, matplotlib, trimesh, Three.js, Vite, TypeScript, Playwright.
