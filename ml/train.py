"""HeartLens training: repeated stratified 5x5 CV with light nested tuning,
model comparison (logistic regression, random forest, XGBoost) per target,
Platt vs isotonic calibration, final refit, ONNX export, SHAP export.

Run: make train   (writes artifacts/, web/public/models, web/public/data)
"""
from __future__ import annotations

import json
import os
import time
import warnings
from itertools import product
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ml.data import FEATURE_META, TARGETS, encode, load_raw, xy

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
WEB = ROOT / "web" / "public"
SEED = 42
N_JOBS = 4
EPS = 1e-6

GRIDS = {
    "logreg": [{"C": c} for c in (0.03, 0.1, 0.3, 1.0)],
    "rf": [{"max_depth": d, "min_samples_leaf": l} for d, l in product((None, 6), (1, 3))],
    "xgb": [{"max_depth": d, "min_child_weight": w} for d, w in product((2, 3), (1, 3))],
}
MODEL_LABEL = {"logreg": "Logistic regression", "rf": "Random forest", "xgb": "XGBoost"}


def make_model(kind: str, params: dict):
    if kind == "logreg":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(C=params["C"], max_iter=5000))])
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=1,
                                      random_state=SEED, **params)
    if kind == "xgb":
        return XGBClassifier(n_estimators=200, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, tree_method="hist", n_jobs=1,
                             random_state=SEED, eval_metric="logloss", **params)
    raise ValueError(kind)


def tune(kind, X, y, seed):
    """Inner loop: 3 fold stratified CV, ROC AUC, small grid."""
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    best, best_auc = None, -1
    for params in GRIDS[kind]:
        aucs = []
        for tr, va in cv.split(X, y):
            m = make_model(kind, params).fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
        if np.mean(aucs) > best_auc:
            best, best_auc = params, float(np.mean(aucs))
    return best, best_auc


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_calibrators(p_oof, y):
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_oof).reshape(-1, 1), y)
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p_oof, y)
    return platt, iso


def apply_platt(platt, p):
    return platt.predict_proba(logit(p).reshape(-1, 1))[:, 1]


def fold_task(target, kind, X, y, rep, fold, tr, te):
    params, inner_auc = tune(kind, X[tr], y[tr], SEED + rep * 10 + fold)
    p_oof = cross_val_predict(make_model(kind, params), X[tr], y[tr],
                              cv=StratifiedKFold(5, shuffle=True, random_state=SEED + fold),
                              method="predict_proba")[:, 1]
    platt, iso = fit_calibrators(p_oof, y[tr])
    model = make_model(kind, params).fit(X[tr], y[tr])
    raw = model.predict_proba(X[te])[:, 1]
    return dict(target=target, kind=kind, rep=rep, fold=fold, params=params, inner_auc=inner_auc,
                te=te, raw=raw, platt=apply_platt(platt, raw), iso=iso.predict(raw))


def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return dict(
        roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
        accuracy=accuracy_score(y, yhat), precision=precision_score(y, yhat, zero_division=0),
        recall=recall_score(y, yhat, zero_division=0), f1=f1_score(y, yhat, zero_division=0),
        brier=brier_score_loss(y, p))


def summarise(vals):
    v = np.asarray(vals)
    return {"mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)),
            "hi": float(np.percentile(v, 97.5)), "sd": float(v.std(ddof=1))}


def export_onnx(kind, model, n_features, path):
    from onnxmltools.convert.common.data_types import FloatTensorType
    if kind == "xgb":
        import onnxmltools
        onx = onnxmltools.convert_xgboost(model, initial_types=[("input", FloatTensorType([None, n_features]))],
                                          target_opset=15)
    else:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType as FT
        clf = model.named_steps["clf"] if kind == "logreg" else model
        onx = convert_sklearn(model, initial_types=[("input", FT([None, n_features]))],
                              options={id(clf): {"zipmap": False}}, target_opset=15)
    path.write_bytes(onx.SerializeToString())
    return onx


def onnx_probs(path, X):
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outs = s.run(None, {"input": X.astype(np.float32)})
    for o in outs:
        o = np.asarray(o)
        if o.ndim == 2 and o.shape[1] == 2:
            return o[:, 1], [x.name for x in s.get_outputs()]
    raise RuntimeError("no probability output")


def main():
    t0 = time.time()
    ART.mkdir(exist_ok=True)
    (WEB / "models").mkdir(parents=True, exist_ok=True)
    (WEB / "data").mkdir(parents=True, exist_ok=True)
    enc = encode(load_raw())
    feats = None
    data = {}
    for t in TARGETS:
        X, y = xy(t, enc)
        feats = list(X.columns)
        data[t] = (X.to_numpy(np.float32), y)
    print(f"{len(enc)} patients, {len(feats)} features, prevalence:",
          {t: round(float(data[t][1].mean()), 3) for t in TARGETS})

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    tasks = []
    for t in TARGETS:
        X, y = data[t]
        for i, (tr, te) in enumerate(rskf.split(X, y)):
            for kind in GRIDS:
                tasks.append(delayed(fold_task)(t, kind, X, y, i // 5, i % 5, tr, te))
    print(f"outer CV tasks: {len(tasks)} on {N_JOBS} workers")
    results = Parallel(n_jobs=N_JOBS, verbose=0)(tasks)
    print(f"CV done in {time.time() - t0:.0f}s")

    # fold level metrics
    rows = []
    for r in results:
        y = data[r["target"]][1][r["te"]]
        for cal in ("raw", "platt", "iso"):
            rows.append(dict(target=r["target"], kind=r["kind"], cal=cal, rep=r["rep"], fold=r["fold"],
                             **metrics(y, r[cal])))
    fm = pd.DataFrame(rows)
    fm.to_csv(ART / "cv_fold_metrics.csv", index=False)

    METRICS = ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "brier"]
    table = {}
    selection = {}
    for t in TARGETS:
        table[t] = {}
        for kind in GRIDS:
            table[t][kind] = {}
            for cal in ("raw", "platt", "iso"):
                sub = fm[(fm.target == t) & (fm.kind == kind) & (fm.cal == cal)]
                table[t][kind][cal] = {m: summarise(sub[m]) for m in METRICS}
        best_kind = max(GRIDS, key=lambda k: table[t][k]["platt"]["roc_auc"]["mean"])
        best_cal = min(("platt", "iso"), key=lambda c: table[t][best_kind][c]["brier"]["mean"])
        selection[t] = {"kind": best_kind, "cal": best_cal}
        print(f"{t}: " + ", ".join(f"{k} AUC {table[t][k]['platt']['roc_auc']['mean']:.3f}" for k in GRIDS)
              + f"  -> {best_kind} + {best_cal}")

    # out of fold predictions of the selected pipeline, averaged over the 5 repeats
    oof = {}
    pooled = {}
    for t in TARGETS:
        n = len(data[t][1])
        acc = np.zeros((5, n))
        for r in results:
            if r["target"] == t and r["kind"] == selection[t]["kind"]:
                acc[r["rep"], r["te"]] = r[selection[t]["cal"]]
                pooled.setdefault(t, {}).setdefault("raw", np.zeros((5, n)))[r["rep"], r["te"]] = r["raw"]
        oof[t] = acc
        pooled[t]["cal"] = acc
        np.save(ART / f"oof_{t}.npy", acc)
        np.save(ART / f"oof_raw_{t}.npy", pooled[t]["raw"])

    # final models on all 303 patients
    final = {}
    for t in TARGETS:
        X, y = data[t]
        kind, cal = selection[t]["kind"], selection[t]["cal"]
        params, inner = tune(kind, X, y, SEED)
        p_oof = cross_val_predict(make_model(kind, params), X, y,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                  method="predict_proba")[:, 1]
        platt, iso = fit_calibrators(p_oof, y)
        model = make_model(kind, params).fit(X, y)
        path = WEB / "models" / f"{t.lower()}.onnx"
        export_onnx(kind, model, X.shape[1], path)
        p_sk = model.predict_proba(X)[:, 1]
        p_ox, outs = onnx_probs(path, X)
        err = float(np.abs(p_sk - p_ox).max())
        assert err < 1e-4, (t, err)
        if cal == "platt":
            calib = {"type": "platt", "a": float(platt.coef_[0, 0]), "b": float(platt.intercept_[0])}
            f_cal = lambda p, platt=platt: apply_platt(platt, p)
        else:
            calib = {"type": "isotonic", "x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
            f_cal = lambda p, iso=iso: iso.predict(p)
        final[t] = dict(kind=kind, params=params, model=model, f_cal=f_cal, calib=calib,
                        onnx_outputs=outs, onnx_max_abs_err=err, path=path)
        print(f"final {t}: {kind} {params} {cal}, onnx max err {err:.2e}, {path.stat().st_size // 1024} KB")

    # SHAP: permutation explainer on the full calibrated pipeline, probability space
    import shap
    shap_out = {}
    for t in TARGETS:
        X, y = data[t]
        f = final[t]
        fn = lambda A, f=f: f["f_cal"](f["model"].predict_proba(np.asarray(A, np.float32))[:, 1])
        rng = np.random.default_rng(SEED)
        bg = X[rng.choice(len(X), 100, replace=False)]
        expl = shap.PermutationExplainer(fn, shap.maskers.Independent(bg, max_samples=100), seed=SEED)
        ts = time.time()
        sv = expl(X, max_evals=2 * X.shape[1] + 1, silent=True)
        vals = np.asarray(sv.values)
        base = float(np.asarray(sv.base_values).mean())
        shap_out[t] = dict(values=vals, base=base)
        np.save(ART / f"shap_{t}.npy", vals)
        recon = base + vals.sum(1)
        print(f"shap {t}: {time.time() - ts:.0f}s, additivity max err {np.abs(recon - fn(X)).max():.2e}")

    # ---- export JSON for the browser ----
    X_all = data["Cath"][0]
    # exact source values (float64 shortest repr) so float32 casts match training bit for bit
    X_src = enc[feats].to_numpy(np.float64)
    feat_meta = []
    for j, k in enumerate(feats):
        label, kindf, unit = FEATURE_META[k]
        col = X_all[:, j]
        isint = bool(np.all(np.mod(col, 1) == 0))
        feat_meta.append(dict(key=k, label=label, kind=kindf, unit=unit, min=float(col.min()),
                              max=float(col.max()), median=float(np.median(col)),
                              step=1 if isint else 0.1))
    targets_meta = []
    for t in TARGETS:
        f = final[t]
        imp = np.abs(shap_out[t]["values"]).mean(0)
        order = np.argsort(-imp)
        targets_meta.append(dict(
            key=t, label={"Cath": "Coronary artery disease", "LAD": "Left anterior descending",
                          "LCX": "Left circumflex", "RCA": "Right coronary"}[t],
            model=f["kind"], model_label=MODEL_LABEL[f["kind"]], params=f["params"], calibration=f["calib"],
            onnx=f"models/{t.lower()}.onnx", onnx_outputs=f["onnx_outputs"], onnx_max_abs_err=f["onnx_max_abs_err"],
            shap_base=shap_out[t]["base"], prevalence=float(data[t][1].mean()),
            global_importance=[{"key": feats[i], "value": float(imp[i])} for i in order[:15]]))

    # calibration curve data (pooled OOF of the selected pipeline, all repeats)
    calib_curves = {}
    for t in TARGETS:
        y = data[t][1]
        for name, P in (("raw", pooled[t]["raw"]), ("calibrated", pooled[t]["cal"])):
            p = P.ravel(); yy = np.tile(y, 5)
            bins = np.linspace(0, 1, 11)
            idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
            pts = [{"pred": float(p[idx == b].mean()), "obs": float(yy[idx == b].mean()), "n": int((idx == b).sum())}
                   for b in range(10) if (idx == b).sum() >= 5]
            calib_curves.setdefault(t, {})[name] = pts
    conf = {}
    for t in TARGETS:
        y = data[t][1]; P = pooled[t]["cal"]
        cm = np.zeros((2, 2))
        for r in range(5):
            yhat = (P[r] >= 0.5).astype(int)
            for a in (0, 1):
                for b in (0, 1):
                    cm[a, b] += np.sum((y == a) & (yhat == b))
        conf[t] = (cm / 5).round(1).tolist()

    meta = dict(
        dataset=dict(name="Extension of Z-Alizadeh Sani dataset", source="UCI Machine Learning Repository, id 411",
                     doi="10.24432/C5461K", license="CC BY 4.0", patients=int(len(enc)), features=len(feats)),
        protocol=dict(outer="RepeatedStratifiedKFold 5 folds x 5 repeats", inner="StratifiedKFold 3, ROC AUC, grid",
                      calibration="Platt or isotonic fitted on inner 5 fold out of fold predictions, chosen by Brier",
                      threshold=0.5, interval="2.5 to 97.5 percentile of the 25 fold level estimates",
                      shap="Permutation SHAP on the calibrated pipeline, 100 row background, probability units"),
        features=feat_meta, targets=targets_meta, cv=table, selection=selection,
        calibration_curves=calib_curves, confusion=conf, grids=GRIDS, train_seconds=None)

    patients = []
    for i in range(len(enc)):
        patients.append(dict(
            id=i + 1, x=[float(v) for v in X_src[i]],
            truth={t: int(data[t][1][i]) for t in TARGETS},
            pred={t: round(float(final[t]["f_cal"](final[t]["model"].predict_proba(X_all[i:i + 1])[:, 1])[0]), 4)
                  for t in TARGETS},
            oof={t: round(float(oof[t][:, i].mean()), 4) for t in TARGETS},
            shap={t: [round(float(v), 4) for v in shap_out[t]["values"][i]] for t in TARGETS}))
    meta["train_seconds"] = round(time.time() - t0, 1)
    (WEB / "data" / "meta.json").write_text(json.dumps(meta, indent=1))
    (WEB / "data" / "patients.json").write_text(json.dumps(patients, separators=(",", ":")))
    (ART / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
