"""Load and encode the UCI extension of the Z-Alizadeh Sani dataset.

Source: UCI Machine Learning Repository, "extention of Z-Alizadeh sani dataset"
(id 411, DOI 10.24432/C5461K), licence CC BY 4.0. The file in data/raw is kept as
downloaded. 303 patients, 55 features, targets LAD, LCX, RCA (stenotic or normal)
and Cath (CAD or normal).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ext" / "extention of Z-Alizadeh sani dataset.xlsx"

TARGETS = ["Cath", "LAD", "LCX", "RCA"]
TARGET_POSITIVE = {"Cath": "CAD", "LAD": "Stenotic", "LCX": "Stenotic", "RCA": "Stenotic"}

YN_COLS = [
    "Obesity", "CRF", "CVA", "Airway disease", "Thyroid Disease", "CHF", "DLP",
    "Weak Peripheral Pulse", "Lung rales", "Systolic Murmur", "Diastolic Murmur",
    "Dyspnea", "Atypical", "Nonanginal", "LowTH Ang", "LVH", "Poor R Progression",
]
VHD_ORDER = {"N": 0, "mild": 1, "Moderate": 2, "Severe": 3}
# constant in this dataset (all "N"), carries no information
DROP_CONSTANT = ["Exertional CP"]


def load_raw() -> pd.DataFrame:
    df = pd.read_excel(RAW, sheet_name=0)
    assert df.shape == (303, 59), df.shape
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric encoding shared by Python and the browser (see web/src/features.ts)."""
    out = df.copy()
    out["Sex"] = (out["Sex"].astype(str) == "Male").astype(int)
    for c in YN_COLS:
        out[c] = (out[c].astype(str) == "Y").astype(int)
    out["LBBB"] = (out["BBB"].astype(str) == "LBBB").astype(int)
    out["RBBB"] = (out["BBB"].astype(str) == "RBBB").astype(int)
    out = out.drop(columns=["BBB"] + DROP_CONSTANT)
    out["VHD"] = out["VHD"].astype(str).map(VHD_ORDER).astype(int)
    for t in TARGETS:
        out[t] = (out[t].astype(str) == TARGET_POSITIVE[t]).astype(int)
    return out


def feature_columns(encoded: pd.DataFrame, target: str) -> list[str]:
    """Leakage rule: when predicting one target, every target column is removed."""
    assert target in TARGETS
    return [c for c in encoded.columns if c not in TARGETS]


def xy(target: str, encoded: pd.DataFrame | None = None):
    enc = encoded if encoded is not None else encode(load_raw())
    cols = feature_columns(enc, target)
    X = enc[cols].astype(np.float32)
    y = enc[target].astype(int).to_numpy()
    return X, y


# Metadata for the UI: label, kind, unit. Order follows the encoded frame.
FEATURE_META = {
    "Age": ("Age", "num", "yr"), "Weight": ("Weight", "num", "kg"), "Length": ("Height", "num", "cm"),
    "Sex": ("Male sex", "bin", ""), "BMI": ("BMI", "num", "kg/m2"), "DM": ("Diabetes", "bin", ""),
    "HTN": ("Hypertension", "bin", ""), "Current Smoker": ("Current smoker", "bin", ""),
    "EX-Smoker": ("Ex smoker", "bin", ""), "FH": ("Family history", "bin", ""),
    "Obesity": ("Obesity", "bin", ""), "CRF": ("Chronic renal failure", "bin", ""),
    "CVA": ("Prior stroke", "bin", ""), "Airway disease": ("Airway disease", "bin", ""),
    "Thyroid Disease": ("Thyroid disease", "bin", ""), "CHF": ("Heart failure", "bin", ""),
    "DLP": ("Dyslipidaemia", "bin", ""), "BP": ("Blood pressure", "num", "mmHg"),
    "PR": ("Pulse rate", "num", "bpm"), "Edema": ("Edema", "bin", ""),
    "Weak Peripheral Pulse": ("Weak peripheral pulse", "bin", ""), "Lung rales": ("Lung rales", "bin", ""),
    "Systolic Murmur": ("Systolic murmur", "bin", ""), "Diastolic Murmur": ("Diastolic murmur", "bin", ""),
    "Typical Chest Pain": ("Typical chest pain", "bin", ""), "Dyspnea": ("Dyspnea", "bin", ""),
    "Function Class": ("Function class", "ord", "0 to 3"), "Atypical": ("Atypical chest pain", "bin", ""),
    "Nonanginal": ("Nonanginal pain", "bin", ""), "LowTH Ang": ("Low threshold angina", "bin", ""),
    "Q Wave": ("Q wave", "bin", ""), "St Elevation": ("ST elevation", "bin", ""),
    "St Depression": ("ST depression", "bin", ""), "Tinversion": ("T inversion", "bin", ""),
    "LVH": ("LV hypertrophy", "bin", ""), "Poor R Progression": ("Poor R progression", "bin", ""),
    "FBS": ("Fasting blood sugar", "num", "mg/dL"), "CR": ("Creatinine", "num", "mg/dL"),
    "TG": ("Triglycerides", "num", "mg/dL"), "LDL": ("LDL cholesterol", "num", "mg/dL"),
    "HDL": ("HDL cholesterol", "num", "mg/dL"), "BUN": ("BUN", "num", "mg/dL"),
    "ESR": ("ESR", "num", "mm/h"), "HB": ("Haemoglobin", "num", "g/dL"), "K": ("Potassium", "num", "mEq/L"),
    "Na": ("Sodium", "num", "mEq/L"), "WBC": ("White cells", "num", "/uL"), "Lymph": ("Lymphocytes", "num", "%"),
    "Neut": ("Neutrophils", "num", "%"), "PLT": ("Platelets", "num", "1000/uL"),
    "EF-TTE": ("Ejection fraction", "num", "%"), "Region RWMA": ("Regional wall motion abn.", "ord", "0 to 4"),
    "VHD": ("Valvular disease", "ord", "0 none to 3 severe"), "LBBB": ("Left bundle branch block", "bin", ""),
    "RBBB": ("Right bundle branch block", "bin", ""),
}
