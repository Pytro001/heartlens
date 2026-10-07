import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from ml.data import FEATURE_META, TARGETS, encode, feature_columns, load_raw, xy


@pytest.fixture(scope="module")
def enc():
    return encode(load_raw())


def test_dataset_shape(enc):
    raw = load_raw()
    assert raw.shape == (303, 59)
    for t in TARGETS:
        assert t in raw.columns


@pytest.mark.parametrize("target", TARGETS)
def test_no_target_column_in_features(enc, target):
    cols = feature_columns(enc, target)
    for t in TARGETS:
        assert t not in cols, f"{t} leaks into features for {target}"
    X, y = xy(target, enc)
    assert not set(TARGETS) & set(X.columns)
    assert len(X) == len(y) == 303


@pytest.mark.parametrize("target", TARGETS)
def test_no_feature_equals_any_target(enc, target):
    """Guard against a renamed copy of a label sneaking in."""
    X, _ = xy(target, enc)
    for t in TARGETS:
        lab = enc[t].to_numpy()
        for c in X.columns:
            assert not np.array_equal(X[c].to_numpy().astype(int), lab), f"{c} duplicates {t}"


def test_feature_metadata_complete(enc):
    X, _ = xy("Cath", enc)
    assert list(X.columns) == [c for c in FEATURE_META if c in X.columns]
    assert set(X.columns) == set(FEATURE_META)


def test_exported_models_use_leakage_safe_features(enc):
    import json
    meta = Path(__file__).resolve().parents[1] / "web" / "public" / "data" / "meta.json"
    if not meta.exists():
        pytest.skip("run make train first")
    m = json.loads(meta.read_text())
    names = [f["key"] for f in m["features"]]
    assert not set(TARGETS) & set(names)
