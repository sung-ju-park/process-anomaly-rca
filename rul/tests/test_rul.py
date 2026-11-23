"""RUL 모듈 불변 조건 테스트.  실행: python -m pytest rul/tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cmapss as C  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return C.load()


def test_rows_and_hash(data):
    tr, te, rul = data
    exp = C.PROTOCOL["data"]["expected_rows"]
    assert (len(tr), len(te), len(rul)) == (exp["train"], exp["test"], exp["rul"])
    assert tr.unit.nunique() == te.unit.nunique() == 100
    for f, h in C.PROTOCOL["data"]["sha256"].items():
        assert C.sha256(C.DATA / f) == h


def test_sensor_rule_matches_protocol(data):
    tr, _, _ = data
    drop = C.low_variance_sensors(tr)
    assert [c for c in C.COLS[5:] if c not in drop] == C.SENSORS


def test_train_label(data):
    tr, _, _ = data
    for _, g in tr.groupby("unit"):
        assert g.RUL.iloc[-1] == 0
        assert (np.diff(g.RUL.to_numpy()) == -1).all()


def test_test_last_rows_align_with_rul_file(data):
    _, te, rul = data
    ev = C.last_rows(te)
    assert list(te.loc[ev, "unit"]) == list(range(1, 101))
    assert te.groupby("unit").cycle.max().min() >= C.MIN_PREFIX


def test_engine_split_no_overlap():
    units = np.arange(1, 101)
    for a, b in GroupKFold(5).split(units, groups=units):
        assert not set(units[a]) & set(units[b])


def test_features_use_only_past(data):
    tr, _, _ = data
    g = tr[tr.unit.isin([1, 2])].copy()
    t = 60
    f1 = C.features(g)
    g2 = g.copy()
    future = (g2.unit == 1) & (g2.cycle > t)
    g2.loc[future, C.SENSORS] = g2.loc[future, C.SENSORS] * 10 + 999
    f2 = C.features(g2)
    past = (g.unit == 1) & (g.cycle <= t)
    pd.testing.assert_frame_equal(f1[past], f2[past])


def test_alarm_flags_use_only_past(data):
    tr, _, _ = data
    det = C.OnsetDetector().fit(tr, range(1, 101))
    g = tr[tr.unit == 3].copy()
    t = 80
    a1 = det.alarm_flags(g)
    g2 = g.copy()
    g2.loc[g2.cycle > t, C.SENSORS] += 1000
    a2 = det.alarm_flags(g2)
    assert (a1[g.cycle <= t] == a2[g.cycle <= t]).all()


def test_rolling_slope_matches_reference(data):
    tr, _, _ = data
    x = tr[tr.unit == 5].s4.astype(float).reset_index(drop=True)
    ref = [C._slope(x.iloc[max(0, i - C.SLOPE_WIN + 1): i + 1].to_numpy()) for i in range(len(x))]
    assert np.allclose(C.rolling_slope(x), ref, atol=1e-8)


def test_truncation_points(data):
    tr, _, _ = data
    idx = C.truncation_points(tr, [1, 2, 3], k=5, seed=0)
    assert len(idx) == 15
    assert (tr.loc[idx, "cycle"] >= C.MIN_PREFIX).all()


def test_nasa_score_hand_cases():
    assert C.nasa_score([0], [10]) == pytest.approx(np.e - 1)   # 늦은 예측 10
    assert C.nasa_score([13], [0]) == pytest.approx(np.e - 1)   # 이른 예측 13
    assert C.nasa_score([5], [5]) == 0
