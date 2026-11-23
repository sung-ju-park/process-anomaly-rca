"""C-MAPSS FD001 잔존수명(RUL) 예측: 데이터, 특징, 라벨, 열화 경보, 지표.

기준은 rul/protocol.json (공식 test 실행 전에 커밋). 여기 숫자는 그 파일과 같아야 한다.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "cmapss"
PROTOCOL = json.loads((ROOT / "rul" / "protocol.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(ROOT / "src"))
from detect import PCAMonitor  # noqa: E402  (TEP 이상 탐지에 쓴 것과 같은 클래스)

COLS = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
SENSORS = PROTOCOL["columns"]["sensors"]
SENSOR_NAMES = {
    "s2": "T24 LPC 출구 온도", "s3": "T30 HPC 출구 온도", "s4": "T50 LPT 출구 온도",
    "s7": "P30 HPC 출구 압력", "s8": "Nf 팬 회전수", "s9": "Nc 코어 회전수",
    "s11": "Ps30 HPC 출구 정압", "s12": "phi 연료유량/Ps30", "s13": "NRf 보정 팬 회전수",
    "s14": "NRc 보정 코어 회전수", "s15": "BPR 바이패스비", "s17": "htBleed 블리드 엔탈피",
    "s20": "W31 HPT 냉각 블리드", "s21": "W32 LPT 냉각 블리드",
}
CAP = 125
MEAN_WIN, SLOPE_WIN, SLOPE_MIN = 10, 30, 5
HEALTHY_CYCLES, MIN_PREFIX = 30, 31
ALARM_RUN, ALARM_Q = 5, 0.99


# ---------------- 데이터 ----------------
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name, sep=r"\s+", header=None, names=COLS)


def load():
    """(train, test, test 정답 RUL 배열). train에는 life, RUL 열을 붙인다."""
    tr, te = read("train_FD001.txt"), read("test_FD001.txt")
    rul = pd.read_csv(DATA / "RUL_FD001.txt", header=None).iloc[:, 0].to_numpy()
    tr["life"] = tr.groupby("unit").cycle.transform("max")
    tr["RUL"] = tr.life - tr.cycle
    return tr, te, rul


def low_variance_sensors(tr: pd.DataFrame) -> list[str]:
    """학습 데이터에서 고유값이 2개 이하인 센서 (protocol의 제외 규칙)."""
    return [c for c in COLS[5:] if tr[c].nunique() <= 2]


# ---------------- 특징 (과거 값만) ----------------
def _slope(x: np.ndarray) -> float:
    n = len(x)
    if n < SLOPE_MIN:
        return 0.0
    t = np.arange(n) - (n - 1) / 2
    return float((t * (x - x.mean())).sum() / (t**2).sum())


def rolling_slope(x: pd.Series) -> np.ndarray:
    """최근 30사이클(짧으면 있는 만큼) 최소제곱 기울기. _slope를 롤링 합으로 계산한 것."""
    t = pd.Series(np.arange(len(x), dtype=float), index=x.index)
    r = lambda s: s.rolling(SLOPE_WIN, min_periods=1).sum()
    n = x.rolling(SLOPE_WIN, min_periods=1).count()
    st, sx, stx, stt = r(t), r(x), r(t * x), r(t * t)
    den = n * stt - st**2
    out = np.where(den > 0, (n * stx - st * sx) / den.replace(0, np.nan), 0.0)
    return np.where(n < SLOPE_MIN, 0.0, out)


def features(df: pd.DataFrame) -> pd.DataFrame:
    """엔진별로 시점 t까지의 값만 써서 특징을 만든다. 행 순서는 입력과 같다."""
    out = []
    for _, g in df.groupby("unit", sort=False):
        f = pd.DataFrame(index=g.index)
        f["cycle"] = g.cycle.values
        for s in SENSORS:
            x = g[s].astype(float)
            f[f"{s}_last"] = x.values
            f[f"{s}_mean"] = x.rolling(MEAN_WIN, min_periods=1).mean().values
            f[f"{s}_slope"] = rolling_slope(x)
        out.append(f)
    return pd.concat(out).loc[df.index]


def smoothed(df: pd.DataFrame) -> pd.DataFrame:
    """열화 경보 입력: 센서별 최근 10사이클 평균 (과거 값만)."""
    return df.groupby("unit", sort=False)[SENSORS].transform(
        lambda x: x.rolling(MEAN_WIN, min_periods=1).mean())


# ---------------- 잘라 낸 평가 시점 ----------------
def truncation_points(tr: pd.DataFrame, units, k: int = 5, seed: int = 0) -> pd.Index:
    """검증 엔진마다 잘린 시점 k개 (31사이클 이상, 수명 이내). 그 시점 행의 index를 돌려준다.
    잘린 시점의 마지막 행만 평가하므로, 특징이 과거 값만 쓰는 한 행을 고르는 것과 같다."""
    rng = np.random.default_rng(seed)
    idx = []
    for u in units:
        g = tr[tr.unit == u]
        cyc = rng.choice(np.arange(MIN_PREFIX, g.cycle.max() + 1), size=k, replace=False)
        idx.extend(g.index[g.cycle.isin(cyc)])
    return pd.Index(idx)


def last_rows(df: pd.DataFrame) -> pd.Index:
    return df.groupby("unit").cycle.idxmax().sort_index().to_numpy()


# ---------------- 열화 경보 ----------------
class OnsetDetector:
    """PCA T²+SPE(src/detect.py)로 '처음 30사이클과 달라진 시점'을 찾는다.
    학습 엔진을 반으로 나눠 한쪽으로 PCA, 다른 쪽 처음 30사이클로 99% 기준선."""

    def fit(self, tr: pd.DataFrame, units, seed: int = 0):
        units = np.array(sorted(units))
        rng = np.random.default_rng(seed)
        perm = rng.permutation(units)
        half = len(perm) // 2
        self.fit_units, self.cal_units = sorted(perm[:half]), sorted(perm[half:])
        X = smoothed(tr)
        healthy = tr.cycle <= HEALTHY_CYCLES
        self.mon = PCAMonitor(cpv=0.90, lags=0).fit(X[healthy & tr.unit.isin(self.fit_units)].to_numpy())
        self.mon.calibrate(X[healthy & tr.unit.isin(self.cal_units)].to_numpy(), q=ALARM_Q)
        return self

    def scores(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.mon.score(smoothed(df).to_numpy()), index=df.index)

    def alarm_flags(self, df: pd.DataFrame) -> pd.Series:
        """각 행 시점에 경보가 이미 울렸는지 (점수>1이 5사이클 연속 나온 적이 있는지). 과거 값만 사용."""
        over = (self.scores(df) > 1).astype(int)
        def per_unit(x):
            run = x.groupby((x == 0).cumsum()).cumsum()
            return (run >= ALARM_RUN).cummax()
        return over.groupby(df.unit, sort=False).transform(per_unit).astype(bool)

    def alarm_cycle(self, df: pd.DataFrame) -> pd.Series:
        """엔진별 첫 경보 사이클 (없으면 NaN)."""
        fl = self.alarm_flags(df)
        return df[fl].groupby("unit").cycle.min().reindex(sorted(df.unit.unique()))


# ---------------- 라벨 ----------------
def label_cap(rul: pd.Series, cap) -> pd.Series:
    return rul if cap is None else rul.clip(upper=cap)


def label_onset(tr: pd.DataFrame, alarm_cyc: pd.Series, fallback: float) -> pd.Series:
    """엔진별 상한 = 경보 시점의 RUL (경보가 없으면 fallback)."""
    knee = (tr.groupby("unit").life.first() - alarm_cyc).fillna(fallback)
    return np.minimum(tr.RUL, tr.unit.map(knee))


# ---------------- 지표 ----------------
def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def nasa_score(y, p) -> float:
    d = np.asarray(p, float) - np.asarray(y, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def late_ratio(y, p) -> float:
    return float(np.mean(np.asarray(p) > np.asarray(y)))


BUCKETS = [(0, 30), (31, 80), (81, 10_000)]


def bucket_rmse(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = {}
    for lo, hi in BUCKETS:
        m = (y >= lo) & (y <= hi)
        key = f"{lo}~{hi}" if hi < 10_000 else f">{lo - 1}"
        out[key] = {"n": int(m.sum()), "rmse": rmse(y[m], p[m]) if m.any() else None}
    return out


def summary(y, p) -> dict:
    return {"rmse": rmse(y, p), "rmse_vs_cap125": rmse(np.minimum(y, CAP), p),
            "nasa": nasa_score(y, p), "late": late_ratio(y, p), "by_rul": bucket_rmse(y, p)}
