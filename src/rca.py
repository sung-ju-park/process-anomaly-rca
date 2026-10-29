"""근본 원인 분석(RCA).

경보가 울린 시점(t_d)부터 W개 샘플(기본 20개 = 1시간)을 '경보 구간'으로 보고,
여러 방법으로 원인 변수 순위를 만든다. 정답 채점은 scripts/run_tep_rca.py.

방법
- zscore  : 정상 대비 평균 |z| (단변량 기준선)
- spe     : PCA 잔차 기여도 (고전적 기여도 그림)
- rbc     : 재구성 기반 기여도 (기여도 '번짐'을 줄인 방법)
- shap    : 정상 vs 경보 구간 분류 모델(XGBoost)의 SHAP 중요도
- onset   : 어느 변수가 먼저 정상 범위를 벗어났는가 (시간 선후)
- rbc+onset : rbc 순위와 onset 순위의 평균 (이 프로젝트의 제안 방법, 사전에 고정)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tep_meta import COLS, UNIT_OF

WINDOW = 20        # 경보 후 1시간
LOOKBACK = 20      # 시간 선후는 경보 1시간 전부터 추적
EWMA_LAMBDA = 0.3
ONSET_Z = 3.0
ONSET_PERSIST = 3


@dataclass
class RCAResult:
    t_alarm: int
    scores: dict          # method -> pd.Series(변수별 점수, 클수록 원인 가능성 높음)
    onset_min: pd.Series  # 변수별 이탈 시작 시각(분, 경보 시점 기준 상대값, 이탈 없으면 NaN)


def ewma(Z: np.ndarray, lam: float) -> np.ndarray:
    out = np.empty_like(Z)
    out[0] = Z[0]
    for i in range(1, len(Z)):
        out[i] = lam * Z[i] + (1 - lam) * out[i - 1]
    return out


def onset_times(Zseg: np.ndarray) -> np.ndarray:
    """각 변수가 |EWMA z| > ONSET_Z 를 ONSET_PERSIST번 연속 넘은 첫 인덱스 (없으면 NaN)."""
    E = np.abs(ewma(Zseg, EWMA_LAMBDA)) > ONSET_Z
    out = np.full(Zseg.shape[1], np.nan)
    for j in range(Zseg.shape[1]):
        run = np.convolve(E[:, j].astype(int), np.ones(ONSET_PERSIST, int), mode="full")[: len(E)]
        idx = np.flatnonzero(run >= ONSET_PERSIST)
        if len(idx):
            out[j] = idx[0] - ONSET_PERSIST + 1
    return out


def rank_score(s: pd.Series) -> pd.Series:
    """점수 → 순위 점수(1등이 가장 큼). 순위 평균 결합용."""
    return s.rank(ascending=True, method="average")


def analyze(X: np.ndarray, t_alarm: int, monitor, normal: np.ndarray, sample_min: int = 3,
            with_shap: bool = True, window: int = WINDOW) -> RCAResult:
    a, b = t_alarm, min(len(X), t_alarm + window)
    Zall = monitor.sc.transform(X)
    win = Zall[a:b]

    scores = {}
    scores["zscore"] = pd.Series(np.abs(win).mean(0), index=COLS)
    # SPE 기여도: 잔차 제곱을 원 변수 단위로 합산
    Zl = monitor._z(X)
    R = Zl - (Zl @ monitor.P) @ monitor.P.T
    nvar = len(COLS)
    spe_c = (R[a:b] ** 2).reshape(b - a, monitor.lags + 1, nvar).sum(1).mean(0)
    scores["spe"] = pd.Series(spe_c, index=COLS)
    scores["rbc"] = pd.Series(monitor.rbc(X)[a:b].mean(0), index=COLS)

    if with_shap:
        import shap
        from xgboost import XGBClassifier

        Zn = monitor.sc.transform(normal)
        Xc = np.vstack([Zn, win])
        yc = np.r_[np.zeros(len(Zn)), np.ones(len(win))]
        clf = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.9,
                            colsample_bytree=0.8, random_state=0, n_jobs=2,
                            scale_pos_weight=len(Zn) / max(1, len(win)), verbosity=0)
        clf.fit(Xc, yc)
        sv = shap.TreeExplainer(clf).shap_values(win)
        scores["shap"] = pd.Series(np.abs(sv).mean(0), index=COLS)

    # 시간 선후: 경보 LOOKBACK 전부터 경보 구간 끝까지
    s0 = max(0, a - LOOKBACK)
    on = onset_times(Zall[s0:b])
    onset_rel = pd.Series((on + s0 - a) * sample_min, index=COLS)  # 경보 시점 기준(분), 음수=경보 전
    # 먼저 벗어날수록 높은 점수, 같은 시각이면 크기(zscore)로 순서
    big = 10_000.0
    ons = pd.Series(np.where(np.isnan(on), big, on), index=COLS)
    scores["onset"] = -ons + 1e-6 * scores["zscore"]

    scores["rbc+onset"] = (rank_score(scores["rbc"]) + rank_score(scores["onset"])) / 2 \
        + 1e-6 * rank_score(scores["rbc"])
    if with_shap:
        scores["shap+onset"] = (rank_score(scores["shap"]) + rank_score(scores["onset"])) / 2 \
            + 1e-6 * rank_score(scores["shap"])
    return RCAResult(t_alarm=t_alarm, scores=scores, onset_min=onset_rel)


def grade(score: pd.Series, truth: list[str], truth_unit: str) -> dict:
    order = list(score.sort_values(ascending=False).index)
    best = min(order.index(v) for v in truth) + 1
    return {
        "hit1": best <= 1,
        "hit3": best <= 3,
        "best_rank": best,
        "rr": 1.0 / best,
        "unit1": UNIT_OF[order[0]] == truth_unit,
        "top3": order[:3],
    }


def propagation(onset_min: pd.Series, top: int = 6) -> pd.DataFrame:
    """먼저 벗어난 순서대로 변수와 공정 단위."""
    s = onset_min.dropna().sort_values().head(top)
    return pd.DataFrame({"var": s.index, "minute_from_alarm": s.values,
                         "unit": [UNIT_OF[v] for v in s.index]})
