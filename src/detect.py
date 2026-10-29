"""이상 탐지 모델.

모든 모델은 같은 인터페이스를 따른다.
    fit(X_normal)      정상 데이터로 학습
    score(X) -> (n,)   이상 점수 (클수록 이상)
임계값은 학습에 쓰지 않은 정상 보정 세트(calib)의 99% 분위수로 정한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def lagged(X: np.ndarray, lags: int) -> np.ndarray:
    """동적 PCA용 시간 지연 행렬. [x_t, x_{t-1}, ..., x_{t-lags}] (앞 lags개 행은 첫 행으로 채움)."""
    if lags == 0:
        return X
    pads = [np.vstack([np.repeat(X[:1], k, axis=0), X[: len(X) - k]]) for k in range(lags + 1)]
    return np.hstack(pads)


class PCAMonitor:
    """PCA 기반 통계적 공정 관리. T²(모델 안쪽 변화) + SPE/Q(모델 바깥 변화).

    lags>0 이면 동적 PCA(DPCA): 시간 상관을 모델에 넣는다.
    """

    def __init__(self, cpv: float = 0.90, lags: int = 0):
        self.cpv, self.lags = cpv, lags

    def fit(self, X):
        X = np.asarray(X, float)
        self.sc = StandardScaler().fit(X)
        Z = lagged(self.sc.transform(X), self.lags)
        self.zm, self.zs = Z.mean(0), Z.std(0) + 1e-12
        Z = (Z - self.zm) / self.zs
        full = PCA().fit(Z)
        self.k = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), self.cpv) + 1)
        self.P = full.components_[: self.k].T  # (m, k)
        self.lam = full.explained_variance_[: self.k]
        return self

    def _z(self, X):
        Z = lagged(self.sc.transform(np.asarray(X, float)), self.lags)
        return (Z - self.zm) / self.zs

    def t2_spe(self, X):
        Z = self._z(X)
        T = Z @ self.P
        t2 = np.sum(T**2 / self.lam, axis=1)
        R = Z - T @ self.P.T
        spe = np.sum(R**2, axis=1)
        return t2, spe

    def calibrate(self, Xc, q=0.99):
        t2, spe = self.t2_spe(Xc)
        self.t2_lim, self.spe_lim = np.quantile(t2, q), np.quantile(spe, q)
        return self

    def score(self, X):
        """T²와 SPE를 각자 기준선으로 나눈 값 중 큰 쪽 (1 넘으면 경보)."""
        t2, spe = self.t2_spe(X)
        return np.maximum(t2 / self.t2_lim, spe / self.spe_lim)

    # ---- 재구성 기반 기여도 (RBC, Alcala & Qin 2009) ----
    def rbc(self, X):
        """변수별 기여도. 결합 지표 φ = P Λ⁻¹ Pᵀ/τ² + (I − P Pᵀ)/δ² 기준.
        동적 PCA면 같은 원 변수의 지연 열들을 합산한다."""
        Z = self._z(X)
        m = Z.shape[1]
        Phi = (self.P / self.lam) @ self.P.T / self.t2_lim + (np.eye(m) - self.P @ self.P.T) / self.spe_lim
        num = (Z @ Phi) ** 2  # (n, m): (ξ_iᵀ φ z)²
        den = np.diag(Phi)  # ξ_iᵀ φ ξ_i
        c = num / den
        nvar = m // (self.lags + 1)
        return c.reshape(len(Z), self.lags + 1, nvar).sum(1)


class IForestMonitor:
    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X):
        self.sc = StandardScaler().fit(X)
        self.m = IsolationForest(n_estimators=300, random_state=self.seed).fit(self.sc.transform(X))
        return self

    def raw(self, X):
        return -self.m.score_samples(self.sc.transform(X))

    def calibrate(self, Xc, q=0.99):
        self.lim = np.quantile(self.raw(Xc), q)
        return self

    def score(self, X):
        return self.raw(X) / self.lim


class AEMonitor:
    """작은 오토인코더(병목 MLP). 재구성 오차를 이상 점수로."""

    def __init__(self, hidden=(32, 12, 32), seed=0):
        self.hidden, self.seed = hidden, seed

    def fit(self, X):
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        self.m = MLPRegressor(hidden_layer_sizes=self.hidden, activation="tanh", alpha=1e-3,
                              max_iter=3000, random_state=self.seed, early_stopping=True,
                              validation_fraction=0.15, n_iter_no_change=50).fit(Z, Z)
        return self

    def raw(self, X):
        Z = self.sc.transform(X)
        return np.sum((Z - self.m.predict(Z)) ** 2, axis=1)

    def calibrate(self, Xc, q=0.99):
        self.lim = np.quantile(self.raw(Xc), q)
        return self

    def score(self, X):
        return self.raw(X) / self.lim


def alarms(score: np.ndarray, persist: int = 1) -> np.ndarray:
    """score>1 이 persist개 연속일 때 경보. persist=1이면 즉시 경보."""
    over = score > 1
    if persist <= 1:
        return over
    run = np.convolve(over.astype(int), np.ones(persist, int), mode="full")[: len(over)]
    return run >= persist


def first_alarm(alarm: np.ndarray, start: int) -> int | None:
    idx = np.flatnonzero(alarm[start:])
    return int(start + idx[0]) if len(idx) else None


MODELS = {
    "PCA": lambda: PCAMonitor(lags=0),
    "DPCA": lambda: PCAMonitor(lags=2),
    "IsolationForest": lambda: IForestMonitor(),
    "AutoEncoder": lambda: AEMonitor(),
}
