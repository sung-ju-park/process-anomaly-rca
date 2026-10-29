"""SECOM 반도체 공정 데이터: 로드와 전처리.

- 1,567개 lot × 센서 590개, 라벨 -1(합격)/1(불량), 측정 시각 포함
- 시간순으로 앞 70%를 학습, 뒤 30%를 테스트로 나눈다 (무작위 분할은 미래 정보가 섞임)
- 전처리 규칙은 학습 구간에서만 정하고 테스트에는 그대로 적용한다
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "secom"


def load() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    X = pd.read_csv(RAW / "secom.data", sep=" ", header=None)
    X.columns = [f"S{i:03d}" for i in range(X.shape[1])]
    lab = pd.read_csv(RAW / "secom_labels.data", sep=" ", header=None, names=["y", "ts"])
    ts = pd.to_datetime(lab.ts.str.strip('"'), format="%d/%m/%Y %H:%M:%S")
    y = (lab.y == 1).astype(int)
    order = np.argsort(ts.values, kind="stable")
    return X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True), ts.iloc[order].reset_index(drop=True)


def time_split(n: int, frac: float = 0.7):
    cut = int(n * frac)
    return np.arange(cut), np.arange(cut, n)


@dataclass
class Prep:
    """학습 구간 기준 전처리. 단계별로 몇 개 센서가 빠졌는지 기록한다."""
    max_missing: float = 0.5
    min_std: float = 1e-8
    dup_corr: float = 0.98
    log: list = field(default_factory=list)

    def fit(self, X: pd.DataFrame):
        cols = list(X.columns)
        self.log = [("원본 센서", len(cols))]
        miss = X[cols].isna().mean()
        cols = [c for c in cols if miss[c] <= self.max_missing]
        self.log.append((f"결측 {int(self.max_missing * 100)}% 초과 제거", len(cols)))
        std = X[cols].std()
        cols = [c for c in cols if std[c] > self.min_std and X[c].nunique() > 1]
        self.log.append(("값이 변하지 않는 센서 제거", len(cols)))
        self.median = X[cols].median()
        Xi = X[cols].fillna(self.median)
        corr = Xi.corr().abs().values
        keep = []
        for j, c in enumerate(cols):
            if all(corr[j, cols.index(k)] < self.dup_corr for k in keep):
                keep.append(c)
        self.log.append((f"상관 {self.dup_corr} 이상 중복 센서 제거", len(keep)))
        self.cols = keep
        self.mu = Xi[keep].mean()
        self.sd = Xi[keep].std().replace(0, 1)
        return self

    def transform(self, X: pd.DataFrame, scale: bool = False) -> pd.DataFrame:
        Z = X[self.cols].fillna(self.median[self.cols])
        return (Z - self.mu) / self.sd if scale else Z
