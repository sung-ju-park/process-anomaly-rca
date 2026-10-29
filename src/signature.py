"""고장 시그니처와 유사 사례 검색.

시그니처 = [변수별 평균 z (부호 포함), 변수별 log 표준편차] 를 길이 1로 정규화한 벡터.
  - 평균 z     : 계단형 고장(값이 한쪽으로 이동)을 잡는다
  - log 표준편차: 랜덤 변동·밸브 고착형 고장(평균은 그대로, 흔들림만 커짐)을 잡는다
라이브러리는 과거 사례(학습용 고장 파일)에서 20개(1시간) 구간을 10개씩 밀며 여러 개 저장하고,
새 경보 구간과 코사인 유사도가 가장 높은 사례를 찾는다.
→ 현장에서 '예전에 이런 패턴이 있었나'를 찾던 일을 자동화

v1(평균 z만, 사례당 1구간)은 테스트에서 top-1 44%에 그쳐 dev에서 위 방식을 검증한 뒤 바꿨다.
(자세한 경위는 README 의사결정 로그)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 20
STEP = 10


def signature_from_z(Z: np.ndarray) -> np.ndarray:
    v = np.r_[Z.mean(0), np.log(Z.std(0) + 1e-3)]
    return v / (np.linalg.norm(v) + 1e-12)


def signature(X: np.ndarray, t_alarm: int, monitor, window: int = WINDOW) -> np.ndarray:
    return signature_from_z(monitor.sc.transform(X)[t_alarm: t_alarm + window])


def build_library(cases: dict, monitor, window: int = WINDOW, step: int = STEP):
    """cases: {사례명: X(원 단위)} → (시그니처 행렬, 사례명 배열)"""
    sigs, names = [], []
    for name, X in cases.items():
        Z = monitor.sc.transform(X)
        for s in range(0, len(Z) - window + 1, step):
            sigs.append(signature_from_z(Z[s: s + window]))
            names.append(name)
    return np.array(sigs), np.array(names)


def nearest(lib_sigs: np.ndarray, lib_names: np.ndarray, sig: np.ndarray, k: int = 3) -> pd.Series:
    """사례별 최고 유사도 상위 k."""
    sims = pd.Series(lib_sigs @ sig).groupby(lib_names).max()
    return sims.sort_values(ascending=False).head(k)


def case_profile(lib_sigs: np.ndarray, lib_names: np.ndarray) -> pd.DataFrame:
    """사례별 평균 시그니처 (클러스터링용)."""
    df = pd.DataFrame(lib_sigs)
    df["case"] = lib_names
    m = df.groupby("case").mean()
    return m.div(np.linalg.norm(m.values, axis=1), axis=0)
