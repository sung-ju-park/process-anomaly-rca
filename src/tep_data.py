"""TEP 데이터 로더.

- 샘플 간격 3분. 테스트 파일(dXX_te.dat)은 960개(48시간), 고장은 8시간 시점(160번째 샘플 이후)부터.
- 학습용 고장 파일(dXX.dat)은 480개이며 전부 고장 발생 이후 구간.
- d00.dat(정상 학습)은 원본이 전치(52×500)돼 있어 바로잡는다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from tep_meta import COLS

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "tep"
SAMPLE_MIN = 3
TEST_ONSET = 160  # 테스트 파일에서 고장이 시작되는 샘플 인덱스(0부터)


def load(fault: int, split: str = "test") -> pd.DataFrame:
    name = f"d{fault:02d}{'_te' if split == 'test' else ''}.dat"
    a = np.loadtxt(RAW / name)
    if a.shape[0] == 52 and a.shape[1] != 52:
        a = a.T
    df = pd.DataFrame(a, columns=COLS)
    df.index = pd.Index(np.arange(len(df)) * SAMPLE_MIN, name="minute")
    return df


def onset(split: str) -> int:
    """고장 시작 샘플 인덱스. 학습 고장 파일은 처음부터 고장."""
    return TEST_ONSET if split == "test" else 0


def normal_sets():
    """정상 데이터 분할.
    fit   : d00.dat (500)            — 모델 학습
    calib : d00_te.dat 앞 480개      — 경보 기준선(임계값) 설정
    hold  : d00_te.dat 뒤 480개      — 오경보율 평가 (모델·임계값 결정에 쓰지 않음)
    """
    fit = load(0, "train")
    te = load(0, "test")
    return fit, te.iloc[:480], te.iloc[480:]
