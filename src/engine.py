"""대시보드·리포트가 쓰는 분석 엔진.

정상 데이터로 DPCA 탐지기를 학습하고, 과거 고장 사례 라이브러리를 만든 뒤
경보가 울리면 원인 후보·전파 경로·유사 사례를 한 번에 돌려준다.
원인 순위는 사전에 정한 'rbc+onset'(기여도 순위 + 먼저 벗어난 순서의 평균).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from detect import PCAMonitor, alarms, first_alarm
from rca import analyze, propagation
from signature import build_library, nearest, signature
from tep_data import SAMPLE_MIN, load, normal_sets
from tep_meta import FAULTS, UNIT_OF, label

PERSIST = 3
PRIMARY = "rbc+onset"

CHECKLIST = {
    "원료 공급": ["원료 유량계와 공급 밸브 개도 확인", "원료 조성 분석값(성분 비율) 확인", "공급 헤더 압력 확인"],
    "반응기": ["반응기 냉각수 입구 온도·유량 확인", "냉각수 밸브 동작(고착 여부) 확인", "반응기 온도 제어 루프 설정값 확인"],
    "응축기": ["응축기 냉각수 입구 온도·유량 확인", "응축기 냉각수 밸브 동작(고착 여부) 확인"],
    "분리기": ["분리기 온도·액위 제어 상태 확인", "분리기 하부 배출 밸브 확인"],
    "압축기·재순환": ["압축기 부하(일) 추이 확인", "재순환 밸브 개도 확인"],
    "퍼지": ["퍼지 밸브 개도 확인", "불활성 성분(B 등) 축적 여부 확인"],
    "스트리퍼·제품": ["스트리퍼 스팀 밸브·유량 확인", "제품 조성 분석값 확인"],
}


@dataclass
class Finding:
    scenario: str
    t_alarm: int
    alarm_min: int
    causes: pd.DataFrame       # 원인 후보 (순위순)
    path: pd.DataFrame         # 전파 경로
    similar: pd.DataFrame      # 유사 과거 사례
    score: np.ndarray          # 이상 점수 (1 넘으면 기준선 초과)
    ranking: dict              # 방법별 전체 순위 점수


class Engine:
    def __init__(self):
        self.fit_df, calib, _ = normal_sets()
        self.mon = PCAMonitor(lags=2).fit(self.fit_df.values).calibrate(calib.values)
        cases = {f"F{f:02d}": load(f, "train").values for f in range(1, 22)}
        self.lib_sigs, self.lib_names = build_library(cases, self.mon)

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.mon.score(X)

    def first_alarm(self, X: np.ndarray, start: int = 0) -> int | None:
        return first_alarm(alarms(self.score(X), PERSIST), start)

    def investigate(self, X: np.ndarray, t_alarm: int, scenario: str = "", with_shap: bool = False) -> Finding:
        res = analyze(X, t_alarm, self.mon, self.fit_df.values, SAMPLE_MIN, with_shap=with_shap)
        prim = res.scores[PRIMARY].sort_values(ascending=False)
        rbc = res.scores["rbc"]
        share = rbc / rbc.sum()
        Z = self.mon.sc.transform(X)[t_alarm: t_alarm + 20]
        zmean = pd.Series(Z.mean(0), index=rbc.index)
        top = prim.index[:5]
        causes = pd.DataFrame({
            "순위": range(1, len(top) + 1),
            "변수": top,
            "이름": [label(v) for v in top],
            "공정 단위": [UNIT_OF[v] for v in top],
            "이탈 시작(경보 기준, 분)": [res.onset_min[v] for v in top],
            "기여도 비중(%)": [round(100 * share[v], 1) for v in top],
            "정상 대비 변화(σ)": [round(zmean[v], 1) for v in top],
        })
        path = propagation(res.onset_min, top=8)
        path["이름"] = [label(v) for v in path["var"]]
        nb = nearest(self.lib_sigs, self.lib_names, signature(X, t_alarm, self.mon), 3)
        similar = pd.DataFrame({
            "과거 사례": nb.index,
            "고장 내용": [FAULTS[int(n[1:])][0] for n in nb.index],
            "유형": [FAULTS[int(n[1:])][1] for n in nb.index],
            "유사도": nb.values.round(3),
        })
        return Finding(scenario, t_alarm, t_alarm * SAMPLE_MIN, causes, path, similar,
                       self.score(X), res.scores)
