"""고장 시그니처 클러스터링과 유사 사례 검색 평가.

- 라이브러리: 학습용 고장 파일 d01~d21 (과거 사례)
- 질의     : 테스트 파일 d01_te~d21_te (새 경보)
- 평가     : 질의마다 가장 비슷한 과거 사례가 같은 고장 번호인지 (top-1, top-3)
             고장 3·9·15는 경보 자체가 오경보 수준이라 평가에서 제외하고 따로 표시
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detect import PCAMonitor, alarms, first_alarm  # noqa: E402
from signature import build_library, case_profile, nearest, signature  # noqa: E402
from tep_data import load, normal_sets, onset  # noqa: E402

OUT = ROOT / "reports"
PERSIST = 3
UNDETECTABLE = {3, 9, 15}

fit, calib, _ = normal_sets()
mon = PCAMonitor(lags=2).fit(fit.values).calibrate(calib.values)


def sig_for(f, split):
    X = load(f, split).values
    t = first_alarm(alarms(mon.score(X), PERSIST), onset(split))
    return None if t is None else signature(X, t, mon)


lib_cases = {f"F{f:02d}": load(f, "train").values for f in range(1, 22)}
lib_sigs, lib_names = build_library(lib_cases, mon)
lib = case_profile(lib_sigs, lib_names)
lib.to_csv(OUT / "tep_signature_library.csv", encoding="utf-8-sig")

rows = []
for f in range(1, 22):
    s = sig_for(f, "test")
    if s is None:
        continue
    nb = nearest(lib_sigs, lib_names, s, 3)
    rows.append({"fault": f, "eval": f not in UNDETECTABLE,
                 "top1": nb.index[0], "top1_sim": round(float(nb.iloc[0]), 3),
                 "top3": list(nb.index), "hit1": nb.index[0] == f"F{f:02d}",
                 "hit3": f"F{f:02d}" in nb.index})
res = pd.DataFrame(rows)
ev = res[res["eval"]]
summary = {"n_eval": int(len(ev)), "top1_acc": float(ev.hit1.mean()), "top3_acc": float(ev.hit3.mean())}

# 클러스터링: 라벨 없이 경보 구간(1시간, 겹치지 않게)을 묶었을 때 같은 고장끼리 모이는가
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA as _PCA  # noqa: E402
from sklearn.metrics import normalized_mutual_info_score  # noqa: E402

det_faults = [f for f in range(1, 22) if f not in UNDETECTABLE]
win_sigs, win_names = build_library({f"F{f:02d}": load(f, "test").values[160:] for f in det_faults},
                                    mon, step=20)
k = len(det_faults)
km = KMeans(k, n_init=10, random_state=0).fit(win_sigs)
summary["cluster_k"] = k
summary["cluster_windows"] = int(len(win_sigs))
summary["cluster_nmi"] = float(normalized_mutual_info_score(win_names, km.labels_))
summary["cluster_ari"] = float(adjusted_rand_score(win_names, km.labels_))
xy = _PCA(2).fit_transform(win_sigs)
pd.DataFrame({"case": win_names, "cluster": km.labels_, "x": xy[:, 0], "y": xy[:, 1]}).to_csv(
    OUT / "tep_window_clusters.csv", index=False, encoding="utf-8-sig")

res.to_csv(OUT / "tep_similar_case.csv", index=False, encoding="utf-8-sig")
json.dump(summary, open(OUT / "tep_signature_summary.json", "w"), ensure_ascii=False, indent=1)
print(res.to_string())
print(json.dumps(summary, ensure_ascii=False, indent=1))
