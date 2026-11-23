"""사후 분석 (공식 test 결과를 본 뒤 계산한 것). README에서 '사후'라고 표시한 숫자를 모두 여기서 만든다.

    python rul/posthoc.py   → rul/results/posthoc.json

공식 test를 다시 평가하지 않는다. 이미 저장된 test_predictions.csv(seed 0)와 학습 데이터만 쓴다.
"""
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmapss as C  # noqa: E402
from run_rul import CAPS, UNITS, xgb_fit  # noqa: E402

OUT = C.ROOT / "rul" / "results"
p = pd.read_csv(OUT / "test_predictions.csv")
y = p.true_rul.to_numpy()
rng = np.random.default_rng(0)
B = 2000
res = {"note": "공식 test 결과를 본 뒤 계산한 사후 분석. 증명이 아니라 설명용."}


def boot(fn, idx_pool):
    vals = []
    for _ in range(B):
        i = rng.choice(idx_pool, size=len(idx_pool), replace=True)
        vals.append(fn(i))
    return [round(float(np.percentile(vals, 2.5)), 2), round(float(np.percentile(vals, 97.5)), 2)]


# 1) test RMSE의 엔진 표본 불확실성 (부트스트랩 95% 구간, seed 0 예측)
allidx = np.arange(len(p))
res["bootstrap_95"] = {
    "A_xgb_rmse": boot(lambda i: C.rmse(y[i], p.A_xgb.to_numpy()[i]), allidx),
    "B_minus_A_rmse": boot(lambda i: C.rmse(y[i], p.B_xgb.to_numpy()[i]) - C.rmse(y[i], p.A_xgb.to_numpy()[i]), allidx),
    "Ridge_minus_A_rmse": boot(lambda i: C.rmse(y[i], p.A_ridge.to_numpy()[i]) - C.rmse(y[i], p.A_xgb.to_numpy()[i]), allidx),
}
al = np.flatnonzero(p.alarmed.to_numpy())
res["D_vs_A_alarmed"] = {
    "n": int(len(al)),
    "D_closer": int((np.abs(p.D_xgb - p.true_rul) < np.abs(p.A_xgb - p.true_rul)).to_numpy()[al].sum()),
    "D_minus_A_rmse_95": boot(lambda i: C.rmse(y[i], p.D_xgb.to_numpy()[i]) - C.rmse(y[i], p.A_xgb.to_numpy()[i]), al),
}

# 2) D가 경보 없는 엔진에서 늦게 예측한 경우 (고정값보다 실제 RUL이 짧은 엔진)
na = p[~p.alarmed]
late_na = na[na.D_xgb > na.true_rul]
res["D_late_not_alarmed"] = {
    "n": int(len(late_na)), "units": late_na.unit.tolist(), "true_rul": late_na.true_rul.tolist(),
    "late_by": (late_na.D_xgb - late_na.true_rul).round(1).tolist(),
    "n_true_rul_le30": int((late_na.true_rul <= 30).sum())}

# 3) 80% 구간 하한 (실제 RUL<=125)
m = p.true_rul <= C.CAP
lo_late = (p.lo80 > p.true_rul) & m
res["interval_lower_bound_rul_le125"] = {
    "n": int(m.sum()), "late_n": int(lo_late.sum()),
    "late_by_median": round(float((p.lo80 - p.true_rul)[lo_late].median()), 1),
    "late_by_max": round(float((p.lo80 - p.true_rul)[lo_late].max()), 1),
    "early_by_median_all": round(float((p.true_rul - p.lo80)[m].median()), 1),
    "point_late_share_all": round(float((p.A_xgb > p.true_rul).mean()), 2)}

# 4) 교차검증 평가 시점의 RUL 분포와, RUL<=145 시점만 썼을 때의 상한 선택
tr, _, _ = C.load()
F = C.features(tr)
folds = list(GroupKFold(n_splits=5).split(UNITS, groups=UNITS))
evs = [C.truncation_points(tr, UNITS[iva], k=5, seed=0) for _, iva in folds]
ev_all = np.concatenate([tr.RUL.loc[e].to_numpy() for e in evs])
res["cv_eval_rul"] = {"n": int(len(ev_all)), "share_gt145": round(float((ev_all > 145).mean()), 3),
                      "max": int(ev_all.max()), "median": float(np.median(ev_all))}
cap_le145 = {}
for c in CAPS:
    errs = []
    for (itr, _), e in zip(folds, evs):
        mtr = tr.unit.isin(UNITS[itr])
        e2 = e[tr.RUL.loc[e].to_numpy() <= 145]
        pr = xgb_fit(F[mtr], C.label_cap(tr.RUL[mtr], c), 0).predict(F.loc[e2])
        errs.append(C.rmse(tr.RUL.loc[e2], pr))
    cap_le145[str(c)] = round(float(np.mean(errs)), 2)
res["cap_cv_rmse_eval_rul_le145"] = cap_le145

res["versions"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                   "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}
(OUT / "posthoc.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
