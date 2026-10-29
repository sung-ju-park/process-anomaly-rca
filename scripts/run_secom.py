"""SECOM 실제 반도체 공정 데이터 분석.

질문: '불량과 관련된 센서'로 보이는 것이 정말 불량 원인 후보인가, 시간에 따른 공정 변화의 흔적인가?
1) 데이터 정리 로그
2) 검증 방식의 함정: 무작위 분할 vs 시간순 분할
3) 단순 비교 vs 같은 주(week) 안에서 비교한 원인 후보 센서
4) SHAP 상위 센서와 부트스트랩 안정성
5) 미래 기간(시간순 뒤 30%)에서 재현되는가
6) 불량률 변화 시점과 그 시점 전후로 바뀐 센서
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from secom import Prep, load, time_split  # noqa: E402

OUT = ROOT / "reports"
rng = np.random.default_rng(0)
X, y, ts = load()
yv = y.values
summary = {"n_lots": len(X), "n_sensors": X.shape[1], "n_fail": int(yv.sum()),
           "period": [str(ts.min()), str(ts.max())]}

# ---- 1) 데이터 정리 ----
tr, te = time_split(len(X))
prep = Prep().fit(X.iloc[tr])
summary["prep_log"] = prep.log
summary["missing_rate"] = float(X.isna().values.mean())
A, B = prep.transform(X.iloc[tr]), prep.transform(X.iloc[te])
ytr, yte = yv[tr], yv[te]
wk = ts.dt.to_period("W").astype(str).values
wk_tr, wk_te = wk[tr], wk[te]
summary["split"] = {"train_until": str(ts.iloc[tr[-1]]), "train_fail_rate": float(ytr.mean()),
                    "test_fail_rate": float(yte.mean())}

weekly = pd.DataFrame({"week": wk, "fail": yv}).groupby("week").fail.agg(["mean", "count"]).reset_index()
weekly.to_csv(OUT / "secom_weekly_fail.csv", index=False, encoding="utf-8-sig")


# ---- 2) 검증 방식의 함정 ----
def cv_auc(splitter):
    aucs, lifts = [], []
    for a, b in splitter.split(X, y):
        p = Prep().fit(X.iloc[a])
        Za, Zb = p.transform(X.iloc[a], True).clip(-10, 10), p.transform(X.iloc[b], True).clip(-10, 10)
        m = RandomForestClassifier(500, min_samples_leaf=3, class_weight="balanced_subsample",
                                   random_state=0, n_jobs=2).fit(Za, yv[a])
        s = m.predict_proba(Zb)[:, 1]
        aucs.append(roc_auc_score(yv[b], s))
        lifts.append(average_precision_score(yv[b], s) / yv[b].mean())
    return {"roc_auc_mean": float(np.mean(aucs)), "roc_auc_std": float(np.std(aucs)),
            "pr_lift_mean": float(np.mean(lifts))}


summary["validation"] = {"random_5fold": cv_auc(StratifiedKFold(5, shuffle=True, random_state=0)),
                         "time_5fold": cv_auc(TimeSeriesSplit(5))}


# ---- 3) 원인 후보 센서: 단순 비교 vs 같은 주 안 비교 ----
def mw(Z: pd.DataFrame, yy: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in Z.columns:
        a, b = Z[c].values[yy == 1], Z[c].values[yy == 0]
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        rows.append((c, p, 2 * u / (len(a) * len(b)) - 1))
    r = pd.DataFrame(rows, columns=["sensor", "p", "delta"]).set_index("sensor")
    r["q"] = multipletests(r.p, method="fdr_bh")[1]
    return r


def within_week(Z: pd.DataFrame, w: np.ndarray) -> pd.DataFrame:
    return Z.groupby(w).rank(pct=True)


naive = mw(A, ytr)
strat = mw(within_week(A, wk_tr), ytr)
time_corr = A.apply(lambda s: np.corrcoef(s.values, np.arange(len(s)))[0, 1])
naive_sig = naive[naive.q < 0.1].index
strat_sig = strat[strat.q < 0.1].index
summary["candidates"] = {
    "naive_q10": int(len(naive_sig)),
    "naive_q10_time_corr_gt_0_3": int((time_corr[naive_sig].abs() > 0.3).sum()),
    "within_week_q10": int(len(strat_sig)),
    "within_week_sensors": list(strat_sig),
}

# ---- 4) SHAP 상위 센서와 안정성 ----
import shap  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402


def xgb():
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.5, scale_pos_weight=(1 - ytr.mean()) / ytr.mean(),
                         random_state=0, n_jobs=2, verbosity=0)


Ztr = prep.transform(X.iloc[tr], True).clip(-10, 10)
m = xgb().fit(Ztr, ytr)
sv = np.abs(shap.TreeExplainer(m).shap_values(Ztr)).mean(0)
shap_rank = pd.Series(sv, index=Ztr.columns).sort_values(ascending=False)
top10 = list(shap_rank.index[:10])
B_ROUNDS = 30
freq = pd.Series(0, index=Ztr.columns, dtype=float)
for i in range(B_ROUNDS):
    idx = rng.choice(len(Ztr), len(Ztr), replace=True)
    mb = xgb().set_params(random_state=i).fit(Ztr.iloc[idx], ytr[idx])
    s = pd.Series(np.abs(shap.TreeExplainer(mb).shap_values(Ztr)).mean(0), index=Ztr.columns)
    freq[s.sort_values(ascending=False).index[:10]] += 1
freq /= B_ROUNDS
summary["shap"] = {
    "top10": top10,
    "top10_bootstrap_freq": {c: float(freq[c]) for c in top10},
    "stable_ge_0_7": [c for c in top10 if freq[c] >= 0.7],
    "top10_time_corr_gt_0_3": [c for c in top10 if abs(time_corr[c]) > 0.3],
}

# ---- 5) 미래 기간 재현 ----
rep_rows = []
for c in sorted(set(strat_sig) | set(top10)):
    r_te = within_week(B[[c]], wk_te)[c].values
    u, p = mannwhitneyu(r_te[yte == 1], r_te[yte == 0], alternative="two-sided")
    d_te = 2 * u / ((yte == 1).sum() * (yte == 0).sum()) - 1
    rep_rows.append({"sensor": c, "train_delta": float(strat.delta[c]), "train_q": float(strat.q[c]),
                     "test_delta": float(d_te), "test_p": float(p),
                     "replicated": bool(p < 0.05 and np.sign(d_te) == np.sign(strat.delta[c]))})
rep = pd.DataFrame(rep_rows)
rep.to_csv(OUT / "secom_replication.csv", index=False, encoding="utf-8-sig")
summary["replication"] = {"tested": int(len(rep)), "replicated": int(rep.replicated.sum()),
                          "replicated_sensors": list(rep[rep.replicated].sensor)}


# ---- 6) 불량률 변화 시점 ----
def ll(k, n):
    if n == 0 or k in (0, n):
        return 0.0
    p = k / n
    return k * np.log(p) + (n - k) * np.log(1 - p)


def best_change(v, margin=100):
    c, n = np.cumsum(v), len(v)
    best = max(range(margin, n - margin), key=lambda t: ll(c[t - 1], t) + ll(c[-1] - c[t - 1], n - t))
    lr = 2 * (ll(c[best - 1], best) + ll(c[-1] - c[best - 1], n - best) - ll(c[-1], n))
    return best, lr


t_cp, lr = best_change(yv)
perm = [best_change(rng.permutation(yv))[1] for _ in range(300)]
p_cp = (1 + sum(p >= lr for p in perm)) / (1 + len(perm))
prep_all = Prep().fit(X)
Zall = prep_all.transform(X)
rows = []
for c in Zall.columns:
    a, b = Zall[c].values[:t_cp], Zall[c].values[t_cp:]
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    rows.append((c, p, 2 * u / (len(a) * len(b)) - 1))
shift = pd.DataFrame(rows, columns=["sensor", "p", "delta_before_vs_after"]).set_index("sensor")
shift["q"] = multipletests(shift.p, method="fdr_bh")[1]
shift = shift.reindex(shift.delta_before_vs_after.abs().sort_values(ascending=False).index)
shifted = shift[(shift.q < 0.01) & (shift.delta_before_vs_after.abs() > 0.5)]
shift.head(30).to_csv(OUT / "secom_shift_sensors.csv", encoding="utf-8-sig")
summary["change_point"] = {
    "index": int(t_cp), "time": str(ts.iloc[t_cp]),
    "fail_rate_before": float(yv[:t_cp].mean()), "fail_rate_after": float(yv[t_cp:].mean()),
    "perm_p": float(p_cp), "n_shifted": int(len(shifted)), "shifted": list(shifted.index),
    "overlap_with_naive_candidates": sorted(set(shifted.index) & set(naive_sig)),
    "overlap_with_shap_top10": sorted(set(shifted.index) & set(top10)),
}
# 차트용: 주별 중앙값
cols = list(shifted.index[:4])
wk_med = pd.DataFrame(Zall[cols].values, columns=cols)
wk_med["week"] = wk
wk_med.groupby("week").median().reset_index().to_csv(OUT / "secom_weekly_sensor_median.csv",
                                                      index=False, encoding="utf-8-sig")
json.dump(summary, open(OUT / "secom_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
