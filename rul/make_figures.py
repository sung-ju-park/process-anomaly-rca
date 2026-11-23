"""RUL 결과 그림 2개 (rul/figures/).

1) test 엔진 3대의 사이클별 예측 RUL(라벨 A XGBoost, 80% 구간)과 실제 RUL, 열화 경보 시점
2) test 100대 마지막 사이클: 실제 RUL 대 오차 (A와 D)

그림 1은 공식 평가와 같은 설정(seed 0)으로 다시 학습해 test 엔진의 모든 사이클을 그린 것이다.
성적표 숫자는 rul/results/test.json(엔진별 마지막 사이클 1회 평가)만 쓴다.
"""
import sys
from pathlib import Path

import logging
import matplotlib
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmapss as C  # noqa: E402
from run_rul import xgb_fit  # noqa: E402

plt.rcParams["font.family"] = ["Malgun Gothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
FIG = C.ROOT / "rul" / "figures"
FIG.mkdir(exist_ok=True)

tr, te, rul = C.load()
pred = pd.read_csv(C.ROOT / "rul" / "results" / "test_predictions.csv")

# ---- 그림 1 ----
F_tr, F_te = C.features(tr), C.features(te)
yA = C.label_cap(tr.RUL, C.CAP)
mid = xgb_fit(F_tr, yA, 0)
lo = xgb_fit(F_tr, yA, 0, objective="reg:quantileerror", quantile_alpha=0.1)
hi = xgb_fit(F_tr, yA, 0, objective="reg:quantileerror", quantile_alpha=0.9)
det = C.OnsetDetector().fit(tr, np.arange(1, 101), seed=0)
alarm = det.alarm_cycle(te)
picks = [int(pred.iloc[(pred.true_rul - t).abs().argmin()].unit) for t in (15, 60, 110)]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for ax, u in zip(axes, picks):
    g = te[te.unit == u]
    true = rul[u - 1] + (g.cycle.max() - g.cycle)
    X = F_te.loc[g.index]
    ax.fill_between(g.cycle, lo.predict(X), hi.predict(X), color="#9ecae1", alpha=0.5, label="80% 예측 구간")
    ax.plot(g.cycle, true, color="black", lw=1.5, label="실제 RUL")
    ax.plot(g.cycle, mid.predict(X), color="#d94801", lw=1.5, label="예측 (XGBoost, 상한 125)")
    ax.axhline(C.CAP, color="gray", ls=":", lw=1)
    if not np.isnan(alarm.get(u, np.nan)):
        ax.axvline(alarm[u], color="#6a51a3", ls="--", lw=1.2, label="열화 경보")
    ax.set_title(f"test 엔진 {u} (마지막 사이클 실제 RUL {rul[u - 1]})")
    ax.set_xlabel("사이클")
axes[0].set_ylabel("잔존수명 RUL (사이클)")
axes[0].legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "rul_trajectories.png", dpi=130)

# ---- 그림 2 ----
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.axhline(0, color="black", lw=0.8)
ax.scatter(pred.true_rul, pred.A_xgb - pred.true_rul, s=18, color="#d94801", label="A: 상한 125")
ax.scatter(pred.true_rul, pred.D_xgb - pred.true_rul, s=18, marker="x", color="#6a51a3", label="D: 경보 후 2단계")
ax.set_xlabel("실제 RUL (test 마지막 사이클)")
ax.set_ylabel("오차 = 예측 - 실제 (위쪽이 늦은 예측)")
ax.legend()
ax.set_title("test 100대: 실제 RUL에 따른 오차")
fig.tight_layout()
fig.savefig(FIG / "rul_errors.png", dpi=130)
print("saved", picks)
