"""RUL 결과 그림 2개 (rul/figures/).

1) test 엔진 3대의 사이클별 예측 RUL(라벨 A XGBoost, 80% 구간)과 실제 RUL, 열화 경보 시점
2) test 100대 마지막 사이클: 실제 RUL 대 오차 (A 중심, D는 비교용)

그림 1은 공식 평가와 같은 설정(seed 0)으로 다시 학습해 test 엔진의 모든 사이클을 그린 것이다.
성적표 숫자는 rul/results/test.json(엔진별 마지막 사이클 1회 평가)만 쓴다.
"""
import logging
import sys
from pathlib import Path

import matplotlib
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmapss as C  # noqa: E402
from run_rul import UNITS, xgb_fit  # noqa: E402

plt.rcParams["font.family"] = ["Malgun Gothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
FIG = C.ROOT / "rul" / "figures"
NEAR = 30  # 고장 임박 기준 (사이클)


def trajectories(tr, te, rul_by_unit, pred):
    F_tr, F_te = C.features(tr), C.features(te)
    yA = C.label_cap(tr.RUL, C.CAP)
    mid = xgb_fit(F_tr, yA, 0)
    lo = xgb_fit(F_tr, yA, 0, objective="reg:quantileerror", quantile_alpha=0.1)
    hi = xgb_fit(F_tr, yA, 0, objective="reg:quantileerror", quantile_alpha=0.9)
    alarm = C.OnsetDetector().fit(tr, UNITS, seed=0).alarm_cycle(te)
    picks = [int(pred.iloc[(pred.true_rul - t).abs().argmin()].unit) for t in (15, 60, 110)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, u in zip(axes, picks):
        g = te[te.unit == u]
        true = rul_by_unit[u] + (g.cycle.max() - g.cycle)
        X = F_te.loc[g.index]
        pm = mid.predict(X)
        ax.fill_between(g.cycle, lo.predict(X), hi.predict(X), color="#9ecae1", alpha=0.5, label="80% 예측 구간")
        ax.plot(g.cycle, true, color="black", lw=1.5, label="실제 RUL")
        ax.plot(g.cycle, np.minimum(true, C.CAP), color="gray", ls="--", lw=1, label="학습 목표 (125에서 자름)")
        ax.plot(g.cycle, pm, color="#d94801", lw=1.5, label="예측 (XGBoost, 상한 125)")
        ax.axhline(NEAR, color="#cb181d", ls=":", lw=1, label=f"고장 임박 {NEAR}사이클")
        if not np.isnan(alarm.get(u, np.nan)):
            ax.axvline(alarm[u], color="#6a51a3", ls="--", lw=1.2, label="열화 경보")
        x_last = g.cycle.max()
        ax.scatter([x_last], [rul_by_unit[u]], color="black", s=40, zorder=5)
        ax.scatter([x_last], [pm[-1]], color="#d94801", s=40, zorder=5)
        ax.set_title(f"엔진 {u}: 평가 시점 실제 {rul_by_unit[u]} / 예측 {pm[-1]:.0f}", fontsize=11)
        ax.set_xlabel("관측한 비행 사이클")
    axes[0].set_ylabel("잔존수명 RUL (비행 사이클)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, fontsize=9, frameon=False)
    fig.text(0.5, 0.005, "점 = 채점하는 마지막 사이클. 남은 수명이 125를 넘는 구간은 125로 예측하도록 설계했다 (가운데·오른쪽).",
             ha="center", fontsize=9, color="#444")
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    fig.savefig(FIG / "rul_trajectories.png", dpi=130)
    return picks


def errors(pred):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhspan(0, 50, color="#fee0d2", alpha=0.6, zorder=0)
    ax.text(100, 44, "수명 과대추정 (정비 지연 위험)", ha="center", va="center", fontsize=9, color="#a50f15")
    ax.axvspan(0, NEAR, color="#e5f5e0", alpha=0.7, zorder=0)
    near = pred.true_rul <= NEAR
    r_near = C.rmse(pred.true_rul[near], pred.A_xgb[near])
    ax.text(NEAR / 2, -75, f"고장 임박\nA의 RMSE {r_near:.1f}", ha="center", fontsize=9, color="#006d2c")
    ax.axvline(C.CAP, color="gray", ls=":", lw=1)
    ax.text(C.CAP + 1, 22, "125 상한 때문에\n이 오른쪽은\n짧게 예측", ha="left", fontsize=8, color="#555")
    ax.axhline(0, color="black", lw=0.8)
    ax.scatter(pred.true_rul, pred.D_xgb - pred.true_rul, s=10, marker="x", color="#bdbdbd", label="D: 경보 후 2단계 (실패 실험)")
    ax.scatter(pred.true_rul, pred.A_xgb - pred.true_rul, s=22, color="#d94801", label="A: XGBoost, 상한 125")
    ax.set_ylim(-95, 50)
    ax.set_xlim(0, 150)
    ax.set_xlabel("실제 잔존수명 (평가 시점, 비행 사이클)")
    ax.set_ylabel("오차 = 예측 - 실제")
    ax.set_title("test 엔진 100대: 고장이 가까울수록 오차가 작다")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "rul_errors.png", dpi=130)


def main():
    FIG.mkdir(exist_ok=True)
    tr, te, rul = C.load()
    rul_by_unit = dict(zip(sorted(te.unit.unique()), rul))
    pred = pd.read_csv(C.ROOT / "rul" / "results" / "test_predictions.csv")
    picks = trajectories(tr, te, rul_by_unit, pred)
    errors(pred)
    print("saved", picks)


if __name__ == "__main__":
    main()
