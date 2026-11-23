"""C-MAPSS FD001 RUL 실험.

    python rul/run_rul.py cv      학습 데이터만으로 엔진 단위 교차검증 (몇 번 돌려도 됨)
    python rul/run_rul.py test    공식 test 1회 평가 (rul/results/test_done.json이 있으면 멈춤)

기준: rul/protocol.json (test 실행 전 커밋)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmapss as C  # noqa: E402

OUT = C.ROOT / "rul" / "results"
OUT.mkdir(parents=True, exist_ok=True)
P = C.PROTOCOL
XGB = {k: v for k, v in P["models"]["xgboost"].items() if k != "seeds"}
SEEDS = P["models"]["xgboost"]["seeds"]
CAPS = [100, 115, 125, 135, 150, None]


def check_data(tr, te, rul):
    exp = P["data"]["expected_rows"]
    assert len(tr) == exp["train"] and len(te) == exp["test"] and len(rul) == exp["rul"]
    assert tr.unit.nunique() == exp["train_units"] and te.unit.nunique() == exp["test_units"]
    for f, h in P["data"]["sha256"].items():
        assert C.sha256(C.DATA / f) == h, f
    drop = C.low_variance_sensors(tr)
    assert drop == ["s1", "s5", "s6", "s10", "s16", "s18", "s19"], drop
    assert sorted(set(C.COLS[5:]) - set(drop), key=lambda s: int(s[1:])) == C.SENSORS


# ---------------- 모델 ----------------
def xgb_fit(X, y, seed, **kw):
    m = xgb.XGBRegressor(**XGB, random_state=seed, n_jobs=4, **kw)
    return m.fit(X, y)


def ridge_fit(X, y):
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)


def predict_all(tr, F_tr, F_ev, ev_df, ev_rows, units_tr, seed, cap_b):
    """학습 엔진(units_tr)으로 맞추고 평가 행(ev_rows)을 예측. 모든 방법이 모든 평가 행에 숫자를 낸다."""
    m = tr.unit.isin(units_tr)
    Xtr, Xev = F_tr[m], F_ev.loc[ev_rows]
    rul_tr = tr.RUL[m]
    pred = {}
    yA = C.label_cap(rul_tr, C.CAP)
    pred["A_const"] = np.full(len(ev_rows), yA.mean())
    pred["A_ridge"] = ridge_fit(Xtr, yA).predict(Xev)
    pred["A_xgb"] = xgb_fit(Xtr, yA, seed).predict(Xev)
    pred["B_xgb"] = xgb_fit(Xtr, C.label_cap(rul_tr, cap_b), seed).predict(Xev)

    det = C.OnsetDetector().fit(tr[m], units_tr, seed=0)
    alarm = det.alarm_cycle(tr[m])
    knee = tr[m].groupby("unit").life.first() - alarm
    F = float(np.nanmedian(knee))
    yC = C.label_onset(tr[m], alarm, F)
    pred["C_xgb"] = xgb_fit(Xtr, yC, seed).predict(Xev)

    fl_tr = det.alarm_flags(tr[m])
    fl_ev = det.alarm_flags(ev_df).loc[ev_rows].to_numpy()
    stage2 = xgb_fit(Xtr[fl_tr.to_numpy()], rul_tr[fl_tr.to_numpy()], seed)
    pD = np.full(len(ev_rows), F)
    if fl_ev.any():
        pD[fl_ev] = stage2.predict(Xev[fl_ev])
    pred["D_xgb"] = pD
    info = {"F": F, "train_alarm_missing": int(alarm.isna().sum()),
            "train_false_alarm_first30": int((alarm <= C.HEALTHY_CYCLES).sum()),
            "train_rul_at_alarm_q": np.nanpercentile(knee, [0, 25, 50, 75, 100]).round(1).tolist(),
            "eval_alarmed": int(fl_ev.sum())}
    return pred, fl_ev, info


# ---------------- 교차검증 (학습 데이터만) ----------------
def run_cv(tr, F_tr):
    units = np.arange(1, 101)
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(units, groups=units))
    # B의 상한 고르기: XGBoost seed 0, 상한별 RMSE (정답은 상한 없는 실제 RUL)
    cap_rmse = {str(c): [] for c in CAPS}
    rows_y, rows_p = [], {}
    for k, (itr, iva) in enumerate(folds):
        u_tr, u_va = units[itr], units[iva]
        ev = C.truncation_points(tr, u_va, k=5, seed=0)
        m = tr.unit.isin(u_tr)
        for c in CAPS:
            p = xgb_fit(F_tr[m], C.label_cap(tr.RUL[m], c), 0).predict(F_tr.loc[ev])
            cap_rmse[str(c)].append(C.rmse(tr.RUL.loc[ev], p))
    cap_mean = {c: float(np.mean(v)) for c, v in cap_rmse.items()}
    best = min(cap_mean, key=cap_mean.get)
    cap_b = None if best == "None" else int(best)

    infos = []
    for k, (itr, iva) in enumerate(folds):
        u_tr, u_va = units[itr], units[iva]
        ev = C.truncation_points(tr, u_va, k=5, seed=0)
        pred, fl, info = predict_all(tr, F_tr, F_tr, tr[tr.unit.isin(u_va)], ev, u_tr, 0, cap_b)
        infos.append(info)
        rows_y.append(tr.RUL.loc[ev].to_numpy())
        for name, p in pred.items():
            rows_p.setdefault(name, []).append(p)
    y = np.concatenate(rows_y)
    cv = {n: C.summary(y, np.concatenate(p)) for n, p in rows_p.items()}

    # 누수 확인: 같은 XGBoost(상한 125)를 행 무작위 KFold vs 엔진 단위 GroupKFold, 둘 다 held-out 모든 행 평가
    yA = C.label_cap(tr.RUL, C.CAP).to_numpy()
    leak = {}
    for name, splitter in [("row_random_kfold", KFold(5, shuffle=True, random_state=0).split(F_tr)),
                           ("engine_groupkfold", GroupKFold(5).split(F_tr, groups=tr.unit))]:
        errs = []
        for a, b in splitter:
            p = xgb_fit(F_tr.iloc[a], yA[a], 0).predict(F_tr.iloc[b])
            errs.append((p - yA[b]) ** 2)
        leak[name] = float(np.sqrt(np.mean(np.concatenate(errs))))
    res = {"cap_cv_rmse": cap_mean, "cap_b": cap_b, "cv": cv, "onset_folds": infos,
           "leak_check_rmse_vs_cap125_all_rows": leak, "n_eval_points": int(len(y))}
    (OUT / "cv.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


# ---------------- 물리 근거 예상 확인 (H5, 학습 데이터만) ----------------
def first_departure(tr):
    sys.path.insert(0, str(C.ROOT / "src"))
    from rca import onset_times  # 기존 RCA의 '먼저 벗어난 순서'와 같은 규칙
    base = tr[tr.cycle <= C.HEALTHY_CYCLES]
    mu = base.groupby("unit")[C.SENSORS].mean()
    sd = (base[C.SENSORS] - mu.loc[base.unit].to_numpy()).std()  # 엔진 안 정상 구간 흔들림
    expected = ["s3", "s4", "s7", "s11", "s12"]
    firsts, in_exp, none = [], 0, 0
    for u, g in tr.groupby("unit"):
        on = onset_times(((g[C.SENSORS] - mu.loc[u]) / sd).to_numpy())
        if np.all(np.isnan(on)):
            none += 1
            continue
        first = [C.SENSORS[j] for j in np.flatnonzero(on == np.nanmin(on))]
        firsts += first
        in_exp += any(s in expected for s in first)
    return {"expected": expected, "first_sensor_counts": pd.Series(firsts).value_counts().to_dict(),
            "engines_first_in_expected": int(in_exp), "engines_no_departure": none,
            "engines": int(tr.unit.nunique())}


# ---------------- 공식 test 1회 ----------------
def run_test(tr, te, rul, F_tr, cap_b):
    done = OUT / "test_done.json"
    if done.exists() and "--force" not in sys.argv:
        raise SystemExit("공식 test는 이미 1회 실행됨 (rul/results/test_done.json). 다시 돌리려면 --force, 그리고 decision_log에 이유를 적을 것.")
    F_te = C.features(te)
    ev = C.last_rows(te)
    units = np.arange(1, 101)
    seeds = {}
    for s in SEEDS:
        pred, fl, info = predict_all(tr, F_tr, F_te, te, ev, units, s, cap_b)
        seeds[s] = pred
    names = list(seeds[SEEDS[0]])
    res = {"cap_b": cap_b, "onset": info, "methods": {}}
    for n in names:
        per = [C.summary(rul, seeds[s][n]) for s in SEEDS]
        res["methods"][n] = {
            "rmse_mean": float(np.mean([r["rmse"] for r in per])), "rmse_sd": float(np.std([r["rmse"] for r in per])),
            "rmse_vs_cap125_mean": float(np.mean([r["rmse_vs_cap125"] for r in per])),
            "nasa_mean": float(np.mean([r["nasa"] for r in per])), "nasa_sd": float(np.std([r["nasa"] for r in per])),
            "late_mean": float(np.mean([r["late"] for r in per])), "seed0": per[0]}
    # D: 경보 있음/없음 엔진별 (seed 0)
    fl = np.asarray(fl)
    res["D_split_seed0"] = {
        "alarmed": {"n": int(fl.sum()), "rmse_D": C.rmse(rul[fl], seeds[0]["D_xgb"][fl]), "rmse_A": C.rmse(rul[fl], seeds[0]["A_xgb"][fl])},
        "not_alarmed": {"n": int((~fl).sum()), "rmse_D": C.rmse(rul[~fl], seeds[0]["D_xgb"][~fl]), "rmse_A": C.rmse(rul[~fl], seeds[0]["A_xgb"][~fl])}}
    # 80% 구간 (라벨 A, seed 0)
    yA = C.label_cap(tr.RUL, C.CAP)
    lo = xgb_fit(F_tr, yA, 0, objective="reg:quantileerror", quantile_alpha=0.1).predict(F_te.loc[ev])
    hi = xgb_fit(F_tr, yA, 0, objective="reg:quantileerror", quantile_alpha=0.9).predict(F_te.loc[ev])
    inside = (rul >= lo) & (rul <= hi)
    m125 = rul <= C.CAP
    res["interval80"] = {"coverage_all": float(inside.mean()), "coverage_rul_le125": float(inside[m125].mean()),
                         "n_rul_le125": int(m125.sum()), "mean_width": float(np.mean(hi - lo))}
    pred_df = pd.DataFrame({"unit": te.loc[ev, "unit"].to_numpy(), "last_cycle": te.loc[ev, "cycle"].to_numpy(),
                            "true_rul": rul, "alarmed": fl, "lo80": lo, "hi80": hi,
                            **{n: seeds[0][n] for n in names}})
    pred_df.to_csv(OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")
    (OUT / "test.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    done.write_text(json.dumps({"ran": True}), encoding="utf-8")
    return res, F_te


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "cv"
    tr, te, rul = C.load()
    check_data(tr, te, rul)
    F_tr = C.features(tr)
    if stage == "cv":
        res = run_cv(tr, F_tr)
        res["physics_H5"] = first_departure(tr)
        (OUT / "cv.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k: res[k] for k in ["cap_cv_rmse", "cap_b", "leak_check_rmse_vs_cap125_all_rows", "physics_H5"]}, ensure_ascii=False, indent=1))
        for n, r in res["cv"].items():
            print(f"{n:8s} RMSE {r['rmse']:6.2f}  NASA {r['nasa']:9.0f}  late {r['late']:.2f}  {r['by_rul']}")
    elif stage == "test":
        cap_b = json.loads((OUT / "cv.json").read_text(encoding="utf-8"))["cap_b"]
        res, _ = run_test(tr, te, rul, F_tr, cap_b)
        print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
