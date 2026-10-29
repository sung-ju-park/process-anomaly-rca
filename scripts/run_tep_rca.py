"""TEP RCA 채점.

1) 개발(dev): 학습용 고장 파일(dXX.dat)로 방법이 정상 동작하는지와 경보 구간 길이 민감도만 확인
2) 테스트: 테스트 파일(dXX_te.dat)로 최종 성적 — 설정값은 dev 이후 바꾸지 않음
정답은 reports/rca_ground_truth.json (결과 확인 전 git 커밋으로 고정)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detect import PCAMonitor, alarms, first_alarm  # noqa: E402
from rca import WINDOW, analyze, grade, propagation  # noqa: E402
from tep_data import SAMPLE_MIN, load, normal_sets, onset  # noqa: E402

OUT = ROOT / "reports"
GT = json.load(open(OUT / "rca_ground_truth.json"))["graded"]
PERSIST = 3

fit, calib, hold = normal_sets()
mon = PCAMonitor(lags=2).fit(fit.values).calibrate(calib.values)


def run(split: str, window: int = WINDOW, with_shap: bool = True):
    rows, details = [], {}
    for f_str, g in GT.items():
        f = int(f_str)
        X = load(f, split).values
        t0 = onset(split)
        t_alarm = first_alarm(alarms(mon.score(X), PERSIST), t0)
        if t_alarm is None:
            rows.append({"fault": f, "detected": False})
            continue
        res = analyze(X, t_alarm, mon, fit.values, SAMPLE_MIN, with_shap=with_shap, window=window)
        d = {"fault": f, "detected": True, "delay_min": (t_alarm - t0) * SAMPLE_MIN}
        for m, s in res.scores.items():
            r = grade(s, g["root_vars"], g["root_unit"])
            d.update({f"{m}|{k}": v for k, v in r.items()})
        rows.append(d)
        details[f] = res
    return pd.DataFrame(rows), details


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    methods = sorted({c.split("|")[0] for c in df.columns if "|" in c},
                     key=["zscore", "spe", "rbc", "shap", "onset", "rbc+onset", "shap+onset"].index)
    out = []
    n = len(df)  # 미탐지 고장은 0점으로 계산 (분모에 포함)
    for m in methods:
        out.append({
            "method": m,
            "hit@1": df[f"{m}|hit1"].fillna(False).astype(bool).sum() / n,
            "hit@3": df[f"{m}|hit3"].fillna(False).astype(bool).sum() / n,
            "MRR": df[f"{m}|rr"].fillna(0).sum() / n,
            "unit@1": df[f"{m}|unit1"].fillna(False).astype(bool).sum() / n,
        })
    return pd.DataFrame(out).set_index("method")


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if split == "dev":
        for w in (10, 20, 40):
            df, _ = run("train", window=w, with_shap=(w == WINDOW))
            print(f"\n[dev] window={w} ({w * SAMPLE_MIN}분)")
            print(summarize(df).round(3))
    else:
        df, details = run("test")
        summ = summarize(df)
        print(summ.round(3))
        cols = ["fault", "delay_min"] + [c for c in df.columns if c.endswith("|best_rank")]
        print(df[cols].to_string())
        df.to_csv(OUT / "tep_rca_by_fault.csv", index=False, encoding="utf-8-sig")
        summ.to_csv(OUT / "tep_rca_summary.csv", encoding="utf-8-sig")
        prop = {f: propagation(r.onset_min).to_dict("records") for f, r in details.items()}
        json.dump(prop, open(OUT / "tep_propagation.json", "w"), ensure_ascii=False, indent=1)
