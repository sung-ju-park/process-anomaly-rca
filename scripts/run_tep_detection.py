"""TEP 이상 탐지 비교.

지표 (테스트 파일 21개 + 정상 hold-out 기준)
- 탐지율   : 고장 이후 샘플 중 기준선을 넘은 비율 (문헌 표준 방식, 연속 조건 없음)
- 오경보율 : 정상 hold-out 480개 + 각 테스트 파일의 고장 전 160개에서 경보가 울린 비율
- 탐지 지연: '3회 연속 초과' 경보 규칙으로 고장 시작 후 처음 경보까지 걸린 분
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detect import MODELS, alarms, first_alarm  # noqa: E402
from tep_data import SAMPLE_MIN, TEST_ONSET, load, normal_sets  # noqa: E402
from tep_meta import FAULTS  # noqa: E402

PERSIST = 3
OUT = ROOT / "reports"

fit, calib, hold = normal_sets()
tests = {f: load(f, "test") for f in range(1, 22)}

rows, far_rows, models = [], [], {}
for name, make in MODELS.items():
    t0 = time.time()
    m = make().fit(fit.values).calibrate(calib.values)
    models[name] = m
    # 오경보: hold-out + 각 테스트 파일 고장 전 구간
    s_hold = m.score(hold.values)
    pre = [m.score(tests[f].values)[:TEST_ONSET] for f in tests]
    s_pre = np.concatenate([s_hold] + pre)
    far_rows.append({
        "model": name,
        "far_sample": float(np.mean(s_pre > 1)),
        "far_alarm3": float(np.mean(np.concatenate([alarms(s_hold, PERSIST)] + [alarms(p, PERSIST) for p in pre]))),
        "fit_sec": round(time.time() - t0, 1),
    })
    for f, df in tests.items():
        s = m.score(df.values)
        post = s[TEST_ONSET:]
        fa = first_alarm(alarms(s, PERSIST), TEST_ONSET)
        rows.append({
            "model": name, "fault": f, "desc": FAULTS[f][0], "type": FAULTS[f][1],
            "detect_rate": float(np.mean(post > 1)),
            "delay_min": None if fa is None else (fa - TEST_ONSET) * SAMPLE_MIN,
        })
    print(name, "done", far_rows[-1])

det = pd.DataFrame(rows)
far = pd.DataFrame(far_rows)
det.to_csv(OUT / "tep_detection_by_fault.csv", index=False, encoding="utf-8-sig")
far.to_csv(OUT / "tep_false_alarm.csv", index=False, encoding="utf-8-sig")

pivot = det.pivot(index="fault", columns="model", values="detect_rate")[list(MODELS)]
delay = det.pivot(index="fault", columns="model", values="delay_min")[list(MODELS)]
print((pivot * 100).round(1))
print(delay)
summary = {
    m: {
        "mean_detect_rate_all": float(pivot[m].mean()),
        "mean_detect_rate_excl_3_9_15": float(pivot[m].drop([3, 9, 15]).mean()),
        "median_delay_min": float(delay[m].dropna().median()),
        "faults_detected_alarm3": int(delay[m].notna().sum()),
        **far.set_index("model").loc[m].to_dict(),
    }
    for m in MODELS
}
json.dump(summary, open(OUT / "tep_detection_summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
