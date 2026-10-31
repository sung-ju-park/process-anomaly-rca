"""로컬 LLM 점검 보고서 평가 (Ollama 필요).

테스트 고장 21종 각각의 첫 경보에 대해 보고서 초안을 만들고 기록한다.
- 첫 시도 통과율, 재시도 후 통과율, 템플릿 대체 횟수
- 걸러진 위반 유형 (지어낸 숫자, 없는 변수, 원인 바꿔치기, 단정 표현 등)
- 1건당 소요 시간
샘플 리포트(고장 4·6·12)는 reports/rca_reports/llm/ 에 저장한다.
"""
import json
import platform
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import llm_report  # noqa: E402
from engine import Engine  # noqa: E402
from report import render  # noqa: E402
from tep_data import TEST_ONSET, load  # noqa: E402
from tep_meta import FAULTS  # noqa: E402

OUT = ROOT / "reports"
ok, msg = llm_report.available()
if not ok:
    sys.exit(f"[중단] {msg}")

E = Engine()
(OUT / "rca_reports" / "llm").mkdir(parents=True, exist_ok=True)
rows, samples = [], {}
for f in range(1, 22):
    X = load(f, "test").values
    ta = E.first_alarm(X, TEST_ONSET)
    if ta is None:
        continue
    F = E.investigate(X, ta, f"테스트 시나리오 {f:02d} — {FAULTS[f][0]} ({FAULTS[f][1]})")
    b = llm_report.make_brief(F)
    first = b.violations[0] if b.violations else []
    rows.append({"fault": f, "source": b.source, "attempts": b.attempts,
                 "first_try_pass": b.attempts >= 1 and not first,
                 "seconds": b.seconds,
                 "violations": " | ".join("; ".join(v) for v in b.violations if v)})
    samples[f"F{f:02d}"] = {"facts": llm_report.build_facts(F), "llm_input": llm_report.render_facts(
                            llm_report.build_facts(F)), "brief": b.content, "source": b.source,
                            "violations": b.violations, "drafts": b.drafts}
    if f in (4, 6, 12):
        (OUT / "rca_reports" / "llm" / f"rca_report_fault{f:02d}_llm.html").write_text(
            render(F, X, E.mon.sc, brief=b), encoding="utf-8")
    print(f"F{f:02d} {b.source:8s} 시도 {b.attempts} {b.seconds:6.1f}s {rows[-1]['violations'][:120]}", flush=True)

df = pd.DataFrame(rows)
# 위반 유형: 시도마다 같은 규칙은 한 번만 센다
kinds = Counter(k for s in samples.values() for v in s["violations"] if v for k in {e.split(":")[0] for e in v})
n = len(df)
summary = {
    "model": llm_report.MODEL, "run_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "machine": platform.processor() or platform.machine(),
    "n_alarms": n,
    "first_try_pass": int(df.first_try_pass.sum()),
    "pass_after_retry": int((df.source == "llm").sum()),
    "fallback_template": int((df.source == "template").sum()),
    "attempts_failing_each_rule": dict(kinds.most_common()),
    "median_seconds": float(df.seconds.median()),
}
df.to_csv(OUT / "llm_report_eval.csv", index=False, encoding="utf-8-sig")
json.dump(summary, open(OUT / "llm_report_eval.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(samples, open(OUT / "llm_report_samples.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
