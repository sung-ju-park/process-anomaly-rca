"""1차 평가(v1)에서 '검증 통과'였던 초안을 v2 검사 규칙으로 다시 검사한다.

v1 검증기는 '없는 숫자·변수·사례를 지어냈는가'만 봤다. 통과한 초안을 읽어 보니
변수 코드만 쓰거나, 입력 항목 이름을 그대로 쓰거나, 목록에 없는 점검 항목을 지어낸 경우가 많았다.
v2에서 추가한 규칙이 이 문제를 실제로 잡는지 v1 초안으로 확인한다. (LLM 없이 실행)
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llm_report import verify  # noqa: E402

d = json.load(open(ROOT / "reports" / "llm_report_samples_v1.json", encoding="utf-8"))
rows = []
for k, v in d.items():
    if v["source"] != "llm":
        continue
    errs = verify(v["brief"], v["facts"])
    rows.append({"case": k, "v2_pass": not errs, "errors": errs})
out = {"v1_passed": len(rows), "still_pass_under_v2": sum(r["v2_pass"] for r in rows),
       "drafts_failing_each_rule": dict(Counter(k for r in rows for k in {e.split(":")[0] for e in r["errors"]})
                                        .most_common()),
       "detail": rows}
json.dump(out, open(ROOT / "reports" / "llm_report_v1_recheck.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "detail"}, ensure_ascii=False, indent=1))
for r in rows:
    print(r["case"], r["v2_pass"], r["errors"][:3])
