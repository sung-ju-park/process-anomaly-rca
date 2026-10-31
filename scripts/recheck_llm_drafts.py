"""이전 평가(v1·v2)의 LLM 초안을 현재 검사 규칙으로 다시 검사한다. (LLM 없이 실행)

- v1 검증기는 '없는 숫자·변수·사례를 지어냈는가'만 봤다. 통과한 초안을 읽어 보니 변수 코드만 쓰거나
  입력 항목 이름을 그대로 쓰는 등 현장에서 읽을 수 없는 문장이 대부분이었다.
- v2에서 통과한 초안에도 '가장 먼저' 순서를 틀리거나 인과를 단정한 문장이 있었다.
현재 규칙이 이런 초안을 실제로 걸러내는지, 그리고 멀쩡한 초안은 통과시키는지 확인한다.
(v1·v2 초안에 있던 '의심원인'·'유사사례' 칸은 v3부터 코드가 채우므로 검사 대상이 아니다)
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llm_report import tidy, verify  # noqa: E402

out = {}
for ver in ("v1", "v2"):
    d = json.load(open(ROOT / "reports" / f"llm_report_samples_{ver}.json", encoding="utf-8"))
    rows = []
    for k, v in d.items():
        if v["source"] != "llm":          # 당시 검증을 통과해 LLM 문장이 채택된 초안만
            continue
        errs = verify(tidy(v["brief"]), v["facts"])
        rows.append({"case": k, "pass_now": not errs, "errors": errs})
    out[ver] = {"passed_then": len(rows), "pass_under_current_rules": sum(r["pass_now"] for r in rows),
                "drafts_failing_each_rule": dict(Counter(k for r in rows for k in {e.split(":")[0] for e in r["errors"]})
                                                 .most_common()),
                "detail": rows}
json.dump(out, open(ROOT / "reports" / "llm_report_recheck.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for ver, o in out.items():
    print(ver, json.dumps({k: v for k, v in o.items() if k != "detail"}, ensure_ascii=False))
