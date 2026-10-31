"""LLM 보고서 검증기 점검 (LLM 없이 실행).

1) 정상 문장(템플릿)은 모든 고장에서 통과해야 한다 → 검증기가 멀쩡한 문장을 막지 않는지
2) 일부러 틀린 내용을 한 가지씩 넣은 문장은 모두 걸러져야 한다 → 지어낸 내용을 잡는지
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from engine import Engine  # noqa: E402
from llm_report import build_facts, checklist_items, render_facts, template_brief, verify  # noqa: E402
from tep_data import TEST_ONSET, load  # noqa: E402
from tep_meta import COLS, label  # noqa: E402


def mutations(b: dict, facts: dict):
    """(이름, 틀린 문장) — 실제 LLM이 저지를 법한 실수 10종 (뒤 3종은 1차 평가에서 실제로 나온 유형)."""
    c = facts["원인후보"]
    used = {x["변수"] for x in c} | {x["변수"] for x in facts["전파경로_먼저벗어난순"]}
    facts_txt = render_facts(facts)
    other = next(v for v in COLS if v not in used and label(v) not in facts_txt)
    cases = {x["사례"] for x in facts["유사과거사례"]}
    fake_case = next(f"F{i:02d}" for i in range(1, 22) if f"F{i:02d}" not in cases)

    def m(fn):
        x = copy.deepcopy(b)
        fn(x)
        return x

    yield "지어낸 숫자", m(lambda x: x["근거"].append("경보 47분 전부터 온도가 8.3% 올랐습니다"))
    yield "입력에 없는 변수", m(lambda x: x["근거"].append(f"{label(other)}도 함께 변했습니다"))
    yield "입력에 없는 변수 코드", m(lambda x: x["근거"].append(f"{other} 값도 확인이 필요합니다"))
    yield "원인 바꿔치기", m(lambda x: x.__setitem__("의심원인", f"{c[-1]['이름']} 이상"))
    yield "없는 과거 사례", m(lambda x: x.__setitem__("유사사례", f"{fake_case}와 비슷합니다"))
    yield "단정 표현", m(lambda x: x.__setitem__("요약", x["요약"] + " 원인은 확실합니다."))
    yield "한자 섞임", m(lambda x: x.__setitem__("요약", x["요약"] + " 冷却水 확인."))
    c1 = c[0]
    yield "입력 항목 이름 노출", m(lambda x: x["근거"].append(f"{c1['이름']}의 이탈시작_경보기준_분 확인"))
    # σ 값을 %로 옮긴 실수. 우연히 같은 값이 기여도에 있으면 의미가 없으므로 기여도 값을 σ로 옮긴 실수로 대신 만든다
    pcts = {round(r["기여도비중_퍼센트"]) for r in c} | {round(s["유사도"] * 100) for s in facts["유사과거사례"]}
    sigs = {round(abs(r["정상대비_시그마"])) for r in c}
    if round(abs(c1["정상대비_시그마"])) not in pcts:
        swapped = f"{c1['이름']} 기여도 {abs(c1['정상대비_시그마'])}%"
    else:
        swapped = f"{c1['이름']} 정상 대비 {c1['기여도비중_퍼센트']}σ" if round(c1["기여도비중_퍼센트"]) not in sigs else None
    if swapped:
        yield "단위 바꿔치기", m(lambda x: x["근거"].append(swapped))
    yield "목록 밖 점검 항목", m(lambda x: x["우선점검"].__setitem__(0, f"{c1['이름']}의 경보 기준값 조정"))


E = Engine()
rows, clean_fail = [], []
for f in range(1, 22):
    X = load(f, "test").values
    ta = E.first_alarm(X, TEST_ONSET)
    if ta is None:
        continue
    facts = build_facts(E.investigate(X, ta))
    b = template_brief(facts)
    errs = verify(b, facts)
    if errs:
        clean_fail.append((f, errs))
    for name, bad in mutations(b, facts):
        rows.append({"fault": f, "mutation": name, "caught": bool(verify(bad, facts))})

by = {}
for r in rows:
    by.setdefault(r["mutation"], []).append(r["caught"])
summary = {
    "faults": len({r["fault"] for r in rows}),
    "clean_pass": f"{len({r['fault'] for r in rows}) - len(clean_fail)}/{len({r['fault'] for r in rows})}",
    "clean_fail_detail": clean_fail,
    "caught_by_type": {k: f"{sum(v)}/{len(v)}" for k, v in by.items()},
    "caught_total": f"{sum(r['caught'] for r in rows)}/{len(rows)}",
}
json.dump(summary, open(ROOT / "reports" / "llm_guard_check.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
