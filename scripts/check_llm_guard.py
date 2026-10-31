"""LLM 보고서 검증기 점검 (LLM 없이 실행).

1) 정상 문장(템플릿)은 모든 고장에서 통과해야 한다 → 검증기가 멀쩡한 문장을 막지 않는지
2) 일부러 틀린 내용을 한 가지씩 넣은 문장은 모두 걸러져야 한다 → 틀린 내용을 잡는지
   틀린 내용 14종 중 '입력 항목 이름 노출'부터는 1·2차 평가에서 실제 LLM이 저지른 유형이다.
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from engine import Engine  # noqa: E402
from llm_report import build_facts, render_facts, template_brief, verify  # noqa: E402
from tep_data import TEST_ONSET, load  # noqa: E402
from tep_meta import COLS, UNIT_OF, UNITS, label  # noqa: E402


def mutations(b: dict, facts: dict):
    c = facts["원인후보"]
    c1 = c[0]
    path = facts["전파경로_먼저벗어난순"]
    src = render_facts(facts)
    other = next(v for v in COLS if label(v) not in src)
    cases = {x["사례"] for x in facts["유사과거사례"]}
    fake_case = next(f"F{i:02d}" for i in range(1, 22) if f"F{i:02d}" not in src)
    wrong_unit = next(u for u in UNITS if u != c1["공정단위"] and u in src) if any(
        u != c1["공정단위"] and u in src for u in UNITS) else None

    def m(fn):
        x = copy.deepcopy(b)
        fn(x)
        return x

    def add(s):
        return m(lambda x: x["근거"].append(s))

    yield "지어낸 숫자", add("경보 47분 전부터 온도가 8.3% 올랐습니다")
    yield "입력에 없는 변수", add(f"{label(other)}도 함께 변했습니다")
    yield "변수 코드 노출", add(f"{other} 값도 확인이 필요합니다")
    yield "없는 과거 사례", add(f"{fake_case} 사례와도 비슷합니다")
    yield "단정 표현", m(lambda x: x.__setitem__("요약", x["요약"] + " 원인은 확실합니다."))
    yield "한자 섞임", m(lambda x: x.__setitem__("요약", x["요약"] + " 冷却水 확인."))
    yield "입력 항목 이름 노출", add(f"{c1['이름']}의 이탈시작_경보기준_분 확인")
    pcts = {round(r["기여도비중_퍼센트"]) for r in c} | {round(s["유사도"] * 100) for s in facts["유사과거사례"]}
    if round(abs(c1["정상대비_시그마"])) not in pcts:
        yield "단위 바꿔치기", add(f"{c1['이름']} 기여도 {abs(c1['정상대비_시그마'])}%")
    yield "목록 밖 점검 항목", m(lambda x: x["우선점검"].__setitem__(0, f"{c1['이름']}의 경보 기준값 조정"))
    yield "원인 대상 바꿔치기", add(f"{c[-1]['이름']}이 원인으로 의심됨")
    if path:
        last = path[-1]["이름"] if len(path) > 1 else c[-1]["이름"]
        if last != path[0]["이름"]:
            yield "순서 뒤바꿈", add(f"{last}이 가장 먼저 벗어남")
    if wrong_unit:
        yield "공정 단위 바꿔치기", add(f"{c1['이름']} ({wrong_unit})의 변화가 큼")
    yield "인과 단정", add(f"{c1['이름']} 때문에 경보가 울림")
    small = min(c, key=lambda r: r["기여도비중_퍼센트"])
    if small["이름"] != max(c, key=lambda r: r["기여도비중_퍼센트"])["이름"]:
        yield "최상급 오류", add(f"{small['이름']}은 기여도가 가장 큰 후보")


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
n = len({r["fault"] for r in rows})
summary = {
    "faults": n,
    "clean_pass": f"{n - len(clean_fail)}/{n}",
    "clean_fail_detail": clean_fail,
    "caught_by_type": {k: f"{sum(v)}/{len(v)}" for k, v in by.items()},
    "caught_total": f"{sum(r['caught'] for r in rows)}/{len(rows)}",
}
json.dump(summary, open(ROOT / "reports" / "llm_guard_check.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
