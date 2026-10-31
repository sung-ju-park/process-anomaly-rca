"""로컬 LLM 점검 보고서 초안 (추가 기능, 2026.10).

원칙
- 분석은 기존 코드가 한다. LLM은 계산된 결과를 현장 담당자가 읽을 문장으로 옮기기만 한다.
- 정답이 하나로 정해진 칸(의심 원인 = 1순위 후보, 유사 사례 = 1위 사례)은 코드가 채운다.
  LLM은 요약·근거 문장을 쓰고, 점검 항목 목록에서 우선 점검을 고른다.
- LLM이 쓴 문장은 코드로 검사한다. 하나라도 걸리면 한 번 고쳐 쓰게 하고, 그래도 틀리면 버리고 템플릿 문장을 쓴다.
- 외부 API 없이 로컬 Ollama로만 동작한다. Ollama가 없으면 템플릿 문장으로 대체되고, 기존 분석 결과는 바뀌지 않는다.

버전 기록 (평가 결과는 reports/llm_report_eval_v*.json, 초안 원문은 llm_report_samples_v*.json)
- v1: 분석 결과 JSON을 그대로 넘김. 검증 통과 18/21이었지만 읽어 보니 변수 코드(XMEAS30)·입력 키 이름
  (이탈시작_경보기준_분)을 그대로 쓰고 숫자 뜻을 잘못 옮긴 문장이 많았다 → 검증 규칙이 너무 느슨했음.
- v2: 코드 없는 문장형 입력 + 작성 예시, 검사 규칙 추가. 읽기는 좋아졌지만 통과 12/21.
  실패의 대부분이 '의심원인' 칸(1순위 대신 가장 먼저 벗어난 변수를 쓰거나, 예시의 공정 단위를 베낌)이었고,
  통과한 초안에도 공정 단위를 틀리게 붙이거나 '가장 먼저' 순서를 틀린 문장이 있었다.
- v3: 정답이 정해진 칸은 코드가 채우고, LLM은 문장만 쓴다. 순서·공정 단위·시점·인과 단정 검사를 추가.

환경 변수: OLLAMA_HOST (기본 http://localhost:11434), RCA_LLM_MODEL (기본 qwen2.5:3b)
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import numpy as np

from engine import CHECKLIST, Finding
from tep_meta import COLS, UNIT_OF, UNITS, label

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("RCA_LLM_MODEL", "qwen2.5:3b")
FIELDS = ["요약", "의심원인", "근거", "유사사례", "우선점검"]     # 보고서에 보이는 칸
LLM_FIELDS = ["요약", "근거", "우선점검"]                       # LLM이 쓰는 칸

SCHEMA = {
    "type": "object",
    "properties": {
        "요약": {"type": "string"},
        "근거": {"type": "array", "items": {"type": "string"}},
        "우선점검": {"type": "array", "items": {"type": "string"}},
    },
    "required": LLM_FIELDS,
}

SYSTEM = """너는 화학 공정 운전팀의 점검 보고서 초안을 쓰는 보조자다. 분석은 이미 끝났고, 너는 분석 결과를 현장 담당자가 바로 읽을 수 있는 짧은 한국어 문장으로 옮긴다.
규칙
1. 분석 결과에 적힌 숫자, 변수 이름, 사례 번호만 쓴다. 없는 숫자·변수·설비를 만들지 않는다.
2. 변수는 적힌 이름 그대로 쓴다. 영문 코드나 밑줄(_)이 들어간 이름을 쓰지 않는다.
3. '원인 후보 순위'와 '먼저 벗어난 순서'는 서로 다르다. 가장 먼저 벗어난 변수는 '먼저 벗어난 순서'의 맨 앞 변수다.
4. 원인은 '1순위 후보' 또는 '의심'으로만 쓴다. '때문에', '인해', '확실' 같은 단정·인과 표현을 쓰지 않는다.
5. 숫자는 적힌 단위 그대로 쓴다. 기여도는 %, 정상 대비 변화는 σ, 시점은 '경보 N분 전/후'로 쓴다.
6. 공정 단위를 쓸 때는 그 변수 옆 [ ] 안에 적힌 단위를 그대로 쓴다.
7. 변화가 '작음'인 변수를 크게 벗어났다고 쓰지 않는다. 먼저 벗어난 변수가 없으면 오경보일 수 있다고 쓴다.
8. 우선점검은 '점검 항목 목록'에서 문장을 바꾸지 말고 그대로 1~3개 고른다.
9. 요약은 2문장 이내, 근거는 3개 이내. 한국어로만 쓰고 한자를 쓰지 않는다.
10. 출력은 요약, 근거, 우선점검 키를 가진 JSON 하나만 쓴다."""

# 예시는 '1순위'와 '가장 먼저 벗어난 변수'가 다른 경우로 골라, 둘을 구분해 쓰는 법을 보여 준다.
EXAMPLE_IN = """경보 시각: 운전 10시간 21분
원인 후보 (분석 코드가 정한 순위 — 기여도와 먼저 벗어난 시점을 함께 본 것)
1순위 분리기 압력 [분리기] — 경보 3분 전 이탈 시작, 기여도 41.5%, 정상 대비 +6.4σ (변화 큼)
2순위 압축기 일 [압축기·재순환] — 경보 9분 전 이탈 시작, 기여도 12.3%, 정상 대비 -2.1σ (변화 보통)
3순위 분리기 액위 [분리기] — 경보 구간에서 뚜렷한 이탈 없음, 기여도 4.4%, 정상 대비 +0.6σ (변화 작음)
먼저 벗어난 순서: 압축기 일(경보 9분 전) → 분리기 압력(경보 3분 전)
가장 비슷한 과거 사례: F05 응축기 냉각수 입구 온도 (계단), 유사도 0.88
점검 항목 목록
- 분리기 온도·액위 제어 상태 확인
- 분리기 하부 배출 밸브 확인
- 압축기 부하(일) 추이 확인
- 재순환 밸브 개도 확인"""

EXAMPLE_OUT = {
    "요약": "운전 10시간 21분에 경보가 울렸습니다. 압축기 일이 가장 먼저 벗어났고, 원인 후보 1순위는 기여도가 가장 큰 분리기 압력입니다.",
    "근거": ["압축기 일이 가장 먼저(경보 9분 전) 정상 범위를 벗어남",
             "분리기 압력은 경보 3분 전부터 벗어났고 기여도 41.5%, 정상 대비 +6.4σ로 변화가 큼",
             "분리기 액위는 변화가 작아 판단 근거로 약함"],
    "우선점검": ["분리기 온도·액위 제어 상태 확인", "압축기 부하(일) 추이 확인", "분리기 하부 배출 밸브 확인"],
}


# ---------------------------------------------------------------- facts
def _num(x, nd=1):
    x = float(x)
    return None if np.isnan(x) else (int(x) if float(x).is_integer() else round(x, nd))


def build_facts(f: Finding) -> dict:
    """분석 결과 요약 (구조화). 시나리오 이름(정답)은 넣지 않는다."""
    c = f.causes.head(3)
    units = list(dict.fromkeys(c["공정 단위"]))
    return {
        "경보": {"운전시각": f"{f.alarm_min // 60}시간 {f.alarm_min % 60}분"},
        "원인후보": [{"순위": int(r["순위"]), "변수": r["변수"], "이름": r["이름"], "공정단위": r["공정 단위"],
                      "이탈시작_경보기준_분": _num(r["이탈 시작(경보 기준, 분)"]),
                      "기여도비중_퍼센트": _num(r["기여도 비중(%)"]),
                      "정상대비_시그마": _num(r["정상 대비 변화(σ)"])} for _, r in c.iterrows()],
        "전파경로_먼저벗어난순": [{"변수": r["var"], "이름": r["이름"], "공정단위": r["unit"],
                                  "경보기준_분": _num(r["minute_from_alarm"])} for _, r in f.path.head(5).iterrows()],
        "유사과거사례": [{"사례": r["과거 사례"], "고장내용": r["고장 내용"], "유형": r["유형"],
                          "유사도": _num(r["유사도"], 2)} for _, r in f.similar.iterrows()],
        "점검항목": {u: CHECKLIST[u] for u in units},
    }


def _when(m) -> str:
    if m is None:
        return "경보 구간에서 뚜렷한 이탈 없음"
    return f"경보 {abs(m)}분 전" if m < 0 else ("경보 시점" if m == 0 else f"경보 {m}분 후")


def _size(s) -> str:
    a = abs(s)
    return "변화 큼" if a >= 3 else ("변화 보통" if a >= 1 else "변화 작음")


def checklist_items(facts: dict) -> list[str]:
    return [i for u in facts["점검항목"].values() for i in u]


def render_facts(facts: dict) -> str:
    """LLM에 넘기는 문장형 입력. 변수 코드와 밑줄 키는 넣지 않는다."""
    L = [f"경보 시각: 운전 {facts['경보']['운전시각']}",
         "원인 후보 (분석 코드가 정한 순위 — 기여도와 먼저 벗어난 시점을 함께 본 것)"]
    for r in facts["원인후보"]:
        on = _when(r["이탈시작_경보기준_분"])
        on = on + " 이탈 시작" if r["이탈시작_경보기준_분"] is not None else on
        L.append(f"{r['순위']}순위 {r['이름']} [{r['공정단위']}] — {on}, 기여도 {r['기여도비중_퍼센트']}%, "
                 f"정상 대비 {r['정상대비_시그마']:+}σ ({_size(r['정상대비_시그마'])})")
    p = facts["전파경로_먼저벗어난순"]
    L.append("먼저 벗어난 순서: " + (" → ".join(f"{x['이름']}({_when(x['경보기준_분'])})" for x in p)
                                   if p else "뚜렷하게 벗어난 변수가 없음 (오경보일 수 있음)"))
    s = facts["유사과거사례"][0]
    L.append(f"가장 비슷한 과거 사례: {s['사례']} {s['고장내용']} ({s['유형']}), 유사도 {s['유사도']}")
    L.append("점검 항목 목록")
    L += [f"- {i}" for i in checklist_items(facts)]
    return "\n".join(L)


# ---------------------------------------------------------------- 코드가 채우는 칸
def fixed_fields(facts: dict) -> dict:
    c1 = facts["원인후보"][0]
    s1 = facts["유사과거사례"][0]
    cause = f"{c1['이름']} ({c1['공정단위']}) — 원인 후보 1순위"
    if not facts["전파경로_먼저벗어난순"]:
        cause += ", 단 뚜렷하게 벗어난 변수가 없어 오경보일 수 있음"
    return {"의심원인": cause, "유사사례": f"{s1['사례']} {s1['고장내용']} ({s1['유형']}), 유사도 {s1['유사도']}"}


def assemble(llm: dict, facts: dict) -> dict:
    """LLM 문장 + 코드가 채운 칸 → 보고서 순서."""
    out = {**fixed_fields(facts), **{k: llm[k] for k in LLM_FIELDS}}
    return {k: out[k] for k in FIELDS}


def template_brief(facts: dict) -> dict:
    c1 = facts["원인후보"][0]
    p = facts["전파경로_먼저벗어난순"]
    if p:
        first = f"{_when(p[0]['경보기준_분'])} {p[0]['이름']}이(가) 가장 먼저 정상 범위를 벗어났습니다."
    else:
        first = "경보 구간에서 뚜렷하게 벗어난 변수가 없어 오경보일 수 있습니다."
    ev = [f"{r['순위']}순위 {r['이름']}: 기여도 {r['기여도비중_퍼센트']}%, 정상 대비 {r['정상대비_시그마']:+}σ"
          for r in facts["원인후보"][:2]]
    if len(p) > 1:
        ev.append("벗어난 순서: " + " → ".join(x["이름"] for x in p[:3]))
    llm_like = {"요약": f"운전 {facts['경보']['운전시각']}에 경보가 울렸습니다. {first}",
                "근거": ev, "우선점검": checklist_items(facts)[:3]}
    return assemble(llm_like, facts)


def tidy(llm: dict) -> dict:
    """형식만 다듬는 후처리: 우선점검 중복 제거, 앞의 3개만. (내용은 바꾸지 않는다)"""
    out = dict(llm)
    if isinstance(out.get("우선점검"), list):
        out["우선점검"] = list(dict.fromkeys(x.strip() for x in out["우선점검"] if isinstance(x, str)))[:3]
    return out


# ---------------------------------------------------------------- 검증
def _base(name: str) -> str:
    return re.sub(r"\s*\(stream \d+\)", "", name)


_LABEL_UNIT = {}
for _c in COLS:
    _LABEL_UNIT[label(_c)] = UNIT_OF[_c]
    _LABEL_UNIT.setdefault(_base(label(_c)), UNIT_OF[_c])
_LABELS = sorted(_LABEL_UNIT, key=len, reverse=True)
_OVERCLAIM = re.compile(r"확정|확실|틀림없|분명히|단정|때문|인해")
_FOREIGN = re.compile(r"[぀-ヿ一-鿿]")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_KEY = re.compile(r"[가-힣A-Za-z]+_[가-힣A-Za-z_]+")
_CODE = re.compile(r"XMEAS\s?\d+|XMV\s?\d+", re.I)
_WHEN = re.compile(r"경보\s?(\d+)\s?분\s?(전|후)|경보\s?시점")


def _text(brief: dict) -> str:
    parts = []
    for k in LLM_FIELDS:
        v = brief.get(k, "")
        parts.extend(v if isinstance(v, list) else [v])
    return "\n".join(str(p) for p in parts)


def _rounds(vals) -> set[float]:
    out = set()
    for v in vals:
        if v is None:
            continue
        v = abs(float(v))
        out |= {v, float(round(v)), round(v, 1)}
    return out


def _norm(s: str) -> str:
    """점검 항목 비교용: 괄호 속 보충 설명, 공백, 마침표는 무시한다."""
    return re.sub(r"[\s.。·]", "", re.sub(r"\([^)]*\)", "", s))


def _find_labels(s: str) -> list[tuple[int, str]]:
    """문장 속 공정 변수 이름 (위치, 이름). 긴 이름부터 찾아 겹치지 않게."""
    found, taken = [], [False] * len(s)
    for name in _LABELS:
        for m in re.finditer(re.escape(name), s):
            if not any(taken[m.start():m.end()]):
                found.append((m.start(), name))
                for i in range(m.start(), m.end()):
                    taken[i] = True
    return sorted(found)


def _same(a: str, b: str) -> bool:
    return a == b or _base(a) == _base(b)


def verify(brief: dict, facts: dict) -> list[str]:
    """LLM이 쓴 칸(요약·근거·우선점검)의 위반 사항 목록. 비어 있으면 통과. '유형: 내용' 형식."""
    errs = [f"형식 오류: {k} 항목 누락" for k in LLM_FIELDS if k not in brief]
    if errs:
        return errs
    if not (isinstance(brief["요약"], str) and brief["요약"].strip()):
        errs.append("형식 오류: 빈 요약")
    if not all(isinstance(brief[k], list) and all(isinstance(x, str) for x in brief[k]) for k in ("근거", "우선점검")):
        return errs + ["형식 오류: 근거·우선점검은 문장 목록이어야 함"]

    text = _text(brief)
    plain = _CODE.sub(" ", text)          # 코드 속 숫자(XMEAS30의 30)는 숫자 검사에서 뺀다 — 코드는 따로 잡음
    src = render_facts(facts)             # LLM이 받은 입력 그대로
    cands = facts["원인후보"]
    path = facts["전파경로_먼저벗어난순"]
    sims = facts["유사과거사례"]
    var_min = {}                          # 변수 이름 → 이탈 시점(분, 경보 기준)
    for r in cands:
        var_min[r["이름"]] = r["이탈시작_경보기준_분"]
    for x in path:
        var_min[x["이름"]] = x["경보기준_분"]

    # 1) 숫자: 입력에 없는 숫자
    allowed = {float(x) for x in _NUM.findall(src)}
    allowed |= _rounds(allowed) | {float(round(s["유사도"] * 100)) for s in sims}
    bad = sorted({n for n in _NUM.findall(plain) if float(n) not in allowed}, key=float)
    if bad:
        errs.append("근거 없는 숫자: " + ", ".join(bad))

    # 2) 단위와 값의 짝: % = 기여도(또는 유사도×100), σ = 정상 대비 변화, N분 = 시점
    pct = _rounds([r["기여도비중_퍼센트"] for r in cands]) | {float(round(s["유사도"] * 100)) for s in sims}
    sig = _rounds([r["정상대비_시그마"] for r in cands])
    mins = _rounds(list(var_min.values()))
    mins |= {float(re.findall(r"\d+", facts["경보"]["운전시각"])[-1])}
    for n in re.findall(r"(\d+(?:\.\d+)?) ?%", plain):
        if float(n) not in pct:
            errs.append(f"단위 불일치: {n}%는 기여도 값이 아님")
    for n in re.findall(r"(\d+(?:\.\d+)?) ?σ", plain) + re.findall(r"시그마[^\d\n]{0,6}?(\d+(?:\.\d+)?)", plain):
        if float(n) not in sig:
            errs.append(f"단위 불일치: {n}σ는 정상 대비 변화 값이 아님")
    for n in re.findall(r"(\d+) ?분(?!리)", plain):
        if float(n) not in mins:
            errs.append(f"단위 불일치: {n}분은 입력에 있는 시점이 아님")

    # 3) 변수 코드·입력 항목 이름 노출, 사례 번호
    codes = sorted(set(_CODE.findall(text)))
    if codes:
        errs.append("변수 코드 노출: " + ", ".join(codes))
    keys = sorted(set(_KEY.findall(text)))
    if keys:
        errs.append("입력 항목 이름 노출: " + ", ".join(keys))
    for case in sorted(set(re.findall(r"\bF\d{2}\b", text))):
        if case not in src:
            errs.append(f"없는 과거 사례: {case}")

    # 4) 변수 이름·공정 단위: 입력에 있는 것만
    for _, name in _find_labels(text):
        if name not in src:
            errs.append(f"없는 변수 이름: {name}")
    for u in UNITS:
        if u in text and u not in src:
            errs.append(f"없는 공정 단위: {u}")

    # 5) 문장 단위 검사: 변수 옆 공정 단위, '가장 먼저', 시점, '원인' 대상
    sents = [s for x in [brief["요약"], *brief["근거"]] for s in re.split(r"(?<=[.。])\s+|\n", x) if s.strip()]
    for s in sents:
        labs = _find_labels(s)
        for pos, name in labs:                            # "이름 (단위)" / "이름 [단위]"
            m = re.match(r"\s*[\(\[]([^)\]]+)[\)\]]", s[pos + len(name):])
            if m and any(u in m.group(1) for u in UNITS) and _LABEL_UNIT[name] not in m.group(1):
                errs.append(f"공정 단위 불일치: {name}의 단위는 {_LABEL_UNIT[name]}")
        if "가장 먼저" in s and labs:
            name = labs[0][1]
            firsts = [x["이름"] for x in path if x["경보기준_분"] == path[0]["경보기준_분"]] if path else []
            if not path:
                errs.append("순서 불일치: 먼저 벗어난 변수가 없는데 '가장 먼저'라고 씀")
            elif not any(_same(name, x) for x in firsts):        # 같은 시점에 함께 벗어난 변수는 모두 인정
                errs.append(f"순서 불일치: 가장 먼저 벗어난 변수는 {path[0]['이름']}")
        w = _WHEN.findall(s)
        if len(labs) == 1 and len(w) == 1:                # 변수 하나 + 시점 하나 → 그 변수의 시점이어야 함
            name = labs[0][1]
            got = 0 if not w[0][0] else (-1 if w[0][1] == "전" else 1) * int(w[0][0])
            exp = next((v for k, v in var_min.items() if _same(k, name)), "없음")
            if exp != "없음" and exp != got:
                errs.append(f"시점 불일치: {name}의 시점은 {_when(exp)}")
        if len(labs) == 1:                                # 변수 하나 + 기여도(%)·σ → 그 변수의 값이어야 함
            name = labs[0][1]
            row = next((r for r in cands if _same(r["이름"], name)), None)
            for n in re.findall(r"(\d+(?:\.\d+)?) ?%", s):
                if "유사" in s:
                    continue
                if row is None or float(n) not in _rounds([row["기여도비중_퍼센트"]]):
                    errs.append(f"값 불일치: {n}%는 {name}의 기여도가 아님")
            for n in re.findall(r"(\d+(?:\.\d+)?) ?σ", s):
                if row is None or float(n) not in _rounds([row["정상대비_시그마"]]):
                    errs.append(f"값 불일치: {n}σ는 {name}의 정상 대비 변화가 아님")
        if re.search(r"원인(?! 미공개)", s) and labs and not _same(labs[0][1], cands[0]["이름"]):
            errs.append(f"원인 대상 불일치: 원인 후보 1순위는 {cands[0]['이름']}")

    # 6) 우선점검은 목록에서 그대로
    items = [_norm(i) for i in checklist_items(facts)]
    for x in brief["우선점검"]:
        if _norm(x) not in items:
            errs.append(f"점검 목록 밖 항목: {x}")
    if not 1 <= len(brief["우선점검"]) <= 3:
        errs.append("형식 오류: 우선점검은 1~3개")

    # 7) 단정·인과 표현, 한국어 외 문자
    if _OVERCLAIM.search(text):
        errs.append("단정·인과 표현: " + ", ".join(sorted(set(_OVERCLAIM.findall(text)))))
    if _FOREIGN.search(text):
        errs.append("한자 섞임: 한국어 외 문자")
    return list(dict.fromkeys(errs))


# ---------------------------------------------------------------- Ollama
def _post(path: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(HOST + path, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def available(model: str = MODEL) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(urllib.request.Request(HOST + "/api/tags"), timeout=3) as r:
            tags = [m["name"] for m in json.loads(r.read().decode("utf-8")).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError) as e:
        return False, f"Ollama에 연결할 수 없음 ({HOST}): {e}"
    want = model if ":" in model else model + ":latest"
    if want not in tags:
        return False, f"모델 {model} 없음 — 'ollama pull {model}' 필요 (설치된 모델: {', '.join(tags) or '없음'})"
    return True, "ok"


def _ask(facts: dict, model: str, feedback: list[str] | None, timeout: float) -> dict:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "분석 결과:\n" + EXAMPLE_IN},
            {"role": "assistant", "content": json.dumps(EXAMPLE_OUT, ensure_ascii=False)},
            {"role": "user", "content": "분석 결과:\n" + render_facts(facts)}]
    if feedback:
        msgs.append({"role": "user", "content": "방금 초안이 다음 규칙을 어겼다. 고쳐서 JSON을 다시 써라:\n- "
                     + "\n- ".join(feedback)})
    opts = {"temperature": 0, "seed": 0, "num_ctx": 4096}
    try:
        r = _post("/api/chat", {"model": model, "messages": msgs, "stream": False, "format": SCHEMA,
                                "options": opts}, timeout)
    except urllib.error.HTTPError:   # 오래된 Ollama: JSON 스키마 미지원 → 일반 JSON 모드
        r = _post("/api/chat", {"model": model, "messages": msgs, "stream": False, "format": "json",
                                "options": opts}, timeout)
    return json.loads(r["message"]["content"])


@dataclass
class Brief:
    content: dict                 # 최종으로 보여줄 칸 (LLM 문장 + 코드가 채운 칸, 또는 템플릿)
    source: str                   # "llm" | "template"
    model: str = ""
    attempts: int = 0             # LLM 호출 횟수
    violations: list = field(default_factory=list)   # 시도별 위반 사항
    drafts: list = field(default_factory=list)       # 시도별 LLM 원문 (거절된 것 포함)
    seconds: float = 0.0
    note: str = ""


def make_brief(f: Finding, use_llm: bool = True, model: str = MODEL, retries: int = 1,
               timeout: float = 300) -> Brief:
    facts = build_facts(f)
    if not use_llm:
        return Brief(template_brief(facts), "template", note="LLM 사용 안 함")
    ok, msg = available(model)
    if not ok:
        return Brief(template_brief(facts), "template", model, note=msg)
    t0, hist, drafts, fb = time.time(), [], [], None
    for i in range(retries + 1):
        try:
            raw = _ask(facts, model, fb, timeout)
            out = tidy(raw)
            errs = verify(out, facts)
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError, urllib.error.URLError, OSError) as e:
            raw, out, errs = None, None, [f"응답 처리 실패: {type(e).__name__}"]
        hist.append(errs)
        drafts.append(raw)
        if not errs:
            return Brief(assemble(out, facts), "llm", model, i + 1, hist, drafts, round(time.time() - t0, 1),
                         "검증 통과")
        fb = errs
    return Brief(template_brief(facts), "template", model, len(hist), hist, drafts, round(time.time() - t0, 1),
                 "검증 실패 → 템플릿 문장으로 대체")
