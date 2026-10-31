"""로컬 LLM 점검 보고서 초안 (추가 기능, 2026.10).

원칙
- 분석은 기존 코드가 한다. LLM은 계산된 결과를 현장 담당자가 읽을 문장으로 옮기기만 한다.
- LLM이 쓴 문장은 코드로 검사한다. 근거 없는 숫자·변수·사례, 단위가 맞지 않는 숫자, 목록 밖 점검 항목이
  하나라도 있으면 한 번 고쳐 쓰게 하고, 그래도 틀리면 버리고 템플릿 문장을 쓴다.
- 외부 API 없이 로컬 Ollama로만 동작한다. Ollama가 없으면 템플릿 문장으로 대체되고, 기존 분석 결과는 바뀌지 않는다.

v1 → v2 (1차 평가에서 발견한 문제를 고침)
- v1은 JSON(변수 코드·영문/밑줄 키)을 그대로 넘겨, 통과한 초안에도 'XMEAS30', '이탈시작_경보기준_분' 같은
  표현이 그대로 나오고 숫자 뜻을 잘못 옮긴 문장이 있었다.
- v2는 코드 없이 읽기 쉬운 문장형 입력 + 작성 예시 1개를 주고, 검사 규칙을 늘렸다.
  (변수 코드·입력 항목 이름 노출, %·σ·분 단위와 값의 짝, 점검 항목은 목록에서만, 원인은 1순위 이름 그대로)

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
from tep_meta import COLS, UNITS, label

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("RCA_LLM_MODEL", "qwen2.5:3b")
FIELDS = ["요약", "의심원인", "근거", "유사사례", "우선점검"]

SCHEMA = {
    "type": "object",
    "properties": {
        "요약": {"type": "string"},
        "의심원인": {"type": "string"},
        "근거": {"type": "array", "items": {"type": "string"}},
        "유사사례": {"type": "string"},
        "우선점검": {"type": "array", "items": {"type": "string"}},
    },
    "required": FIELDS,
}

SYSTEM = """너는 화학 공정 운전팀의 점검 보고서 초안을 쓰는 보조자다. 분석은 이미 끝났고, 너는 분석 결과를 현장 담당자가 바로 읽을 수 있는 짧은 한국어 문장으로 옮긴다.
규칙
1. 분석 결과에 적힌 숫자, 변수 이름, 사례 번호만 쓴다. 없는 숫자·변수·설비를 만들지 않는다.
2. 변수는 적힌 이름 그대로 쓴다. 영문 코드나 밑줄(_)이 들어간 이름을 쓰지 않는다.
3. 의심원인에는 원인 후보 1순위 변수의 이름을 그대로 쓴다. 원인은 '후보' 또는 '의심'으로만 표현하고 단정하지 않는다.
4. 숫자는 적힌 단위 그대로 쓴다. 기여도는 %, 정상 대비 변화는 σ, 시점은 '경보 N분 전/후'로 쓴다.
5. 변화가 '작음'으로 적힌 변수를 크게 벗어났다고 쓰지 않는다.
6. 유사사례에는 '가장 비슷한 과거 사례'의 사례 번호를 쓴다.
7. 우선점검은 '점검 항목 목록'에서 문장을 바꾸지 말고 그대로 최대 3개 고른다.
8. 요약은 2문장 이내, 근거는 3개 이내. 한국어로만 쓰고 한자를 쓰지 않는다.
9. 출력은 요약, 의심원인, 근거, 유사사례, 우선점검 키를 가진 JSON 하나만 쓴다."""

EXAMPLE_IN = """경보 시각: 운전 10시간 21분
원인 후보 (분석 코드가 정한 순위)
1순위 분리기 압력 [분리기] — 경보 9분 전 이탈 시작, 기여도 41.5%, 정상 대비 +6.4σ (변화 큼)
2순위 압축기 일 [압축기·재순환] — 경보 3분 전 이탈 시작, 기여도 12.3%, 정상 대비 -2.1σ (변화 보통)
3순위 분리기 액위 [분리기] — 경보 구간에서 뚜렷한 이탈 없음, 기여도 4.4%, 정상 대비 +0.6σ (변화 작음)
먼저 벗어난 순서: 분리기 압력(경보 9분 전) → 압축기 일(경보 3분 전)
가장 비슷한 과거 사례: F05 응축기 냉각수 입구 온도 (계단), 유사도 0.88
그다음 비슷한 사례: F12 응축기 냉각수 입구 온도 (랜덤 변동) 유사도 0.71
점검 항목 목록
- 분리기 온도·액위 제어 상태 확인
- 분리기 하부 배출 밸브 확인
- 압축기 부하(일) 추이 확인
- 재순환 밸브 개도 확인"""

EXAMPLE_OUT = {
    "요약": "운전 10시간 21분에 경보가 울렸습니다. 분리기 압력이 경보 9분 전에 가장 먼저 벗어났고, 이어서 압축기 일이 변했습니다.",
    "의심원인": "분리기 압력 (분리기) — 원인 후보 1순위",
    "근거": ["분리기 압력이 가장 먼저(경보 9분 전) 정상 범위를 벗어남",
             "분리기 압력의 기여도 41.5%로 가장 크고, 정상 대비 +6.4σ로 변화가 큼",
             "압축기 일은 경보 3분 전부터 벗어나 분리기 압력보다 늦게 변함"],
    "유사사례": "F05 응축기 냉각수 입구 온도 고장과 가장 비슷함 (유사도 0.88)",
    "우선점검": ["분리기 온도·액위 제어 상태 확인", "분리기 하부 배출 밸브 확인", "압축기 부하(일) 추이 확인"],
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
    L = [f"경보 시각: 운전 {facts['경보']['운전시각']}", "원인 후보 (분석 코드가 정한 순위)"]
    for r in facts["원인후보"]:
        on = _when(r["이탈시작_경보기준_분"])
        on = on + " 이탈 시작" if r["이탈시작_경보기준_분"] is not None else on
        L.append(f"{r['순위']}순위 {r['이름']} [{r['공정단위']}] — {on}, 기여도 {r['기여도비중_퍼센트']}%, "
                 f"정상 대비 {r['정상대비_시그마']:+}σ ({_size(r['정상대비_시그마'])})")
    p = facts["전파경로_먼저벗어난순"]
    L.append("먼저 벗어난 순서: " + (" → ".join(f"{x['이름']}({_when(x['경보기준_분'])})" for x in p)
                                   if p else "뚜렷하게 벗어난 변수가 없음 (오경보일 수 있음)"))
    s = facts["유사과거사례"]
    L.append(f"가장 비슷한 과거 사례: {s[0]['사례']} {s[0]['고장내용']} ({s[0]['유형']}), 유사도 {s[0]['유사도']}")
    if len(s) > 1:
        L.append("그다음 비슷한 사례: " + "; ".join(f"{x['사례']} {x['고장내용']} ({x['유형']}) 유사도 {x['유사도']}"
                                               for x in s[1:]))
    L.append("점검 항목 목록")
    L += [f"- {i}" for i in checklist_items(facts)]
    return "\n".join(L)


# ---------------------------------------------------------------- template (대체 문장)
def template_brief(facts: dict) -> dict:
    c1 = facts["원인후보"][0]
    p = facts["전파경로_먼저벗어난순"]
    s1 = facts["유사과거사례"][0]
    if p:
        first = f"{_when(p[0]['경보기준_분'])} {p[0]['이름']}이(가) 가장 먼저 정상 범위를 벗어났습니다."
    else:
        first = "경보 구간에서 뚜렷하게 벗어난 변수가 없어 오경보일 수 있습니다."
    ev = [f"{r['순위']}순위 {r['이름']}: 기여도 {r['기여도비중_퍼센트']}%, 정상 대비 {r['정상대비_시그마']:+}σ"
          for r in facts["원인후보"][:2]]
    if len(p) > 1:
        ev.append("벗어난 순서: " + " → ".join(x["이름"] for x in p[:3]))
    return {
        "요약": f"운전 {facts['경보']['운전시각']}에 경보가 울렸습니다. {first}",
        "의심원인": f"{c1['이름']} ({c1['공정단위']}) — 원인 후보 1순위",
        "근거": ev,
        "유사사례": f"{s1['사례']} {s1['고장내용']} ({s1['유형']}), 유사도 {s1['유사도']}",
        "우선점검": checklist_items(facts)[:3],
    }


# ---------------------------------------------------------------- 검증
def _base(name: str) -> str:
    return re.sub(r"\s*\(stream \d+\)", "", name)


_LABELS = sorted({label(c) for c in COLS} | {_base(label(c)) for c in COLS}, key=len, reverse=True)
_OVERCLAIM = re.compile(r"확정|확실|틀림없|분명히|단정")
_FOREIGN = re.compile(r"[぀-ヿ一-鿿]")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_KEY = re.compile(r"[가-힣A-Za-z]+_[가-힣A-Za-z_]+")
_CODE = re.compile(r"XMEAS\s?\d+|XMV\s?\d+", re.I)


def _text(brief: dict) -> str:
    parts = []
    for k in FIELDS:
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


def verify(brief: dict, facts: dict) -> list[str]:
    """위반 사항 목록. 비어 있으면 통과."""
    errs = [f"형식 오류: {k} 항목 누락" for k in FIELDS if k not in brief]
    if errs:
        return errs
    if not all(isinstance(brief[k], str) and brief[k].strip() for k in ("요약", "의심원인", "유사사례")):
        errs.append("형식 오류: 빈 항목")
    if not all(isinstance(brief[k], list) and all(isinstance(x, str) for x in brief[k]) for k in ("근거", "우선점검")):
        return errs + ["형식 오류: 근거·우선점검은 문장 목록이어야 함"]

    text = _text(brief)
    plain = _CODE.sub(" ", text)          # 코드 속 숫자(XMEAS30의 30)는 숫자 검사에서 뺀다 — 코드는 3)에서 따로 잡음
    src = render_facts(facts)            # LLM이 받은 입력 그대로
    cands = facts["원인후보"]
    sims = facts["유사과거사례"]

    # 1) 숫자: 입력에 없는 숫자
    allowed = {float(x) for x in _NUM.findall(src)}
    allowed |= _rounds(allowed) | {float(round(s["유사도"] * 100)) for s in sims}
    bad = sorted({n for n in _NUM.findall(plain) if float(n) not in allowed}, key=float)
    if bad:
        errs.append("근거 없는 숫자: " + ", ".join(bad))

    # 2) 단위와 값의 짝: % = 기여도(또는 유사도×100), σ = 정상 대비 변화, N분 = 시점
    pct = _rounds([r["기여도비중_퍼센트"] for r in cands]) | {float(round(s["유사도"] * 100)) for s in sims}
    sig = _rounds([r["정상대비_시그마"] for r in cands])
    mins = _rounds([r["이탈시작_경보기준_분"] for r in cands] + [x["경보기준_분"] for x in facts["전파경로_먼저벗어난순"]])
    hh, mm = map(int, re.findall(r"\d+", facts["경보"]["운전시각"]))
    mins |= {float(mm)}
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

    # 4) 변수 이름·공정 단위: 입력에 있는 것만 (긴 이름부터 찾고 지움)
    masked = text
    for name in _LABELS:
        if name in masked:
            if name not in src:
                errs.append(f"없는 변수 이름: {name}")
            masked = masked.replace(name, " ")
    for u in UNITS:
        if u in text and u not in src:
            errs.append(f"없는 공정 단위: {u}")

    # 5) 의심원인 = 1순위 이름, 유사사례 = 가장 비슷한 사례 번호
    c1 = cands[0]
    if c1["이름"] not in brief["의심원인"] and _base(c1["이름"]) not in brief["의심원인"]:
        errs.append(f"의심원인 불일치: 1순위 후보 이름({c1['이름']})이 없음")
    if sims[0]["사례"] not in brief["유사사례"]:
        errs.append(f"유사사례 불일치: 가장 비슷한 사례 번호({sims[0]['사례']})가 없음")

    # 6) 우선점검은 목록에서 그대로
    items = [_norm(i) for i in checklist_items(facts)]
    for x in brief["우선점검"]:
        if _norm(x) not in items:
            errs.append(f"점검 목록 밖 항목: {x}")
    if len(brief["우선점검"]) == 0 or len(brief["우선점검"]) > 3:
        errs.append("형식 오류: 우선점검은 1~3개")

    # 7) 단정 표현, 한국어 외 문자
    if _OVERCLAIM.search(text):
        errs.append("단정 표현: " + ", ".join(sorted(set(_OVERCLAIM.findall(text)))))
    if _FOREIGN.search(text):
        errs.append("한자 섞임: 한국어 외 문자")
    return errs


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
    content: dict                 # 최종으로 보여줄 문장 (LLM 또는 템플릿)
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
            out = _ask(facts, model, fb, timeout)
            errs = verify(out, facts)
        except (json.JSONDecodeError, KeyError, TypeError, urllib.error.URLError, OSError) as e:
            out, errs = None, [f"응답 처리 실패: {type(e).__name__}"]
        hist.append(errs)
        drafts.append(out)
        if not errs:
            return Brief(out, "llm", model, i + 1, hist, drafts, round(time.time() - t0, 1), "검증 통과")
        fb = errs
    return Brief(template_brief(facts), "template", model, len(hist), hist, drafts, round(time.time() - t0, 1),
                 "검증 실패 → 템플릿 문장으로 대체")
