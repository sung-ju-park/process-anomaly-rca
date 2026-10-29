"""경보 1건에 대한 1장짜리 RCA 리포트(HTML).

문장은 계산된 값만으로 채운다(템플릿). 생성형 AI가 숫자를 지어낼 여지를 없앤다.
"""
from __future__ import annotations

import html
from datetime import datetime

import numpy as np

from engine import CHECKLIST, Finding
from tep_meta import UNITS

CSS = """
*{box-sizing:border-box}body{font-family:'Malgun Gothic','Noto Sans KR','Noto Sans CJK KR',sans-serif;
color:#1d2733;margin:0;background:#fff;font-size:12.5px;line-height:1.55}
.page{max-width:820px;margin:0 auto;padding:28px 34px}
h1{font-size:20px;margin:0 0 2px}h2{font-size:14px;margin:18px 0 6px;color:#0f3d5e;border-bottom:2px solid #0f3d5e;padding-bottom:3px}
.meta{color:#5b6672;font-size:11.5px}.sum{background:#f2f6f9;border-left:4px solid #c2410c;padding:10px 14px;margin:12px 0;font-size:13px}
table{border-collapse:collapse;width:100%}th{background:#0f3d5e;color:#fff;font-weight:600;padding:5px 6px;font-size:11.5px}
td{border-bottom:1px solid #dde3e8;padding:5px 6px;text-align:center}td.l{text-align:left}
.tag{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;background:#fde8dc;color:#9a3412}
.flow{display:flex;flex-wrap:wrap;gap:6px;align-items:center}.step{border:1px solid #c9d3dc;border-radius:6px;padding:4px 8px;background:#fff}
.step b{color:#c2410c}.arr{color:#8a96a3}ul{margin:4px 0 0 18px;padding:0}.foot{margin-top:18px;color:#6b7580;font-size:10.5px;border-top:1px solid #dde3e8;padding-top:8px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
"""


def _spark(values: np.ndarray, t_alarm: int, width=360, height=90, limit: float | None = 1.0) -> str:
    v = np.asarray(values, float)
    n = len(v)
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    if limit is not None:
        hi = max(hi, limit)
    hi = hi if hi > lo else lo + 1

    def X(i):
        return 4 + (width - 8) * i / max(1, n - 1)

    def Y(x):
        return height - 6 - (height - 12) * (x - lo) / (hi - lo)

    pts = " ".join(f"{X(i):.1f},{Y(x):.1f}" for i, x in enumerate(v))
    parts = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
             f'<rect width="{width}" height="{height}" fill="#f7f9fb"/>']
    if limit is not None:
        parts.append(f'<line x1="4" x2="{width - 4}" y1="{Y(limit):.1f}" y2="{Y(limit):.1f}" stroke="#c2410c" stroke-dasharray="4 3"/>')
    parts.append(f'<line x1="{X(t_alarm):.1f}" x2="{X(t_alarm):.1f}" y1="2" y2="{height - 2}" stroke="#0f3d5e" stroke-width="1.5"/>')
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#2f6fb0" stroke-width="1.4"/></svg>')
    return "".join(parts)


def render(f: Finding, X: np.ndarray, sc, title: str = "공정 이상 RCA 리포트") -> str:
    c = f.causes
    first = f.path.iloc[0] if len(f.path) else None
    top_unit = c.iloc[0]["공정 단위"]
    if first is not None:
        when = int(first["minute_from_alarm"])
        when_txt = f"경보 {abs(when)}분 전" if when < 0 else ("경보 시점" if when == 0 else f"경보 {when}분 후")
        lead = f"{when_txt}에 <b>{html.escape(first['이름'])}</b>이(가) 가장 먼저 정상 범위를 벗어났습니다. "
    else:
        lead = ""
    sim = f.similar.iloc[0]
    summary = (f"{lead}원인 후보 1순위는 <b>{html.escape(c.iloc[0]['이름'])}</b>({top_unit})이며, "
               f"과거 사례 중 <b>{html.escape(sim['고장 내용'])}</b>과(와) 가장 비슷합니다(유사도 {sim['유사도']:.2f}).")

    rows = "".join(
        f"<tr><td>{r['순위']}</td><td class='l'>{html.escape(r['이름'])}<br><span class='meta'>{r['변수']}</span></td>"
        f"<td>{r['공정 단위']}</td><td>{'-' if np.isnan(r['이탈 시작(경보 기준, 분)']) else int(r['이탈 시작(경보 기준, 분)'])}</td>"
        f"<td>{r['기여도 비중(%)']}</td><td>{r['정상 대비 변화(σ)']:+.1f}</td></tr>"
        for _, r in c.head(3).iterrows())
    flow = '<span class="arr">→</span>'.join(
        f"<span class='step'><b>{int(r['minute_from_alarm']):+d}분</b> {html.escape(r['이름'])} <span class='meta'>({r['unit']})</span></span>"
        for _, r in f.path.head(5).iterrows()) or "<span class='meta'>경보 구간에서 뚜렷이 벗어난 변수가 없습니다.</span>"
    sims = "".join(
        f"<tr><td>{r['과거 사례']}</td><td class='l'>{html.escape(r['고장 내용'])}</td><td>{r['유형']}</td><td>{r['유사도']:.3f}</td></tr>"
        for _, r in f.similar.iterrows())
    units = list(dict.fromkeys(c.head(3)["공정 단위"]))
    checks = "".join(f"<li><b>{u}</b>: {', '.join(CHECKLIST[u])}</li>" for u in units)

    a0, a1 = max(0, f.t_alarm - 60), min(len(X), f.t_alarm + 60)
    spark_score = _spark(f.score[a0:a1], f.t_alarm - a0)
    z = sc.transform(X)
    v1 = c.iloc[0]["변수"]
    from tep_meta import COLS
    spark_v1 = _spark(z[a0:a1, COLS.index(v1)], f.t_alarm - a0, limit=None)

    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>{title}</title><style>{CSS}</style></head>
<body><div class="page">
<h1>{title}</h1>
<div class="meta">시나리오: {html.escape(f.scenario)} · 경보 시각: 운전 {f.alarm_min // 60}시간 {f.alarm_min % 60}분 · 생성: {datetime.now():%Y-%m-%d %H:%M} · 탐지: 동적 PCA(T²·SPE), 3회 연속 기준선 초과</div>
<div class="sum">{summary}</div>
<h2>원인 후보 Top 3</h2>
<table><tr><th>순위</th><th>변수</th><th>공정 단위</th><th>이탈 시작(분)</th><th>기여도 비중(%)</th><th>정상 대비(σ)</th></tr>{rows}</table>
<div class="meta" style="margin-top:4px">순위 = 재구성 기반 기여도 순위와 '먼저 벗어난 순서'의 평균 (사전에 정한 방법). 이탈 시작은 경보 시점 기준, 음수는 경보 전.</div>
<h2>이상 전파 경로 (먼저 벗어난 순서)</h2><div class="flow">{flow}</div>
<div class="grid" style="margin-top:12px">
<div><div class="meta">이상 점수 (점선 = 기준선, 세로선 = 경보)</div>{spark_score}</div>
<div><div class="meta">1순위 변수 {html.escape(c.iloc[0]['이름'])} (정상 대비 σ)</div>{spark_v1}</div></div>
<h2>유사 과거 사례</h2>
<table><tr><th>사례</th><th>고장 내용</th><th>유형</th><th>유사도</th></tr>{sims}</table>
<h2>점검 권고</h2><ul>{checks}</ul>
<div class="foot">이 리포트의 문장과 수치는 모두 계산값으로 채운 템플릿입니다. 원인 '후보'이며 현장 확인이 필요합니다.
데이터: Tennessee Eastman Process 벤치마크 (Downs &amp; Vogel 1993; Braatz 연구실 공개 데이터).</div>
</div></body></html>"""
