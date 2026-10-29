"""공정 이상 탐지 · 근본 원인 분석(RCA) 대시보드.

실행: streamlit run app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from engine import CHECKLIST, Engine  # noqa: E402
from report import render  # noqa: E402
from tep_data import SAMPLE_MIN, TEST_ONSET, load  # noqa: E402
from tep_meta import COLS, FAULTS, GROUND_TRUTH, UNIT_OF, label  # noqa: E402

REP = ROOT / "reports"
NAVY, ORANGE, BLUE, GREY = "#0f3d5e", "#c2410c", "#2f6fb0", "#8a96a3"

st.set_page_config(page_title="공정 이상 RCA", page_icon="🏭", layout="wide")
st.markdown(f"""<style>
.block-container{{padding-top:1.6rem;max-width:1280px}}
h1,h2,h3{{color:{NAVY}}}
div[data-testid="stMetric"]{{background:#f4f7fa;border:1px solid #e1e7ed;border-radius:10px;padding:12px 16px}}
div[data-testid="stMetricLabel"] p{{font-size:.85rem;color:#4b5866}}
.card{{border:1px solid #e1e7ed;border-radius:10px;padding:12px 14px;background:#fff;height:100%}}
.card .rk{{font-size:.8rem;color:{ORANGE};font-weight:700}}.card .nm{{font-size:1.02rem;font-weight:700;color:{NAVY}}}
.card .sub{{font-size:.82rem;color:#5b6672}}
.pill{{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.78rem;margin-right:4px;background:#eef3f8;color:{NAVY}}}
.pill.o{{background:#fde8dc;color:#9a3412}}
.note{{font-size:.82rem;color:#5b6672}}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="탐지 모델과 과거 사례 라이브러리를 준비하는 중…")
def engine():
    return Engine()


@st.cache_data
def scenario(f: int) -> np.ndarray:
    return load(f, "test").values


def jload(name):
    return json.load(open(REP / name, encoding="utf-8"))


def alarm_runs(score: np.ndarray, persist: int = 3):
    """경보 구간(연속 초과 구간)의 시작 인덱스 목록."""
    over = score > 1
    runs, i = [], 0
    while i < len(over):
        if over[i]:
            j = i
            while j < len(over) and over[j]:
                j += 1
            if j - i >= persist:
                runs.append(i + persist - 1)
            i = j
        else:
            i += 1
    return runs


def flow_svg(order: dict) -> str:
    """공정 흐름도. order: {공정 단위: 이탈 순서(1부터)}"""
    pos = {"원료 공급": (20, 95), "반응기": (190, 95), "응축기": (360, 95), "분리기": (530, 95),
           "스트리퍼·제품": (700, 95), "압축기·재순환": (445, 205), "퍼지": (615, 10)}
    w, h = 140, 52
    col = {1: ORANGE, 2: "#ea8a4f", 3: "#f2b48c"}
    s = ['<svg viewBox="0 0 860 270" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">',
         '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
         f'<path d="M0,0 L8,4 L0,8 z" fill="{GREY}"/></marker></defs>']

    def edge(x1, y1, x2, y2):
        s.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{GREY}" stroke-width="2" marker-end="url(#a)"/>')
    for a, b in [("원료 공급", "반응기"), ("반응기", "응축기"), ("응축기", "분리기"), ("분리기", "스트리퍼·제품")]:
        edge(pos[a][0] + w, pos[a][1] + h / 2, pos[b][0], pos[b][1] + h / 2)
    edge(pos["분리기"][0] + 50, pos["분리기"][1] + h, pos["압축기·재순환"][0] + 100, pos["압축기·재순환"][1])
    s.append(f'<polyline points="{pos["압축기·재순환"][0]},{pos["압축기·재순환"][1] + h / 2} 260,{pos["압축기·재순환"][1] + h / 2} 260,{pos["반응기"][1] + h}" '
             f'fill="none" stroke="{GREY}" stroke-width="2" marker-end="url(#a)"/>')
    edge(pos["분리기"][0] + 110, pos["분리기"][1], pos["퍼지"][0] + 40, pos["퍼지"][1] + h)
    for u, (x, y) in pos.items():
        k = order.get(u)
        fill = col.get(k, "#fde8dc") if k else "#f4f7fa"
        stroke = ORANGE if k else "#c9d3dc"
        tc = "#fff" if k in (1, 2) else NAVY
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        s.append(f'<text x="{x + w / 2}" y="{y + h / 2 + 5}" text-anchor="middle" font-size="17" font-weight="700" fill="{tc}">{u}</text>')
        if k:
            s.append(f'<circle cx="{x + w - 4}" cy="{y + 4}" r="12" fill="{NAVY}"/>'
                     f'<text x="{x + w - 4}" y="{y + 9}" text-anchor="middle" font-size="13" font-weight="700" fill="#fff">{k}</text>')
    s.append("</svg>")
    return "".join(s)


def score_chart(score, now, runs, sel=None):
    t = np.arange(len(score)) * SAMPLE_MIN / 60
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[: now + 1], y=score[: now + 1], mode="lines", line=dict(color=BLUE, width=1.6),
                             name="이상 점수"))
    fig.add_hline(y=1, line=dict(color=ORANGE, dash="dash"), annotation_text="기준선", annotation_position="bottom right")
    for r in runs:
        if r <= now:
            fig.add_vline(x=r * SAMPLE_MIN / 60, line=dict(color=NAVY if r == sel else "#c9d3dc", width=2 if r == sel else 1))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(type="log", tickvals=[0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], title="점수 (기준선=1, 로그)"),
                      xaxis_title="운전 시간 (h)", showlegend=False, plot_bgcolor="#fafbfc")
    return fig


# ------------------------------------------------------------------ 사이드바
st.sidebar.markdown("### 🏭 공정 이상 탐지 · RCA")
page = st.sidebar.radio("메뉴", ["개요", "모니터링 · 원인 분석", "성능 평가", "실데이터 분석 (SECOM)", "방법 · 의사결정"],
                        label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("데이터: Tennessee Eastman Process(공정 시뮬레이션 벤치마크), "
                   "SECOM(UCI, 실제 반도체 공정 센서). 모든 수치는 reports/ 폴더의 실행 결과.")

det = jload("tep_detection_summary.json")
rca_s = pd.read_csv(REP / "tep_rca_summary.csv", index_col=0)
sig = jload("tep_signature_summary.json")
sec = jload("secom_summary.json")

# ------------------------------------------------------------------ 개요
if page == "개요":
    st.title("공정 이상 탐지 · 근본 원인 분석(RCA)")
    st.markdown("센서 데이터에서 **이상을 잡고 → 어떤 센서가 먼저 흔들렸는지 추적하고 → 원인 후보와 비슷한 과거 사례를 "
                "리포트로** 내는 파이프라인입니다. 원인 지목이 맞았는지 **사전에 고정한 정답으로 채점**했습니다.")
    c = st.columns(5)
    c[0].metric("평균 탐지율 (동적 PCA)", f"{det['DPCA']['mean_detect_rate_all']:.1%}", help="고장 21종, 고장 후 샘플 중 기준선 초과 비율")
    c[1].metric("오경보율", f"{det['DPCA']['far_alarm3']:.1%}", help="3회 연속 초과 규칙, 정상 hold-out + 고장 전 구간")
    c[2].metric("탐지 지연 중앙값", f"{det['DPCA']['median_delay_min']:.0f}분")
    c[3].metric("원인 1순위 적중", f"{round(rca_s.loc['onset', 'hit@1'] * 11)}/11",
                help="정답이 정해진 고장 11종. '먼저 벗어난 순서' 기준. 탐지된 9종만 보면 8/9")
    c[4].metric("유사 과거 사례 적중", f"{sig['top1_acc']:.0%}", help=f"고장 {sig['n_eval']}종, top-1")
    st.markdown("#### 파이프라인")
    steps = [("① 전처리·EDA", "정상 구간 기준 표준화, 분석계 샘플링·제어 루프 확인"),
             ("② 이상 탐지", "PCA·동적 PCA·Isolation Forest·오토인코더 비교"),
             ("③ 원인 추적", "기여도(RBC) + 먼저 벗어난 순서 → 원인 후보 순위"),
             ("④ 유사 사례", "평균 이동 + 변동 크기 시그니처로 과거 고장 검색"),
             ("⑤ 리포트", "계산값만으로 채운 1장짜리 RCA 리포트")]
    cols = st.columns(5)
    for col, (a, b) in zip(cols, steps):
        col.markdown(f"<div class='card'><div class='nm'>{a}</div><div class='sub'>{b}</div></div>", unsafe_allow_html=True)
    st.markdown("#### 무엇이 다른가")
    a, b = st.columns(2)
    a.markdown(f"""<div class='card'><div class='rk'>차별점 1</div><div class='nm'>RCA를 '채점'한다</div>
<div class='sub'>고장 설명과 공정 구조만으로 정답 원인 변수를 먼저 정하고 git에 올린 뒤 채점했습니다.
그 결과 <b>SHAP 같은 설명 기법은 원인을 1순위로 {round(rca_s.loc['shap', 'hit@1'] * 11)}/11만 맞혔고</b>,
'먼저 벗어난 순서'가 {round(rca_s.loc['onset', 'hit@1'] * 11)}/11로 가장 정확했습니다.
'정상과 구분되는 변수'와 '원인 변수'는 다르다는 걸 숫자로 확인했습니다.</div></div>""", unsafe_allow_html=True)
    b.markdown(f"""<div class='card'><div class='rk'>차별점 2</div><div class='nm'>전파 경로 + 유사 사례 + 자동 리포트</div>
<div class='sub'>경보를 누르면 이상이 어느 공정 단위에서 시작해 어디로 번졌는지 흐름도에 표시하고,
과거 고장 라이브러리에서 가장 비슷한 사례(top-1 적중 {sig['top1_acc']:.0%})를 찾아 점검 항목과 함께 1장 리포트로 만듭니다.</div></div>""",
               unsafe_allow_html=True)
    st.markdown("#### 실제 반도체 데이터(SECOM)에서 확인한 것")
    cp = sec["change_point"]
    st.markdown(f"- 무작위로 섞어 검증하면 불량 예측 ROC-AUC **{sec['validation']['random_5fold']['roc_auc_mean']:.2f}**, "
                f"시간순으로 검증하면 **{sec['validation']['time_5fold']['roc_auc_mean']:.2f}** → 무작위 분할은 성능을 부풀린다\n"
                f"- '불량 관련 센서' {sec['candidates']['naive_q10']}개 중 같은 주 안에서 비교해도 남는 센서는 "
                f"{sec['candidates']['within_week_q10']}개, 그 뒤 기간에서 재현된 센서는 {sec['replication']['replicated']}개\n"
                f"- {cp['time'][:10]}을 기점으로 불량률 {cp['fail_rate_before']:.1%} → {cp['fail_rate_after']:.1%} "
                f"(순열검정 p={cp['perm_p']:.3f}). 이때 크게 바뀐 센서 {cp['n_shifted']}개 중 "
                f"{len(cp['overlap_with_naive_candidates'])}개가 '불량 관련 센서'와 겹침")

# ------------------------------------------------------------------ 모니터링
elif page == "모니터링 · 원인 분석":
    E = engine()
    st.title("모니터링 · 원인 분석")
    left, right = st.columns([1, 2.2])
    with left:
        opts = {0: "정상 운전"} | {f: f"고장 시나리오 {f:02d}" for f in range(1, 22)}
        blind = st.toggle("정답 가리고 보기", value=True, help="켜면 어떤 고장인지 숨기고 분석 결과만 봅니다")
        f = st.selectbox("시나리오 (48시간 운전 기록, 3분 간격)", list(opts),
                         format_func=lambda k: opts[k] if blind or k == 0 else f"{opts[k]} — {FAULTS[k][0]}", index=6)
        X = scenario(f)
        score = E.score(X)
        runs = alarm_runs(score)
        now = st.slider("현재 시각 (운전 시간, h)", 0.0, (len(X) - 1) * SAMPLE_MIN / 60, 24.0, 0.25)
        now_i = int(now * 60 / SAMPLE_MIN)
        seen = [r for r in runs if r <= now_i]
        st.markdown(f"<span class='pill'>경보 {len(seen)}건</span>"
                    f"<span class='pill'>고장 투입: {'없음' if f == 0 else '8시간 시점' if not blind else '숨김'}</span>",
                    unsafe_allow_html=True)
        if not seen:
            st.info("아직 경보가 없습니다. 시각을 뒤로 옮겨 보세요.")
            sel = None
        else:
            sel = st.radio("분석할 경보", seen, format_func=lambda r: f"{r * SAMPLE_MIN // 60}시간 {r * SAMPLE_MIN % 60}분 경보",
                           index=next((i for i, r in enumerate(seen) if r >= TEST_ONSET), 0) if f else 0)
    with right:
        st.plotly_chart(score_chart(score, now_i, runs, sel), use_container_width=True)

    if sel is not None:
        F = E.investigate(X, sel, opts[f] if blind or f == 0 else f"{opts[f]} — {FAULTS[f][0]}")
        st.markdown("### 원인 후보")
        cards = st.columns(3)
        for col, (_, r) in zip(cards, F.causes.head(3).iterrows()):
            on = r["이탈 시작(경보 기준, 분)"]
            on_txt = "경보 구간에서 뚜렷한 이탈 없음" if np.isnan(on) else (f"경보 {abs(int(on))}분 전 이탈 시작" if on < 0 else f"경보 후 {int(on)}분 이탈")
            col.markdown(f"<div class='card'><div class='rk'>{r['순위']}순위 · {r['공정 단위']}</div>"
                         f"<div class='nm'>{r['이름']}</div><div class='sub'>{r['변수']} · {on_txt}<br>"
                         f"기여도 비중 {r['기여도 비중(%)']}% · 정상 대비 {r['정상 대비 변화(σ)']:+.1f}σ</div></div>",
                         unsafe_allow_html=True)
        a, b = st.columns([1.35, 1])
        with a:
            st.markdown("#### 이상 전파 경로")
            order = {}
            for _, r in F.path.iterrows():
                if r["unit"] not in order and len(order) < 3:
                    order[r["unit"]] = len(order) + 1
            st.markdown(flow_svg(order), unsafe_allow_html=True)
            if len(F.path):
                st.markdown(" → ".join(f"**{int(r['minute_from_alarm']):+d}분** {r['이름']}" for _, r in F.path.head(5).iterrows()))
            else:
                st.caption("정상 범위를 뚜렷하게 벗어난 변수가 없습니다 — 오경보일 가능성이 높습니다.")
        with b:
            st.markdown("#### 상위 변수 추이 (정상 대비 σ)")
            Z = E.mon.sc.transform(X)
            a0, a1 = max(0, sel - 60), min(len(X), sel + 60)
            fig = go.Figure()
            for v, c in zip(F.causes["변수"].head(3), [ORANGE, BLUE, GREY]):
                fig.add_trace(go.Scatter(x=np.arange(a0, a1) * SAMPLE_MIN / 60, y=Z[a0:a1, COLS.index(v)],
                                         name=label(v), line=dict(color=c, width=1.6)))
            fig.add_vline(x=sel * SAMPLE_MIN / 60, line=dict(color=NAVY))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25),
                              plot_bgcolor="#fafbfc", xaxis_title="운전 시간 (h)")
            st.plotly_chart(fig, use_container_width=True)
        a, b = st.columns([1.35, 1])
        with a:
            st.markdown("#### 비슷한 과거 사례")
            st.dataframe(F.similar, hide_index=True, use_container_width=True)
        with b:
            st.markdown("#### 점검 권고")
            for u in dict.fromkeys(F.causes.head(3)["공정 단위"]):
                st.markdown(f"**{u}** — " + ", ".join(CHECKLIST[u]))
            st.download_button("📄 RCA 리포트 내려받기 (HTML)", render(F, X, E.mon.sc).encode("utf-8"),
                               file_name=f"rca_report_{f:02d}_{sel}.html", mime="text/html", use_container_width=True)
        if f in GROUND_TRUTH:
            with st.expander("정답 확인 (채점 기준)"):
                gt, unit, why = GROUND_TRUTH[f]
                rank = F.ranking["rbc+onset"].sort_values(ascending=False).index.tolist()
                best = min(rank.index(v) for v in gt) + 1
                st.markdown(f"**실제 고장:** {FAULTS[f][0]} ({FAULTS[f][1]})  \n**정답 원인 변수:** "
                            + ", ".join(f"{label(v)}({v})" for v in gt)
                            + f"  \n**근거:** {why}  \n**이번 분석에서 정답 변수의 최고 순위:** {best}위")
        elif f:
            with st.expander("정답 확인"):
                st.markdown(f"**실제 고장:** {FAULTS[f][0]} ({FAULTS[f][1]}) — 단일 원인 변수를 정할 수 없어 채점에서 제외한 고장입니다.")

# ------------------------------------------------------------------ 성능 평가
elif page == "성능 평가":
    st.title("성능 평가")
    st.caption("TEP 테스트 파일 21종(고장 8시간 시점 투입). 모델·기준선·설정값은 테스트를 보기 전에 정상 데이터와 학습 파일로만 정했습니다.")
    st.markdown("### ① 이상 탐지 4종 비교")
    d = pd.read_csv(REP / "tep_detection_by_fault.csv")
    piv = d.pivot(index="fault", columns="model", values="detect_rate")[["PCA", "DPCA", "IsolationForest", "AutoEncoder"]]
    a, b = st.columns([1.5, 1])
    with a:
        fig = go.Figure(go.Heatmap(z=piv.values * 100, x=["PCA", "동적 PCA", "Isolation Forest", "오토인코더"],
                                   y=[f"{i:02d} {FAULTS[i][0]}" for i in piv.index], colorscale="Blues",
                                   text=np.round(piv.values * 100).astype(int), texttemplate="%{text}", zmin=0, zmax=100,
                                   colorbar=dict(title="탐지율%")))
        fig.update_layout(height=640, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    with b:
        t = pd.DataFrame({{"PCA": "PCA", "DPCA": "동적 PCA", "IsolationForest": "IForest", "AutoEncoder": "AE"}[m]: {"평균 탐지율": f"{v['mean_detect_rate_all']:.1%}",
                              "탐지율 (3·9·15 제외)": f"{v['mean_detect_rate_excl_3_9_15']:.1%}",
                              "오경보율 (3회 연속)": f"{v['far_alarm3']:.2%}",
                              "탐지 지연 중앙값": f"{v['median_delay_min']:.0f}분"} for m, v in det.items()})
        st.dataframe(t, use_container_width=True)
        st.markdown("<div class='note'>고장 3·9·15번은 모든 방법이 거의 못 잡습니다. 문헌에서도 같은 결과가 보고된 고장입니다. "
                    "머신러닝(Isolation Forest)이 상관 구조 기반 PCA보다 오히려 낮았고, 시간 상관을 넣은 동적 PCA가 가장 좋았습니다.</div>",
                    unsafe_allow_html=True)
    st.markdown("### ② RCA 정답 채점")
    names = {"zscore": "단변량 z", "spe": "PCA 잔차 기여도", "rbc": "재구성 기반 기여도(RBC)", "shap": "SHAP",
             "onset": "먼저 벗어난 순서", "rbc+onset": "RBC + 순서 (사전 지정)", "shap+onset": "SHAP + 순서"}
    r = rca_s.rename(index=names)
    fig = go.Figure()
    for col, c in [("hit@1", ORANGE), ("hit@3", BLUE), ("unit@1", GREY)]:
        fig.add_trace(go.Bar(x=r.index, y=r[col] * 100, name={"hit@1": "정답 변수 1순위", "hit@3": "정답 변수 3순위 안",
                                                               "unit@1": "공정 단위 1순위"}[col], marker_color=c,
                             text=[f"{v * 11:.0f}/11" for v in r[col]], textposition="outside"))
    fig.update_layout(barmode="group", height=360, yaxis_title="%", margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.12), plot_bgcolor="#fafbfc")
    st.plotly_chart(fig, use_container_width=True)
    bf = pd.read_csv(REP / "tep_rca_by_fault.csv")
    rk = bf[["fault", "delay_min"] + [f"{m}|best_rank" for m in names]].copy()
    rk.columns = ["고장", "탐지 지연(분)"] + list(names.values())
    rk.insert(1, "내용", [FAULTS[i][0] for i in rk["고장"]])
    st.dataframe(rk, hide_index=True, use_container_width=True)
    st.markdown("<div class='note'>표의 숫자 = 정답 원인 변수가 받은 가장 높은 순위(1이 최고). 15·21번은 경보가 고장 후 12시간 이상 지나서야 울려 사실상 미탐지. "
                "1번(원료 조성 변화)은 모든 방법이 실패 — 유량 제어기가 변화를 가려 압력이 먼저 반응했기 때문입니다.</div>",
                unsafe_allow_html=True)
    st.markdown("### ③ 유사 과거 사례 검색 · 경보 구간 클러스터링")
    a, b = st.columns([1, 1.2])
    with a:
        sc = pd.read_csv(REP / "tep_similar_case.csv")
        sc = sc[sc["eval"]][["fault", "top1", "top1_sim", "hit1", "hit3"]]
        sc.columns = ["질의 고장", "가장 비슷한 과거 사례", "유사도", "1순위 적중", "3순위 안 적중"]
        st.metric("top-1 / top-3 적중", f"{sig['top1_acc']:.0%} / {sig['top3_acc']:.0%}", help="평가 고장 18종 (3·9·15 제외)")
        st.dataframe(sc, hide_index=True, use_container_width=True, height=300)
        st.caption("첫 버전(평균 이동만)은 44% → 랜덤 변동·밸브 고착형 고장을 놓치는 원인을 찾아 '변동 크기'를 추가 (학습 파일로 검증 후 반영).")
    with b:
        wc = pd.read_csv(REP / "tep_window_clusters.csv")
        fig = go.Figure(go.Scatter(x=wc.x, y=wc.y, mode="markers", text=wc.case,
                                   marker=dict(color=wc.cluster, colorscale="Turbo", size=6, opacity=0.8)))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#fafbfc",
                          title=dict(text=f"경보 구간 {sig['cluster_windows']}개를 라벨 없이 {sig['cluster_k']}개 군집으로 (NMI {sig['cluster_nmi']:.2f})",
                                     font=dict(size=13)))
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ SECOM
elif page == "실데이터 분석 (SECOM)":
    st.title("실데이터 분석 — SECOM 반도체 공정")
    st.caption(f"{sec['n_lots']:,}개 lot × 센서 {sec['n_sensors']}개, 불량 {sec['n_fail']}건, "
               f"{sec['period'][0][:10]} ~ {sec['period'][1][:10]}. 센서 이름은 익명(S000~S589).")
    a, b, c = st.columns(3)
    log = sec["prep_log"]
    a.markdown("#### ① 센서 정리")
    fig = go.Figure(go.Funnel(y=[x[0] for x in log], x=[x[1] for x in log], marker=dict(color=[NAVY, BLUE, "#6c9bd2", "#a9c4e6"])))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
    a.plotly_chart(fig, use_container_width=True)
    b.markdown("#### ② 검증 방식의 함정")
    v = sec["validation"]
    fig = go.Figure(go.Bar(x=["무작위 5-fold", "시간순 5-fold"], y=[v["random_5fold"]["roc_auc_mean"], v["time_5fold"]["roc_auc_mean"]],
                           marker_color=[GREY, ORANGE], text=[f"{v['random_5fold']['roc_auc_mean']:.2f}", f"{v['time_5fold']['roc_auc_mean']:.2f}"],
                           textposition="outside"))
    fig.add_hline(y=0.5, line=dict(color="#c9d3dc", dash="dot"), annotation_text="무작위 추측(0.5)", annotation_position="bottom left")
    fig.update_layout(height=280, yaxis=dict(range=[0, 1], title="불량 예측 ROC-AUC"), margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="#fafbfc")
    b.plotly_chart(fig, use_container_width=True)
    c.markdown("#### ③ 원인 후보 센서")
    cand = sec["candidates"]
    vals = [cand["naive_q10"], cand["within_week_q10"], sec["replication"]["replicated"]]
    fig = go.Figure(go.Bar(y=["단순 비교로 유의", "같은 주 안에서도 유의", "뒤 기간에서 재현"], x=vals, orientation="h",
                           marker_color=[GREY, BLUE, ORANGE], text=[f"{v}개" for v in vals], textposition="outside"))
    fig.update_layout(height=280, margin=dict(l=10, r=30, t=10, b=10), yaxis=dict(autorange="reversed"),
                      xaxis=dict(range=[0, max(vals) * 1.25], title="센서 수"), plot_bgcolor="#fafbfc")
    c.plotly_chart(fig, use_container_width=True)
    st.markdown("#### ④ 불량률 변화 시점과 그때 바뀐 센서")
    wk = pd.read_csv(REP / "secom_weekly_fail.csv")
    med = pd.read_csv(REP / "secom_weekly_sensor_median.csv")
    cp = sec["change_point"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=wk.week.str[:10], y=wk["mean"] * 100, name="주별 불량률(%)", marker_color="#f2b48c", yaxis="y"))
    for s_, c_ in zip([x for x in med.columns if x != "week"], [NAVY, BLUE, GREY, "#6c9bd2"]):
        z = (med[s_] - med[s_].mean()) / (med[s_].std() + 1e-12)
        fig.add_trace(go.Scatter(x=med.week.str[:10], y=z, name=f"{s_} 주별 중앙값(표준화)", line=dict(color=c_), yaxis="y2"))
    cpw = next((w for w in wk.week if w[:10] <= cp["time"][:10] <= w[-10:]), wk.week.iloc[0])
    fig.add_vline(x=cpw[:10], line=dict(color=ORANGE, dash="dash"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="#fafbfc",
                      yaxis=dict(title="불량률 %"), yaxis2=dict(overlaying="y", side="right", title="센서(표준화)"),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"- 변화 시점 **{cp['time'][:16]}** — 불량률 {cp['fail_rate_before']:.1%} → {cp['fail_rate_after']:.1%} (이항 우도비 + 순열검정 p={cp['perm_p']:.3f})\n"
                f"- 이 시점 전후로 크게 바뀐 센서 {cp['n_shifted']}개: {', '.join(cp['shifted'])}\n"
                f"- 그중 {len(cp['overlap_with_naive_candidates'])}개가 단순 비교의 '불량 관련 센서'와 겹침 → 불량의 직접 원인이라기보다 공정 상태 변화의 흔적일 가능성\n"
                f"- SHAP 상위 10개 중 부트스트랩 30회에서 70% 이상 다시 뽑힌 센서: {', '.join(sec['shap']['stable_ge_0_7'])}")
    st.info("결론: 불량 원인이 기간마다 달라지는 데이터에서는 '불량 예측 모델 하나'보다 **공정 변화 시점 감지 → 시점 전후 센서 비교 → 기간별 재검증** 체계가 현실적입니다.")

# ------------------------------------------------------------------ 방법
else:
    st.title("방법 · 의사결정")
    st.markdown(open(ROOT / "docs" / "decision_log.md", encoding="utf-8").read())
    st.markdown("""#### 한계
- TEP는 시뮬레이션 데이터라 실제 설비의 센서 고장·결측·운전 조건 변경이 없습니다. 그래서 실제 데이터(SECOM)를 함께 분석했습니다.
- RCA 정답은 공개된 고장 설명과 공정 구조로 정한 것이라, 문헌마다 정답 변수 세트가 조금씩 다를 수 있습니다.
- SECOM 센서는 익명이라 원인 후보를 물리적으로 확인할 수 없습니다. 결론은 '후보'와 '재현성'까지입니다.
- 코드와 분석은 생성형 AI(Claude)와 함께 작성했습니다. 모든 수치는 reports/ 폴더의 스크립트 실행 결과이며, 다시 실행해 같은 값이 나오는지 확인했습니다.""")
