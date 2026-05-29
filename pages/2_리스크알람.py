from __future__ import annotations

import os
import sys
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sentiment import analyze_dataframe, calc_risk_score

# ── 상수 ─────────────────────────────────────────────────────
PLAYERS      = ["손흥민", "이강인", "김민재", "황희찬", "설영우"]
MAX_DATE     = pd.Timestamp("2026-05-29")
RISK_DANGER  = 70
RISK_CAUTION = 40

GRADE_STYLE = {
    "위험": {"color": "#c62828", "bg": "#ffebee", "emoji": "🔴"},
    "주의": {"color": "#e65100", "bg": "#fff3e0", "emoji": "🟡"},
    "안전": {"color": "#2e7d32", "bg": "#e8f5e9", "emoji": "🟢"},
}
PERIOD_DAYS = {"최근 7일": 7, "최근 30일": 30, "전체": None}

# ── 페이지 설정 ───────────────────────────────────────────────
st.set_page_config(
    page_title="리스크 알람 대시보드",
    page_icon="🚨",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1b5e20 0%, #2e7d32 55%, #388e3c 100%);
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: white !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.25); }
[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='color:white;text-align:center;margin-bottom:2px;'>⚽ MPRA</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#a5d6a7;text-align:center;font-size:0.8rem;margin-top:0;'>"
        "Mental &amp; Public Risk AI</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        "<p style='color:#c8e6c9;font-size:0.75rem;letter-spacing:1px;margin-bottom:6px;'>"
        "NAVIGATION</p>",
        unsafe_allow_html=True,
    )
    st.page_link("app.py",                        label="🏠 대시보드")
    st.page_link("pages/1_감성분석.py",            label="💬 감성 분석")
    st.page_link("pages/2_리스크알람.py",           label="🚨 리스크 알람")
    st.page_link("pages/3_선수비교.py",             label="📊 선수 비교")
    st.page_link("pages/4_이적가시뮬레이터.py",     label="🔄 이적가 시뮬레이터")
    st.page_link("pages/5_AI대응문구.py",           label="🤖 AI 대응 문구")
    st.markdown("---")
    st.markdown(
        "<p style='color:#81c784;font-size:0.75rem;text-align:center;'>데이터 기준: 2026년</p>",
        unsafe_allow_html=True,
    )


# ── 데이터 / 분석 캐시 ─────────────────────────────────────────
@st.cache_data
def load_data(_mtime: float = 0.0) -> pd.DataFrame:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return pd.read_csv(os.path.join(base, "data", "sample_data.csv"), encoding="utf-8-sig")


@st.cache_data
def get_full_analyzed(df: pd.DataFrame) -> pd.DataFrame:
    """전체 150개 댓글 일괄 분석 — 세션당 1회만 실행."""
    return analyze_dataframe(df)


# ── 헤더 ─────────────────────────────────────────────────────
st.html("""
<div style="
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
    border-left: 4px solid #1D9E75;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
">
    <div style="font-size: 1.4rem; font-weight: 700; color: #0a2e1f;">
        🚨 리스크 알람 센터
    </div>
    <div style="font-size: 0.85rem; color: #888; margin-top: 4px;">
        위험 등급 선수를 즉시 감지하고 알람을 발송합니다
    </div>
</div>
""")

# ── 기간 필터 ─────────────────────────────────────────────────
col_filter, col_meta = st.columns([2, 3])
with col_filter:
    period_label: str = st.radio(
        "📅 분석 기간",
        list(PERIOD_DAYS.keys()),
        index=2,
        horizontal=True,
    )

# ── 데이터 로딩 + 분석 ─────────────────────────────────────────
df_all = load_data(_mtime=os.path.getmtime(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_data.csv")))

with st.spinner("🔄 전체 선수 감성 분석 중... 처음 실행 시 잠시 기다려주세요."):
    full_analyzed = get_full_analyzed(df_all)

# 기간 필터 적용
days = PERIOD_DAYS[period_label]
if days is not None:
    cutoff = MAX_DATE - pd.Timedelta(days=days)
    period_df = full_analyzed[pd.to_datetime(full_analyzed["date"]) >= cutoff].copy()
else:
    period_df = full_analyzed.copy()

with col_meta:
    st.info(
        f"📋 **{period_label}** 기준 분석 완료 — "
        f"댓글 **{len(period_df)}개** / 선수 **{len(PLAYERS)}명**"
    )

# ── 선수별 리스크 계산 ────────────────────────────────────────
def grade_of(score: float) -> str:
    if score >= RISK_DANGER:
        return "위험"
    if score >= RISK_CAUTION:
        return "주의"
    return "안전"


risk_data: list[dict] = []
for player in PLAYERS:
    pdf = period_df[period_df["player_name"] == player]
    total = len(pdf)
    if total == 0:
        risk = 0.0
        pos = neg = 0
    else:
        risk = calc_risk_score(pdf)
        pos  = int((pdf["sentiment"] == "긍정").sum())
        neg  = int((pdf["sentiment"] == "부정").sum())
    risk_data.append(
        dict(player=player, risk=risk, grade=grade_of(risk),
             total=total, pos=pos, neg=neg)
    )

# 리스크 점수 높은 순 정렬
risk_data.sort(key=lambda x: x["risk"], reverse=True)

# 알람 시각 (시뮬레이션 기준 2026년)
# Fix 2: 윤년 2월 29일에 .replace(year=2026) → ValueError (2026은 평년)
try:
    alarm_dt = datetime.now().replace(year=2026)
except ValueError:
    alarm_dt = datetime(2026, 3, 1)
alarm_str = alarm_dt.strftime("%Y년 %m월 %d일 %H:%M:%S")

# ── 알람 배너 ─────────────────────────────────────────────────
st.subheader("🔔 자동 알람")

danger_list  = [r for r in risk_data if r["grade"] == "위험"]
caution_list = [r for r in risk_data if r["grade"] == "주의"]
safe_list    = [r for r in risk_data if r["grade"] == "안전"]

if not danger_list:
    st.success("✅ 현재 **위험** 등급 선수가 없습니다. 모든 선수 여론이 안정적입니다.")

for r in danger_list:
    neg_pct = round(r["neg"] / r["total"] * 100, 1) if r["total"] > 0 else 0.0
    st.error(
        f"⚠️ **{r['player']}** 의 부정 여론이 급증했습니다. 즉각 대응이 필요합니다.  \n"
        f"리스크 점수: **{r['risk']:.1f}점** | 부정: {r['neg']}개 ({neg_pct}%) / 전체 {r['total']}개  \n"
        f"🕐 알람 발생 시각: {alarm_str}"
    )

for r in caution_list:
    neg_pct = round(r["neg"] / r["total"] * 100, 1) if r["total"] > 0 else 0.0
    st.warning(
        f"🟡 **{r['player']}** 주의 단계입니다. 여론 모니터링을 강화하세요.  \n"
        f"리스크 점수: **{r['risk']:.1f}점** | 부정: {r['neg']}개 ({neg_pct}%) / 전체 {r['total']}개  \n"
        f"🕐 감지 시각: {alarm_str}"
    )

for r in safe_list:
    st.success(
        f"✅ **{r['player']}** 안전 단계 — 리스크 점수: {r['risk']:.1f}점"
    )

st.markdown("---")

# ── 게이지 차트 ───────────────────────────────────────────────
st.subheader("📊 선수별 리스크 점수 게이지")
st.caption(f"🟢 안전 < {RISK_CAUTION}점  |  🟡 주의 {RISK_CAUTION}~{RISK_DANGER-1}점  |  🔴 위험 ≥ {RISK_DANGER}점")


def make_gauge(value: float, player: str, grade: str) -> go.Figure:
    style = GRADE_STYLE[grade]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={
            "suffix": "점",
            "font": {"size": 22, "color": style["color"]},
        },
        title={
            "text": f"<b>{player}</b>",
            "font": {"size": 14, "color": "#333"},
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#999",
                "tickvals": [0, 40, 70, 100],
                "ticktext":  ["0", "40", "70", "100"],
                "tickfont": {"size": 10},
            },
            "bar": {"color": style["color"], "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": "#e0e0e0",
            "steps": [
                {"range": [0,            RISK_CAUTION], "color": GRADE_STYLE["안전"]["bg"]},
                {"range": [RISK_CAUTION, RISK_DANGER],  "color": GRADE_STYLE["주의"]["bg"]},
                {"range": [RISK_DANGER,  100],          "color": GRADE_STYLE["위험"]["bg"]},
            ],
            "threshold": {
                "line": {"color": "#b71c1c", "width": 3},
                "thickness": 0.8,
                "value": RISK_DANGER,
            },
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=230,
        margin=dict(t=70, b=10, l=15, r=15),
        paper_bgcolor="white",
        font=dict(family='sans-serif', color='#2C2C2C'),
    )
    return fig


# 원래 선수 순서대로 5개 나란히 표시
gauge_cols = st.columns(5)
for i, player in enumerate(PLAYERS):
    # Fix 3: next() 기본값 없으면 매칭 실패 시 StopIteration
    r = next((x for x in risk_data if x["player"] == player), None)
    if r is None:
        continue
    style = GRADE_STYLE[r["grade"]]
    with gauge_cols[i]:
        st.plotly_chart(
            make_gauge(r["risk"], r["player"], r["grade"]),
            use_container_width=True,
        )
        # Fix 1: Python < 3.12 에서 f-string 내 백슬래시 이스케이프 SyntaxError
        #         → 변수로 먼저 꺼낸 뒤 f-string에 사용
        _c  = style["color"]
        _bg = style["bg"]
        st.markdown(
            f"<p style='text-align:center;font-weight:bold;"
            f"color:{_c};background:{_bg};"
            f"border-radius:6px;padding:3px 0;margin-top:-10px;font-size:0.85rem;'>"
            f"{style['emoji']} {r['grade']}</p>",
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── 리스크 요약 테이블 ─────────────────────────────────────────
st.subheader("📋 리스크 요약 테이블")
st.caption("리스크 점수 높은 순 정렬")

rows = []
for r in risk_data:
    s = GRADE_STYLE[r["grade"]]
    neg_pct = round(r["neg"] / r["total"] * 100, 1) if r["total"] > 0 else 0.0
    rows.append({
        "선수명":     r["player"],
        "리스크 점수": r["risk"],
        "등급":       f"{s['emoji']} {r['grade']}",
        "분석 댓글":  r["total"],
        "긍정 💚":    r["pos"],
        "부정 🔴":    r["neg"],
        "부정 비율":  f"{neg_pct}%",
    })

summary_df = pd.DataFrame(rows)

def color_risk(val):
    try:
        v = float(val)
        if v >= 70:
            return 'background-color: #ffcccc; color: #8b0000; font-weight: bold;'
        elif v >= 40:
            return 'background-color: #fff3cc; color: #856404; font-weight: bold;'
        else:
            return 'background-color: #ccf0e0; color: #0a5c36; font-weight: bold;'
    except Exception:
        return ''

_styler = summary_df.style.map(
    color_risk, subset=["리스크 점수"]
).format({"리스크 점수": "{:.1f}"})

st.dataframe(_styler, use_container_width=True, hide_index=True)
