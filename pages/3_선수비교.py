from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sentiment import analyze_dataframe, calc_risk_score

# ── 상수 ─────────────────────────────────────────────────────
PLAYERS = ["손흥민", "이강인", "김민재", "황희찬", "설영우"]

PLAYER_COLORS = {
    "손흥민": "#1565c0",
    "이강인": "#6a1b9a",
    "김민재": "#e65100",
    "황희찬": "#2e7d32",
    "설영우": "#b71c1c",
}

GRADE_STYLE = {
    "위험": {"color": "#c62828", "bg": "#ffebee", "emoji": "🔴"},
    "주의": {"color": "#e65100", "bg": "#fff3e0", "emoji": "🟡"},
    "안전": {"color": "#2e7d32", "bg": "#e8f5e9", "emoji": "🟢"},
}


def grade_of(score: float) -> str:
    if score >= 70:
        return "위험"
    if score >= 40:
        return "주의"
    return "안전"


def hex_to_rgba(hex_color: str, alpha: float = 0.18) -> str:
    """HEX 색상 → rgba() 문자열 (fill 투명도 조절용)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── 페이지 설정 ───────────────────────────────────────────────
st.set_page_config(
    page_title="선수 비교 대시보드",
    page_icon="📊",
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
        "<p style='color:#c8e6c9;font-size:0.85rem;font-weight:bold;margin-bottom:4px;'>"
        "⚽ 비교 선수 선택</p>",
        unsafe_allow_html=True,
    )
    selected: list[str] = st.multiselect(
        "선수",
        PLAYERS,
        default=PLAYERS,
        label_visibility="collapsed",
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


# ── 데이터 캐시 ───────────────────────────────────────────────
@st.cache_data
def load_data(_mtime: float = 0.0) -> pd.DataFrame:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return pd.read_csv(os.path.join(base, "data", "sample_data.csv"), encoding="utf-8-sig")


@st.cache_data
def get_full_analyzed(df: pd.DataFrame) -> pd.DataFrame:
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
        👥 선수 비교 분석
    </div>
    <div style="font-size: 0.85rem; color: #888; margin-top: 4px;">
        전체 선수의 리스크를 한눈에 비교합니다
    </div>
</div>
""")

# 2명 미만 선택 시 조기 종료
if len(selected) < 2:
    st.warning("⚠️ 비교하려면 사이드바에서 **2명 이상** 선택해주세요.")
    st.stop()

# ── 데이터 로딩 + 분석 ─────────────────────────────────────────
df_all = load_data(_mtime=os.path.getmtime(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_data.csv")))

with st.spinner("🔄 선수 데이터 감성 분석 중..."):
    full_analyzed = get_full_analyzed(df_all)

# ── 선수별 통계 계산 ──────────────────────────────────────────
stats: dict[str, dict] = {}
for player in selected:
    pdf = full_analyzed[full_analyzed["player_name"] == player]
    total = len(pdf)
    if total == 0:
        stats[player] = {
            "total": 0, "pos": 0, "neg": 0,
            "pos_pct": 0.0, "neg_pct": 0.0,
            "risk": 0.0, "grade": "안전",
        }
        continue
    pos  = int((pdf["sentiment"] == "긍정").sum())
    neg  = int((pdf["sentiment"] == "부정").sum())
    risk = calc_risk_score(pdf)
    stats[player] = {
        "total": total,
        "pos": pos, "neg": neg,
        "pos_pct": round(pos / total * 100, 1),
        "neg_pct": round(neg / total * 100, 1),
        "risk": risk,
        "grade": grade_of(risk),
    }

# ── 요약 카드 ─────────────────────────────────────────────────
safest   = min(selected, key=lambda p: stats[p]["risk"])
riskiest = max(selected, key=lambda p: stats[p]["risk"])
avg_risk = round(sum(stats[p]["risk"] for p in selected) / len(selected), 1)

c1, c2, c3 = st.columns(3)
c1.metric("📊 선택 선수 평균 리스크", f"{avg_risk}점")
c2.metric(
    "🟢 가장 안전한 선수",
    safest,
    delta=f"{stats[safest]['risk']:.1f}점",
)
c3.metric(
    "🔴 가장 위험한 선수",
    riskiest,
    delta=f"{stats[riskiest]['risk']:.1f}점",
    delta_color="inverse",
)

st.markdown("---")

# ── Row 1: 레이더 차트 + 리스크 바 차트 ─────────────────────
col_radar, col_risk = st.columns(2)

with col_radar:
    st.subheader("긍정 / 부정 비율 비교")

    stack_fig = go.Figure()
    stack_fig.add_trace(go.Bar(
        name="긍정 💚",
        y=selected,
        x=[stats[p]["pos_pct"] for p in selected],
        orientation="h",
        marker_color="#4caf50",
        text=[f"{stats[p]['pos_pct']:.1f}%" for p in selected],
        textposition="inside",
        textfont=dict(color="white", size=12, family="Arial Black"),
        hovertemplate="%{y} 긍정: %{x:.1f}%<extra></extra>",
    ))
    stack_fig.add_trace(go.Bar(
        name="부정 🔴",
        y=selected,
        x=[stats[p]["neg_pct"] for p in selected],
        orientation="h",
        marker_color="#f44336",
        text=[f"{stats[p]['neg_pct']:.1f}%" for p in selected],
        textposition="inside",
        textfont=dict(color="white", size=12, family="Arial Black"),
        hovertemplate="%{y} 부정: %{x:.1f}%<extra></extra>",
    ))
    stack_fig.update_layout(
        barmode="stack",
        xaxis=dict(title="비율 (%)", range=[0, 100], gridcolor="#e0e0e0"),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(t=40, b=40, l=70, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    stack_fig.update_yaxes(showgrid=False)
    stack_fig.update_layout(paper_bgcolor='white', plot_bgcolor='#F8F9FA', font=dict(family='sans-serif', color='#2C2C2C'), margin=dict(t=40, b=40, l=40, r=40))
    st.plotly_chart(stack_fig, use_container_width=True)

with col_risk:
    st.subheader("리스크 점수 비교")

    bar_order  = sorted(selected, key=lambda p: stats[p]["risk"], reverse=True)
    bar_colors = [GRADE_STYLE[stats[p]["grade"]]["color"] for p in bar_order]

    risk_fig = go.Figure(go.Bar(
        x=[stats[p]["risk"] for p in bar_order],
        y=bar_order,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{stats[p]['risk']:.1f}점" for p in bar_order],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}점<extra></extra>",
    ))
    # 기준선
    risk_fig.add_vline(
        x=70, line_dash="dash", line_color="#c62828", line_width=1.5,
        annotation_text="위험 기준(70)", annotation_position="top right",
        annotation_font_size=10,
    )
    risk_fig.add_vline(
        x=40, line_dash="dot", line_color="#e65100", line_width=1.5,
        annotation_text="주의 기준(40)", annotation_position="bottom right",
        annotation_font_size=10,
    )
    risk_fig.update_layout(
        xaxis=dict(title="리스크 점수", range=[0, 120], gridcolor="#e0e0e0"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
        margin=dict(t=30, b=40, l=80, r=80),
    )
    risk_fig.update_yaxes(showgrid=False)
    risk_fig.update_layout(paper_bgcolor='white', plot_bgcolor='#F8F9FA', font=dict(family='sans-serif', color='#2C2C2C'), margin=dict(t=40, b=40, l=40, r=40))
    st.plotly_chart(risk_fig, use_container_width=True)

st.markdown("---")

# ── Row 2: 감성 분포 그룹 바 차트 ────────────────────────────
st.subheader("감성 분포 상세 비교")

sent_config = [
    ("긍정 💚", "pos", "#4caf50"),
    ("부정 🔴", "neg", "#f44336"),
]

grouped_fig = go.Figure()
for label, key, color in sent_config:
    grouped_fig.add_trace(go.Bar(
        name=label,
        x=selected,
        y=[stats[p][key] for p in selected],
        marker_color=color,
        text=[stats[p][key] for p in selected],
        textposition="inside",
        textfont=dict(color="white", size=12),
        hovertemplate="%{x} " + label + ": %{y}개<extra></extra>",
    ))

grouped_fig.update_layout(
    barmode="group",
    xaxis_title="선수",
    yaxis=dict(title="댓글 수", gridcolor="#e0e0e0"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=320,
    margin=dict(t=40, b=40, l=50, r=20),
    plot_bgcolor="white",
    paper_bgcolor="white",
)
grouped_fig.update_xaxes(showgrid=False)
grouped_fig.update_layout(paper_bgcolor='white', plot_bgcolor='#F8F9FA', font=dict(family='sans-serif', color='#2C2C2C'), margin=dict(t=40, b=40, l=40, r=40))
st.plotly_chart(grouped_fig, use_container_width=True)

st.markdown("---")

# ── 상세 비교 테이블 ──────────────────────────────────────────
st.subheader("📋 선수별 상세 비교")
st.caption("리스크 점수 높은 순 정렬")

rows = []
for player in selected:
    s  = stats[player]
    gs = GRADE_STYLE[s["grade"]]
    rows.append({
        "선수명":       player,
        "리스크 점수":  s["risk"],
        "등급":         gs["emoji"] + " " + s["grade"],
        "총 댓글":      s["total"],
        "긍정 💚":      s["pos"],
        "부정 🔴":      s["neg"],
        "긍정 비율":    f"{s['pos_pct']}%",
        "부정 비율":    f"{s['neg_pct']}%",
    })

cmp_df = (
    pd.DataFrame(rows)
    .sort_values("리스크 점수", ascending=False)
    .reset_index(drop=True)
)

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

_styler = cmp_df.style.map(
    color_risk, subset=["리스크 점수"]
).format({"리스크 점수": "{:.1f}"})

st.dataframe(_styler, use_container_width=True, hide_index=True)
