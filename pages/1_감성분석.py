from __future__ import annotations

import os
import re
import sys
from collections import Counter

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sentiment import analyze_dataframe

# ── 상수 ─────────────────────────────────────────────────────
PLAYERS = ["이강인", "손흥민", "김민재", "황희찬", "설영우"]
COLOR_POS = "#4caf50"
COLOR_NEG = "#f44336"
COLOR_NEU = "#9e9e9e"

STOP_WORDS = {
    # 조사
    "이", "가", "은", "는", "을", "를", "의", "에", "에서", "으로", "로", "와", "과",
    # 의미없는 단어
    "것", "수", "좀", "더", "거", "데", "듯", "때", "건", "거나",
    "진짜", "그냥", "아직", "너무",
    # 동사 어미
    "같음", "같아", "같은데", "하는데", "하고", "했는데",
    # 추가 불용어
    "있는", "없는", "있어", "없어", "있다", "없다", "있음", "없음",
    "하는", "하지", "않아", "않고", "이미", "그런", "이런", "어떤",
    "그래서", "하지만", "그리고", "많이", "정말", "아닌가", "않는",
    "이랑", "이나", "이제", "솔직히", "보임", "느낌", "생각", "부분",
    "입장에서", "비교하면", "이후로", "입장", "느껴짐", "것같음",
    "ㅋㅋ", "ㄷㄷ", "ㄹㅇ", "ㅎㅎ", "뭐", "왜", "안", "못", "다", "또",
    # 선수명 (선수별 탭에서 이미 필터링되므로 무의미)
    "이강인", "손흥민", "김민재", "황희찬", "설영우",
}

# ── 페이지 설정 ───────────────────────────────────────────────
st.set_page_config(
    page_title="감성 분석 대시보드",
    page_icon="💬",
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
        "⚽ 선수 선택</p>",
        unsafe_allow_html=True,
    )
    selected_player = st.selectbox(
        "선수",
        PLAYERS,
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<p style='color:#c8e6c9;font-size:0.75rem;letter-spacing:1px;margin-bottom:6px;'>"
        "NAVIGATION</p>",
        unsafe_allow_html=True,
    )
    st.page_link("app.py", label="🏠 대시보드")
    st.page_link("pages/1_감성분석.py", label="💬 감성 분석")
    st.page_link("pages/2_리스크알람.py", label="🚨 리스크 알람")
    st.page_link("pages/3_선수비교.py", label="📊 선수 비교")
    st.page_link("pages/4_이적가시뮬레이터.py", label="🔄 이적가 시뮬레이터")
    st.page_link("pages/5_AI대응문구.py", label="🤖 AI 대응 문구")
    st.markdown("---")
    st.markdown(
        "<p style='color:#81c784;font-size:0.75rem;text-align:center;'>데이터 기준: 2026년</p>",
        unsafe_allow_html=True,
    )


# ── 데이터 로딩 ───────────────────────────────────────────────
@st.cache_data
def load_data(_mtime: float = 0.0) -> pd.DataFrame:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return pd.read_csv(os.path.join(base, "data", "sample_data.csv"), encoding="utf-8-sig")


@st.cache_data
def get_analyzed_player(df: pd.DataFrame, player: str) -> pd.DataFrame:
    """player를 캐시 키로 사용, _df는 해시 제외."""
    return analyze_dataframe(df)


df_all = load_data(_mtime=os.path.getmtime(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_data.csv")))
player_df = df_all[df_all["player_name"] == selected_player].copy()

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
        📊 감성 분석 대시보드
    </div>
    <div style="font-size: 0.85rem; color: #888; margin-top: 4px;">
        선수별 SNS 여론을 실시간으로 분석합니다
    </div>
</div>
""")

with st.spinner(f"🔄 {selected_player} 댓글 감성 분석 중..."):
    analyzed = get_analyzed_player(player_df, selected_player)

# ── 집계 ─────────────────────────────────────────────────────
total     = len(analyzed)
pos_count = int((analyzed["sentiment"] == "긍정").sum())
neg_count = int((analyzed["sentiment"] == "부정").sum())
pos_ratio = round(pos_count / total * 100, 1)
neg_ratio = round(neg_count / total * 100, 1)

# ── 지표 카드 ─────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("📋 전체 댓글 수", f"{total}개")
c2.metric("💚 긍정 비율", f"{pos_ratio}%", delta=f"{pos_count}개")
c3.metric(
    "🔴 부정 비율",
    f"{neg_ratio}%",
    delta=f"{neg_count}개",
    delta_color="inverse",
)

st.markdown("---")

# ── 도넛 차트 + 라인 차트 ─────────────────────────────────────
col_donut, col_line = st.columns([1, 2])

with col_donut:
    st.subheader("감성 비율")
    donut = go.Figure(go.Pie(
        labels=["긍정", "부정"],
        values=[pos_count, neg_count],
        hole=0.55,
        marker=dict(colors=[COLOR_POS, COLOR_NEG]),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value}개 (%{percent})<extra></extra>",
    ))
    donut.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        height=300,
    )
    donut.update_layout(paper_bgcolor='white', plot_bgcolor='#F8F9FA', font=dict(family='sans-serif', color='#2C2C2C'), margin=dict(t=40, b=40, l=40, r=40))
    st.plotly_chart(donut, use_container_width=True)

with col_line:
    st.subheader("월별 긍정 비율 추이")

    # Fix 1: 캐시된 analyzed 직접 변경 금지 → 별도 복사본 사용
    # Fix 2: .apply(lambda p: p.to_timestamp()) 불안정 → .dt.to_timestamp() 사용
    chart_df = analyzed.copy()
    chart_df["_month"] = (
        pd.to_datetime(chart_df["date"])
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    monthly = (
        chart_df.groupby("_month")["sentiment"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["긍정", "부정"]:
        if col not in monthly.columns:
            monthly[col] = 0
    denom = (monthly["긍정"] + monthly["부정"]).replace(0, 1)
    monthly["긍정률"] = (monthly["긍정"] / denom * 100).fillna(0)
    monthly["label"] = monthly["_month"].dt.strftime("%m월")

    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(
        x=monthly["label"],
        y=monthly["긍정률"],
        mode="lines+markers",
        line=dict(color="#2e7d32", width=2.5),
        marker=dict(size=9, color=COLOR_POS, line=dict(color="white", width=1.5)),
        fill="tozeroy",
        fillcolor="rgba(76,175,80,0.12)",
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ))
    line_fig.update_layout(
        xaxis_title="",
        yaxis=dict(title="긍정 비율 (%)", range=[0, 100], gridcolor="#e0e0e0"),
        margin=dict(t=10, b=30, l=50, r=20),
        height=300,
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    line_fig.update_xaxes(showgrid=False)
    line_fig.update_layout(paper_bgcolor='white', plot_bgcolor='#F8F9FA', font=dict(family='sans-serif', color='#2C2C2C'), margin=dict(t=40, b=40, l=40, r=40))
    st.plotly_chart(line_fig, use_container_width=True)

st.markdown("---")

# ── 댓글 상세 테이블 ──────────────────────────────────────────
st.subheader("댓글 상세 내역")
st.caption("최신 날짜 순 정렬 | 감성 셀 색상: 💚 긍정 / 🔴 부정")

display_df = (
    analyzed[["date", "platform", "comment", "sentiment", "confidence"]]
    .copy()
    .sort_values("date", ascending=False)
    .reset_index(drop=True)
)
display_df.columns = ["날짜", "플랫폼", "댓글 내용", "감성", "신뢰도"]
display_df["신뢰도"] = display_df["신뢰도"].map(lambda x: f"{x:.1%}")


def _highlight_sentiment(val: str) -> str:
    if val == "긍정":
        return "background-color:#c8e6c9;color:#1b5e20;font-weight:bold;"
    if val == "부정":
        return "background-color:#ffcdd2;color:#b71c1c;font-weight:bold;"
    return "background-color:#f5f5f5;color:#616161;"


# Fix 4: pandas < 2.1.0 에서는 style.map 없음 → hasattr로 버전 분기
_styler = display_df.style
styled_table = (
    _styler.map(_highlight_sentiment, subset=["감성"])
    if hasattr(_styler, "map")
    else _styler.applymap(_highlight_sentiment, subset=["감성"])
)
st.dataframe(styled_table, use_container_width=True, hide_index=True, height=420)

st.markdown("---")

# ── 부정 여론 주요 키워드 TOP 5 ──────────────────────────────
st.subheader("부정 여론 주요 키워드 TOP 5")
st.caption("부정 댓글 전체 텍스트에서 의미 있는 단어 추출 (조사·불용어 제외)")

neg_texts = analyzed[analyzed["sentiment"] == "부정"]["comment"].tolist()


def extract_keywords(texts: list[str], top_n: int = 5) -> list[tuple[str, int]]:
    words: list[str] = []
    for text in texts:
        for token in re.split(r"\s+", str(text)):
            token = re.sub(r"[^\w]", "", token)
            if len(token) >= 2 and token not in STOP_WORDS:
                words.append(token)
    return Counter(words).most_common(top_n)


keywords = extract_keywords(neg_texts)

if keywords:
    kw_df = pd.DataFrame(keywords, columns=["키워드", "빈도"])
    bar_fig = go.Figure(go.Bar(
        x=kw_df["빈도"],
        y=kw_df["키워드"],
        orientation="h",
        marker=dict(
            color=kw_df["빈도"],
            colorscale=[[0, "#ffcdd2"], [1, "#c62828"]],
            showscale=False,
        ),
        text=kw_df["빈도"],
        textposition="outside",
        hovertemplate="%{y}: %{x}회<extra></extra>",
    ))
    bar_fig.update_layout(
        xaxis=dict(title="언급 횟수", gridcolor="#e0e0e0"),
        yaxis=dict(autorange="reversed"),
        margin=dict(t=10, b=40, l=150, r=80),
        height=280,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    bar_fig.update_yaxes(showgrid=False)
    bar_fig.update_layout(paper_bgcolor='white', plot_bgcolor='#F8F9FA', font=dict(family='sans-serif', color='#2C2C2C'), margin=dict(t=40, b=40, l=40, r=40))
    st.plotly_chart(bar_fig, use_container_width=True)
else:
    st.info("부정 댓글이 없어 키워드를 추출할 수 없습니다.")
