from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.sentiment import analyze_dataframe, calc_risk_score

st.set_page_config(
    page_title="MPRA – Player Risk AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0a2e1f; }
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebarNav"] { display: none; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.18); }
.stButton button { background-color: #1D9E75; color: white; border: none; }
.stButton button:hover { background-color: #17876a; }

.mpra-header {
    background: linear-gradient(135deg, #0a2e1f 0%, #1a6b4a 60%, #1D9E75 100%);
    border-radius: 16px;
    padding: 44px 52px;
    margin-bottom: 32px;
}
.mpra-logo  { font-size: 3.4rem; font-weight: 900; color: #fff; margin: 0 0 10px; line-height: 1; }
.mpra-slogan { font-size: 1.3rem; color: #a8f0d9; margin: 0 0 6px; font-weight: 600; }
.mpra-subtitle { font-size: 0.95rem; color: #6fcfb0; margin: 0; letter-spacing: 0.3px; }

.metric-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 22px 22px 18px;
    border-left: 5px solid #1D9E75;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    height: 100%;
}
.metric-card.danger  { border-left-color: #E24B4A; }
.metric-card.warning { border-left-color: #EF9F27; }
.metric-label { font-size: 0.75rem; color: #999; font-weight: 700;
                text-transform: uppercase; letter-spacing: 0.8px; margin: 0 0 8px; }
.metric-value { font-size: 2.4rem; font-weight: 900; color: #1a1a1a;
                margin: 0; line-height: 1.1; }
.metric-unit  { font-size: 1rem; font-weight: 400; color: #bbb; margin-left: 2px; }
.metric-sub   { font-size: 0.75rem; color: #bbb; margin: 8px 0 0; }

.section-title {
    font-size: 1.05rem; font-weight: 700; color: #1a1a1a;
    border-left: 4px solid #1D9E75; padding-left: 10px;
    margin: 32px 0 12px;
}

.alarm-card {
    background: #fff5f5;
    border: 1px solid #fcc;
    border-left: 5px solid #E24B4A;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.alarm-card.warning {
    background: #fffbf0;
    border-color: #fde68a;
    border-left-color: #EF9F27;
}
</style>
""", unsafe_allow_html=True)

# ── 사이드바 ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 10px;'>
      <div style='font-size:2.2rem;line-height:1;'>⚽</div>
      <div style='font-size:1.4rem;font-weight:900;letter-spacing:2px;margin-top:6px;'>MPRA</div>
      <div style='font-size:0.7rem;color:#6fcfb0;margin-top:4px;'>v1.0 &nbsp;·&nbsp; 2026</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.7rem;letter-spacing:2px;color:#6fcfb0;margin-bottom:10px;'>MENU</p>",
        unsafe_allow_html=True,
    )
    st.page_link("app.py",                        label="🏠 대시보드")
    st.page_link("pages/1_감성분석.py",            label="📊 감성 분석")
    st.page_link("pages/2_리스크알람.py",           label="🚨 리스크 알람")
    st.page_link("pages/3_선수비교.py",             label="👥 선수 비교")
    st.page_link("pages/4_이적가시뮬레이터.py",     label="💰 이적가 시뮬레이터")
    st.page_link("pages/5_AI대응문구.py",           label="🤖 AI 대응 문구")
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;padding:6px 0 12px;'>
      <div style='font-size:0.7rem;color:#6fcfb0;'>Powered by Claude AI</div>
    </div>
    """, unsafe_allow_html=True)

# ── 데이터 로딩 ──────────────────────────────────────────────
@st.cache_data
def load_data(_mtime: float = 0.0) -> pd.DataFrame:
    base = os.path.dirname(os.path.abspath(__file__))
    return pd.read_csv(os.path.join(base, "data", "sample_data.csv"), encoding="utf-8-sig")


@st.cache_data
def get_risk_summary(df: pd.DataFrame) -> pd.DataFrame:
    analyzed = analyze_dataframe(df)
    rows = []
    for player in df["player_name"].unique():
        pdf = analyzed[analyzed["player_name"] == player]
        risk = calc_risk_score(pdf)
        pos = int((pdf["sentiment"] == "긍정").sum())
        neg = int((pdf["sentiment"] == "부정").sum())
        rows.append({
            "선수명": player,
            "총 댓글": len(pdf),
            "긍정": pos,
            "부정": neg,
            "리스크 점수": round(float(risk), 1),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("리스크 점수", ascending=False)
        .reset_index(drop=True)
    )


_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_data.csv")
df = load_data(_mtime=os.path.getmtime(_csv_path))
with st.spinner("AI 감성 분석 중..."):
    risk_df = get_risk_summary(df)

# ── 헤더 배너 ────────────────────────────────────────────────
st.markdown("""
<div class="mpra-header">
  <p class="mpra-logo">⚽ MPRA</p>
  <p class="mpra-slogan">경기장 밖의 데이터로 선수의 가치를 지킵니다</p>
  <p class="mpra-subtitle">AI 기반 선수 멘탈 리스크 &amp; 팬덤 여론 관리 시스템</p>
</div>
""", unsafe_allow_html=True)

# ── 모니터링 현황 카드 4개 ───────────────────────────────────
total_players  = len(risk_df)
total_comments = int(risk_df["총 댓글"].sum())
today_comments = max(1, round(total_comments / 150))
danger_count   = int((risk_df["리스크 점수"] >= 60).sum())
avg_risk       = round(float(risk_df["리스크 점수"].mean()), 1)

c1, c2, c3, c4 = st.columns(4, gap="medium")

with c1:
    st.markdown(f"""
    <div class="metric-card">
      <p class="metric-label">모니터링 선수</p>
      <p class="metric-value">{total_players}<span class="metric-unit">명</span></p>
      <p class="metric-sub">실시간 모니터링 중</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
      <p class="metric-label">오늘 수집 댓글</p>
      <p class="metric-value">{today_comments:,}<span class="metric-unit">건</span></p>
      <p class="metric-sub">2026년 5월 29일 기준</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    card_cls   = "danger" if danger_count > 0 else ""
    val_color  = "#E24B4A" if danger_count > 0 else "#1a1a1a"
    st.markdown(f"""
    <div class="metric-card {card_cls}">
      <p class="metric-label">위험 등급 선수</p>
      <p class="metric-value" style="color:{val_color};">{danger_count}<span class="metric-unit">명</span></p>
      <p class="metric-sub">리스크 점수 60점 이상</p>
    </div>
    """, unsafe_allow_html=True)

with c4:
    risk_color = "#E24B4A" if avg_risk >= 60 else "#EF9F27" if avg_risk >= 35 else "#1D9E75"
    card_cls   = "danger" if avg_risk >= 60 else "warning" if avg_risk >= 35 else ""
    st.markdown(f"""
    <div class="metric-card {card_cls}">
      <p class="metric-label">평균 리스크 점수</p>
      <p class="metric-value" style="color:{risk_color};">{avg_risk}<span class="metric-unit">/100</span></p>
      <p class="metric-sub">전체 선수 평균</p>
    </div>
    """, unsafe_allow_html=True)

# ── 전체 선수 리스크 현황 테이블 ─────────────────────────────
st.html("<p style='font-size:1.05rem;font-weight:700;color:#1a1a1a;border-left:4px solid #1D9E75;padding-left:10px;margin:32px 0 4px;'>전체 선수 리스크 현황</p>")
st.caption("리스크 점수: 부정 댓글 가중 비율 × 100  |  🔴 위험 ≥ 60  /  🟡 주의 ≥ 35  /  🟢 안전 < 35")


def _badge(score: float) -> str:
    if score >= 60:
        return "<span style='background:#E24B4A;color:#fff;padding:3px 12px;border-radius:20px;font-size:0.75rem;font-weight:700;'>위험</span>"
    if score >= 35:
        return "<span style='background:#EF9F27;color:#fff;padding:3px 12px;border-radius:20px;font-size:0.75rem;font-weight:700;'>주의</span>"
    return "<span style='background:#1D9E75;color:#fff;padding:3px 12px;border-radius:20px;font-size:0.75rem;font-weight:700;'>안전</span>"


def _score_color(score: float) -> str:
    if score >= 60:
        return "#E24B4A"
    if score >= 35:
        return "#EF9F27"
    return "#1D9E75"


rows_html = ""
for i, row in risk_df.iterrows():
    score = row["리스크 점수"]
    color = _score_color(score)
    bg    = "#fff8f8" if score >= 60 else "#fffbf0" if score >= 35 else "#ffffff"
    rows_html += f"""
    <tr style='background:{bg};border-bottom:1px solid #f0f0f0;'>
      <td style='padding:12px 16px;font-weight:700;'>{row['선수명']}</td>
      <td style='padding:12px 16px;text-align:center;'>{row['총 댓글']:,}</td>
      <td style='padding:12px 16px;text-align:center;color:#1D9E75;font-weight:600;'>{row['긍정']}</td>
      <td style='padding:12px 16px;text-align:center;color:#E24B4A;font-weight:600;'>{row['부정']}</td>
      <td style='padding:12px 16px;text-align:center;'>
        <span style='color:{color};font-size:1.1rem;font-weight:900;'>{score}</span>
        <div style='margin:4px auto 0;width:80px;height:6px;background:#eee;border-radius:3px;'>
          <div style='width:{min(score,100):.0f}%;height:6px;background:{color};border-radius:3px;'></div>
        </div>
      </td>
      <td style='padding:12px 16px;text-align:center;'>{_badge(score)}</td>
    </tr>
    """

table_html = (
    "<div style='background:#fff;border-radius:14px;overflow:hidden;"
    "box-shadow:0 2px 12px rgba(0,0,0,0.08);margin-bottom:8px;'>"
    "<table style='width:100%;border-collapse:collapse;font-size:0.9rem;'>"
    "<thead><tr style='background:#0a2e1f;color:#fff;'>"
    "<th style='padding:13px 16px;text-align:left;font-weight:600;'>선수명</th>"
    "<th style='padding:13px 16px;text-align:center;font-weight:600;'>총 댓글</th>"
    "<th style='padding:13px 16px;text-align:center;font-weight:600;'>긍정</th>"
    "<th style='padding:13px 16px;text-align:center;font-weight:600;'>부정</th>"
    "<th style='padding:13px 16px;text-align:center;font-weight:600;'>리스크 점수</th>"
    "<th style='padding:13px 16px;text-align:center;font-weight:600;'>등급</th>"
    "</tr></thead>"
    f"<tbody>{rows_html}</tbody>"
    "</table></div>"
)
st.html(table_html)

# ── 최근 알람 발생 내역 ──────────────────────────────────────
st.html("<p style='font-size:1.05rem;font-weight:700;color:#1a1a1a;border-left:4px solid #1D9E75;padding-left:10px;margin:32px 0 4px;'>최근 알람 발생 내역</p>")

_alarm_msgs = [
    "부정 댓글 비율이 전일 대비 급증하여 리스크 경보가 발령됐습니다.",
    "SNS 언급량이 전일 대비 38% 증가하며 집중 모니터링이 시작됐습니다.",
    "커뮤니티 내 비판성 게시물이 집중 감지되어 알람이 트리거됐습니다.",
]
_alarm_dates = ["2026.05.29", "2026.05.28", "2026.05.27"]

for i, (_, row) in enumerate(risk_df.head(3).iterrows()):
    score    = row["리스크 점수"]
    severity = "🔴 위험" if score >= 60 else "🟡 주의"
    bg       = "#fff5f5" if score >= 60 else "#fffbf0"
    border   = "#E24B4A" if score >= 60 else "#EF9F27"
    st.html(
        f"<div style='background:{bg};border:1px solid {border}33;"
        f"border-left:5px solid {border};border-radius:10px;"
        f"padding:14px 18px;margin-bottom:10px;'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;'>"
        f"<span style='font-weight:700;font-size:0.95rem;'>{row['선수명']} — {severity}</span>"
        f"<span style='font-size:0.78rem;color:#aaa;'>{_alarm_dates[i]}</span>"
        f"</div>"
        f"<div style='font-size:0.85rem;color:#555;'>{_alarm_msgs[i]}</div>"
        f"<div style='margin-top:7px;font-size:0.78rem;color:#bbb;'>"
        f"리스크 점수: <b style='color:#333;'>{score}</b> / 100"
        f" &nbsp;·&nbsp; "
        f"부정 댓글: <b style='color:#E24B4A;'>{row['부정']}건</b>"
        f"</div></div>"
    )

# ── 하단 안내 ────────────────────────────────────────────────
st.html(
    "<div style='background:#f5fffa;border:1px solid #c3ead9;border-radius:10px;"
    "padding:14px 20px;margin-top:32px;'>"
    "<span style='font-size:0.85rem;color:#555;'>"
    "💡 <b>MPRA</b>는 한국어 특화 AI 감성 분석 엔진과 <b>Claude AI</b>를 결합하여 "
    "선수 여론 리스크를 조기에 감지하고 맞춤형 위기 대응 문구를 자동 생성합니다."
    "</span></div>"
)
