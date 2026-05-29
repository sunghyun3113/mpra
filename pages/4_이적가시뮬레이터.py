from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sentiment import analyze_dataframe

# ── 상수 ─────────────────────────────────────────────────────
PLAYERS = ["손흥민", "이강인", "김민재", "황희찬", "설영우"]

PLAYER_TEAMS: dict[str, str] = {
    "이강인": "PSG",
    "손흥민": "LAFC",
    "김민재": "바이에른 뮌헨",
    "황희찬": "울버햄튼",
    "설영우": "즈베즈다",
}

# Transfermarkt 2026년 기준 · 환율 1,650원/€ 적용
DEFAULT_VALUES: dict[str, int] = {
    "김민재":  528,   # €3,200만
    "이강인":  412,   # €2,500만
    "손흥민":  330,   # €2,000만
    "황희찬":  198,   # €1,200만
    "설영우":    7,   # €45만
}

# base_drop: 심각도 중간(4~6) 기준 이적가 하락 비율
ISSUE_PARAMS: dict[str, dict] = {
    "사생활 논란":            {"base_drop": 0.15, "desc": "개인 사생활 관련 미디어 노출 — 타격 가장 큼"},
    "팬과의 갈등":            {"base_drop": 0.10, "desc": "팬 커뮤니티와의 관계 악화"},
    "SNS 논란":               {"base_drop": 0.08, "desc": "소셜미디어 부적절 발언·행동"},
    "경기 부진 (3경기 연속)": {"base_drop": 0.06, "desc": "3경기 연속 저조한 경기력"},
    "부상 소식":              {"base_drop": 0.05, "desc": "부상으로 인한 출전 불가 예상 — 타격 가장 적음"},
}

# mitigation    : 초기 충격 방어율 (0~1)
# recovery_rate : 주당 회복 비율 (양수=회복, 음수=계속 하락)
# deepening     : 2주차 추가 하락 비율
RESPONSE_PARAMS: dict[str, dict] = {
    "즉시 공식 사과문 발표": {
        "mitigation": 0.70,   # 70% 방어 — 가장 효과적
        "recovery_rate": 0.55, # 1~2주 안에 빠른 회복
        "deepening": 0.00,
    },
    "선수 직접 해명 영상": {
        "mitigation": 0.55,   # 55% 방어
        "recovery_rate": 0.28, # 3~4주에 걸쳐 서서히 회복
        "deepening": 0.00,
    },
    "강경 대응 (법적 조치 예고)": {
        "mitigation": 0.30,   # 30% 방어
        "recovery_rate": 0.12, # 5~6주부터 느린 회복
        "deepening": 0.08,    # 초반 소폭 추가 하락
    },
    "조용히 무마 (대응 없음)": {
        "mitigation": 0.10,   # 10% 방어 — 가장 비효과적
        "recovery_rate": -0.08, # 음수 = 8주 내내 계속 하락
        "deepening": 0.15,    # 2주차 추가 악화
    },
}

RESPONSE_COLORS: dict[str, str] = {
    "즉시 공식 사과문 발표":       "#1565c0",
    "조용히 무마 (대응 없음)":     "#e53935",
    "강경 대응 (법적 조치 예고)":  "#f57c00",
    "선수 직접 해명 영상":          "#2e7d32",
}

WEEK_LABELS: list[str] = ["현재"] + [f"{w}주차" for w in range(1, 9)]

ISSUE_ADVICE: dict[str, str] = {
    "사생활 논란":            "사생활 이슈는 침묵보다 적극적 해명이 효과적입니다. 단, 불필요한 세부 정보 공개는 자제하세요.",
    "경기 부진 (3경기 연속)": "경기력 이슈는 미디어 대응보다 실제 퍼포먼스 회복이 최우선입니다. 훈련 영상 공개가 효과적입니다.",
    "팬과의 갈등":            "팬과의 직접 소통이 핵심입니다. 선수의 진정성 있는 메시지가 빠른 여론 진화로 이어집니다.",
    "SNS 논란":               "SNS 논란은 48시간 이내 대응하지 않으면 바이럴 확산됩니다. 신속 대응이 최우선입니다.",
    "부상 소식":              "정확한 의학 정보 공개와 복귀 일정 제시가 불확실성을 줄이고 이적가를 방어합니다.",
}

REFERENCE_CASES = [
    {
        "player": "이강인 (PSG)",      "issue": "팬과의 갈등",   "year": "2026.02",
        "response": "즉시 공식 사과문 발표",
        "before": "412억 원", "after": "402억 원", "change": "-2.5%", "defense": "76%",
        "result": "✅ 이적가 방어 성공",
        "bg": "#e3f2fd", "border": "#1565c0",
    },
    {
        "player": "손흥민 (LAFC)",     "issue": "은퇴 루머 확산", "year": "2026.03",
        "response": "선수 직접 해명 영상",
        "before": "330억 원", "after": "322억 원", "change": "-2.3%", "defense": "82%",
        "result": "✅ 신속 대응으로 방어",
        "bg": "#e8f5e9", "border": "#2e7d32",
    },
    {
        "player": "김민재 (바이에른)", "issue": "부상 소식",      "year": "2026.04",
        "response": "조용히 무마 (대응 없음)",
        "before": "528억 원", "after": "443억 원", "change": "-16.1%", "defense": "0%",
        "result": "🔴 미대응으로 이적가 급락",
        "bg": "#ffebee", "border": "#e53935",
    },
]

# 선수별 주요 이슈 타임라인 이벤트 (2026년 가상 시나리오)
PLAYER_EVENTS: dict[str, list[dict]] = {
    "손흥민": [
        {"date": "2026-01-20", "label": "LAFC 이적 후 기량 하락 논란",   "type": "issue"},
        {"date": "2026-02-08", "label": "MLS 적응 완료 선언 인터뷰",     "type": "response"},
        {"date": "2026-03-12", "label": "LAFC 연속 골·어시스트 여론 회복", "type": "recovery"},
    ],
    "이강인": [
        {"date": "2026-01-18", "label": "SNS 논란 발생",               "type": "issue"},
        {"date": "2026-02-03", "label": "공식 사과문 발표",             "type": "response"},
        {"date": "2026-03-25", "label": "PSG 주전 입지 강화 여론 호전", "type": "recovery"},
    ],
    "김민재": [
        {"date": "2026-02-10", "label": "부상 소식 공개",                   "type": "issue"},
        {"date": "2026-03-01", "label": "복귀 일정 공식 발표",              "type": "response"},
        {"date": "2026-04-15", "label": "복귀 후 안정된 수비 여론 회복",    "type": "recovery"},
    ],
    "황희찬": [
        {"date": "2026-02-18", "label": "빅클럽 이적 협상 결렬 루머", "type": "issue"},
        {"date": "2026-03-10", "label": "팀 잔류 의사 공식 표명",    "type": "response"},
        {"date": "2026-04-22", "label": "4경기 연속 골 여론 상승",    "type": "recovery"},
    ],
    "설영우": [
        {"date": "2026-02-15", "label": "즈베즈다 리그 수준 논란",              "type": "issue"},
        {"date": "2026-03-05", "label": "챔피언스리그 예선 인상적 활약 공개",   "type": "response"},
        {"date": "2026-05-05", "label": "세르비아 올시즌 베스트 풀백 선정",     "type": "recovery"},
    ],
}


@st.cache_data
def load_data(_mtime: float = 0.0) -> pd.DataFrame:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return pd.read_csv(os.path.join(base, "data", "sample_data.csv"), encoding="utf-8-sig")


@st.cache_data
def get_full_analyzed(df: pd.DataFrame) -> pd.DataFrame:
    """전체 댓글 일괄 분석 — 세션당 1회."""
    return analyze_dataframe(df)


# ── 시뮬레이션 ────────────────────────────────────────────────
def simulate_value(
    base: float,
    issue: str,
    response: str,
    severity: int,
) -> list[float]:
    """
    이슈·대응·심각도를 바탕으로 8주 이적가 변화 시뮬레이션.

    심각도 배수:
      - 1~3 (경미): ×0.5
      - 4~6 (보통): ×1.0
      - 7~10 (심각): ×1.5

    회복 모델:
      - 1주차: initial_impact 즉시 하락
      - 2주차: peak (= initial × (1 + deepening))
      - 3~8주: peak × (1 - recovery_rate)^(w-2)
               recovery_rate < 0 이면 계속 하락
    """
    ip = ISSUE_PARAMS[issue]
    rp = RESPONSE_PARAMS[response]

    if severity <= 3:
        sev_scale = 0.5
    elif severity <= 6:
        sev_scale = 1.0
    else:
        sev_scale = 1.5

    max_drop       = ip["base_drop"] * sev_scale
    initial_impact = max_drop * (1.0 - rp["mitigation"])
    recovery       = rp["recovery_rate"]
    peak_impact    = initial_impact * (1.0 + rp["deepening"])

    values: list[float] = [round(base, 2)]
    for w in range(1, 9):
        if w == 1:
            impact = initial_impact
        elif w == 2:
            impact = peak_impact
        else:
            impact = peak_impact * ((1.0 - recovery) ** (w - 2))
        val = base * (1.0 - impact)
        values.append(round(max(val, base * 0.20), 2))  # 최저 20% 방어선

    return values


def simulate_all(base: float, issue: str, severity: int) -> dict[str, list[float]]:
    return {r: simulate_value(base, issue, r, severity) for r in RESPONSE_PARAMS}


def calc_defense_rate(
    base: float,
    selected_vals: list[float],
    worst_vals: list[float],
) -> float:
    """최악 시나리오 대비 이적가 손실 방어율 (0~100 %)."""
    potential_loss = base - worst_vals[-1]
    if potential_loss < 0.01:          # 실질 손실 없으면 100 %
        return 100.0
    rate = (selected_vals[-1] - worst_vals[-1]) / potential_loss * 100.0
    return round(min(max(rate, 0.0), 100.0), 1)


def get_ai_strategy(
    issue: str,
    severity: int,
    selected: str,
    base: float,
    selected_vals: list[float],
    worst_vals: list[float],
) -> str:
    worst_loss_pct = round((base - worst_vals[-1]) / base * 100, 1)
    sel_loss_pct   = round((base - selected_vals[-1]) / base * 100, 1)
    defended_pct   = round(worst_loss_pct - sel_loss_pct, 1)

    if severity <= 3:
        sev_tag, sev_label = "🟢 저심각도", "경미한 이슈"
    elif severity <= 6:
        sev_tag, sev_label = "🟡 중심각도", "중간 수준 이슈"
    else:
        sev_tag, sev_label = "🔴 고심각도", "심각한 이슈"

    lines = [f"**{sev_tag} — {issue} | 심각도 {severity}/10 — 대응 전략 분석**\n"]

    # 무대응 선택 시 경고
    if selected == "조용히 무마 (대응 없음)":
        lines.append(
            f"🚨 **즉각 대응하지 않으면 이적가의 최대 {worst_loss_pct:.1f}% 손실이 예상됩니다.**"
        )
        if severity >= 7:
            lines.append(
                f"⚡ **48시간 이내 공식 입장 발표를 강력히 권고합니다.** "
                f"무대응 지속 시 8주 후 {worst_loss_pct:.1f}% 이상 하락이 불가피합니다."
            )
        else:
            lines.append(
                f"현재 {sev_label}이나, 방치 시 여론 악화로 손실이 커질 수 있습니다. "
                f"최소한 단기 모니터링 강화를 권장합니다."
            )
    # 강경 대응 시
    elif selected == "강경 대응 (법적 조치 예고)":
        lines.append(
            f"⚠️ 강경 대응은 초반 여론 악화를 수반합니다. "
            f"이적가는 단기적으로 추가 하락 후 5~6주부터 회복이 시작됩니다."
        )
        lines.append(
            f"무대응 대비 **{defended_pct:.1f}%p 방어** 효과가 있으나, "
            f"'즉시 공식 사과문'으로 전환 시 더 빠른 회복이 가능합니다."
        )
    # 선수 해명 선택 시
    elif selected == "선수 직접 해명 영상":
        lines.append(
            f"✅ 선수 직접 해명은 팬 신뢰 회복에 효과적입니다. "
            f"3~4주에 걸쳐 이적가가 서서히 회복됩니다."
        )
        lines.append(
            f"무대응 대비 **{defended_pct:.1f}%p 방어** 달성 예상. "
            f"해명 영상은 48시간 이내 공개해야 최대 효과를 냅니다."
        )
    # 즉시 사과 선택 시
    else:
        if severity <= 3:
            lines.append(
                f"✅ {sev_label}로 적극적 대응 시 이적가 영향은 미미합니다. "
                f"언론 노출을 최소화하며 경기력으로 증명하는 전략을 권장합니다."
            )
        else:
            lines.append(
                f"✅ 가장 효과적인 대응입니다. 1~2주 안에 빠른 회복이 예상됩니다."
            )
            lines.append(
                f"무대응 대비 **{defended_pct:.1f}%p 방어** — "
                f"이적가 손실을 {sel_loss_pct:.1f}% 수준으로 최소화합니다."
            )

    # 이슈별 추가 조언
    lines.append(f"\n📌 **{issue} 전문 조언**: {ISSUE_ADVICE.get(issue, '')}")

    # 고심각도 긴급 경고
    if severity >= 8 and selected != "즉시 공식 사과문 발표":
        lines.append(
            f"\n⚡ **긴급 권고**: 심각도 {severity}/10 수준에서는 "
            f"**'즉시 공식 사과문 발표'** 가 최선입니다. "
            f"현재 전략보다 추가로 {max(0, round(defended_pct * 0.4, 1))}%p 방어 가능합니다."
        )

    return "\n".join(lines)


# ── 페이지 설정 ───────────────────────────────────────────────
st.set_page_config(
    page_title="이적가 시뮬레이터",
    page_icon="🔄",
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
        💰 이적가 방어 시뮬레이터
    </div>
    <div style="font-size: 0.85rem; color: #888; margin-top: 4px;">
        위기 대응 시나리오별 이적가 변화를 예측합니다
    </div>
</div>
""")

# ── 메인 레이아웃 (좌 1 : 우 2) ──────────────────────────────
col_left, col_right = st.columns([1, 2], gap="large")

# ╔═══════════════════════════════╗
# ║        좌측 입력 패널          ║
# ╚═══════════════════════════════╝
with col_left:
    st.subheader("⚙️ 시뮬레이션 조건 설정")

    player = st.selectbox(
        "선수 선택", PLAYERS, index=1,
        format_func=lambda p: f"{p} ({PLAYER_TEAMS[p]})",
    )

    # Fix 1: key=f"base_value_{player}" → 선수 변경 시 위젯 상태 리셋,
    #         올바른 기본 이적가로 자동 갱신
    base_value: float = float(
        st.number_input(
            "현재 이적가 (억 원)",
            min_value=1,
            max_value=1000,
            value=DEFAULT_VALUES[player],
            step=5,
            format="%d",
            key=f"base_value_{player}",
        )
    )

    st.markdown("---")

    issue_type: str = st.selectbox(
        "이슈 유형",
        list(ISSUE_PARAMS.keys()),
        index=1,   # 기본: 팬과의 갈등
    )
    st.caption(f"ℹ️ {ISSUE_PARAMS[issue_type]['desc']}")

    response_type: str = st.selectbox(
        "구단 대응 방식",
        list(RESPONSE_PARAMS.keys()),
        index=0,
    )

    st.markdown("---")

    severity: int = st.slider(
        "이슈 심각도",
        min_value=1,
        max_value=10,
        value=8,
        help="1=경미 / 5=보통 / 10=매우 심각",
    )
    sev_tag = "🔴 매우 심각" if severity >= 8 else "🟡 중간 수준" if severity >= 5 else "🟢 경미한 수준"
    st.markdown(f"현재 심각도: **{sev_tag}** ({severity}/10)")

    st.markdown("---")
    st.markdown("**📌 파라미터 요약**")

    ip = ISSUE_PARAMS[issue_type]
    rp = RESPONSE_PARAMS[response_type]
    if severity <= 3:
        sev_scale_val = 0.5
    elif severity <= 6:
        sev_scale_val = 1.0
    else:
        sev_scale_val = 1.5
    effective_drop = ip["base_drop"] * sev_scale_val * (1.0 - rp["mitigation"])
    param_df = pd.DataFrame(
        {
            "항목": ["이슈 기본 하락폭", "심각도 배수", "방어율", "실질 예상 하락"],
            "값":   [
                f"{int(ip['base_drop'] * 100)}%",
                f"×{sev_scale_val}",
                f"{int(rp['mitigation'] * 100)}%",
                f"≈{effective_drop * 100:.1f}%",
            ],
        }
    )
    st.dataframe(param_df, use_container_width=True, hide_index=True)


# ╔═══════════════════════════════╗
# ║        우측 결과 패널          ║
# ╚═══════════════════════════════╝
with col_right:
    st.subheader("📈 시뮬레이션 결과")

    scenarios = simulate_all(base_value, issue_type, severity)

    worst_key    = "조용히 무마 (대응 없음)"
    selected_vals = scenarios[response_type]
    worst_vals    = scenarios[worst_key]
    best_key      = max(scenarios, key=lambda k: scenarios[k][-1])
    best_vals     = scenarios[best_key]

    defense_rate = calc_defense_rate(base_value, selected_vals, worst_vals)

    # ── 핵심 지표 계산 ───────────────────────────────────────
    final_val      = selected_vals[-1]
    val_change     = round(final_val - base_value, 2)
    pct_change     = round(val_change / base_value * 100, 1)
    min_val        = min(selected_vals)
    min_pct        = round((min_val - base_value) / base_value * 100, 1)
    worst_loss     = round(base_value - worst_vals[-1], 0)    # 무대응 시 손실액
    defended_amt   = round(final_val - worst_vals[-1], 0)     # 방어한 금액

    # ── 방어 금액 강조 배너 ──────────────────────────────────
    if response_type != "조용히 무마 (대응 없음)" and defended_amt > 0:
        st.html(
            f"<div style='background:linear-gradient(135deg,#1b5e20,#2e7d32);"
            f"border-radius:12px;padding:18px 24px;text-align:center;margin-bottom:16px;'>"
            f"<div style='font-size:0.85rem;color:#a8f0d9;margin-bottom:4px;'>이 대응으로 지킨 금액</div>"
            f"<div style='font-size:2rem;font-weight:900;color:#fff;'>"
            f"{defended_amt:,.0f}억 원을 지켰습니다 ✅</div>"
            f"<div style='font-size:0.8rem;color:#c8e6c9;margin-top:6px;'>"
            f"무대응 시 예상 손실: {worst_loss:,.0f}억 원 → 실제 손실: {abs(val_change):,.0f}억 원</div>"
            f"</div>"
        )
    else:
        st.html(
            f"<div style='background:#b71c1c;border-radius:12px;padding:18px 24px;"
            f"text-align:center;margin-bottom:16px;'>"
            f"<div style='font-size:0.85rem;color:#ffcdd2;margin-bottom:4px;'>무대응 선택 — 예상 손실액</div>"
            f"<div style='font-size:2rem;font-weight:900;color:#fff;'>"
            f"{worst_loss:,.0f}억 원 손실 예상 🚨</div>"
            f"<div style='font-size:0.8rem;color:#ffcdd2;margin-top:6px;'>"
            f"즉시 공식 사과문 발표로 전환 시 최대 {worst_loss * 0.7:,.0f}억 원 방어 가능</div>"
            f"</div>"
        )

    # ── 핵심 지표 3개 ────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "📊 8주 후 이적가",
        f"{final_val:,.0f}억 원",
        delta=f"{val_change:+,.0f}억 ({pct_change:+.1f}%)",
        delta_color="normal",
    )
    m2.metric(
        "🛡️ 이적가 방어율",
        f"{defense_rate:.1f}%",
        help="최악 시나리오(무대응) 대비 방어된 손실 비율",
    )
    m3.metric(
        "📉 최대 하락 시점",
        f"{min_val:,.0f}억 원",
        delta=f"{min_pct:+.1f}%",
        delta_color="normal",
    )

    st.markdown("---")

    # ── 시나리오 비교 라인 차트 ───────────────────────────────
    fig = go.Figure()

    for resp_name, vals in scenarios.items():
        is_sel   = resp_name == response_type
        is_worst = resp_name == worst_key
        is_best  = resp_name == best_key

        # 레이블 결정
        if is_sel and is_best:
            label = resp_name + " ★선택(최선)"
        elif is_sel and is_worst:
            label = resp_name + " ★선택(최악)"
        elif is_sel:
            label = resp_name + " ★ 선택"
        elif is_best:
            label = resp_name + " (최선)"
        elif is_worst:
            label = resp_name + " (최악)"
        else:
            label = resp_name

        # 선 스타일 결정
        if is_sel:
            line = dict(color=RESPONSE_COLORS[resp_name], width=3.5, dash="solid")
            opacity, marker_sz = 1.0, 7
        elif is_worst:
            line = dict(color="#e53935", width=2.0, dash="dash")
            opacity, marker_sz = 0.9, 5
        elif is_best:
            line = dict(color="#2e7d32", width=2.0, dash="solid")
            opacity, marker_sz = 0.9, 5
        else:
            line = dict(color="#bdbdbd", width=1.5, dash="dot")
            opacity, marker_sz = 0.6, 4

        fig.add_trace(go.Scatter(
            x=WEEK_LABELS,
            y=vals,
            mode="lines+markers",
            name=label,
            line=line,
            opacity=opacity,
            marker=dict(size=marker_sz),
            hovertemplate="%{x}: %{y:,.0f}억 원<extra>" + resp_name + "</extra>",
        ))

    # 현재가 기준선
    fig.add_hline(
        y=base_value,
        line_dash="dot", line_color="#9e9e9e", line_width=1,
        annotation_text=f"현재 {base_value:,.0f}억 원",
        annotation_position="top left",
        annotation_font_size=10,
    )

    fig.update_layout(
        xaxis_title="대응 후 경과 기간",
        yaxis_title="이적가 (억 원)",
        yaxis=dict(gridcolor="#e0e0e0"),
        legend=dict(
            orientation="v",
            yanchor="bottom", y=0.01,
            xanchor="right",  x=0.99,
            bgcolor="rgba(255,255,255,0.85)",
            font=dict(size=10),
        ),
        hovermode="x unified",
        height=360,
        margin=dict(t=10, b=40, l=60, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_layout(paper_bgcolor='white', plot_bgcolor='#F8F9FA', font=dict(family='sans-serif', color='#2C2C2C'), margin=dict(t=40, b=40, l=40, r=40))
    st.plotly_chart(fig, use_container_width=True)

    # ── 최선/최악/선택 비교 테이블 ───────────────────────────
    st.subheader("📋 시나리오 비교 테이블")

    table_rows = []
    scenarios_to_compare = [
        ("🟢 최선", best_key,      best_vals),
        ("🔴 최악", worst_key,     worst_vals),
        ("➡️ 선택", response_type, selected_vals),
    ]
    for tag, key, vals in scenarios_to_compare:
        v8   = vals[-1]
        chg  = round(v8 - base_value, 2)
        pct  = round(chg / base_value * 100, 1)
        mv   = round(min(vals), 2)
        mvp  = round((mv - base_value) / base_value * 100, 1)
        dr   = calc_defense_rate(base_value, vals, worst_vals)
        table_rows.append({
            "구분":          tag,
            "대응 방식":     key,
            "8주 후 이적가": f"{v8:,.0f}억 원",
            "변화율":        f"{pct:+.1f}%",
            "최저 시점":     f"{mv:,.0f}억 원 ({mvp:+.1f}%)",
            "방어율":        f"{dr:.1f}%",
        })

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # ── AI 권장 대응 전략 ─────────────────────────────────────
    st.markdown("---")
    st.subheader("🤖 AI 권장 대응 전략")
    st.markdown(get_ai_strategy(
        issue_type, severity, response_type,
        base_value, selected_vals, worst_vals,
    ))


# ── 하단: 실제 사례 참고 카드 ─────────────────────────────────
st.markdown("---")
st.subheader("📚 실제 사례 참고 (2026년 기준 가상 시나리오)")

case_cols = st.columns(3)
for i, case in enumerate(REFERENCE_CASES):
    bg     = case["bg"]
    border = case["border"]
    with case_cols[i]:
        # f-string 내 단일 따옴표 dict 접근은 Python 3.8+ 에서 유효
        st.markdown(
            f"<div style='background:{bg};border-left:5px solid {border};"
            f"border-radius:8px;padding:14px 16px;'>"
            f"<b>⚽ {case['player']}</b>&nbsp;"
            f"<span style='font-size:0.8rem;color:#666;'>({case['year']})</span><br><br>"
            f"<b>📌 이슈:</b> {case['issue']}<br>"
            f"<b>💬 대응:</b> {case['response']}<br>"
            f"<b>💰 이적가:</b> {case['before']} → {case['after']}"
            f"&nbsp;<span style='color:#666;'>({case['change']})</span><br>"
            f"<b>🛡️ 방어율:</b> {case['defense']}<br><br>"
            f"<b>{case['result']}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ── 여론 타임라인 섹션 ────────────────────────────────────────
st.markdown("---")
st.subheader(f"📅 {player} ({PLAYER_TEAMS[player]}) — 여론 타임라인 (2026년)")
st.caption(
    "개별 댓글 감성 점수와 14일 이동평균으로 여론 흐름 표시  |  "
    "⚠️ 이슈 발생  💬 공식 대응  ✅ 여론 회복"
)

# 데이터 로딩 + 분석 (캐시 사용 — 첫 실행만 느림)
_df_raw = load_data(_mtime=os.path.getmtime(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_data.csv")))
with st.spinner(f"🔄 {player} 여론 타임라인 분석 중..."):
    _df_analyzed = get_full_analyzed(_df_raw)

_pdf = _df_analyzed[_df_analyzed["player_name"] == player].copy()

if _pdf.empty:
    st.info("해당 선수의 타임라인 데이터가 없습니다.")
else:
    # ── 감성 점수 계산 ─────────────────────────────────────────
    # 긍정=+confidence, 부정=-confidence, 중립=0
    def _to_score(row: pd.Series) -> float:
        if row["sentiment"] == "긍정":
            return float(row["confidence"])
        if row["sentiment"] == "부정":
            return -float(row["confidence"])
        return 0.0

    _pdf["score"]   = _pdf.apply(_to_score, axis=1)
    _pdf["date_dt"] = pd.to_datetime(_pdf["date"])
    _pdf = _pdf.sort_values("date_dt").reset_index(drop=True)

    # ── 14일 이동평균 계산 ─────────────────────────────────────
    _full_range = pd.date_range("2026-01-01", "2026-05-29", freq="D")
    _daily_raw  = _pdf.groupby("date_dt")["score"].mean()
    _daily_full = _daily_raw.reindex(_full_range, fill_value=float("nan"))

    # nan 채움 순서: 롤링 → ffill → bfill → 0 (pandas 2.0+ 대응)
    _smooth = (
        _daily_full
        .rolling(window=14, min_periods=1, center=True)
        .mean()
        .ffill()
        .bfill()
        .fillna(0.0)
    )
    _smooth_df = pd.DataFrame({"date": _full_range, "smooth": _smooth.values})

    # ── 차트 ───────────────────────────────────────────────────
    _tl = go.Figure()

    # 긍정 영역 (연초록 fill)
    _tl.add_trace(go.Scatter(
        x=_smooth_df["date"],
        y=_smooth_df["smooth"].clip(lower=0),
        fill="tozeroy",
        fillcolor="rgba(76,175,80,0.20)",
        line=dict(color="rgba(0,0,0,0)"),
        name="긍정 여론",
        hoverinfo="skip",
    ))
    # 부정 영역 (연빨강 fill)
    _tl.add_trace(go.Scatter(
        x=_smooth_df["date"],
        y=_smooth_df["smooth"].clip(upper=0),
        fill="tozeroy",
        fillcolor="rgba(244,67,54,0.20)",
        line=dict(color="rgba(0,0,0,0)"),
        name="부정 여론",
        hoverinfo="skip",
    ))
    # 이동평균 메인 라인
    _tl.add_trace(go.Scatter(
        x=_smooth_df["date"],
        y=_smooth_df["smooth"],
        mode="lines",
        line=dict(color="#1565c0", width=2.5),
        name="여론 지수 (14일 이동평균)",
        hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra>여론 지수</extra>",
    ))
    # 개별 댓글 점수 dot
    _color_map = {"긍정": "#4caf50", "부정": "#f44336"}
    _dot_colors = _pdf["sentiment"].map(_color_map).fillna("#bdbdbd").tolist()
    _tl.add_trace(go.Scatter(
        x=_pdf["date_dt"],
        y=_pdf["score"],
        mode="markers",
        marker=dict(size=7, color=_dot_colors, opacity=0.75),
        name="개별 댓글",
        hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra>개별 댓글</extra>",
    ))

    # ── 주요 이벤트 마커 ───────────────────────────────────────
    _ev_style = {
        "issue":    {"color": "#e53935", "dash": "dash",  "icon": "⚠️"},
        "response": {"color": "#1565c0", "dash": "solid", "icon": "💬"},
        "recovery": {"color": "#2e7d32", "dash": "solid", "icon": "✅"},
    }
    for _idx, _ev in enumerate(PLAYER_EVENTS.get(player, [])):
        _es    = _ev_style[_ev["type"]]
        _lbl   = _es["icon"] + " " + _ev["label"]
        _x_str = str(_ev["date"])          # 문자열 날짜 그대로 사용
        _at_top = (_idx % 2 == 0)          # 짝수=위, 홀수=아래 (겹침 방지)

        # add_vline 대신 add_shape + add_annotation 으로 직접 그리기
        # → add_vline 은 내부에서 annotation offset 을 x값에 정수로 더하려다
        #   datetime 축 문자열과 충돌해 TypeError 발생하므로 사용 안 함

        # 수직선
        _tl.add_shape(
            type="line",
            x0=_x_str, x1=_x_str,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color=_es["color"], width=1.5, dash=_es["dash"]),
        )
        # 텍스트 어노테이션
        _tl.add_annotation(
            x=_x_str,
            y=1.0 if _at_top else 0.0,
            xref="x",
            yref="paper",
            text=_lbl,
            showarrow=False,
            font=dict(size=9, color=_es["color"]),
            yanchor="bottom" if _at_top else "top",
            xanchor="left",
            bgcolor="rgba(255,255,255,0.75)",
            borderpad=2,
        )

    _tl.update_layout(
        xaxis=dict(
            title="날짜 (2026년)",
            tickformat="%m월",
            gridcolor="#e0e0e0",
            dtick="M1",
        ),
        yaxis=dict(
            title="여론 지수",
            range=[-1.1, 1.1],
            gridcolor="#e0e0e0",
            zeroline=True,
            zerolinecolor="#9e9e9e",
            zerolinewidth=1.5,
            tickvals=[-1.0, -0.5, 0.0, 0.5, 1.0],
            ticktext=["-1.0 (부정)", "-0.5", "0", "+0.5", "+1.0 (긍정)"],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right",  x=1.0,
        ),
        hovermode="x unified",
        height=420,
        margin=dict(t=60, b=50, l=90, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    _tl.update_xaxes(showgrid=False)
    _tl.update_layout(paper_bgcolor='white', plot_bgcolor='#F8F9FA', font=dict(family='sans-serif', color='#2C2C2C'), margin=dict(t=40, b=40, l=40, r=40))
    st.plotly_chart(_tl, use_container_width=True)

    st.caption(
        "**⚠️ 이슈 발생** (빨강 점선) &nbsp;&nbsp; "
        "**💬 공식 대응** (파랑 실선) &nbsp;&nbsp; "
        "**✅ 여론 회복** (초록 실선) &nbsp;&nbsp;|&nbsp;&nbsp; "
        "점: 개별 댓글 감성 점수 (💚긍정 / 🔴부정)"
    )

st.markdown("---")
st.caption("※ 이적가 기준: Transfermarkt 2026년 데이터 · 환율 1,650원/€ 적용")
