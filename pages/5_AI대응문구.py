from __future__ import annotations

import io
import os
import sys
import time
from xml.sax.saxutils import escape as _xml_escape

import pandas as pd
import streamlit as st

try:
    from reportlab.lib import colors as _rlc
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )
    _REPORTLAB_OK = True
except ImportError:
    _REPORTLAB_OK = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sentiment import analyze_dataframe, calc_risk_score

# ── 상수 ─────────────────────────────────────────────────────
PLAYERS        = ["손흥민", "이강인", "김민재", "황희찬", "설영우"]
RESPONSE_TYPES = ["공식 구단 성명서", "선수 개인 SNS 게시물", "팬 커뮤니티 공지", "언론 인터뷰 멘트"]
TONES          = ["공식적", "친근한", "사과 중심", "적극 해명"]

# ── 템플릿 구성 요소 ──────────────────────────────────────────
# 각 tone × 3 버전 (v0/v1/v2) 오프닝 문장
_OPENINGS: dict[str, list[str]] = {
    "공식적": [
        "{player} 선수와 관련된 {sit}에 대해 구단 공식 입장을 다음과 같이 밝힙니다.",
        "{player} 선수 관련 사안에 대해 구단은 아래와 같이 성명을 발표합니다.",
        "최근 {player} 선수 관련 {sit}에 대한 각계의 우려를 인지하고 공식 입장을 전달합니다.",
    ],
    "친근한": [
        "안녕하세요, {player}를 응원하는 팬 여러분! 최근 {sit}에 대해 직접 소식을 전합니다 😊",
        "팬 여러분, 반갑습니다! {player} 관련해서 솔직하게 말씀드릴 내용이 있어요.",
        "안녕하세요! {player} 관련 {sit}으로 걱정하시는 분들을 위해 공식 입장을 전합니다 💚",
    ],
    "사과 중심": [
        "먼저, {player} 선수 관련 {sit}으로 팬 여러분께 심려를 끼쳐 드린 점 진심으로 사과드립니다.",
        "{sit}으로 인해 팬 여러분의 마음에 상처를 드린 {player} 선수와 구단은 깊이 반성합니다.",
        "팬 여러분께 가장 먼저 사과드립니다. {player} 선수 관련 {sit}에 대해 진심으로 죄송합니다.",
    ],
    "적극 해명": [
        "{player} 선수 관련 {sit}과 관련하여 잘못된 정보가 퍼지고 있어 사실 관계를 명확히 밝힙니다.",
        "최근 {sit}을 둘러싼 오해가 커지고 있어 {player} 측 공식 해명을 진행합니다.",
        "{player} 선수 관련 허위 정보 확산에 대해 구단은 즉각적이고 명확한 해명에 나섭니다.",
    ],
}

# 각 response_type × 3 버전 본문
_BODIES: dict[str, list[str]] = {
    "공식 구단 성명서": [
        "구단은 해당 사안을 매우 엄중히 받아들이고 있으며, {player} 선수와 충분한 소통을 통해 상황을 "
        "파악하고 있습니다. 구단은 선수 개인의 인격 보호를 최우선으로 하며, 필요한 모든 지원을 아끼지 "
        "않을 것을 약속드립니다.\n\n"
        "구단은 이번 사안을 계기로 선수 관리 및 커뮤니케이션 체계를 더욱 강화하겠습니다.",

        "구단은 {player} 선수를 중심으로 팀 내 긴밀한 소통을 통해 이 상황에 대처하고 있습니다. "
        "사안의 전말을 파악하고 있으며, 투명한 소통을 통해 팬 여러분과 함께 해결책을 찾아나가겠습니다.\n\n"
        "구단은 {player} 선수가 최고의 경기력을 발휘할 수 있는 환경 조성에 집중하겠습니다.",

        "구단은 외부의 부정적 여론과 무관하게 {player} 선수의 전문성과 헌신을 높이 평가합니다. "
        "구단은 선수와 함께 이 상황을 극복해 나갈 것이며, 팬 여러분의 지속적인 응원을 기다립니다.\n\n"
        "이번 사안에 대한 체계적인 대응 방안을 수립하여 실행에 옮길 것입니다.",
    ],
    "선수 개인 SNS 게시물": [
        "여러분의 걱정과 응원 메시지 정말 감사합니다. 쉽지 않은 상황이지만 여러분의 따뜻한 마음 덕분에 "
        "하루하루 버텨나갈 수 있습니다.\n\n"
        "저는 지금 이 상황을 정면으로 받아들이고 있으며, 앞으로 행동으로 신뢰를 보여드리겠습니다. "
        "계속 응원해 주세요 🙏",

        "팬 여러분께 직접 말씀드리고 싶었습니다. 지금 많이 힘들지만, 여러분의 응원이 저에게 큰 힘이 "
        "됩니다.\n\n"
        "좋은 선수이기 전에, 좋은 사람이 되도록 노력하겠습니다. 실망시켜 드린 부분이 있다면 진심으로 "
        "사과드리며, 그라운드에서 결과로 보여드리겠습니다 ⚽",

        "솔직하게 말씀드릴게요. 이번 상황이 저에게도 쉽지 않았습니다. 하지만 팬 여러분의 응원을 보면서 "
        "다시 마음을 다잡게 됩니다.\n\n"
        "앞으로는 더 성숙한 모습으로, 더 좋은 경기력으로 보답하겠습니다. 항상 함께해 주셔서 감사합니다 💪",
    ],
    "팬 커뮤니티 공지": [
        "[공지] 운영진은 최근 {player} 선수 관련 {sit}으로 커뮤니티 내 논란이 증가하고 있는 것을 "
        "인지하고 있습니다.\n\n"
        "운영진은 구단 공식 입장 확인 후, 미확인 정보 확산을 최소화하기 위한 모니터링을 강화합니다. "
        "건전한 토론 문화 유지를 위해 회원 여러분의 협조를 부탁드립니다.",

        "안녕하세요, 운영진입니다. {player} 선수 관련 많은 게시글과 댓글이 올라오고 있어 운영진 차원의 "
        "공식 입장을 공유합니다.\n\n"
        "구단 공식 발표 기반의 정보만 공유해 주시고, 미확인 루머 확산 방지에 협조해 주세요. "
        "커뮤니티 여러분의 따뜻한 응원이 선수에게 직접 전달됩니다 💚",

        "[팬 커뮤니티 공식 공지]\n\n"
        "{player} 선수 관련 상황을 지켜보며 걱정하시는 회원 분들이 많다는 것 잘 알고 있습니다. "
        "운영진은 구단과 협력하여 정확한 정보를 전달하기 위해 노력하고 있습니다.\n\n"
        "사실 확인이 되지 않은 내용의 무분별한 확산은 자제해 주시고, 어려운 시기일수록 선수를 "
        "응원하는 목소리를 높여주세요!",
    ],
    "언론 인터뷰 멘트": [
        "질문해 주셔서 감사합니다. {player} 선수 관련 이번 사안에 대해 구단은 매우 진지하게 "
        "받아들이고 있습니다.\n\n"
        "현재 사실 관계를 파악하는 과정에 있으며, 확인이 완료되는 대로 공식적인 입장을 발표할 "
        "예정입니다. 현 시점에서의 추가 발언은 자제하겠습니다.",

        "이번 사안과 관련해서 말씀드리겠습니다. {player} 선수는 구단의 핵심 자산이며, "
        "구단은 선수를 전적으로 지지하는 입장입니다.\n\n"
        "다만 현재 상황이 진행 중인 만큼, 성급한 판단보다는 충분한 사실 확인 후 공식 입장을 "
        "발표하겠습니다. 보도에 있어 신중을 기해 주시기 바랍니다.",

        "{player} 선수와 관련된 질문에 답변 드립니다. 구단은 이번 사안의 심각성을 충분히 "
        "인지하고 있으며, 내부적으로 철저한 검토를 진행 중입니다.\n\n"
        "선수의 명예와 팬 여러분의 신뢰를 지키기 위해 최선을 다하겠습니다. "
        "앞으로의 공식 발표를 통해 투명하게 소통하겠습니다. 이상입니다.",
    ],
}

# 각 tone × 3 버전 클로징 문장
_CLOSINGS: dict[str, list[str]] = {
    "공식적": [
        "구단은 앞으로도 팬 여러분과 투명하게 소통하며 신뢰를 쌓아나가겠습니다.\n[2026년 구단 공식 발표]",
        "팬 여러분의 이해와 지속적인 성원에 깊이 감사드립니다.\n[2026년 구단 홍보팀]",
        "앞으로의 상황 변화에 대해 공식 채널을 통해 지속 소통하겠습니다.\n[2026년]",
    ],
    "친근한": [
        "언제나 응원해 주시는 팬 여러분, 진심으로 감사드립니다! 앞으로도 좋은 소식으로 찾아올게요 😊\n[2026년]",
        "팬 여러분의 응원이 {player}에게 가장 큰 힘이 된답니다! 앞으로도 많이 응원해 주세요 💚\n[2026년]",
        "항상 함께해 주시는 팬 여러분께 정말 감사드려요. 좋은 모습으로 보답할게요! 🙏\n[2026년]",
    ],
    "사과 중심": [
        "다시 한번 진심으로 사과드립니다. 앞으로의 행동으로 신뢰를 회복하겠습니다.\n[2026년 구단]",
        "팬 여러분의 실망에 진심으로 사과드리며, 재발 방지를 위해 최선을 다하겠습니다.\n[2026년]",
        "이번 상황에 대한 책임을 통감하며 더 나은 모습으로 거듭나겠습니다. 진심으로 죄송합니다.\n[2026년 구단]",
    ],
    "적극 해명": [
        "사실에 기반한 정보만을 신뢰하며, 허위 정보에 대해서는 강력히 대응할 것입니다.\n[2026년 구단 공식 해명]",
        "정확한 사실이 왜곡되는 일이 없도록 구단은 모든 역량을 총동원하겠습니다.\n[2026년]",
        "사실에 근거한 정론이 바로 설 수 있도록 구단은 끝까지 대응할 것입니다.\n[2026년 구단 강경 대응]",
    ],
}

# 악성 댓글 분류 규칙
COMMENT_RULES: list[dict] = [
    {
        "category": "인신공격",
        "keywords": ["병신", "쓰레기", "죽어", "꺼져", "멍청", "찐따", "미쳤",
                     "정신병", "돌았", "바보", "돼지", "못생", "개새끼", "빡대가리"],
        "color":    "#ffcdd2",
        "severity": "🔴 법적 검토 필요",
        "guide": (
            "직접적 인신공격으로 **법적 조치 검토** 대상입니다.\n\n"
            "✅ 즉시 스크린샷 저장 후 증거 보존\n"
            "✅ 플랫폼 신고 처리 요청\n"
            "✅ 반복 발생 시 법무팀 전달"
        ),
    },
    {
        "category": "허위사실",
        "keywords": ["불법", "도박", "마약", "폭행", "사기", "음주운전",
                     "성범죄", "비리", "탈세", "카더라", "~라고 함", "들었는데", "썰"],
        "color":    "#ffe0b2",
        "severity": "🟠 팩트체크 후 대응",
        "guide": (
            "허위사실 유포 가능성이 있습니다. **명예훼손** 소지가 있습니다.\n\n"
            "✅ 사실관계 즉시 확인\n"
            "✅ 오보 시 공식 반박문 발표\n"
            "✅ 확산 방지를 위한 신속 대응"
        ),
    },
    {
        "category": "은퇴강요",
        "keywords": ["은퇴", "그만해", "뛰지마", "늙었", "나이 많", "물러나",
                     "내려와", "나가", "노땅", "은퇴해", "그만 뛰"],
        "color":    "#fff9c4",
        "severity": "🟡 심리 모니터링",
        "guide": (
            "선수 정신건강에 영향을 줄 수 있는 댓글입니다.\n\n"
            "✅ 선수 심리 상태 모니터링\n"
            "✅ 필요 시 심리 상담 지원 검토\n"
            "✅ 커뮤니티 건전화 캠페인 검토"
        ),
    },
    {
        "category": "일반부정",
        "keywords": ["별로", "못함", "실망", "아쉽", "부족", "형편없",
                     "못해", "안타깝", "개선", "부진", "떨어", "최악"],
        "color":    "#f5f5f5",
        "severity": "⚪ 일반 모니터링",
        "guide": (
            "스포츠 팬의 일반적인 비판 의견으로, 정상 범위의 여론입니다.\n\n"
            "✅ 별도 조치 없이 추이만 모니터링\n"
            "✅ 유사 패턴 급증 시 재분류 검토"
        ),
    },
]


# ── 헬퍼 함수 ─────────────────────────────────────────────────

def generate_responses(
    player: str,
    situation: str,
    resp_type: str,
    tone: str,
) -> list[str]:
    """템플릿 조합으로 대응 문구 3가지 버전 생성."""
    # situation 첫 문장 또는 60자 요약
    first_sentence = situation.split("。")[0].split(".")[0].split("\n")[0].strip()
    sit_short = (first_sentence[:60] + "…") if len(first_sentence) > 60 else first_sentence
    if not sit_short:
        sit_short = "관련 상황"

    def _fill(tmpl: str) -> str:
        return tmpl.replace("{player}", player).replace("{sit}", sit_short)

    versions: list[str] = []
    for v in range(3):
        opening = _fill(_OPENINGS[tone][v])
        body    = _fill(_BODIES[resp_type][v])
        closing = _fill(_CLOSINGS[tone][v])

        # SNS 게시물은 포멀한 클로징 생략
        if resp_type == "선수 개인 SNS 게시물":
            text = opening + "\n\n" + body
        else:
            text = opening + "\n\n" + body + "\n\n" + closing
        versions.append(text)

    return versions


def classify_comment(text: str) -> dict:
    """키워드 기반 댓글 분류. 매칭 없으면 '정상댓글'."""
    for rule in COMMENT_RULES:
        if any(kw in text for kw in rule["keywords"]):
            return rule
    return {
        "category": "정상댓글",
        "color":    "#e8f5e9",
        "severity": "🟢 정상",
        "guide":    "일반 팬 의견입니다. 별도 조치가 필요하지 않습니다.",
    }


@st.cache_resource
def get_korean_font() -> str:
    """reportlab 맑은 고딕 폰트 등록 후 폰트명 반환. 실패 시 Helvetica."""
    if not _REPORTLAB_OK:
        return "Helvetica"
    try:
        pdfmetrics.registerFont(TTFont("Malgun", "C:/Windows/Fonts/malgun.ttf"))
        return "Malgun"
    except Exception:
        return "Helvetica"


@st.cache_data
def load_data(_mtime: float = 0.0) -> pd.DataFrame:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return pd.read_csv(os.path.join(base, "data", "sample_data.csv"), encoding="utf-8-sig")


@st.cache_data
def get_player_stats(_df: pd.DataFrame, player: str) -> dict:
    """선수별 감성 분석 요약 통계 (캐시)."""
    pdf = _df[_df["player_name"] == player]
    if pdf.empty:
        return {"total": 0, "pos": 0, "neg": 0, "risk": 0.0}
    analyzed = analyze_dataframe(pdf)
    total = len(analyzed)
    pos   = int((analyzed["sentiment"] == "긍정").sum())
    neg   = int((analyzed["sentiment"] == "부정").sum())
    risk  = calc_risk_score(analyzed)
    return {"total": total, "pos": pos, "neg": neg, "risk": risk}


def build_pdf(player: str, stats: dict, responses: list[str], font_name: str) -> bytes:
    """reportlab으로 MPRA PDF 리포트 생성."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2.5*cm, leftMargin=2.5*cm,
        topMargin=2*cm,     bottomMargin=2*cm,
    )

    green  = _rlc.HexColor("#1b5e20")
    green2 = _rlc.HexColor("#2e7d32")
    blue   = _rlc.HexColor("#1565c0")
    grey   = _rlc.HexColor("#757575")

    def _sty(name: str, **kw) -> ParagraphStyle:
        return ParagraphStyle(name, fontName=font_name, **kw)

    h1     = _sty("h1", fontSize=20, textColor=green,  spaceAfter=6,  leading=26)
    h2     = _sty("h2", fontSize=13, textColor=green2, spaceAfter=6,  leading=18)
    h3     = _sty("h3", fontSize=11, textColor=blue,   spaceAfter=4,  leading=15)
    normal = _sty("n",  fontSize=10, spaceAfter=4,  leading=15)
    small  = _sty("sm", fontSize=8,  textColor=grey,   spaceAfter=3,  leading=12)

    story: list = []

    # 표지
    story.append(Paragraph("MPRA 여론 리스크 리포트", h1))
    story.append(Paragraph(f"선수: {player} | 생성일: 2026년", normal))
    story.append(HRFlowable(width="100%", thickness=2, color=_rlc.HexColor("#4caf50")))
    story.append(Spacer(1, 0.4*cm))

    # 1. 리스크 점수
    story.append(Paragraph("1. 리스크 점수", h2))
    grade = "위험" if stats["risk"] >= 70 else "주의" if stats["risk"] >= 40 else "안전"
    t1 = Table(
        [["항목", "수치"],
         ["리스크 점수", f"{stats['risk']:.1f} / 100"],
         ["리스크 등급", grade],
         ["분석 댓글",   f"{stats['total']}개"]],
        colWidths=[6*cm, 8*cm],
    )
    t1.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), green2),
        ("TEXTCOLOR",     (0, 0), (-1, 0), _rlc.white),
        ("FONTNAME",      (0, 0), (-1, -1), font_name),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("GRID",          (0, 0), (-1, -1), 0.5, _rlc.HexColor("#e0e0e0")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [_rlc.white, _rlc.HexColor("#f5f5f5")]),
        ("ALIGN",         (1, 0), (1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t1)
    story.append(Spacer(1, 0.5*cm))

    # 2. 감성 분석 요약
    story.append(Paragraph("2. 감성 분석 요약", h2))
    total = stats["total"] or 1
    t2 = Table(
        [["감성", "댓글 수", "비율"],
         ["긍정", str(stats["pos"]), f"{stats['pos']/total*100:.1f}%"],
         ["부정", str(stats["neg"]), f"{stats['neg']/total*100:.1f}%"],
         ["합계", str(stats["total"]), "100%"]],
        colWidths=[5*cm, 4*cm, 5*cm],
    )
    t2.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), green2),
        ("TEXTCOLOR",     (0, 0), (-1, 0), _rlc.white),
        ("FONTNAME",      (0, 0), (-1, -1), font_name),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("GRID",          (0, 0), (-1, -1), 0.5, _rlc.HexColor("#e0e0e0")),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [_rlc.white, _rlc.HexColor("#f5f5f5")]),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.5*cm))

    # 3. AI 생성 대응 문구
    if responses:
        story.append(Paragraph("3. AI 생성 대응 문구", h2))
        for idx, resp in enumerate(responses, 1):
            story.append(Paragraph(f"버전 {idx}", h3))
            safe = _xml_escape(resp).replace("\n", "<br/>")
            story.append(Paragraph(safe, normal))
            story.append(Spacer(1, 0.3*cm))

    # 푸터
    story.append(Spacer(1, 0.8*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_rlc.HexColor("#bdbdbd")))
    story.append(Paragraph(
        "본 리포트는 MPRA (Mental &amp; Public Risk AI) 시스템에 의해 자동 생성되었습니다. | 2026년",
        small,
    ))

    doc.build(story)
    return buf.getvalue()


# ── 페이지 설정 ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI 대응 문구 생성기",
    page_icon="🤖",
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
st.title("🤖 AI 대응 문구 생성기")
st.markdown("템플릿 기반 위기 대응 문구 자동 생성 · 악성 댓글 분류 · PDF 리포트 | 2026년")
st.markdown("---")

# ╔══════════════════════════════════════════════════╗
# ║  섹션 1: 입력 + 문구 생성                          ║
# ╚══════════════════════════════════════════════════╝
col_in, col_out = st.columns([1, 2], gap="large")

with col_in:
    st.subheader("⚙️ 입력 조건")

    player_input: str = st.text_input(
        "선수 이름",
        value="이강인",
        placeholder="예: 이강인",
        key="player_input",
    )

    situation: str = st.text_area(
        "위기 상황 설명",
        placeholder="예: SNS에 부적절한 발언이 확산되어 팬들의 비판이 급증하고 있습니다.",
        height=130,
        key="situation_input",
    )

    st.markdown("---")

    resp_type: str = st.selectbox(
        "대응 문구 유형",
        RESPONSE_TYPES,
        index=0,
        key="resp_type",
    )

    tone: str = st.selectbox(
        "톤",
        TONES,
        index=0,
        key="tone",
    )

    st.markdown("---")

    generate_btn = st.button(
        "✨ AI 문구 생성",
        type="primary",
        use_container_width=True,
    )

with col_out:
    st.subheader("📝 생성된 대응 문구")

    if generate_btn:
        if not player_input.strip():
            st.warning("⚠️ 선수 이름을 입력해주세요.")
        elif not situation.strip():
            st.warning("⚠️ 위기 상황을 설명해주세요.")
        else:
            with st.spinner("🤖 AI가 대응 문구를 생성하고 있습니다..."):
                time.sleep(2)      # 실제처럼 느껴지는 로딩 효과
                versions = generate_responses(player_input, situation, resp_type, tone)

            st.session_state["generated_versions"] = versions
            st.session_state["generated_player"]   = player_input
            st.success(f"✅ **{player_input}** 대응 문구 3가지 버전이 생성됐습니다.")

    if "generated_versions" in st.session_state:
        tab1, tab2, tab3 = st.tabs(["📄 버전 1", "📄 버전 2", "📄 버전 3"])
        for i, (tab, ver) in enumerate(
            zip([tab1, tab2, tab3], st.session_state["generated_versions"])
        ):
            with tab:
                st.text_area(
                    f"버전 {i+1}  (드래그 후 Ctrl+C 로 복사)",
                    value=ver,
                    height=300,
                    key=f"ver_output_{i}",
                )
    else:
        st.info("왼쪽 입력창을 작성하고 **AI 문구 생성** 버튼을 누르세요.")

st.markdown("---")

# ╔══════════════════════════════════════════════════╗
# ║  섹션 2: 악성 댓글 자동 분류                       ║
# ╚══════════════════════════════════════════════════╝
st.subheader("🔍 악성 댓글 자동 분류")
st.caption("댓글을 입력하면 유형을 분류하고 대응 가이드를 제공합니다.")

cls_l, cls_r = st.columns([1, 1], gap="large")

with cls_l:
    comment_input: str = st.text_area(
        "분류할 댓글 입력",
        placeholder="예: 슬슬 은퇴 준비해야 하는 거 아님? 예전 같지 않아",
        height=120,
        key="comment_input",
    )
    classify_btn = st.button("🔎 분류하기", use_container_width=True, key="classify_btn")

with cls_r:
    if classify_btn:
        if not comment_input.strip():
            st.warning("⚠️ 댓글을 입력해주세요.")
        else:
            result = classify_comment(comment_input)
            st.session_state["cls_result"]  = result
            st.session_state["cls_comment"] = comment_input

    if "cls_result" in st.session_state:
        r  = st.session_state["cls_result"]
        bg = r["color"]
        st.markdown(
            f"<div style='background:{bg};border-radius:8px;padding:14px 16px;"
            f"border-left:4px solid #555;margin-bottom:12px;'>"
            f"<b>분류 결과:</b> {r['category']}<br>"
            f"<b>심각도:</b> {r['severity']}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("**📋 대응 가이드**")
        st.markdown(r["guide"])
    else:
        st.info("분류 결과가 여기에 표시됩니다.")

# 분류 기준 요약표
with st.expander("📌 분류 유형 기준 보기"):
    guide_rows = [
        {"유형": rule["category"], "심각도": rule["severity"],
         "대응 방침": rule["guide"].split("\n")[0]}
        for rule in COMMENT_RULES
    ]
    guide_rows.append(
        {"유형": "정상댓글", "심각도": "🟢 정상", "대응 방침": "일반 팬 의견. 별도 조치 불필요."}
    )
    st.dataframe(pd.DataFrame(guide_rows), use_container_width=True, hide_index=True)

st.markdown("---")

# ╔══════════════════════════════════════════════════╗
# ║  섹션 3: PDF 리포트 생성                           ║
# ╚══════════════════════════════════════════════════╝
st.subheader("📥 PDF 리포트 생성")
st.caption("선수를 선택하면 리스크 점수 + 감성 분석 + AI 대응 문구를 PDF로 저장합니다.")

pdf_l, pdf_r = st.columns([1, 2], gap="large")

with pdf_l:
    pdf_player: str = st.selectbox(
        "PDF 대상 선수",
        PLAYERS,
        index=1,
        key="pdf_player",
    )
    include_responses = st.checkbox(
        "생성된 AI 문구 포함",
        value=True,
        help="AI 문구를 먼저 생성해야 포함됩니다.",
    )
    pdf_btn = st.button(
        "📄 PDF 생성",
        type="primary",
        use_container_width=True,
        disabled=not _REPORTLAB_OK,
    )
    if not _REPORTLAB_OK:
        st.warning("`pip install reportlab` 이 필요합니다.")

with pdf_r:
    if pdf_btn and _REPORTLAB_OK:
        df_all = load_data(_mtime=os.path.getmtime(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_data.csv")))
        with st.spinner(f"🔄 {pdf_player} 감성 분석 중... (첫 실행 시 모델 로딩 포함)"):
            stats = get_player_stats(df_all, pdf_player)

        font_name = get_korean_font()

        responses: list[str] = []
        if include_responses and "generated_versions" in st.session_state:
            responses = st.session_state["generated_versions"]

        with st.spinner("📄 PDF 생성 중..."):
            try:
                pdf_bytes = build_pdf(pdf_player, stats, responses, font_name)
                fname     = f"MPRA_리포트_{pdf_player}_2026.pdf"

                st.success(f"✅ **{pdf_player}** PDF 리포트 생성 완료!")

                if font_name == "Helvetica":
                    st.warning(
                        "⚠️ 한국어 폰트를 찾지 못해 Helvetica로 대체됐습니다.  \n"
                        "한글이 깨질 수 있습니다. Malgun Gothic 또는 NanumGothic 설치를 권장합니다."
                    )

                grade = "위험" if stats["risk"] >= 70 else "주의" if stats["risk"] >= 40 else "안전"
                st.markdown(
                    f"**포함 내용**: 리스크 {stats['risk']:.1f}점 ({grade}) · "
                    f"댓글 {stats['total']}개 · AI 문구 {len(responses)}버전"
                )

                st.download_button(
                    label="📥 PDF 다운로드",
                    data=pdf_bytes,
                    file_name=fname,
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"❌ PDF 생성 오류: {exc}")
    elif not pdf_btn:
        st.info(
            "왼쪽에서 선수를 선택하고 **PDF 생성** 버튼을 누르세요.\n\n"
            "AI 문구를 먼저 생성하면 PDF에 자동으로 포함됩니다."
        )

# ── 하단 안내 문구 ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='background:#f5f5f5;border-radius:8px;padding:14px 18px;"
    "border-left:4px solid #2e7d32;'>"
    "<p style='margin:0;font-size:0.9rem;color:#555;'>"
    "💡 본 시스템은 AI 기반 문구 생성 엔진을 탑재하고 있으며, "
    "실제 서비스에서는 <b>Claude API</b>와 연동하여 더욱 정교한 "
    "맞춤형 문구를 생성합니다."
    "</p>"
    "</div>",
    unsafe_allow_html=True,
)
