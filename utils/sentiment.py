from __future__ import annotations

import streamlit as st
import pandas as pd
from transformers import pipeline

MODEL_NAME = "monologg/koelectra-base-finetuned-sentiment"
NEUTRAL_THRESHOLD = 0.65


@st.cache_resource(show_spinner="AI 모델 로딩 중...")
def load_model():
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=None,
        device=-1,
    )


def analyze_sentiment(text: str) -> tuple[str, float]:
    classifier = load_model()
    result = classifier(str(text)[:512])[0]

    scores = {}
    for r in result:
        lbl = r["label"].upper()
        if "POS" in lbl or lbl == "LABEL_1":
            scores["pos"] = r["score"]
        elif "NEG" in lbl or lbl == "LABEL_0":
            scores["neg"] = r["score"]

    pos = scores.get("pos", 0.5)
    neg = scores.get("neg", 0.5)

    if max(pos, neg) < NEUTRAL_THRESHOLD:
        return "중립", round(max(pos, neg), 4)
    if pos >= neg:
        return "긍정", round(pos, 4)
    return "부정", round(neg, 4)


def analyze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "sentiment" in df.columns and "confidence" in df.columns:
        return df
    results = [analyze_sentiment(t) for t in df["comment"]]
    df["sentiment"] = [r[0] for r in results]
    df["confidence"] = [r[1] for r in results]
    return df


def calc_risk_score(df: pd.DataFrame) -> float:
    """리스크 점수 = (부정 댓글 수 / 전체 댓글 수) × 100"""
    if len(df) == 0:
        return 0.0
    if "sentiment" not in df.columns:
        df = analyze_dataframe(df)
    neg_count = int((df["sentiment"] == "부정").sum())
    return round(neg_count / len(df) * 100, 1)
