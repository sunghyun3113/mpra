from __future__ import annotations

import pandas as pd

try:
    import streamlit as st
    from transformers import pipeline
    _TRANSFORMERS_OK = True
except ImportError:
    _TRANSFORMERS_OK = False

MODEL_NAME = "monologg/koelectra-base-finetuned-sentiment"


def _load_model():
    if not _TRANSFORMERS_OK:
        return None
    try:
        return st.cache_resource(lambda: pipeline(
            "text-classification",
            model=MODEL_NAME,
            top_k=None,
            device=-1,
        ))()
    except Exception:
        return None


def analyze_sentiment(text: str) -> tuple[str, float]:
    classifier = _load_model()
    if classifier is None:
        return "부정", 0.80

    try:
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
        if pos >= neg:
            return "긍정", round(pos, 4)
        return "부정", round(neg, 4)
    except Exception:
        return "부정", 0.80


def analyze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "sentiment" in df.columns and "confidence" in df.columns:
        return df  # CSV 사전 계산값 사용
    results = [analyze_sentiment(t) for t in df["comment"]]
    df["sentiment"]  = [r[0] for r in results]
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
