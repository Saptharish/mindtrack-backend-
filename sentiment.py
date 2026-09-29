import re

_POSITIVE_WORDS = ["good","great","happy","better","wonderful","amazing",
                   "love","excited","joy","grateful","peaceful","hopeful","well"]
_NEGATIVE_WORDS = ["bad","sad","terrible","awful","depressed","anxious",
                   "worried","stressed","tired","angry","hopeless","lonely"]

def analyze_sentiment(text: str) -> dict:
    text_lower = text.lower()
    pos = sum(1 for w in _POSITIVE_WORDS if re.search(r'\b' + w + r'\b', text_lower))
    neg = sum(1 for w in _NEGATIVE_WORDS if re.search(r'\b' + w + r'\b', text_lower))
    if pos > neg:
        score = min(0.5 + pos * 0.1, 0.95)
        return {"label": "POSITIVE", "confidence": score,
                "mood_score": round(score * 10, 1)}
    score = max(0.05, 0.5 - neg * 0.1)
    return {"label": "NEGATIVE", "confidence": 1 - score,
            "mood_score": max(1.0, round(score * 10, 1))}
