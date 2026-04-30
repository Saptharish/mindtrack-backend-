try:
    from transformers import pipeline
    _classifier = None

    def get_classifier():
        global _classifier
        if _classifier is None:
            _classifier = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
        return _classifier

    def analyze_sentiment(text: str) -> dict:
        classifier = get_classifier()
        result     = classifier(text[:512])[0]
        label      = result["label"]
        score      = result["score"]
        mood_score = round(score * 10, 1) if label == "POSITIVE" \
                     else max(1.0, round((1 - score) * 10, 1))
        return {"label": label, "confidence": round(score, 3),
                "mood_score": mood_score}

except Exception:
    def analyze_sentiment(text: str) -> dict:
        positive_words = ["good","great","happy","better","wonderful","amazing",
                          "love","excited","joy","grateful","peaceful","hopeful","well"]
        negative_words = ["bad","sad","terrible","awful","depressed","anxious",
                          "worried","stressed","tired","angry","hopeless","lonely"]
        text_lower = text.lower()
        pos = sum(1 for w in positive_words if w in text_lower)
        neg = sum(1 for w in negative_words if w in text_lower)
        if pos > neg:
            score = min(0.5 + pos * 0.1, 0.95)
            return {"label":"POSITIVE","confidence":score,
                    "mood_score":round(score*10,1)}
        score = max(0.05, 0.5 - neg * 0.1)
        return {"label":"NEGATIVE","confidence":1-score,
                "mood_score":max(1.0,round(score*10,1))}