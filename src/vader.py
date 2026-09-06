from __future__ import annotations

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


def build_vader():
    """
    Initialize VADER sentiment analyzer.
    """
    try:
        nltk.data.find(
            "sentiment/vader_lexicon"
        )
    except LookupError:
        nltk.download(
            "vader_lexicon",
            quiet=True,
        )

    return SentimentIntensityAnalyzer()


def predict(
    texts,
    analyzer=None,
):
    """
    Generate binary sentiment predictions using VADER.

    VADER is evaluated on original text so that punctuation,
    capitalization and intensifiers remain available.
    """
    if analyzer is None:
        analyzer = build_vader()

    predictions = []

    for text in texts:
        scores = analyzer.polarity_scores(
            text
        )

        compound = scores["compound"]

        if compound >= 0.05:
            prediction = "pos"

        elif compound <= -0.05:
            prediction = "neg"

        else:
            prediction = (
                "pos"
                if scores["pos"] > scores["neg"]
                else "neg"
            )

        predictions.append(prediction)

    return predictions