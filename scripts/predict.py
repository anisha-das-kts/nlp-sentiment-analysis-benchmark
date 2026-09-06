from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT,
    )


from src.preprocessing import (
    clean_text,
)

from src.traditional_ml import (
    load_model,
    predict,
    predict_proba,
)


VECTOR_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "tfidf_vectorizer.joblib",
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "logistic_regression.joblib",
)


def main():
    if len(sys.argv) < 2:
        print(
            'Usage:\n'
            'python scripts/predict.py '
            '"Your review text here"'
        )
        raise SystemExit(1)

    text = " ".join(
        sys.argv[1:]
    )

    cleaned = clean_text(
        text
    )

    vectorizer, model = load_model(
        VECTOR_PATH,
        MODEL_PATH,
    )

    prediction = predict(
        vectorizer,
        model,
        [cleaned],
    )[0]

    probability = predict_proba(
        vectorizer,
        model,
        [cleaned],
    )[0]

    confidence = (
        probability
        if prediction == "pos"
        else 1 - probability
    )

    print(
        f"\nSentiment: "
        f"{prediction.upper()}"
    )

    print(
        f"Confidence: "
        f"{confidence:.2%}"
    )


if __name__ == "__main__":
    main()