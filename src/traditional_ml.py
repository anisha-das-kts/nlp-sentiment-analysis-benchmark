from __future__ import annotations

from typing import Tuple

import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def build_tfidf_model(
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    seed: int = 42,
):
    """
    Build TF-IDF vectorizer and Logistic Regression classifier.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=seed,
    )

    return vectorizer, model


def train_model(
    x_train,
    y_train,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    seed: int = 42,
):
    """
    Fit TF-IDF and Logistic Regression on training data only.
    """
    vectorizer, model = build_tfidf_model(
        max_features=max_features,
        ngram_range=ngram_range,
        seed=seed,
    )

    x_train_tfidf = vectorizer.fit_transform(x_train)

    model.fit(
        x_train_tfidf,
        y_train,
    )
    
    param_grid = {'C': [0.1, 1, 10], 'penalty': ['l2']}
    grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=seed),
                        param_grid, cv=3, scoring='f1_macro')
    grid.fit(x_train_tfidf, y_train)

    return vectorizer, model,  vectorizer, grid.best_estimator_


def predict(
    vectorizer,
    model,
    texts,
):
    """
    Generate class predictions.
    """
    features = vectorizer.transform(texts)

    return model.predict(features)


def predict_proba(
    vectorizer,
    model,
    texts,
):
    """
    Generate positive-class probabilities.
    """
    features = vectorizer.transform(texts)

    probabilities = model.predict_proba(features)

    return probabilities[:, 1]


def save_model(
    vectorizer,
    model,
    vectorizer_path: str,
    model_path: str,
) -> None:
    """
    Save TF-IDF vectorizer and Logistic Regression model.
    """
    joblib.dump(
        vectorizer,
        vectorizer_path,
    )

    joblib.dump(
        model,
        model_path,
    )


def load_model(
    vectorizer_path: str,
    model_path: str,
):
    """
    Load TF-IDF vectorizer and Logistic Regression model.
    """
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)

    return vectorizer, model