from __future__ import annotations

import re
from typing import Tuple

import nltk
import pandas as pd

from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split


def download_nltk_resources() -> None:
    """
    Download all NLTK resources required by the project.
    """
    resources = [
        ("corpora/movie_reviews", "movie_reviews"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]

    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


def load_movie_reviews() -> pd.DataFrame:
    """
    Load the NLTK Movie Reviews corpus into a DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns:
        - Review
        - Label
    """
    download_nltk_resources()

    documents = []

    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            documents.append(
                (
                    movie_reviews.raw(fileid),
                    category,
                )
            )

    return pd.DataFrame(
        documents,
        columns=["Review", "Label"],
    )


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate reviews only when duplicate reviews
    do not contain conflicting sentiment labels.
    """
    df = df.copy()

    duplicate_mask = df.duplicated(
        subset=["Review"],
        keep=False,
    )

    conflicting_reviews = (
        df.groupby("Review")["Label"]
        .nunique()
        .loc[lambda values: values > 1]
    )

    if duplicate_mask.sum() > 0 and len(conflicting_reviews) == 0:
        df = (
            df.drop_duplicates(
                subset=["Review"],
                keep="first",
            )
            .reset_index(drop=True)
        )

    return df


def get_stop_words() -> set[str]:
    """
    Return English stop words while preserving sentiment-bearing
    negation words.
    """
    download_nltk_resources()

    words = set(stopwords.words("english"))

    return words - {"no", "nor", "not"}


def clean_text(
    text: str,
    stop_words: set[str] | None = None,
) -> str:
    """
    Clean an input review.

    Processing steps:
    1. Lowercase
    2. Normalize contracted negation
    3. Remove URLs and HTML
    4. Remove punctuation/numbers
    5. Tokenize
    6. Remove stop words while preserving negations
    7. Remove one-character tokens
    8. Normalize whitespace
    """
    if pd.isna(text):
        return ""

    if stop_words is None:
        stop_words = get_stop_words()

    text = str(text).lower()

    text = re.sub(
        r"n['’]t\b",
        " not",
        text,
    )

    text = re.sub(
        r"https?://\S+|www\.\S+|<[^>]+>",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-z\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    tokens = word_tokenize(text)

    tokens = [
        token
        for token in tokens
        if token not in stop_words and len(token) > 1
    ]

    return " ".join(tokens)


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate, remove safe duplicates, and create cleaned reviews.
    """
    required_columns = {"Review", "Label"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = remove_duplicates(df)

    stop_words = get_stop_words()

    df["Clean_Review"] = df["Review"].apply(
        lambda text: clean_text(
            text,
            stop_words=stop_words,
        )
    )

    if (df["Clean_Review"] == "").any():
        raise ValueError(
            "At least one review became empty after preprocessing."
        )

    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a stratified train/test split.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["Label"],
    )

    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )