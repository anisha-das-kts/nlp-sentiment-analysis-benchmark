from __future__ import annotations

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


# ============================================================
# Project path
# ============================================================

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


# ============================================================
# Project imports
# ============================================================

from src.evaluation import (
    calculate_metrics,
    save_inference_chart,
    save_metric_bar_chart,
    save_results,
    save_training_time_chart,
)

from src.lstm_model import (
    LSTMClassifier,
    MovieReviewDataset,
    encode_and_pad,
)

from src.preprocessing import (
    load_movie_reviews,
    prepare_dataset,
    split_dataset,
)

from src.traditional_ml import (
    load_model,
    predict,
    predict_proba,
)

from src.utils import set_seed

from src.vader import (
    build_vader,
    predict as vader_predict,
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
)

FIGURES_DIR = os.path.join(
    RESULTS_DIR,
    "figures",
)

TRAINING_METRICS_PATH = os.path.join(
    RESULTS_DIR,
    "metrics.csv",
)

COMPARISON_PATH = os.path.join(
    RESULTS_DIR,
    "comparison.csv",
)


# ============================================================
# Load training-time metrics
# ============================================================

def load_training_metrics() -> pd.DataFrame:
    """
    Load preparation/training timing measurements generated
    by scripts/train.py.
    """

    if not os.path.exists(
        TRAINING_METRICS_PATH
    ):
        raise FileNotFoundError(
            "Training metrics file not found:\n"
            f"{TRAINING_METRICS_PATH}\n\n"
            "Run:\n"
            "python scripts/train.py\n"
            "before running evaluation."
        )

    return pd.read_csv(
        TRAINING_METRICS_PATH
    )


# ============================================================
# Logistic Regression evaluation
# ============================================================

def evaluate_logistic_regression(
    test_df: pd.DataFrame,
):
    """
    Evaluate TF-IDF + Logistic Regression.
    """

    vectorizer, model = load_model(
        os.path.join(
            MODEL_DIR,
            "tfidf_vectorizer.joblib",
        ),
        os.path.join(
            MODEL_DIR,
            "logistic_regression.joblib",
        ),
    )

    start = time.perf_counter()

    predictions = predict(
        vectorizer,
        model,
        test_df["Clean_Review"],
    )

    inference_time = (
        time.perf_counter()
        - start
    )

    probabilities = predict_proba(
        vectorizer,
        model,
        test_df["Clean_Review"],
    )

    metrics = calculate_metrics(
        test_df["Label"],
        predictions,
        positive_label="pos",
    )

    return (
        metrics,
        inference_time,
        probabilities,
        predictions,
    )


# ============================================================
# LSTM evaluation
# ============================================================

def evaluate_lstm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Evaluate the saved PyTorch LSTM.
    """

    model_path = os.path.join(
        MODEL_DIR,
        "lstm_model.pt",
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
    )

    vocab = checkpoint["vocab"]
    max_len = checkpoint["max_len"]

    test_encoded = [
        encode_and_pad(
            text,
            vocab,
            max_len,
        )
        for text in test_df[
            "Clean_Review"
        ]
    ]

    test_sequences = [
        item[0]
        for item in test_encoded
    ]

    test_lengths = [
        item[1]
        for item in test_encoded
    ]

    test_labels = (
        test_df["Label"]
        .map(
            {
                "neg": 0,
                "pos": 1,
            }
        )
        .to_numpy()
    )

    test_dataset = MovieReviewDataset(
        test_sequences,
        test_lengths,
        test_labels,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = LSTMClassifier(
        vocabulary_size=len(vocab),
        padding_index=vocab["<PAD>"],
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    predictions = []
    actual = []

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():

        for (
            reviews,
            lengths,
            labels,
        ) in test_loader:

            reviews = reviews.to(device)

            logits = model(
                reviews,
                lengths,
            )

            predicted = (
                logits >= 0
            ).long()

            predictions.extend(
                predicted
                .cpu()
                .tolist()
            )

            actual.extend(
                labels
                .long()
                .tolist()
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_time = (
        time.perf_counter()
        - start
    )

    metrics = calculate_metrics(
        actual,
        predictions,
        positive_label=1,
    )

    return (
        metrics,
        inference_time,
        np.array(predictions),
        np.array(actual),
    )


# ============================================================
# VADER evaluation
# ============================================================

def evaluate_vader(
    test_df: pd.DataFrame,
):
    """
    Evaluate VADER on the original test reviews.
    """

    analyzer = build_vader()

    start = time.perf_counter()

    predictions = vader_predict(
        test_df["Review"],
        analyzer,
    )

    inference_time = (
        time.perf_counter()
        - start
    )

    metrics = calculate_metrics(
        test_df["Label"],
        predictions,
        positive_label="pos",
    )

    return (
        metrics,
        inference_time,
        predictions,
    )


# ============================================================
# Add latency information
# ============================================================

def add_latency_columns(
    comparison: pd.DataFrame,
    test_size: int,
) -> pd.DataFrame:
    """
    Add per-review inference latency.
    """

    comparison[
        "Inference per Review (ms)"
    ] = (
        comparison[
            "Test Inference Time (seconds)"
        ]
        / test_size
        * 1000
    )

    return comparison


# ============================================================
# Main evaluation pipeline
# ============================================================

def main():

    set_seed(
        SEED
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    os.makedirs(
        FIGURES_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load training metrics
    # --------------------------------------------------------

    training_metrics = (
        load_training_metrics()
    )

    training_time_lookup = dict(
        zip(
            training_metrics["Model"],
            training_metrics[
                "Preparation + Training Time (seconds)"
            ],
        )
    )

    # --------------------------------------------------------
    # Load and prepare dataset
    # --------------------------------------------------------

    print(
        "Loading NLTK Movie Reviews..."
    )

    df = prepare_dataset(
        load_movie_reviews()
    )

    train_df, test_df = split_dataset(
        df,
        test_size=0.20,
        seed=SEED,
    )

    print(
        f"Training samples: "
        f"{len(train_df)}"
    )

    print(
        f"Testing samples: "
        f"{len(test_df)}"
    )

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    (
        ml_metrics,
        ml_inference,
        ml_probabilities,
        ml_predictions,
    ) = evaluate_logistic_regression(
        test_df
    )

    results = []

    results.append(
        {
            "Model": "Logistic Regression",
            **ml_metrics,
            "Preparation + Training Time (seconds)": (
                training_time_lookup[
                    "Logistic Regression"
                ]
            ),
            "Test Inference Time (seconds)": (
                ml_inference
            ),
        }
    )

    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------

    (
        lstm_metrics,
        lstm_inference,
        lstm_predictions,
        lstm_actual,
    ) = evaluate_lstm(
        train_df,
        test_df,
    )

    results.append(
        {
            "Model": "LSTM",
            **lstm_metrics,
            "Preparation + Training Time (seconds)": (
                training_time_lookup[
                    "LSTM"
                ]
            ),
            "Test Inference Time (seconds)": (
                lstm_inference
            ),
        }
    )

    # --------------------------------------------------------
    # VADER
    # --------------------------------------------------------

    (
        vader_metrics,
        vader_inference,
        vader_predictions,
    ) = evaluate_vader(
        test_df
    )

    results.append(
        {
            "Model": "VADER",
            **vader_metrics,
            "Preparation + Training Time (seconds)": 0.0,
            "Test Inference Time (seconds)": (
                vader_inference
            ),
        }
    )

    # --------------------------------------------------------
    # Build comparison DataFrame
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        results
    )

    comparison = add_latency_columns(
        comparison,
        len(test_df),
    )

    comparison = comparison[
        [
            "Model",
            "Accuracy",
            "Macro F1",
            "Positive-Class F1",
            "Preparation + Training Time (seconds)",
            "Test Inference Time (seconds)",
            "Inference per Review (ms)",
        ]
    ]

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print(
        "\nFinal Comparison\n"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save comparison CSV
    # --------------------------------------------------------

    save_results(
        comparison,
        COMPARISON_PATH,
    )

    print(
        "\nComparison results saved to:"
    )

    print(
        COMPARISON_PATH
    )

    # --------------------------------------------------------
    # Generate figures
    # --------------------------------------------------------

    save_metric_bar_chart(
        comparison,
        "Accuracy",
        "Accuracy Comparison",
        "Accuracy",
        os.path.join(
            FIGURES_DIR,
            "accuracy_comparison.png",
        ),
    )

    save_metric_bar_chart(
        comparison,
        "Macro F1",
        "Macro F1 Comparison",
        "Macro F1",
        os.path.join(
            FIGURES_DIR,
            "f1_comparison.png",
        ),
    )

    save_training_time_chart(
        comparison,
        os.path.join(
            FIGURES_DIR,
            "training_time_comparison.png",
        ),
    )

    save_inference_chart(
        comparison,
        os.path.join(
            FIGURES_DIR,
            "inference_latency.png",
        ),
    )

    print(
        "\nFigures saved to:"
    )

    print(
        FIGURES_DIR
    )

    print(
        "\nEvaluation completed successfully."
    )


if __name__ == "__main__":
    main()