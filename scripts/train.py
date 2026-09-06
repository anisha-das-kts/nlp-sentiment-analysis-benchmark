from __future__ import annotations

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


# ============================================================
# Project path setup
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

from src.lstm_model import (
    LSTMClassifier,
    MovieReviewDataset,
    build_vocabulary,
    encode_and_pad,
    train_lstm_with_validation,
)

from src.preprocessing import (
    load_movie_reviews,
    prepare_dataset,
    split_dataset,
)

from src.traditional_ml import (
    save_model,
    train_model,
)

from src.utils import set_seed


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

METRICS_PATH = os.path.join(
    RESULTS_DIR,
    "metrics.csv",
)


# ============================================================
# Traditional ML
# ============================================================

def train_traditional(
    train_df: pd.DataFrame,
) -> dict:
    """
    Train TF-IDF + Logistic Regression.

    Returns
    -------
    dict
        Training-time information for experiment tracking.
    """

    print(
        "Training TF-IDF + Logistic Regression..."
    )

    start = time.perf_counter()

    vectorizer, model = train_model(
        train_df["Clean_Review"],
        train_df["Label"],
        seed=SEED,
    )[:2]

    training_time = (
        time.perf_counter()
        - start
    )

    # Save trained artifacts
    save_model(
        vectorizer,
        model,
        os.path.join(
            MODEL_DIR,
            "tfidf_vectorizer.joblib",
        ),
        os.path.join(
            MODEL_DIR,
            "logistic_regression.joblib",
        ),
    )

    print(
        f"Logistic Regression training time: "
        f"{training_time:.4f}s"
    )

    return {
        "Model": "Logistic Regression",
        "Preparation Time (seconds)": 0.0,
        "Training Time (seconds)": training_time,
        "Preparation + Training Time (seconds)": training_time,
    }


# ============================================================
# Deep Learning
# ============================================================

def train_deep_learning(
    train_df: pd.DataFrame,
) -> dict:
    """
    Train the PyTorch LSTM using a validation split
    and early stopping.

    Returns
    -------
    dict
        Training-time information and training history.
    """

    print(
        "Training LSTM..."
    )

    # --------------------------------------------------------
    # 1. Preparation
    # --------------------------------------------------------

    preparation_start = time.perf_counter()

    # Build vocabulary using training data only
    vocab = build_vocabulary(
        train_df["Clean_Review"]
    )

    # Determine sequence length using the 90th percentile
    lengths = (
        train_df["Clean_Review"]
        .str.split()
        .str.len()
    )

    max_len = int(
        min(
            500,
            np.percentile(
                lengths,
                90,
            ),
        )
    )

    max_len = max(
        max_len,
        1,
    )

    # Encode reviews
    encoded = [
        encode_and_pad(
            text,
            vocab,
            max_len,
        )
        for text in train_df["Clean_Review"]
    ]

    sequences = [
        item[0]
        for item in encoded
    ]

    sequence_lengths = [
        item[1]
        for item in encoded
    ]

    labels = (
        train_df["Label"]
        .map(
            {
                "neg": 0,
                "pos": 1,
            }
        )
        .to_numpy()
    )

    # --------------------------------------------------------
    # 2. Train / validation split
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        y_train,
        y_val,
        l_train,
        l_val,
    ) = train_test_split(
        sequences,
        labels,
        sequence_lengths,
        test_size=0.10,
        random_state=SEED,
        stratify=labels,
    )

    # --------------------------------------------------------
    # 3. PyTorch datasets
    # --------------------------------------------------------

    train_dataset = MovieReviewDataset(
        X_train,
        l_train,
        y_train,
    )

    val_dataset = MovieReviewDataset(
        X_val,
        l_val,
        y_val,
    )

    # Reproducible DataLoader shuffling
    generator = torch.Generator()

    generator.manual_seed(
        SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
    )

    preparation_time = (
        time.perf_counter()
        - preparation_start
    )

    print(
        f"LSTM preparation time: "
        f"{preparation_time:.4f}s"
    )

    # --------------------------------------------------------
    # 4. Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"LSTM device: {device}"
    )

    # --------------------------------------------------------
    # 5. Model
    # --------------------------------------------------------

    model = LSTMClassifier(
        vocabulary_size=len(vocab),
        padding_index=vocab["<PAD>"],
    ).to(device)

    # --------------------------------------------------------
    # 6. Train with validation and early stopping
    # --------------------------------------------------------

    training_start = time.perf_counter()

    model, history = train_lstm_with_validation(
        model,
        train_loader,
        val_loader,
        device,
        epochs=5,
        learning_rate=0.001,
        patience=3,
    )

    training_time = (
        time.perf_counter()
        - training_start
    )

    total_time = (
        preparation_time
        + training_time
    )

    # --------------------------------------------------------
    # 7. Save best model
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "lstm_model.pt",
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "max_len": max_len,
        },
        model_path,
    )

    # --------------------------------------------------------
    # 8. Print timing information
    # --------------------------------------------------------

    print(
        f"LSTM training time: "
        f"{training_time:.4f}s"
    )

    print(
        f"LSTM total time: "
        f"{total_time:.4f}s"
    )

    return {
        "Model": "LSTM",
        "Preparation Time (seconds)": preparation_time,
        "Training Time (seconds)": training_time,
        "Preparation + Training Time (seconds)": total_time,
        "history": history,
    }


# ============================================================
# Save experiment metrics
# ============================================================

def save_training_metrics(
    logistic_result: dict,
    lstm_result: dict,
) -> None:
    """
    Save training/preparation timing information to CSV.
    """

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    records = [
        {
            "Model": logistic_result["Model"],
            "Preparation Time (seconds)": (
                logistic_result[
                    "Preparation Time (seconds)"
                ]
            ),
            "Training Time (seconds)": (
                logistic_result[
                    "Training Time (seconds)"
                ]
            ),
            "Preparation + Training Time (seconds)": (
                logistic_result[
                    "Preparation + Training Time (seconds)"
                ]
            ),
        },
        {
            "Model": lstm_result["Model"],
            "Preparation Time (seconds)": (
                lstm_result[
                    "Preparation Time (seconds)"
                ]
            ),
            "Training Time (seconds)": (
                lstm_result[
                    "Training Time (seconds)"
                ]
            ),
            "Preparation + Training Time (seconds)": (
                lstm_result[
                    "Preparation + Training Time (seconds)"
                ]
            ),
        },
    ]

    metrics_df = pd.DataFrame(
        records
    )

    metrics_df.to_csv(
        METRICS_PATH,
        index=False,
    )

    print(
        f"\nTraining metrics saved to:\n"
        f"{METRICS_PATH}"
    )

    print(
        "\nTraining metrics:"
    )

    print(
        metrics_df.to_string(
            index=False
        )
    )


# ============================================================
# Main
# ============================================================

def main():
    """
    Execute the complete training pipeline.
    """

    # Reproducibility
    set_seed(
        SEED
    )

    # Required directories
    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(
        "Loading NLTK Movie Reviews..."
    )

    df = load_movie_reviews()

    df = prepare_dataset(
        df
    )

    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------

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
    # Traditional ML
    # --------------------------------------------------------

    logistic_result = train_traditional(
        train_df
    )

    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------

    lstm_result = train_deep_learning(
        train_df
    )

    # --------------------------------------------------------
    # Save timing metrics
    # --------------------------------------------------------

    save_training_metrics(
        logistic_result,
        lstm_result,
    )

    print(
        "\nTraining completed successfully."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()