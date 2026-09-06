from __future__ import annotations

import os
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def calculate_metrics(
    y_true,
    y_pred,
    positive_label=1,
):
    """
    Calculate the project's primary classification metrics.
    """
    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "Macro F1": f1_score(
            y_true,
            y_pred,
            average="macro",
        ),
        "Positive-Class F1": f1_score(
            y_true,
            y_pred,
            pos_label=positive_label,
        ),
    }


def classification_summary(
    y_true,
    y_pred,
    labels=None,
    target_names=None,
) -> str:
    """
    Return a readable classification report.
    """
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )


def create_comparison_dataframe(
    records: Iterable[dict],
) -> pd.DataFrame:
    """
    Convert model result records into a comparison DataFrame.
    """
    return pd.DataFrame(records)


def save_results(
    dataframe: pd.DataFrame,
    path: str,
) -> None:
    """
    Save results to CSV.
    """
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    dataframe.to_csv(
        path,
        index=False,
    )


def save_confusion_matrix(
    y_true,
    y_pred,
    labels,
    display_labels,
    title,
    output_path,
):
    """
    Save a confusion matrix figure.
    """
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    image = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks(
        range(len(display_labels))
    )

    ax.set_yticks(
        range(len(display_labels))
    )

    ax.set_xticklabels(display_labels)
    ax.set_yticklabels(display_labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax)

    plt.tight_layout()

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_metric_bar_chart(
    dataframe: pd.DataFrame,
    metric_column: str,
    title: str,
    ylabel: str,
    output_path: str,
):
    """
    Save a comparison bar chart.
    """
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    bars = ax.bar(
        dataframe["Model"],
        dataframe[metric_column],
    )

    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)

    if "F1" in metric_column or metric_column == "Accuracy":
        ax.set_ylim(0, 1)

    for bar, value in zip(
        bars,
        dataframe[metric_column],
    ):
        ax.text(
            bar.get_x() +
            bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_training_time_chart(
    dataframe: pd.DataFrame,
    output_path: str,
):
    """
    Save trained-model preparation/training comparison.
    """
    trained = dataframe[
        dataframe["Model"] != "VADER"
    ]

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    bars = ax.bar(
        trained["Model"],
        trained[
            "Preparation + Training Time (seconds)"
        ],
    )

    ax.set_title(
        "Preparation and Training Time"
    )

    ax.set_xlabel(
        "Trained Model"
    )

    ax.set_ylabel(
        "Time in Seconds"
    )

    for bar, value in zip(
        bars,
        trained[
            "Preparation + Training Time (seconds)"
        ],
    ):
        ax.text(
            bar.get_x() +
            bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}s",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_inference_chart(
    dataframe: pd.DataFrame,
    output_path: str,
):
    """
    Save test-set inference-time comparison.
    """
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    bars = ax.bar(
        dataframe["Model"],
        dataframe[
            "Test Inference Time (seconds)"
        ],
    )

    ax.set_title(
        "Test-Set Inference Time"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        "Time in Seconds"
    )

    for bar, value in zip(
        bars,
        dataframe[
            "Test Inference Time (seconds)"
        ],
    ):
        ax.text(
            bar.get_x() +
            bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}s",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def calculate_roc_auc(
    y_true,
    y_probability,
) -> float:
    """
    Calculate ROC-AUC.
    """
    return roc_auc_score(
        y_true,
        y_probability,
    )


def calculate_average_precision(
    y_true,
    y_probability,
) -> float:
    """
    Calculate area under the Precision-Recall curve.
    """
    return average_precision_score(
        y_true,
        y_probability,
    )


def save_roc_curve(
    y_true,
    y_probability,
    output_path,
):
    """
    Save ROC curve.
    """
    fpr, tpr, _ = roc_curve(
        y_true,
        y_probability,
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.plot(
        fpr,
        tpr,
        label="Model",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random",
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC Curve"
    )

    ax.legend()

    plt.tight_layout()

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()