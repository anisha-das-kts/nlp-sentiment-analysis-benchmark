from src.evaluation import calculate_metrics


def test_metrics_are_perfect():
    y_true = [
        "neg",
        "pos",
        "neg",
        "pos",
    ]

    y_pred = [
        "neg",
        "pos",
        "neg",
        "pos",
    ]

    metrics = calculate_metrics(
        y_true,
        y_pred,
        positive_label="pos",
    )

    assert metrics["Accuracy"] == 1.0
    assert metrics["Macro F1"] == 1.0
    assert metrics["Positive-Class F1"] == 1.0


def test_metrics_are_between_zero_and_one():
    y_true = [
        "neg",
        "pos",
        "neg",
        "pos",
    ]

    y_pred = [
        "pos",
        "pos",
        "neg",
        "neg",
    ]

    metrics = calculate_metrics(
        y_true,
        y_pred,
        positive_label="pos",
    )

    for value in metrics.values():
        assert 0.0 <= value <= 1.0