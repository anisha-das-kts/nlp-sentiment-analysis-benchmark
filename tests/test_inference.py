from src.vader import build_vader, predict


def test_vader_returns_valid_labels():
    analyzer = build_vader()

    predictions = predict(
        [
            "This movie was wonderful.",
            "This movie was terrible.",
        ],
        analyzer,
    )

    assert len(predictions) == 2

    assert all(
        prediction in {"pos", "neg"}
        for prediction in predictions
    )