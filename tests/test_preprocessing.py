from src.preprocessing import clean_text


def test_lowercase():
    result = clean_text(
        "THIS Movie Was GREAT"
    )

    assert result == result.lower()


def test_negation_is_preserved():
    result = clean_text(
        "This movie was not good."
    )

    assert "not" in result


def test_punctuation_is_removed():
    result = clean_text(
        "Amazing!!! Movie???"
    )

    assert "!" not in result
    assert "?" not in result


def test_empty_input():
    assert clean_text("") == ""