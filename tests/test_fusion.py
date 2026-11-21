from app.fusion import fuse


def test_concordant_positive():
    result = fuse("happy", 0.8, "positive", 0.9)
    assert result.state == "positive"
    assert result.score > 0.25


def test_concordant_negative():
    result = fuse("angry", 0.8, "negative", 0.9)
    assert result.state == "negative"
    assert result.score < -0.25


def test_mixed_case():
    result = fuse("angry", 0.9, "positive", 0.9)
    assert result.state == "mixed"
    assert result.score == 0.0


def test_invalid_alpha():
    import pytest
    with pytest.raises(ValueError):
        fuse("happy", 0.8, "positive", 0.9, alpha=1.1)
