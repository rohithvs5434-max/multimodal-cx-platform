import pytest

from emotion_model.ravdess import parse_ravdess, speaker_split


def test_parse_ravdess_speech_filename() -> None:
    result = parse_ravdess("03-01-05-01-02-01-12.wav")
    assert result.emotion == "angry"
    assert result.speaker == 12


def test_speaker_split_is_deterministic_and_exclusive() -> None:
    assert speaker_split(1) == "train"
    assert speaker_split(6) == "validation"
    assert speaker_split(9) == "test"


def test_invalid_filename_is_rejected() -> None:
    with pytest.raises(ValueError):
        parse_ravdess("not-a-ravdess-file.wav")
