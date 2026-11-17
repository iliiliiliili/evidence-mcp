import pytest

from evidence.models.ollama import extract_json


def test_plain_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_block():
    text = 'Here you go:\n```json\n{"a": [1, 2]}\n```\nDone.'
    assert extract_json(text) == {"a": [1, 2]}


def test_array():
    assert extract_json('[{"x": 1}]') == [{"x": 1}]


def test_embedded_in_prose():
    text = 'The answer is {"intent": "content", "confidence": 0.9} hope that helps'
    assert extract_json(text) == {"intent": "content", "confidence": 0.9}


def test_invalid_raises():
    with pytest.raises(ValueError):
        extract_json("no json here at all")
