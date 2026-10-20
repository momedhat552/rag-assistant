import pytest

from rag import build_prompt, chunk_text, cosine_similarity


def test_chunk_text_overlap():
    assert chunk_text("ABCDEFGHIJKL", 5, 2) == ["ABCDE", "DEFGH", "GHIJK", "JKL"]


def test_chunk_text_short_text():
    assert chunk_text("hi", 500, 50) == ["hi"]


def test_chunk_text_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello", 5, 5)


def test_cosine_identical():
    assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_cosine_unrelated():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_zero_vector():
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_build_prompt_contains_question_and_context():
    chunks = [(0.9, {"source": "docker.txt", "text": "Docker packages apps."})]
    prompt = build_prompt("What is Docker?", chunks)
    assert "What is Docker?" in prompt
    assert "Docker packages apps." in prompt
    assert "docker.txt" in prompt