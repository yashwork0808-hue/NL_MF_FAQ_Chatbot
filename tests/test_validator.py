from src.query.retriever import RetrievalResult, RetrievedChunk
from src.query.validator import AnswerValidator


ELSS_URL = "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth"
MID_CAP_URL = "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth"


def retrieval(url=ELSS_URL):
    chunk = RetrievedChunk(
        chunk_id="chunk-1",
        text="The ELSS scheme has a lock-in period of three years.",
        metadata={
            "source_url": url,
            "scheme": "HDFC ELSS Tax Saver Fund",
            "scraped_date": "2026-11-03",
        },
        distance=0.1,
        score=0.9,
    )
    return RetrievalResult(
        question="What is the lock-in period?",
        chunks=[chunk],
        citation=chunk,
        source_date="2026-11-03",
    )


def test_valid_answer_gets_date_line():
    answer = f"The ELSS scheme has a lock-in period of three years. {ELSS_URL}"
    result = AnswerValidator().validate(answer, retrieval())

    assert result.ok is True
    assert result.errors == []
    assert result.citation_url == ELSS_URL
    assert result.answer.endswith("Last updated from sources: 2026-11-03")


def test_missing_url_is_rejected():
    result = AnswerValidator().validate(
        "The lock-in period is three years.",
        retrieval(),
    )

    assert result.ok is False
    assert "missing_citation_url" in result.errors


def test_multiple_urls_are_rejected():
    answer = (
        f"The lock-in period is three years. {ELSS_URL} {MID_CAP_URL}"
    )
    result = AnswerValidator().validate(answer, retrieval())

    assert result.ok is False
    assert "expected_exactly_one_url_found_2" in result.errors


def test_unapproved_url_is_rejected():
    answer = "The lock-in period is three years. https://example.com/fund"
    result = AnswerValidator().validate(answer, retrieval())

    assert result.ok is False
    assert "url_not_approved" in result.errors


def test_too_many_sentences_are_rejected():
    answer = (
        "The fund has a lock-in period. It applies for three years. "
        f"This is a factual detail. Extra sentence. {ELSS_URL}"
    )
    result = AnswerValidator().validate(answer, retrieval())

    assert result.ok is False
    assert "too_many_sentences_4" in result.errors


def test_advice_language_is_rejected():
    answer = f"You should invest in this fund for three years. {ELSS_URL}"
    result = AnswerValidator().validate(answer, retrieval())

    assert result.ok is False
    assert "advice_language" in result.errors


def test_performance_language_is_rejected():
    answer = f"The fund produced strong returns last year. {ELSS_URL}"
    result = AnswerValidator().validate(answer, retrieval())

    assert result.ok is False
    assert "performance_language" in result.errors


def test_markdown_link_is_normalized():
    answer = f"The lock-in period is three years. [Source]({ELSS_URL})"
    result = AnswerValidator().validate(answer, retrieval())

    assert result.ok is True
    assert ELSS_URL in result.answer
    assert "](http" not in result.answer
