import json

from src.config import Settings
from src.llm.base import LLMClient, LLMResponse
from src.pipeline.query_pipeline import QueryPipeline
from src.query.retriever import RetrievalResult, RetrievedChunk


ELSS_URL = "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth"


class FakeRetriever:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def retrieve(self, question, source=None):
        self.calls += 1
        return self.result


class FakeLLM(LLMClient):
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def generate(self, system_prompt, user_prompt):
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return LLMResponse(text, "fake", "fake-model", 1.0)


def make_retrieval():
    chunk = RetrievedChunk(
        chunk_id="chunk-1",
        text="The ELSS scheme has a lock-in period of three years.",
        metadata={
            "source_url": ELSS_URL,
            "scheme": "HDFC ELSS Tax Saver Fund",
            "scraped_date": "2026-11-03",
        },
        distance=0.1,
        score=0.9,
    )
    return RetrievalResult(
        question="What is the ELSS lock-in period?",
        chunks=[chunk],
        citation=chunk,
        source_date="2026-11-03",
    )


def test_advisory_question_is_refused_without_llm(tmp_path):
    llm = FakeLLM(["unused"])
    pipeline = QueryPipeline(
        settings=Settings(source_updated_date="2026-11-03"),
        retriever=FakeRetriever(make_retrieval()),
        llm=llm,
        trace_dir=tmp_path,
    )

    result = pipeline.answer("Should I invest in HDFC Flexi Cap?")

    assert result.category == "advisory"
    assert llm.calls == 0
    assert ELSS_URL in result.answer or "hdfc-equity-fund" in result.answer
    assert result.answer.endswith("Last updated from sources: 2026-11-03")
    assert result.trace_path is not None


def test_pii_is_rejected_and_not_written_to_trace(tmp_path):
    pipeline = QueryPipeline(
        settings=Settings(source_updated_date="2026-11-03"),
        retriever=FakeRetriever(make_retrieval()),
        trace_dir=tmp_path,
    )

    result = pipeline.answer(
        "My PAN is ABCDE1234F and OTP is 123456. What is the ELSS lock-in period?"
    )

    assert result.category == "pii"
    assert "ABCDE1234F" not in result.answer
    trace_text = result.trace_path.read_text(encoding="utf-8")
    trace = json.loads(trace_text)
    assert "ABCDE1234F" not in trace_text
    assert trace["pii"]["detected"] is True
    assert trace["pii"]["types"] == ["PAN", "OTP"]


def test_factual_answer_is_validated(tmp_path):
    answer = f"The ELSS scheme has a lock-in period of three years. {ELSS_URL}"
    llm = FakeLLM([answer])
    pipeline = QueryPipeline(
        settings=Settings(source_updated_date="2026-11-03"),
        retriever=FakeRetriever(make_retrieval()),
        llm=llm,
        trace_dir=tmp_path,
    )

    result = pipeline.answer("What is the ELSS lock-in period?")

    assert result.category == "factual"
    assert result.generation_attempts == 1
    assert llm.calls == 1
    assert result.answer.endswith("Last updated from sources: 2026-11-03")


def test_invalid_answer_retries_then_falls_back(tmp_path):
    bad_answer = f"You should invest in this fund. {ELSS_URL}"
    llm = FakeLLM([bad_answer, bad_answer])
    pipeline = QueryPipeline(
        settings=Settings(source_updated_date="2026-11-03"),
        retriever=FakeRetriever(make_retrieval()),
        llm=llm,
        trace_dir=tmp_path,
    )

    result = pipeline.answer("What is the ELSS lock-in period?")

    assert result.category == "fallback"
    assert result.reason == "validation_failed"
    assert result.generation_attempts == 2
    assert llm.calls == 2
    assert ELSS_URL in result.answer
    assert result.answer.endswith("Last updated from sources: 2026-11-03")


def test_missing_key_returns_clear_fallback(tmp_path):
    pipeline = QueryPipeline(
        settings=Settings(
            source_updated_date="2026-11-03",
            groq_api_key=None,
        ),
        retriever=FakeRetriever(make_retrieval()),
        trace_dir=tmp_path,
    )

    result = pipeline.answer("What is the ELSS lock-in period?")

    assert result.category == "fallback"
    assert result.reason == "llm_unavailable"
    assert "GROQ_API_KEY" in result.error
    assert result.trace_path is not None
