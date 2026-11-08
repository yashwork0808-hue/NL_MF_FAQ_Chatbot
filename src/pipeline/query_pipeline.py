from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import ConfigurationError, Settings, settings as default_settings
from src.llm.base import LLMClient, LLMResponse
from src.llm.factory import create_llm_client
from src.query.intent_gate import IntentGate, IntentResult, OUT_OF_SCOPE
from src.query.pii_guard import PIIGuard
from src.query.prompt_builder import PromptBuilder
from src.query.refusal import RefusalHandler
from src.query.retriever import RetrievalResult, Retriever
from src.query.trace import TraceWriter
from src.query.validator import AnswerValidator, ValidationResult


LOGGER = logging.getLogger(__name__)


@dataclass
class QueryResult:
    answer: str
    category: str
    reason: str
    source_url: str
    source_date: str
    trace_path: Optional[Path] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    generation_attempts: int = 0
    error: Optional[str] = None


class QueryPipeline:
    def __init__(
        self,
        settings: Settings = default_settings,
        retriever: Optional[Retriever] = None,
        llm: Optional[LLMClient] = None,
        intent_gate: Optional[IntentGate] = None,
        refusal_handler: Optional[RefusalHandler] = None,
        validator: Optional[AnswerValidator] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        pii_guard: Optional[PIIGuard] = None,
        trace_dir: Optional[Path] = None,
    ) -> None:
        self.settings = settings
        self.intent_gate = intent_gate or IntentGate()
        self.refusal_handler = refusal_handler or RefusalHandler()
        self.retriever = retriever or Retriever(settings=settings)
        self.validator = validator or AnswerValidator()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.pii_guard = pii_guard or PIIGuard()
        self.trace_writer = TraceWriter(trace_dir=trace_dir, settings=settings)
        self._llm = llm
        self._source_date_cache: Optional[str] = None

    def answer(self, question: str) -> QueryResult:
        trace: Dict[str, Any] = {
            "question": "",
            "pii": {},
            "intent": {},
            "retrieval": {},
            "llm": {"attempts": []},
            "validation": {"attempts": []},
        }
        pii = self.pii_guard.scan(question or "")
        safe_question = pii.redacted_text
        trace["question"] = safe_question
        trace["pii"] = {
            "detected": pii.has_pii,
            "types": list(pii.detected_types),
        }

        if pii.has_pii:
            source = self.intent_gate.detect_source(safe_question)
            intent = IntentResult(
                category=OUT_OF_SCOPE,
                reason="pii_detected",
                source=source,
                source_url=source.url if source else None,
                scheme=source.scheme if source else None,
            )
            trace["intent"] = self._intent_trace(intent)
            refusal = self.refusal_handler.build(
                intent,
                self._default_source_date(),
                category="pii",
                reason="pii_detected",
            )
            return self._finalize(
                answer=refusal.answer,
                category="pii",
                reason="pii_detected",
                source_url=refusal.source_url,
                source_date=refusal.source_date,
                trace=trace,
            )

        intent = self.intent_gate.classify(safe_question)
        trace["intent"] = self._intent_trace(intent)
        if intent.category != "factual":
            refusal = self.refusal_handler.build(
                intent,
                self._default_source_date(),
            )
            return self._finalize(
                answer=refusal.answer,
                category=intent.category,
                reason=intent.reason,
                source_url=refusal.source_url,
                source_date=refusal.source_date,
                trace=trace,
            )

        retrieval = self.retriever.retrieve(safe_question, source=intent.source)
        trace["retrieval"] = self._retrieval_trace(retrieval)
        if retrieval.citation is None:
            return self._fallback(
                intent=intent,
                retrieval=retrieval,
                reason="no_approved_citation",
                error="No retrieved chunk had an approved source URL",
                trace=trace,
            )

        try:
            bundle = self.prompt_builder.build(retrieval, safe_question)
        except Exception as exc:
            return self._fallback(
                intent=intent,
                retrieval=retrieval,
                reason="prompt_build_error",
                error=str(exc),
                trace=trace,
            )

        try:
            llm = self._get_llm()
        except Exception as exc:
            return self._fallback(
                intent=intent,
                retrieval=retrieval,
                reason="llm_unavailable",
                error=str(exc),
                trace=trace,
            )

        last_validation: Optional[ValidationResult] = None
        user_prompt = bundle.user_prompt
        for attempt in range(1, 3):
            attempt_trace: Dict[str, Any] = {"attempt": attempt}
            try:
                response: LLMResponse = llm.generate(
                    bundle.system_prompt,
                    user_prompt,
                )
            except Exception as exc:
                attempt_trace["error"] = str(exc)
                trace["llm"]["attempts"].append(attempt_trace)
                if attempt == 1:
                    continue
                return self._fallback(
                    intent=intent,
                    retrieval=retrieval,
                    reason="llm_error",
                    error=str(exc),
                    trace=trace,
                )

            attempt_trace.update(
                {
                    "provider": response.provider,
                    "model": response.model,
                    "latency_ms": response.latency_ms,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                }
            )
            trace["llm"]["attempts"].append(attempt_trace)
            trace["llm"].update(
                {
                    "provider": response.provider,
                    "model": response.model,
                }
            )

            validation = self.validator.validate(response.text, retrieval)
            last_validation = validation
            trace["validation"]["attempts"].append(
                {
                    "attempt": attempt,
                    "ok": validation.ok,
                    "citation_url": validation.citation_url,
                    "errors": list(validation.errors),
                }
            )
            if validation.ok:
                return self._finalize(
                    answer=validation.answer,
                    category="factual",
                    reason=intent.reason,
                    source_url=validation.citation_url or bundle.citation_url,
                    source_date=retrieval.source_date,
                    trace=trace,
                    llm_provider=response.provider,
                    llm_model=response.model,
                    generation_attempts=attempt,
                )

            if attempt == 1:
                user_prompt = self._repair_prompt(bundle.user_prompt, validation)

        errors = ", ".join(last_validation.errors) if last_validation else "unknown"
        return self._fallback(
            intent=intent,
            retrieval=retrieval,
            reason="validation_failed",
            error=f"Answer failed validation: {errors}",
            trace=trace,
            generation_attempts=2,
        )

    def _get_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = create_llm_client(self.settings)
        return self._llm

    def _default_source_date(self) -> str:
        if self._source_date_cache is not None:
            return self._source_date_cache
        try:
            metadatas = self.retriever.store.get_metadatas()
            dates = [
                str(metadata.get("scraped_date"))
                for metadata in metadatas
                if metadata.get("scraped_date")
            ]
            if dates:
                self._source_date_cache = max(dates)
                return self._source_date_cache
        except Exception:
            LOGGER.debug("Could not read source dates from the vector store")
        self._source_date_cache = self.settings.source_date
        return self._source_date_cache

    def _fallback(
        self,
        intent: IntentResult,
        retrieval: Optional[RetrievalResult],
        reason: str,
        error: str,
        trace: Dict[str, Any],
        generation_attempts: int = 0,
    ) -> QueryResult:
        if retrieval is not None and retrieval.citation is not None:
            source_url = str(retrieval.citation.metadata["source_url"])
            source_date = retrieval.source_date
        else:
            source_url = intent.source_url or self.refusal_handler.default_source.url
            source_date = self._default_source_date()
        answer = (
            "I’m unable to produce a verified answer from the indexed sources right "
            f"now. Please try again later, or refer to the official scheme page: "
            f"{source_url}.\n"
            f"Last updated from sources: {source_date}"
        )
        trace["fallback"] = {"reason": reason, "error": error}
        return self._finalize(
            answer=answer,
            category="fallback",
            reason=reason,
            source_url=source_url,
            source_date=source_date,
            trace=trace,
            generation_attempts=generation_attempts,
            error=error,
        )

    def _finalize(
        self,
        answer: str,
        category: str,
        reason: str,
        source_url: str,
        source_date: str,
        trace: Dict[str, Any],
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        generation_attempts: int = 0,
        error: Optional[str] = None,
    ) -> QueryResult:
        trace["final"] = {
            "category": category,
            "reason": reason,
            "source_url": source_url,
            "source_date": source_date,
            "answer": answer,
            "error": error,
        }
        trace_path: Optional[Path] = None
        try:
            trace_path = self.trace_writer.write(trace)
        except Exception:
            LOGGER.exception("Unable to write query trace")
        LOGGER.info("Query handled: category=%s reason=%s", category, reason)
        return QueryResult(
            answer=answer,
            category=category,
            reason=reason,
            source_url=source_url,
            source_date=source_date,
            trace_path=trace_path,
            llm_provider=llm_provider,
            llm_model=llm_model,
            generation_attempts=generation_attempts,
            error=error,
        )

    @staticmethod
    def _intent_trace(intent: IntentResult) -> Dict[str, Any]:
        return {
            "category": intent.category,
            "reason": intent.reason,
            "source_id": intent.source.source_id if intent.source else None,
            "source_url": intent.source_url,
            "scheme": intent.scheme,
        }

    @staticmethod
    def _retrieval_trace(retrieval: RetrievalResult) -> Dict[str, Any]:
        return {
            "filtered_by_scheme": retrieval.filtered_by_scheme,
            "chunk_ids": [chunk.chunk_id for chunk in retrieval.chunks],
            "scores": [round(chunk.score, 6) for chunk in retrieval.chunks],
            "citation_url": (
                retrieval.citation.metadata.get("source_url")
                if retrieval.citation
                else None
            ),
            "source_date": retrieval.source_date,
            "warnings": list(retrieval.warnings),
        }

    @staticmethod
    def _repair_prompt(user_prompt: str, validation: ValidationResult) -> str:
        errors = ", ".join(validation.errors)
        return (
            f"{user_prompt}\n\n"
            "Your previous answer was rejected by the safety validator. "
            f"Problems: {errors}. Return a corrected answer using only one approved "
            "source URL, at most three sentences, and the required final date line."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Answer a factual mutual fund FAQ question.",
    )
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="Directory for JSON traces (default: data/traces)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    pipeline = QueryPipeline(settings=default_settings, trace_dir=args.trace_dir)
    try:
        result = pipeline.answer(args.question)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    print(result.answer)
    if result.error:
        print(f"[{result.category}] {result.error}", file=sys.stderr)
    if result.trace_path is not None:
        print(f"Trace: {result.trace_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
