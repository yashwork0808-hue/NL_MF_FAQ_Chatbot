from __future__ import annotations

import re

import streamlit as st

from src.config import settings
from src.pipeline.query_pipeline import QueryPipeline, QueryResult


EXAMPLE_QUESTIONS = (
    "What is the expense ratio?",
    "What is the ELSS lock-in period?",
    "Should I invest in HDFC Flexi Cap?",
)
URL_PATTERN = re.compile(r"https?://[^\s)\]>,]+", re.IGNORECASE)
DATE_LINE_PATTERN = re.compile(
    r"(?im)^\s*last updated from sources\s*:.*$",
    re.IGNORECASE,
)


@st.cache_resource(show_spinner=False)
def get_pipeline() -> QueryPipeline:
    return QueryPipeline(settings=settings)


def display_body(answer: str) -> str:
    body = DATE_LINE_PATTERN.sub("", answer or "").strip()
    body = re.sub(
        r"(?im)^\s*(?:source|link)\s*:\s*https?://\S+\s*$",
        "",
        body,
    ).strip()
    body = re.sub(
        r"(?i)\b(please see|refer to|visit|available at)\s*:?\s*https?://\S+\s*$",
        r"\1 the official source page",
        body,
    )
    body = re.sub(r"\s*https?://\S+\s*$", "", body).strip()
    body = URL_PATTERN.sub("the official source page", body)
    body = re.sub(r"[:\-–—]\s*$", "", body).strip()
    if body and body[-1] not in ".!?":
        body += "."
    return body


def show_answer(result: QueryResult) -> None:
    if result.category == "fallback":
        if result.reason == "llm_unavailable":
            st.warning(
                "The answer service is not configured right now. You can still use "
                "the official source link below."
            )
        else:
            st.warning(
                "I couldn’t produce a verified answer from the indexed sources. "
                "Please try again or use the official source link below."
            )
    st.markdown(display_body(result.answer))
    st.link_button("View official source", result.source_url)
    st.caption(f"Last updated from sources: {result.source_date}")


st.set_page_config(page_title="Groww Mutual Fund FAQ Assistant")
st.title("Groww Mutual Fund FAQ Assistant")
st.write(
    "Ask a factual question about the covered HDFC Mutual Fund schemes. "
    "Answers use only the indexed Groww pages and include one official source."
)
st.info("Facts-only. No investment advice.")

if "current_result" not in st.session_state:
    st.session_state.current_result = None

st.subheader("Example questions")
example_columns = st.columns(len(EXAMPLE_QUESTIONS))
for column, example in zip(example_columns, EXAMPLE_QUESTIONS):
    with column:
        if st.button(example, use_container_width=True):
            st.session_state.question = example
            st.session_state.current_result = None

with st.form("question_form", clear_on_submit=False):
    question = st.text_input(
        "Your question",
        key="question",
        placeholder="e.g. What is the exit load for HDFC Large Cap Fund?",
    )
    submitted = st.form_submit_button("Ask")

if submitted:
    if not question.strip():
        st.warning("Please enter a factual question about the covered schemes.")
    else:
        with st.spinner("Checking approved sources..."):
            try:
                st.session_state.current_result = get_pipeline().answer(question)
            except Exception:
                st.session_state.current_result = None
                st.error(
                    "I couldn’t process that request. Please try again without "
                    "personal details."
                )

if st.session_state.current_result is not None:
    show_answer(st.session_state.current_result)

st.divider()
st.caption(
    "Scope: five approved HDFC Mutual Fund pages on Groww. "
    "No account access, personal data, returns calculations, or recommendations."
)
