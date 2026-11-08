# Groww Mutual Fund FAQ Assistant

Facts-only RAG assistant for five approved HDFC Mutual Fund pages on Groww. It answers factual questions, refuses investment advice and performance comparisons, rejects PII, and cites exactly one approved source.

## Scope

- Five fixed Groww sources only; no crawling or third-party citations.
- Topics: expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark, fund facts, and statement downloads.
- Streamlit UI only.
- No account access, personal data storage, return calculations, recommendations, or portfolio guidance.

The source allowlist is documented in [`sources.md`](sources.md) and enforced in `src/ingest/registry.py`.

## Stack

| Layer | Choice |
|-------|--------|
| UI | Streamlit |
| LLM | Groq `qwen/qwen3.8-27b` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB collection `mf_faq` |
| Ingestion | requests + BeautifulSoup |

## Setup

```text
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `GROQ_API_KEY` in `.env`. Keep `.env` private; it is excluded by `.gitignore`. If the configured Groq model is unavailable in your project, update `LLM_MODEL` to an available model.

## Build the Index

Run the stages in order:

```text
python -m src.pipeline.ingest_pipeline --stage scrape
python -m src.pipeline.ingest_pipeline --stage load
python -m src.pipeline.ingest_pipeline --stage clean
python -m src.pipeline.ingest_pipeline --stage chunk
python -m src.pipeline.ingest_pipeline --stage embed
python -m src.pipeline.ingest_pipeline --stage store
```

Artifacts are stored under `data/`:

- `data/raw/`: raw HTML and scrape manifest
- `data/cleaned/`: cleaned text and cleaning report
- `data/chunks/`: `chunks.jsonl` and summary
- `data/embeddings/`: vectors, ID map, manifest
- `data/chroma/`: persisted ChromaDB collection
- `data/traces/`: sanitized query traces

Chunk IDs are stable across re-ingestion, and stale vectors are pruned from the collection.

## Query from the CLI

```text
python -m src.pipeline.query_pipeline --question "What is the ELSS lock-in period?"
```

The pipeline runs PII detection, intent classification, retrieval, prompt construction, Groq generation, validation, and trace writing. Invalid answers are retried once and then replaced with a templated safe answer. A missing key or unavailable model produces a clear fallback and trace error.

## Run the UI

```text
streamlit run src/ui/app.py
```

The UI provides a welcome message, three example questions, a question box, the answer, one source link, the updated date, and the note `Facts-only. No investment advice.` It keeps only the current result in session state; there is no chat history.

## Tests

```text
pytest -q
```

Tests cover PII patterns, intent classification, validator rules, citation allowlisting, retry/fallback behavior, and sanitized traces. Reference Q&A examples are in [`eval/sample_questions.md`](eval/sample_questions.md).

## Guardrails

- PII is detected before retrieval and redacted from traces: PAN, Aadhaar, account numbers, OTPs, email, and phone.
- Advisory, portfolio, performance, comparison, and prompt-injection requests are refused.
- Answers are limited to three sentences, one approved URL, and a final `Last updated from sources: <scraped_date>` line.
- The validator rejects advice/performance language, unapproved or multiple URLs, and excessive sentences.
- Traces store sanitized questions and derived fields, never raw PII.

## Limitations

- Answers depend on the indexed pages and Groq availability.
- The cleaner may not preserve every page section; the scheme riskometer is not always present as cleaned text.
- The indexed pages do not include a step-by-step capital-gains statement download guide, so that question returns a safe unavailable answer.
- The assistant does not access accounts, transactions, statements, or personal holdings.
- This is an informational FAQ tool, not investment advice.
