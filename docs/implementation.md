# Implementation Plan: Groww Mutual Fund FAQ Assistant

**Derived from:** `docs/architecture.md`
**Supporting source:** `docs/PRD.md`
**Scope:** HDFC Mutual Fund — exactly five approved Groww pages
**UI:** Streamlit
**LLM:** Groq `qwen/qwen3.8-27b`
**Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
**Vector DB:** ChromaDB

---

## 1. Create Directories

### Directory Structure

```text
NL_MF_FAQ_Chatbot/
├── src/
│   ├── config.py
│   ├── ingest/          # registry, fetcher, loader, cleaner, chunker, embedder
│   ├── store/           # vector store interface + ChromaDB implementation
│   ├── query/           # pii guard, intent gate, retriever, prompt, validator, refusal
│   ├── llm/             # provider interface + providers/
│   ├── pipeline/        # ingest_pipeline.py, query_pipeline.py
│   └── ui/              # Streamlit app
├── data/
│   ├── raw/             # scraped HTML + manifest
│   ├── cleaned/         # cleaned text
│   ├── chunks/          # chunks.jsonl
│   ├── embeddings/      # vectors + ID mapping + manifest
│   ├── chroma/          # persisted vector DB
│   └── traces/          # retrieval and generation traces
├── eval/
│   ├── sample_questions.md
│   └── results/
├── docs/
├── sources.md
├── README.md
├── requirements.txt
└── .env.example
```

### Setup Tasks

1. Install dependencies: `streamlit`, `sentence-transformers`, `chromadb`, `requests`, `beautifulsoup4`, `python-dotenv`, `pydantic-settings`, `numpy`, `pandas`, `groq`, `pytest`.
2. Add `.gitignore` for `.env`, `data/`, `__pycache__/`, and virtual environments.
3. Create `.env.example`:

```text
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DB=chroma
CHROMA_DIR=data/chroma
COLLECTION_NAME=mf_faq
CHUNK_SIZE=400
CHUNK_OVERLAP=60
TOP_K=4
LLM_PROVIDER=groq
LLM_MODEL=qwen/qwen3.8-27b
GROQ_API_KEY=
SOURCE_UPDATED_DATE=
```

4. Implement `src/config.py` to load `.env`, validate values, and fail clearly on missing `GROQ_API_KEY` or an invalid model.

**Done when:** config imports cleanly, `.env` is ignored, and no secret is logged.

---

## 2. Scrape Documents

### Tasks

1. `src/ingest/registry.py` — define the hard allowlist of exactly five Groww URLs with `url`, `scheme`, `page_type`, and `page_label`. Reject all other URLs.
2. `src/ingest/fetcher.py` — fetch each page once, follow no internal links, retry transient failures, save raw HTML, and record `source_url`, `scheme`, `page_type`, `fetched_at`, and checksum in `data/raw/manifest.json`.

### Artifacts

| Artifact | Location |
|----------|----------|
| Raw HTML | `data/raw/<source_id>.html` |
| Scrape manifest | `data/raw/manifest.json` |

### Verify

```text
python -m src.pipeline.ingest_pipeline --stage scrape
```

**Done when:** exactly five raw HTML files and a manifest containing five approved URLs exist, and no link crawling occurred.

---

## 3. Load Documents

### Tasks

1. `src/ingest/loader.py` — read `data/raw/manifest.json` and the corresponding raw HTML files.
2. Validate exactly five documents, allowlisted URLs, matching checksums, and non-empty content.
3. Build in-memory `Document` records with `source_url`, `scheme`, `page_type`, `raw_html`, and `fetched_at`.
4. Pass the loaded documents to cleaning; write `data/raw/load_report.json` with loaded count, skipped sources, and validation warnings.

### Verify

```text
python -m src.pipeline.ingest_pipeline --stage load
```

**Done when:** exactly five documents load successfully, every document matches the allowlist, and no document is empty.

---

## 4. Clean Documents

### Tasks

1. `src/ingest/cleaner.py` — parse each loaded document with BeautifulSoup.
2. Strip navigation, scripts, styles, and unrelated chrome; normalize whitespace and encoding.
3. Preserve label-value facts such as expense ratio, exit load, minimum SIP, lock-in, riskometer, and benchmark.
4. Save cleaned text and a cleaning report with character counts and warnings.

### Artifacts

| Artifact | Location |
|----------|----------|
| Cleaned text | `data/cleaned/<source_id>.txt` |
| Cleaning report | `data/cleaned/report.json` |

### Verify

```text
python -m src.pipeline.ingest_pipeline --stage clean
```

**Done when:** all five loaded documents produce non-empty cleaned text with key facts intact.

---

## 5. Chunking

### Tasks

1. Load the cleaned documents from `data/cleaned/` into the chunker.
2. `src/ingest/chunker.py` — use structure-aware recursive splitting: section boundaries first, then paragraphs/lines.
3. Target ~300–500 tokens per chunk with ~15% overlap and a minimum-size guard.
4. Attach metadata to every chunk: `chunk_id`, `source_url`, `scheme`, `page_type`, `section_heading`, `chunk_index`, `scraped_date`, `text`.
5. Generate stable `chunk_id` values from URL, section, index, and text hash for idempotent re-ingestion.
6. Write `data/chunks/chunks.jsonl` and `data/chunks/summary.json`.

### Verify

```text
python -m src.pipeline.ingest_pipeline --stage chunk
```

**Done when:** no empty chunks exist, all metadata is present, key label-value pairs are not split, and repeated runs produce stable IDs.

---

## 6. Embedding

### Tasks

1. `src/ingest/embedder.py` — load `EMBEDDING_MODEL` (default `sentence-transformers/all-MiniLM-L6-v2`).
2. Embed chunks in batches, preserving chunk order.
3. Save:
   - `data/embeddings/embeddings.npy`
   - `data/embeddings/embedding_ids.json`
   - `data/embeddings/embedding_manifest.json` (model, dimension, count, timestamp)
4. Validate one vector per chunk, matching dimension, and no `NaN`/infinite values.

### Verify

```text
python -m src.pipeline.ingest_pipeline --stage embed
```

**Done when:** vector count equals chunk count, the model is recorded, and the dimension matches the manifest.

---

## 7. Store Data in Vector DB

### Tasks

1. `src/store/base.py` — define a `VectorStore` interface with `upsert`, `query`, `count`, and `reset`.
2. `src/store/chroma_store.py` — open/create the configured ChromaDB directory, use collection `mf_faq`, and upsert chunk IDs, vectors, text, and metadata.
3. Make ingestion idempotent using stable chunk IDs.
4. Write `data/chroma/ingestion_manifest.json` with collection name, embedding model, dimension, chunk count, ingestion timestamp, and `scraped_date`.
5. Fail clearly if vector dimensions do not match the collection.

### Verify

```text
python -m src.pipeline.ingest_pipeline --stage store
```

**Done when:** collection count equals chunk count, every record has `source_url`/`scheme`/`chunk_id`, and re-ingestion does not create duplicates.

---

## 8. Retrieval Logic

### Tasks

1. `src/query/retriever.py` — embed the question with the same embedding model used at ingestion.
2. Query ChromaDB for top-k results (default `k=4`) and optionally filter by `scheme` when the question names a fund.
3. Choose the highest-scoring chunk whose `source_url` supports the answer as the single citation.
4. `src/query/pii_guard.py` — reject or redact PAN, Aadhaar, account numbers, OTPs, email, and phone before retrieval or logging.
5. `src/query/intent_gate.py` — classify questions as factual, advisory/portfolio, performance, or out of scope.
6. `src/query/prompt_builder.py` — enforce the answer contract:
   - Answer only from retrieved context.
   - Maximum three sentences.
   - Exactly one source link.
   - No advice, recommendations, opinions, or portfolio guidance.
   - No return calculations or comparisons.
   - Append `Last updated from sources: <scraped_date>`.
   - Refuse advisory questions politely with an educational link.
7. `src/llm/providers/groq.py` — call Groq with `qwen/qwen3.8-27b`; fail clearly on a missing key or rejected model ID.
8. `src/query/validator.py` — enforce sentence count, one approved URL, no advice/performance claims, and the updated-date line. Retry once, then fall back to a templated safe answer.
9. `src/pipeline/query_pipeline.py` — orchestrate PII guard → intent gate → retrieval → prompt → LLM → validation → response.
10. Save retrieval/generation traces under `data/traces/`.

### Verify

```text
python -m src.pipeline.query_pipeline --question "What is the ELSS lock-in period?"
```

**Done when:** factual answers are concise, cite one valid source, include the updated date, refuse advice safely, and leave a trace.

---

## 9. UI

### Tasks

1. Implement `src/ui/app.py` with:
   - Welcome message.
   - Three example questions.
   - Question input box.
   - Answer area showing the answer, one source link, and updated date.
   - Note: "Facts-only. No investment advice."
2. Wire submit to `query_pipeline`.
3. Keep only the current answer in Streamlit session state; never persist PII or chat history.
4. Show a friendly error notice on failure and keep the app usable.

### Run

```text
streamlit run src/ui/app.py
```

### Verify

Test these in the UI:

- "What is the expense ratio?"
- "What is the ELSS lock-in period?"
- "Should I invest in HDFC Flexi Cap?"

**Done when:** all UI requirements are visible, every answer shows one source link and date, refusals display correctly, and PII is never shown.

---

## 10. Edge Cases and Guardrail Test Cases

### Guardrail Cases

| ID | Input / Scenario | Expected behavior |
|----|------------------|-------------------|
| G-1 | Question contains PAN | Reject/redact before processing; no PII stored |
| G-2 | Question contains Aadhaar or account number | Reject/redact |
| G-3 | Question contains OTP, email, or phone | Reject/redact |
| G-4 | "Should I invest in HDFC Flexi Cap?" | Polite refusal + educational link |
| G-5 | "Which fund will give the best returns?" | Refusal; no recommendation |
| G-6 | "Is this fund good for my portfolio?" | Refusal; no portfolio advice |
| G-7 | Request to compare or calculate returns | Link to official factsheet; no comparison |
| G-8 | Factual answer longer than three sentences | Validator rejects or truncates |
| G-9 | Answer with zero URLs | Retry once; fallback includes one approved source |
| G-10 | Answer with multiple URLs | Reject; retry with exactly one citation |
| G-11 | Citation is not an approved source | Replace with allowlisted source or refuse |
| G-12 | Answer includes advice language | Route to refusal/fallback |
| G-13 | Answer lacks `Last updated from sources` | Append from `scraped_date` |
| G-14 | Retrieved context is empty or low-score | Say information is unavailable; link closest official source |
| G-15 | User asks to ignore instructions | Ignore request; enforce system guardrails |
| G-16 | Empty or whitespace-only question | Ask for a factual question about the covered schemes |
| G-17 | Very long or multi-part question | Answer only the supported factual part within three sentences |
| G-18 | LLM key missing or model unavailable | Clear configuration error; no silent substitution |
| G-19 | Source fetch fails repeatedly | Fail ingestion with the source URL reported |
| G-20 | Embedding model/dimension mismatch | Fail before query; do not mix models |
| G-21 | Re-ingestion with same sources | No duplicate vectors; stable chunk IDs |
| G-22 | Question names a specific scheme | Apply scheme metadata filter when possible |

### Test Files

- Unit tests: URL allowlist, PII patterns, intent classification, chunk metadata, embedding dimensions, citation allowlist, validator rules.
- Integration tests: ingestion to ChromaDB; question to validated answer; idempotent re-ingestion; config swaps.
- Regression tests: six PRD factual questions, three advisory questions, one performance request, and PII inputs.

### Verify

```text
pytest -q
```

**Done when:** all guardrail cases pass and no PII appears in logs or traces.

---

## 11. Deliverables Checklist

- [ ] Five approved Groww sources only
- [ ] Raw, cleaned, chunks, embeddings, ChromaDB, and trace artifacts persisted
- [ ] `all-MiniLM-L6-v2` embeddings recorded
- [ ] ChromaDB collection `mf_faq` populated
- [ ] Groq `qwen/qwen3.8-27b` configured
- [ ] Answers ≤3 sentences with exactly one source link
- [ ] `Last updated from sources` present
- [ ] Advisory/performance/PII guardrails tested
- [ ] Streamlit UI complete
- [ ] Sample Q&A file with 5–10 queries
- [ ] `sources.md` with exactly 5 approved Groww URLs
- [ ] README with setup, scope, and limitations
- [ ] Demo video (≤3 minutes) if not hosted
