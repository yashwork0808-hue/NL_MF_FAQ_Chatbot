# PRD: Groww Mutual Fund FAQ Assistant

**Status:** Draft
**Source:** `docs/ProblemStatement.txt`
**Product type:** Retrieval-augmented generation (RAG) FAQ assistant
**Primary provider:** Groq

---

## 1. Overview

Build a small, facts-only FAQ assistant for Groww that answers factual questions about mutual fund schemes using only official public pages. The assistant demonstrates a complete, inspectable RAG pipeline — from data loading through LLM response generation and UI — with every stage's output stored explicitly.

The assistant answers questions about:

- Expense ratio
- Exit load
- Minimum SIP
- ELSS lock-in period
- Riskometer
- Benchmark
- How to download mutual fund statements

Every answer must include exactly one source link. The assistant must not provide investment advice.

---

## 2. Problem

Retail users comparing mutual fund schemes and support/content teams answering repetitive mutual fund questions need quick, trustworthy, factual answers. Existing general-purpose chatbots are risky here: they may hallucinate figures, cite unreliable third-party blogs, or drift into investment advice.

## 3. Goals

1. Answer factual mutual fund queries accurately using only five official Groww pages.
2. Always cite one clear source link per answer.
3. Politely refuse opinionated or portfolio-related questions and offer a relevant educational link.
4. Demonstrate each RAG stage explicitly (load → clean → chunk → embed → store → retrieve → generate → UI).
5. Keep the pipeline modular and configurable (embedding model, vector DB, LLM provider/model).

## 4. Non-Goals

- No investment advice, recommendations, or portfolio guidance.
- No return calculations, performance comparisons, or performance claims.
- No crawling beyond the five specified pages or following internal links.
- No handling or storage of PII (PAN, Aadhaar, account numbers, OTPs, email, phone).
- No third-party blogs, app screenshots, or non-public sources.
- No multi-AMC coverage in this version (HDFC Mutual Fund only).

---

## 5. Target Users

| User | Need |
|------|------|
| Retail investors comparing schemes | Quick factual answers with a trustworthy citation |
| Support / content teams | Reduce repetitive factual replies with a safe, citable assistant |

---

## 6. Scope

### 6.1 Data Sources (exactly five pages)

Scrape **only** these pages. Do not follow internal links or crawl other pages.

1. AMC Overview — HDFC Mutual Fund
   `https://groww.in/mutual-funds/amc/hdfc-mutual-funds`
2. Large Cap — HDFC Large Cap Fund Direct Growth
   `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth`
3. Flexi Cap — HDFC Flexi Cap Direct Plan Growth
   `https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth`
4. ELSS — HDFC ELSS Tax Saver Fund Direct Plan Growth
   `https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth`
5. Mid Cap — HDFC Mid Cap Fund Direct Growth
   `https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth`

### 6.2 Schemes Covered

HDFC AMC overview, HDFC Large Cap Fund, HDFC Flexi Cap Fund, HDFC ELSS Tax Saver Fund, HDFC Mid Cap Fund.

---

## 7. Functional Requirements

### 7.1 Question Answering

| ID | Requirement |
|----|-------------|
| FR-1 | Accept a free-text factual question about the covered schemes. |
| FR-2 | Retrieve the most relevant content from the indexed corpus. |
| FR-3 | Answer in **three sentences or fewer**. |
| FR-4 | Include exactly **one** source link in every answer. |
| FR-5 | Append `Last updated from sources: [date]` to answers. |
| FR-6 | Handle the example questions below. |

**Example questions:**

- What is the expense ratio?
- What is the ELSS lock-in period?
- What is the minimum SIP?
- What is the exit load?
- What is the riskometer or benchmark?
- How do I download my capital-gains statement?

### 7.2 Safe Refusals

| ID | Requirement |
|----|-------------|
| FR-7 | Detect opinionated, advisory, or portfolio-related questions. |
| FR-8 | Refuse politely without giving advice. |
| FR-9 | Provide a relevant educational link when refusing. |

**Examples of questions to refuse:** "Should I invest in HDFC Flexi Cap?", "Which fund will give me the best returns?", "Is this fund good for my portfolio?"

### 7.3 Performance Requests

| ID | Requirement |
|----|-------------|
| FR-10 | Do not calculate or compare investment returns. |
| FR-11 | If performance information is requested, link to the official factsheet. |

### 7.4 Privacy / PII

| ID | Requirement |
|----|-------------|
| FR-12 | Do not accept or store PAN, Aadhaar, account numbers, OTPs, email addresses, or phone numbers. |
| FR-13 | Do not log, persist, or transmit PII entered by users. |

### 7.5 User Interface

| ID | Requirement |
|----|-------------|
| FR-14 | Show a welcome message. |
| FR-15 | Show three example questions. |
| FR-16 | Provide a question input box. |
| FR-17 | Provide an answer area displaying the answer, source link, and updated date. |
| FR-18 | Display the note: "Facts-only. No investment advice." |

---

## 8. RAG Pipeline Requirements

The project must demonstrate the complete pipeline, storing the output of every stage explicitly.

| Stage | Requirement |
|-------|-------------|
| 1. Data loading | Collect and load the five specified source documents. |
| 2. Data cleaning | Normalize extracted text; store cleaned output. |
| 3. Data chunking | Apply a chunking strategy chosen for the data; document the choice and store chunks. |
| 4. Data embedding | Embed chunks with `sentence-transformers/all-MiniLM-L6-v2`. Store embeddings. |
| 5. Vector DB storage | Store embeddings in ChromaDB; persist the store. |
| 6. Data retrieval | Retrieve using the **same** embedding model and vector DB used at ingestion. |
| 7. LLM generation | Generate answers via Groq; store prompts/responses. |
| 8. User interface | Expose the assistant through the simple UI. |

### 8.1 Configurability

The following must be configurable (not hard-coded) so they can change later:

- Embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- Vector database (default: ChromaDB)
- LLM provider and model (default: Groq)

### 8.2 LLM Provider

- **Primary:** Groq, with `GROQ_API_KEY` in `.env`.
- **Model:** `qwen/qwen3.8-27b` (as specified in the trainer's brief).
- **Also support:** Gemini, OpenAI, Anthropic, Mistral.
- Model/provider selection must be flexible and configurable.

> Note: `qwen/qwen3.8-27b` is specified verbatim in the brief. If the provider rejects this model ID, surface a clear configuration error rather than silently substituting another model.

---

## 9. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-1 | Sources must be public only; no app-backend screenshots or third-party blogs. |
| NFR-2 | Answers must be within three sentences and always cite one source. |
| NFR-3 | Pipeline must be modular, understandable, and demonstrable stage by stage. |
| NFR-4 | Secrets (API keys) must live in `.env`, never in source control. |
| NFR-5 | Usable with no more than the five indexed documents; small-corpus retrieval. |
| NFR-6 | Every stage's artifacts must be persisted for inspection/demo. |

---

## 10. Deliverables

1. Working prototype link (app/notebook) **or** a demo video of up to three minutes if hosting is not possible.
2. Source list (CSV/MD) containing 15–25 URLs used.
3. `README` with setup steps, project scope (AMC + schemes), and known limitations.
4. Sample Q&A file with 5–10 queries, assistant answers, and links.
5. Disclaimer snippet used in the UI (facts-only, no investment advice).

---

## 11. Success Criteria

- All six example question types produce correct, concise answers with one valid source link.
- Advisory/portfolio questions are refused politely with an educational link.
- No PII is accepted, stored, or logged.
- No return calculations or performance comparisons are produced.
- Every RAG stage's output is stored and can be demonstrated independently.
- Embedding model, vector DB, and LLM provider/model are configurable, and ingestion/retrieval use matching settings.

---

## 12. Out of Scope

- Multi-AMC or multi-scheme expansion beyond the five pages.
- Live price, NAV tracking, or performance analytics.
- Transaction flows, account linking, or personalization.
- User authentication and persistent chat history.

---

## 13. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Groww page layout changes break scraping | Isolate scraping in Phase 1; store raw HTML/text artifacts and validate selectors. |
| LLM hallucinates figures | Constrain generation to retrieved context; require citation; keep answers short. |
| Model ID `qwen/qwen3.8-27b` unavailable on Groq | Fail with a clear config error; keep model configurable. |
| Advisory drift | Explicit refusal prompt + intent check before answering; educational link fallback. |
| PII leakage | Input validation/sanitization; no logging of user-entered sensitive fields. |
| Stale source content | Surface `Last updated from sources: [date]` on every answer. |

---

## 14. Skills Being Tested

- **W1 — Thinking Like a Model:** Identify the exact fact asked and decide whether to answer or refuse.
- **W2 — LLMs & Prompting:** Instruction style, concise phrasing, polite safe refusals, citation wording.
- **W3 — RAGs:** Small-corpus retrieval with accurate citations from AMC/SEBI/AMFI pages.

---

## 15. Implementation Phases

| Phase | Description |
|-------|-------------|
| Phase 1 — Data Loading | Collect and load the specified source documents. |
| Phase 2 — Data Chunking | Choose and document a chunking strategy for the data. |
| Phase 3 — Data Embedding | Embed with `sentence-transformers/all-MiniLM-L6-v2`. |
| Phase 4 — Vector Database | Store in ChromaDB with configurable embedding model and vector DB. |
| Phase 5 — Retrieval | Retrieve using the same embedding model and vector DB as ingestion. |
| Phase 6 — Generation | Generate answers with Groq (`qwen/qwen3.8-27b`), keeping providers flexible. |
| Phase 7 — UI | Ship the simple interface with welcome, examples, input, answer + source, disclaimer. |

---

## 16. Open Questions

1. Should the source list include AMC/SEBI/AMFI pages beyond the five Groww pages for educational-refusal links, or remain strictly the five sources?
2. Where will the prototype be hosted, or is a demo video the chosen deliverable?
3. What update cadence should drive the `Last updated from sources` date?
4. Which educational link(s) should refusals point to by default?
