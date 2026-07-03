# CheatSheet RAG — Build Plan
**Project:** Interactive Q&A over the Kaggle "Data Science Cheat Sheets" dataset
**Pattern:** Same as your reference card — Problem → What to Build → How it Works → Use Cases
**Builder:** You (architect/PM) directing Google Antigravity (execution agent)

---

## 0. Reality Check on the Dataset First

`timoboz/data-science-cheat-sheets` = **251 files, ~625MB**, sourced from the `abhat222/Data-Science--Cheat-Sheet` GitHub repo. Two things matter before you write a single prompt to Antigravity:

1. **It's mostly PDFs, and many are dense single-page infographics** (Pandas, Seaborn, Scikit-learn, Neural Networks cheat sheets) — not clean paragraph text. Plain `pypdf` text extraction on these will produce garbled, out-of-order text because layout ≠ reading order.
2. **625MB / 251 files is too much for an MVP.** Don't ingest the whole thing on day one — it burns embedding calls, agent tokens, and debugging time on files you don't need yet.

**Decision: build in two tiers.**
- **Tier 1 (MVP, build this first):** ~20–25 hand-picked high-value cheat sheets — Pandas, NumPy, Scikit-learn, SQL, Matplotlib/Seaborn, Probability & Statistics, Linear Algebra, Deep Learning, Git, Python basics. This is small enough to iterate fast and cheap.
- **Tier 2 (scale-up, only after MVP works):** the remaining corpus, added via the same pipeline once it's proven.

This alone will save you the most tokens/credits — both Antigravity's and any embedding API you use.

---

## 1. Architecture (maps to your "How It Works" card)

```
User Question
     │
     ▼
[1] Query received (chat UI)
     │
     ▼
[2] Embed query → similarity search in vector DB
     │             (filtered by metadata: topic/source file if detected)
     ▼
[3] Top-k relevant chunks retrieved (with source + page metadata)
     │
     ▼
[4] LLM generates answer, grounded ONLY in retrieved chunks,
     with citation of which cheat sheet / section it came from
     │
     ▼
Answer + source reference shown to user
```

### Pipeline stages (offline, run once per tier)
```
Raw PDFs (251 files)
   → [A] Extraction (text + layout-aware / OCR for infographic PDFs)
   → [B] Chunking (semantic/section-based, not fixed-token blind split)
   → [C] Metadata tagging (topic, filename, page, section header)
   → [D] Embedding (batch, cached — never re-embed unchanged files)
   → [E] Vector store (Chroma, local, persisted to disk)
```

---

## 2. Tech Stack (optimized for cost + your skill level)

| Layer | Recommendation | Why |
|---|---|---|
| Extraction | `PyMuPDF (fitz)` for text-based PDFs; `pdf2image` + `pytesseract` OCR for infographic-style sheets that are basically images | Cheat sheets are visually dense — pure text extraction loses table/diagram structure. Tesseract is local and free, no vendor model needed |
| Chunking | Section/heading-aware splitting (fall back to ~500-token overlapping chunks where no clear structure exists) | Cheat sheets have natural sections (e.g. "groupby", "joins") — chunk on those, not blindly every N tokens |
| Embeddings | **Local, free**: `sentence-transformers` (`bge-small-en-v1.5` or `all-MiniLM-L6-v2`) run on CPU | Avoids burning any paid API quota on ~250 files re-embedded during dev iteration |
| Vector DB | **Chroma** (local, file-based, zero cost) | Simplest to set up, no server, plays nicely with an agent building it in one pass |
| Retrieval | Top-k (start k=4) + metadata filter | Keep prompt small = fewer generation tokens |
| Generation LLM | **Claude API** (Anthropic) — your own key, your own model choice | This is the model actually answering the user's question, so it should be the one you deliberately chose, not whatever ships free with the IDE |
| UI | **Streamlit** (fastest to build, good enough for a portfolio demo) | You've done Power BI/dashboards before — Streamlit is the Python-native equivalent for this |
| Orchestration | Plain Python + LangChain **or** no framework at all (manual retrieval + prompt) | For a project this size, a framework adds token overhead for Antigravity to reason about. A ~150-line manual pipeline is easier for an agent to build correctly in one pass and easier for you to explain in interviews. |

**Recommendation: skip LangChain for this build.** Manual pipeline = fewer moving parts, fewer agent tool-calls, fully explainable by you in interviews (important since your target roles value "I understand what I built," not "I imported a black box").

---

## 3. Build Phases (give these to Antigravity one at a time — NOT all at once)

Feeding the whole plan as a single mega-prompt is the #1 way agentic IDEs blow through tokens: they either try to do everything in one giant uncontrolled pass, or lose context mid-way and redo work. Break it into checkpointed phases, review each output, then proceed.

### Phase 1 — Project scaffold + extraction (Tier 1 only, 20-25 files)
- Set up Python project structure, `requirements.txt`
- Script to extract text from the 20-25 selected PDFs (PyMuPDF, with a fallback flag for image-heavy pages)
- Output: raw extracted text saved to `/data/extracted/*.txt` with source filename preserved

### Phase 2 — Chunking + metadata
- Section-aware chunker
- Attach metadata: `{source_file, topic, chunk_id}`
- Output: `/data/chunks.jsonl`

### Phase 3 — Embedding + vector store
- Load `bge-small-en-v1.5` locally
- Embed all chunks once, persist to Chroma at `/data/chroma_db`
- **Critical instruction to Antigravity: "Never re-run embedding on unchanged files — check a hash/manifest first."** This single line prevents accidental re-embedding loops that waste time and compute on every rerun.

### Phase 4 — Retrieval + generation pipeline
- Function: `query -> embed -> retrieve top-k -> build grounded prompt -> call Claude API -> return answer + sources`
- Use the Anthropic Python SDK (`anthropic` package) with your own API key, read from an environment variable — never hardcoded
- System prompt must instruct: *answer only from provided context; if not found, say so; always cite which cheat sheet the answer came from*

### Phase 5 — Streamlit UI
- Simple chat interface: input box, answer, expandable "sources" section showing which cheat sheet/section was used
- This is what becomes your demo screenshot/GIF for portfolio

### Phase 6 — Evaluation (do this — it's what separates a toy from a real RAG project)
- Write 15-20 test questions across different cheat sheets (e.g. "How do I merge two dataframes in pandas?", "What's the formula for Bayes' theorem?")
- Manually check: did it retrieve the right chunk? Was the answer grounded and correct?
- Log a simple table of results — this becomes evidence you can show in interviews that you *evaluated* your RAG system, not just built it

### Phase 7 (optional, only after Tier 1 works end-to-end) — Scale to Tier 2
- Rerun Phases 1-3 on the remaining ~225 files
- Same pipeline, no architecture changes needed if Tier 1 was designed cleanly

---

## 4. Token/Cost Discipline — for both Antigravity and any LLM API

**Two separate things use tokens here — don't conflate them:**
1. **Antigravity's own coding engine** (whatever model it runs on under the hood) — this is what writes your code, and its usage cap is a separate pool from your app's actual LLM.
2. **Your app's generation model** — Claude API, called by the code Antigravity writes, using your own key. This is the one your app depends on at runtime, and the one you're deliberately choosing.

**On Antigravity specifically (coding-engine usage):**
- Use **Plan Mode**, not Fast/Autopilot mode, for Phases 1-4. Plan Mode makes Antigravity produce an implementation plan artifact *before* writing code — review it, correct scope before it burns tool-calls on the wrong approach.
- Give each phase as a **separate, scoped prompt** referencing this document rather than one giant instruction. Scoped prompts = smaller context per turn = fewer wasted tokens if something needs correcting.
- After each phase, explicitly tell it to stop and wait for your review rather than cascading into the next phase — this is the main way agent sessions run away with token/credit usage.
- Treat Phase 1-4 build-and-debug cycles as your main consumption; don't waste turns on Tier 2 scale-up until Tier 1 is fully verified.

**On your app's Claude API usage:**
- Local embeddings (sentence-transformers) = zero marginal cost, so iterate freely there — no API calls involved at all.
- During Phase 4 dev/debugging, test retrieval correctness first with the chunks printed to console — you don't need to call Claude just to verify the right chunks are being retrieved.
- Once retrieval looks right, call Claude sparingly while tuning the prompt — a handful of calls, not a loop re-running on every keystroke.
- Cache everything: extraction and embeddings should never re-run on unchanged input.

---

## 5. Ready-to-paste First Prompt for Antigravity

```
I'm building a RAG Q&A system over a curated set of ~20-25 data science
cheat sheet PDFs (pandas, numpy, sklearn, SQL, matplotlib/seaborn,
probability/stats, linear algebra, deep learning, git, python basics).

Work in PLAN MODE. Produce an implementation plan for Phase 1 only:
- Python project scaffold
- PDF text extraction using PyMuPDF, with a flag/log for pages that
  look image-heavy/low-text-density (so I know which ones need OCR
  via pytesseract as a fallback)
- Save extracted text to /data/extracted/<filename>.txt

Generation later in this project will call the Claude API directly
(anthropic Python SDK, my own key) — keep that in mind for architecture,
but don't build it yet.

Do not proceed past producing the plan until I approve it.
Do not touch embeddings, vector stores, or UI yet — that's later phases.
```

Once Phase 1 is approved and built, come back with Phase 2's scoped prompt, and so on.

---

## 6. Portfolio Framing (ties back to your positioning)

When this is done, frame it as: *"Built an end-to-end RAG system over a 250-document technical corpus — designed the ingestion/chunking/retrieval architecture, directed an AI coding agent for implementation, and ran a 20-question grounded-answer evaluation."* This fits your "AI-directed builder" positioning: you're the architect and evaluator, the agent is the execution layer — exactly the story your target roles (Founder's Office, AI Product Associate, APM) want to hear.
