# Architecture Walkthrough

A guide to explaining this codebase out loud — in the Sprint 0 design review, or
to a panel at demo time. Each section has **what it is**, **why it is that way**,
and **the one line to say**.

---

## 1. The shape of it

```
hackathon_template/
│
├── rag/                    the pipeline — all logic lives here
│   ├── config.py           every tunable, in one place
│   ├── loaders.py          files  →  Documents
│   ├── chunking.py         Documents  →  Chunks
│   ├── store.py            Chunks  →  searchable index (saved to disk)
│   ├── retrieval.py        question  →  the right Chunks
│   ├── llm.py              provider failover
│   ├── pipeline.py         question  →  grounded, cited Answer
│   ├── evaluate.py         "is it actually any good?"
│   └── cli.py              python -m rag ...
│
├── app.py                  Streamlit chat UI (a thin shell over rag/)
├── eval/questions.json     the scoreboard: 19 graded cases
├── tests/test_rag.py       25 tests, no API key needed
├── data/                   sample documents
├── notebooks/              Colab fallback if the laptop dies
└── storage/                the built index (git-ignored, rebuildable)
```

> **Say this:** "The UI is disposable. All the logic is in `rag/`, which is why we
> can run the exact same pipeline from the command line, from Streamlit, or from
> a Colab notebook."

---

## 2. The flow, end to end

```
   ┌─────────────┐
   │   FILES     │  PDF · scanned PDF · CSV · XLSX · DOCX · TXT
   └──────┬──────┘
          │  loaders.py
          │  • PDF with text  → pypdf, one Document per page
          │  • PDF, no text   → render → Tesseract OCR
          │  • CSV/XLSX/DOCX  → one atomic Document per row
          ▼
   ┌─────────────┐
   │  DOCUMENTS  │  text + source + page/row + how it was read
   └──────┬──────┘
          │  chunking.py
          │  • split on sections → paragraphs → sentences
          │  • table rows are ATOMIC — never split
          ▼
   ┌─────────────┐
   │   CHUNKS    │  ~900 chars, 150 overlap, heading prefixed
   └──────┬──────┘
          │  store.py          ──────────────┐
          │                                  │  saved to storage/
          ├── FAISS   (dense, semantic)      │  survives a restart
          └── BM25    (sparse, literal)   ───┘
          │
          │  retrieval.py
          │  ① both indexes rank independently
          │  ② reciprocal rank fusion merges the two rankings
          │  ③ MMR drops near-duplicates
          │  ④ confidence gate: too weak → refuse
          ▼
   ┌─────────────┐
   │  TOP CHUNKS │  numbered [1] [2] [3] …
   └──────┬──────┘
          │  pipeline.py → llm.py
          │  Groq 120b  →  Groq 20b  →  Gemini      (automatic failover)
          ▼
   ┌─────────────┐
   │   ANSWER    │  grounded · cited · or an honest refusal
   └─────────────┘
```

---

## 3. The four decisions worth defending

### 3.1 Hybrid retrieval, not just vectors

```
Question: "What issue did Daniel Oyelaran report?"

   dense (FAISS)   →  misses it. "Oyelaran" has no semantic neighbours.
   sparse (BM25)   →  finds it at rank 2. It is a literal token match.
   fused (RRF)     →  rank 1. ✅
```

Dense embeddings are good at **paraphrase** ("time off" → "annual leave") and bad
at **rare literals** (`T-1009`, `FR-4`, `0.35`, a person's name). BM25 is exactly
the reverse. Reciprocal Rank Fusion merges the two *rankings*, so the two scoring
scales never have to be made comparable.

> **Say this:** "Vector search alone fails on identifiers and names. Keyword
> search alone fails on paraphrase. We run both and fuse the rankings."

### 3.2 Table rows are atomic

```
  ✗ naive chunking            ✓ atomic rows
  ──────────────────          ──────────────────
  ...Lena Fischer,            [row 9] Lena Fischer … Rejected
  Returns, Rejected,          [row 10] Arjun Bhatia … Resolved
  Arjun Bhatia, Tech...
      ↑ two customers                ↑ one record per chunk
        in one chunk                   cannot be confused
```

> **Say this:** "A chunk boundary through the middle of a table row lets the model
> attribute one customer's status to another. Rows are never split."

### 3.3 The system is allowed to say "I don't know"

Two independent defences, because one is not enough:

| Defence | Catches | Cost |
|---|---|---|
| Cosine similarity gate | Obvious nonsense ("capital of France", 0.097) | Free, pre-LLM |
| `NOT_IN_CONTEXT` instruction | Plausible-but-absent ("Q3 revenue forecast", 0.361) | One LLM call |

The thresholds **overlap** — the weakest valid question scores 0.210, the hardest
absent one scores 0.361. No single number separates them, which is precisely why
there are two layers.

> **Say this:** "A RAG system that always answers is a RAG system that
> hallucinates. We measure refusal as a feature, with two negative cases in the
> eval set."

### 3.4 Everything is measured

```bash
python -m rag eval --retrieval-only    # seconds, free, no API calls
python -m rag eval                     # full scoring
```

```
  hit@k      100%     was a correct source retrieved at all
  MRR        1.00     how high up the first correct source landed
  accuracy    94%     expected facts present, or correctly refused
  cited      100%     answers carrying [n] markers
  median     1.7s     end-to-end latency
```

> **Say this:** "We can change a parameter and know within seconds whether it
> helped. Tuning by vibes is how you lose an afternoon."

---

## 4. Things this deliberately does not do

Being able to name your own limitations is worth more than pretending there are
none.

| Limitation | Why we accepted it |
|---|---|
| Counting across rows ("how many billing tickets?") | Needs all rows in context; `top_k` shows 5. The system refuses rather than guesses — a tracked, known case in the eval set. |
| No cross-encoder reranker | ~100 ms per candidate on CPU. Precision is already sufficient at this corpus size. |
| No multi-hop decomposition | Single-shot retrieval. A comparison question retrieves for one hop only. |
| Scanned **tables** lose columns | OCR flattens alignment to whitespace. Scanned prose is fine. |
| Index is a snapshot | Re-run `ingest` after documents change. No incremental update. |

---

## 5. Three-minute demo script

| # | Do | Shows |
|---|---|---|
| 1 | `python -m rag ingest` | 5 files, 30 chunks, mixed formats, one of them OCR |
| 2 | Ask *"How long do I have to claim a damaged item?"* | Correct figure, cited, page + section named |
| 3 | Ask the **scanned invoice** for its total | OCR path — this PDF has **zero** text layer |
| 4 | Ask *"What is our Q3 revenue forecast?"* | It refuses. Does not invent. |
| 5 | Open the Sources expander | Per-chunk similarity, which chunks were actually cited |
| 6 | `python -m rag eval` | The scoreboard. Numbers, not opinions. |

> **Close with:** "Retrieval is 100% on our eval set, answers 94%, and the 6% is a
> known aggregation limitation we chose not to paper over."

---

## 6. Where to change things

Nothing here needs a code edit — all of it reads from `.env`:

| Symptom | Knob |
|---|---|
| Answers miss relevant context | `TOP_K=8` |
| Model quotes the wrong nearby fact | `CHUNK_CHARS=600` |
| Rare IDs/names not found | check `USE_HYBRID=1` |
| Too many near-duplicate chunks | `MMR_LAMBDA=0.3` |
| Refuses valid questions | lower `MIN_SIMILARITY` |
| Invents answers | raise `MIN_SIMILARITY` |
| Documents not in English | `EMBED_PROFILE=multilingual` |
| Hindi/other scans | `OCR_LANG=eng+hin` |

After any change:

```bash
python -m rag eval --retrieval-only
```

Keep it only if the numbers improved.
