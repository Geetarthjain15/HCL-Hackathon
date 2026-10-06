# HCL Hackathon — RAG starter

A retrieval-augmented generation pipeline you can point at a pile of documents
and demo in minutes. Built to survive the things that actually go wrong at a
hackathon: scanned PDFs, rate limits, a dead laptop, and questions whose answers
are not in the corpus.

## Problem

<!-- Who hurts, how often, and what it costs them today. Name the user. -->

## Approach

<!-- The idea in 3-4 sentences. What you chose not to build, and why. -->

## Architecture

```
files (PDF / CSV / XLSX / DOCX / TXT / MD)
   |
   |  loaders.py
   +-- PDF with text --> pypdf, one Document per page (keeps page numbers)
   +-- PDF, no text ---> pypdfium2 render -> Tesseract OCR
   +-- CSV / XLSX -----> pandas, one atomic Document per row
   +-- DOCX -----------> paragraphs + one atomic Document per table row
   |
   |  chunking.py   structure-aware: sections -> paragraphs -> sentences
   v                 atomic rows are never split
 chunks
   |
   |  store.py      persisted to storage/
   +-- FAISS IndexFlatIP over normalised MiniLM vectors   (semantic)
   +-- BM25Okapi over the same chunks                     (literal)
   |
   |  retrieval.py
   +-- reciprocal rank fusion of both rankings
   +-- MMR to drop near-duplicate chunks
   +-- confidence gate on raw cosine similarity
   |
   |  pipeline.py   numbered context -> cited answer, or refusal
   v
 llm.py:  Groq gpt-oss-120b  ->  Groq gpt-oss-20b  ->  Gemini 3.8 flash
```

| Layer | Choice | Why |
| --- | --- | --- |
| Embeddings | `all-MiniLM-L6-v2` | CPU-only, cached locally, 384 dims |
| Multilingual | `paraphrase-multilingual-MiniLM-L12-v2` | Same dims, matches across languages |
| Retrieval | Hybrid BM25 + dense, fused by RRF | Dense handles paraphrase, BM25 handles rare literals like `T-1009` or `FR-4`; RRF needs no score calibration |
| Diversity | MMR (`MMR_LAMBDA`, default 0.5) | Stops four chunks of one paragraph crowding out the one that completes the answer |
| Vector store | FAISS flat (exact) | Under ~100k chunks, ANN costs recall and buys nothing |
| Persistence | `storage/` on disk | The index survives a restart; no re-embedding mid-demo |
| Refusal | Cosine gate + `NOT_IN_CONTEXT` instruction | Two independent defences against confident nonsense |
| LLM | Groq, with 2 fallbacks | A 429 on the free tier is the likeliest demo-killer |

**Why the confidence gate uses cosine, not the fused score.** RRF scores depend
only on *rank*, so the top result scores the same whether it is a perfect match
or unrelated. Only the raw similarity can tell those apart. (This was a real bug
here, caught by the negative cases in the eval set.)

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
cp .env.example .env           # then fill in the keys

python -m rag ingest           # build the index from data/
python -m rag info             # check config, keys, index
streamlit run app.py
```

| Variable | Where to get it |
| --- | --- |
| `GROQ_API_KEY` | console.groq.com/keys |
| `GEMINI_API_KEY` | aistudio.google.com/apikey |

Use a **personal** Google account for the Gemini key. Workspace and education
accounts are blocked at the project level and return `403 PERMISSION_DENIED` no
matter how many keys you generate.

Groq's free tier serves `openai/gpt-oss-120b`, `openai/gpt-oss-20b` and
`qwen/qwen3.8-27b`. The Llama models most tutorials use are **not** available —
a copied `llama-3.3-70b-versatile` will 404.

For scanned PDFs:

```bash
winget install UB-Mannheim.TesseractOCR    # Windows
sudo apt install tesseract-ocr             # Debian / Ubuntu
```

### Command line

```bash
python -m rag ingest [path]        # index a file or directory
python -m rag ask "your question"  # one-shot answer with sources
python -m rag eval                 # score the pipeline
python -m rag eval --retrieval-only   # same, without spending API tokens
python -m rag info                 # config, providers, index contents
```

### Tuning without touching code

Every setting reads from `.env`: `TOP_K`, `CHUNK_CHARS`, `CHUNK_OVERLAP`,
`MIN_SIMILARITY`, `MMR_LAMBDA`, `USE_HYBRID=0`, `EMBED_PROFILE=multilingual`,
`GROQ_MODEL`, `TEMPERATURE`. Change one, re-run `python -m rag eval
--retrieval-only`, and keep it only if the numbers improve.

## Evaluation

`eval/questions.json` holds 19 cases: paraphrases, exact figures with
distractors nearby, OCR-only questions, row-level table lookups, and two
**negative cases** whose correct answer is a refusal.

```bash
python -m rag eval --retrieval-only          # seconds, free
python -m rag eval                           # adds answer scoring
python -m rag eval --min-hit-rate 0.9        # exits 1 below the floor, for CI
```

Metrics: `hit@k` (was a correct source retrieved), `MRR` (how high), `accuracy`
(expected strings present, or correctly refused), `cited` (share of answers
carrying `[n]` markers), and median latency.

Add your own cases the moment you see the real hackathon data — a case with no
`expect` and no `sources` is a negative case, and those are the ones that catch
a system which cheerfully invents answers.

## Demo

<!-- Script the 3-minute run-through. -->

1. `python -m rag ingest` then `streamlit run app.py`
2. Upload `data/refund_policy.pdf`, ask *"How long do I have to claim a damaged item?"* → 10 days, cited
3. Ask the same of `data/scanned_invoice.pdf` — **no text layer at all**, answered via OCR
4. Ask *"What is our Q3 revenue forecast?"* → it refuses instead of inventing
5. Open the Sources expander to show per-chunk similarity and which chunks were actually cited

| Sample file | Shows |
| --- | --- |
| `data/refund_policy.pdf` | Normal PDF. "claim a damaged item" → 10 days |
| `data/employee_handbook.pdf` | Distractor numbers in one document (18 / 12 / 26 / 10) |
| `data/project_orion_spec.pdf` | Literal lookups — "0.35", "FR-4" — where BM25 earns its place |
| `data/scanned_invoice.pdf` | Image-only PDF, OCR path |
| `data/support_tickets.csv` | Row-level retrieval; rows never merge |

## Limitations

- **Aggregation is weak.** "How many tickets are about billing?" needs counting
  across rows; retrieval returns `top_k` chunks, so any count beyond that is a
  guess. Known-shaky case, kept in the eval set deliberately.
- **OCR loses table structure.** Columns become whitespace, so a scanned table
  answers worse than a scanned paragraph. Check this early if the real data is
  scanned tables.
- **The index is a snapshot.** Re-run `ingest` after documents change; there is
  no incremental update or deletion.
- **No reranker.** A cross-encoder would improve precision, at roughly 100ms per
  candidate on CPU — deliberately skipped for latency.
- **Single-shot retrieval.** No query decomposition, so multi-hop questions
  ("compare the leave policy with the notice period") retrieve for only one hop.
- **The refusal gate is a threshold, not a judgement.** `MIN_SIMILARITY` is
  tuned on this sample corpus; re-check it against real data.

## Layout

```
rag/
  config.py      settings, all env-overridable
  loaders.py     files -> Documents (incl. OCR)
  chunking.py    Documents -> Chunks
  store.py       FAISS + BM25, save/load
  retrieval.py   RRF fusion, MMR, confidence gate
  llm.py         provider chain with failover
  pipeline.py    ask() / ask_stream()
  evaluate.py    metrics
  cli.py         python -m rag ...
app.py           Streamlit chat UI
eval/            question set
tests/           24 tests, no API key needed
notebooks/       Colab fallback
```

### If the laptop fails

`notebooks/colab_rag.ipynb` runs the same pipeline on Colab. Open
colab.research.google.com → GitHub tab → paste this repo URL. Keys go in via
`getpass`, so nothing sensitive is saved into the notebook.
