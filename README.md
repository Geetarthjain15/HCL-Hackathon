# HCL Hackathon

One-line pitch goes here.

## Problem

<!-- Who hurts, how often, and what it costs them today. Name the user. -->

## Approach

<!-- The idea in 3-4 sentences. Why retrieval over fine-tuning, what you chose not to build. -->

## Architecture

<!-- Replace with your real flow. -->

```
PDF / CSV upload
      |
      v
  chunking (1000 chars, 150 overlap)
      |
      v
  all-MiniLM-L6-v2  ->  384-dim vectors
      |
      v
  FAISS IndexFlatIP (cosine)
      |
      v
  top-4 chunks -> prompt -> Groq (gpt-oss-120b)
                              |  on rate limit
                              v
                            Gemini (gemini-3.8-flash)
```

| Layer | Choice | Why |
| --- | --- | --- |
| Embeddings | `all-MiniLM-L6-v2` | Runs on CPU, cached locally, 384-dim keeps the index small |
| Vector store | FAISS flat index | Exact search; dataset is small enough that ANN adds nothing |
| LLM | Groq `openai/gpt-oss-120b` | Fast inference on the free tier |
| Fallback | Gemini `gemini-3.8-flash` | Separate quota, so a Groq rate limit does not end the demo |
| UI | Streamlit | Fastest path to a demo-able interface |

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt

cp .env.example .env           # then fill in both keys
streamlit run app.py
```

Keys needed in `.env`:

| Variable | Where to get it |
| --- | --- |
| `GROQ_API_KEY` | console.groq.com/keys |
| `GEMINI_API_KEY` | aistudio.google.com/apikey |

Use a personal Google account for the Gemini key — Workspace/education accounts are
blocked at the project level and return `403 PERMISSION_DENIED`.

The embedding model downloads once (~90 MB) on first run and is cached in
`~/.cache/huggingface` thereafter, so the app works offline after that.

## Demo

<!-- Script the 3-minute run-through: which sample file, which question, what to point at. -->

1. Upload `data/refund_policy.pdf`
2. Ask: "How long do I have to return a damaged item?"
3. Open the "Retrieved context" expander to show grounding

## Limitations

- No OCR — scanned/image-only PDFs extract no text
- Index is rebuilt in memory on every upload; nothing persists across restarts
- Fixed-size chunking splits tables and lists mid-structure
- Retrieval is single-shot; no query rewriting or reranking
- No evaluation set, so answer quality is unmeasured
