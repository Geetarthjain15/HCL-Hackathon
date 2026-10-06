# HCL Hackathon

One-line pitch goes here.

## Problem

<!-- Who hurts, how often, and what it costs them today. Name the user. -->

## Approach

<!-- The idea in 3-4 sentences. Why retrieval over fine-tuning, what you chose not to build. -->

## Architecture

<!-- Replace with your real flow. -->

```
PDF / CSV / DOCX / XLSX / TXT upload
      |
      +-- PDF with text layer --> pypdf
      +-- PDF with no text ------> pypdfium2 render -> Tesseract OCR
      +-- CSV / XLSX ------------> pandas, one chunk per row
      +-- DOCX ------------------> python-docx (paragraphs + tables)
      |
      v
  chunking (1000 chars, 150 overlap)
      |
      v
  all-MiniLM-L6-v2  ->  384-dim vectors
  (or paraphrase-multilingual-MiniLM-L12-v2 for non-English)
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
| Multilingual | `paraphrase-multilingual-MiniLM-L12-v2` | Same 384 dims, matches across languages; selectable in the sidebar |
| Scanned PDFs | pypdfium2 + Tesseract | Triggered automatically when the text layer is near-empty |
| Vector store | FAISS flat index | Exact search; dataset is small enough that ANN adds nothing |
| LLM | Groq `openai/gpt-oss-120b` | Fast inference on the free tier |
| Fallback | Gemini `gemini-3.8-flash` | Separate quota, so a Groq rate limit does not end the demo |
| UI | Streamlit | Fastest path to a demo-able interface |

Note: Groq's free tier serves `openai/gpt-oss-120b`, `openai/gpt-oss-20b` and
`qwen/qwen3.8-27b`. The Llama models most tutorials use are not available, so a
copied `llama-3.3-70b-versatile` will 404.

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

Embedding models download once on first run and are cached in
`~/.cache/huggingface`, so the app works offline after that.

For scanned PDFs, install Tesseract as well — `app.py` finds it on PATH or at
the default install location:

```bash
winget install UB-Mannheim.TesseractOCR    # Windows
sudo apt install tesseract-ocr             # Debian / Ubuntu
```

### If the laptop fails

[`notebooks/colab_rag.ipynb`](notebooks/colab_rag.ipynb) runs the same pipeline
on Google Colab. Open colab.research.google.com, use the GitHub tab, paste this
repo URL, and run the cells top to bottom. Keys are entered via `getpass`, so
nothing sensitive is saved into the notebook.

## Sample data

| File | Use it to show |
| --- | --- |
| `data/refund_policy.pdf` | Normal text-layer PDF. "How long do I have to claim a damaged item?" -> 10 days |
| `data/employee_handbook.pdf` | Multi-topic retrieval. "How much annual leave do I get?" -> 18 days |
| `data/project_orion_spec.pdf` | Requirement lookup. "What is the latency target?" -> 5s at p95 |
| `data/scanned_invoice.pdf` | Image-only PDF, no text layer. Proves the OCR path |
| `data/support_tickets.csv` | Row-level retrieval over tabular data |

## Demo

<!-- Script the 3-minute run-through: which sample file, which question, what to point at. -->

1. Upload `data/refund_policy.pdf`
2. Ask: "How long do I have to return a damaged item?"
3. Open the "Retrieved context" expander to show grounding

## Limitations

- OCR accuracy degrades on low-resolution or skewed scans, and OCR'd tables lose
  their column structure
- Index is rebuilt in memory on every upload; nothing persists across restarts
- Fixed-size chunking splits tables and lists mid-structure
- Retrieval is single-shot; no query rewriting or reranking
- No evaluation set, so answer quality is unmeasured
