"""Minimal RAG template: PDFs/CSV -> MiniLM embeddings -> FAISS -> LLM answer.

Primary LLM is Groq; if Groq errors or rate-limits, it falls back to Gemini.
Run with:  streamlit run app.py
"""

import os
from dataclasses import dataclass

import faiss
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-3.8-flash"
CHUNK_CHARS = 1000
CHUNK_OVERLAP = 150
TOP_K = 4

SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "If the context does not contain the answer, say so plainly. "
    "Cite the source filename for each claim."
)


@dataclass
class Chunk:
    text: str
    source: str


# --- loading -----------------------------------------------------------------


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


def split(text: str, source: str) -> list[Chunk]:
    """Fixed-size character chunks with overlap, snapped to whitespace."""
    chunks, start = [], 0
    text = " ".join(text.split())
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(piece, source))
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)  # +1 guards against no forward progress
    return chunks


def read_upload(file) -> list[Chunk]:
    name = file.name
    if name.lower().endswith(".pdf"):
        pages = [p.extract_text() or "" for p in PdfReader(file).pages]
        return split("\n".join(pages), name)
    if name.lower().endswith(".csv"):
        df = pd.read_csv(file)
        # One chunk per row keeps tabular facts from bleeding into each other.
        return [
            Chunk("; ".join(f"{c}: {r[c]}" for c in df.columns), f"{name} row {i + 1}")
            for i, r in df.iterrows()
        ]
    return split(file.read().decode("utf-8", errors="ignore"), name)


# --- index -------------------------------------------------------------------


def build_index(chunks: list[Chunk]):
    vecs = get_embedder().encode(
        [c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False
    )
    vecs = np.asarray(vecs, dtype="float32")
    index = faiss.IndexFlatIP(vecs.shape[1])  # inner product == cosine, vectors normalized
    index.add(vecs)
    return index


def retrieve(index, chunks: list[Chunk], question: str, k: int = TOP_K) -> list[Chunk]:
    q = np.asarray(
        get_embedder().encode([question], normalize_embeddings=True), dtype="float32"
    )
    _, ids = index.search(q, min(k, len(chunks)))
    return [chunks[i] for i in ids[0] if i >= 0]


# --- generation --------------------------------------------------------------


def ask_groq(question: str, context: str) -> str:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def ask_gemini(question: str, context: str) -> str:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    r = requests.post(
        url,
        headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
        json={
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {"parts": [{"text": f"Context:\n{context}\n\nQuestion: {question}"}]}
            ],
            # This model spends tokens on hidden reasoning before it emits any
            # text, so a small cap returns an empty response, not a short one.
            "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.2},
        },
        timeout=90,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def answer(question: str, context: str) -> tuple[str, str]:
    """Return (answer, which_model). Falls back to Gemini if Groq is unavailable."""
    if os.getenv("GROQ_API_KEY"):
        try:
            return ask_groq(question, context), f"Groq / {GROQ_MODEL}"
        except Exception as exc:  # rate limit, quota, outage
            st.warning(f"Groq unavailable ({type(exc).__name__}), falling back to Gemini.")
    if os.getenv("GEMINI_API_KEY"):
        return ask_gemini(question, context), f"Gemini / {GEMINI_MODEL}"
    raise RuntimeError("No usable API key. Set GROQ_API_KEY or GEMINI_API_KEY in .env")


# --- ui ----------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="RAG Template", page_icon="📄")
    st.title("📄 Ask your documents")

    with st.sidebar:
        st.subheader("Documents")
        files = st.file_uploader(
            "PDF, CSV or TXT", type=["pdf", "csv", "txt"], accept_multiple_files=True
        )
        st.caption(
            f"Groq key: {'✅' if os.getenv('GROQ_API_KEY') else '❌'} · "
            f"Gemini key: {'✅' if os.getenv('GEMINI_API_KEY') else '❌'}"
        )

    if not files:
        st.info("Upload a document in the sidebar to begin. Samples are in `data/`.")
        return

    signature = tuple(sorted(f.name for f in files))
    if st.session_state.get("signature") != signature:
        with st.spinner("Indexing..."):
            chunks: list[Chunk] = []
            for f in files:
                chunks.extend(read_upload(f))
            if not chunks:
                st.error("No text could be extracted. Is the PDF a scan?")
                return
            st.session_state.chunks = chunks
            st.session_state.index = build_index(chunks)
            st.session_state.signature = signature
        st.success(f"Indexed {len(chunks)} chunks from {len(files)} file(s).")

    question = st.text_input("Question", placeholder="What does the policy say about refunds?")
    if not question:
        return

    hits = retrieve(st.session_state.index, st.session_state.chunks, question)
    context = "\n\n".join(f"[{c.source}] {c.text}" for c in hits)

    with st.spinner("Thinking..."):
        text, model = answer(question, context)

    st.markdown(text)
    st.caption(f"Answered by {model}")
    with st.expander(f"Retrieved context ({len(hits)} chunks)"):
        for c in hits:
            st.markdown(f"**{c.source}**")
            st.text(c.text)


if __name__ == "__main__":
    main()
