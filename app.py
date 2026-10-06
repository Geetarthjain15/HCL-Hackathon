"""Streamlit chat UI over the rag package.

    streamlit run app.py

The index is loaded from storage/ if it exists, so the app starts instantly
after `python -m rag ingest`. Uploads are indexed in-session on top of that.
"""

from __future__ import annotations

import streamlit as st

from rag import Store, settings, split_documents
from rag.config import EMBED_MODELS
from rag.llm import available_providers
from rag.loaders import find_tesseract, load_bytes
from rag.pipeline import ask_stream

st.set_page_config(page_title="RAG", page_icon="📄", layout="wide")


# --- state -------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading saved index...")
def load_saved_index(model_name: str):
    """Cached on the model name so switching profiles rebuilds correctly."""
    if not Store.exists():
        return None
    try:
        return Store.load()
    except ValueError:
        return None  # index built with a different model


def index_uploads(files, _model_name: str) -> tuple[Store | None, list[str], list[str]]:
    docs, notes, errors = [], [], []
    for f in files:
        try:
            got = load_bytes(f.name, f.getvalue())
        except Exception as exc:
            errors.append(f"{f.name}: {exc}")
            continue
        docs += got
        how = ", ".join(sorted({d.how for d in got}))
        notes.append(f"{f.name} ({how})")
    if not docs:
        return None, notes, errors
    chunks = split_documents(docs)
    return Store.build(chunks), notes, errors


# --- sidebar -----------------------------------------------------------------

with st.sidebar:
    st.subheader("Documents")

    profile = st.selectbox(
        "Embedding model",
        list(EMBED_MODELS),
        index=list(EMBED_MODELS).index(settings.embed_profile)
        if settings.embed_profile in EMBED_MODELS
        else 0,
        help="Switch to multilingual if the documents are not in English.",
    )
    settings.embed_profile = profile

    uploads = st.file_uploader(
        "Add files",
        type=["pdf", "csv", "xlsx", "xlsm", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Scanned PDFs are OCR'd automatically.",
    )

    settings.top_k = st.slider("Chunks retrieved", 1, 12, settings.top_k)
    settings.use_hybrid = st.toggle(
        "Hybrid search", settings.use_hybrid, help="Combine keyword (BM25) with vector search."
    )

    st.divider()
    providers = available_providers()
    st.caption(
        f"**LLM:** {providers[0].name if providers else 'none - check .env'}  \n"
        f"**Fallbacks:** {len(providers) - 1 if providers else 0}  \n"
        f"**OCR:** {'ready' if find_tesseract() else 'not installed'}"
    )

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --- index selection ---------------------------------------------------------

store, notes, errors = None, [], []
if uploads:
    signature = (tuple(sorted(f.name for f in uploads)), settings.embed_model)
    if st.session_state.get("upload_sig") != signature:
        with st.spinner("Indexing uploads..."):
            store, notes, errors = index_uploads(uploads, settings.embed_model)
        st.session_state.upload_sig = signature
        st.session_state.upload_store = store
        st.session_state.upload_notes = notes
        st.session_state.upload_errors = errors
    else:
        store = st.session_state.get("upload_store")
        notes = st.session_state.get("upload_notes", [])
        errors = st.session_state.get("upload_errors", [])
else:
    store = load_saved_index(settings.embed_model)

for err in errors:
    st.sidebar.error(err)

st.title("📄 Ask your documents")

if store is None:
    st.info(
        "No index yet. Either upload files in the sidebar, or build one from "
        "`data/` on the command line:\n\n```\npython -m rag ingest\n```"
    )
    st.stop()

caption = f"{len(store)} chunks from {len(store.sources)} source(s)"
if notes:
    caption += " — " + "; ".join(notes)
st.caption(caption)
with st.expander("Indexed sources"):
    for s in store.sources:
        st.write(f"- {s}")


# --- chat --------------------------------------------------------------------

st.session_state.setdefault("messages", [])

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + ", ".join(msg["sources"]))

question = st.chat_input("Ask something about the indexed documents")
if not question:
    st.stop()

st.session_state.messages.append({"role": "user", "content": question})
with st.chat_message("user"):
    st.markdown(question)

with st.chat_message("assistant"):
    placeholder = st.empty()
    parts: list[str] = []
    answer = None

    try:
        for kind, payload in ask_stream(question, store):
            if kind == "delta":
                parts.append(payload)
                placeholder.markdown("".join(parts) + "▌")
            elif kind == "done":
                answer = payload
    except Exception as exc:
        placeholder.error(f"{type(exc).__name__}: {exc}")
        st.stop()

    placeholder.markdown(answer.text)

    if not answer.grounded:
        st.warning("Not answered from the documents — treat this as a miss, not a fact.")
    for note in answer.notes:
        st.caption(f"⚠ {note}")
    if answer.provider:
        st.caption(f"Answered by {answer.provider}")

    shown = answer.cited or answer.hits
    if shown:
        with st.expander(f"Sources ({len(shown)})"):
            for i, hit in enumerate(answer.hits, 1):
                marker = "**cited**" if hit in answer.cited else "retrieved"
                st.markdown(
                    f"**[{i}] {hit.chunk.label}** — {marker} · "
                    f"similarity {hit.similarity:.2f} · {hit.chunk.how}"
                )
                st.text(hit.chunk.text[:1200])
                st.divider()

st.session_state.messages.append(
    {"role": "assistant", "content": answer.text, "sources": answer.sources}
)
