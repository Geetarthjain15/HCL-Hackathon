"""A small, dependency-light RAG pipeline.

    from rag import Store, ask, load_dir, split_documents

    store = Store.build(split_documents(load_dir("data")))
    print(ask("How long do refunds take?", store).text)
"""

from .chunking import Chunk, split_document, split_documents
from .config import EMBED_MODELS, settings
from .loaders import Document, find_tesseract, load_bytes, load_dir, load_path
from .pipeline import Answer, ask, ask_stream
from .retrieval import build_context, search
from .store import Hit, Store

__all__ = [
    "Answer", "Chunk", "Document", "EMBED_MODELS", "Hit", "Store",
    "ask", "ask_stream", "build_context", "find_tesseract", "load_bytes",
    "load_dir", "load_path", "search", "settings", "split_document",
    "split_documents",
]
