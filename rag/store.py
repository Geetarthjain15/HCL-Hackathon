"""Persistent hybrid index: FAISS for dense vectors, BM25 for keywords.

Saving the index to disk means a demo survives a restart - the laptop can die
between the dry run and the judging and the corpus is still there.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np

from .chunking import Chunk
from .config import settings

FORMAT_VERSION = 2
TOKEN = re.compile(r"[a-z0-9]+")


@lru_cache(maxsize=4)
def get_embedder(model_name: str):
    """Cached across calls; loading the model is the slow part."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed(texts: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    vecs = get_embedder(model_name).encode(
        texts,
        normalize_embeddings=True,  # so inner product == cosine similarity
        batch_size=batch_size,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype="float32")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@dataclass
class Hit:
    chunk: Chunk
    score: float            # fused RRF score, for ordering only
    similarity: float = 0.0  # raw cosine to the query, for the confidence gate
    dense_rank: int | None = None
    sparse_rank: int | None = None


class Store:
    """Chunks + a FAISS index + a BM25 index, saved and loaded together."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, model_name: str):
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self.chunks = chunks
        self.vectors = vectors
        self.model_name = model_name
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self._bm25 = None

    # --- construction ---

    @classmethod
    def build(cls, chunks: list[Chunk], model_name: str | None = None) -> "Store":
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks")
        model_name = model_name or settings.embed_model
        return cls(chunks, embed([c.text for c in chunks], model_name), model_name)

    @property
    def bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])
        return self._bm25

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks})

    # --- search ---

    def dense(self, query: str, k: int) -> list[tuple[int, float]]:
        q = embed([query], self.model_name)
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def sparse(self, query: str, k: int) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    # --- persistence ---

    def save(self, directory: str | Path | None = None) -> Path:
        directory = Path(directory or settings.index_dir)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / "index.faiss"))
        np.save(directory / "vectors.npy", self.vectors)
        with open(directory / "chunks.jsonl", "w", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(
                    json.dumps(
                        {"text": c.text, "source": c.source, "how": c.how, "meta": c.meta},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "format": FORMAT_VERSION,
                    "model": self.model_name,
                    "dim": int(self.vectors.shape[1]),
                    "chunks": len(self.chunks),
                    "sources": self.sources,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path | None = None) -> "Store":
        directory = Path(directory or settings.index_dir)
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No index at {directory}. Build one with: python -m rag ingest"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT_VERSION:
            raise ValueError(
                f"Index at {directory} is format {manifest.get('format')}, "
                f"this code expects {FORMAT_VERSION}. Re-run: python -m rag ingest"
            )

        chunks = []
        with open(directory / "chunks.jsonl", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                chunks.append(Chunk(d["text"], d["source"], d["how"], d.get("meta", {})))

        store = cls.__new__(cls)
        store.chunks = chunks
        store.vectors = np.load(directory / "vectors.npy")
        store.model_name = manifest["model"]
        store.index = faiss.read_index(str(directory / "index.faiss"))
        store._bm25 = None

        if store.model_name != settings.embed_model:
            # Mixing models silently would produce meaningless scores.
            raise ValueError(
                f"Index was built with '{store.model_name}' but settings say "
                f"'{settings.embed_model}'. Re-ingest or set EMBED_PROFILE to match."
            )
        return store

    @classmethod
    def exists(cls, directory: str | Path | None = None) -> bool:
        return (Path(directory or settings.index_dir) / "manifest.json").exists()
