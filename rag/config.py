"""Central configuration. Every value can be overridden by an env var."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

EMBED_MODELS = {
    "english": "all-MiniLM-L6-v2",
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    # --- paths ---
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    index_dir: Path = field(default_factory=lambda: ROOT / "storage")

    # --- embeddings ---
    embed_profile: str = os.getenv("EMBED_PROFILE", "english")

    # --- chunking ---
    chunk_chars: int = _int("CHUNK_CHARS", 900)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 150)
    min_chunk_chars: int = _int("MIN_CHUNK_CHARS", 80)

    # --- retrieval ---
    top_k: int = _int("TOP_K", 5)
    candidate_k: int = _int("CANDIDATE_K", 25)
    # Below this raw cosine similarity we treat retrieval as a miss and refuse.
    # RRF scores depend only on rank, so they cannot measure relevance - this
    # gate has to use the actual embedding similarity.
    min_similarity: float = _float("MIN_SIMILARITY", 0.18)
    mmr_lambda: float = _float("MMR_LAMBDA", 0.5)  # 1.0 = pure relevance, 0 = pure diversity
    use_hybrid: bool = os.getenv("USE_HYBRID", "1") != "0"

    # --- ocr ---
    ocr_dpi: int = _int("OCR_DPI", 200)
    ocr_trigger_chars: int = _int("OCR_TRIGGER_CHARS", 40)
    ocr_lang: str = os.getenv("OCR_LANG", "eng")

    # --- generation ---
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    groq_fallback_model: str = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    temperature: float = _float("TEMPERATURE", 0.1)
    max_context_chars: int = _int("MAX_CONTEXT_CHARS", 12000)
    request_timeout: int = _int("REQUEST_TIMEOUT", 90)

    @property
    def embed_model(self) -> str:
        return EMBED_MODELS.get(self.embed_profile, self.embed_profile)

    @property
    def groq_key(self) -> str | None:
        return os.getenv("GROQ_API_KEY") or None

    @property
    def gemini_key(self) -> str | None:
        return os.getenv("GEMINI_API_KEY") or None


settings = Settings()
