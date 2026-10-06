"""Split Documents into retrievable Chunks.

Splitting happens on the largest natural boundary that fits: blank lines first,
then sentence ends, then whitespace. Atomic documents (table rows) are never
split, which is what keeps one record's fields from leaking into another's.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .config import settings
from .loaders import Document

# A heading is a short line that is numbered, all-caps, or markdown-style.
HEADING = re.compile(
    r"^\s*(?:#{1,6}\s+\S.*|\d+(?:\.\d+)*[.)]?\s+\S.{0,70}|[A-Z][A-Z0-9 ,&/'-]{3,70})\s*$"
)
SENTENCE_END = re.compile(r"(?<=[.!?])\s")


@dataclass
class Chunk:
    text: str
    source: str
    how: str
    meta: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return hashlib.sha1(f"{self.source}|{self.text}".encode()).hexdigest()[:16]

    @property
    def label(self) -> str:
        """Human-readable citation target, e.g. 'policy.pdf p.2'."""
        bits = [self.source]
        if page := self.meta.get("page"):
            bits.append(f"p.{page}")
        if row := self.meta.get("row"):
            bits.append(f"row {row}")
        if section := self.meta.get("section"):
            bits.append(f"- {section}")
        return " ".join(bits)


def _normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _sections(text: str) -> list[tuple[str | None, str]]:
    """Group lines under the most recent heading."""
    out: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in text.split("\n"):
        if HEADING.match(line) and len(line.strip()) > 3:
            out.append((line.strip().lstrip("# ").strip(), []))
        else:
            out[-1][1].append(line)
    return [(h, "\n".join(body).strip()) for h, body in out if "\n".join(body).strip()]


def _pack(pieces: list[str], limit: int) -> list[str]:
    """Greedily combine pieces without crossing the size limit."""
    packed, current = [], ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + len(piece) + 2 <= limit:
            current = f"{current}\n\n{piece}" if "\n" in piece or len(piece) > 60 else f"{current} {piece}"
        else:
            packed.append(current)
            current = piece
    if current:
        packed.append(current)
    return packed


def _hard_split(text: str, limit: int, overlap: int) -> list[str]:
    """Last resort for a single piece larger than the limit."""
    out, start = [], 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("\n"), window.rfind(" "))
            if cut > limit // 2:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return out


def split_document(doc: Document) -> list[Chunk]:
    if doc.atomic:
        text = _normalise(doc.text)
        return [Chunk(text, doc.source, doc.how, dict(doc.meta))] if text else []

    limit, overlap = settings.chunk_chars, settings.chunk_overlap
    chunks: list[Chunk] = []

    for heading, body in _sections(_normalise(doc.text)):
        paragraphs: list[str] = []
        for para in re.split(r"\n\s*\n", body):
            para = para.strip()
            if not para:
                continue
            if len(para) <= limit:
                paragraphs.append(para)
            else:
                sentences = [s.strip() for s in SENTENCE_END.split(para) if s.strip()]
                for piece in _pack(sentences, limit):
                    paragraphs.extend(
                        [piece] if len(piece) <= limit else _hard_split(piece, limit, overlap)
                    )

        meta = dict(doc.meta)
        if heading:
            meta["section"] = heading

        for i, text in enumerate(_pack(paragraphs, limit)):
            # Too small to retrieve alone - attach to the previous chunk, but
            # only within the same section, or the heading label is lost and two
            # topics end up in one chunk.
            if (
                i > 0
                and len(text) < settings.min_chunk_chars
                and chunks
                and chunks[-1].meta.get("section") == meta.get("section")
                and len(chunks[-1].text) + len(text) <= limit * 1.3
            ):
                prev = chunks[-1]
                chunks[-1] = Chunk(f"{prev.text}\n{text}", prev.source, prev.how, prev.meta)
                continue
            # Prefixing the heading gives the embedding its topic back.
            chunks.append(
                Chunk(f"{heading}\n{text}" if heading else text, doc.source, doc.how, meta)
            )

    return chunks


def split_documents(docs: list[Document]) -> list[Chunk]:
    seen: set[str] = set()
    out: list[Chunk] = []
    for doc in docs:
        for chunk in split_document(doc):
            if chunk.id not in seen:  # drop exact duplicates across files
                seen.add(chunk.id)
                out.append(chunk)
    return out
