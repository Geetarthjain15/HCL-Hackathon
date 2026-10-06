"""Question in, grounded answer out."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import llm
from .config import settings
from .retrieval import build_context, is_confident, search
from .store import Hit, Store

SYSTEM_PROMPT = """You answer questions strictly from the numbered context given to you.

Rules:
- Use only the context. Do not add outside knowledge.
- Cite the bracketed number after each claim, like [1] or [2][3].
- Use plain ASCII square brackets for citations. Never use other bracket styles.
- If the context does not contain the answer, reply exactly:
  NOT_IN_CONTEXT
  followed by one sentence saying what is missing.
- Quote figures, dates and names exactly as written.
- Answer the field that was asked for. If the question names a labelled field
  (for example "payment terms"), quote that label's value, not a nearby one.
- Be concise. No preamble.
"""

REFUSAL = (
    "I could not find anything relevant to that in the indexed documents. "
    "Try rephrasing, or check that the right file is indexed."
)

# Models reach for fullwidth/CJK brackets surprisingly often - gpt-oss emits
# U+3010 and U+3011 rather than ASCII. Accept them all, then normalise.
CITATION = re.compile(r"[\[【［]\s*(\d+)\s*[\]】］]")
NARROW_SPACE = dict.fromkeys(map(ord, "   "), " ")


def normalise_citations(text: str) -> str:
    """Rewrite any bracket style to [n] and tidy exotic spaces."""
    return CITATION.sub(lambda m: f"[{m.group(1)}]", text).translate(NARROW_SPACE)


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    cited: list[Hit] = field(default_factory=list)
    provider: str = ""
    grounded: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def sources(self) -> list[str]:
        seen = {}
        for hit in self.cited or self.hits:
            seen.setdefault(hit.chunk.label, None)
        return list(seen)


def build_prompt(question: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def _cited_hits(text: str, used: list[Hit]) -> list[Hit]:
    """Map [n] markers in the answer back to the hits they refer to."""
    out = []
    for n in dict.fromkeys(CITATION.findall(text)):
        i = int(n) - 1
        if 0 <= i < len(used):
            out.append(used[i])
    return out


def ask(question: str, store: Store, k: int | None = None) -> Answer:
    question = question.strip()
    if not question:
        raise ValueError("Empty question")

    hits = search(store, question, k or settings.top_k)

    if not is_confident(hits):
        return Answer(
            text=REFUSAL,
            hits=hits,
            grounded=False,
            notes=["retrieval below confidence threshold"],
        )

    context, used = build_context(hits)
    notes: list[str] = []
    text, provider = llm.complete(
        SYSTEM_PROMPT,
        build_prompt(question, context),
        on_fallback=lambda name, exc: notes.append(f"{name} failed ({type(exc).__name__})"),
    )
    text = normalise_citations((text or "").strip())

    if text.startswith("NOT_IN_CONTEXT"):
        return Answer(
            text=text.replace("NOT_IN_CONTEXT", "").strip() or REFUSAL,
            hits=used,
            provider=provider,
            grounded=False,
            notes=notes + ["model reported the context did not cover it"],
        )

    cited = _cited_hits(text, used)
    if not cited:
        notes.append("answer carried no citations - verify it against the sources")

    return Answer(
        text=text, hits=used, cited=cited, provider=provider, grounded=True, notes=notes
    )


def ask_stream(question: str, store: Store, k: int | None = None):
    """Yield (kind, payload): ('hits', list), ('delta', str), ('done', Answer)."""
    question = question.strip()
    hits = search(store, question, k or settings.top_k)
    yield "hits", hits

    if not is_confident(hits):
        yield "done", Answer(text=REFUSAL, hits=hits, grounded=False)
        return

    context, used = build_context(hits)
    notes: list[str] = []
    parts, provider = [], ""

    for piece in llm.stream(
        SYSTEM_PROMPT,
        build_prompt(question, context),
        on_fallback=lambda name, exc: notes.append(f"{name} failed ({type(exc).__name__})"),
    ):
        if isinstance(piece, dict):
            provider = piece["provider"]
            break
        parts.append(piece)
        yield "delta", piece

    text = normalise_citations("".join(parts).strip())
    grounded = not text.startswith("NOT_IN_CONTEXT")
    if not grounded:
        text = text.replace("NOT_IN_CONTEXT", "").strip() or REFUSAL

    yield "done", Answer(
        text=text,
        hits=used,
        cited=_cited_hits(text, used) if grounded else [],
        provider=provider,
        grounded=grounded,
        notes=notes,
    )
