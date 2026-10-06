"""Measure retrieval and answer quality against a fixed question set.

Retrieval metrics (no API calls, so run these constantly while tuning):
  hit@k  - was a correct source retrieved at all
  MRR    - how high up the first correct source appeared

Answer metrics (one API call per question):
  correct   - every expected string appears in the answer
  cited     - the answer carried at least one [n] citation
  refused   - the pipeline declined to answer

A question with no `expect` is a negative case: the right behaviour is to
refuse, which is how you catch a system that happily invents answers.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import ROOT, settings
from .pipeline import ask
from .retrieval import search
from .store import Store

DEFAULT_SET = ROOT / "eval" / "questions.json"


@dataclass
class Case:
    question: str
    expect: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    note: str = ""
    known_limitation: bool = False  # reported separately, never counted as a regression

    @property
    def is_negative(self) -> bool:
        return not self.expect and not self.sources


@dataclass
class Result:
    case: Case
    retrieved: list[str]
    hit: bool
    rank: int | None
    answer: str = ""
    correct: bool | None = None
    cited: bool = False
    refused: bool = False
    missing: list[str] = field(default_factory=list)
    seconds: float = 0.0


def load_cases(path: str | Path | None = None) -> list[Case]:
    path = Path(path or DEFAULT_SET)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Case(
            question=d["question"],
            expect=d.get("expect", []),
            sources=d.get("sources", []),
            note=d.get("note", ""),
            known_limitation=d.get("known_limitation", False),
        )
        for d in raw
    ]


def _flatten(text: str) -> str:
    """Lowercase and strip the separators that differ between model outputs."""
    for ch in ",   ":
        text = text.replace(ch, " " if ch != "," else "")
    return " ".join(text.lower().split())


def _matches(case: Case, label: str) -> bool:
    return any(s.lower() in label.lower() for s in case.sources)


def evaluate_retrieval(store: Store, cases: list[Case], k: int | None = None) -> list[Result]:
    k = k or settings.top_k
    results = []
    for case in cases:
        hits = search(store, case.question, k)
        labels = [h.chunk.label for h in hits]
        rank = next((i + 1 for i, lab in enumerate(labels) if _matches(case, lab)), None)
        results.append(
            Result(case=case, retrieved=labels, hit=rank is not None, rank=rank)
        )
    return results


def evaluate_answers(store: Store, results: list[Result], k: int | None = None) -> list[Result]:
    for result in results:
        started = time.perf_counter()
        answer = ask(result.case.question, store, k)
        result.seconds = time.perf_counter() - started
        result.answer = answer.text
        result.cited = bool(answer.cited)
        result.refused = not answer.grounded

        if result.case.is_negative:
            result.correct = result.refused  # refusing is the win here
        else:
            flat = _flatten(answer.text)
            # "a|b" means either spelling counts, so an answer saying
            # "optical character recognition" matches an expected "OCR".
            result.missing = [
                e for e in result.case.expect
                if not any(_flatten(alt) in flat for alt in e.split("|"))
            ]
            result.correct = not result.missing and not result.refused
    return results


def summarise(results: list[Result]) -> dict:
    scored = [r for r in results if not r.case.is_negative]
    answered = [
        r for r in results if r.correct is not None and not r.case.known_limitation
    ]
    known = [r for r in results if r.case.known_limitation]
    return {
        "cases": len(results),
        "known_limitations": len(known),
        "hit_rate": sum(r.hit for r in scored) / len(scored) if scored else 0.0,
        "mrr": (
            sum(1 / r.rank for r in scored if r.rank) / len(scored) if scored else 0.0
        ),
        "accuracy": (
            sum(bool(r.correct) for r in answered) / len(answered) if answered else None
        ),
        "cited_rate": (
            sum(r.cited for r in answered if not r.refused)
            / max(sum(not r.refused for r in answered), 1)
            if answered
            else None
        ),
        "median_seconds": (
            sorted(r.seconds for r in answered)[len(answered) // 2] if answered else None
        ),
    }


def format_report(results: list[Result], summary: dict) -> str:
    lines = []
    for r in results:
        if r.correct is None:
            # Retrieval-only run: no answer was generated, so judge nothing else.
            if r.case.is_negative:
                mark, detail = "----", "negative case (needs an answer run)"
            else:
                mark = "HIT " if r.hit else "MISS"
                detail = f"rank {r.rank}" if r.rank else f"got {r.retrieved[:2]}"
        elif r.case.is_negative:
            mark = "PASS" if r.correct else "FAIL"
            detail = "refused correctly" if r.correct else "answered when it should not have"
        elif r.case.known_limitation and not r.correct:
            mark, detail = "KNOWN", "known limitation (see note)"
        else:
            mark = "PASS" if r.correct else "FAIL"
            detail = "ok" if r.correct else (
                "refused" if r.refused else f"missing {r.missing}"
            )
        lines.append(f"  [{mark}] {r.case.question[:62]:62} {detail}")

    lines.append("")
    lines.append(f"  cases      {summary['cases']}"
                 + (f"  ({summary['known_limitations']} known limitation)"
                    if summary.get("known_limitations") else ""))
    lines.append(f"  hit@k      {summary['hit_rate']:.0%}")
    lines.append(f"  MRR        {summary['mrr']:.2f}")
    if summary["accuracy"] is not None:
        lines.append(f"  accuracy   {summary['accuracy']:.0%}")
        lines.append(f"  cited      {summary['cited_rate']:.0%}")
        lines.append(f"  median     {summary['median_seconds']:.1f}s")
    return "\n".join(lines)


def run(
    store: Store | None = None,
    path: str | Path | None = None,
    answers: bool = True,
    k: int | None = None,
) -> tuple[list[Result], dict]:
    store = store or Store.load()
    cases = load_cases(path)
    results = evaluate_retrieval(store, cases, k)
    if answers:
        results = evaluate_answers(store, results, k)
    return results, summarise(results)
