"""Tests that need no API key and no network (models load from the local cache)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rag.chunking import Chunk, split_document, split_documents
from rag.config import settings
from rag.loaders import Document, load_bytes, load_dir
from rag.retrieval import mmr, reciprocal_rank_fusion, search, is_confident
from rag.store import Store, tokenize

DATA = settings.data_dir


# --- chunking ----------------------------------------------------------------


def test_atomic_documents_are_never_split():
    long_row = Document("; ".join(f"col{i}: value{i}" for i in range(400)), "t.csv", "table row", atomic=True)
    assert len(long_row.text) > settings.chunk_chars
    assert len(split_document(long_row)) == 1


def test_chunks_respect_the_size_limit():
    text = " ".join(f"Sentence number {i} with filler words." for i in range(600))
    chunks = split_document(Document(text, "big.txt", "plain text"))
    assert len(chunks) > 1
    assert all(len(c.text) <= settings.chunk_chars * 1.35 for c in chunks)


def test_headings_are_attached_to_their_section():
    doc = Document(
        "# Refund Policy\nRefunds take seven days.\n\n# Shipping\nShipping takes two days.",
        "p.md",
        "plain text",
    )
    chunks = split_document(doc)
    sections = {c.meta.get("section") for c in chunks}
    assert "Refund Policy" in sections and "Shipping" in sections
    refund = next(c for c in chunks if c.meta.get("section") == "Refund Policy")
    assert "seven days" in refund.text
    assert "two days" not in refund.text  # sections must not bleed together


def test_duplicate_chunks_are_dropped():
    docs = [Document("Identical text here.", "a.txt", "plain text")] * 3
    assert len(split_documents(docs)) == 1


def test_chunk_label_includes_page_and_row():
    assert Chunk("x", "f.pdf", "text layer", {"page": 4}).label == "f.pdf p.4"
    assert Chunk("x", "f.csv", "table row", {"row": 9}).label == "f.csv row 9"


def test_empty_document_yields_nothing():
    assert split_document(Document("   \n  ", "empty.txt", "plain text")) == []


# --- loaders -----------------------------------------------------------------


def test_csv_rows_stay_separate():
    csv = b"name,city\nAda,London\nGrace,Boston\n"
    docs = load_bytes("people.csv", csv)
    rows = [d for d in docs if d.meta.get("kind") != "schema"]
    assert len(rows) == 2
    assert all(d.atomic for d in rows)
    assert not any("Ada" in d.text and "Grace" in d.text for d in rows)


def test_unsupported_type_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        load_bytes("thing.xyz", b"data")


@pytest.mark.skipif(not (DATA / "refund_policy.pdf").exists(), reason="sample missing")
def test_pdf_keeps_page_numbers():
    docs = load_bytes("refund_policy.pdf", (DATA / "refund_policy.pdf").read_bytes())
    assert docs and all("page" in d.meta for d in docs)
    assert docs[0].how == "text layer"


@pytest.mark.skipif(not (DATA / "scanned_invoice.pdf").exists(), reason="sample missing")
def test_scanned_pdf_routes_to_ocr():
    from rag.loaders import find_tesseract

    if not find_tesseract():
        pytest.skip("tesseract not installed")
    docs = load_bytes("scanned_invoice.pdf", (DATA / "scanned_invoice.pdf").read_bytes())
    assert docs and docs[0].how == "OCR (scanned)"
    assert "BLUE RIDGE" in " ".join(d.text for d in docs).upper()


# --- fusion and mmr ----------------------------------------------------------

def test_rrf_rewards_agreement_between_rankings():
    # 1 is top in both rankings; 2 and 3 are second in one ranking each.
    fused = reciprocal_rank_fusion([[1, 2], [1, 3]])
    assert fused[1] > fused[2] == pytest.approx(fused[3])


def test_rrf_respects_weights():
    a = reciprocal_rank_fusion([[1], [2]], [1.0, 0.1])
    assert a[1] > a[2]


# index 0 and 1 are identical; 2 points elsewhere and is less relevant.
DUP_REL = np.array([1.0, 1.0, 0.6], dtype="float32")
DUP_CANDS = np.array([[1.0, 0.0], [1.0, 0.0], [0.6, 0.8]], dtype="float32")


def test_mmr_drops_near_duplicates_when_diversity_is_favoured():
    picked = mmr(DUP_REL, DUP_CANDS, [0, 1, 2], k=2, lambda_=0.3)
    assert 2 in picked and picked.count(0) + picked.count(1) == 1


def test_mmr_keeps_duplicates_when_relevance_dominates():
    # lambda=1 disables the redundancy penalty entirely, so the two identical
    # top vectors both survive. This documents what the knob actually does.
    assert mmr(DUP_REL, DUP_CANDS, [0, 1, 2], k=2, lambda_=1.0) == [0, 1]


def test_mmr_relevance_is_not_recomputed_from_vectors():
    """A chunk BM25 found but cosine rates poorly must still win on fused score.

    This is the bug that made hybrid search pointless: MMR used to derive
    relevance from cosine, discarding the BM25 half of the ranking.
    """
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    fused_relevance = np.array([0.2, 1.0], dtype="float32")  # index 1 wins
    assert mmr(fused_relevance, vecs, [0, 1], k=1, lambda_=1.0) == [1]


def test_mmr_is_a_noop_when_candidates_fit():
    assert mmr(np.array([1.0, 0.5]), np.eye(2, dtype="float32"), [0, 1], k=5, lambda_=0.5) == [0, 1]


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, World! FR-4") == ["hello", "world", "fr", "4"]


# --- store -------------------------------------------------------------------


@pytest.fixture
def workdir():
    """A scratch directory. pytest's workdir cleanup can fail on Windows."""
    import shutil
    import tempfile

    path = Path(tempfile.mkdtemp(prefix="ragtest-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="module")
def store() -> Store:
    chunks = [
        Chunk("Refunds for card payments take 7 to 10 business days.", "policy.pdf", "text layer"),
        Chunk("Damaged items must be claimed within 10 calendar days.", "policy.pdf", "text layer"),
        Chunk("Employees accrue 18 days of annual leave per year.", "hr.pdf", "text layer"),
        Chunk("ticket: T-1009; status: Rejected; reason: outside window", "t.csv", "table row"),
    ]
    return Store.build(chunks)


def test_store_builds_and_searches(store):
    hits = search(store, "How long do card refunds take?", k=2)
    assert hits and "card payments" in hits[0].chunk.text


def test_bm25_finds_rare_literal_tokens(store):
    # "T-1009" is a token dense embeddings handle poorly; BM25 should carry it.
    ids = [i for i, _ in store.sparse("T-1009", 3)]
    assert store.chunks[ids[0]].source == "t.csv"


def test_confidence_gate_rejects_unrelated_questions(store):
    assert is_confident(search(store, "How long do card refunds take?"))
    assert not is_confident(search(store, "What is the capital of France?"))


def test_store_round_trips_through_disk(store, workdir):
    store.save(workdir)
    assert json.loads((workdir / "manifest.json").read_text())["chunks"] == len(store)

    loaded = Store.load(workdir)
    assert len(loaded) == len(store)
    assert loaded.sources == store.sources
    np.testing.assert_allclose(loaded.vectors, store.vectors)

    before = [h.chunk.text for h in search(store, "annual leave", k=2)]
    after = [h.chunk.text for h in search(loaded, "annual leave", k=2)]
    assert before == after  # results must survive a restart


def test_loading_a_mismatched_model_is_refused(store, workdir):
    store.save(workdir)
    manifest = workdir / "manifest.json"
    data = json.loads(manifest.read_text())
    data["model"] = "some-other-model"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="built with"):
        Store.load(workdir)


def test_loading_a_missing_index_explains_how_to_fix_it(workdir):
    with pytest.raises(FileNotFoundError, match="rag ingest"):
        Store.load(workdir / "nothing")


def test_empty_index_is_rejected():
    with pytest.raises(ValueError, match="zero chunks"):
        Store.build([])


# --- end to end (no LLM) -----------------------------------------------------


@pytest.mark.skipif(not DATA.exists(), reason="no sample data")
def test_real_documents_index_and_retrieve():
    chunks = split_documents(load_dir(DATA))
    assert len(chunks) > 5
    store = Store.build(chunks)
    hits = search(store, "How many days to claim a damaged item?")
    assert any("refund_policy" in h.chunk.source for h in hits)
