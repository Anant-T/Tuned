"""chunks.py - segments -> 800-1,500 token chunks landed as `seed` rows.

Fixtures are structural shapes with invented prose (word-count paragraphs) -
no verbatim S.C.R./eval text anywhere, matching every other builder test in
this suite.
"""

import hashlib
import json
import os
import random
import re
from pathlib import Path

import pytest
from pipeline_fakes import temp_config

from tuned.data.acquire import SC_SOURCE_ID
from tuned.data.chunks import (
    CHUNK_VERSION,
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    Chunk,
    chunk_documents,
    chunk_id_for,
    chunk_seed_row,
    chunk_seed_rows,
    pack_chunks,
)
from tuned.data.roles_infer import BACKEND_SUBPROCESS, ROLES_VERSION
from tuned.data.seeds import INJUDGEMENTS_SOURCE_ID
from tuned.data.segment import (
    MAX_PARA_STEP,
    SEGMENT_VERSION,
    TIER_PACKING,
    TIER_ROLES,
    TIER_TOC,
    Segment,
    segment_document,
)
from tuned.data.config import load_build_config
from tuned.data.paths import build_paths
from tuned.data.store import Store

CHUNKS_SRC = Path(__file__).parent.parent / "src" / "tuned" / "data" / "chunks.py"


class FakeTokenizer:
    """One token per whitespace word - the same shape assemble.py's own
    fixture tokenizer uses, without the chat-template half chunks.py has no
    use for (chunks are raw text, not rendered messages)."""

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(len(text.split())))


def words(n: int) -> str:
    return ("word " * n).strip()


def numbered_paragraphs(counts: list[int], *, start: int = 1) -> str:
    """counts[i] words in paragraph (start + i)."""
    return "\n\n".join(f"{start + i}. {words(n)}." for i, n in enumerate(counts))


# --------------------------------------------------------------------------
# pack_chunks: the band, the oversize rule, determinism, the merge heuristic.
# --------------------------------------------------------------------------


def test_never_splits_inside_a_segment():
    text = numbered_paragraphs([2000])  # one huge paragraph, alone
    segments = segment_document(text).segments
    chunks = pack_chunks(text, segments, FakeTokenizer())
    assert len(chunks) == 1
    assert chunks[0].start == 0 and chunks[0].end == len(text)
    assert chunks[0].oversize is True


def test_a_paragraph_over_max_is_emitted_alone_and_flagged_never_truncated():
    text = numbered_paragraphs([100, 100, 2000, 100])
    segments = segment_document(text).segments
    chunks = pack_chunks(text, segments, FakeTokenizer())
    oversized = [c for c in chunks if c.oversize]
    assert len(oversized) == 1
    # The oversize chunk's full text is present, byte for byte - never cut.
    big_para_text = text[text.index("3. "):text.index("4. ")].rstrip("\n")
    assert oversized[0].text.strip().startswith("3.")
    assert oversized[0].token_count == 2001  # "3." + 2000 words counted as tokens


def test_ordinary_chunks_land_within_the_band_when_enough_material_exists():
    # Six paragraphs of 300 words each = 1800 "tokens" of raw material -
    # enough to fill at least one full [800, 1500] chunk without relying on
    # the small-tail merge.
    text = numbered_paragraphs([300] * 6)
    segments = segment_document(text).segments
    chunks = pack_chunks(text, segments, FakeTokenizer())
    assert len(chunks) >= 1
    for chunk in chunks[:-1]:
        assert MIN_CHUNK_TOKENS <= chunk.token_count <= MAX_CHUNK_TOKENS


def test_no_two_adjacent_ordinary_chunks_could_still_be_combined_under_the_band():
    # A mathematical property of the greedy packer, not an active merge
    # step (there isn't one - see pack_chunks' docstring for the proof that
    # one would always be a no-op here): whatever chunk immediately follows
    # a flush was flushed BECAUSE adding the next chunk's own first segment
    # would already have overflowed max_tokens, so two adjacent ordinary
    # chunks can never sum to <= max_tokens. Regression protection for that
    # invariant, not a test of a merge that does not exist.
    text = numbered_paragraphs([350, 350, 350, 350, 50])
    segments = segment_document(text).segments
    chunks = pack_chunks(text, segments, FakeTokenizer())
    for i in range(len(chunks) - 1):
        if not chunks[i].oversize and not chunks[i + 1].oversize:
            assert chunks[i].token_count + chunks[i + 1].token_count > MAX_CHUNK_TOKENS


def _sized_segments(sizes: list[int]) -> tuple[str, list[Segment]]:
    """Contiguous segments whose FakeTokenizer word count is EXACTLY
    `sizes[i]` each - precise enough to test the >, >= boundary itself,
    which prose built from `numbered_paragraphs` cannot promise."""
    parts, segs, cursor = [], [], 0
    for i, n in enumerate(sizes):
        piece = " ".join(f"s{i}w{j}" for j in range(n)) + " "
        segs.append(Segment(cursor, cursor + len(piece), str(i + 1)))
        parts.append(piece)
        cursor += len(piece)
    return "".join(parts), segs


def test_a_segment_exactly_at_max_tokens_is_not_oversize():
    text, segs = _sized_segments([MAX_CHUNK_TOKENS])
    chunks = pack_chunks(text, segs, FakeTokenizer())
    assert len(chunks) == 1
    assert chunks[0].token_count == MAX_CHUNK_TOKENS
    assert chunks[0].oversize is False


def test_a_segment_one_token_over_max_is_oversize():
    text, segs = _sized_segments([MAX_CHUNK_TOKENS + 1])
    chunks = pack_chunks(text, segs, FakeTokenizer())
    assert len(chunks) == 1
    assert chunks[0].oversize is True


def test_two_segments_summing_to_exactly_max_tokens_share_one_chunk():
    text, segs = _sized_segments([MAX_CHUNK_TOKENS - 700, 700])
    chunks = pack_chunks(text, segs, FakeTokenizer())
    assert len(chunks) == 1
    assert chunks[0].token_count == MAX_CHUNK_TOKENS
    assert chunks[0].oversize is False


def test_one_token_past_the_max_sum_forces_a_second_chunk():
    text, segs = _sized_segments([MAX_CHUNK_TOKENS - 700, 701])
    chunks = pack_chunks(text, segs, FakeTokenizer())
    assert len(chunks) == 2
    assert chunks[0].token_count == MAX_CHUNK_TOKENS - 700
    assert chunks[1].token_count == 701
    assert not chunks[0].oversize and not chunks[1].oversize


def test_chunks_reconstruct_the_source_text_byte_for_byte():
    text = numbered_paragraphs([100, 2000, 300, 300, 300, 300, 50])
    segments = segment_document(text).segments
    chunks = pack_chunks(text, segments, FakeTokenizer())
    assert "".join(c.text for c in chunks) == text
    for chunk in chunks:
        assert text[chunk.start : chunk.end] == chunk.text


def test_no_segments_means_no_chunks():
    assert pack_chunks("", [], FakeTokenizer()) == []


def test_packing_is_deterministic_regardless_of_repeated_calls():
    text = numbered_paragraphs([120] * 10)
    segments = segment_document(text).segments
    tok = FakeTokenizer()
    first = pack_chunks(text, segments, tok)
    second = pack_chunks(text, segments, tok)
    assert [(c.start, c.end) for c in first] == [(c.start, c.end) for c in second]


# --------------------------------------------------------------------------
# chunk_id_for: content-derived, order-independent.
# --------------------------------------------------------------------------


def test_chunk_id_is_a_pure_function_of_source_object_key_and_offsets():
    chunk = Chunk(start=10, end=200, text="x" * 190, token_count=190, labels=(), oversize=False)
    a = chunk_id_for(SC_SOURCE_ID, "year=2020/x.pdf", chunk)
    b = chunk_id_for(SC_SOURCE_ID, "year=2020/x.pdf", chunk)
    assert a == b


def test_chunk_id_differs_by_document_by_offsets_and_by_source():
    chunk = Chunk(start=10, end=200, text="x" * 190, token_count=190, labels=(), oversize=False)
    base = chunk_id_for(SC_SOURCE_ID, "k1", chunk)
    assert chunk_id_for(SC_SOURCE_ID, "k2", chunk) != base
    assert chunk_id_for("other-source", "k1", chunk) != base
    shifted = Chunk(start=11, end=200, text="x" * 189, token_count=189, labels=(), oversize=False)
    assert chunk_id_for(SC_SOURCE_ID, "k1", shifted) != base


# --------------------------------------------------------------------------
# chunk_seed_row: the seed-table shape.
# --------------------------------------------------------------------------


def test_chunk_seed_row_carries_offsets_and_versions_for_the_audit():
    chunk = Chunk(start=5, end=55, text="y" * 50, token_count=50, labels=("2",), oversize=False)
    row = chunk_seed_row(
        source_id=SC_SOURCE_ID,
        object_key="k1",
        chunk=chunk,
        tier="packing",
        why="fallback",
        degradation={"from": "roles", "reason": "roles_backend_none"},
        extract_version=5,
        doc_sha256="ab" * 32,
        text_path="corpus/text/k1.txt",
        court="Supreme Court of India",
        case_type="criminal",
        code_era=None,
        provenance={"case_id": "C1", "citation": "[2020] 1 SCR 1", "year": 2020},
    )
    assert row["text"] == "y" * 50
    assert row["token_count"] == 50
    assert row["roles_json"] is None  # only the roles tier populates this
    meta = row["meta_json"]
    assert (meta["start"], meta["end"]) == (5, 55)
    assert meta["text_path"] == "corpus/text/k1.txt"
    assert meta["segment_version"] == SEGMENT_VERSION
    assert meta["chunk_version"] == CHUNK_VERSION
    assert meta["case_id"] == "C1"
    assert meta["kind"] == "chunk"


def test_roles_tier_chunk_carries_its_labels_in_roles_json():
    chunk = Chunk(start=0, end=10, text="z" * 10, token_count=10, labels=("FAC", "ISSUE"), oversize=False)
    row = chunk_seed_row(
        source_id=SC_SOURCE_ID, object_key="k1", chunk=chunk, tier="roles", why="roles available",
        degradation=None, extract_version=5, doc_sha256="cd" * 32, text_path="t.txt",
    )
    assert row["roles_json"] == ["FAC", "ISSUE"]


# --------------------------------------------------------------------------
# The document-table driver: real store, real files on disk.
# --------------------------------------------------------------------------


def _write_doc(store, tmp_path, key, text, *, extract_version=5, **doc_over):
    path = tmp_path / f"{key}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    row = {
        "status": "ok", "text_path": str(path), "case_id": f"C-{key}",
        "citation": "[2020] 1 SCR 1", "year": 2020, "pages": 10,
        "page_start": 1, "page_end": 10, "chars": len(text), "headnote_chars": 0,
        "marker": "judgment_delivered_by", "sha256": sha, "extract_version": extract_version,
        "meta": {},
    }
    row.update(doc_over)
    store.record_document(SC_SOURCE_ID, key, row)
    return sha


@pytest.fixture
def doc_store(tmp_path):
    with Store.open(tmp_path / "state.sqlite3") as store:
        store.upsert_source(SC_SOURCE_ID, "Public Domain")
        yield store, tmp_path


def test_chunk_documents_writes_one_manifest_row_and_at_least_one_chunk(doc_store):
    store, tmp_path = doc_store
    _write_doc(store, tmp_path, "k1", numbered_paragraphs([200] * 5))
    stats = chunk_documents(store, tokenizer=FakeTokenizer())
    assert stats["chunked"] == 1
    assert stats["chunks_written"] >= 1
    manifest = store.chunk_manifest(SC_SOURCE_ID, "k1")
    assert manifest["status"] == "ok"
    assert manifest["chunk_count"] == stats["chunks_written"]
    assert json.loads(manifest["seed_ids_json"])
    assert store.seed_count(SC_SOURCE_ID) == stats["chunks_written"]


def test_only_status_ok_documents_are_chunked(doc_store):
    store, tmp_path = doc_store
    store.record_document(SC_SOURCE_ID, "bad", {
        "status": "quarantined", "reason": "no_text", "text_path": None,
        "case_id": None, "citation": None, "year": None, "pages": 0,
        "page_start": None, "page_end": None, "chars": None, "headnote_chars": None,
        "marker": None, "sha256": None, "extract_version": 5, "meta": {},
    })
    stats = chunk_documents(store, tokenizer=FakeTokenizer())
    assert stats["considered"] == 0
    assert stats["chunked"] == 0


def test_resume_skips_an_already_chunked_document(doc_store):
    store, tmp_path = doc_store
    _write_doc(store, tmp_path, "k1", numbered_paragraphs([200] * 5))
    chunk_documents(store, tokenizer=FakeTokenizer())
    before = store.seed_count(SC_SOURCE_ID)
    stats2 = chunk_documents(store, tokenizer=FakeTokenizer())
    assert stats2["chunked"] == 0
    assert stats2["skipped"] == 1
    assert store.seed_count(SC_SOURCE_ID) == before


def test_a_changed_sha_replaces_the_chunks_never_duplicates(doc_store):
    store, tmp_path = doc_store
    _write_doc(store, tmp_path, "k1", numbered_paragraphs([200] * 5))
    chunk_documents(store, tokenizer=FakeTokenizer())
    old_ids = set(json.loads(store.chunk_manifest(SC_SOURCE_ID, "k1")["seed_ids_json"]))

    # A rule change re-extracted the same document with different bytes.
    new_text = numbered_paragraphs([250] * 6)
    _write_doc(store, tmp_path, "k1", new_text, extract_version=6)
    stats = chunk_documents(store, tokenizer=FakeTokenizer())
    assert stats["chunked"] == 1
    assert stats["replaced"] == 1

    new_ids = set(json.loads(store.chunk_manifest(SC_SOURCE_ID, "k1")["seed_ids_json"]))
    assert new_ids != old_ids
    for old_id in old_ids:
        assert store.get_seed(old_id) is None  # replaced, not duplicated
    assert store.seed_count(SC_SOURCE_ID) == len(new_ids)


def test_force_rechunks_even_when_nothing_changed(doc_store):
    store, tmp_path = doc_store
    _write_doc(store, tmp_path, "k1", numbered_paragraphs([200] * 5))
    chunk_documents(store, tokenizer=FakeTokenizer())
    stats = chunk_documents(store, tokenizer=FakeTokenizer(), force=True)
    assert stats["chunked"] == 1
    assert stats["replaced"] == 1


def _prior(**over) -> dict:
    """A manifest row at exactly today's rules; `over` moves one field."""
    row = {
        "extract_version": 5,
        "sha256": "aa",
        "segment_version": SEGMENT_VERSION,
        "chunk_version": CHUNK_VERSION,
        "roles_version": ROLES_VERSION,
    }
    row.update(over)
    return row


@pytest.mark.parametrize("field", ["segment_version", "chunk_version", "roles_version"])
def test_each_rule_version_on_its_own_forces_rechunking_via_the_manifest(field):
    # Simulated without actually bumping the module constants: writes a
    # manifest row claiming an OLDER version of ONE rule and checks the real
    # resume decision reads it. Parametrized because the first cut tested
    # only the segment_version half - deleting the chunk_version half of the
    # decision's `or` left the whole suite green, and roles_version was
    # compared nowhere at all.
    from tuned.data.chunks import _doc_chunk_decision

    doc = {"extract_version": 5, "sha256": "aa"}
    assert _doc_chunk_decision(_prior(**{field: -1}), doc, force=False) == "chunk"
    assert _doc_chunk_decision(_prior(), doc, force=False) == "skip"
    assert _doc_chunk_decision(None, doc, force=False) == "chunk"
    assert _doc_chunk_decision(_prior(), doc, force=True) == "chunk"


@pytest.mark.parametrize("field", ["segment_version", "chunk_version", "roles_version"])
def test_each_rule_version_is_recorded_on_the_manifest_row_it_will_be_compared_to(doc_store, field):
    # The other half: comparing a version the writer never records is a
    # comparison against None that happens to be right by accident.
    store, tmp_path = doc_store
    _write_doc(store, tmp_path, "k1", numbered_paragraphs([200] * 5))
    chunk_documents(store, tokenizer=FakeTokenizer())
    manifest = store.chunk_manifest(SC_SOURCE_ID, "k1")
    current = {
        "segment_version": SEGMENT_VERSION,
        "chunk_version": CHUNK_VERSION,
        "roles_version": ROLES_VERSION,
    }
    assert manifest[field] == current[field]
    row = store.seeds_by_source(SC_SOURCE_ID)[0]
    assert json.loads(row["meta_json"])[field] == current[field]


def test_a_roles_version_bump_rechunks_a_document_end_to_end(doc_store, monkeypatch):
    # Through the real driver rather than the decision function alone: the
    # manifest carries the version the run was made under, so moving the
    # constant must make the next run redo the work.
    store, tmp_path = doc_store
    _write_doc(store, tmp_path, "k1", numbered_paragraphs([200] * 5))
    chunk_documents(store, tokenizer=FakeTokenizer())
    assert chunk_documents(store, tokenizer=FakeTokenizer())["chunked"] == 0

    monkeypatch.setattr("tuned.data.chunks.ROLES_VERSION", ROLES_VERSION + 1)
    stats = chunk_documents(store, tokenizer=FakeTokenizer())
    assert stats["chunked"] == 1
    assert stats["replaced"] == 1
    assert store.chunk_manifest(SC_SOURCE_ID, "k1")["roles_version"] == ROLES_VERSION + 1


def test_sha_changing_alone_forces_rechunk_even_with_extract_version_unchanged():
    # Independent of the extract_version test above: the source PDF was
    # re-extracted under the SAME rule version (extract_version unchanged)
    # but produced different bytes - sha256 alone must be enough to trigger
    # a re-chunk. Isolated from test_a_changed_sha_replaces_the_chunks (the
    # end-to-end version), which always changes extract_version alongside
    # sha and so cannot tell the two branches apart on its own.
    from tuned.data.chunks import _doc_chunk_decision

    doc = {"extract_version": 5, "sha256": "new-sha"}
    assert _doc_chunk_decision(_prior(sha256="old-sha"), doc, force=False) == "chunk"


def test_extract_version_changing_alone_forces_rechunk_even_with_sha_unchanged():
    # The mirror case: rules changed (extract_version bumped) but this
    # particular document's bytes happen to be identical either way (a real
    # possibility - not every rule change moves every document's text).
    from tuned.data.chunks import _doc_chunk_decision

    doc = {"extract_version": 6, "sha256": "same-sha"}
    assert _doc_chunk_decision(_prior(sha256="same-sha"), doc, force=False) == "chunk"


def test_neither_sha_nor_extract_version_changing_skips():
    from tuned.data.chunks import _doc_chunk_decision

    doc = {"extract_version": 5, "sha256": "same-sha"}
    assert _doc_chunk_decision(_prior(sha256="same-sha"), doc, force=False) == "skip"


def test_oversize_paragraphs_are_counted_in_the_run_stats(doc_store):
    store, tmp_path = doc_store
    _write_doc(store, tmp_path, "k1", numbered_paragraphs([2000]))
    stats = chunk_documents(store, tokenizer=FakeTokenizer())
    assert stats["oversize_chunks"] == 1


def test_tier_usage_is_tallied_per_run(doc_store):
    store, tmp_path = doc_store
    _write_doc(store, tmp_path, "k1", numbered_paragraphs([200] * 5))
    stats = chunk_documents(store, tokenizer=FakeTokenizer())
    assert stats["tiers"] == {"packing": 1}


def test_case_type_and_court_are_set_on_document_derived_chunks(doc_store):
    store, tmp_path = doc_store
    text = "JUDGMENT\n\nRAO, J.\n\n" + numbered_paragraphs(
        [200], start=1
    ).replace("1. word", "1. The accused was charged under the Indian Penal Code with murder. word")
    _write_doc(store, tmp_path, "k1", text)
    chunk_documents(store, tokenizer=FakeTokenizer())
    rows = store.seeds_by_source(SC_SOURCE_ID)
    assert rows[0]["court"] == "Supreme Court of India"
    assert rows[0]["case_type"] == "criminal"


# --------------------------------------------------------------------------
# Determinism, across processing order.
# --------------------------------------------------------------------------


def test_chunking_two_documents_in_either_order_gives_identical_ids_and_bounds(tmp_path):
    text_a = numbered_paragraphs([180] * 4)
    text_b = numbered_paragraphs([220] * 5)

    results = {}
    for order in (("a", "b"), ("b", "a")):
        with Store.open(tmp_path / f"order-{''.join(order)}.sqlite3") as store:
            store.upsert_source(SC_SOURCE_ID, "Public Domain")
            docs = {"a": text_a, "b": text_b}
            for key in order:
                _write_doc(store, tmp_path / f"order-{''.join(order)}-files", f"doc-{key}", docs[key])
            stats = chunk_documents(store, tokenizer=FakeTokenizer())
            rows = sorted(
                store.seeds_by_source(SC_SOURCE_ID), key=lambda r: r["seed_id"]
            )
            results[order] = [(r["seed_id"], json.loads(r["meta_json"])["start"], json.loads(r["meta_json"])["end"]) for r in rows]

    assert results[("a", "b")] == results[("b", "a")]


def test_shuffled_document_processing_order_gives_the_same_chunk_ids():
    # Shuffles the CHUNKING ITSELF, not the insertion order. Driving
    # chunk_documents with a shuffled store cannot fail: store.documents()
    # returns rows in object_key order, so the driver processes them in the
    # same order however the caller inserted them, and a shuffle test around
    # it passes for a reason that has nothing to do with the property. The
    # property being claimed is that a chunk's id and bounds are a pure
    # function of its own document - so it is the per-document calls that
    # have to be shuffled.
    random.seed(0)
    texts = {f"doc{i}": numbered_paragraphs([150 + i * 10] * 4) for i in range(4)}
    tok = FakeTokenizer()

    def run(order):
        out = {}
        for key in order:
            chunks = pack_chunks(texts[key], segment_document(texts[key]).segments, tok)
            out[key] = [(chunk_id_for(SC_SOURCE_ID, key, c), c.start, c.end) for c in chunks]
        return out

    keys = list(texts)
    shuffled = keys[:]
    random.shuffle(shuffled)
    assert shuffled != keys  # the shuffle really moved something
    assert run(keys) == run(shuffled)


def test_chunk_documents_walks_its_documents_in_object_key_order_whatever_the_insert_order(tmp_path):
    # The reason the shuffle above had to move: this is the real property
    # the end-to-end version was measuring, stated as itself.
    keys = ["doc3", "doc1", "doc2"]
    with Store.open(tmp_path / "order.sqlite3") as store:
        store.upsert_source(SC_SOURCE_ID, "Public Domain")
        for key in keys:
            _write_doc(store, tmp_path / "order-files", key, numbered_paragraphs([180] * 4))
        assert [d["object_key"] for d in store.documents(SC_SOURCE_ID, status="ok")] == sorted(keys)


# --------------------------------------------------------------------------
# The seed-table driver: whole InJudgements-shaped rows -> replaced by chunks.
# --------------------------------------------------------------------------


@pytest.fixture
def seed_store(tmp_path):
    with Store.open(tmp_path / "seedstate.sqlite3") as store:
        store.upsert_source(INJUDGEMENTS_SOURCE_ID, "Apache-2.0")
        yield store


def _whole_seed(store, seed_id, text, **over):
    row = {
        "seed_id": seed_id, "source_id": INJUDGEMENTS_SOURCE_ID, "native_id": f"url-{seed_id}",
        "court": "Delhi High Court", "case_type": "criminal", "code_era": "ipc",
        "text": text, "token_count": len(text) // 4, "meta_json": {"estimator": "chars/4"},
    }
    row.update(over)
    store.upsert_seeds([row])


def test_chunk_seed_rows_replaces_the_parent_with_its_chunks(seed_store):
    text = numbered_paragraphs([250] * 6)
    _whole_seed(seed_store, "whole1", text)
    stats = chunk_seed_rows(seed_store, tokenizer=FakeTokenizer())
    assert stats["chunked"] == 1
    assert seed_store.get_seed("whole1") is None
    rows = seed_store.seeds_by_source(INJUDGEMENTS_SOURCE_ID)
    assert len(rows) == stats["chunks_written"]
    for row in rows:
        meta = json.loads(row["meta_json"])
        assert meta["kind"] == "chunk"
        assert meta["parent_seed_id"] == "whole1"
        assert row["court"] == "Delhi High Court"
        assert row["code_era"] == "ipc"


def test_chunk_seed_rows_never_reprocesses_its_own_output(seed_store):
    text = numbered_paragraphs([250] * 6)
    _whole_seed(seed_store, "whole1", text)
    chunk_seed_rows(seed_store, tokenizer=FakeTokenizer())
    before = seed_store.seed_count(INJUDGEMENTS_SOURCE_ID)
    stats2 = chunk_seed_rows(seed_store, tokenizer=FakeTokenizer())
    assert stats2["considered"] == 0
    assert stats2["chunked"] == 0
    assert seed_store.seed_count(INJUDGEMENTS_SOURCE_ID) == before


def test_chunk_seed_rows_force_rechunks_and_replaces(seed_store):
    text = numbered_paragraphs([250] * 6)
    _whole_seed(seed_store, "whole1", text)
    chunk_seed_rows(seed_store, tokenizer=FakeTokenizer())
    old_ids = {r["seed_id"] for r in seed_store.seeds_by_source(INJUDGEMENTS_SOURCE_ID)}
    stats = chunk_seed_rows(seed_store, tokenizer=FakeTokenizer(), force=True)
    assert stats["chunked"] == 0  # nothing to re-chunk: the parent is gone
    # force only affects rows still present as WHOLE candidates; once
    # replaced, the parent's own identity is gone from `seed` and cannot be
    # "found" again by this driver - this is the expected, documented
    # boundary (a real re-chunk of already-chunked output would need
    # SEGMENT_VERSION/CHUNK_VERSION to move, not --force alone).
    assert {r["seed_id"] for r in seed_store.seeds_by_source(INJUDGEMENTS_SOURCE_ID)} == old_ids


def test_a_resurrected_whole_row_at_the_current_rules_is_skipped_but_still_deleted(seed_store):
    # The one scenario that DOES reach the "prior manifest matches current
    # rules" skip branch inside chunk_seed_rows: seeds.py's own upsert is
    # content-derived and idempotent, so a later normalization run over the
    # same raw HF row can legitimately recreate a whole-text seed under the
    # SAME seed_id this driver already chunked and deleted once. The
    # CHUNKING work is skipped (nothing moved, and it would reproduce the
    # identical ids anyway) - but the resurrected parent row must not be
    # left sitting next to its own already-existing chunks, which would let
    # the wave planner see both the whole judgment and its chunks at once.
    text = numbered_paragraphs([250] * 6)
    _whole_seed(seed_store, "whole1", text)
    chunk_seed_rows(seed_store, tokenizer=FakeTokenizer())
    manifest_before = seed_store.chunk_manifest(INJUDGEMENTS_SOURCE_ID, "whole1")
    ids_before = set(json.loads(manifest_before["seed_ids_json"]))

    _whole_seed(seed_store, "whole1", text)  # seeds.py re-normalizing the same raw row
    assert seed_store.get_seed("whole1") is not None  # really resurrected
    stats = chunk_seed_rows(seed_store, tokenizer=FakeTokenizer())
    assert stats["considered"] == 1
    assert stats["chunked"] == 0
    assert stats["skipped"] == 1
    assert seed_store.get_seed("whole1") is None  # resurrected parent removed again
    manifest_after = seed_store.chunk_manifest(INJUDGEMENTS_SOURCE_ID, "whole1")
    assert set(json.loads(manifest_after["seed_ids_json"])) == ids_before
    assert {r["seed_id"] for r in seed_store.seeds_by_source(INJUDGEMENTS_SOURCE_ID)} == ids_before


def test_predex_and_tathyanyaya_rows_are_never_touched_by_the_seed_driver(seed_store):
    # Only INJUDGEMENTS_SOURCE_ID rows are whole-text-unchunked per seeds.py's
    # own docstring; this driver must not reach into another source's rows
    # even if asked (the default source_id already scopes it, this proves it
    # is not accidentally global).
    other_source = "L-NLProc/PredEx_Instruction-Tuning_Pred-Exp"
    with Store.open(Path(seed_store.path).parent / "other.sqlite3") as other_store:
        other_store.upsert_source(other_source, "Apache-2.0")
        other_store.upsert_seeds([{
            "seed_id": "predex1", "source_id": other_source, "text": "short excerpt " * 50,
            "case_type": "criminal", "code_era": "ipc", "token_count": 100,
        }])
        stats = chunk_seed_rows(other_store, source_id=INJUDGEMENTS_SOURCE_ID, tokenizer=FakeTokenizer())
        assert stats["considered"] == 0
        assert other_store.get_seed("predex1") is not None


# --------------------------------------------------------------------------
# The real tokenizer path (importorskip - mirrors assemble.py's own test).
# --------------------------------------------------------------------------


def test_the_real_pinned_tokenizer_counts_tokens_the_same_way_the_fake_claims():
    transformers = pytest.importorskip("transformers")
    from tuned.data.config import load_build_config

    cfg = load_build_config("configs/data_law_v1.yaml")
    tok = transformers.AutoTokenizer.from_pretrained(cfg.model_repo, revision=cfg.model_revision)
    text = numbered_paragraphs([300] * 4)
    segments = segment_document(text).segments
    chunks = pack_chunks(text, segments, tok)
    assert chunks
    for chunk in chunks:
        assert chunk.token_count == len(tok.encode(chunk.text, add_special_tokens=False))


# --------------------------------------------------------------------------
# What the OTHER two tiers actually yield. Neither was ever driven through
# pack_chunks: both were tested for whether they fire, never for the chunk
# shapes they produce - which is how a tier with priority over packing came
# to produce chunks packing would never have made.
# --------------------------------------------------------------------------


def _toc_document(section_words: int = 400, per_section: int = 4) -> str:
    """A validated ToC whose sections are far larger than the token band."""
    parts, number = ["JUDGMENT\n"], 1
    for letter, heading in (("A", "Factual Matrix"), ("B", "Issues"), ("C", "Analysis")):
        parts.append(f"\n{letter}. {heading}\n\n")
        for _ in range(per_section):
            parts.append(f"{number}. {words(section_words)}.\n\n")
            number += 1
    return "".join(parts)


def test_a_toc_tier_document_is_packed_into_the_band_like_any_other(doc_store):
    store, tmp_path = doc_store
    text = _toc_document()
    _write_doc(store, tmp_path, "k1", text)
    result = segment_document(text)
    assert result.tier == TIER_TOC  # the tier really is the one under test
    stats = chunk_documents(store, tokenizer=FakeTokenizer())
    assert stats["tiers"] == {"toc": 1}
    assert stats["oversize_chunks"] == 0
    assert stats["chunks_written"] > 3  # more chunks than there are sections
    for row in store.seeds_by_source(SC_SOURCE_ID):
        assert row["token_count"] <= MAX_CHUNK_TOKENS


def test_a_toc_tier_chunk_carries_its_section_heading_once_not_once_per_paragraph(doc_store):
    store, tmp_path = doc_store
    _write_doc(store, tmp_path, "k1", _toc_document(section_words=120))
    chunk_documents(store, tokenizer=FakeTokenizer())
    seen = set()
    for row in store.seeds_by_source(SC_SOURCE_ID):
        labels = json.loads(row["meta_json"])["labels"]
        assert labels == list(dict.fromkeys(labels))  # deduplicated
        seen.update(labels)
    assert seen == {"Factual Matrix", "Issues", "Analysis"}


def test_no_tier_produces_an_oversize_chunk_that_packing_would_have_avoided(doc_store):
    # The rule stated as a comparison, on the one document where the tiers
    # actually disagree: whichever tier wins, its chunk set is at least as
    # in-band as the packing tier's, because its segments are a refinement.
    store, tmp_path = doc_store
    text = _toc_document()
    tok = FakeTokenizer()
    toc_chunks = pack_chunks(text, segment_document(text).segments, tok)
    from tuned.data.segment import _normalize_segments, _packing_tier

    packing_chunks = pack_chunks(text, _normalize_segments(text, _packing_tier(text)), tok)
    assert sum(c.oversize for c in toc_chunks) <= sum(c.oversize for c in packing_chunks)
    assert max(c.token_count for c in toc_chunks) <= max(c.token_count for c in packing_chunks)
    assert "".join(c.text for c in toc_chunks) == text


def test_a_roles_tier_document_is_packed_into_the_band_like_any_other(doc_store):
    store, tmp_path = doc_store
    text = numbered_paragraphs([400] * 8)
    _write_doc(store, tmp_path, "k1", text)
    half = len(text) // 2
    spawn = _fake_spawn({"spans": [[0, half, "FAC"], [half, len(text), "ANALYSIS"]]})
    stats = chunk_documents(
        store, tokenizer=FakeTokenizer(), roles_backend=BACKEND_SUBPROCESS, roles_spawn=spawn
    )
    assert stats["tiers"] == {"roles": 1}
    assert stats["oversize_chunks"] == 0
    rows = store.seeds_by_source(SC_SOURCE_ID)
    assert len(rows) > 2  # not one chunk per role span
    for row in rows:
        assert row["token_count"] <= MAX_CHUNK_TOKENS
        assert json.loads(row["roles_json"])  # the tier's labels travel


def _fake_spawn(reply):
    def run(args, *, input, capture_output, text, timeout):
        class R:
            returncode = 0
            stdout = json.dumps(reply)
            stderr = ""

        return R()

    return run


# --------------------------------------------------------------------------
# Band accounting: the acceptance criterion, observable from the run itself.
# --------------------------------------------------------------------------


def test_the_run_reports_the_band_it_was_asked_to_hit(doc_store):
    store, tmp_path = doc_store
    _write_doc(store, tmp_path, "k1", numbered_paragraphs([300] * 12))
    _write_doc(store, tmp_path, "k2", numbered_paragraphs([40]))  # one tiny chunk
    _write_doc(store, tmp_path, "k3", numbered_paragraphs([2000]))  # one oversize
    stats = chunk_documents(store, tokenizer=FakeTokenizer())

    assert sum(stats["token_histogram"].values()) == stats["chunks_written"]
    assert stats["oversize_chunks"] == 1
    assert stats["under_min_chunks"] >= 1
    assert stats["in_band_chunks"] + stats["under_min_chunks"] + stats["oversize_chunks"] == (
        stats["chunks_written"]
    )
    assert "0-99" in stats["token_histogram"]  # the 41-token document


def test_a_chunk_of_exactly_the_maximum_counts_as_in_band_not_as_a_bucket_artifact():
    # MAX_CHUNK_TOKENS is inclusive and sits on a histogram bucket edge, so
    # the in-band count is tallied exactly rather than read off the buckets.
    from tuned.data.chunks import _new_stats, _tally_chunks

    stats = _new_stats()
    _tally_chunks(stats, [Chunk(0, 1, "x", MAX_CHUNK_TOKENS, (), False)])
    assert stats["in_band_chunks"] == 1
    assert stats["under_min_chunks"] == 0
    assert stats["token_histogram"] == {"1500-1599": 1}


def test_token_count_is_the_encoding_of_the_chunk_not_the_sum_of_its_segments():
    # L6: the two differ exactly when a segment boundary is not a token
    # boundary. This tokenizer makes that visible on purpose - it counts
    # RUNS of a letter, so joining two segments that both touch the same run
    # yields fewer tokens than the parts do separately, and a chunk sized by
    # the sum would claim a number its own text does not have.
    class RunTokenizer:
        def encode(self, text, add_special_tokens=False):
            out, prev = [], None
            for ch in text:
                if ch != prev:
                    out.append(ch)
                prev = ch
            return out

    text = "aaaa" + "aaaa" + "bbbb"
    segments = [Segment(0, 4, "1"), Segment(4, 8, "2"), Segment(8, 12, "3")]
    chunks = pack_chunks(text, segments, RunTokenizer(), max_tokens=10)
    assert len(chunks) == 1
    tok = RunTokenizer()
    assert chunks[0].token_count == len(tok.encode(chunks[0].text))
    assert chunks[0].token_count == 2  # not 3, which the per-segment sum gives


# --------------------------------------------------------------------------
# Per-document failure containment: one bad document costs one document.
# --------------------------------------------------------------------------


def test_a_missing_text_file_fails_that_document_and_the_pass_continues(doc_store):
    store, tmp_path = doc_store
    _write_doc(store, tmp_path, "k1", numbered_paragraphs([200] * 4))
    _write_doc(store, tmp_path, "k2", numbered_paragraphs([200] * 4))
    _write_doc(store, tmp_path, "k3", numbered_paragraphs([200] * 4))
    Path(store.document(SC_SOURCE_ID, "k2")["text_path"]).unlink()

    stats = chunk_documents(store, tokenizer=FakeTokenizer())
    assert stats["chunked"] == 2
    assert stats["failed"] == 1
    assert [f["object_key"] for f in stats["failures"]] == ["k2"]
    assert "FileNotFoundError" in stats["failures"][0]["reason"]
    # no manifest row for the failure: the next run must try it again
    assert store.chunk_manifest(SC_SOURCE_ID, "k2") is None
    assert store.chunk_manifest(SC_SOURCE_ID, "k3") is not None


def test_a_document_that_failed_once_is_retried_by_the_next_pass(doc_store):
    store, tmp_path = doc_store
    text = numbered_paragraphs([200] * 4)
    _write_doc(store, tmp_path, "k1", text)
    path = Path(store.document(SC_SOURCE_ID, "k1")["text_path"])
    path.unlink()
    assert chunk_documents(store, tokenizer=FakeTokenizer())["failed"] == 1
    path.write_text(text, encoding="utf-8")
    stats = chunk_documents(store, tokenizer=FakeTokenizer())
    assert stats["chunked"] == 1 and stats["failed"] == 0


def test_a_foreign_key_error_from_delete_seeds_fails_that_document_only(doc_store):
    # The measured case: chunking interleaves with wave planning over a
    # weeks-long run, so a routine extract_version bump on a document a wave
    # was already planned against hits the task foreign key. That is a real
    # signal about ONE document, not a reason to end the pass after earlier
    # documents were already rewritten.
    store, tmp_path = doc_store
    for key in ("k1", "k2"):
        _write_doc(store, tmp_path, key, numbered_paragraphs([200] * 4))
    chunk_documents(store, tokenizer=FakeTokenizer())

    planned = json.loads(store.chunk_manifest(SC_SOURCE_ID, "k1")["seed_ids_json"])[0]
    store.create_tasks([{
        "task_id": "t1", "seed_id": planned, "stream": "s", "task_type": "irac_analysis",
        "prompt_id": "p", "prompt_sha": "x", "sample_ix": 0,
    }])
    for key in ("k1", "k2"):
        _write_doc(store, tmp_path, key, numbered_paragraphs([250] * 5), extract_version=6)

    stats = chunk_documents(store, tokenizer=FakeTokenizer())
    assert stats["failed"] == 1
    assert stats["failures"][0]["object_key"] == "k1"
    assert "IntegrityError" in stats["failures"][0]["reason"]
    assert stats["chunked"] == 1  # k2 got through
    assert store.get_seed(planned) is not None  # k1's rows are intact


def test_the_seed_driver_contains_a_per_row_failure_too(seed_store):
    text = numbered_paragraphs([250] * 6)
    _whole_seed(seed_store, "whole1", text)
    _whole_seed(seed_store, "whole2", text)
    chunk_seed_rows(seed_store, tokenizer=FakeTokenizer())
    child = seed_store.seeds_by_source(INJUDGEMENTS_SOURCE_ID)[0]["seed_id"]
    seed_store.create_tasks([{
        "task_id": "t1", "seed_id": child, "stream": "s", "task_type": "irac_analysis",
        "prompt_id": "p", "prompt_sha": "x", "sample_ix": 0,
    }])
    _whole_seed(seed_store, "whole1", text)  # resurrected parent, skip path
    _whole_seed(seed_store, "whole2", text)

    stats = chunk_seed_rows(seed_store, tokenizer=FakeTokenizer(), force=True)
    assert stats["failed"] + stats["chunked"] == 2
    assert stats["failed"] == 1
    assert "IntegrityError" in stats["failures"][0]["reason"]


# --------------------------------------------------------------------------
# The behaviours nothing constrained: --limit, the CLI, the durability
# ordering, and chunk_seed_rows' own version guard.
# --------------------------------------------------------------------------


def test_limit_caps_documents_chunked_not_documents_considered(doc_store):
    store, tmp_path = doc_store
    for key in ("k1", "k2", "k3"):
        _write_doc(store, tmp_path, key, numbered_paragraphs([200] * 4))
    stats = chunk_documents(store, tokenizer=FakeTokenizer(), limit=2)
    assert stats["chunked"] == 2
    assert stats["considered"] == 2
    assert store.chunk_manifest_count(SC_SOURCE_ID) == 2


def test_limit_spends_its_cap_on_new_work_not_on_re_confirming_skips(doc_store):
    # The docstring's specific claim, and the reason `limit` gates on
    # `chunked` rather than `considered`: a resumed run must advance.
    store, tmp_path = doc_store
    for key in ("k1", "k2", "k3"):
        _write_doc(store, tmp_path, key, numbered_paragraphs([200] * 4))
    chunk_documents(store, tokenizer=FakeTokenizer(), limit=1)
    stats = chunk_documents(store, tokenizer=FakeTokenizer(), limit=1)
    assert stats["skipped"] == 1
    assert stats["chunked"] == 1
    assert store.chunk_manifest_count(SC_SOURCE_ID) == 2


def test_limit_zero_chunks_nothing(doc_store):
    store, tmp_path = doc_store
    _write_doc(store, tmp_path, "k1", numbered_paragraphs([200] * 4))
    stats = chunk_documents(store, tokenizer=FakeTokenizer(), limit=0)
    assert stats == {**stats, "chunked": 0, "considered": 0}
    assert store.seed_count(SC_SOURCE_ID) == 0


def test_limit_caps_the_seed_driver_too(seed_store):
    for key in ("w1", "w2", "w3"):
        _whole_seed(seed_store, key, numbered_paragraphs([250] * 6))
    stats = chunk_seed_rows(seed_store, tokenizer=FakeTokenizer(), limit=2)
    assert stats["chunked"] == 2
    assert seed_store.get_seed("w3") is not None  # untouched, still whole


def test_the_parent_row_is_removed_only_after_its_chunks_are_written(seed_store):
    # The durability ordering the module comments at length about, with the
    # consequence made observable: under the other order a crash between the
    # two destroys the whole judgment with nothing written to replace it.
    order = []
    real_upsert, real_delete = seed_store.upsert_seeds, seed_store.delete_seeds

    def spy_upsert(rows):
        rows = list(rows)
        order.append(("upsert", tuple(r["seed_id"] for r in rows)))
        return real_upsert(rows)

    def spy_delete(ids):
        ids = list(ids)
        order.append(("delete", tuple(ids)))
        return real_delete(ids)

    seed_store.upsert_seeds, seed_store.delete_seeds = spy_upsert, spy_delete
    try:
        _whole_seed(seed_store, "whole1", numbered_paragraphs([250] * 6))
        order.clear()
        chunk_seed_rows(seed_store, tokenizer=FakeTokenizer())
    finally:
        seed_store.upsert_seeds, seed_store.delete_seeds = real_upsert, real_delete

    parent_delete = next(i for i, (op, ids) in enumerate(order) if op == "delete" and "whole1" in ids)
    children_upsert = next(i for i, (op, _ids) in enumerate(order) if op == "upsert")
    assert children_upsert < parent_delete


@pytest.mark.parametrize("field", ["segment_version", "chunk_version", "roles_version"])
def test_the_seed_driver_rechunks_when_one_rule_version_moved(seed_store, monkeypatch, field):
    # chunk_seed_rows' own four-line version guard, which no test reached:
    # with no manifest ever differing from current, the whole `if` collapsed
    # to "skip whenever a prior row exists" with nothing failing.
    text = numbered_paragraphs([250] * 6)
    _whole_seed(seed_store, "whole1", text)
    chunk_seed_rows(seed_store, tokenizer=FakeTokenizer())
    old_ids = {r["seed_id"] for r in seed_store.seeds_by_source(INJUDGEMENTS_SOURCE_ID)}
    _whole_seed(seed_store, "whole1", text)  # seeds.py recreating the parent

    constant = {
        "segment_version": "SEGMENT_VERSION",
        "chunk_version": "CHUNK_VERSION",
        "roles_version": "ROLES_VERSION",
    }[field]
    monkeypatch.setattr(f"tuned.data.chunks.{constant}", 99)
    stats = chunk_seed_rows(seed_store, tokenizer=FakeTokenizer())
    assert stats["chunked"] == 1
    assert stats["skipped"] == 0
    assert seed_store.get_seed("whole1") is None
    assert seed_store.chunk_manifest(INJUDGEMENTS_SOURCE_ID, "whole1")[field] == 99
    assert {r["seed_id"] for r in seed_store.seeds_by_source(INJUDGEMENTS_SOURCE_ID)} == old_ids


def test_the_seed_driver_skips_when_every_rule_version_matches(seed_store):
    text = numbered_paragraphs([250] * 6)
    _whole_seed(seed_store, "whole1", text)
    chunk_seed_rows(seed_store, tokenizer=FakeTokenizer())
    _whole_seed(seed_store, "whole1", text)
    stats = chunk_seed_rows(seed_store, tokenizer=FakeTokenizer())
    assert stats["skipped"] == 1 and stats["chunked"] == 0


def test_the_cli_chunks_a_real_store_and_reports_the_band(tmp_path, capsys, monkeypatch):
    # There were no chunks.main tests at all. os._exit is stubbed for the
    # same reason assemble.py's own CLI tests stub it - the module hard-exits
    # by design, and that is itself worth asserting.
    import tuned.data.chunks as chunks_mod

    config = temp_config(tmp_path)
    paths = build_paths(load_build_config(config).build.workdir).ensure()
    with Store.open(paths.state_db) as store:
        store.upsert_source(SC_SOURCE_ID, "Public Domain")
        _write_doc(store, tmp_path / "texts", "k1", numbered_paragraphs([300] * 6))

    exits = []
    monkeypatch.setattr(os, "_exit", lambda code: exits.append(code))
    chunks_mod.main(["--config", config, "--skip-seed-rows"], tokenizer=FakeTokenizer())
    out = capsys.readouterr().out
    assert exits == [0]
    assert "documents  considered 1  chunked 1" in out
    assert f"band[{MIN_CHUNK_TOKENS}-{MAX_CHUNK_TOKENS}]" in out
    with Store.open(paths.state_db) as store:
        assert store.seed_count(SC_SOURCE_ID) > 0


def test_the_cli_limit_flag_reaches_the_driver(tmp_path, capsys, monkeypatch):
    import tuned.data.chunks as chunks_mod

    config = temp_config(tmp_path)
    paths = build_paths(load_build_config(config).build.workdir).ensure()
    with Store.open(paths.state_db) as store:
        store.upsert_source(SC_SOURCE_ID, "Public Domain")
        for key in ("k1", "k2", "k3"):
            _write_doc(store, tmp_path / "texts", key, numbered_paragraphs([300] * 4))

    monkeypatch.setattr(os, "_exit", lambda code: None)
    chunks_mod.main(
        ["--config", config, "--limit", "1", "--skip-seed-rows"], tokenizer=FakeTokenizer()
    )
    assert "chunked 1" in capsys.readouterr().out
    with Store.open(paths.state_db) as store:
        assert store.chunk_manifest_count(SC_SOURCE_ID) == 1


# --------------------------------------------------------------------------
# Conventions.
# --------------------------------------------------------------------------


def test_cli_hard_exits_after_success():
    assert "os._exit(" in CHUNKS_SRC.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "module, name, value",
    [
        ("segment", "SEGMENT_VERSION", SEGMENT_VERSION),
        ("chunks", "CHUNK_VERSION", CHUNK_VERSION),
        ("roles_infer", "ROLES_VERSION", ROLES_VERSION),
    ],
)
def test_every_rule_version_has_a_ledger_line_for_the_value_it_currently_holds(
    module, name, value
):
    # These three constants are compared, never read for their value, so
    # every assertion about them elsewhere is `meta[x] == THE_CONSTANT` and
    # stays true however far the constant moves - measured: bumping
    # SEGMENT_VERSION to 999 left the whole suite green. What the convention
    # actually asks for is that a bump ARRIVES WITH ITS REASON, in the
    # numbered ledger comment above the assignment. That is checkable, and
    # this is where it is checked.
    source = (CHUNKS_SRC.parent / f"{module}.py").read_text(encoding="utf-8")
    lines = source.splitlines()
    at = next(i for i, line in enumerate(lines) if line.startswith(f"{name} = "))
    block, i = [], at - 1
    while i >= 0 and lines[i].startswith("#"):
        block.append(lines[i])
        i -= 1
    assert block, f"{name} carries no ledger comment at all"
    entries = [re.match(r"#\s{2,}(\d+)\s{2,}\S", line) for line in block]
    numbered = {int(m.group(1)) for m in entries if m}
    assert value in numbered, (
        f"{module}.{name} is {value} but its ledger names only {sorted(numbered)} - "
        "a version bump has to arrive with the reason it moved"
    )


def test_all_sql_stays_in_store_py():
    source = CHUNKS_SRC.read_text(encoding="utf-8")
    for banned in ("store.conn.execute", "self._conn.execute", "sqlite3.connect"):
        assert banned not in source
