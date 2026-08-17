"""chunks.py - segments -> 800-1,500 token chunks landed as `seed` rows.

Fixtures are structural shapes with invented prose (word-count paragraphs) -
no verbatim S.C.R./eval text anywhere, matching every other builder test in
this suite.
"""

import hashlib
import json
import random
from pathlib import Path

import pytest

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
from tuned.data.seeds import INJUDGEMENTS_SOURCE_ID
from tuned.data.segment import SEGMENT_VERSION, Segment, segment_document
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


def test_a_segment_version_bump_forces_rechunking_via_the_manifest():
    # Simulated without actually bumping the module constant: writes a
    # manifest row claiming an OLDER segment_version, then checks the real
    # resume decision function reads it correctly.
    from tuned.data.chunks import _doc_chunk_decision

    doc = {"extract_version": 5, "sha256": "aa"}
    stale = {"extract_version": 5, "sha256": "aa", "segment_version": 0, "chunk_version": CHUNK_VERSION}
    current = {"extract_version": 5, "sha256": "aa", "segment_version": SEGMENT_VERSION, "chunk_version": CHUNK_VERSION}
    assert _doc_chunk_decision(stale, doc, force=False) == "chunk"
    assert _doc_chunk_decision(current, doc, force=False) == "skip"
    assert _doc_chunk_decision(None, doc, force=False) == "chunk"
    assert _doc_chunk_decision(current, doc, force=True) == "chunk"


def test_sha_changing_alone_forces_rechunk_even_with_extract_version_unchanged():
    # Independent of the extract_version test above: the source PDF was
    # re-extracted under the SAME rule version (extract_version unchanged)
    # but produced different bytes - sha256 alone must be enough to trigger
    # a re-chunk. Isolated from test_a_changed_sha_replaces_the_chunks (the
    # end-to-end version), which always changes extract_version alongside
    # sha and so cannot tell the two branches apart on its own.
    from tuned.data.chunks import _doc_chunk_decision

    doc = {"extract_version": 5, "sha256": "new-sha"}
    prior = {
        "extract_version": 5, "sha256": "old-sha",
        "segment_version": SEGMENT_VERSION, "chunk_version": CHUNK_VERSION,
    }
    assert _doc_chunk_decision(prior, doc, force=False) == "chunk"


def test_extract_version_changing_alone_forces_rechunk_even_with_sha_unchanged():
    # The mirror case: rules changed (extract_version bumped) but this
    # particular document's bytes happen to be identical either way (a real
    # possibility - not every rule change moves every document's text).
    from tuned.data.chunks import _doc_chunk_decision

    doc = {"extract_version": 6, "sha256": "same-sha"}
    prior = {
        "extract_version": 5, "sha256": "same-sha",
        "segment_version": SEGMENT_VERSION, "chunk_version": CHUNK_VERSION,
    }
    assert _doc_chunk_decision(prior, doc, force=False) == "chunk"


def test_neither_sha_nor_extract_version_changing_skips():
    from tuned.data.chunks import _doc_chunk_decision

    doc = {"extract_version": 5, "sha256": "same-sha"}
    prior = {
        "extract_version": 5, "sha256": "same-sha",
        "segment_version": SEGMENT_VERSION, "chunk_version": CHUNK_VERSION,
    }
    assert _doc_chunk_decision(prior, doc, force=False) == "skip"


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


def test_shuffled_seed_row_processing_order_gives_the_same_chunk_ids(tmp_path):
    random.seed(0)
    texts = {f"doc{i}": numbered_paragraphs([150 + i * 10] * 4) for i in range(4)}

    def run(order):
        with Store.open(tmp_path / f"shuffle-{'-'.join(order)}.sqlite3") as store:
            store.upsert_source(SC_SOURCE_ID, "Public Domain")
            for key in order:
                _write_doc(store, tmp_path / f"shuffle-{'-'.join(order)}-files", key, texts[key])
            chunk_documents(store, tokenizer=FakeTokenizer())
            return {r["seed_id"] for r in store.seeds_by_source(SC_SOURCE_ID)}

    keys = list(texts)
    first = run(keys)
    shuffled = keys[:]
    random.shuffle(shuffled)
    second = run(shuffled)
    assert first == second


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
# Conventions.
# --------------------------------------------------------------------------


def test_cli_hard_exits_after_success():
    assert "os._exit(" in CHUNKS_SRC.read_text(encoding="utf-8")


def test_all_sql_stays_in_store_py():
    source = CHUNKS_SRC.read_text(encoding="utf-8")
    for banned in ("store.conn.execute", "self._conn.execute", "sqlite3.connect"):
        assert banned not in source
