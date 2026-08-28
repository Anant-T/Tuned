"""segment.py's segments -> 800-1,500 token chunks, recorded as `seed` rows.

TWO INPUTS, one packer. The plan's Layer A middle sits between extract.py's
output and the teacher-generation streams that read `store.seed`, and there
are two places whole judgment text is sitting unchunked when this module's
work begins:

  THE `document` TABLE       extract.py's own output - `text_path` on disk,
                              one row per judgment, `status='ok'` meaning the
                              headnote is already stripped and the text is
                              the court's own words. chunk_documents drives
                              this one.
  `seed` ROWS FROM `seeds.py`  specifically the InJudgements rows -
                              seeds.py's own docstring says so directly:
                              "Text is kept WHOLE here (no chunking) -
                              segment.py chunks it later by segment."
                              chunk_seed_rows drives this one, and it
                              REPLACES the whole-text parent row with its
                              chunks (deletes the parent, inserts the
                              children) rather than leaving both in the
                              table - a wave planner that can see both would
                              plan against the 50,000-word original as
                              readily as against its own chunks, which is
                              exactly the prompt-budget blowout chunking
                              exists to prevent.

PREDEX AND TATHYANYAYA ARE LEFT WHOLE ON PURPOSE, AND NOT BECAUSE THEY ARE
SHORT. An earlier version of this docstring said they "are already short
excerpts", which is false: measured 2026-08-26 over all 60,603 seeds, 92.8%
of PredEx and 69.9% of TathyaNyaya rows exceed MAX_CHUNK_TOKENS, with p99s of
10,389 and 11,734 tokens and maxima of 50,369 and 30,098
(docs/reports/2026-08-26-row-length-under-deepseek-traces.md). Chunking them
would look like an obvious win and is a trap.

The real reason is that each of those seeds CARRIES ITS OWN ANSWER.
seeds.predex_seed builds its text as `facts + "\n\n" + the court's reasoning`
and tathyanyaya_seed is documented as the same treatment, so a chunk of one is
either an unanswerable fragment (facts, reasoning removed) or the answer with
no question (reasoning, facts removed). There is no split of a whole case that
preserves the task, which is why this module takes the InJudgements source
alone.

Their LENGTH is still a real problem, and it is solved one layer up rather
than here: tasks.seed_token_budget refuses to plan a wave against a seed with
no room left for its own reply. Every row assemble.py has ever dropped for
length came from these two sources, precisely because the `oversize` flag
below can only exist on rows this module chunked.

Both drivers funnel through the same core: segment_document (segment.py)
produces tier-selected segments, pack_chunks bins them into the token band
without ever splitting one, and chunk_seed_row shapes the result into the
`seed` table's existing schema - no parallel store, per the brief.

CONTENT-DERIVED IDS. A chunk's seed_id is seed_id_for(source_id,
"{object_key}:{start}:{end}") - the SAME derivation seeds.py already uses
for its own rows, not a second hashing scheme. Because start/end are a pure
function of the source bytes plus SEGMENT_VERSION/CHUNK_VERSION, chunking the
same document twice - in any order, interleaved with any other document -
produces the same ids and the same boundaries. Nothing here reads global run
state or a counter.

RESUME, PER DOCUMENT. chunk_manifest (store.py) is this module's twin of
extract.py's document-index resume: one row per (source_id, object_key)
naming the tier used, the rules' versions, the source sha, and the exact
seed_ids it wrote. A document is re-chunked only when its extract_version or
sha256 changed (the source moved) or SEGMENT_VERSION/CHUNK_VERSION/
ROLES_VERSION changed (the rules moved) or --force. On any of those the OLD seed_ids are deleted
before the new ones are written - replaced, never duplicated, and never left
as orphaned rows nothing points at any more once a document's chunk
boundaries shift.

Build:  python -m tuned.data.chunks --config data/configs/data_law_v1.yaml
        [--limit N] [--force] [--roles-backend opennyai-subprocess]
"""

import json
import sqlite3
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tuned.data import roles_infer
from tuned.data.acquire import SC_SOURCE_ID
from tuned.data.extract import STATUS_OK as DOC_STATUS_OK
from tuned.data.roles_infer import ROLES_VERSION
from tuned.data.seeds import INJUDGEMENTS_SOURCE_ID, classify_case_type, seed_id_for
from tuned.data.segment import SEGMENT_VERSION, TIER_ROLES, Segment, segment_document

# Bump when the packing rule itself changes what a chunk's boundaries are -
# independent of SEGMENT_VERSION, which governs the segments a chunk is
# packed FROM. Both travel on every chunk's meta_json and on its
# chunk_manifest row, because a chunk's identity is a function of both.
#
#   1  greedy left-to-right bin-packing up to 1500 tokens, a lone oversize
#      segment emitted alone and flagged, never truncated. A "merge the
#      small trailing chunk into its predecessor" pass was written and then
#      removed before shipping - proved mathematically (and by 200,000
#      randomised trials, zero counterexamples) unreachable given this same
#      greedy algorithm's own flush condition; see pack_chunks' docstring.
CHUNK_VERSION = 1

MIN_CHUNK_TOKENS = 800
MAX_CHUNK_TOKENS = 1500

# Court metadata that is true of the WHOLE document-table corpus by
# construction (acquire.py's bucket is exclusively the S.C.R. reprint of
# Supreme Court judgments) rather than a per-document fact this module
# infers - safe to set unconditionally, unlike a citation or a decision date
# extract.py never parsed.
DOCUMENT_COURT = "Supreme Court of India"

CHUNK_KIND = "chunk"


@dataclass(frozen=True)
class Chunk:
    start: int
    end: int
    text: str
    token_count: int
    labels: tuple[str, ...]
    oversize: bool


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def pack_chunks(
    text: str,
    segments: Sequence[Segment],
    tokenizer,
    *,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[Chunk]:
    """Whole segments, greedily binned into chunks of AT MOST max_tokens
    under `tokenizer`. NEVER splits inside a segment: a segment whose own
    token count already exceeds max_tokens is emitted ALONE, flagged
    `oversize`, and never truncated - the hard rule the brief states and the
    one property every other choice here is subordinate to.

    MIN_CHUNK_TOKENS (800) is the corpus-level TARGET this module reports
    against, not a floor this function enforces: greedy, order-preserving
    bin-packing that may never split a segment cannot also guarantee every
    bin clears a minimum, and a document that runs out of segments leaves
    its last chunk however large the remaining material is. Measured on the
    15 staged documents it is missed 19 times in 145 chunks, and 11 of those
    19 are INTERIOR chunks rather than trailing ones - a chunk that ran into
    an oversize segment it could not absorb and was flushed early. Both
    counts are reported by the drivers below (`under_min_chunks`,
    `in_band_chunks`, `token_histogram`), because the brief's band-adherence
    criterion is not checkable from a run that only counts chunks written.

    Greedy, not globally optimal: segments are added to the current chunk in
    order until the NEXT one would overflow max_tokens, at which point the
    current chunk is flushed and a new one starts.

    A "merge the small trailing chunk into its predecessor when they still
    fit together" pass was written and then proven dead rather than shipped:
    whatever chunk immediately precedes a flush, call its total T, was
    flushed BECAUSE the next segment (of size R, R>0) could not be added -
    i.e. T + R > max_tokens. The chunk built from that rejected segment
    starts at R or more, so T + (that chunk's own total) is T + R or larger
    - always > max_tokens by the same inequality that caused the flush in
    the first place. A merge that only fires when the combination still fits
    under max_tokens can therefore never fire on two chunks this function
    itself produced (checked exhaustively over thousands of randomised
    segment-size combinations, zero counterexamples). The mathematics say
    the same thing the search does, so the branch is gone rather than kept
    as code nothing can reach.
    """
    if not segments:
        return []
    sized = [(seg, _token_count(tokenizer, text[seg.start : seg.end])) for seg in segments]

    chunks: list[Chunk] = []
    current: list[tuple[Segment, int]] = []
    current_tok = 0

    def flush(items: list[tuple[Segment, int]], *, oversize: bool) -> None:
        start = items[0][0].start
        end = items[-1][0].end
        chunks.append(
            Chunk(
                start=start,
                end=end,
                text=text[start:end],
                # The JOINED span, not the sum of the parts. The two differ
                # whenever a segment boundary is not also a token boundary,
                # which never happens on the packing tier (every paragraph
                # start is newline-preceded and this tokenizer's BPE breaks
                # at newlines - measured: 0 of 94 real chunks differed) and
                # can happen on the roles tier, whose spans are not
                # line-anchored. Recording the sum there would let a chunk
                # sit over the band with a number saying it did not. The
                # PACKING decision below still uses per-segment sizes, so
                # this function encodes each segment once and each chunk
                # once rather than re-encoding the whole chunk per candidate.
                token_count=_token_count(tokenizer, text[start:end]),
                # First-seen order, deduplicated: on the packing tier the
                # labels are distinct paragraph numbers and this is a no-op,
                # while on the ToC and roles tiers every piece of one
                # section carries that section's heading, and a chunk of
                # four of them means "this chunk is in ANALYSIS", not
                # "ANALYSIS, ANALYSIS, ANALYSIS, ANALYSIS".
                labels=tuple(dict.fromkeys(seg.label for seg, _tok in items if seg.label is not None)),
                oversize=oversize,
            )
        )

    for seg, tok in sized:
        if tok > max_tokens:
            if current:
                flush(current, oversize=False)
                current, current_tok = [], 0
            flush([(seg, tok)], oversize=True)
            continue
        if current and current_tok + tok > max_tokens:
            flush(current, oversize=False)
            current, current_tok = [], 0
        current.append((seg, tok))
        current_tok += tok
    if current:
        flush(current, oversize=False)
    return chunks


def chunk_id_for(source_id: str, object_key: str, chunk: Chunk) -> str:
    """Content-derived id: seeds.py's own seed_id_for over (source_id,
    "object_key:start:end"). Deterministic given the source bytes and the
    two version constants that decide where start/end fall - never a
    function of processing order, an autoincrement, or which other
    documents were chunked in the same run."""
    return seed_id_for(source_id, f"{object_key}:{chunk.start}:{chunk.end}")


def chunk_seed_row(
    *,
    source_id: str,
    object_key: str,
    chunk: Chunk,
    tier: str,
    why: str,
    degradation: dict | None,
    extract_version: int | None,
    doc_sha256: str | None,
    text_path: str | None,
    court: str | None = None,
    case_type: str | None = None,
    code_era: str | None = None,
    decision_date: str | None = None,
    provenance: dict | None = None,
) -> dict:
    """One Chunk -> one `seed` row, in the table's existing schema.

    `roles_json` carries the chunk's role labels ONLY on the roles tier -
    None elsewhere is the honest reading ("this chunk's rhetorical role was
    never assessed"), not a placeholder. `meta_json` carries everything an
    auditor needs to relocate the chunk byte-for-byte in its source
    (text_path + start/end + the source's own sha256) plus the two version
    numbers that decided the boundary and every upstream provenance field
    the caller has (case_id, citation, year, marker, or - for a
    seed-table-origin chunk - the parent seed_id it replaced).
    """
    meta = {
        "kind": CHUNK_KIND,
        "object_key": object_key,
        "text_path": text_path,
        "start": chunk.start,
        "end": chunk.end,
        "tier": tier,
        "why": why,
        "degradation": degradation,
        "labels": list(chunk.labels),
        "oversize": chunk.oversize,
        "segment_version": SEGMENT_VERSION,
        "chunk_version": CHUNK_VERSION,
        "roles_version": ROLES_VERSION,
        "extract_version": extract_version,
        "doc_sha256": doc_sha256,
        "token_estimator": "tokenizer.encode",
        **(provenance or {}),
    }
    return {
        "seed_id": chunk_id_for(source_id, object_key, chunk),
        "source_id": source_id,
        "native_id": f"{object_key}:{chunk.start}:{chunk.end}",
        "court": court,
        "decision_date": decision_date,
        "offence_date": None,
        "case_type": case_type,
        "code_era": code_era,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "roles_json": list(chunk.labels) if tier == TIER_ROLES else None,
        "answer_key_json": None,
        "meta_json": meta,
    }


# --------------------------------------------------------------------------
# Driver 1: the `document` table (extract.py's output).
# --------------------------------------------------------------------------


def _rules_current(prior: dict | None) -> bool:
    """True when `prior` was written under exactly today's three rule
    versions. The roles version is in here for the same reason the other two
    are: it governs what a roles-tier chunk's boundaries ARE, so a bump to
    it has to invalidate the chunks it produced. Left out (as it was), a
    roles-tier chunk is stale forever - the tier travels on the manifest and
    was never compared to anything.
    """
    return (
        prior is not None
        and prior.get("segment_version") == SEGMENT_VERSION
        and prior.get("chunk_version") == CHUNK_VERSION
        and prior.get("roles_version") == ROLES_VERSION
    )


def _doc_chunk_decision(prior: dict | None, doc: dict, *, force: bool) -> str:
    """"chunk" or "skip" - the whole resume policy for one document.

    Three ways to be out of date and one way to be current: no manifest row
    at all, the source document's own extract_version or sha256 moved (the
    text this document points at changed), or this module's own rules moved
    (SEGMENT_VERSION/CHUNK_VERSION/ROLES_VERSION). --force re-chunks
    everything regardless.
    """
    if force or prior is None:
        return "chunk"
    if prior.get("extract_version") != doc.get("extract_version"):
        return "chunk"
    if prior.get("sha256") != doc.get("sha256"):
        return "chunk"
    if not _rules_current(prior):
        return "chunk"
    return "skip"


# Width of one bucket in the token histogram below. 100 tokens is fine
# enough to read p50/p90 off the band edges (800 and 1500 both fall on a
# bucket boundary) and coarse enough that a 15,000-chunk run's stats stay a
# couple of dozen integers rather than a 15,000-element list in the event log.
TOKEN_BUCKET = 100


def _new_stats() -> dict:
    return {
        "considered": 0,
        "chunked": 0,
        "skipped": 0,
        "replaced": 0,
        "empty": 0,
        "failed": 0,
        "failures": [],
        "chunks_written": 0,
        "oversize_chunks": 0,
        # The brief's acceptance criterion is band adherence, and a run that
        # reports only chunks_written cannot answer it - the first real-data
        # numbers for this module had to be computed by an ad-hoc external
        # script because MIN_CHUNK_TOKENS appears in no conditional anywhere
        # in src/. These two make it observable from the pipeline itself.
        "under_min_chunks": 0,
        "in_band_chunks": 0,
        "token_histogram": {},
        "tiers": {},
    }


def _tally_chunks(stats: dict, chunks: Sequence[Chunk]) -> None:
    """Fold one document's chunks into the run stats' band accounting."""
    for chunk in chunks:
        if chunk.oversize:
            stats["oversize_chunks"] += 1
        if chunk.token_count < MIN_CHUNK_TOKENS:
            stats["under_min_chunks"] += 1
        # Counted exactly, not read off the histogram - MAX_CHUNK_TOKENS is
        # inclusive and falls on a bucket boundary, so a chunk of exactly
        # 1500 tokens lands in the 1500-1599 bucket and a bucket-derived
        # share would report it out of band.
        if MIN_CHUNK_TOKENS <= chunk.token_count <= MAX_CHUNK_TOKENS:
            stats["in_band_chunks"] += 1
        bucket = (chunk.token_count // TOKEN_BUCKET) * TOKEN_BUCKET
        key = f"{bucket}-{bucket + TOKEN_BUCKET - 1}"
        stats["token_histogram"][key] = stats["token_histogram"].get(key, 0) + 1


def _record_failure(stats: dict, key: str, exc: BaseException) -> None:
    """This document failed; the pass continues.

    Every other per-document failure in this pipeline is recorded and
    stepped over, and chunking is designed to interleave with wave planning
    over a weeks-long run - so a routine extract_version bump on one
    already-planned document (delete_seeds hits the task foreign key) or a
    text file that was pruned out from under the document row must cost that
    ONE document, not the whole pass after earlier documents were already
    rewritten. Deliberately NO chunk_manifest row is written here: the next
    run must find this document unrecorded and try it again, which is the
    same resume contract every other status carries.
    """
    stats["failed"] += 1
    stats["failures"].append({"object_key": key, "reason": f"{type(exc).__name__}: {exc}"})


# What "this document is unusable right now" looks like coming out of the
# store or the filesystem. sqlite3.Error covers delete_seeds' FK
# IntegrityError; OSError covers a missing or unreadable text_path. Anything
# else - a bug in this module, a KeyboardInterrupt - is not a per-document
# degradation and is left to end the run loudly.
_DOCUMENT_FAILURE = (sqlite3.Error, OSError, UnicodeDecodeError)


def chunk_documents(
    store,
    *,
    source_id: str = SC_SOURCE_ID,
    tokenizer,
    roles_backend: str = roles_infer.BACKEND_NONE,
    roles_python_bin: str | None = None,
    roles_timeout: float = roles_infer.DEFAULT_TIMEOUT_S,
    roles_spawn=subprocess.run,
    force: bool = False,
    limit: int | None = None,
) -> dict:
    """Chunk every `status='ok'` document under `source_id`, resumably.

    `limit` caps documents actually CHUNKED (not considered), matching
    extract.py's own reasoning: a resumed run should advance past what it
    already has rather than spend the cap re-confirming skips.
    """
    manifest_index = store.chunk_manifest_index(source_id)
    stats = _new_stats()

    for doc in store.documents(source_id, status=DOC_STATUS_OK):
        if limit is not None and stats["chunked"] >= limit:
            break
        stats["considered"] += 1
        object_key = doc["object_key"]
        prior = manifest_index.get(object_key)
        if _doc_chunk_decision(prior, doc, force=force) == "skip":
            stats["skipped"] += 1
            continue

        try:
            text = Path(doc["text_path"]).read_text(encoding="utf-8")
            result = segment_document(
                text,
                roles_backend=roles_backend,
                roles_python_bin=roles_python_bin,
                roles_timeout=roles_timeout,
                roles_spawn=roles_spawn,
            )
            chunks = pack_chunks(text, result.segments, tokenizer)
            rows = [
                chunk_seed_row(
                    source_id=source_id,
                    object_key=object_key,
                    chunk=chunk,
                    tier=result.tier,
                    why=result.why,
                    degradation=result.degradation,
                    extract_version=doc.get("extract_version"),
                    doc_sha256=doc.get("sha256"),
                    text_path=doc.get("text_path"),
                    court=DOCUMENT_COURT,
                    case_type=classify_case_type(chunk.text),
                    code_era=None,
                    provenance={
                        "case_id": doc.get("case_id"),
                        "citation": doc.get("citation"),
                        "year": doc.get("year"),
                        "marker": doc.get("marker"),
                    },
                )
                for chunk in chunks
            ]

            if prior is not None:
                old_ids = json.loads(prior.get("seed_ids_json") or "[]")
                if old_ids:
                    store.delete_seeds(old_ids)
                stats["replaced"] += 1

            if rows:
                store.upsert_seeds(rows)
            store.record_chunk_manifest(
                source_id,
                object_key,
                {
                    "status": "ok" if rows else "empty",
                    "reason": None if rows else "empty_text",
                    "tier": result.tier,
                    "why": result.why,
                    "chunk_count": len(rows),
                    "seed_ids_json": [row["seed_id"] for row in rows],
                    "sha256": doc.get("sha256"),
                    "extract_version": doc.get("extract_version"),
                    "segment_version": SEGMENT_VERSION,
                    "chunk_version": CHUNK_VERSION,
                    "roles_version": ROLES_VERSION,
                    "meta_json": {"degradation": result.degradation},
                },
            )
        except _DOCUMENT_FAILURE as exc:
            _record_failure(stats, object_key, exc)
            continue

        stats["chunked"] += 1
        stats["chunks_written"] += len(rows)
        _tally_chunks(stats, chunks)
        stats["tiers"][result.tier] = stats["tiers"].get(result.tier, 0) + 1
        if not rows:
            stats["empty"] += 1
    return stats


# --------------------------------------------------------------------------
# Driver 2: whole-text `seed` rows (seeds.py's InJudgements, by its own
# stated boundary - see the module docstring).
# --------------------------------------------------------------------------


def _is_chunk_row(row: dict) -> bool:
    """True for a seed row THIS module already produced - never re-chunked,
    and never mistaken for a whole document waiting to be chunked. Reads
    meta_json rather than a dedicated column: `seed` is the wave planner's
    table and this task's contract is to land chunks in its EXISTING
    schema, not to grow it."""
    try:
        meta = json.loads(row.get("meta_json") or "{}")
    except ValueError:
        return False
    return isinstance(meta, dict) and meta.get("kind") == CHUNK_KIND


def chunk_seed_rows(
    store,
    *,
    source_id: str = INJUDGEMENTS_SOURCE_ID,
    tokenizer,
    roles_backend: str = roles_infer.BACKEND_NONE,
    roles_python_bin: str | None = None,
    roles_timeout: float = roles_infer.DEFAULT_TIMEOUT_S,
    roles_spawn=subprocess.run,
    force: bool = False,
    limit: int | None = None,
) -> dict:
    """Chunk every whole-text `seed` row under `source_id`, replacing each
    parent row with its chunks.

    A row already carrying `meta_json.kind == "chunk"` (this module's own
    prior output) is never treated as a new whole document to chunk - that
    is what stops a re-run chunking its own chunks. Everything else under
    `source_id` is a candidate exactly once: this driver's resume key is the
    PARENT row's own seed_id (there is no object_key here - seed_id is
    already the stable, content-derived identity seeds.py gives it), and
    once a parent is replaced its id no longer appears in `seed` at all, so
    a second run naturally has nothing left to reconsider for that document
    unless SEGMENT_VERSION/CHUNK_VERSION moved or --force was passed.

    THE ONE CASE THAT DOES REACH A "skip" HERE: seeds.py's own upsert is
    content-derived and idempotent, so a LATER normalization run over the
    same raw HF row can legitimately recreate a whole-text row under the
    SAME seed_id this driver already chunked and deleted once. That
    resurrected parent is still deleted below even on the skip path - only
    the CHUNKING work is skipped (nothing about the rules or the bytes
    moved, so redoing it would waste work and, being content-derived,
    reproduce the identical chunk ids) - because leaving the resurrected
    parent in `seed` would let the wave planner see both the whole judgment
    and its own already-existing chunks, which is exactly the prompt-budget
    duplication chunking exists to prevent.
    """
    manifest_index = store.chunk_manifest_index(source_id)
    stats = _new_stats()

    for row in store.iter_seeds_by_source(source_id):
        if _is_chunk_row(row):
            continue
        if limit is not None and stats["chunked"] >= limit:
            break
        stats["considered"] += 1
        parent_id = row["seed_id"]
        prior = manifest_index.get(parent_id)
        # A whole seed row is content-fixed once seeds.py upserts it (its
        # own seed_id IS a hash of its content), so there is no source
        # sha/extract_version to compare here - only whether THIS module's
        # own rules moved, or --force.
        if not force and _rules_current(prior):
            try:
                store.delete_seeds([parent_id])
            except _DOCUMENT_FAILURE as exc:
                _record_failure(stats, parent_id, exc)
                continue
            stats["skipped"] += 1
            continue

        try:
            text = row.get("text") or ""
            result = segment_document(
                text,
                roles_backend=roles_backend,
                roles_python_bin=roles_python_bin,
                roles_timeout=roles_timeout,
                roles_spawn=roles_spawn,
            )
            chunks = pack_chunks(text, result.segments, tokenizer)
            rows = [
                chunk_seed_row(
                    source_id=source_id,
                    object_key=parent_id,
                    chunk=chunk,
                    tier=result.tier,
                    why=result.why,
                    degradation=result.degradation,
                    extract_version=None,
                    doc_sha256=None,
                    text_path=None,
                    court=row.get("court"),
                    case_type=row.get("case_type") or classify_case_type(chunk.text),
                    code_era=row.get("code_era"),
                    decision_date=row.get("decision_date"),
                    provenance={"parent_seed_id": parent_id, "native_id": row.get("native_id")},
                )
                for chunk in chunks
            ]

            if prior is not None:
                old_ids = json.loads(prior.get("seed_ids_json") or "[]")
                if old_ids:
                    store.delete_seeds(old_ids)
                stats["replaced"] += 1

            if rows:
                store.upsert_seeds(rows)
            # The parent is removed LAST, after its children are durably
            # written: a crash between the two leaves both the whole row and
            # its chunks in the table, which a re-run resolves the same way
            # record_document resolves a crash between text and index - by
            # finding the manifest row absent (or stale) and redoing the
            # work, never by losing it. The other order would destroy the
            # whole judgment with nothing yet written to replace it.
            store.delete_seeds([parent_id])
            store.record_chunk_manifest(
                source_id,
                parent_id,
                {
                    "status": "ok" if rows else "empty",
                    "reason": None if rows else "empty_text",
                    "tier": result.tier,
                    "why": result.why,
                    "chunk_count": len(rows),
                    "seed_ids_json": [r["seed_id"] for r in rows],
                    "sha256": None,
                    "extract_version": None,
                    "segment_version": SEGMENT_VERSION,
                    "chunk_version": CHUNK_VERSION,
                    "roles_version": ROLES_VERSION,
                    "meta_json": {"degradation": result.degradation},
                },
            )
        except _DOCUMENT_FAILURE as exc:
            _record_failure(stats, parent_id, exc)
            continue

        stats["chunked"] += 1
        stats["chunks_written"] += len(rows)
        _tally_chunks(stats, chunks)
        stats["tiers"][result.tier] = stats["tiers"].get(result.tier, 0) + 1
        if not rows:
            stats["empty"] += 1
    return stats


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------


def _print_band(stats: dict) -> None:
    """The band histogram, in-band share and the in-band bucket's edges.

    The acceptance criterion for this layer is chunks in [800, 1500] tokens,
    and until this printed, answering "how did the run do against it" needed
    a script outside the pipeline reading the store by hand.
    """
    histogram = stats["token_histogram"]
    if not histogram:
        return
    total = sum(histogram.values())
    in_band = stats["in_band_chunks"]
    print(f"    band[{MIN_CHUNK_TOKENS}-{MAX_CHUNK_TOKENS}]: {in_band}/{total} "
          f"({100 * in_band / total:.1f}%)")
    for key in sorted(histogram, key=lambda k: int(k.split("-")[0])):
        print(f"      {key:>12}: {histogram[key]}")


def main(argv: Sequence[str] | None = None, *, tokenizer=None) -> int:
    import argparse
    import os
    import sys

    from tuned.data.assemble import load_tokenizer
    from tuned.data.config import load_build_config
    from tuned.data.paths import build_paths
    from tuned.data.store import Store

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="data/configs/data_law_v1.yaml")
    parser.add_argument("--limit", type=int, default=None, help="stop after N documents chunked")
    parser.add_argument("--force", action="store_true", help="re-chunk everything, ignoring resume")
    parser.add_argument(
        "--roles-backend",
        default=roles_infer.BACKEND_NONE,
        choices=roles_infer.BACKENDS,
        help="OpenNyAI subprocess bridge backend (default: none - packing only)",
    )
    parser.add_argument("--roles-python-bin", default=None, help="interpreter with opennyai installed")
    parser.add_argument("--roles-timeout", type=float, default=roles_infer.DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--skip-seed-rows",
        action="store_true",
        help="only chunk_documents (the document table) - skip re-chunking whole-text seed rows",
    )
    args = parser.parse_args(argv)

    cfg = load_build_config(args.config)
    paths = build_paths(cfg.build.workdir).ensure()
    store = Store.open(paths.state_db)
    if tokenizer is None:
        tokenizer = load_tokenizer(cfg)

    code = 0
    try:
        doc_stats = chunk_documents(
            store,
            tokenizer=tokenizer,
            roles_backend=args.roles_backend,
            roles_python_bin=args.roles_python_bin,
            roles_timeout=args.roles_timeout,
            force=args.force,
            limit=args.limit,
        )
        print(
            f"documents  considered {doc_stats['considered']}  chunked {doc_stats['chunked']}  "
            f"skipped {doc_stats['skipped']}  replaced {doc_stats['replaced']}  "
            f"failed {doc_stats['failed']}  "
            f"chunks {doc_stats['chunks_written']}  oversize {doc_stats['oversize_chunks']}  "
            f"under-min {doc_stats['under_min_chunks']}"
        )
        for tier, count in sorted(doc_stats["tiers"].items()):
            print(f"    tier[{tier}]: {count}")
        _print_band(doc_stats)
        for failure in doc_stats["failures"]:
            print(f"    failed[{failure['object_key']}]: {failure['reason']}")

        seed_stats = _new_stats()
        if not args.skip_seed_rows:
            seed_stats = chunk_seed_rows(
                store,
                tokenizer=tokenizer,
                roles_backend=args.roles_backend,
                roles_python_bin=args.roles_python_bin,
                roles_timeout=args.roles_timeout,
                force=args.force,
                limit=args.limit,
            )
            print(
                f"seed rows  considered {seed_stats['considered']}  chunked {seed_stats['chunked']}  "
                f"skipped {seed_stats['skipped']}  failed {seed_stats['failed']}  "
                f"chunks {seed_stats['chunks_written']}  "
                f"oversize {seed_stats['oversize_chunks']}  "
                f"under-min {seed_stats['under_min_chunks']}"
            )
            for tier, count in sorted(seed_stats["tiers"].items()):
                print(f"    tier[{tier}]: {count}")
            _print_band(seed_stats)
            for failure in seed_stats["failures"]:
                print(f"    failed[{failure['object_key']}]: {failure['reason']}")

        store.log_event(
            "chunking_pass",
            {"documents": doc_stats, "seed_rows": seed_stats, "segment_version": SEGMENT_VERSION,
             "chunk_version": CHUNK_VERSION},
        )
        print(f"seed_count total -> {store.seed_count()} ({paths.state_db})")
    finally:
        store.close()

    # Same reasoning as extract.py/seeds.py: an optionally-loaded real
    # tokenizer (transformers/torch) can leave non-daemon threads that wedge
    # interpreter shutdown after all output is written.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
