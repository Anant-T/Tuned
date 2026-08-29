"""acquire.py - S3/HF corpus intake.

Nothing here touches the network: the bucket and the HF snapshot are both
injected seams (FakeBucket below, and a plain callable standing in for
snapshot_download), so every resume/durability property is driven from
fixtures.
"""

import ast
import hashlib
import os
import sys
from pathlib import Path

import pytest
from pipeline_fakes import temp_config

from tuned.data.acquire import (
    DEV_YEARS,
    HF_IDS_VERIFIED_AT,
    HF_SOURCES,
    PART_SUFFIX,
    SC_BUCKET,
    SC_LICENSE,
    SC_SOURCE_ID,
    AcquisitionError,
    GatedSourceError,
    HfSource,
    ObjectEntry,
    S3Bucket,
    acquire_hf,
    acquire_objects,
    download_object,
    entries_for,
    fetch_decision,
    index_tree,
    is_gated_error,
    local_path_for,
    main,
    parse_year,
    parse_years,
    sha256_file,
    year_prefixes,
)
from tuned.data.acquire import _READ_BLOCK as READ_BLOCK
from tuned.data.store import Store

ACQUIRE_SRC = Path(__file__).parent.parent / "src" / "tuned" / "data" / "acquire.py"


# --------------------------------------------------------------------------
# Doubles.
# --------------------------------------------------------------------------

class FakeBucket:
    """list_objects/fetch over an in-memory {key: bytes}.

    `fail_keys` raises mid-stream (a dropped connection); `truncate_keys`
    ends the body early with a clean EOF - the nastier case, because nothing
    raises and the file simply is not all there.
    """

    def __init__(self, objects: dict, *, fail_keys=(), truncate_keys=None, chunk: int = 8):
        self.objects = dict(objects)
        self.fail_keys = set(fail_keys)
        self.truncate_keys = dict(truncate_keys or {})
        self.chunk = chunk
        self.prefixes: list[str] = []
        self.fetched: list[str] = []

    def list_objects(self, prefix: str):
        self.prefixes.append(prefix)
        for key in sorted(self.objects):
            if key.startswith(prefix):
                yield ObjectEntry(key=key, size=len(self.objects[key]), etag=f"etag-{key}")

    def fetch(self, key: str):
        self.fetched.append(key)
        data = self.objects[key]
        if key in self.truncate_keys:
            data = data[: self.truncate_keys[key]]
        for i in range(0, len(data), self.chunk):
            if key in self.fail_keys and i:
                raise OSError("connection reset by peer")
            yield data[i : i + self.chunk]


class RefusesToFetch:
    """A bucket that fails the test if anything is downloaded through it."""

    def __init__(self, objects: dict):
        self.objects = dict(objects)

    def list_objects(self, prefix: str):
        for key in sorted(self.objects):
            if key.startswith(prefix):
                yield ObjectEntry(key=key, size=len(self.objects[key]), etag=f"etag-{key}")

    def fetch(self, key: str):
        raise AssertionError(f"re-downloaded {key!r}, which was already on disk")


class IndexFailsAt:
    """Store proxy whose Nth record_artifact raises - the process dying
    between the bytes landing and the row that points at them."""

    def __init__(self, store, at: int = 1):
        self._store = store
        self._at = at
        self.calls = 0

    def __getattr__(self, name):
        attr = getattr(self._store, name)
        if name != "record_artifact":
            return attr

        def recording(*args, **kwargs):
            self.calls += 1
            if self.calls >= self._at:
                raise RuntimeError("index write died")
            return attr(*args, **kwargs)

        return recording


PDF_KEY = "data/pdf/year=2015/english/2015_1_1_20_EN.pdf"


def _objects(n: int = 3, year: int = 2015) -> dict:
    return {
        f"data/pdf/year={year}/english/{year}_1_{i}_{i + 9}_EN.pdf": (f"pdf-{i}-".encode() * 20)
        for i in range(n)
    }


@pytest.fixture
def store(tmp_path):
    with Store.open(tmp_path / "state" / "law_v1.sqlite3") as s:
        yield s


# --------------------------------------------------------------------------
# Scope + bucket layout (both established in P0; neither is guessable).
# --------------------------------------------------------------------------

def test_the_dev_build_scope_is_2010_to_2025():
    # v1 is born-digital SC only: OCR (pre-1990 scans) is out of scope, and
    # 2026 is a partial year the bucket is still filling.
    assert DEV_YEARS[0] == 2010
    assert DEV_YEARS[-1] == 2025
    assert len(DEV_YEARS) == 16
    assert 2026 not in DEV_YEARS
    assert 1990 not in DEV_YEARS


def test_parse_years_accepts_a_range_a_list_and_a_mix():
    assert parse_years("2010-2012") == (2010, 2011, 2012)
    assert parse_years("2015") == (2015,)
    assert parse_years("2020,2010-2011") == (2020, 2010, 2011)
    assert parse_years("2011,2011") == (2011,)  # deduped, order kept


@pytest.mark.parametrize("spec", ["", "   ", "abc", "2020-2019", "2010-", "2010-2012-2014"])
def test_parse_years_rejects_nonsense(spec):
    with pytest.raises(ValueError):
        parse_years(spec)


def test_year_prefixes_match_the_layout_p0_walked():
    assert year_prefixes("pdf", (2015, 2016)) == (
        "data/pdf/year=2015/english/",
        "data/pdf/year=2016/english/",
    )
    assert year_prefixes("pdf", (2015,), language="regional") == ("data/pdf/year=2015/regional/",)
    assert year_prefixes("metadata", (2015,)) == ("metadata/parquet/year=2015/",)


def test_year_prefixes_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="tar"):
        year_prefixes("tar", (2015,))


def test_parse_year_reads_the_partition_out_of_a_key():
    assert parse_year(PDF_KEY) == 2015
    assert parse_year("metadata/parquet/year=2026/part-0.parquet") == 2026
    assert parse_year("data/tar/everything.tar") is None


# --------------------------------------------------------------------------
# Local layout + hashing.
# --------------------------------------------------------------------------

def test_local_path_for_mirrors_the_key_under_the_root(tmp_path):
    assert local_path_for(tmp_path, PDF_KEY) == tmp_path / PDF_KEY
    # dots inside a NAME are not a traversal
    ok = "data/pdf/year=2015/english/a.b..c.pdf"
    assert local_path_for(tmp_path, ok) == tmp_path / ok


@pytest.mark.parametrize(
    "key",
    ["../secret", "/etc/passwd", "C:/Windows/x", "a\\b", "data/../../x", "", "dir/"],
)
def test_local_path_for_refuses_a_key_that_escapes_the_root(tmp_path, key):
    # Keys come from a remote listing; one of them deciding where this
    # process writes is not a risk worth leaving open for free.
    with pytest.raises(ValueError):
        local_path_for(tmp_path, key)


def test_sha256_file_hashes_the_whole_file_not_the_first_block(tmp_path):
    # The fixture has to EXCEED _READ_BLOCK (1 MiB) or a first-block-only
    # hash passes this test: the metadata parquet and the InJudgements
    # snapshot are both larger than that, so a wrong hash would go into
    # artifact.sha256 and --verify would be blind past byte 1,048,576.
    blob = bytes(range(256)) * 8192  # 2,097,152 bytes = two full blocks
    assert len(blob) > READ_BLOCK
    path = tmp_path / "big.bin"
    path.write_bytes(blob)
    assert sha256_file(path) == hashlib.sha256(blob).hexdigest()
    assert sha256_file(path) != hashlib.sha256(blob[:READ_BLOCK]).hexdigest()


# --------------------------------------------------------------------------
# The resume decision.
# --------------------------------------------------------------------------

def _entry(size: int = 10) -> ObjectEntry:
    return ObjectEntry(key=PDF_KEY, size=size, etag="e")


def _row(size: int = 10, sha: str = "aa") -> dict:
    return {"size_bytes": size, "sha256": sha, "local_path": "p"}


def test_a_missing_file_is_fetched(tmp_path):
    assert fetch_decision(_entry(), tmp_path / "absent.pdf", None) == "fetch"


def test_a_file_of_the_wrong_size_is_re_fetched_never_adopted(tmp_path):
    path = tmp_path / "short.pdf"
    path.write_bytes(b"x" * 4)
    # Even with an index row claiming it is complete: the bytes on disk are
    # the authority, and 4 != 10.
    assert fetch_decision(_entry(10), path, _row(10)) == "fetch"


def test_a_file_the_index_agrees_with_is_skipped(tmp_path):
    path = tmp_path / "ok.pdf"
    path.write_bytes(b"x" * 10)
    assert fetch_decision(_entry(10), path, _row(10)) == "skip"


def test_a_complete_file_with_no_index_row_is_adopted(tmp_path):
    # The crash window: os.replace landed, record_artifact never ran.
    path = tmp_path / "ok.pdf"
    path.write_bytes(b"x" * 10)
    assert fetch_decision(_entry(10), path, None) == "adopt"


def test_a_stale_index_row_is_adopted_not_re_downloaded(tmp_path):
    path = tmp_path / "ok.pdf"
    path.write_bytes(b"x" * 10)
    assert fetch_decision(_entry(10), path, _row(7)) == "adopt"


def test_verify_re_reads_a_file_the_index_already_agrees_with(tmp_path):
    path = tmp_path / "ok.pdf"
    path.write_bytes(b"x" * 10)
    assert fetch_decision(_entry(10), path, _row(10), verify=True) == "adopt"


def test_an_object_re_uploaded_under_a_key_we_already_hold_is_fetched_again(tmp_path):
    # The SC bucket is a rolling release: an object CAN change under a key we
    # already hold, and at the same length only the ETag says so. Nothing may
    # verify content AGAINST an ETag (multipart is "<md5>-<parts>"), but
    # inequality between what we recorded and what the listing now reports is
    # still evidence the object was re-uploaded - and it is free, because the
    # listing carries it either way.
    path = tmp_path / "ok.pdf"
    path.write_bytes(b"x" * 10)
    held = {"size_bytes": 10, "sha256": "aa", "local_path": "p", "etag": "one"}

    assert fetch_decision(ObjectEntry(PDF_KEY, 10, etag="two"), path, held) == "fetch"
    assert fetch_decision(ObjectEntry(PDF_KEY, 10, etag="one"), path, held) == "skip"
    # An ETag missing on either side is not evidence of anything, so it must
    # not turn a settled corpus into a re-download of all 100k objects.
    assert fetch_decision(ObjectEntry(PDF_KEY, 10, etag=None), path, held) == "skip"
    assert fetch_decision(ObjectEntry(PDF_KEY, 10, etag="two"), path, {**held, "etag": None}) == "skip"
    assert fetch_decision(ObjectEntry(PDF_KEY, 10, etag="two"), path, _row(10)) == "skip"


# --------------------------------------------------------------------------
# One object, durably.
# --------------------------------------------------------------------------

def test_download_writes_the_bytes_and_returns_size_and_hash(tmp_path):
    body = b"a judgment pdf, more or less" * 40
    bucket = FakeBucket({PDF_KEY: body})
    dest = tmp_path / "sc" / PDF_KEY
    size, digest = download_object(bucket, ObjectEntry(PDF_KEY, len(body)), dest)
    assert dest.read_bytes() == body
    assert size == len(body)
    assert digest == hashlib.sha256(body).hexdigest()


def test_the_destination_only_appears_after_the_last_byte(tmp_path):
    dest = tmp_path / "sc" / PDF_KEY
    body = b"0123456789" * 4
    seen: list[bool] = []

    class Peeking:
        def fetch(self, key):
            yield body[:20]
            seen.append(dest.exists())
            yield body[20:]

    download_object(Peeking(), ObjectEntry(PDF_KEY, len(body)), dest)
    # Writing straight to `dest` would leave a half file at the path every
    # later run reads as complete. It has to be a rename of a finished file.
    assert seen == [False]
    assert dest.read_bytes() == body


def test_the_bytes_are_fsynced_before_the_rename_that_publishes_them(tmp_path, monkeypatch):
    # The whole durability argument rests on this primitive: os.replace is
    # what makes the file visible at the path later runs read as complete, so
    # a rename that reaches the directory before the data reaches the platter
    # can survive a power loss as a present-but-empty judgment.
    order: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        order.append("fsync")
        return real_fsync(fd)

    def replace(src, dst):
        order.append("replace")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)

    body = b"a judgment" * 8
    download_object(FakeBucket({PDF_KEY: body}), ObjectEntry(PDF_KEY, len(body)), tmp_path / PDF_KEY)
    assert order == ["fsync", "replace"]


def test_an_interrupted_transfer_leaves_nothing_behind(tmp_path):
    body = b"z" * 64
    bucket = FakeBucket({PDF_KEY: body}, fail_keys=[PDF_KEY])
    dest = tmp_path / "sc" / PDF_KEY
    with pytest.raises(OSError):
        download_object(bucket, ObjectEntry(PDF_KEY, len(body)), dest)
    assert not dest.exists()
    assert not dest.with_name(dest.name + PART_SUFFIX).exists()


def test_a_body_that_ends_early_is_never_adopted_as_complete(tmp_path):
    body = b"z" * 64
    bucket = FakeBucket({PDF_KEY: body}, truncate_keys={PDF_KEY: 24})
    dest = tmp_path / "sc" / PDF_KEY
    with pytest.raises(AcquisitionError) as exc:
        download_object(bucket, ObjectEntry(PDF_KEY, len(body)), dest)
    assert PDF_KEY in str(exc.value)
    assert "24" in str(exc.value) and "64" in str(exc.value)
    assert not dest.exists()
    assert not dest.with_name(dest.name + PART_SUFFIX).exists()


# --------------------------------------------------------------------------
# A whole acquisition.
# --------------------------------------------------------------------------

def _acquire(store, bucket, root, objects=None, **kwargs):
    entries = [ObjectEntry(k, len(v), f"etag-{k}") for k, v in sorted((objects or {}).items())]
    return acquire_objects(store, bucket, entries, root=root, **kwargs)


def test_an_acquisition_lands_the_bytes_then_indexes_them(store, tmp_path):
    objects = _objects(3)
    bucket = FakeBucket(objects)
    stats = _acquire(store, bucket, tmp_path / "sc", objects)

    assert stats["fetched"] == 3
    assert stats["skipped"] == 0
    assert stats["bytes"] == sum(len(v) for v in objects.values())
    for key, body in objects.items():
        path = tmp_path / "sc" / key
        assert path.read_bytes() == body
        row = store.artifact(SC_SOURCE_ID, key)
        assert row["sha256"] == hashlib.sha256(body).hexdigest()
        assert row["size_bytes"] == len(body)
        assert row["etag"] == f"etag-{key}"
        assert Path(row["local_path"]) == path

    src = store.conn.execute(
        "SELECT license, url FROM source WHERE source_id = ?", (SC_SOURCE_ID,)
    ).fetchone()
    assert src["license"] == SC_LICENSE == "CC-BY-4.0"
    assert SC_BUCKET in src["url"]


def test_re_running_an_acquisition_downloads_nothing(store, tmp_path):
    objects = _objects(3)
    _acquire(store, FakeBucket(objects), tmp_path / "sc", objects)

    stats = _acquire(store, RefusesToFetch(objects), tmp_path / "sc", objects)
    assert stats["skipped"] == 3
    assert stats["fetched"] == 0
    assert stats["adopted"] == 0
    assert stats["bytes"] == 0
    assert store.artifact_count(SC_SOURCE_ID) == 3


def test_a_crash_between_the_bytes_and_the_index_row_costs_no_download(store, tmp_path):
    objects = _objects(3)
    root = tmp_path / "sc"
    # First run dies on the second index write, with the second object's
    # bytes already renamed into place.
    with pytest.raises(RuntimeError, match="index write died"):
        _acquire(IndexFailsAt(store, at=2), FakeBucket(objects), root, objects)
    assert store.artifact_count(SC_SOURCE_ID) == 1
    assert sum(1 for k in objects if (root / k).exists()) == 2

    # Second run adopts what is already durable and downloads only the rest.
    bucket = FakeBucket(objects)
    stats = _acquire(store, bucket, root, objects)
    assert (stats["skipped"], stats["adopted"], stats["fetched"]) == (1, 1, 1)
    assert bucket.fetched == [sorted(objects)[2]]
    for key, body in objects.items():
        assert store.artifact(SC_SOURCE_ID, key)["sha256"] == hashlib.sha256(body).hexdigest()


def test_one_bad_object_does_not_end_the_run(store, tmp_path):
    objects = _objects(3)
    bad = sorted(objects)[1]
    bucket = FakeBucket(objects, truncate_keys={bad: 3})
    stats = _acquire(store, bucket, tmp_path / "sc", objects)

    assert stats["fetched"] == 2
    assert stats["failed"] == 1
    assert [f["key"] for f in stats["failures"]] == [bad]
    assert store.artifact(SC_SOURCE_ID, bad) is None
    assert store.artifact_count(SC_SOURCE_ID) == 2
    events = store.events("acquire_failed")
    assert len(events) == 1
    assert bad in events[0]["detail_json"]


def test_a_run_that_is_failing_wholesale_stops_early(store, tmp_path):
    objects = _objects(6)
    bucket = FakeBucket(objects, truncate_keys={k: 1 for k in objects})
    with pytest.raises(AcquisitionError, match="2 failures"):
        _acquire(store, bucket, tmp_path / "sc", objects, max_failures=2)
    # Stopped at the cap rather than walking all six.
    assert len(bucket.fetched) == 2


def test_a_dry_run_touches_neither_the_disk_nor_the_index(store, tmp_path):
    objects = _objects(3)
    bucket = FakeBucket(objects)
    stats = _acquire(store, bucket, tmp_path / "sc", objects, dry_run=True)
    assert stats["dry_run"] is True
    assert stats["fetched"] == 3  # what a real run would have done
    assert bucket.fetched == []
    assert store.artifact_count(SC_SOURCE_ID) == 0
    assert not (tmp_path / "sc").exists()


def test_limit_stops_after_n_objects(store, tmp_path):
    objects = _objects(5)
    stats = _acquire(store, FakeBucket(objects), tmp_path / "sc", objects, limit=2)
    assert stats["considered"] == 2
    assert store.artifact_count(SC_SOURCE_ID) == 2


def test_limit_counts_work_done_so_a_resumed_run_advances(store, tmp_path):
    # Counting objects EXAMINED would spend the whole cap re-deciding what is
    # already local: `--limit 1000` on a resumed run would then walk the same
    # first 1000 keys forever and never fetch anything.
    objects = _objects(5)
    root = tmp_path / "sc"
    first = _acquire(store, FakeBucket(objects), root, objects, limit=2)
    assert first["fetched"] == 2

    bucket = FakeBucket(objects)
    second = _acquire(store, bucket, root, objects, limit=2)
    assert (second["skipped"], second["fetched"]) == (2, 2)
    # And it advanced onto the NEXT two keys rather than re-deciding the
    # first two for the rest of the corpus's life.
    assert bucket.fetched == sorted(objects)[2:4]
    assert store.artifact_count(SC_SOURCE_ID) == 4


def test_a_failure_spends_the_cap_the_same_way_a_download_does(store, tmp_path):
    # `--limit` caps WORK, and a failure is work: it cost a connection and a
    # slot in the failure list. Counting only the successes would let a run
    # whose objects are all failing walk the entire 100k-key listing under
    # `--limit 10`, which is the opposite of what an operator reaches for a
    # cap to do.
    objects = _objects(4)
    keys = sorted(objects)
    bucket = FakeBucket(objects, fail_keys=keys[:2])
    stats = _acquire(store, bucket, tmp_path / "sc", objects, limit=2)

    assert (stats["failed"], stats["fetched"], stats["adopted"]) == (2, 0, 0)
    assert stats["considered"] == 2
    # The two good keys behind the failures were never reached, so the cap
    # cannot be read as "N successes".
    assert bucket.fetched == keys[:2]
    assert store.artifact_count(SC_SOURCE_ID) == 0


def test_a_re_upload_is_recorded_as_changed_on_a_plain_run_not_only_under_verify(store, tmp_path):
    # `--verify` used to be the only way into the changed-hash branch, and the
    # comment there still said so. The ETag decision routes a genuine
    # re-upload - same length, new bytes, new ETag - through a PLAIN run's
    # fetch, and that run has to record it: this event is the provenance trail
    # for an object whose content moved under a key the corpus already cites.
    root = tmp_path / "sc"
    original = {PDF_KEY: b"y" * 40}
    _acquire(store, FakeBucket(original), root, original)
    was = store.artifact(SC_SOURCE_ID, PDF_KEY)["sha256"]

    replaced = {PDF_KEY: b"n" * 40}
    entries = [ObjectEntry(PDF_KEY, 40, "etag-after-the-re-upload")]
    stats = acquire_objects(store, FakeBucket(replaced), entries, root=root)

    assert (stats["fetched"], stats["skipped"], stats["changed"]) == (1, 0, 1)
    events = store.events("artifact_hash_changed")
    assert len(events) == 1
    assert was in events[0]["detail_json"]
    assert store.artifact(SC_SOURCE_ID, PDF_KEY)["sha256"] == hashlib.sha256(b"n" * 40).hexdigest()


def test_verify_notices_a_local_file_that_changed_under_us(store, tmp_path):
    objects = {PDF_KEY: b"y" * 40}
    root = tmp_path / "sc"
    _acquire(store, FakeBucket(objects), root, objects)
    original = store.artifact(SC_SOURCE_ID, PDF_KEY)["sha256"]

    # Same length, different bytes: only a hash can see this.
    (root / PDF_KEY).write_bytes(b"n" * 40)
    plain = _acquire(store, RefusesToFetch(objects), root, objects)
    assert (plain["skipped"], plain["changed"]) == (1, 0)
    assert store.artifact(SC_SOURCE_ID, PDF_KEY)["sha256"] == original

    checked = _acquire(store, RefusesToFetch(objects), root, objects, verify=True)
    assert checked["changed"] == 1
    assert store.artifact(SC_SOURCE_ID, PDF_KEY)["sha256"] == hashlib.sha256(b"n" * 40).hexdigest()
    assert len(store.events("artifact_hash_changed")) == 1


# --------------------------------------------------------------------------
# Listing.
# --------------------------------------------------------------------------

def test_entries_for_walks_exactly_the_year_prefixes_asked_for():
    # MORE THAN ONE YEAR on purpose: the dev scope is a 16-year walk, and a
    # single-year tuple cannot tell a full walk from one that stops after the
    # first prefix.
    objects = {**_objects(2, year=2015), **_objects(2, year=2016), **_objects(2, year=2009)}
    bucket = FakeBucket(objects)
    keys = [e.key for e in entries_for(bucket, "pdf", (2015, 2016))]
    assert bucket.prefixes == ["data/pdf/year=2015/english/", "data/pdf/year=2016/english/"]
    assert len(keys) == 4
    assert sorted({parse_year(k) for k in keys}) == [2015, 2016]


@pytest.mark.parametrize(
    ("kind", "payload", "noise"),
    [
        (
            "metadata",
            "metadata/parquet/year=2015/part-0.parquet",
            ("metadata/parquet/year=2015/_SUCCESS", "metadata/parquet/year=2015/manifest.json"),
        ),
        (
            "pdf",
            "data/pdf/year=2015/english/2015_1_1_20_EN.pdf",
            (
                "data/pdf/year=2015/english/index.json",
                "data/pdf/year=2015/english/2015_1_1_20_EN.pdf.md5",
            ),
        ),
    ],
)
def test_entries_for_drops_objects_that_are_not_this_kinds_payload(kind, payload, noise):
    # BOTH kinds: listings carry markers, manifests and checksums, and
    # downloading those would index bytes no later stage can read.
    objects = {payload: b"payload", **{key: b"noise" for key in noise}}
    entries = list(entries_for(FakeBucket(objects), kind, (2015,)))
    assert [e.key for e in entries] == [payload]


# --------------------------------------------------------------------------
# HuggingFace snapshots.
# --------------------------------------------------------------------------

def test_the_snapshot_registry_is_exactly_these_six_repos():
    """The literal pin that was lost when a count became len(HF_SOURCES).

    That change was right for what it was doing - the left side there is DB
    event rows from the production loop - but it stopped pinning the registry
    itself: dropping a source, or a typo in one of the three eval repo ids,
    passed the whole suite. decontaminate.py REFUSES to run without the eval
    sets and prints the repo id as the thing to fix, so a silent edit to one
    of them is exactly the wrong gap to leave open.

    THE THREE EVAL IDS WERE VERIFIED AGAINST THE HUB ON 2026-08-14 and this
    pin holds the verified spellings, so an edit to one of them is a decision
    and not a typo-fix. `opennyaiorg/aibe` was a 404 at that check - the real
    id is `opennyaiorg/aibe_dataset` - which is exactly the failure this pin
    exists to make loud.
    """
    assert {key: source.repo_id for key, source in HF_SOURCES.items()} == {
        "predex": "L-NLProc/PredEx_Instruction-Tuning_Pred-Exp",
        "tathyanyaya": "L-NLProc/TathyaNyaya-and-FactLegalLlama-NyayaFacts-Datasets",
        "injudgements": "opennyaiorg/InJudgements_dataset",
        "bbl": "bharatgenai/BhashaBench-Legal",
        "iltur": "Exploration-Lab/IL-TUR",
        "aibe": "opennyaiorg/aibe_dataset",
    }
    # Every key is its own entry's key, so a copy-paste in the registry cannot
    # leave two entries pointing at one source.
    assert all(key == source.key for key, source in HF_SOURCES.items())
    assert len({s.source_id for s in HF_SOURCES.values()}) == len(HF_SOURCES)


def test_every_eval_set_decontamination_refuses_without_is_a_registered_snapshot():
    """The join between the two registries: decontaminate.py names the set,
    acquire.py owns where it comes from, and the refusal it prints is an
    `acquire --hf-source KEY` command that has to exist."""
    from tuned.data.decontaminate import EVAL_SETS

    assert set(EVAL_SETS) <= set(HF_SOURCES)
    assert {EVAL_SETS[key].repo_id for key in EVAL_SETS} == {
        "bharatgenai/BhashaBench-Legal", "Exploration-Lab/IL-TUR", "opennyaiorg/aibe_dataset",
    }

FAKE_SOURCE = HfSource(
    key="fake",
    repo_id="somebody/a-gated-corpus",
    license="Apache-2.0",
    gated=True,
)


class GatedRepoError(Exception):
    """Named exactly as huggingface_hub names it - the classifier is
    duck-typed, so the real class never has to be imported to test this."""


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code


class FakeHttpError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.response = _Response(status_code)


def test_is_gated_error_reads_both_shapes_huggingface_uses():
    assert is_gated_error(GatedRepoError("gated"))
    assert is_gated_error(FakeHttpError(401))
    assert is_gated_error(FakeHttpError(403))
    assert not is_gated_error(FakeHttpError(404))
    assert not is_gated_error(FakeHttpError(500))
    assert not is_gated_error(ConnectionError("dns"))


def test_a_gated_dataset_fails_with_the_page_to_open(store, tmp_path):
    def snapshot(**kwargs):
        raise GatedRepoError("401 Client Error")

    with pytest.raises(GatedSourceError) as exc:
        acquire_hf(store, FAKE_SOURCE, root=tmp_path / "hf", snapshot_fn=snapshot)

    message = str(exc.value)
    # The fabricated source is what makes this non-circular: none of these
    # strings exist anywhere in acquire.py.
    assert FAKE_SOURCE.repo_id in message
    assert FAKE_SOURCE.url in message
    assert "HF_TOKEN" in message
    assert store.artifact_count() == 0


def test_a_network_failure_is_not_dressed_up_as_a_permissions_problem(store, tmp_path):
    def snapshot(**kwargs):
        raise ConnectionError("temporary failure in name resolution")

    with pytest.raises(ConnectionError):
        acquire_hf(store, FAKE_SOURCE, root=tmp_path / "hf", snapshot_fn=snapshot)


def _write_snapshot(local_dir, files: dict) -> str:
    for name, body in files.items():
        path = Path(local_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    return str(local_dir)


def test_a_snapshot_is_indexed_by_what_actually_landed(store, tmp_path):
    files = {
        "README.md": b"# card",
        "data/train-0.parquet": b"rows" * 10,
        ".cache/huggingface/download/train-0.metadata": b"bookkeeping",
    }

    def snapshot(*, repo_id, local_dir, **kwargs):
        assert repo_id == FAKE_SOURCE.repo_id
        return _write_snapshot(local_dir, files)

    stats = acquire_hf(store, FAKE_SOURCE, root=tmp_path / "hf", snapshot_fn=snapshot)

    # The hub's own bookkeeping under .cache/ is not corpus data.
    assert stats["indexed"] == 2
    index = store.artifact_index(FAKE_SOURCE.source_id)
    assert sorted(index) == ["README.md", "data/train-0.parquet"]
    assert index["data/train-0.parquet"]["sha256"] == hashlib.sha256(b"rows" * 10).hexdigest()
    src = store.conn.execute(
        "SELECT license FROM source WHERE source_id = ?", (FAKE_SOURCE.source_id,)
    ).fetchone()
    assert src["license"] == "Apache-2.0"


def test_re_indexing_a_snapshot_adds_nothing_but_a_changed_file_is_re_read(store, tmp_path):
    files = {"data/train-0.parquet": b"rows"}
    root = tmp_path / "hf"

    def snapshot(*, repo_id, local_dir, **kwargs):
        return _write_snapshot(local_dir, files)

    acquire_hf(store, FAKE_SOURCE, root=root, snapshot_fn=snapshot)
    again = acquire_hf(store, FAKE_SOURCE, root=root, snapshot_fn=snapshot)
    assert (again["indexed"], again["skipped"]) == (0, 1)

    files["data/train-0.parquet"] = b"rows and more rows"
    grown = acquire_hf(store, FAKE_SOURCE, root=root, snapshot_fn=snapshot)
    assert grown["indexed"] == 1
    row = store.artifact(FAKE_SOURCE.source_id, "data/train-0.parquet")
    assert row["sha256"] == hashlib.sha256(b"rows and more rows").hexdigest()


def test_the_gated_sources_are_the_ones_the_operator_queue_names():
    """All four gated repos, not just the one. The three eval sets report
    gated="auto" on the Hub - the same situation injudgements is in - and
    carried `gated=False` beside a comment saying so, which left the three sets
    that BLOCK decontamination out of the list of access grants to make."""
    gated = {key for key, src in HF_SOURCES.items() if src.gated}
    assert gated == {"injudgements", "bbl", "iltur", "aibe"}
    assert HF_SOURCES["injudgements"].repo_id == "opennyaiorg/InJudgements_dataset"


def test_acquiring_injudgements_reuses_the_source_row_seeds_py_writes(store, tmp_path):
    # One dataset, one source row: seeds.py registers the same id when it
    # normalises the rows, and two spellings would fork the provenance.
    from tuned.data.seeds import INJUDGEMENTS_SOURCE_ID

    store.upsert_source(INJUDGEMENTS_SOURCE_ID, "Apache-2.0")

    def snapshot(*, repo_id, local_dir, **kwargs):
        return _write_snapshot(local_dir, {"train.parquet": b"x"})

    acquire_hf(store, HF_SOURCES["injudgements"], root=tmp_path / "hf", snapshot_fn=snapshot)
    assert store.conn.execute("SELECT COUNT(*) FROM source").fetchone()[0] == 1


def test_index_tree_keys_are_posix_relative_paths_and_skip_partials(store, tmp_path):
    store.upsert_source("src", "Apache-2.0")
    tree = tmp_path / "tree"
    (tree / "a" / "b").mkdir(parents=True)
    (tree / "a" / "b" / "c.parquet").write_bytes(b"x")
    (tree / ("leftover.parquet" + PART_SUFFIX)).write_bytes(b"y")
    stats = index_tree(store, tree, source_id="src")
    # Keys are "/"-joined on every platform, and a half-written .part file is
    # not corpus data.
    assert sorted(store.artifact_index("src")) == ["a/b/c.parquet"]
    assert stats["indexed"] == 1


# --------------------------------------------------------------------------
# Seams and CLI.
# --------------------------------------------------------------------------

def test_module_import_never_touches_the_heavy_clients():
    tree = ast.parse(ACQUIRE_SRC.read_text(encoding="utf-8"))
    banned = {"boto3", "botocore", "huggingface_hub", "datasets", "pyarrow"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = {alias.name.split(".")[0] for alias in node.names}
            assert not (names & banned), f"top-level import of {names & banned}"
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in banned, f"top-level from {node.module}"


def test_a_missing_boto3_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", None)
    with pytest.raises(AcquisitionError) as exc:
        list(S3Bucket().list_objects("data/pdf/"))
    assert "boto3" in str(exc.value)
    assert "[build]" in str(exc.value)


def test_cli_hard_exits_after_success():
    assert "os._exit(" in ACQUIRE_SRC.read_text(encoding="utf-8")


def test_cli_acquires_pdfs_into_the_build_corpus(tmp_path, capsys):
    from tuned.data.config import load_build_config
    from tuned.data.paths import build_paths

    config = temp_config(tmp_path)
    objects = _objects(2)
    bucket = FakeBucket(objects)

    assert main(["--config", config, "--kind", "pdf", "--years", "2015"], fetcher=bucket) == 0

    cfg = load_build_config(config, allow_unpinned=True)
    paths = build_paths(cfg.build.workdir)
    for key, body in objects.items():
        assert (paths.corpus_dir / "sc" / key).read_bytes() == body
    # A count, not the word "fetched": the header prints that either way.
    assert "artifacts indexed -> 2" in capsys.readouterr().out

    with Store.open(paths.state_db) as opened:
        assert opened.artifact_count(SC_SOURCE_ID) == 2


def test_a_dry_run_does_not_report_itself_as_a_clean_run(tmp_path, capsys):
    """`--dry-run` acquired NOTHING and printed "no failures" and exited 0 -
    byte-identical to a real run in which everything landed, on the one command
    whose whole point is that it did not touch anything."""
    config = temp_config(tmp_path)

    def snapshot(**kwargs):
        raise AssertionError("a dry run must not call the snapshot seam")

    code = main(["--config", config, "--kind", "hf", "--dry-run"], snapshot_fn=snapshot)
    out = capsys.readouterr().out
    assert code == 0
    assert "no failures" not in out
    assert "DRY RUN: nothing was downloaded, indexed or checked" in out
    for source in HF_SOURCES.values():
        assert f"would snapshot {source.repo_id}" in out


def test_cli_reports_a_gated_dataset_and_exits_nonzero(tmp_path, capsys):
    def snapshot(**kwargs):
        raise GatedRepoError("403")

    code = main(
        ["--config", temp_config(tmp_path), "--kind", "hf", "--hf-source", "injudgements"],
        snapshot_fn=snapshot,
    )
    out = capsys.readouterr().out
    assert code == 2
    assert HF_SOURCES["injudgements"].url in out


def test_a_run_that_lost_sources_is_not_reported_as_ordinary_first_run_gating(tmp_path, capsys):
    """A wrong repo id fails exactly like a hub 5xx. It used to be
    indistinguishable from the BENIGN case: `code = 2`
    was an assignment rather than a max, gating sorted after the failures, and
    the summary line mentioned no failures at all. Three FAILED lines scrolled
    above a six-line access-grant block under an exit code that means 'go and
    accept some terms'."""
    config = temp_config(tmp_path)

    def snapshot(*, repo_id, **kwargs):
        if repo_id == HF_SOURCES["injudgements"].repo_id:
            raise GatedRepoError("403")
        raise ConnectionError("Repository Not Found for url")

    code = main(["--config", config, "--kind", "hf"], snapshot_fn=snapshot)
    out = capsys.readouterr().out
    assert code == 1, "a lost source outranks a gate: the remedies are different"
    assert f"FAILED ({len(HF_SOURCES) - 1})" in out
    assert "GATED (1): injudgements" in out
    # And it no longer tells the operator the eval ids are unchecked: they
    # were checked, by the same commit that added that line, and pointing at
    # the one thing that had been verified is worse than saying nothing.
    assert "never been checked against the Hub" not in out
    assert f"checked against the Hub on {HF_IDS_VERIFIED_AT} and all six resolved" in out
    # The summary is BELOW the access-grant block, which is what stops it
    # scrolling away.
    assert out.index(f"FAILED ({len(HF_SOURCES) - 1})") > out.index("Agree and access repository")
    # Every lost source is named, not just counted.
    for key in set(HF_SOURCES) - {"injudgements"}:
        assert HF_SOURCES[key].repo_id in out


def test_a_run_with_nothing_but_a_gate_still_exits_two(tmp_path, capsys):
    """The other side: gating alone keeps its own exit code and says so."""
    config = temp_config(tmp_path)

    def snapshot(*, repo_id, local_dir, **kwargs):
        if repo_id == HF_SOURCES["injudgements"].repo_id:
            raise GatedRepoError("403")
        return _write_snapshot(local_dir, {"data/train-0.parquet": b"rows"})

    code = main(["--config", config, "--kind", "hf"], snapshot_fn=snapshot)
    out = capsys.readouterr().out
    assert code == 2
    assert "GATED (1): injudgements" in out
    assert "FAILED" not in out


def test_a_clean_run_says_there_were_no_failures(tmp_path, capsys):
    """The summary reads on a clean run too: an operator who has learned to
    look for it must not have to infer its absence."""
    def snapshot(*, local_dir, **kwargs):
        return _write_snapshot(local_dir, {"data/train-0.parquet": b"rows"})

    code = main(["--config", temp_config(tmp_path), "--kind", "hf"], snapshot_fn=snapshot)
    out = capsys.readouterr().out
    assert code == 0
    assert "no failures" in out


def _corpus_paths(config):
    from tuned.data.config import load_build_config
    from tuned.data.paths import build_paths

    return build_paths(load_build_config(config, allow_unpinned=True).build.workdir)


def test_a_gated_dataset_does_not_cost_the_days_long_pdf_pull(tmp_path, capsys):
    # InJudgements is gated RIGHT NOW, so `--kind all` - the documented
    # default command - hits this on the first real run. The whole point of
    # exit 2 rather than a raise is that the PDFs are still pulled.
    config = temp_config(tmp_path)
    objects = _objects(2)

    def snapshot(**kwargs):
        raise GatedRepoError("403")

    code = main(
        ["--config", config, "--kind", "all", "--years", "2015"],
        fetcher=FakeBucket(objects),
        snapshot_fn=snapshot,
    )
    out = capsys.readouterr().out
    assert code == 2
    assert "GATED" in out
    paths = _corpus_paths(config)
    for key, body in objects.items():
        assert (paths.corpus_dir / "sc" / key).read_bytes() == body


def test_a_transient_hub_failure_does_not_cost_the_days_long_pdf_pull_either(tmp_path, capsys):
    # Same event class as a gated set, demoted the same way: a 5xx from the
    # hub after the metadata already landed must not propagate out of main
    # and leave the PDF pull unstarted. The exit code still says the run was
    # not clean.
    config = temp_config(tmp_path)
    objects = _objects(2)

    def snapshot(**kwargs):
        raise ConnectionError("temporary failure in name resolution")

    code = main(
        ["--config", config, "--kind", "all", "--years", "2015"],
        fetcher=FakeBucket(objects),
        snapshot_fn=snapshot,
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "ConnectionError" in out
    paths = _corpus_paths(config)
    for key, body in objects.items():
        assert (paths.corpus_dir / "sc" / key).read_bytes() == body
    with Store.open(paths.state_db) as opened:
        # One per registered snapshot, read off the registry rather than
        # frozen as a literal: the eval corpora decontaminate.py refuses to
        # run without are snapshots too, and this test is about "every one of
        # them was counted", not about how many there happen to be.
        assert len(opened.events("acquire_failed")) == len(HF_SOURCES)


def test_a_run_that_lost_objects_does_not_report_success(tmp_path, capsys):
    # A zero exit from a run that dropped judgments would clear a shell `&&`
    # chain and start extraction on a corpus with holes in it.
    config = temp_config(tmp_path)
    objects = _objects(3)
    bad = sorted(objects)[1]
    bucket = FakeBucket(objects, truncate_keys={bad: 3})

    code = main(["--config", config, "--kind", "pdf", "--years", "2015"], fetcher=bucket)
    assert code == 1
    assert bad in capsys.readouterr().out


def test_every_lost_object_is_named_not_just_the_first_ten(tmp_path, capsys):
    # --max-failures defaults to 25, so truncating the printed list at 10
    # hides failures of a run that never reached the cap.
    config = temp_config(tmp_path)
    objects = _objects(12)
    bucket = FakeBucket(objects, truncate_keys={k: 1 for k in objects})

    assert main(["--config", config, "--kind", "pdf", "--years", "2015"], fetcher=bucket) == 1
    out = capsys.readouterr().out
    assert [key for key in sorted(objects) if key not in out] == []


def test_a_verify_pass_does_not_call_re_reading_the_corpus_an_adoption(tmp_path, capsys):
    # Under --verify every object the index agrees with takes the adopt path,
    # so the summary would read "adopted 100000 skipped 0" on a corpus where
    # nothing whatever was adopted.
    config = temp_config(tmp_path)
    objects = _objects(2)
    main(["--config", config, "--kind", "pdf", "--years", "2015"], fetcher=FakeBucket(objects))
    capsys.readouterr()

    main(
        ["--config", config, "--kind", "pdf", "--years", "2015", "--verify"],
        fetcher=RefusesToFetch(objects),
    )
    out = capsys.readouterr().out
    assert "re-hashed" in out
    assert "adopted" not in out


# --------------------------------------------------------------------------
# rebase_under_corpus: an absolute local_path outlives the checkout it names.
# --------------------------------------------------------------------------

def test_rebase_recovers_an_eval_set_from_a_deleted_worktree(tmp_path):
    """The real 2026-08-29 failure: every hf eval set was indexed inside
    a linked worktree, the restructure deleted it, and
    decontaminate refused to run because it could not read a set it is
    measured against. The files never moved relative to `corpus`."""
    from tuned.data.acquire import rebase_under_corpus

    corpus = tmp_path / "data" / "build" / "corpus"
    key = "data/train-00000-of-00001.parquet"
    landed = corpus / "hf" / "aibe" / key
    landed.parent.mkdir(parents=True)
    landed.write_bytes(b"parquet")

    recorded = (
        r"C:\old-checkout\worktrees\law-v1-data-pipeline"
        r"\data\build\corpus\hf\aibe\data"
        r"\train-00000-of-00001.parquet"
    )
    assert rebase_under_corpus(recorded, key, corpus) == landed
    # The sub-root under corpus/ is preserved, not guessed: an sc/ object
    # must not be re-rooted into hf/.
    sc_key = "metadata/year=2018/x.parquet"
    sc_recorded = Path("/kaggle/working/data/build/corpus/sc") / sc_key
    assert rebase_under_corpus(sc_recorded, sc_key, corpus) == corpus / "sc" / sc_key


def test_rebase_leaves_a_live_path_alone_and_never_invents_one(tmp_path):
    from tuned.data.acquire import rebase_under_corpus

    corpus = tmp_path / "corpus"
    live = corpus / "hf" / "bbl" / "a.parquet"
    live.parent.mkdir(parents=True)
    live.write_bytes(b"x")
    # An existing path is returned untouched - no re-rooting, no stat games.
    assert rebase_under_corpus(live, "a.parquet", corpus) == live
    # A path with no `corpus` component cannot be re-rooted: return it
    # unchanged so the caller reports the same missing file it would have.
    orphan = tmp_path / "somewhere" / "else" / "a.parquet"
    assert rebase_under_corpus(orphan, "a.parquet", corpus) == orphan
