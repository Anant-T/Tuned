"""Corpus intake: get the raw material onto disk, resumably.

Two sources, no credentials on the S3 side and no teacher-model spend on
either: the public AWS Open Data bucket of Supreme Court judgments
(`s3://indian-supreme-court-judgments/` - metadata parquet under
`metadata/parquet/year=YYYY/`, PDFs under
`data/pdf/year=YYYY/{english,regional}/`, both walked in P0), and
HuggingFace snapshots of the dataset-shaped sources the corpus phase reads
more than once.

RESUMABILITY IS THE DESIGN, not a feature
-----------------------------------------
This runs for hours over ~100k small objects and WILL be interrupted, so
every object goes through the same three-outcome decision (fetch_decision):

    fetch   nothing usable is on disk - download it
    adopt   the bytes are already durable but no index row points at them
    skip    disk and index agree with the listing - do nothing

and each download follows the house durability rule the generation path
uses - bytes durable BEFORE the row claiming they exist:

  * the body is streamed to `<dest>.part`, fsynced, and only then renamed
    onto `dest`, so a killed process never leaves a half file at the path a
    later run reads as complete;
  * a body that ends early (a clean EOF short of the listed size - the
    failure that raises nothing) is discarded, not adopted;
  * `record_artifact` runs after the rename, so a crash in that window
    costs an index row and no download: the next run sees complete bytes
    with no row, hashes them and adopts. That is the same SHAPE as
    store.reconcile_raw rebuilding generation rows from the raw logs, but
    NOT the same cost - reconcile_raw rebuilds from local raw logs alone,
    while adopting needs a live bucket listing to know which keys to look
    for. Losing the database therefore costs a full re-listing plus a
    re-hash of ~15 GB, which is why the index is worth keeping.

Size is what the resume decision compares, because hashing ~15 GB of PDFs
on every restart would cost more than the sync; `--verify` re-hashes and
reports (run_event artifact_hash_changed) a file that changed under us
without changing length. An ETag that differs from the recorded one is the
one free exception - see fetch_decision.

Gated HuggingFace datasets are an ACCESS GRANT the operator has to make on
the dataset page, and they fail with that instruction rather than a
traceback.

boto3/huggingface_hub are imported lazily inside the seams, never at module
import time, matching replay.py/seeds.py.

Build:  python -m tuned.data.acquire --config configs/data_law_v1.yaml
        [--kind metadata|pdf|hf|all] [--years 2010-2025] [--language english]
        [--limit N] [--verify] [--dry-run] [--hf-source injudgements]
"""

import hashlib
import os
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from tuned.data.seeds import (
    INJUDGEMENTS_SOURCE_ID,
    PREDEX_SOURCE_ID,
    TATHYANYAYA_SOURCE_ID,
)

# --------------------------------------------------------------------------
# The bucket, and the slice of it v1 wants.
# --------------------------------------------------------------------------

SC_BUCKET = "indian-supreme-court-judgments"
SC_REGION = "ap-south-1"
SC_SOURCE_ID = f"s3://{SC_BUCKET}"
SC_URL = f"s3://{SC_BUCKET}/"
# Indian judgments are not copyrightable (s.52(1)(q)(iv)); the bucket
# publishes under CC-BY-4.0 and the dataset card must carry the attribution.
SC_LICENSE = "CC-BY-4.0"

# 2010-2025 ONLY. Those years are ~100% born-digital and single-column, so
# v1 needs no OCR; pre-1990 is scans (deferred, +4-6 days) and 2026 is a
# partial year the bucket is still filling.
DEV_YEARS = tuple(range(2010, 2026))

ENGLISH = "english"
REGIONAL = "regional"

# prefix template -> the suffix that is this kind's actual payload. Listings
# also carry markers and manifests; downloading those would index bytes no
# later stage can read.
_KINDS = {
    "pdf": ("data/pdf/year={year}/{language}/", ".pdf"),
    "metadata": ("metadata/parquet/year={year}/", ".parquet"),
}

PART_SUFFIX = ".part"
_READ_BLOCK = 1 << 20

# Enough failures in a row and the problem is the network or the bucket, not
# the object: stop with the count rather than log 100k identical errors.
DEFAULT_MAX_FAILURES = 25

_YEAR_RE = re.compile(r"(?:^|/)year=(\d{4})(?:/|$)")

# A key that is absolute, drive-qualified, backslashed or carries a ".."
# segment would put this process's writes somewhere the operator did not
# choose. Keys come off a remote listing, so this is checked, not assumed.
_UNSAFE_KEY = re.compile(r"^/|^[A-Za-z]:|(?:^|/)\.\.(?:/|$)|\\")


class AcquisitionError(RuntimeError):
    """Something upstream could not be acquired. Actionable by construction:
    the message says what to do, not what raised."""


class GatedSourceError(AcquisitionError):
    """The dataset needs an access grant this account has not been given."""


@dataclass(frozen=True)
class ObjectEntry:
    """One object as the listing describes it. `size` is required and is the
    resume decision's whole basis, so a listing that cannot report it is not
    usable here."""

    key: str
    size: int
    etag: str | None = None


@dataclass(frozen=True)
class HfSource:
    key: str
    repo_id: str
    license: str
    gated: bool
    allow_patterns: tuple[str, ...] | None = None

    @property
    def url(self) -> str:
        return f"https://huggingface.co/datasets/{self.repo_id}"

    @property
    def source_id(self) -> str:
        # The repo id IS the source id, so a snapshot and the seed rows
        # normalised out of it share one `source` row rather than forking
        # the provenance in two spellings.
        return self.repo_id


# Snapshotted because the corpus phase reads them more than once and they
# are small. The bulk replay sources (WildChat-4.8M, OpenThoughts-114k,
# smoltalk2, Nemotron-v2) are deliberately NOT here: replay.py streams them
# once, and mirroring terabytes to re-read 300 rows would be the most
# expensive no-op in the build.
HF_SOURCES = {
    "predex": HfSource(
        key="predex", repo_id=PREDEX_SOURCE_ID, license="Apache-2.0", gated=False
    ),
    "tathyanyaya": HfSource(
        key="tathyanyaya", repo_id=TATHYANYAYA_SOURCE_ID, license="Apache-2.0", gated=False
    ),
    # Both a full-text source and the pre-computed most-cited landmark list
    # select.py joins against - which is why it is worth having locally
    # rather than streaming twice. gated="auto" on HF.
    "injudgements": HfSource(
        key="injudgements", repo_id=INJUDGEMENTS_SOURCE_ID, license="Apache-2.0", gated=True
    ),
    # The eval corpora. NEVER TRAINED ON - they are here because
    # decontaminate.py screens the dataset against them and REFUSES to run
    # without them, so "is it on disk" has to be an acquire-time fact like
    # every other input. IL-TUR and AIBE are NC/ND and are read-only for that
    # screen; BBL is the forgetting guard the charter's headline number comes
    # from, which is exactly why a leak of it into training is unrecoverable.
    #
    # THESE THREE REPO IDS HAVE NEVER BEEN CHECKED AGAINST THE HUB - P0 did
    # not enumerate them and this build has no network. A wrong id fails here,
    # loudly, and then fails decontaminate.py as `not_acquired`, which is a
    # refusal. It cannot fail quietly.
    "bbl": HfSource(
        key="bbl", repo_id="bharatgenai/BhashaBench-Legal", license="CC-BY-4.0", gated=False
    ),
    "iltur": HfSource(
        key="iltur", repo_id="Exploration-Lab/IL-TUR",
        license="non-commercial (EVAL/DECONTAMINATION ONLY)", gated=False,
    ),
    "aibe": HfSource(
        key="aibe", repo_id="opennyaiorg/aibe",
        license="no-derivatives (EVAL/DECONTAMINATION ONLY)", gated=False,
    ),
}


# --------------------------------------------------------------------------
# Pure helpers.
# --------------------------------------------------------------------------

def parse_years(spec: str) -> tuple[int, ...]:
    """"2010-2025" / "2015" / "2020,2010-2011" -> the years, in order given."""
    years: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            bounds = part.split("-")
            if len(bounds) != 2:
                raise ValueError(f"--years range must be LO-HI, got {part!r}")
            lo, hi = (int(b) for b in bounds)
            if hi < lo:
                raise ValueError(f"--years range runs backwards: {part!r}")
            years.extend(range(lo, hi + 1))
        else:
            years.append(int(part))
    if not years:
        raise ValueError(f"--years parsed to nothing: {spec!r}")
    return tuple(dict.fromkeys(years))


def _kind(kind: str) -> tuple[str, str]:
    try:
        return _KINDS[kind]
    except KeyError:
        raise ValueError(f"unknown kind {kind!r}, expected one of {sorted(_KINDS)}") from None


def year_prefixes(kind: str, years: Iterable[int], language: str = ENGLISH) -> tuple[str, ...]:
    template, _ = _kind(kind)
    return tuple(template.format(year=year, language=language) for year in years)


def parse_year(key: str) -> int | None:
    """The year partition a key sits in, or None if it is outside one."""
    match = _YEAR_RE.search(key)
    return int(match.group(1)) if match else None


def local_path_for(root: str | Path, key: str) -> Path:
    """Where `key` lives locally: the bucket layout, mirrored under `root`.

    Keeping the remote key as the local path is what makes the index
    (source_id, object_key) enough to find a file again after a crash, and
    what lets `parse_year` work on either side.
    """
    root = Path(root)
    if not key or key.endswith("/") or _UNSAFE_KEY.search(key):
        raise ValueError(f"unsafe object key {key!r}: refusing to write outside {root}")
    dest = root / key
    # Belt and braces: the regex is the readable rule, containment is the
    # guarantee.
    if not dest.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"object key {key!r} resolves outside {root}")
    return dest


def sha256_file(path: str | Path, block: int = _READ_BLOCK) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_decision(
    entry: ObjectEntry, path: str | Path, indexed: dict | None, *, verify: bool = False
) -> str:
    """"fetch" | "adopt" | "skip" for one object - the whole resume policy.

    The bytes on disk are the authority and the listing is the reference:
    anything whose length disagrees with the listing is re-fetched and never
    adopted, which is what stops a truncated leftover from ever being read
    as complete. A file that agrees with the listing but has no (or a stale)
    index row is ADOPTED - hashed and recorded, not downloaded again -
    because that is exactly the state a crash between the rename and the row
    leaves behind.

    Length, not hash, because re-hashing the whole corpus on every restart
    would cost more than the sync it is protecting; `verify=True` buys the
    hash back for a run that wants it.

    The ETag is the one exception, and it is free: nothing may verify content
    AGAINST an ETag (a multipart upload's is "<md5>-<parts>", not the body's
    md5), but an ETag that DIFFERS from the one recorded means the object was
    re-uploaded under a key already held - and the SC bucket is a rolling
    release, so that happens. The local bytes are then the old object at the
    same length, which is precisely what a size comparison cannot see.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size != entry.size:
        return "fetch"
    recorded = indexed.get("etag") if indexed is not None else None
    if entry.etag and recorded and entry.etag != recorded:
        # Absent on either side is not evidence of anything: index_tree
        # records no ETag at all, and a run that re-fetched 100k objects
        # because the listing stopped reporting them would be far worse than
        # the one stale object this catches.
        return "fetch"
    if indexed is None or indexed["size_bytes"] != entry.size or verify:
        return "adopt"
    return "skip"


def download_object(fetcher, entry: ObjectEntry, dest: str | Path) -> tuple[int, str]:
    """Stream one object to `dest` atomically; returns (bytes, sha256).

    Written to a sibling `.part` and renamed, so `dest` either does not
    exist or is the whole object - there is no state in which it is a
    prefix of one. A short body raises instead of renaming: a clean EOF
    below the listed size is the one truncation nothing else would notice.
    """
    dest = Path(dest)
    part = dest.with_name(dest.name + PART_SUFFIX)
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    try:
        with part.open("wb") as handle:
            for chunk in fetcher.fetch(entry.key):
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if size != entry.size:
            raise AcquisitionError(
                f"{entry.key}: body ended at {size} bytes but the listing says "
                f"{entry.size} - discarding the partial instead of adopting it as complete"
            )
    except BaseException:
        # A partial must never be adoptable, so it does not survive the
        # failure that produced it.
        part.unlink(missing_ok=True)
        raise
    os.replace(part, dest)
    return size, digest.hexdigest()


def entries_for(
    fetcher, kind: str, years: Iterable[int] = DEV_YEARS, language: str = ENGLISH
) -> Iterator[ObjectEntry]:
    """Every payload object of `kind` in those year partitions."""
    _, suffix = _kind(kind)
    for prefix in year_prefixes(kind, years, language=language):
        for entry in fetcher.list_objects(prefix):
            if entry.key.lower().endswith(suffix):
                yield entry


# --------------------------------------------------------------------------
# Assembly.
# --------------------------------------------------------------------------

def acquire_objects(
    store,
    fetcher,
    entries: Iterable[ObjectEntry],
    *,
    root: str | Path,
    source_id: str = SC_SOURCE_ID,
    license_: str = SC_LICENSE,
    url: str = SC_URL,
    verify: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
    max_failures: int = DEFAULT_MAX_FAILURES,
) -> dict:
    """Acquire `entries` under `root`, indexing each object once it is durable.

    Idempotent: a second run over the same entries downloads nothing.
    Interruptible: whatever landed stays landed, and the index is re-derived
    from the files themselves rather than from anything the dead run held in
    memory.

    A per-object failure is counted and the run continues - one bad object
    out of 100k must not cost the other 99,999 - but `max_failures`
    consecutive-ish failures stop the run, because at that point the fault
    is not in the objects. A failure to write the INDEX is deliberately not
    caught: that is the database, not the network, and continuing past it
    would produce a corpus nothing can find.

    `limit` caps WORK - objects fetched, adopted or failed - and not objects
    examined. Counting the examined ones would spend the whole cap on the
    keys a resumed run skips, so `--limit 1000` would re-decide the same
    first 1000 objects on every restart and never advance.
    """
    root = Path(root)
    stats: dict = {
        "considered": 0,
        "skipped": 0,
        "fetched": 0,
        "adopted": 0,
        "failed": 0,
        "changed": 0,
        "bytes": 0,
        "failures": [],
        "dry_run": dry_run,
    }
    if not dry_run:
        store.upsert_source(source_id, license_, url=url)
    # One read for the whole run: the decision below is taken per key, and a
    # SELECT apiece would make resuming cost more than the sync.
    index = store.artifact_index(source_id)

    for entry in entries:
        if limit is not None and stats["fetched"] + stats["adopted"] + stats["failed"] >= limit:
            break
        stats["considered"] += 1
        try:
            dest = local_path_for(root, entry.key)
            indexed = index.get(entry.key)
            decision = fetch_decision(entry, dest, indexed, verify=verify)
            if decision == "skip":
                stats["skipped"] += 1
                continue
            if dry_run:
                stats["fetched" if decision == "fetch" else "adopted"] += 1
                if decision == "fetch":
                    stats["bytes"] += entry.size
                continue
            if decision == "fetch":
                size, digest = download_object(fetcher, entry, dest)
            else:
                size, digest = dest.stat().st_size, sha256_file(dest)
        except Exception as exc:
            stats["failed"] += 1
            detail = {"key": entry.key, "error": f"{type(exc).__name__}: {exc}"}
            stats["failures"].append(detail)
            store.log_event("acquire_failed", {"source_id": source_id, **detail})
            if stats["failed"] >= max_failures:
                raise AcquisitionError(
                    f"acquire: stopping after {stats['failed']} failures "
                    f"(last {entry.key!r}) - at this rate the fault is upstream, "
                    f"not in the objects"
                ) from exc
            continue

        # The bytes are durable from here; the row may now claim they exist.
        if (
            indexed is not None
            and indexed["size_bytes"] == size
            and indexed["sha256"] != digest
        ):
            # Same length, different content: the local file changed under
            # us, or upstream replaced the object in place. TWO ways in, and
            # only the first needs a flag - --verify, which forces the re-read
            # that can see a local file rotting under an unchanged listing;
            # and a plain run whose ETag decision already said "fetch",
            # because a re-upload at the same length is exactly what an ETag
            # mismatch reports.
            stats["changed"] += 1
            store.log_event(
                "artifact_hash_changed",
                {
                    "source_id": source_id,
                    "key": entry.key,
                    "was": indexed["sha256"],
                    "now": digest,
                    "size_bytes": size,
                },
            )
        store.record_artifact(
            source_id,
            entry.key,
            local_path=dest,
            size_bytes=size,
            sha256=digest,
            etag=entry.etag,
        )
        index[entry.key] = {"size_bytes": size, "sha256": digest, "local_path": str(dest)}
        stats["fetched" if decision == "fetch" else "adopted"] += 1
        if decision == "fetch":
            stats["bytes"] += size
    return stats


def index_tree(store, root: str | Path, *, source_id: str) -> dict:
    """Index every file under `root` as an artifact of `source_id`.

    The adopt path for downloads somebody else performed - huggingface_hub
    does its own atomic-rename resume, so acquire_hf lets it, then records
    what actually landed. Dot-directories are skipped: `.cache/huggingface`
    under a `local_dir` snapshot is the hub's bookkeeping, not corpus data.
    """
    root = Path(root)
    index = store.artifact_index(source_id)
    stats = {"indexed": 0, "skipped": 0, "bytes": 0}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        if any(part.startswith(".") for part in rel.parts) or path.name.endswith(PART_SUFFIX):
            continue
        key = rel.as_posix()
        size = path.stat().st_size
        row = index.get(key)
        if row is not None and row["size_bytes"] == size:
            stats["skipped"] += 1
            continue
        store.record_artifact(
            source_id, key, local_path=path, size_bytes=size, sha256=sha256_file(path)
        )
        stats["indexed"] += 1
        stats["bytes"] += size
    return stats


def is_gated_error(exc: BaseException) -> bool:
    """Is this "you have not been granted access" rather than a real fault?

    Duck-typed on purpose: huggingface_hub raises GatedRepoError (an
    HfHubHTTPError carrying .response), and classifying by NAME plus HTTP
    status means neither the classifier nor its tests have to import the
    library.
    """
    if type(exc).__name__ in _GATED_EXC_NAMES:
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in _GATED_STATUSES


_GATED_EXC_NAMES = frozenset({"GatedRepoError"})
_GATED_STATUSES = frozenset({401, 403})


def gated_message(source: HfSource) -> str:
    return (
        f"{source.repo_id} is gated on HuggingFace and the token in HF_TOKEN cannot read it.\n"
        f"  1. open {source.url} while signed in to the account that owns the token\n"
        f"  2. accept the terms there (the 'Agree and access repository' button)\n"
        f"  3. re-run this command with HF_TOKEN set to that account's token\n"
        f"This is an access grant to make on the dataset page, not a fault in the build."
    )


def acquire_hf(
    store, source: HfSource, *, root: str | Path, snapshot_fn=None, token: str | None = None
) -> dict:
    """Snapshot one HF dataset under `root/<key>` and index what landed."""
    dest = Path(root) / source.key
    snapshot = _real_snapshot if snapshot_fn is None else snapshot_fn
    try:
        local = snapshot(
            repo_id=source.repo_id,
            local_dir=dest,
            token=token,
            allow_patterns=source.allow_patterns,
        )
    except Exception as exc:
        if is_gated_error(exc):
            raise GatedSourceError(gated_message(source)) from exc
        raise
    store.upsert_source(source.source_id, source.license, url=source.url)
    stats = index_tree(store, local or dest, source_id=source.source_id)
    stats["repo_id"] = source.repo_id
    stats["local_dir"] = str(local or dest)
    return stats


# --------------------------------------------------------------------------
# The real seams. Nothing above this line imports a client.
# --------------------------------------------------------------------------

class S3Bucket:
    """Anonymous reader for a public AWS Open Data bucket.

    No credentials: the bucket is public and signing it with whatever the
    operator happens to have in their environment is how an unrelated AWS
    account gets billed for someone else's judgments.
    """

    def __init__(self, bucket: str = SC_BUCKET, region: str = SC_REGION, *, chunk_size: int = _READ_BLOCK):
        self.bucket = bucket
        self.region = region
        self.chunk_size = chunk_size
        self._s3 = None

    def _client(self):
        if self._s3 is None:
            try:
                import boto3
                from botocore import UNSIGNED
                from botocore.config import Config
            except ImportError as exc:
                raise AcquisitionError(
                    "boto3/botocore are needed to read the public judgment bucket and "
                    "are not installed - run: pip install -e .[build]"
                ) from exc
            self._s3 = boto3.client(
                "s3",
                region_name=self.region,
                config=Config(
                    signature_version=UNSIGNED, retries={"max_attempts": 5, "mode": "standard"}
                ),
            )
        return self._s3

    def list_objects(self, prefix: str) -> Iterator[ObjectEntry]:
        paginator = self._client().get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", ()):
                key = obj["Key"]
                if key.endswith("/"):
                    continue  # a directory marker, not an object
                etag = (obj.get("ETag") or "").strip('"') or None
                yield ObjectEntry(key=key, size=int(obj["Size"]), etag=etag)

    def fetch(self, key: str) -> Iterator[bytes]:
        body = self._client().get_object(Bucket=self.bucket, Key=key)["Body"]
        try:
            yield from body.iter_chunks(self.chunk_size)
        finally:
            body.close()


def _real_snapshot(*, repo_id, local_dir, token=None, allow_patterns=None):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise AcquisitionError(
            "huggingface_hub is needed for dataset snapshots and is not installed - "
            "run: pip install -e .[build]"
        ) from exc
    return snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        token=token,
        allow_patterns=list(allow_patterns) if allow_patterns else None,
    )


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def _fmt_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"  # pragma: no cover - loop returns first


def _print_object_stats(label: str, stats: dict, *, verify: bool = False) -> None:
    # Under --verify every object the index already agrees with takes the
    # adopt path, so calling that column "adopted" would report 100k
    # adoptions on a run that adopted nothing.
    adopted = "re-hashed" if verify else "adopted"
    print(
        f"{label:<26}considered {stats['considered']:>7}  fetched {stats['fetched']:>7}  "
        f"{adopted} {stats['adopted']:>6}  skipped {stats['skipped']:>7}  "
        f"failed {stats['failed']:>5}  {_fmt_bytes(stats['bytes'])}"
    )
    if stats["changed"]:
        print(f"{'':<26}CHANGED UNDER US {stats['changed']} (see run_event artifact_hash_changed)")
    # Every one of them: the list is already bounded by --max-failures, and
    # truncating at ten hid failures of a run that never reached the cap.
    for failure in stats["failures"]:
        print(f"{'':<26}failed {failure['key']}: {failure['error']}")


def main(argv: Sequence[str] | None = None, *, fetcher=None, snapshot_fn=None) -> int:
    import argparse

    from tuned.data.config import load_build_config
    from tuned.data.paths import build_paths
    from tuned.data.store import Store

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data_law_v1.yaml")
    parser.add_argument(
        "--kind",
        default="all",
        choices=("all", "metadata", "pdf", "hf"),
        help="all = metadata, then the HF snapshots, then the (much larger) PDFs",
    )
    parser.add_argument("--years", default=None, help=f"default {DEV_YEARS[0]}-{DEV_YEARS[-1]}")
    parser.add_argument("--language", default=ENGLISH, choices=(ENGLISH, REGIONAL))
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after N objects fetched, adopted or FAILED per kind; a failure "
        "spends the cap like a download does, and objects already local are "
        "skipped without spending it, so a resumed run advances",
    )
    parser.add_argument("--max-failures", type=int, default=DEFAULT_MAX_FAILURES)
    parser.add_argument(
        "--verify", action="store_true", help="re-hash files the index already agrees with"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="decide for every object, download nothing"
    )
    parser.add_argument(
        "--hf-source",
        action="append",
        default=None,
        choices=sorted(HF_SOURCES),
        help="repeatable; default is every registered snapshot",
    )
    args = parser.parse_args(argv)

    years = parse_years(args.years) if args.years else DEV_YEARS
    kinds = ("metadata", "hf", "pdf") if args.kind == "all" else (args.kind,)

    cfg = load_build_config(args.config)
    paths = build_paths(cfg.build.workdir).ensure()
    corpus = paths.corpus_dir
    store = Store.open(paths.state_db)
    token = os.environ.get("HF_TOKEN")
    code = 0
    try:
        # min/max, not first/last: --years takes a comma list, and printing
        # "2020-2011" for "2020,2010-2011" would misdescribe the run.
        print(
            f"years {min(years)}-{max(years)} ({len(years)} partitions)  root {corpus}"
            + ("  [DRY RUN]" if args.dry_run else "")
        )
        for kind in kinds:
            if kind == "hf":
                for key in args.hf_source or sorted(HF_SOURCES):
                    source = HF_SOURCES[key]
                    if args.dry_run:
                        print(f"hf:{key:<23}[DRY RUN] would snapshot {source.repo_id}")
                        continue
                    try:
                        stats = acquire_hf(
                            store,
                            source,
                            root=corpus / "hf",
                            snapshot_fn=snapshot_fn,
                            token=token,
                        )
                    except GatedSourceError as exc:
                        # Not fatal to the run: the PDF pull is days of work
                        # and must not wait on an access grant.
                        print(f"hf:{key:<23}GATED - not acquired")
                        print(str(exc))
                        code = 2
                        continue
                    except Exception as exc:
                        # Nor is a hub 5xx, a DNS blip or a missing client
                        # library. Exactly the same reasoning as the gated
                        # branch above: this is a snapshot of a few hundred
                        # megabytes, the next kind is days of PDFs, and
                        # letting it propagate out of main would mean a
                        # transient failure after the metadata already landed
                        # costs the whole pull.
                        print(f"hf:{key:<23}FAILED - {type(exc).__name__}: {exc}")
                        store.log_event(
                            "acquire_failed",
                            {
                                "source_id": source.source_id,
                                "key": source.repo_id,
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                        )
                        code = max(code, 1)
                        continue
                    print(
                        f"hf:{key:<23}indexed {stats['indexed']:>7}  "
                        f"skipped {stats['skipped']:>7}  {_fmt_bytes(stats['bytes'])}  "
                        f"-> {stats['local_dir']}"
                    )
                continue

            bucket = fetcher if fetcher is not None else S3Bucket()
            stats = acquire_objects(
                store,
                bucket,
                entries_for(bucket, kind, years, language=args.language),
                root=corpus / "sc",
                verify=args.verify,
                dry_run=args.dry_run,
                limit=args.limit,
                max_failures=args.max_failures,
            )
            _print_object_stats(f"{kind}:{args.language}", stats, verify=args.verify)
            if stats["failed"]:
                code = max(code, 1)
        print(f"artifacts indexed -> {store.artifact_count()} ({paths.state_db})")
    finally:
        store.close()
    return code


if __name__ == "__main__":
    import sys

    exit_code = main()
    # Same reasoning as replay.py/seeds.py: hf_xet/datasets leave non-daemon
    # threads that can wedge interpreter shutdown after all output is
    # written. Skip shutdown entirely.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
