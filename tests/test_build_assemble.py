"""assemble.py - a row is emitted byte-identical or dropped with a reason.

Fixtures are structural shapes with filler prose; no real judgment or eval
text appears anywhere here.
"""

import json
from pathlib import Path

import pytest
from pipeline_fakes import paths_for, temp_config
from test_build_decontaminate import prose, row

from tuned.data.acquire import sha256_file
from tuned.data.assemble import (
    ASSEMBLE_VERSION,
    DROP_EMPTY_ANSWER,
    DROP_EMPTY_USER,
    DROP_NO_CONTENT,
    DROP_NO_SCAFFOLD,
    DROP_TOO_LONG,
    DROP_TURNS,
    DROP_UNCLOSED_SCAFFOLD,
    MANIFEST_FILENAME,
    assemble_rows,
    built_row,
    conforms,
    raw_fields,
    rendered,
    token_length,
)
from tuned.data.assemble import main as assemble_main
from tuned.data.jsonl import read_jsonl, write_jsonl
from tuned.data.replay import empty_think
from tuned.data.split import manifest_digests

ASSEMBLE_SRC = Path(__file__).parent.parent / "src" / "tuned" / "data" / "assemble.py"

OPEN, CLOSE = "<think>", "</think>"


# --------------------------------------------------------------------------
# The injected tokenizer.
# --------------------------------------------------------------------------

class FakeTokenizer:
    """The two calls assemble.py and stats.py make, and nothing else.

    The template is Qwen3's SHAPE - `<|im_start|>role\\n...<|im_end|>` - not
    the bare content: the length being gated is the RENDERED length, and a fake
    that returned content alone would under-count per-turn overhead by a
    constant and hide a gate calibrated against it.

    `scale` is tokens per whitespace word. Real tokenizers are nowhere near 1
    on this corpus (Devanagari runs several tokens to the word), so a knob is
    honest as well as convenient: it is what lets a 500-word fixture sit above
    an 8192-token bucket without a 50 KB test file.

    The keyword asserts are the point of the double, not paranoia: they are
    what fails when the production call drifts to `add_generation_prompt=True`
    or lets the tokenizer add its own specials on top of a template that has
    already emitted them.
    """

    def __init__(self, scale: int = 1):
        self.scale = scale
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False, "assemble renders to text, then encodes - like sft.py"
        assert add_generation_prompt is False, "this is a completed turn, not a prompt"
        self.calls.append({"messages": messages, "tokenize": tokenize,
                           "add_generation_prompt": add_generation_prompt})
        return "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        )

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False, "the template already emitted the specials"
        return list(range(len(text.split()) * self.scale))


# --------------------------------------------------------------------------
# Fixtures.
# --------------------------------------------------------------------------

def reasoning_row(seed: int, words: int = 40, **prov) -> dict:
    return row(prose(seed, 30), f"{OPEN}{prose(seed + 1, words)}{CLOSE}{prose(seed + 2, 25)}",
               reasoning=True, **prov)


def empty_think_row(seed: int, **prov) -> dict:
    return row(prose(seed, 30), empty_think(OPEN, CLOSE) + prose(seed + 2, 25),
               reasoning=False, **prov)


def corpus(n: int = 10) -> list[dict]:
    return [
        reasoning_row(i * 10) if i % 5 else empty_think_row(i * 10)
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# Shape, in both directions.
# --------------------------------------------------------------------------

def test_the_three_shipped_builders_all_conform_despite_three_whitespace_layouts():
    """format_example, generated_rows and replay/curated lay the same shape out
    three different ways. A byte-equality check against format_example would
    reject the whole generated stream, which is the reading this is not."""
    from tuned.data.smoke import format_example

    formatted = format_example("q", "trace", "answer", OPEN, CLOSE)
    generated = row("q", f"{OPEN}\ntrace\n{CLOSE}\n\nanswer")
    replayed = row("q", empty_think(OPEN, CLOSE) + "answer")
    for candidate in (formatted, generated, replayed):
        assert conforms(candidate, OPEN, CLOSE) is None
    # ...and they really are byte-different, or this proves nothing.
    contents = {c["messages"][1]["content"] for c in (formatted, generated, replayed)}
    assert len(contents) == 3


@pytest.mark.parametrize(
    "candidate,reason",
    [
        ({}, DROP_TURNS),
        ({"messages": []}, DROP_TURNS),
        ({"messages": [{"role": "user", "content": "q"}]}, DROP_TURNS),
        ({"messages": [{"role": "user", "content": "q"},
                       {"role": "assistant", "content": f"{OPEN}t{CLOSE}a"},
                       {"role": "user", "content": "more"}]}, DROP_TURNS),
        # Right count, wrong order - the trainer masks on the marker order.
        ({"messages": [{"role": "assistant", "content": f"{OPEN}t{CLOSE}a"},
                       {"role": "user", "content": "q"}]}, DROP_TURNS),
        ({"messages": [{"role": "system", "content": "s"},
                       {"role": "assistant", "content": f"{OPEN}t{CLOSE}a"}]}, DROP_TURNS),
        ({"messages": [{"role": "user", "content": None},
                       {"role": "assistant", "content": f"{OPEN}t{CLOSE}a"}]}, DROP_TURNS),
        ({"messages": ["q", "a"]}, DROP_TURNS),
        ({"messages": [{"role": "user", "content": "   "},
                       {"role": "assistant", "content": f"{OPEN}t{CLOSE}a"}]}, DROP_EMPTY_USER),
        # A bare answer: what generated_rows emits when the teacher returned no
        # trace at all.
        ({"messages": [{"role": "user", "content": "q"},
                       {"role": "assistant", "content": "answer"}]}, DROP_NO_SCAFFOLD),
        # Opened somewhere in the middle rather than at the start - the
        # scaffold has to be the FIRST thing the trainer sees.
        ({"messages": [{"role": "user", "content": "q"},
                       {"role": "assistant", "content": f"Sure! {OPEN}t{CLOSE}a"}]},
         DROP_NO_SCAFFOLD),
        ({"messages": [{"role": "user", "content": "q"},
                       {"role": "assistant", "content": f"{OPEN}trace with no close"}]},
         DROP_UNCLOSED_SCAFFOLD),
        ({"messages": [{"role": "user", "content": "q"},
                       {"role": "assistant", "content": f"{OPEN}t{CLOSE}   "}]},
         DROP_EMPTY_ANSWER),
        ({"messages": [{"role": "user", "content": "q"},
                       {"role": "assistant", "content": empty_think(OPEN, CLOSE)}]},
         DROP_EMPTY_ANSWER),
    ],
)
def test_each_shape_fault_names_itself(candidate, reason):
    assert conforms(candidate, OPEN, CLOSE) == reason


def test_a_close_tag_inside_the_trace_is_not_mistaken_for_the_opening_one():
    """`content[len(open):]` and not `content.split(open)` - a trace that
    quotes the open tag must not shift where the close is looked for."""
    content = f"{OPEN}the model wrote {OPEN} inside its trace{CLOSE}the answer"
    assert conforms(row("q", content), OPEN, CLOSE) is None


def test_the_close_is_looked_for_PAST_the_open_tag_not_inside_it():
    """The reason `tail` is a slice rather than the whole content.

    Under the shipped tags the two are indistinguishable - `</think>` is not a
    substring of `<think>`, so the first close is in the same place either way,
    and a mutant replacing the slice with the whole string survived the suite.
    They come apart the moment a config's open tag CONTAINS its close tag,
    which the think tags are config (`data.think_open`/`data.think_close`) and
    nothing forbids. Searching the whole content would then find the close
    inside the opening tag and call an unclosed block closed.
    """
    nested_open, nested_close = "<a></a>", "</a>"
    unclosed = row("q", f"{nested_open}a trace that never closes")
    assert conforms(unclosed, nested_open, nested_close) == DROP_UNCLOSED_SCAFFOLD
    # ...and a genuinely closed one under the same tags still passes.
    closed = row("q", f"{nested_open}trace{nested_close}answer")
    assert conforms(closed, nested_open, nested_close) is None


# --------------------------------------------------------------------------
# Building, and never rewriting.
# --------------------------------------------------------------------------

def test_a_built_row_comes_back_as_the_same_object():
    """Not equal - the SAME object. Byte-identical emission is a property of
    not touching the row, and an equal copy would let a rewrite hide in a
    round-trip."""
    source = reasoning_row(1)
    built, reason = built_row(source, think_open=OPEN, think_close=CLOSE)
    assert built is source and reason is None


def test_the_empty_think_scaffold_survives_byte_for_byte():
    """stats.py measures this with a byte comparison, so a whitespace tidy here
    would move a number there while looking like a cleanup."""
    source = empty_think_row(7)
    built, _ = built_row(source, think_open=OPEN, think_close=CLOSE)
    assert built["messages"][1]["content"].startswith(f"{OPEN}\n\n{CLOSE}")
    assert built["messages"][1]["content"] == source["messages"][1]["content"]
    # And a one-newline scaffold is NOT the empty-think shape - it conforms as
    # a row and is simply not that row.
    near = row("q", f"{OPEN}\n{CLOSE}answer")
    assert conforms(near, OPEN, CLOSE) is None
    assert not near["messages"][1]["content"].startswith(empty_think(OPEN, CLOSE))


def test_raw_fields_render_through_format_example_and_keep_their_provenance():
    from tuned.data.smoke import format_example

    source = {"problem": "q", "reasoning": "trace", "solution": "answer",
              "_prov": {"source": "somewhere", "license": "Apache-2.0"}}
    built, reason = built_row(source, think_open=OPEN, think_close=CLOSE)
    assert reason is None
    assert built["messages"] == format_example("q", "trace", "answer", OPEN, CLOSE)["messages"]
    # _prov survives to the emitted row: stats.py grades on it and push.py's
    # card counts licenses off it.
    assert built["_prov"] == source["_prov"]


def test_raw_fields_without_provenance_do_not_grow_an_empty_one():
    built, _ = built_row({"problem": "q", "reasoning": "t", "solution": "a"},
                         think_open=OPEN, think_close=CLOSE)
    assert "_prov" not in built


def test_raw_fields_are_all_three_or_none():
    assert raw_fields({"problem": "q", "reasoning": "t", "solution": "a"}) == ("q", "t", "a")
    assert raw_fields({"problem": "q", "reasoning": "t"}) is None
    assert raw_fields({"problem": "q", "reasoning": "t", "solution": 5}) is None
    assert raw_fields({}) is None


def test_a_row_that_is_neither_shape_is_dropped_by_name():
    built, reason = built_row({"text": "just a string"}, think_open=OPEN, think_close=CLOSE)
    assert built is None and reason == DROP_NO_CONTENT


def test_raw_fields_with_an_empty_solution_are_dropped_not_shipped_hollow():
    """format_example will happily build an empty answer; the shape check runs
    on the BUILT row for exactly that reason."""
    built, reason = built_row({"problem": "q", "reasoning": "t", "solution": "  "},
                              think_open=OPEN, think_close=CLOSE)
    assert built is None and reason == DROP_EMPTY_ANSWER


def test_the_accepted_generation_export_already_emits_messages():
    """The brief expected a raw-fields export to be the generated path. It is
    not: decontaminate.generated_rows builds messages rows and store_items
    feeds them to the screen, so a generation is shape 1 long before this pass.
    Shape 2 is a seam for a future exporter, and this is what says so.
    """
    import inspect

    from tuned.data import decontaminate

    source = inspect.getsource(decontaminate.generated_rows)
    assert '"messages"' in source and '"_prov"' in source
    assert not any(f'"{field}"' in source for field in ("problem", "solution"))


# --------------------------------------------------------------------------
# Length.
# --------------------------------------------------------------------------

def test_length_is_measured_on_the_rendered_row_not_the_raw_content():
    tok = FakeTokenizer()
    messages = [{"role": "user", "content": "one two"},
                {"role": "assistant", "content": "three"}]
    text = rendered(tok, messages)
    assert "<|im_start|>user" in text and "<|im_end|>" in text
    # Five, not the three words of content: the per-turn markers are part of
    # what the trainer tokenizes and a length gate that ignored them would pass
    # rows the trainer then truncates.
    assert token_length(tok, messages) == len(text.split()) == 5
    assert token_length(tok, messages) > sum(len(m["content"].split()) for m in messages)
    assert tok.calls[-1]["add_generation_prompt"] is False


def test_a_row_over_the_bucket_is_dropped_and_a_row_at_it_is_kept():
    tok = FakeTokenizer()
    short, long_ = reasoning_row(1, words=10), reasoning_row(2, words=60)
    limit = token_length(tok, short["messages"])
    assert token_length(tok, long_["messages"]) > limit
    kept, drops, stats = assemble_rows(
        [short, long_], tokenizer=tok, think_open=OPEN, think_close=CLOSE, max_tokens=limit
    )
    # `>` and not `>=`: a row exactly at the bucket fits it.
    assert kept == [short] and stats["kept"] == 1
    assert drops[0]["reason"] == DROP_TOO_LONG and drops[0]["limit"] == limit
    assert drops[0]["tokens"] > limit
    # And with the limit one token higher the long row comes back, so the
    # boundary is the boundary and not a filter on something else.
    kept, _drops, _stats = assemble_rows(
        [short, long_], tokenizer=tok, think_open=OPEN, think_close=CLOSE,
        max_tokens=token_length(tok, long_["messages"]),
    )
    assert kept == [short, long_]


def test_the_boundary_is_tested_AT_the_boundary_and_one_token_past_it():
    """The two fixtures above are 50 tokens apart, so `> limit` and
    `> limit + 1` agree on both of them and the off-by-one is invisible.

    A row exactly ONE token over is the whole point of drop-never-truncate: it
    ships under the mutant, and the trainer then truncates it - the outcome
    assemble.py exists to prevent. So the pair is built from ONE row measured
    against two limits, its own length and one below it.
    """
    tok = FakeTokenizer()
    fitted = reasoning_row(11, words=30)
    exact = token_length(tok, fitted["messages"])

    at_the_limit, drops, stats = assemble_rows(
        [fitted], tokenizer=tok, think_open=OPEN, think_close=CLOSE, max_tokens=exact
    )
    assert at_the_limit == [fitted] and drops == [] and stats["max_tokens_kept"] == exact

    one_over, drops, stats = assemble_rows(
        [fitted], tokenizer=tok, think_open=OPEN, think_close=CLOSE, max_tokens=exact - 1
    )
    assert one_over == [] and stats["kept"] == 0
    assert stats["by_reason"] == {DROP_TOO_LONG: 1}
    assert (drops[0]["tokens"], drops[0]["limit"]) == (exact, exact - 1)
    assert drops[0]["tokens"] == drops[0]["limit"] + 1  # ...exactly one over


def test_an_over_length_row_is_dropped_whole_and_never_trimmed():
    tok = FakeTokenizer()
    fat = reasoning_row(3)
    kept, drops, _stats = assemble_rows(
        [fat], tokenizer=tok, think_open=OPEN, think_close=CLOSE, max_tokens=5
    )
    assert kept == []
    # The drop record carries provenance and a measurement, not a shortened row.
    assert set(drops[0]) == {"index", "reason", "tokens", "limit", "prov"}
    assert fat["messages"][1]["content"].endswith(prose(5, 25))  # untouched in place


def test_the_drop_record_names_the_row_by_provenance():
    tok = FakeTokenizer()
    bad = row("q", "no scaffold at all", source="somewhere", license="Apache-2.0")
    _kept, drops, _stats = assemble_rows(
        [bad], tokenizer=tok, think_open=OPEN, think_close=CLOSE, max_tokens=1000
    )
    assert drops[0]["prov"]["source"] == "somewhere"


def test_the_stats_count_every_reason_that_fired_and_no_reason_that_did_not():
    tok = FakeTokenizer()
    rows = [reasoning_row(1), row("q", "bare"), {"nothing": True}, reasoning_row(2)]
    _kept, _drops, stats = assemble_rows(
        rows, tokenizer=tok, think_open=OPEN, think_close=CLOSE, max_tokens=10_000
    )
    assert stats["total"] == 4 and stats["kept"] == 2 and stats["dropped"] == 2
    assert stats["by_reason"] == {DROP_NO_CONTENT: 1, DROP_NO_SCAFFOLD: 1}
    assert DROP_TOO_LONG not in stats["by_reason"]


@pytest.mark.parametrize("scale,expected_kept", [(1, 2), (5000, 0)])
def test_the_bucket_is_the_trainers_number_not_this_modules(tmp_path, scale, expected_kept):
    """8192 comes from train.main.max_seq_length through config.py. The same
    two rows survive or die on the tokenizer alone."""
    from tuned.data.config import load_build_config

    cfg = load_build_config(temp_config(tmp_path), allow_unpinned=True)
    assert cfg.max_seq_length == 8192
    kept, _drops, _stats = assemble_rows(
        [reasoning_row(1), reasoning_row(2)], tokenizer=FakeTokenizer(scale),
        think_open=OPEN, think_close=CLOSE, max_tokens=cfg.max_seq_length,
    )
    assert len(kept) == expected_kept


# --------------------------------------------------------------------------
# The CLI.
# --------------------------------------------------------------------------

def split_output(tmp_path, train_rows, eval_rows, *, manifest="default"):
    """split.py's two files plus the manifest that vouches for both."""
    cfg = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    train = paths.out_dir / "split_train.jsonl"
    evaluation = paths.out_dir / "split_eval.jsonl"
    n_train = write_jsonl(train, train_rows)
    n_eval = write_jsonl(evaluation, eval_rows)
    if manifest == "default":
        manifest = {
            "stage": "split",
            "split_version": 1,
            "outputs": [
                {"path": str(train), "rows": n_train, "sha256": sha256_file(train)},
                {"path": str(evaluation), "rows": n_eval, "sha256": sha256_file(evaluation)},
            ],
            "dedupe": {"stage": "dedupe", "dedupe_version": 4,
                       "decontamination": {"decon_version": 4}},
        }
    if manifest is not None:
        (paths.out_dir / "split.json").write_text(json.dumps(manifest), encoding="utf-8")
    return cfg, paths


def test_the_emitted_bytes_are_the_input_bytes(tmp_path):
    train_rows, eval_rows = corpus(8), corpus(3)
    cfg, paths = split_output(tmp_path, train_rows, eval_rows)
    assert assemble_main(["--config", cfg], tokenizer=FakeTokenizer()) == 0
    assert (paths.out_dir / "law_v1_train.jsonl").read_bytes() == (
        paths.out_dir / "split_train.jsonl"
    ).read_bytes()
    assert list(read_jsonl(paths.out_dir / "law_v1_eval.jsonl")) == eval_rows


def test_provenance_stays_on_the_emitted_rows(tmp_path):
    """push.py decides what goes public; stats.py cannot grade without it. The
    trainer drops it anyway - sft.py maps messages -> text with
    remove_columns=ds.column_names - so stripping it here would only cost the
    two readers that need it."""
    cfg, paths = split_output(tmp_path, corpus(6), corpus(2))
    assert assemble_main(["--config", cfg], tokenizer=FakeTokenizer()) == 0
    emitted = list(read_jsonl(paths.out_dir / "law_v1_train.jsonl"))
    assert all("_prov" in r and r["_prov"]["license"] for r in emitted)


def test_the_manifest_carries_the_chain_and_names_its_instrument(tmp_path):
    cfg, paths = split_output(tmp_path, corpus(8), corpus(3))
    assert assemble_main(["--config", cfg], tokenizer=FakeTokenizer()) == 0
    manifest = json.loads((paths.out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["assemble_version"] == ASSEMBLE_VERSION == 1
    assert manifest["split_check"]["status"] == "verified"
    # The whole chain, three links deep.
    assert manifest["split"]["dedupe"]["decontamination"]["decon_version"] == 4
    # WHICH tokenizer measured the lengths - a gate whose instrument is not
    # recorded is a number nobody can reproduce.
    assert manifest["tokenizer"]["repo"] == "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    assert manifest["max_tokens"] == 8192
    assert manifest["packing"] is False
    assert manifest["counts"]["train"]["kept"] == 8
    assert manifest_digests(manifest) == {
        sha256_file(paths.out_dir / "law_v1_train.jsonl"),
        sha256_file(paths.out_dir / "law_v1_eval.jsonl"),
    }


def test_drops_are_written_with_their_side_and_reason(tmp_path):
    train_rows = corpus(6) + [row("q", "bare answer, no scaffold")]
    cfg, paths = split_output(tmp_path, train_rows, corpus(2))
    assert assemble_main(["--config", cfg], tokenizer=FakeTokenizer()) == 0
    drops = list(read_jsonl(paths.out_dir / "assemble_drops.jsonl"))
    assert [(d["side"], d["reason"]) for d in drops] == [("train", DROP_NO_SCAFFOLD)]


def test_a_generation_with_no_trace_is_reported_not_repaired(tmp_path, capsys):
    cfg, _paths = split_output(tmp_path, corpus(6) + [row("q", "bare")], corpus(2))
    assert assemble_main(["--config", cfg], tokenizer=FakeTokenizer()) == 0
    out = capsys.readouterr().out
    assert f"drop[{DROP_NO_SCAFFOLD}]: 1" in out
    assert "adding the empty block here would be a rewrite" in out


def test_a_clean_run_does_not_print_the_scaffold_warning(tmp_path, capsys):
    cfg, _paths = split_output(tmp_path, corpus(6), corpus(2))
    assert assemble_main(["--config", cfg], tokenizer=FakeTokenizer()) == 0
    assert "would be a rewrite" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "manifest,banner",
    [
        (None, "NO UPSTREAM MANIFEST"),
        ({"stage": "split", "outputs": []}, "NO OUTPUT DIGEST"),
        ({"stage": "split", "outputs": [{"path": "x", "rows": 1, "sha256": "0" * 64}]},
         "DESCRIBES DIFFERENT ROWS"),
    ],
)
def test_a_broken_chain_refuses_before_anything_is_written(tmp_path, capsys, manifest, banner):
    cfg, paths = split_output(tmp_path, corpus(8), corpus(3), manifest=manifest)
    assert assemble_main(["--config", cfg], tokenizer=FakeTokenizer()) == 2
    assert banner in capsys.readouterr().out
    assert not (paths.out_dir / "law_v1_train.jsonl").exists()


def test_one_verified_side_is_not_enough(tmp_path):
    """BOTH inputs must be outputs the manifest recorded. A manifest that
    vouches for the train file only would let an eval file from anywhere at all
    into the dataset under a verified stamp."""
    cfg, paths = split_output(tmp_path, corpus(8), corpus(3))
    write_jsonl(paths.out_dir / "split_eval.jsonl", corpus(4))
    assert assemble_main(["--config", cfg], tokenizer=FakeTokenizer()) == 2


def test_everything_dropped_from_a_non_empty_input_exits_1(tmp_path, capsys):
    cfg, _paths = split_output(tmp_path, [row("q", "bare")] * 3, corpus(2))
    assert assemble_main(["--config", cfg], tokenizer=FakeTokenizer()) == 1
    assert "EVERY TRAIN ROW WAS DROPPED" in capsys.readouterr().out


def test_an_emptied_eval_side_exits_1_rather_than_shipping_an_unscoreable_build(
    tmp_path, capsys
):
    cfg, _paths = split_output(tmp_path, corpus(8), [row("q", "bare")])
    assert assemble_main(["--config", cfg], tokenizer=FakeTokenizer()) == 1
    assert "EVERY EVAL ROW WAS DROPPED" in capsys.readouterr().out


def test_an_empty_input_exits_1(tmp_path, capsys):
    cfg, _paths = split_output(tmp_path, [], [])
    assert assemble_main(["--config", cfg], tokenizer=FakeTokenizer()) == 1
    assert "NOTHING READ" in capsys.readouterr().out


def test_a_missing_input_names_the_command_that_makes_it(tmp_path, capsys):
    cfg = temp_config(tmp_path)
    paths_for(tmp_path)
    assert assemble_main(["--config", cfg], tokenizer=FakeTokenizer()) == 2
    assert "tuned.data.split" in capsys.readouterr().out


def test_the_run_is_logged_to_the_store(tmp_path):
    from tuned.data.store import Store

    cfg, paths = split_output(tmp_path, corpus(8), corpus(3))
    assert assemble_main(["--config", cfg], tokenizer=FakeTokenizer()) == 0
    event = json.loads(Store.open(paths.state_db).events("assemble")[0]["detail_json"])
    assert event["stage"] == "assemble"


def test_the_run_says_it_did_not_pack(tmp_path, capsys):
    cfg, _paths = split_output(tmp_path, corpus(8), corpus(3))
    assert assemble_main(["--config", cfg], tokenizer=FakeTokenizer()) == 0
    assert "UNPACKED" in capsys.readouterr().out


# --------------------------------------------------------------------------
# The real tokenizer.
# --------------------------------------------------------------------------

def test_the_real_template_renders_what_the_fake_pretends_to():
    """Skips locally (no [train] extra); runs where transformers is installed.

    The fake's shape is a claim about the real one, and this is where the claim
    is checked: markers present, the reasoning block surviving once, and the
    two-step render-then-encode agreeing with a direct tokenized call.
    """
    transformers = pytest.importorskip("transformers")
    from tuned.data.config import load_build_config

    cfg = load_build_config("configs/data_law_v1.yaml")
    tok = transformers.AutoTokenizer.from_pretrained(cfg.model_repo, revision=cfg.model_revision)
    messages = reasoning_row(1)["messages"]
    text = rendered(tok, messages)
    assert cfg.instruction_part in text and cfg.response_part in text
    response = text[text.index(cfg.response_part):]
    assert response.count(cfg.think_open) == 1 and response.count(cfg.think_close) == 1
    direct = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    assert token_length(tok, messages) == len(direct)


# --------------------------------------------------------------------------
# Conventions.
# --------------------------------------------------------------------------

def test_cli_hard_exits_after_success():
    assert "os._exit(" in ASSEMBLE_SRC.read_text(encoding="utf-8")


def test_the_version_ledger_describes_the_version_the_module_ships():
    import re

    source = ASSEMBLE_SRC.read_text(encoding="utf-8")
    entries = [int(n) for n in re.findall(r"^# (\d+)  ", source, re.M)]
    assert entries == sorted(entries)
    assert entries[-1] == ASSEMBLE_VERSION
    assert entries[0] == 1
