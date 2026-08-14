import json
import random

import pytest
from pipeline_fakes import paths_for, temp_config
from test_build_decontaminate import (
    _cli_bytes,
    all_eval_snapshots,
    install_fake_semhash,
    items,
    prose,
    row,
    shuffled,
)

from tuned.data import dedupe as dedupe_module
from tuned.data.decontaminate import (
    EVAL_SETS,
    SEMANTIC_UNUSABLE,
    EvalIndex,
    EvalItem,
    NGRAM as DECON_NGRAM,
    decontaminate_items,
    gram_hashes,
    item_of,
    jaccard,
    tokens,
)
from tuned.data.decontaminate import main as decon_main
from tuned.data.dedupe import (
    CNR_CAP,
    NGRAM,
    PROMPT_JACCARD,
    REASON_CAP,
    REASON_EXACT,
    REASON_NEAR_PROMPT,
    REASON_NEAR_ROW,
    REASON_SEMANTIC,
    ROW_JACCARD,
    PrefixIndex,
    apply_cap,
    candidate_of,
    cap_survivors,
    case_id_of,
    dedupe_items,
    flagged_indexes,
    prefix_length,
    score_of,
)
from tuned.data.dedupe import main as dedupe_main
from tuned.data.jsonl import write_jsonl
from tuned.data.store import Store


def grams(text: str) -> frozenset[int]:
    return gram_hashes(tokens(text), NGRAM)


def variant(text: str, keep: float, seed: int) -> str:
    """`text` with its tail replaced, so the pair's Jaccard is controllable
    and the fixture can sit NEAR a threshold instead of at 0 or 1."""
    words = text.split()
    cut = int(len(words) * keep)
    return " ".join(words[:cut]) + " " + prose(seed, len(words) - cut)


# --------------------------------------------------------------------------
# The exact candidate generator.
# --------------------------------------------------------------------------

def test_prefix_length_is_the_bound_the_proof_gives():
    assert prefix_length(100, 0.9) == 11  # 100 - ceil(90) + 1
    assert prefix_length(100, 0.85) == 16
    assert prefix_length(1, 0.9) == 1
    assert prefix_length(0, 0.9) == 0
    # ~10% of a real row's grams, which is where the memory saving comes from.
    assert prefix_length(2500, ROW_JACCARD) == 251


def test_the_prefix_index_finds_every_pair_brute_force_finds():
    """Exactness asserted against brute force, not argued in a comment: LSH's
    error direction is missed pairs, and this replaces LSH precisely because
    it has no such direction."""
    rng = random.Random(3)
    positives = 0
    for _ in range(120):
        threshold = rng.choice([0.6, 0.85, 0.9])
        sets: list[frozenset[int]] = []
        for _ in range(10):
            # Half fresh, half near-copies of an earlier set: random sets alone
            # never reach 0.85 and the fixture would assert nothing.
            if sets and rng.random() < 0.5:
                base = sorted(rng.choice(sets))
                keep = rng.sample(base, max(1, int(len(base) * rng.uniform(0.8, 1.0))))
                sets.append(frozenset(keep) | frozenset(rng.sample(range(200, 400), rng.randint(0, 3))))
            else:
                sets.append(frozenset(rng.sample(range(120), rng.randint(5, 60))))
        index = PrefixIndex(threshold)
        for i, current in enumerate(sets):
            best = None
            for j in range(i):
                score = jaccard(sets[j], current)
                if score >= threshold and (best is None or score > best[1]):
                    best = (j, score)
            found = index.verified(current)
            assert (found is None) == (best is None), "the index lost a real pair"
            if best is not None:
                assert found[0] == best[0] and found[1] == pytest.approx(best[1])
                positives += 1
            index.add(current)
    assert positives > 10, "the fixture never produced a pair above a threshold"


def test_one_gram_less_than_the_bound_loses_a_pair(monkeypatch):
    """The prefix length is a BOUND, not a tuning knob - a shorter prefix
    silently misses genuine duplicates, which is what makes the +1 in it
    load-bearing.

    The worst case the bound exists for, constructed: B is A with its four
    LOWEST grams removed, so the two prefixes are disjoint one gram before the
    bound and touch exactly at it.
    """
    base = frozenset(range(1, 41))
    near = frozenset(range(5, 41))
    assert jaccard(base, near) == pytest.approx(ROW_JACCARD)
    assert prefix_length(len(base), ROW_JACCARD) == 5
    assert set(sorted(base)[:4]).isdisjoint(sorted(near)[:4])

    index = PrefixIndex(ROW_JACCARD)
    index.add(base)
    assert index.verified(near) is not None

    monkeypatch.setattr(
        dedupe_module, "prefix_length", lambda size, threshold: max(1, prefix_length(size, threshold) - 1)
    )
    short = PrefixIndex(ROW_JACCARD)
    short.add(base)
    assert short.verified(near) is None


# --------------------------------------------------------------------------
# The stack.
# --------------------------------------------------------------------------

def test_an_exact_duplicate_drops_the_later_row_and_names_the_survivor():
    text = prose(1, 80)
    kept, drops, stats = dedupe_items(items(row(text), row(prose(2, 80)), row(text)))
    assert len(kept) == 2
    assert stats["by_reason"] == {REASON_EXACT: 1}
    assert drops[0]["origin"] == "fixture#2"
    assert drops[0]["duplicate_of"] == kept[0].key


def test_the_prompt_threshold_decides_in_both_directions():
    base = prose(11, 400)
    near = variant(base, 0.93, 12)
    far = variant(base, 0.88, 13)
    over = jaccard(grams(base), grams(near))
    under = jaccard(grams(base), grams(far))
    # The fixture straddles the constant CLOSELY on both sides: a pair at 0.4
    # would leave the threshold free to slide to 0.7 unnoticed.
    assert under < PROMPT_JACCARD < over
    assert 0.75 < under and over < 0.95

    kept, drops, _ = dedupe_items(items(row(base, "x"), row(near, "y"), row(far, "z")))
    assert [d["reason"] for d in drops] == [REASON_NEAR_PROMPT]
    assert drops[0]["origin"] == "fixture#1"
    assert drops[0]["duplicate_of"] == kept[0].key
    assert drops[0]["jaccard"] == pytest.approx(round(over, 4))
    assert len(kept) == 2


def test_the_row_threshold_decides_in_both_directions():
    """Rule 3's own case: prompts too different for rule 2, whole rows close
    enough to be one example."""
    prompt_a, prompt_b = prose(21, 90), prose(22, 90)
    shared_answer = prose(23, 2000)
    assert jaccard(grams(prompt_a), grams(prompt_b)) < PROMPT_JACCARD
    left, right = row(prompt_a, shared_answer), row(prompt_b, shared_answer)
    over = jaccard(grams(f"{prompt_a}\n{shared_answer}"), grams(f"{prompt_b}\n{shared_answer}"))
    assert over > ROW_JACCARD

    kept, drops, _ = dedupe_items(items(left, right))
    assert len(kept) == 1 and drops[0]["reason"] == REASON_NEAR_ROW

    # ... and a pair just UNDER the row threshold survives: same construction,
    # a longer differing part. The fixture straddles the constant.
    long_a, long_b = prose(24, 200), prose(25, 200)
    under = jaccard(grams(f"{long_a}\n{shared_answer}"), grams(f"{long_b}\n{shared_answer}"))
    assert 0.8 < under < ROW_JACCARD < over
    assert len(dedupe_items(items(row(long_a, shared_answer), row(long_b, shared_answer)))[0]) == 2


def test_the_prompt_rule_only_fires_within_one_question_form():
    """The plan's 'J >= 0.85 on prompts' deletes the wave planner's own
    design when it is applied across forms: four tasks on ONE seed share the
    grounding and differ only by an instruction, so their prompts are ~0.99
    alike BY CONSTRUCTION. Measured here in both configurations."""
    grounding = prose(31, 1200)
    forms = ("irac_analysis", "statute_qa", "drafting", "issue_spotting")
    rows = [row(f"{grounding} task: {form}", prose(35 + i, 900), form=form)
            for i, form in enumerate(forms)]
    pair = jaccard(grams(rows[0]["messages"][0]["content"]),
                   grams(rows[1]["messages"][0]["content"]))
    assert pair > 0.98, "the premise: these four prompts really are near-identical"
    # ... while the WHOLE rows are not, so rule 3 is not what is being tested.
    assert jaccard(grams(rows[0]["messages"][0]["content"] + rows[0]["messages"][1]["content"]),
                   grams(rows[1]["messages"][0]["content"] + rows[1]["messages"][1]["content"])
                   ) < ROW_JACCARD

    kept, drops, _ = dedupe_items(items(*rows))
    assert len(kept) == 4 and not drops

    # The same four rows under ONE form: now they ARE duplicates, and the
    # rule fires - so the guard is what separates the two, not the texts.
    same_form = [row(r["messages"][0]["content"], r["messages"][1]["content"], form="one")
                 for r in rows]
    kept, drops, _ = dedupe_items(items(*same_form))
    assert len(kept) == 1
    assert [d["reason"] for d in drops] == [REASON_NEAR_PROMPT] * 3


def test_a_duplicate_that_changed_its_form_is_still_caught_by_the_row_rule():
    """The union's other side: rule 2 is blind across forms by design, so
    rule 3 has to carry the row that was re-labelled."""
    text = prose(41, 900)
    left = row(text, "same answer", form="irac_analysis")
    right = row(variant(text, 0.97, 42), "same answer", form="statute_qa")
    # Not byte-identical, so this is rule 3's case and not the exact rule's.
    assert left["messages"][0]["content"] != right["messages"][0]["content"]
    kept, drops, _ = dedupe_items(items(left, right))
    assert len(kept) == 1
    assert drops[0]["reason"] == REASON_NEAR_ROW


def test_a_prompt_duplicate_with_a_genuinely_different_answer_needs_the_prompt_rule():
    """And rule 2's own case: the whole rows are too different for rule 3."""
    prompt = prose(51, 300)
    left = row(prompt, prose(52, 900), form="f")
    right = row(prompt, prose(53, 900), form="f")
    assert jaccard(grams(f"{prompt}\n{prose(52, 900)}"), grams(f"{prompt}\n{prose(53, 900)}")) < ROW_JACCARD
    kept, drops, _ = dedupe_items(items(left, right))
    assert len(kept) == 1 and drops[0]["reason"] == REASON_NEAR_PROMPT


# --------------------------------------------------------------------------
# The per-case cap and its selection rule.
# --------------------------------------------------------------------------

def case_rows(n, *, forms, scores=None, cnr="DLHC010001232020"):
    scores = scores or [None] * n
    return [
        row(f"{prose(60 + i, 200)}", f"answer {i}", cnr=cnr, form=forms[i], score=scores[i])
        for i in range(n)
    ]


def test_the_cap_is_three_rows_per_case():
    assert CNR_CAP == 3  # the literal, beside the rule that spends it
    rows = case_rows(5, forms=["a", "b", "c", "d", "e"], scores=[5, 4, 3, 2, 1])
    kept, drops, stats = dedupe_items(items(*rows))
    assert len(kept) == CNR_CAP
    assert stats["capped"] == 2
    assert {d["reason"] for d in drops} == {REASON_CAP}
    assert stats["cases"] == 1 and stats["cases_over_cap"] == 1


def test_the_cap_takes_a_second_form_over_a_higher_scoring_second_row_of_the_first():
    """The selection rule, in the one case that distinguishes it from 'top 3
    by score': three rows in three forms teach three things, three rows in one
    form teach one thing three times."""
    rows = case_rows(4, forms=["irac", "irac", "irac", "statute"], scores=[5, 4.9, 4.8, 1])
    kept, _, _ = dedupe_items(items(*rows))
    forms = [k.row["_prov"]["form"] for k in kept]
    assert forms.count("statute") == 1, "the weakest row of a NEW form must take a slot"
    assert forms.count("irac") == 2
    # ... and the two irac survivors are its two best, not an arbitrary pair.
    assert [k.row["_prov"]["score"] for k in kept if k.row["_prov"]["form"] == "irac"] == [5, 4.9]


def test_a_group_at_the_cap_keeps_everything_and_still_ranks_it():
    group = [candidate_of(i) for i in items(*case_rows(3, forms=["a", "b", "c"],
                                                       scores=[1, 5, 3]))]
    survivors = cap_survivors(group)
    assert len(survivors) == len(group) == CNR_CAP
    # Ranked, not input order - one contract for every group size.
    assert [c.score for c in survivors] == [5, 3, 1]


def test_when_the_forms_run_out_the_cap_fills_by_score():
    group = [candidate_of(i) for i in items(*case_rows(4, forms=["a", "a", "b", "b"],
                                                       scores=[1, 5, 2, 4]))]
    survivors = cap_survivors(group)
    assert [c.score for c in survivors] == [5, 4, 2]


def test_an_unscored_row_does_not_outrank_a_scored_one():
    """`None` is not zero and it is not five: an unjudged stream row must not
    take a slot from a judged generation just because it sorted first."""
    group = [candidate_of(i) for i in items(*case_rows(4, forms=["a", "a", "a", "a"],
                                                       scores=[None, 3, None, 4]))]
    survivors = cap_survivors(group)
    assert [c.score for c in survivors] == [4, 3, None]


def test_ties_break_on_the_content_key_so_the_survivors_never_move():
    group = [candidate_of(i) for i in items(*case_rows(5, forms=["a"] * 5, scores=[3] * 5))]
    first = [c.key for c in cap_survivors(group)]
    assert first == sorted(c.key for c in group)[:CNR_CAP]
    assert first == [c.key for c in cap_survivors(list(reversed(group)))]


def test_rows_with_no_case_identifier_are_never_capped():
    """Grouping them under one None bucket would cap the entire replay stream
    - thousands of rows that share no case at all - at three."""
    rows = [row(prose(70 + i, 120), f"answer {i}") for i in range(10)]
    kept, drops, stats = dedupe_items(items(*rows))
    assert len(kept) == 10 and not drops
    assert stats["uncapped_rows"] == 10 and stats["cases"] == 0


def test_the_cap_bucket_prefers_the_strongest_identifier_a_row_carries():
    both = item_of(
        row("q", cnr="DLHC010001232020", neutral_citation="2020 INSC 484",
            case_title="Government of India v ISRO Drivers Association"),
        "x#0",
    )
    assert case_id_of(both) == "cnr:DLHC010001232020"
    assert case_id_of(item_of(row("q", neutral_citation="2020 INSC 484"), "x#1")) == "cit:2020 INSC 484"
    assert case_id_of(item_of(row("q"), "x#2")) is None


def test_the_cap_can_be_switched_off_without_touching_the_rest():
    rows = case_rows(5, forms=list("abcde"))
    assert len(dedupe_items(items(*rows), cap=None)[0]) == 5
    assert len(dedupe_items(items(*rows))[0]) == CNR_CAP


def test_a_cap_that_did_not_run_is_null_and_not_zero(tmp_path, capsys):
    """`uncapped_rows: 0, cases: 0, cases_over_cap: 0` is byte-identical to
    "the cap ran and reached every row" - the opposite of what --no-cap did.
    Years later, that is the difference between a dataset one judgment may
    dominate and one it cannot."""
    rows = case_rows(5, forms=list("abcde"))
    cfg, paths = _decontaminated(tmp_path, rows)
    capsys.readouterr()

    assert dedupe_main(["--config", cfg, "--no-cap"]) == 0
    out = capsys.readouterr().out
    manifest = json.loads((paths.out_dir / "dedupe.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["kept"] == 5
    for key in ("uncapped_rows", "cases", "cases_over_cap"):
        assert manifest["counts"][key] is None, key
    assert manifest["thresholds"]["cap"] is None
    assert "THE PER-CASE CAP DID NOT RUN" in out

    # ... and the run that DID cap reports numbers, so null is a statement and
    # not the shape of every run.
    assert dedupe_main(["--config", cfg]) == 0
    manifest = json.loads((paths.out_dir / "dedupe.json").read_text(encoding="utf-8"))
    assert (manifest["counts"]["cases"], manifest["counts"]["uncapped_rows"]) == (1, 0)


def test_apply_cap_writes_rows_back_in_input_order():
    rows = case_rows(4, forms=["a", "b", "c", "d"], scores=[1, 4, 3, 2])
    candidates = [candidate_of(i) for i in items(*rows)]
    kept, drops, _ = apply_cap(candidates)
    assert [c.item.origin for c in kept] == sorted(c.item.origin for c in kept)
    assert len(kept) == CNR_CAP and len(drops) == 1


def test_score_of_reads_the_judge_score_and_survives_junk():
    assert score_of(item_of(row("q", score=4.25), "x#0")) == 4.25
    assert score_of(item_of(row("q", score="not a number"), "x#1")) is None
    assert score_of(item_of(row("q"), "x#2")) is None


def test_an_unscored_row_is_not_a_row_that_scored_zero():
    """The docstring says "None is NOT zero" and the [None, 3, None, 4]
    fixture cannot show it: every judge score is positive, so reading None as
    0 sorts it last too and both readings keep the same three rows. It takes a
    row that genuinely scored 0 - which three zeroed judge dimensions produce
    - and a cap of one.

    The content keys are sha256 of the row, so the fixture's seeds are chosen
    (not arranged) to put the unscored row's key FIRST: under "None is zero"
    the two tie and the key hands the slot to the unscored one.
    """
    scored, unscored = items(
        row("zzz " + prose(301, 40), "a", cnr="DLHC010001232020", form="f", score=0.0),
        row("aaa " + prose(307, 40), "b", cnr="DLHC010001232020", form="f"),
    )
    group = [candidate_of(scored), candidate_of(unscored)]
    assert candidate_of(unscored).key < candidate_of(scored).key, "the premise: the tie-break"
    assert [c.score for c in cap_survivors(group, cap=1)] == [0.0]


# --------------------------------------------------------------------------
# The composition order. Correctness, not style.
# --------------------------------------------------------------------------

def _twin_fixture():
    """A duplicate cluster whose FIRST member is contaminated and whose twin
    is clean - the exact shape that decides the order of the two passes."""
    common = prose(81, 1200)
    question = "which of the following is the correct measure of " + prose(82, 25)
    contaminated = row(f"{common} {question}", "answer", form="f")
    clean = row(f"{common} {prose(83, 30)}", "answer", form="f")
    index = EvalIndex([EvalItem("bbl", "bbl#0", question, frozenset())])
    assert jaccard(
        grams(contaminated["messages"][0]["content"]), grams(clean["messages"][0]["content"])
    ) > PROMPT_JACCARD, "the premise: these two really are a duplicate cluster"
    return contaminated, clean, index


def test_decontaminating_first_lets_the_clean_twin_ship():
    contaminated, clean, index = _twin_fixture()
    screened, drops, _ = decontaminate_items(items(contaminated, clean), index)
    assert [d["origin"] for d in drops] == ["fixture#0"]
    kept, _, _ = dedupe_items(screened)
    assert len(kept) == 1
    assert kept[0].row == clean


def test_deduping_first_loses_the_clean_twin_as_well():
    """The reason the order is a correctness property: dedupe keeps the first
    member of a cluster, and if that member is contaminated the twin dies as a
    duplicate before decontamination ever sees it - so BOTH are lost."""
    contaminated, clean, index = _twin_fixture()
    deduped, drops, _ = dedupe_items(items(contaminated, clean))
    assert len(deduped) == 1 and deduped[0].row == contaminated
    assert drops[0]["reason"] == REASON_NEAR_PROMPT
    screened, _, _ = decontaminate_items(deduped, index)
    assert screened == [], "wrong order: the cluster is gone entirely"


# --------------------------------------------------------------------------
# The semantic layer: optional, and recorded either way.
# --------------------------------------------------------------------------

def test_a_semantic_backend_only_ever_adds_drops_and_names_them():
    rows = [row(prose(91 + i, 120), f"answer {i}") for i in range(3)]
    candidates = items(*rows)

    def fake(kept):
        return {kept[-1].key: kept[0].key}

    kept, drops, stats = dedupe_items(candidates, semantic=fake)
    assert len(kept) == 2
    assert drops[0]["reason"] == REASON_SEMANTIC
    assert drops[0]["duplicate_of"] == candidates[0].key
    assert stats["by_reason"] == {REASON_SEMANTIC: 1}


def test_the_real_seam_drops_a_semantic_duplicate_and_records_that_it_ran(tmp_path, monkeypatch, capsys):
    """The seam itself, not a hand-written callable standing in for it: the
    two `semantic=` tests below inject their own function, so before this the
    only code that ever touched semhash's API had never been executed by
    anything."""
    text = prose(161, 200)
    cfg, paths = _decontaminated(tmp_path, [row(text, "a"), row(shuffled(text, 162), "b"),
                                            row(prose(163, 200), "c")])
    capsys.readouterr()
    install_fake_semhash(monkeypatch)
    assert dedupe_main(["--config", cfg]) == 0
    manifest = json.loads((paths.out_dir / "dedupe.json").read_text(encoding="utf-8"))
    drops = [json.loads(line) for line in
             (paths.out_dir / "dedupe_drops.jsonl").read_text().splitlines()]
    assert manifest["semantic"] == "ran"
    assert [d["reason"] for d in drops] == [REASON_SEMANTIC]
    assert manifest["counts"]["kept"] == 2


def test_a_drifted_semhash_result_never_reads_as_a_deduplicated_corpus(tmp_path, monkeypatch, capsys):
    """The silent half of the seam, reproduced: with survivors named
    `.deduplicated`, the old `getattr(result, "selected", texts)` default read
    as 'semhash kept everything', so the pass flagged NOTHING and the manifest
    said `semantic: ran` over a corpus the layer never compared. The two
    directions were opposite in the two modules and neither was pinned."""
    text = prose(161, 200)
    cfg, paths = _decontaminated(tmp_path, [row(text, "a"), row(shuffled(text, 162), "b"),
                                            row(prose(163, 200), "c")])
    capsys.readouterr()
    install_fake_semhash(monkeypatch, attr="deduplicated")
    assert dedupe_main(["--config", cfg]) == 0
    out = capsys.readouterr().out
    manifest = json.loads((paths.out_dir / "dedupe.json").read_text(encoding="utf-8"))
    assert manifest["semantic"] == SEMANTIC_UNUSABLE
    assert manifest["semantic"] != "ran"
    assert ".selected" in manifest["semantic_detail"]
    assert "semantic self-dedupe did NOT run" in out
    assert manifest["counts"]["kept"] == 3  # the exact stack still ran


@pytest.mark.parametrize("answer", ["keep-everything", "keep-nothing"])
def test_a_seam_that_ignores_its_input_fails_the_control_in_either_direction(
    tmp_path, monkeypatch, capsys, answer
):
    text = prose(161, 200)
    cfg, paths = _decontaminated(tmp_path, [row(text, "a"), row(shuffled(text, 162), "b")])
    capsys.readouterr()
    install_fake_semhash(monkeypatch, answer=answer)
    assert dedupe_main(["--config", cfg]) == 0
    manifest = json.loads((paths.out_dir / "dedupe.json").read_text(encoding="utf-8"))
    assert manifest["semantic"] == SEMANTIC_UNUSABLE
    assert "the only correct answer is [1]" in manifest["semantic_detail"]


def test_two_candidates_carrying_the_same_text_are_both_reachable(monkeypatch):
    """Survivors are matched back BY COUNT: under set membership, two rows
    with the same joined text are both unflaggable no matter what semhash
    says, because the survivor's own text is in the set."""
    install_fake_semhash(monkeypatch)
    duplicate = prose(171, 60)
    assert flagged_indexes([duplicate, duplicate, prose(172, 60)]) == [1]


def test_the_semantic_layer_runs_after_the_exact_stack_not_instead_of_it():
    text = prose(101, 200)
    calls = {}

    def fake(kept):
        calls["seen"] = [c.item.origin for c in kept]
        return {}

    dedupe_items(items(row(text), row(text)), semantic=fake)
    # The exact duplicate is already gone by the time the semantic pass runs,
    # so it can neither rescue it nor be asked about it.
    assert calls["seen"] == ["fixture#0"]


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def _decontaminated(tmp_path, rows, *, eval_records=None):
    """Run the real decontamination pass so dedupe's input is what production
    hands it - manifest included."""
    cfg = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    write_jsonl(paths.streams_dir / "stream.jsonl", rows)
    store = Store.open(paths.state_db)
    all_eval_snapshots(store, tmp_path / "hf", eval_records or {})
    store.close()
    assert decon_main(["--config", cfg, "--no-generated"]) == 0
    return cfg, paths


def test_the_cli_refuses_a_missing_input_and_sends_the_operator_to_decontaminate(tmp_path, capsys):
    cfg = temp_config(tmp_path)
    paths_for(tmp_path)
    assert dedupe_main(["--config", cfg]) == 2
    out = capsys.readouterr().out
    assert "decontamination runs FIRST" in out
    assert "tuned.data.decontaminate" in out


def test_the_manifest_carries_an_upstream_waiver_forward(tmp_path, capsys):
    cfg = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    write_jsonl(paths.streams_dir / "stream.jsonl", [row(prose(111, 120))])
    store = Store.open(paths.state_db)
    all_eval_snapshots(store, tmp_path / "hf")
    store.conn.execute("DELETE FROM artifact WHERE source_id = ?", (EVAL_SETS["aibe"].source_id,))
    store.close()
    assert decon_main(["--config", cfg, "--no-generated", "--allow-missing-eval", "aibe"]) == 0
    capsys.readouterr()

    assert dedupe_main(["--config", cfg]) == 0
    out = capsys.readouterr().out
    manifest = json.loads((paths.out_dir / "dedupe.json").read_text(encoding="utf-8"))
    assert manifest["decontamination"]["eval_sets"]["aibe"]["allowed_missing"] is True
    assert "eval sets WAIVED upstream: aibe" in out
    # The counterfactual half: a screened input must NOT print the warning
    # that its rows were never screened.
    assert "NO DECONTAMINATION MANIFEST" not in out


def test_an_input_that_never_went_through_decontamination_is_recorded_as_such(tmp_path, capsys):
    """The misreading this exists to reject is CONSTRUCTED, not assumed away:
    a real decontamination pass has run and left its manifest in out/, and the
    input is a different file that never went through it. Reading the manifest
    from the OUTPUT's directory instead of the input's - which is what the
    code comment claimed to prevent - finds that manifest here, so the two
    readings no longer agree and the fixture can tell them apart."""
    text = prose(121, 120)
    cfg, paths = _decontaminated(tmp_path, [row(text), row(prose(122, 120))])
    capsys.readouterr()
    assert (paths.out_dir / "decontamination.json").exists(), "the premise: a manifest exists"

    raw = paths.corpus_dir / "unscreened.jsonl"
    write_jsonl(raw, [row(prose(123, 120))])
    assert dedupe_main(["--config", cfg, "--in", str(raw)]) == 0
    out = capsys.readouterr().out
    manifest = json.loads((paths.out_dir / "dedupe.json").read_text(encoding="utf-8"))
    assert manifest["decontamination"] is None
    assert manifest["decontamination_check"]["status"] == "no_manifest"
    assert manifest["decontamination_check"]["manifest"] is None
    assert "NO DECONTAMINATION MANIFEST" in out


def test_rows_that_are_not_the_screened_rows_do_not_inherit_the_manifest(tmp_path, capsys):
    """Reproduced end to end before it was fixed: never-screened rows shipped
    beside a manifest asserting a decontamination pass over OTHER rows, exit 0,
    and the NO DECONTAMINATION MANIFEST banner did not print. The counts
    visibly disagreed and nothing looked. Custody is bound to the digest of
    the bytes decontaminate.py wrote, so a manifest in the same directory
    cannot be inherited by rows it does not describe."""
    cfg, paths = _decontaminated(tmp_path, [row(prose(124, 120)), row(prose(125, 120))])
    capsys.readouterr()
    upstream = json.loads((paths.out_dir / "decontamination.json").read_text(encoding="utf-8"))
    assert upstream["output"]["rows"] == 2
    assert upstream["output"]["sha256"]

    # Four never-screened rows, dropped into the very directory the screened
    # pair and its manifest live in.
    smuggled = paths.out_dir / "smuggled.jsonl"
    write_jsonl(smuggled, [row(prose(126 + i, 120)) for i in range(4)])
    assert dedupe_main(["--config", cfg, "--in", str(smuggled)]) == 0
    out = capsys.readouterr().out
    manifest = json.loads((paths.out_dir / "dedupe.json").read_text(encoding="utf-8"))
    assert manifest["decontamination"] is None
    check = manifest["decontamination_check"]
    assert check["status"] == "content_mismatch"
    assert check["manifest"].endswith("decontamination.json")
    assert check["input_sha256"] != check["manifest_output"]["sha256"]
    assert "DESCRIBES DIFFERENT ROWS" in out


def test_the_screened_rows_themselves_do_verify(tmp_path, capsys):
    """The other half: the real output of the real pass carries its record
    forward, so the check above is rejecting a mismatch rather than rejecting
    everything."""
    cfg, paths = _decontaminated(tmp_path, [row(prose(128, 120)), row(prose(129, 120))])
    capsys.readouterr()
    assert dedupe_main(["--config", cfg]) == 0
    manifest = json.loads((paths.out_dir / "dedupe.json").read_text(encoding="utf-8"))
    check = manifest["decontamination_check"]
    assert check["status"] == "verified"
    assert check["input_sha256"] == check["manifest_output"]["sha256"]
    assert manifest["decontamination"]["counts"]["kept"] == 2
    assert "NO DECONTAMINATION MANIFEST" not in capsys.readouterr().out


def test_a_manifest_with_no_output_digest_is_not_inherited_either(tmp_path, capsys):
    """A manifest written by an older decon_version cannot say whether it
    describes these rows, and 'cannot tell' is not 'yes'."""
    cfg, paths = _decontaminated(tmp_path, [row(prose(130, 120)), row(prose(131, 120))])
    capsys.readouterr()
    path = paths.out_dir / "decontamination.json"
    stale = json.loads(path.read_text(encoding="utf-8"))
    stale.pop("output")
    stale["decon_version"] = 1
    path.write_text(json.dumps(stale), encoding="utf-8")

    assert dedupe_main(["--config", cfg]) == 0
    out = capsys.readouterr().out
    manifest = json.loads((paths.out_dir / "dedupe.json").read_text(encoding="utf-8"))
    assert manifest["decontamination"] is None
    assert manifest["decontamination_check"]["status"] == "no_output_digest"
    assert "NO OUTPUT DIGEST" in out


def test_several_inputs_cannot_all_be_the_screened_output(tmp_path, capsys):
    cfg, paths = _decontaminated(tmp_path, [row(prose(132, 120)), row(prose(133, 120))])
    capsys.readouterr()
    other = paths.out_dir / "extra.jsonl"
    write_jsonl(other, [row(prose(134, 120))])
    screened = paths.out_dir / "decontaminated.jsonl"
    assert dedupe_main(["--config", cfg, "--in", str(screened), "--in", str(other)]) == 0
    manifest = json.loads((paths.out_dir / "dedupe.json").read_text(encoding="utf-8"))
    assert manifest["decontamination"] is None
    assert manifest["decontamination_check"]["status"] == "several_inputs"
    assert "SEVERAL INPUTS" in capsys.readouterr().out


def test_the_case_id_lever_carries_through_from_the_screen_to_the_cap(tmp_path, capsys):
    """`--no-case-id-from-text` is the approved remedy for the case-identifier
    level's false-positive risk, and it only worked on one of the two passes:
    stream_items always asked for identifiers from text, so the screen and the
    cap disagreed about which case a row was about.

    Four rows naming ONE landmark citation in their grounding and nothing in
    _prov: with the lever off they are one case and the cap takes three; with
    it on they carry no identifier at all and none of them is capped.
    """
    cfg = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    rows = [row(f"as held in 2020 INSC 484 {prose(180 + i, 200)}", f"answer {i}")
            for i in range(4)]
    write_jsonl(paths.streams_dir / "stream.jsonl", rows)
    store = Store.open(paths.state_db)
    all_eval_snapshots(store, tmp_path / "hf")
    store.close()

    assert decon_main(["--config", cfg, "--no-generated"]) == 0
    capsys.readouterr()
    assert dedupe_main(["--config", cfg]) == 0
    capsys.readouterr()
    manifest = json.loads((paths.out_dir / "dedupe.json").read_text(encoding="utf-8"))
    assert manifest["thresholds"]["case_ids_from_text"] is True
    assert manifest["counts"]["kept"] == CNR_CAP
    assert manifest["counts"]["cases"] == 1

    # Now with the lever, passed to the SCREEN only - dedupe inherits it from
    # the manifest, so the two passes cannot disagree.
    assert decon_main(["--config", cfg, "--no-generated", "--no-case-id-from-text"]) == 0
    capsys.readouterr()
    assert dedupe_main(["--config", cfg]) == 0
    capsys.readouterr()
    manifest = json.loads((paths.out_dir / "dedupe.json").read_text(encoding="utf-8"))
    assert manifest["thresholds"]["case_ids_from_text"] is False
    assert manifest["counts"]["kept"] == 4
    assert manifest["counts"]["cases"] == 0
    assert manifest["counts"]["uncapped_rows"] == 4


def test_the_lever_can_be_passed_to_dedupe_directly_too(tmp_path, capsys):
    """An input with no verifiable custody has no upstream setting to inherit,
    so the flag has to work on its own."""
    cfg = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    raw = paths.corpus_dir / "unscreened.jsonl"
    write_jsonl(raw, [row(f"as held in 2020 INSC 484 {prose(190 + i, 200)}", f"answer {i}")
                      for i in range(4)])
    assert dedupe_main(["--config", cfg, "--in", str(raw)]) == 0
    capsys.readouterr()
    assert json.loads(
        (paths.out_dir / "dedupe.json").read_text(encoding="utf-8")
    )["counts"]["kept"] == CNR_CAP

    assert dedupe_main(["--config", cfg, "--in", str(raw), "--no-case-id-from-text"]) == 0
    capsys.readouterr()
    manifest = json.loads((paths.out_dir / "dedupe.json").read_text(encoding="utf-8"))
    assert manifest["thresholds"]["case_ids_from_text"] is False
    assert manifest["counts"]["kept"] == 4


def test_the_cli_writes_rows_drops_and_a_manifest(tmp_path, capsys):
    text = prose(131, 200)
    cfg, paths = _decontaminated(tmp_path, [row(text), row(text), row(prose(132, 200))])
    capsys.readouterr()
    assert dedupe_main(["--config", cfg]) == 0
    out = capsys.readouterr().out
    kept = (paths.out_dir / "deduped.jsonl").read_text().splitlines()
    drops = [json.loads(line) for line in (paths.out_dir / "dedupe_drops.jsonl").read_text().splitlines()]
    manifest = json.loads((paths.out_dir / "dedupe.json").read_text(encoding="utf-8"))
    assert len(kept) == 2 and len(drops) == 1
    assert json.loads(kept[0]) == row(text)
    assert manifest["thresholds"] == {
        "ngram": NGRAM, "prompt_jaccard": PROMPT_JACCARD,
        "row_jaccard": ROW_JACCARD, "cap": CNR_CAP, "case_ids_from_text": True,
    }
    assert manifest["counts"]["total"] == 3
    assert "drop[exact]: 1" in out
    event = json.loads(Store.open(paths.state_db).events("dedupe")[0]["detail_json"])
    assert event["counts"]["kept"] == 2


def test_nothing_read_exits_1(tmp_path, capsys):
    cfg = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    empty = paths.corpus_dir / "empty.jsonl"
    write_jsonl(empty, [])
    assert dedupe_main(["--config", cfg, "--in", str(empty)]) == 1
    assert "NOTHING READ" in capsys.readouterr().out


def test_two_runs_under_different_hash_seeds_are_byte_identical(tmp_path):
    """Run twice for real: the survivor of a duplicate cluster and the three
    rows a case keeps are exactly where a set iteration order would leak in,
    and downstream split.py assigns at case level - so a shifting survivor set
    moves the train/test boundary."""
    text = prose(141, 200)
    rows = (
        [row(text), row(text)]
        + case_rows(5, forms=list("abcde"), scores=[3, 3, 3, 3, 3])
        + [row(prose(150 + i, 150)) for i in range(8)]
    )
    cfg, paths = _decontaminated(tmp_path, rows)
    first = _cli_bytes(tmp_path, "0", "tuned.data.dedupe", ["--config", cfg])
    second = _cli_bytes(tmp_path, "1", "tuned.data.dedupe", ["--config", cfg])
    assert first["deduped.jsonl"] == second["deduped.jsonl"]
    assert first["dedupe_drops.jsonl"] == second["dedupe_drops.jsonl"]
    left, right = json.loads(first["dedupe.json"]), json.loads(second["dedupe.json"])
    left.pop("at"), right.pop("at")
    left["decontamination"].pop("at"), right["decontamination"].pop("at")
    assert left == right
    dropped = json.loads(first["dedupe.json"])["counts"]["dropped"]
    assert dropped >= 3, "a run that drops nothing cannot demonstrate a stable survivor set"


def test_the_decontamination_window_and_the_dedupe_window_are_different_constants():
    """Not one shared knob: 13 grams answers 'did this text appear inside
    that one', 5 answers 'are these two rows the same example'."""
    assert (NGRAM, DECON_NGRAM) == (5, 13)


def _scattered(text: str, k: int, seed: int = 7) -> str:
    """`text` with `k` substitutions spread evenly through it.

    A TAIL variant (the `variant` helper above) barely moves with the window -
    the two texts differ at one seam, so only ~n grams change whatever n is.
    Scattered edits are what the window size actually decides about, because
    each one destroys n grams.
    """
    rng = random.Random(seed)
    words = text.split()
    step = len(words) / (k + 1)
    for i in range(k):
        words[int(step * (i + 1))] = "xreplacedx" + str(rng.randrange(10_000))
    return " ".join(words)


def test_the_five_gram_window_decides_in_both_directions():
    """The window had no behavioural pin: its only killer was a test naming
    the constant, which is satisfied by any implementation that also names it.

    Two pairs of prompts either side of PROMPT_JACCARD AT FIVE, chosen so that
    a WIDER window would keep the first pair and a NARROWER one would drop the
    second - a wider window is more fragile under scattered edits and a
    narrower one more forgiving, so a fixture that only drops can never pin
    the narrowing direction.
    """
    base = prose(1001, 400)
    dropped = _scattered(base, 6)     # J = 0.859 at 5, 0.833 at 6
    survives = _scattered(base, 7)    # J = 0.868 at 4, 0.838 at 5
    measured = {
        n: (jaccard(gram_hashes(tokens(base), n), gram_hashes(tokens(dropped), n)),
            jaccard(gram_hashes(tokens(base), n), gram_hashes(tokens(survives), n)))
        for n in (4, NGRAM, 6)
    }
    assert measured[NGRAM][0] >= PROMPT_JACCARD > measured[6][0], "a wider window keeps the pair"
    assert measured[4][1] >= PROMPT_JACCARD > measured[NGRAM][1], "a narrower window drops it"

    # ... and the whole rows are far enough apart that rule 3 is not what is
    # deciding either of these.
    for other in (dropped, survives):
        assert jaccard(grams(f"{base}\nanswer one"), grams(f"{other}\nanswer two")) < ROW_JACCARD

    kept, drops, _ = dedupe_items(items(row(base, "answer one", form="f"),
                                        row(dropped, "answer two", form="f")))
    assert len(kept) == 1 and drops[0]["reason"] == REASON_NEAR_PROMPT

    kept, drops, _ = dedupe_items(items(row(base, "answer one", form="f"),
                                        row(survives, "answer two", form="f")))
    assert len(kept) == 2 and not drops
