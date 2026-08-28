"""Export a human-labellable anchor set. READ-ONLY on every store it touches.

Selection is the whole failure population, not a sample: every generation
underlying a grounding<=3 judgement, plus the tiebreak-derived accepts (the
rows the saturated arbiter waved through), plus a few clean accepts as
control. Zero sampling error over the population that matters.
"""
import argparse, json, sqlite3, sys
from pathlib import Path

# The venv's editable install already puts src/ on sys.path (see the
# _editable_impl_tuned.pth in site-packages), but this script is also
# runnable from a checkout that only ever did `PYTHONPATH=src ...`, so the
# fallback is kept explicit rather than assumed.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path and (_SRC / "tuned").is_dir():
    sys.path.insert(0, str(_SRC))

from tuned.data import generate, prompt_registry  # noqa: E402

SELECT_FAILING = """
select distinct g.gen_id from judgement j join generation g on g.gen_id = j.gen_id
where j.grounding <= 3
"""
SELECT_TIEBREAK = """
select distinct g.gen_id from judgement j join generation g on g.gen_id = j.gen_id
where j.judge_slot = 'tiebreak'
"""
SELECT_CONTROL = """
select distinct g.gen_id from judgement j join generation g on g.gen_id = j.gen_id
where j.grounding = 5 limit 3
"""

# What replaces a MATERIAL slot (GROUNDING_SLOTS, plus transition's
# {scenario}) before the generator's prompt is rendered for the labeller.
# seed_text is already its own key in the exported row - the labeller needs
# the ask, not a second copy of the materials the generator was handed.
REDACTED_MATERIAL = "[omitted here - see seed_text]"


def render_task_instruction(prompt_id, task_type, seed_id, seed_text, case_type, meta_json):
    """The RENDERED instruction the generator actually saw for this row,
    materials redacted - not the raw template, and not task_type again.

    Rendering goes through the real generator path (generate.build_slots +
    prompt_registry.render) so paraphrase, focus_issue, document_kind and
    the rest come out exactly as the teacher read them, word for word,
    minus the seed material this export already carries separately.
    build_slots takes a `cfg` argument but never reads it - grep confirms
    the body never references it - so this calls it with None rather than
    constructing or importing a build config; that is what keeps a full
    build config out of an export script, per the brief's own test for
    whether the rendered form is worth shipping.

    Falls back to the raw template text (system + user, unresolved
    {slot} placeholders, prompt_id noted) when a seed cannot actually
    render for this task type - e.g. a statute_qa seed with no distinct
    section_text, or a transition seed missing a required date. That
    fallback is visible (mode="template" in the caller), never silent.
    """
    task_row = {"task_type": task_type, "seed_id": seed_id}
    seed_row = {"text": seed_text, "case_type": case_type, "meta_json": meta_json}
    try:
        slots = generate.build_slots(None, task_row, seed_row)
        redacted = dict(slots)
        for name in (*generate.GROUNDING_SLOTS, "scenario"):
            if redacted.get(name):
                redacted[name] = REDACTED_MATERIAL
        messages = prompt_registry.render(prompt_id, **redacted)
        text = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
        return text, "rendered"
    except (generate.SlotError, KeyError) as exc:
        template = prompt_registry.load(prompt_id)
        parts = []
        if template.system:
            parts.append(f"[system]\n{template.system}")
        parts.append(f"[user]\n{template.user}")
        parts.append(
            f"[fallback: {prompt_id!r} could not render for seed {seed_id!r} "
            f"({exc}); raw template shown instead]"
        )
        return "\n\n".join(parts), "template"


def build_rows(conn, *, limit=40):
    ids = []
    for sql in (SELECT_FAILING, SELECT_TIEBREAK, SELECT_CONTROL):
        for (gen_id,) in conn.execute(sql):
            if gen_id not in ids:
                ids.append(gen_id)
    rows = []
    for gen_id in ids[:limit]:
        gen = conn.execute(
            "select gen_id, task_id, think, answer from generation where gen_id=?",
            (gen_id,),
        ).fetchone()
        task = conn.execute(
            "select task_type, seed_id, prompt_id from task where task_id=?",
            (gen[1],),
        ).fetchone()
        seed = conn.execute(
            "select text, case_type, meta_json from seed where seed_id=?",
            (task[1],),
        ).fetchone()
        seed_text = seed[0] if seed else ""
        case_type = seed[1] if seed else None
        meta_json = seed[2] if seed else None
        instruction, source_mode = render_task_instruction(
            task[2], task[0], task[1], seed_text, case_type, meta_json
        )
        rows.append({
            "gen_id": gen[0],
            "task_type": task[0],
            "task_instruction": instruction,
            "task_instruction_source": source_mode,
            "prompt_id": task[2],
            "seed_text": seed_text,
            "think": gen[2],
            "answer": gen[3],
            "asserts_false": None,
            "asserts_unsupported": None,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()
    conn = sqlite3.connect(f"file:{Path(args.store).as_posix()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    rows = build_rows(conn, limit=args.limit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
