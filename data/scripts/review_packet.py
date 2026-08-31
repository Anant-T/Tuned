"""Render the legal-accuracy review packet the dataset card requires.

    python data/scripts/review_packet.py --state state.sqlite3 --out packet.html

The card makes a human read of 50 accepted examples a ship prerequisite - "the
only legal-accuracy check in this pipeline" - because every automated gate
scores FORM. Row 412b8d1c5430 passed twelve gates and a judge while concluding
the opposite of its own answer key; that is the class of defect this packet
exists to surface, and no gate will ever catch it.

Read-only. Opens the store with mode=ro and writes nothing but the HTML, so it
is safe to point at a pulled copy of the baton while CI holds the real one.
"""
import argparse
import html
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tuned.data import review  # noqa: E402
from tuned.data.decontaminate import answer_without_preamble  # noqa: E402

# The trimmed preamble is not deleted evidence - it moves into the trace, where
# the reader can still see everything the teacher wrote.
CUT = "\n\n--- cut from the head of the answer ---\n\n"

STYLE = """
:root { --bg:#fbfaf8; --fg:#1a1a19; --dim:#6b6a66; --line:#e3e0da; --card:#fff;
        --warn:#8a5a00; --warnbg:#fff5e0; --bad:#8a1f1f; --badbg:#fdeaea;
        --ok:#2c6a4a; --okbg:#e9f5ee; }
@media (prefers-color-scheme: dark) { :root:not([data-theme=light]) {
  --bg:#16161a; --fg:#e8e6e1; --dim:#9a9791; --line:#2e2e34; --card:#1d1d22;
  --warn:#e0b064; --warnbg:#3a2e14; --bad:#f0a0a0; --badbg:#3a1c1c;
  --ok:#8fd6b0; --okbg:#173327; } }
:root[data-theme=dark] { --bg:#16161a; --fg:#e8e6e1; --dim:#9a9791; --line:#2e2e34;
  --card:#1d1d22; --warn:#e0b064; --warnbg:#3a2e14; --bad:#f0a0a0; --badbg:#3a1c1c;
  --ok:#8fd6b0; --okbg:#173327; }
body { background:var(--bg); color:var(--fg);
       font:15px/1.6 -apple-system,Segoe UI,system-ui,sans-serif;
       max-width:920px; margin:0 auto; padding:2rem 1.2rem 8rem; }
h1 { font-size:1.5rem; margin:0 0 .3rem; }
.lede { color:var(--dim); margin:0 0 1.6rem; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
        padding:1.1rem 1.2rem; margin:0 0 1.4rem; }
.card header { display:flex; gap:.5rem; align-items:baseline; flex-wrap:wrap; }
.n { font-weight:700; }
.tag { font-size:.75rem; background:var(--line); padding:.12rem .5rem; border-radius:99px; }
.tag.alt { background:transparent; border:1px solid var(--line); color:var(--dim); }
.src { font-size:.72rem; color:var(--dim); margin-left:auto;
       font-family:ui-monospace,monospace; }
.meta { color:var(--dim); font-size:.85rem; margin:.4rem 0 0; }
.flags { margin:.6rem 0 .2rem; display:flex; gap:.4rem; flex-wrap:wrap; }
.flag { font-size:.75rem; padding:.15rem .5rem; border-radius:5px;
        background:var(--warnbg); color:var(--warn); }
.flag.ok { background:var(--okbg); color:var(--ok); }
.flag.lowscore, .flag.audit, .flag.unsourced { background:var(--badbg); color:var(--bad); }
h3 { font-size:.9rem; margin:1rem 0 .3rem; text-transform:uppercase;
     letter-spacing:.04em; color:var(--dim); }
.m { color:var(--dim); font-weight:400; text-transform:none; letter-spacing:0; }
.answer, .think, .seed { white-space:pre-wrap; overflow-wrap:anywhere; }
.think, .seed { font-size:.88rem; color:var(--dim); max-height:26rem; overflow:auto;
                border-left:2px solid var(--line); padding-left:.8rem; }
details { margin:.5rem 0; } summary { cursor:pointer; color:var(--dim); font-size:.85rem; }
.j p { margin:.2rem 0 .8rem; color:var(--dim); font-size:.9rem; }
.verdict { margin-top:1rem; padding-top:.8rem; border-top:1px solid var(--line);
           display:flex; gap:.9rem; flex-wrap:wrap; align-items:center; font-size:.86rem; }
.note { flex:1 1 100%; padding:.4rem .6rem; border:1px solid var(--line);
        border-radius:6px; background:transparent; color:var(--fg); }
#bar { position:fixed; left:0; right:0; bottom:0; background:var(--card);
       border-top:1px solid var(--line); padding:.7rem 1.2rem; display:flex;
       gap:1rem; align-items:center; font-size:.87rem; }
button { font:inherit; padding:.35rem .9rem; border-radius:6px;
         border:1px solid var(--line); background:transparent; color:var(--fg);
         cursor:pointer; }
#out { width:100%; height:9rem; margin-top:1rem; font-family:ui-monospace,monospace;
       font-size:.8rem; background:var(--card); color:var(--fg);
       border:1px solid var(--line); border-radius:6px; padding:.6rem; }
"""

SCRIPT = """
var N = %(n)d;
var ids = %(ids)s;
function tally() {
  var n = 0;
  for (var i = 1; i <= N; i++) {
    if (document.querySelector('input[name=v' + i + ']:checked')) n++;
  }
  document.getElementById('count').textContent = n + ' / ' + N + ' reviewed';
}
document.addEventListener('change', tally);
function dump() {
  var out = [];
  for (var i = 1; i <= N; i++) {
    var v = document.querySelector('input[name=v' + i + ']:checked');
    var note = document.querySelector('.note[data-row="' + i + '"]').value;
    out.push({row: i, task_id: ids[i - 1], verdict: v ? v.value : null,
              note: note || null});
  }
  document.getElementById('out').value = JSON.stringify(out, null, 1);
}
"""

CARD = """
<article class=card id="row{i}">
  <header>
    <span class=n>{i}/{n}</span>
    <span class=tag>{stream}</span>
    <span class=tag>{ttype}</span>
    <span class="tag alt">{prompt}</span>
    <span class=src>{source} &middot; {task_id}</span>
  </header>
  {meta_html}
  <div class=flags>{flags}</div>

  <h3>Answer <span class=m>({model}, as it will ship)</span></h3>
  <div class=answer>{answer}</div>

  <details><summary>Reasoning trace ({ttok} tok){cutnote}</summary>
    <div class=think>{think}</div></details>
  <details><summary>Source document ({stok} tok)</summary>
    <div class=seed>{seedtext}</div></details>
  <details><summary>Judge</summary>{judge}</details>

  <div class=verdict>
    <label><input type=radio name="v{i}" value=sound> Legally sound</label>
    <label><input type=radio name="v{i}" value=minor> Minor error (ships with a note)</label>
    <label><input type=radio name="v{i}" value=wrong> Wrong law, wrong section, or invented authority</label>
    <label><input type=radio name="v{i}" value=unsure> Cannot judge from this material</label>
    <input class=note type=text placeholder="what is wrong, and where" data-row="{i}">
  </div>
</article>"""

LEDE = (
    "{n} accepted examples, stratified over every stream and task type with a floor of "
    "{floor} per cell so the small streams are not washed out by a proportional draw. "
    "Every automated gate has already passed on all of them; what is left is the "
    "judgement no gate makes, which is whether well-formed legal reasoning is "
    "<em>right</em>. The dataset card names this read as the only legal-accuracy check "
    "in the pipeline. Flags mark where the machine recorded a doubt it was not allowed "
    "to act on, or where an authority appears that the source never named &mdash; start "
    "there."
)


def e(value):
    return html.escape(str(value if value is not None else ""))


def build(db, rows, floor):
    cards, ids, flagged = [], [], 0
    for i, task in enumerate(rows, 1):
        gen = db.execute(
            "SELECT * FROM generation WHERE task_id=? ORDER BY gen_id DESC LIMIT 1",
            (task["task_id"],),
        ).fetchone()
        seed = db.execute(
            "SELECT * FROM seed WHERE seed_id=?", (task["seed_id"],)
        ).fetchone()
        gates = db.execute(
            "SELECT gate, passed FROM gate_result WHERE gen_id=? ORDER BY gate",
            (gen["gen_id"],),
        ).fetchall()
        judged = db.execute(
            "SELECT judge_slot, model, grounding, validity, coverage, rationale "
            "FROM judgement WHERE gen_id=? ORDER BY judge_slot",
            (gen["gen_id"],),
        ).fetchall()

        shipped, dropped = answer_without_preamble(
            gen["answer"] or "", task["task_type"]
        )
        source_text = (seed["text"] if seed else "") or ""
        unsourced = review.unsourced_references(shipped, source_text)

        # Every ENFORCED gate passed here or the row would not be accepted. What
        # can still be failing is a DIAGNOSTIC gate - recorded, never enforced -
        # and that is exactly where the machine had a doubt it was not allowed
        # to act on. Send the reader there first.
        failed = [g["gate"] for g in gates if not g["passed"]]
        flags = []
        if failed:
            flags.append(("diagnostic", "diagnostic gate failed: " + ", ".join(failed)))
        if (task["disposition"] or "").startswith("audit:"):
            flags.append(("audit", "accepted on gates alone - no judge read this"))
        lows = [
            "{}:validity {}".format(j["judge_slot"], j["validity"])
            for j in judged
            if j["validity"] is not None and j["validity"] <= 3
        ]
        if lows:
            flags.append(("lowscore", "judge doubted the reasoning (" + ", ".join(lows) + ")"))
        if task["attempts"] and task["attempts"] > 1:
            flags.append(("retry", "accepted on attempt {}".format(task["attempts"])))
        refs = list(unsourced.sections) + list(unsourced.citations)
        if refs:
            flagged += 1
            # On transition this is expected by construction: the task is
            # which-enactment-governs, decided on BNS 358 / BNSS 531 / BSA 170,
            # which are cited from law and never appear in the source judgment.
            # Whether each LIMB matches seed.answer_key_json is the real check.
            why = (
                "the savings provisions - verify each limb against the key"
                if task["task_type"] == "transition"
                else "not in the source; check it is real"
            )
            flags.append(
                ("unsourced", "cites {} ({})".format(", ".join(refs[:6]), why))
            )
        if dropped:
            flags.append(
                (
                    "trim",
                    "{} chars of second deliberation cut from the answer "
                    "(shown with the trace)".format(dropped),
                )
            )

        meta = " &middot; ".join(
            x
            for x in [
                e(seed["court"]) if seed and seed["court"] else "",
                e(seed["neutral_citation"]) if seed and seed["neutral_citation"] else "",
                e(seed["decision_date"]) if seed and seed["decision_date"] else "",
                "code era " + e(seed["code_era"]) if seed and seed["code_era"] else "",
            ]
            if x
        )
        judge_html = "".join(
            "<div class=j><b>{}</b> <span class=m>{}</span> grounding {} &middot; "
            "validity {} &middot; coverage {}<p>{}</p></div>".format(
                e(j["judge_slot"]), e(j["model"]), e(j["grounding"]),
                e(j["validity"]), e(j["coverage"]), e(j["rationale"]),
            )
            for j in judged
        ) or (
            "<div class=j><em>No judge scored this row - it was accepted in audit "
            "mode, on the gates alone.</em></div>"
        )
        flag_html = "".join(
            '<span class="flag {}">{}</span>'.format(k, e(v)) for k, v in flags
        ) or '<span class="flag ok">no automated doubt recorded</span>'

        ids.append(task["task_id"])
        cards.append(
            CARD.format(
                i=i,
                n=len(rows),
                stream=e(task["stream"]),
                ttype=e(task["task_type"]),
                prompt=e(task["prompt_id"]),
                source=e(seed["source_id"]) if seed else "",
                task_id=e(task["task_id"]),
                meta_html="<div class=meta>" + meta + "</div>" if meta else "",
                flags=flag_html,
                model=e(gen["model"]),
                answer=e(shipped),
                ttok=e(gen["think_tokens"]),
                think=e(
                    (gen["think"] or "")
                    + (CUT + (gen["answer"] or "")[:dropped] if dropped else "")
                ),
                cutnote=(
                    " + the {} chars cut from the answer".format(dropped)
                    if dropped
                    else ""
                ),
                stok=e(seed["token_count"]) if seed else "?",
                seedtext=e(source_text) if seed else "(seed missing)",
                judge=judge_html,
            )
        )

    page = (
        "<!doctype html><meta charset=utf-8>"
        "<title>law_v1 legal review</title>"
        "<style>" + STYLE + "</style>"
        "<h1>law_v1 &mdash; legal accuracy review</h1>"
        "<p class=lede>" + LEDE.format(n=len(rows), floor=floor) + "</p>"
        + "".join(cards)
        + "<textarea id=out placeholder=\"verdicts appear here\"></textarea>"
        '<div id=bar><span id=count>0 / {} reviewed</span>'
        '<button onclick=dump()>Export verdicts</button></div>'.format(len(rows))
        + "<script>"
        + SCRIPT % {"n": len(rows), "ids": json.dumps(ids)}
        + "</script>"
    )
    return page, flagged


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, help="path to the build state sqlite3")
    parser.add_argument("--out", required=True, help="path to write the HTML packet to")
    parser.add_argument("--n", type=int, default=review.DEFAULT_SAMPLE)
    parser.add_argument(
        "--floor",
        type=int,
        default=review.DEFAULT_FLOOR,
        help="minimum rows per stream/task-type cell, so small streams survive the draw",
    )
    parser.add_argument(
        "--salt",
        default="law_v1-review",
        help="the draw is deterministic in this; keep it to re-render the SAME 50 "
        "rows after the corpus grows, change it to read a fresh sample",
    )
    args = parser.parse_args(argv)

    db = sqlite3.connect("file:{}?mode=ro".format(args.state), uri=True)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT task_id, seed_id, stream, task_type, prompt_id, disposition, attempts "
        "FROM task WHERE state='accepted' ORDER BY task_id"
    ).fetchall()
    if not rows:
        parser.error("no accepted rows in {}".format(args.state))

    sample = review.stratified_sample(
        [dict(r) for r in rows],
        n=args.n,
        floor=args.floor,
        key=lambda r: "{}/{}".format(r["stream"], r["task_type"]),
        salt=args.salt,
    )
    page, flagged = build(db, sample, args.floor)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")

    print("accepted rows: {}   drawn: {}".format(len(rows), len(sample)))
    cells = {}
    for row in sample:
        cells["{}/{}".format(row["stream"], row["task_type"])] = (
            cells.get("{}/{}".format(row["stream"], row["task_type"]), 0) + 1
        )
    for cell in sorted(cells):
        print("  {:<32}{}".format(cell, cells[cell]))
    print(
        "{} of {} cite an authority the source never names - read those first".format(
            flagged, len(sample)
        )
    )
    print("wrote {} ({:.1f} KB)".format(out, out.stat().st_size / 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
