"""Export a human-labellable anchor set. READ-ONLY on every store it touches.

Selection is the whole failure population, not a sample: every generation
underlying a grounding<=3 judgement, plus the tiebreak-derived accepts (the
rows the saturated arbiter waved through), plus a few clean accepts as
control. Zero sampling error over the population that matters.
"""
import argparse, json, sqlite3
from pathlib import Path

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
            "select task_type, seed_id from task where task_id=?", (gen[1],)
        ).fetchone()
        seed = conn.execute(
            "select text from seed where seed_id=?", (task[1],)
        ).fetchone()
        rows.append({
            "gen_id": gen[0],
            "task_type": task[0],
            "task_instruction": task[0],
            "seed_text": seed[0] if seed else "",
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
