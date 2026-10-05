"""Per-family table for a single-path replay directory (markdown): baseline solves, retained,
lost (by kind), CERT gains, ADV gains with witness sources, conflicts.  Usage:
    python summarize_final.py n039_full_replay_v14 > results/final_table_v14.md"""
import collections, glob, json, sys

d = sys.argv[1]
rows = []
for f in sorted(glob.glob(f"{d}/w*.jsonl")):
    for l in open(f):
        rows.append(json.loads(l))
by = collections.defaultdict(list)
for r in rows:
    by[r["family"]].append(r)
tot = collections.Counter()
print(f"| family | rows | baseline CERT+ADV | retained | lost | CERT gains | ADV gains (sources) | conflicts |")
print("|---|---:|---:|---:|---|---:|---|---:|")
for fam in sorted(by):
    rs = by[fam]
    base = [r for r in rs if r["baseline"] in ("CERT", "ADV")]
    kept = [r for r in base if r["outcome"] == r["baseline"]]
    lost = collections.Counter(f"{r['baseline']}->{r['outcome']}" for r in base if r["outcome"] != r["baseline"])
    cg = [r for r in rs if r["baseline"] not in ("CERT", "ADV") and r["outcome"] == "CERT"]
    ag = [r for r in rs if r["baseline"] not in ("CERT", "ADV") and r["outcome"] == "ADV"]
    src = collections.Counter(r.get("witness_source") for r in ag)
    conf = sum(1 for r in rs if r.get("conflict"))
    print(f"| {fam} | {len(rs)} | {len(base)} | {len(kept)} | {', '.join(f'{k} {v}' for k, v in lost.items()) or '-'} | "
          f"{len(cg)} | {len(ag)} ({', '.join(f'{k} {v}' for k, v in src.items()) or '-'}) | {conf} |")
    tot.update(rows=len(rs), base=len(base), kept=len(kept), lost=sum(lost.values()), cg=len(cg), ag=len(ag), conf=conf)
print(f"| **total** | {tot['rows']} | {tot['base']} | {tot['kept']} | {tot['lost']} | {tot['cg']} | {tot['ag']} | {tot['conf']} |")
