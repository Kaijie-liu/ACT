"""Summarise the N039 replay: per-family retention, losses, gains, conflicts, promotion gate."""
import collections
import glob
import json
import sys

rows = []
for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "n039_full_replay_v1/w*.jsonl")):
    rows += [json.loads(l) for l in open(f)]
SOLVED = ("CERT", "ADV")
fam = collections.defaultdict(lambda: collections.Counter())
gains, losses, conflicts = [], [], []
for r in rows:
    f = r["family"]; b = r["baseline"]; o = r["outcome"]
    fam[f]["rows"] += 1
    fam[f]["base_solved"] += b in SOLVED
    fam[f]["new_solved"] += o in SOLVED
    if b in SOLVED and o == b:
        fam[f]["retained"] += 1
    elif b in SOLVED:
        fam[f][f"lost_{b}->{o}"] += 1; losses.append((f, r["iid"], b, o))
    elif o in SOLVED:
        fam[f][f"gain_{o}"] += 1; gains.append((f, r["iid"], b, o, r.get("witness_source")))
    if r.get("conflict"):
        conflicts.append((f, r["iid"], b, o))
print(f"rows completed: {len(rows)} / 2413")
print(f"{'family':22s} {'rows':>5s} {'base':>5s} {'kept':>5s} {'new':>5s}  details")
tot = collections.Counter()
for f in sorted(fam):
    c = fam[f]
    det = {k: v for k, v in c.items() if k.startswith(("lost", "gain"))}
    print(f"{f:22s} {c['rows']:5d} {c['base_solved']:5d} {c['retained']:5d} {c['new_solved']:5d}  {det}")
    for k in ("rows", "base_solved", "retained", "new_solved"):
        tot[k] += c[k]
print(f"{'TOTAL':22s} {tot['rows']:5d} {tot['base_solved']:5d} {tot['retained']:5d} {tot['new_solved']:5d}")
print("conflicts:", conflicts)
print("gains:", gains)
gate = len(rows) == 2413 and not losses and not conflicts and gains
print("formal promotion gate:", "PASS" if gate else "FAIL", f"(losses {len(losses)}, conflicts {len(conflicts)}, gains {len(gains)})")
