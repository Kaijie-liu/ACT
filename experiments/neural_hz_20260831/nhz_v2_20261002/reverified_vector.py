"""Re-verified vector of N039 v14 after N113: a v14 CERT counts only if it was LP-only (stage B) or
reproduced by the path v7.2 re-run; ADVs count if S1-valid in the final witness audit.  Writes
results/reverified_v14_table.md and results/reverified_v14_summary.json."""
import json, glob, collections, os
D = os.path.dirname(os.path.abspath(__file__))
def milp_excluded(r):
    return any(m.get('excluded') for b in r.get('boxes', []) for m in (b.get('milp') or {}).values())
v14 = [json.loads(l) for f in sorted(glob.glob(f"{D}/n039_full_replay_v14/w*.jsonl")) for l in open(f)]
rev = {}
for w in ('wA', 'wB'):
    p = f"{D}/n113_reverify_v14_milp_certs/{w}.jsonl"
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l); rev[(r['family'], r['iid'])] = r['outcome']
wit = {}
p = f"{D}/results/final_n039_full_replay_v14_witness_audit.jsonl"
if os.path.exists(p):
    for l in open(p):
        r = json.loads(l); wit[(r['family'], r['iid'])] = r['S1_valid']
by = collections.defaultdict(lambda: collections.Counter()); withdrawn = []; pending = []
for r in v14:
    k = (r['family'], r['iid']); base = r['baseline'] in ('CERT', 'ADV'); out = r['outcome']
    final = out
    if out == 'CERT' and milp_excluded(r):
        if k not in rev: final = 'PENDING'; pending.append(k)
        elif rev[k] != 'CERT': final = 'WITHDRAWN'; withdrawn.append((k, r['baseline'], rev[k]))
    if out == 'ADV' and not wit.get(k, False): final = 'WITHDRAWN'; withdrawn.append((k, r['baseline'], 'witness'))
    f = by[r['family']]; f['rows'] += 1; f['base'] += base
    if final in ('CERT', 'ADV'):
        if base and final == r['baseline']: f['kept'] += 1
        elif not base: f['gain_' + final] += 1
    if final == 'WITHDRAWN': f['withdrawn'] += 1
    if final == 'PENDING': f['pending'] += 1
tot = collections.Counter()
lines = ["| family | rows | baseline solves | retained (re-verified) | CERT gains | ADV gains | withdrawn | pending |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
for fam in sorted(by):
    f = by[fam]; tot.update(f)
    lines.append(f"| {fam} | {f['rows']} | {f['base']} | {f['kept']} | {f['gain_CERT']} | {f['gain_ADV']} | {f['withdrawn']} | {f['pending']} |")
lines.append(f"| **total** | {tot['rows']} | {tot['base']} | {tot['kept']} | {tot['gain_CERT']} | {tot['gain_ADV']} | {tot['withdrawn']} | {tot['pending']} |")
open(f"{D}/results/reverified_v14_table.md", "w").write("\n".join(lines) + "\n")
json.dump({"retained": tot['kept'], "baseline": tot['base'], "cert_gains": tot['gain_CERT'], "adv_gains": tot['gain_ADV'],
           "withdrawn": withdrawn, "pending": pending}, open(f"{D}/results/reverified_v14_summary.json", "w"), indent=1, default=str)
print("\n".join(lines)); print("withdrawn:", withdrawn); print("pending:", len(pending))
