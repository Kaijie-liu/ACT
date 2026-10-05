"""Post-run audits and final table for a single-path replay directory.
Usage: python finalize_replay.py n039_full_replay_v14 [--samples 2000] [--per-family 20]
  1. witness audit (independent VNNLIB reader, exact rationals, S1 and S2) of every ADV witness;
  2. CERT sampling audit of every CERT gain plus a fixed-seed sample of retained CERTs per family;
  3. per-family table (summarize_final.py) and a JSON summary with the gate verdict.
Audits never create verdicts; they can only invalidate them."""
import argparse, glob, json, os, random, subprocess, sys

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("d"); ap.add_argument("--samples", type=int, default=2000)
ap.add_argument("--per-family", type=int, default=20); a = ap.parse_args()
d = a.d.rstrip("/"); tag = os.path.basename(d)
os.makedirs(f"{HERE}/results", exist_ok=True)
rows = [json.loads(l) for f in sorted(glob.glob(f"{d}/w*.jsonl")) for l in open(f)]
# 1. witnesses
wit_out = f"{HERE}/results/final_{tag}_witness_audit.jsonl"
if not os.path.exists(wit_out):
    subprocess.run([PY, f"{HERE}/audit_n039_witness.py", "--glob", f"{d}/w*_witness/x_*.npy", "--out", wit_out], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
wit = {(r["family"], r["iid"]): r for r in (json.loads(l) for l in open(wit_out))} if os.path.exists(wit_out) else {}
# 2. CERT sample
random.seed(11)
certs = [r for r in rows if r["outcome"] == "CERT"]
gains = [r for r in certs if r["baseline"] not in ("CERT", "ADV")]
kept = {}
for r in certs:
    if r["baseline"] == "CERT":
        kept.setdefault(r["family"], []).append(r)
sample = list(gains) + [r for rs in kept.values() for r in random.sample(rs, min(a.per_family, len(rs)))]
sp = f"{HERE}/results/final_{tag}_cert_sample_rows.jsonl"
open(sp, "w").write("\n".join(json.dumps(r) for r in sample) + "\n")
cert_out = f"{HERE}/results/final_{tag}_cert_audit.jsonl"
if not os.path.exists(cert_out):
    subprocess.run([PY, f"{HERE}/audit_n039_cert_sampling.py", sp, "--samples", str(a.samples), "--out", cert_out], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
cert = [json.loads(l) for l in open(cert_out)] if os.path.exists(cert_out) else []
# 3. table + summary
table = subprocess.run([PY, f"{HERE}/summarize_final.py", d], capture_output=True, text=True).stdout
open(f"{HERE}/results/final_{tag}_table.md", "w").write(table)
adv = [r for r in rows if r["outcome"] == "ADV"]
s1_bad = [k for k, w in wit.items() if not w["S1_valid"]]
s2_bad = [k for k, w in wit.items() if not w["S2_valid"]]
missing = [(r["family"], r["iid"]) for r in adv if (r["family"], r["iid"]) not in wit]
base = [r for r in rows if r["baseline"] in ("CERT", "ADV")]
lost = [(r["family"], r["iid"], r["baseline"], r["outcome"]) for r in base if r["outcome"] != r["baseline"]]
summary = {"replay": tag, "rows": len(rows), "baseline_solves": len(base), "retained": len(base) - len(lost), "lost": lost,
           "conflicts": [(r["family"], r["iid"]) for r in rows if r.get("conflict")],
           "cert_gains": [(r["family"], r["iid"]) for r in gains],
           "adv_gains": [(r["family"], r["iid"], r.get("witness_source")) for r in adv if r["baseline"] not in ("CERT", "ADV")],
           "witness_audit": {"audited": len(wit), "S1_invalid": s1_bad, "S2_invalid_count": len(s2_bad), "unaudited_adv": missing},
           "cert_audit": {"audited": len(cert), "violations": sum(c["violations"] for c in cert),
                          "box_disagreements": [c["row"] for c in cert if not c["independent_box_agrees"]]},
           "gate": "PASS" if (not lost and not s1_bad and not missing and len(rows) == 2413
                              and not any(r.get("conflict") for r in rows)) else "FAIL"}
json.dump(summary, open(f"{HERE}/results/final_{tag}_summary.json", "w"), indent=1, default=str)
print(table); print(json.dumps({k: v for k, v in summary.items() if k not in ("lost", "cert_gains", "adv_gains")}, indent=1, default=str))
print("lost:", lost); print("CERT gains:", summary["cert_gains"]); print("ADV gains:", summary["adv_gains"])
