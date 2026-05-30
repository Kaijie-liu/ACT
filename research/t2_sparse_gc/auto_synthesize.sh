#!/bin/bash
# Auto-synthesize all running B3/D experiments into a single report.
# Run when user wakes from nap.
set -u

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
REPORT="/data1/Kane/ACT/research/t2_sparse_gc/AUTO_REPORT_${STAMP}.md"

echo "# Auto-synthesized results — ${STAMP}" > "$REPORT"
echo "" >> "$REPORT"
date >> "$REPORT"
echo "" >> "$REPORT"

# === GPU autosweep ===
echo "## D filter GPU autosweep (Phase 1)" >> "$REPORT"
echo "" >> "$REPORT"
GPU_SWEEP_DIR=$(ls -td /data1/Kane/ACT/audit_results/d_gpu_autosweep_*/ 2>/dev/null | head -1)
if [ -n "$GPU_SWEEP_DIR" ]; then
    /data1/Kane/miniconda3/envs/act-py312/bin/python <<EOF >> "$REPORT"
import json, glob, os
from collections import defaultdict

root = "${GPU_SWEEP_DIR%/}"
results = defaultdict(dict)
for sub in sorted(os.listdir(root)):
    full = os.path.join(root, sub)
    if not os.path.isdir(full): continue
    if '_baseline' in sub:
        bench = sub.replace('_baseline', '')
        mode = 'baseline'
    elif '_d_filter' in sub:
        bench = sub.replace('_d_filter', '')
        mode = 'd_filter'
    else:
        continue
    fs = sorted([f for f in glob.glob(full + '/per_instance_*.json') if 'watchdog' not in f])
    cert = fal = unk = to = rss = err = 0
    walls = []
    for fname in fs:
        try:
            d = json.load(open(fname))
            for p in d.get('per_instance', []):
                v = p.get('cli_normalized','?')
                if v == 'CERTIFIED': cert += 1
                elif v == 'FALSIFIED': fal += 1
                elif v == 'UNKNOWN': unk += 1
                elif v == 'UNKNOWN_TIMEOUT': to += 1
                elif v == 'UNKNOWN_RESOURCE_LIMIT': rss += 1
                else: err += 1
                if p.get('wall_s'): walls.append(float(p['wall_s']))
                break
        except: pass
    mw = sum(walls)/max(len(walls),1)
    n = cert+fal+unk+to+rss+err
    results[bench][mode] = {'cert': cert, 'fal': fal, 'unk': unk, 'to': to,
                            'rss': rss, 'err': err, 'mean_wall': mw, 'n': n}

print("| Benchmark | mode | V | A | U | TO | RSS | E | n | mean_wall |")
print("|---|---|---|---|---|---|---|---|---|---|")
winners = []
for bench in sorted(results.keys()):
    for mode in ('baseline', 'd_filter'):
        if mode not in results[bench]: continue
        r = results[bench][mode]
        print(f"| {bench} | {mode} | {r['cert']} | {r['fal']} | {r['unk']} | {r['to']} | {r['rss']} | {r['err']} | {r['n']} | {r['mean_wall']:.1f}s |")
    # Check for winner
    if 'baseline' in results[bench] and 'd_filter' in results[bench]:
        b = results[bench]['baseline']
        d = results[bench]['d_filter']
        b_va = b['cert'] + b['fal']
        d_va = d['cert'] + d['fal']
        if d_va > b_va:
            winners.append((bench, b_va, d_va))
print()
if winners:
    print("**D filter WINNERS** (V+A increased):")
    for bench, bva, dva in winners:
        print(f"- {bench}: baseline V+A = {bva} → D V+A = {dva} (+{dva-bva})")
else:
    print("**No benchmark showed D > baseline in V+A**")
EOF
else
    echo "(autosweep not started or directory not found)" >> "$REPORT"
fi
echo "" >> "$REPORT"

# === Metaroom CPU full ===
echo "## Metaroom CPU full sweep (B3 v2)" >> "$REPORT"
echo "" >> "$REPORT"
META_DIR=$(ls -td /data1/Kane/ACT/audit_results/metaroom_full_b3_v2_*/ 2>/dev/null | head -1)
if [ -n "$META_DIR" ]; then
    /data1/Kane/miniconda3/envs/act-py312/bin/python <<EOF >> "$REPORT"
import json, glob
from collections import Counter
fs = sorted([f for f in glob.glob("${META_DIR%/}/per_instance_*.json") if 'watchdog' not in f])
c = Counter(); cert_iids = []
walls = []
for f in fs:
    try:
        d = json.load(open(f))
        for p in d.get('per_instance', []):
            v = p.get('cli_normalized', '?')
            c[v] += 1
            if v == 'CERTIFIED':
                cert_iids.append(p.get('official_instance_id', p.get('instance_index')))
            if p.get('wall_s'): walls.append(float(p['wall_s']))
            break
    except: pass
print(f"- Done: {len(fs)}/100 iids")
print(f"- Verdicts: {dict(c)}")
print(f"- mean wall: {sum(walls)/max(len(walls),1):.0f}s")
print(f"- CERT iids: {cert_iids[:20]}{'...' if len(cert_iids) > 20 else ''} (total {len(cert_iids)})")
print()
print(f"Baseline r93 CPU metaroom: 37 CERT + 60 RSS-bound + 2 TO + 1 UNK")
print(f"B3 v2 result: {len(cert_iids)} CERT  (+{len(cert_iids) - 37} vs baseline so far)")
EOF
fi
echo "" >> "$REPORT"

# === Tinyimagenet long-wall ===
echo "## Tinyimagenet CPU long-wall (10 iids @ wall=600s)" >> "$REPORT"
echo "" >> "$REPORT"
TINY_DIR=$(ls -td /data1/Kane/ACT/audit_results/step1_tiny_longwall_*/ 2>/dev/null | head -1)
if [ -n "$TINY_DIR" ]; then
    /data1/Kane/miniconda3/envs/act-py312/bin/python <<EOF >> "$REPORT"
import json, glob
from collections import Counter
fs = sorted([f for f in glob.glob("${TINY_DIR%/}/per_instance_*.json") if 'watchdog' not in f])
c = Counter(); walls = []
for f in fs:
    try:
        d = json.load(open(f))
        for p in d.get('per_instance', []):
            c[p.get('cli_normalized', '?')] += 1
            if p.get('wall_s'): walls.append(float(p['wall_s']))
            break
    except: pass
print(f"- Done: {len(fs)}/10 iids")
print(f"- Verdicts: {dict(c)}")
print(f"- mean wall: {sum(walls)/max(len(walls),1):.0f}s")
EOF
fi

echo "Auto-report written: $REPORT"
