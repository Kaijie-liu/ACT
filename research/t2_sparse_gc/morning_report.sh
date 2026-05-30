#!/bin/bash
# Synthesize morning report from overnight sweep results.
# Run after overnight_sweep.sh completes.
set -u

SWEEP_DIR=$(ls -td /data1/Kane/ACT/audit_results/overnight_b3_*/ 2>/dev/null | head -1)
if [ -z "$SWEEP_DIR" ]; then
    echo "No sweep directory found"
    exit 1
fi
echo "Sweep dir: $SWEEP_DIR"

/data1/Kane/miniconda3/envs/act-py312/bin/python <<EOF
import json, glob, os
from collections import Counter, defaultdict

sweep = "$SWEEP_DIR".rstrip('/')
results = defaultdict(lambda: defaultdict(dict))  # bench -> mode -> {verdicts, rss_list, wall_list}

for sub in sorted(os.listdir(sweep)):
    if not os.path.isdir(os.path.join(sweep, sub)): continue
    parts = sub.split('_')
    # bench_mode pattern: cifar100_2024_t2b_only or tinyimagenet_2024_b3_full_kmax500
    if sub.startswith('cifar100_2024_'):
        bench = 'cifar100_2024'
        mode = sub[len('cifar100_2024_'):]
    elif sub.startswith('tinyimagenet_2024_'):
        bench = 'tinyimagenet_2024'
        mode = sub[len('tinyimagenet_2024_'):]
    else:
        continue
    ws_path = os.path.join(sweep, sub, 'watchdog_summary.json')
    if not os.path.exists(ws_path): continue
    try:
        ws = json.load(open(ws_path))
    except: continue
    verdicts = Counter()
    rsses = []
    walls = []
    iid_verdicts = {}
    for r in ws.get('results', []):
        v = r.get('cli_normalized', '?')
        verdicts[v] += 1
        if r.get('peak_rss_mb'):
            rsses.append(float(r['peak_rss_mb']))
        if r.get('wall_s'):
            walls.append(float(r['wall_s']))
        iid_verdicts[r.get('instance_id')] = v
    results[bench][mode] = {
        'verdicts': dict(verdicts),
        'mean_rss': sum(rsses)/max(len(rsses),1),
        'mean_wall': sum(walls)/max(len(walls),1),
        'n': len(rsses),
        'iid_verdicts': iid_verdicts,
    }

print("=" * 88)
print("Overnight B3 sweep results")
print("=" * 88)
for bench, modes in sorted(results.items()):
    print(f"\n## {bench}")
    print(f"{'mode':30s}  {'verdicts':40s}  {'mean_RSS':>10s}  {'mean_wall':>10s}")
    print("-" * 95)
    for mode in ['t2b_only', 'b3_full_kmax500', 'b3_full_kmax2000', 'b3_compact_kmax2000', 'b3_compact_kmax5000']:
        if mode not in modes: continue
        r = modes[mode]
        vs = str(r['verdicts'])[:38]
        print(f"{mode:30s}  {vs:40s}  {r['mean_rss']:>10.0f}  {r['mean_wall']:>10.1f}")
    # Show per-iid for t2b_only vs best b3
    if 't2b_only' in modes:
        print(f"  per-iid (t2b_only vs others):")
        iids = sorted(set().union(*[m['iid_verdicts'].keys() for m in modes.values()]))
        for iid in iids:
            row = [f"iid={iid}"]
            for mode in ['t2b_only', 'b3_full_kmax500', 'b3_compact_kmax2000']:
                v = modes.get(mode, {}).get('iid_verdicts', {}).get(iid, '-')
                row.append(f"{mode}={v}")
            print("    " + ' | '.join(row))
EOF

echo ""
echo "Report saved at $SWEEP_DIR/log.txt"
