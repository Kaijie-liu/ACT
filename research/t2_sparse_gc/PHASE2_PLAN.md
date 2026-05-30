# Phase 2 plan — after GPU autosweep completes

## Trigger
After `d_gpu_autosweep_<stamp>/MAIN.log` shows "ALL DONE", analyze
each `(bench, baseline vs d_filter)` pair.

## Decision criteria
For each benchmark, compare V+A counts:
- D wins: D's (V+A) > baseline's (V+A) by ≥ 1
- D neutral: same V+A, but D shorter mean wall (≥ 20% reduction)
- D no-effect: same V+A, same wall
- D regression: D fewer V+A than baseline (unlikely given proven sound)

## Expansion criteria (Phase 2)
Only expand benchmarks where D wins or D-neutral-faster on Phase 1.

| Tier | Action | Cost |
|---|---|---|
| D-wins | Full benchmark sweep with D ON, GPU | ~30 min - 2 hr per |
| D-neutral | Larger sample to verify wall reduction is real | ~30 min |
| D-no-effect | Skip — record as not-applicable | 0 |

## Resource budget
- User nap: 2 hr
- Auto-launch only if total estimated time < 2 hr
- Otherwise: stop, prepare clean report for user decision

## Currently in flight
- d_gpu_autosweep: 8 benches × 2 modes ≈ 90-120 min ETA
- metaroom CPU 100 iids: ~3 hr ETA, 14/14 CERT so far
- tinyimagenet long-wall: ~70 min ETA
- metaroom GPU B3 v2 quick sanity: ~10 min

## Synthesis report location
`/data1/Kane/ACT/research/t2_sparse_gc/AUTO_REPORT_<stamp>.md` —
will be auto-generated when autosweep ALL DONE detected.
