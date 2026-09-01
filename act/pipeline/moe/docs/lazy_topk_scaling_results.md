# Lazy top-k E-scaling result

Status: completed and independently audited on 2026-09-01.

The frozen study crossed `E={4,8,16,32,64}`, three deterministic router
families, and partial-MIP-start off/on. All 30 conditions completed. Every
`E<=8` result matched exhaustive enumeration, every complete paired condition
returned identical route sets, and the independent audit reported zero
issues.

## Scaling result

For the no-start condition, the all-tied top-2 family enumerated every legal
pair and took:

| E | legal sets | seconds |
|---:|---:|---:|
| 4 | 6 | 0.0053 |
| 8 | 28 | 0.0569 |
| 16 | 120 | 0.4719 |
| 32 | 496 | 3.6897 |
| 64 | 2,016 | 50.2289 |

At E=64, the strictly stable family completed in 0.0616 seconds with one
route set, whereas the fixed random-affine family took 50.0417 seconds for its
reachable family. The experiment therefore supports a scaling statement in
terms of the number of feasible route sets, not expert count alone. The
all-tied family remains an explicit combinatorial worst case and is not a
natural-model prevalence estimate.

## Partial MIP-start result

HiGHS accepted the partial start submissions, but the public interface does
not expose whether it used them internally. Across the 15 paired instances,
the median with-start/no-start wall-time ratio was `1.1280` (range
`0.9978--1.4265`). Thus no performance benefit was observed; starts were
typically slower. The result is attributed only to the measured condition,
not to an unobservable solver mechanism. The retained implementation can keep
submission disabled by default while preserving incremental model and cut
reuse.

## Frozen evidence

- Config: `act/pipeline/moe/configs/lazy_topk_scaling_r1.json`
- Raw rows SHA-256: `032f9fe670f9337c2174cfe0ab36cc264a2d0c00d415b356b37049726dfb78f5`
- Raw summary SHA-256: `c75edc97b45efb1af5a3e054e00f988747a6d27326bc539f974b8729e1ca0785`
- Independent audit:
  `act/pipeline/moe/results/lazy_topk_scaling_20260901_r1_audit.json`

Raw runtime artifacts remain under
`data/moe/results/lazy_topk_scaling_r1` and are never committed as source.
