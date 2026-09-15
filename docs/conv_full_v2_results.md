# Convolutional full V2: reviewed final results

Execution `2d2477e4bd3ff5f2622e17577136a49fd45bd947` completed all 90
requests on the original 30 frozen inputs. No retry, replacement, budget,
support, 25% scheduling or numerical-gate change occurred. Read the immutable
`docs/conv_full_v2.md` for the protocol. R1 remains a failed smoke; V2 does
not relabel its results. The seed17 convolutional model selected at epoch89
has 67.06% full-test clean accuracy; epsilon is 2/255, each request capped at
300 seconds, both ACT arms bound to the same V2 budget contract.

| Arm (30 requests each) | HZ-policy SAFE | CROWN numerical positive | Replayed UNSAFE | Complete UNKNOWN | Internal TIMEOUT | Outer timeout | Mean / median seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Adaptive | 0 | — | 17 | 0 | 13 | 0 | 188.31 / 153.25 |
| Matched monolithic | 0 | — | 11 | 0 | 19 | 0 | 217.50 / 295.63 |
| ACT routing + static weighted plain CROWN | — | 1 | 7 | 22 | 0 | 0 | 4.40 / 4.35 |

Automatic, separate-process and fresh archival audits agree exactly: PASS,
zero issues, 90 complete records, 30/30 common-fact pairs equal and 35 full-model
UNSAFE replays. The replay count comprises method runs, covering 18 distinct
inputs. Seventeen inputs have one legal pair and thirteen have multiple pairs.
Launch through automatic audited completion took 3.739 hours, including
resource waits and audits. Means above charge all requests, including internal
timeouts; external resource waits and post-run audits are reported separately
in the archive, not silently added to one method's request time.

Adaptive gains six solved inputs over matched (indices 8,20,47,56,69,75),
all UNSAFE, with no matched-only solved input. There is **no SAFE increment**.
The paired mean time difference is -29.19 seconds but its median is +0.014
seconds: this is not a uniform speedup. The degenerate SAFE bootstrap interval
[0,0] is not evidence of population equivalence. This full run does not
establish a cross-architecture certificate advantage.

CROWN's positive is index98, a single-pair request {1,2}; both ACT arms time
out there. It remains a numerical filter, not formal SAFE. Conversely, CROWN
finds a replayed witness at index113 where both ACT arms time out. The six
adaptive-only solved requests above therefore do not imply dominance over
the external path. No method outcome is retroactively relabelled using
another arm's result. Negative bounds alone are not counterexamples.

## Reproducible archive

`act/pipeline/moe/results/conv_full_v2_review_20260915.json` includes the
complete descriptive comparisons, 90 terminal rows, distinct-witness context,
and a byte-size/SHA-256 inventory of every frozen raw artifact. Raw checkpoints,
input tensors and external repositories are not committed. Reconstruct using:

```
/data1/Kane/miniconda3/envs/act-py312/bin/python -m scripts.review_conv_full_v2 --check
```

This runs the frozen identity/journal/package/witness auditor again, not model
verification searches. Structural audit is not independent proof of SAFE
bounds. Next, separately analyze saved property/native-call journals: an
internal TIMEOUT is not evidence that completed relaxation crosses zero, nor
that extra time would prove the property. Do not modify the frozen full-run
scripts, raw files or protocol while producing that derived analysis.
