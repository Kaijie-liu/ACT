# C5 live / actual ReLU36 checkpoint

Read `C5_LIVE_AND_RELU36_AUDIT_20260905.md` first. Four completed runs are
`c5_live_transaction_20260905_v1`, `c5_live_transaction_20260905_v2`,
`c5_integrated_prefix_20260905_v1`, and `c5_integrated_prefix_20260905_v2`.
Failures are retained, not merged with successes. V2 integrated propagation
really has HZ at ReLU36; V1's loop label did not.

The additive `CHECKPOINT_C5_LIVE_RELU36_20260905_SHA256SUMS` covers all new
sources, prerequisites, tests, both live evidence files and all four result
directories, and links the five earlier immutable manifests. Verify all six
from this experiment directory. No score/default/production-source changes.

Next work: audit ReLU36 on a fresh live prefix with the same C5 V2 runtime,
all four sources and native stages. Do not treat a pre-ReLU20 qualification,
loop counter, or snapshot size as its complete physical/equivalence proof.
Only then proceed along the fixed deeper-layer and cohort campaign. Formal
1,870/2,413 and E0 61/400 remain unchanged; overall goal is active.
