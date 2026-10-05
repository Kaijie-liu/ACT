# Complete ReLU36 boundary proof / actual ReLU63 checkpoint

Read in order:

1. `C5_LIVE_AND_RELU36_AUDIT_20260905.md` (preceding work, now separately sealed);
2. `C5_RELU36_BOUNDARY_AUDIT_20260905.md` (closed full-reference rejection and
   successful conservative complete-state proof, same runtime/metric);
3. `C5_RELU63_PREFIX_AUDIT_20260905.md` (fresh ReLU63 state and the exact
   ReLU44 admitted-phase gate still needed before ReLU71).

New runs `c5_relu36_live_20260905_v1`, `c5_relu36_live_20260905_v2`, and
`c5_integrated_relu63_20260905_v1` are all terminal; do not restart them.
Tests respectively 49, 59 and 73 passed. First physical-allocation diagnostic
is a retained failure, second proves the strict whole-state inequality by
reference lower bound, third reaches actual ReLU63 in 11.749476 propagation
seconds. No joined verdict or terminal witness is claimed.

The additive `CHECKPOINT_C5_RELU36_PROOF_RELU63_20260905_SHA256SUMS` binds all
new sources, preregistrations, tests, audits, raw runs and three evidence
records, plus the preceding six manifests. Verify all seven from this
experiment directory. Native/default code and historical HyZor data remain
unchanged; candidate is opt-in on redu-hz. Formal 1870/2413, E0 61/400 and
TLL capability 27/32 are unchanged. Goal is active, not complete or blocked.
