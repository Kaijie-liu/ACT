# Additive checkpoint: actual ReLU44, actual ReLU71, first terminal failure

Branch `redu-hz`, base commit `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.
Existing production edits and all historical HyZor data remain untouched.
The preceding seven checkpoint manifests are immutable and included by hash
in the eighth additive seal. No candidate is promoted; the goal stays active.

- ReLU44: 124 tests, native admitted phase and real publication/consumption
  audited. Complete live storage at actual return 147,339,420 bytes and
  15,670,409 entries, both strictly below the same verified expanded-reference
  lower bound. See `C5_RELU44_ADMITTED_AUDIT_20260905.md`.
- ReLU71: 129 tests, all four actual zero suffix frontiers preserve every
  predicate and slot. Complete 601-root live state 175,235,772 bytes and
  18,443,567 entries below 314,673,156 bytes and 26,214,400 entries respectively.
  See `C5_RELU71_ZERO_SUFFIX_AUDIT_20260905.md`.
- Full terminal: 129 tests, source/configuration freeze intact, timeout at
  240 seconds in ReLU78 composition, no solver/result/witness. Closed without
  a retry. See `C5_FIRST_TERMINAL_TIMEOUT_AUDIT_20260905.md` and its hashed
  actual-state postmortem. Full attempt failure does not become a prefix solve.

The next target remains the same residual CNN affine suffix, not another
instance chosen to pass. Preregister a uniform exact support-carrying affine
rule, its complete 14-term work accounting, source-bound equivalence tests,
and construction/whole-state gates before evaluating a changed candidate.
Keep the original accumulation order where required by the frozen numerical
reference. No full terminal rerun until its changed suffix is qualified.

Formal 1870/2413 and E0 61/400 (CIFAR25, Tiny36) remain unchanged. No full
2413-case or 400-case candidate retention run has passed. All current jobs
are terminal and their evidence is retained; no background job is promised.
