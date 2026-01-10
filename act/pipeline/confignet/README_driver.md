ConfigNet Unified Driver: Immutable Semantics

This document freezes semantics for the unified ConfigNet L1/L2 driver.
These rules are mandatory. The driver only orchestrates existing logic and
must not alter algorithmic meaning.

1) Level 1 policy (must hold)
- If a concrete counterexample is found, CERTIFIED is forbidden.
- This is enforced via policy gating (see `act/pipeline/confignet/policy.py`).

2) Level 2 strict behavior (must hold)
- Any alignment failure, missing bounds, or NaN/Inf in bounds is an ERROR.
- Strict mode must preserve this behavior (see `validate_bounds_per_neuron`).

3) Bounds compare output fields (must hold)
- The per-neuron comparison outputs the canonical fields:
  `violations_total`, `violations_topk`, `layerwise_stats`.
- These field names are contractually stable.

Driver rules
- Orchestration only: the driver calls existing L1/L2 functions.
- Do not re-implement or change L1/L2 semantics.
- JSONL v1/v2 coexist: v1 is untouched; v2 adds fields.
