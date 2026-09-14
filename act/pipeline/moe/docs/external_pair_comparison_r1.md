# Complete-cost external static-pair comparison R1

Separate from rational-proof R3 and all sealed confirmation/ablation results.
Use exactly the first ten inputs of the 30-input R2 selection (the relation
ablation inputs), all three frozen checkpoints, 2/255, no route-based sample
selection: 60 requests. Six old index3000 smoke requests precede and gate full
execution by conformance/completeness, never by positive count. Full execution
has no resume, replacement or outcome-based stopping. Errors fail-stop and
are retained; UNKNOWN/TIMEOUT are normal denominator-preserving outcomes.

Arms: unchanged `relation_shared_r1.json` ACT adaptive versus ACT route
frontend + plain CROWN whole-box variable-weight static pairs. The latter
enumerates all 28 unordered pairs itself under the existing ACT/HiGHS policy,
then verifies every feasible pair with shared input, actual router logits,
two expert outputs and selected softmax weights. No free historical census,
guard-conditioned CROWN, alpha, BaB, dtype fallback or fixed gate weights.
It is NOT standalone alpha-beta-CROWN verifying the original dynamic graph.

Every request has one 300-second process-group watchdog including interpreter
startup, checkpoint/tensor loading, route analysis, serialization/handoff,
external interpreter startup, graph conversion, ALL bounds and worker result
submission. The entire owned process group is killed on expiry. Audit/ledger
maintenance after execution is separately excluded. Input images/boxes are
materialized once in ACT float64 before either arm, frozen by SHA and the
selection's tensor identities; this common raw-data preparation is recorded
and excluded equally. Every arm loads this SAME tensor file; the external
environment never runs its own ToTensor. Both environments check model-state,
input tensors and property identity. No computed verification facts are shared.

Plain `CROWN`, `conv_mode=matrix`, CPU/float64, one thread. Pinned source
commits/environment are those in `external_compatibility.py`; actual import
path, Python/Torch version and options are recorded. No dependencies installed.
External route queries receive at most min(10 seconds, remaining request
budget). Negative bounds are UNKNOWN. Five fixed conformance probes may find
a full dynamic prediction change; that is independently replayed in ACT before
UNSAFE is accepted. No PGD is introduced and absence of a witness is not proof.

Report positive intersections/arm-only results, all states, per-model and
single/multiple-pair strata, mean and median paired observed costs. Keep HZ
policy SAFE and CROWN numerical POSITIVE in distinct evidence classes; neither
is labeled independently checked rational LP. No requirement that ACT wins.
Thirty model-input pairs are ten correlated inputs, not 60 independent samples.

Run `python -m act.pipeline.moe.external_pair_comparison --pipeline` using
act-py312. A lock prevents another paired solver run. Do not edit source during
the frozen pipeline. Final structural audit is automatic; independent review
and compact-result commit/push follow completion, not inferred from launch.
