# HybridZ ImageStar-inspired simplification trial

Date: 2026-08-30  
Branch: `redu-hz`  
Status: experimental; do not count toward the 13-family headline until the
frozen replay gate passes.

## Hard boundary

The frozen 13-family result is the non-regression floor:

- 2,413 instances;
- 1,063 certificates and 807 validated adversarial results (1,870 solved);
- no per-family solved-count loss is allowed;
- no new invalid SAT/adversarial result is allowed.

This trial may use exact Hybrid-Zonotope identities only. It must retain the
continuous factors, binary factors, equality/inequality predicates, shared
latent identities, and fail-closed verdict semantics. Replacing the state by a
zonotope or constrained zonotope is out of scope. Attack, splitting, backward
relaxation, and dual-verifier rescue are not part of this representation trial.

## ImageStar idea translated to HZ

ImageStar keeps the image-valued affine basis separate from its linear
predicate. The corresponding HZ principle is to keep only latent coordinates
that occur in the current value basis or predicate, while retaining every
binary/nonconvex relation. Affine tensor operators may act on the value basis
without changing the predicate.

ACT already applies dense HZ convolution directly to the center and generator
images in `hybridz_tf/tf_cnn.py`. Sparse HZ intentionally uses a sparse linear
operator because an L-infinity image input has thousands of one-pixel factors;
dense ImageStar-style convolution of every such factor would increase work.
The useful missing simplification on updated `main` is therefore at the final
latent/predicate boundary, not a domain replacement.

## Candidate retained on `redu-hz`

The final HZ-to-MILP lowering now performs two exact operations uniformly:

1. Remove a continuous or binary factor only if its column is structurally
   zero in the value map and all predicate matrices.
2. Merge byte-identical parallel or antiparallel predicate rows by exact
   interval intersection. No tolerance matching or coefficient normalization
   is used.

The lowering records source-column maps. Counterexample reconstruction puts
retained input factors back in their original coordinates; unused continuous
input factors receive the valid value zero and unused binary input factors the
valid value minus one. The simplification can be disabled at the `HZSolver`
constructor for paired audit, but is one uniform rule when enabled.

These operations were present in the frozen 13-family solver but absent from
the updated `main` used to create `redu-hz`. Restoring them is therefore a
baseline-preserving recovery, with explicit tests added for the new mainline
implementation.

Before final lowering, the verifier also releases intermediate dense/sparse HZ
states after retaining ordinary Python references to the output HZ and optional
input HZ. This changes object lifetime only: the final represented set and the
input state used for witness reconstruction are not mutated.

## Evidence collected

### Unit and environment gates

- Candidate-specific tests: 11/11 passed under the available pytest environment.
- `act-py312`: import, bytecode compilation, and a dense/sparse lowering smoke
  passed.
- The repository serialization tests cannot currently start because
  `act/back_end/examples/nets` contains no JSON fixtures. This predates the
  candidate and is not treated as a candidate failure or pass.

### Frozen ViT structural audit

On frozen `vit_2023` iid 0, projected exact-ReLU arm:

- final sparse HZ: 4,106 continuous factors, 127 binary factors, 674 predicate
  rows;
- exact lowering: 3,806 continuous factors, 127 binary factors, 578 rows;
- reduction: 300 unused continuous factors (7.3%) and 96 duplicate or
  antiparallel rows;
- verdict remained `UNKNOWN` under the 10-second solver budget.

This is the same pruning/coalescing rule used by the frozen solver, not a new
relaxation.

### Updated-main ViT capability result

On VNNLIB 2.0 `vit_2023` iid 101 (`ibp_3_3_8_9191`), the updated mainline
initially failed before solving:

1. a non-finite optional HZ tightening polluted valid interval bounds and
   triggered `slice produced invalid bounds (lb > ub)`;
2. after endpoint-level interval fallback, the final HZ materialized but
   lowering raised `MemoryError` under the 16 GB worker limit while 126
   intermediate sparse HZ states remained live.

With non-finite candidate endpoints ignored and intermediate HZ caches released
after retaining the final object, the same instance reports `CERTIFIED` through
the normal `verify_once` production entry point under the same 16 GB cap:

- final propagated HZ: 10,802 continuous factors, 40 binary factors, 18,112
  predicate rows, 30,801,360 stored nonzeros;
- exact lowered HZ MILP: 5,400 continuous factors, all 40 binary factors, 399
  rows;
- removed/coalesced: 5,402 unused continuous factors and 17,713 exact duplicate
  or antiparallel rows;
- released before lowering: 1 dense and 126 sparse intermediate HZ states;
- final solver elapsed: 2.50 seconds; end-to-end verification time: 69.25
  seconds.

The transition is `ERROR -> CERT`, not a relaxation-derived certificate. The
binary/nonconvex state is unchanged and the final MILP is exactly equivalent to
the propagated HZ.

### Frozen dense-family control

On `safenlp_2024` iid 0, the projected dense HZ had no unused factor and no
duplicate predicate row. The rule made zero structural changes and preserved
the result, demonstrating the intended identity behavior when there is nothing
to simplify.

## Rejected extension

A more aggressive exact projection would eliminate output-inactive continuous
factors defined by one equality and move their box feasibility into predicate
inequalities. Frozen ViT iid 0 had 127 such candidates, but every defining row
had width 3,394 and each candidate also occurred in an inequality. Eliminating
them would substantially increase predicate nonzeros, so this extension is
stopped rather than retained. It does not satisfy the goal of simplifying HZ.

A second extension attempted a conservative numerical fallback for the
score-box vertex used by sparse softmax-value propagation. On VNNLIB 2.0 ViT
PGD iid 0 it changed `UNKNOWN(missing_hz_state)` to
`UNKNOWN(empty_hz)`, but the same candidate regressed the retained IBP iid 101
capability result from `CERTIFIED` to `UNKNOWN(missing_hz_state)`. The extension
and its test were therefore rolled back in full. It is not part of the retained
candidate. This paired check is the first explicit no-regression rejection for
the trial.

## Remaining blockers before a new 13-family replay

One sampled VNNLIB 2.0 PGD ViT instance still reaches `missing_hz_state`; it was
localized to numerical failure in the optional score-aware softmax-value
refinement. The attempted fallback is deliberately not retained because it
regressed the IBP certificate. The IBP Slice failure above is fixed and reaches
a certificate with the retained candidate.

They prevent treating a new mainline run as a clean 13-family candidate replay.
The frozen 13-family records remain authoritative until these baseline issues
are separated or fixed. No headline result should be updated from the current
trial alone.
