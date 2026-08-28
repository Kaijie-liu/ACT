# Restricted weighted top-2 fallback

This document freezes the soundness contract for the selected-softmax top-2
fallback used after Experiment 1C. F0 is a range-only relaxation. F1 is a
separately gated future ablation and is not part of the F0 implementation.

## Scope and routing semantics

F0 applies only to output-level MoEs with `top_k=2` and the
`selected_softmax` gate. For every exact feasible unordered pair
`rho={a,b}`, the implementation first uses the canonical order `a < b` and
attaches the tie-inclusive guard

```text
r_j <= r_a and r_j <= r_b, for every j outside {a,b}.
```

The guard does not allocate a route-selection binary. Equality is allowed, so
all legal pairs induced by a router tie must be enumerated and proved. A row is
`SAFE` only when every property row is proved on every exact feasible pair.

With `m = r_a-r_b` and `lambda = sigmoid(m)`, the concrete pair output is

```text
F = E_b + lambda * (E_a-E_b).
```

For a linear safety row `q^T F+c >= 0`, F0 constructs only

```text
u = q^T E_b+c
d = q^T(E_a-E_b)
y = u + lambda*d.
```

It does not introduce product variables for all output logits.

## Shared-input pair propagation

Both experts are propagated from the same guarded input HZ. The merge preserves
one identity for every input continuous generator, input binary, router
constraint, and pair-guard constraint. Expert-local factors are remapped into
disjoint suffixes:

```text
[shared input continuous | expert-a private continuous | expert-b private continuous]
[shared input binary     | expert-a private binary     | expert-b private binary]
```

The merge rejects an expert HZ if its shared equality or inequality prefix has
changed. This prevents two unsound failure modes: independently cloning a shared
input generator, or aliasing the two experts' private ReLU binaries.

## F0 range-only encoding

Support optimization under the pair guard gives
`m in [m_lower,m_upper]`. Numeric monotonic evaluation gives the outward-rounded
constant interval

```text
lambda in [sigmoid(m_lower), sigmoid(m_upper)].
```

The symbolic encoding contains no exponential, division, sigmoid segment, or
sigmoid binary. Expert-pair propagation gives a sound support interval
`d in [d_lower,d_upper]`. For `w=lambda*d`, the four standard McCormick rows are

```text
w >= lambda_lower*d + d_lower*lambda - lambda_lower*d_lower
w >= lambda_upper*d + d_upper*lambda - lambda_upper*d_upper
w <= lambda_upper*d + d_lower*lambda - lambda_upper*d_lower
w <= lambda_lower*d + d_upper*lambda - lambda_lower*d_upper.
```

The checked scalar is `y=u+w`. The McCormick HZ is marked as a relaxation even
when all preceding router and expert HZs are exact.

Only a solver-certified, outward-corrected, strictly positive lower bound proves
the property row. For an MILP, the certificate uses a finite HiGHS
`mip_dual_bound` from a successful status-0 solve; for a pure LP, it uses the
status-0 optimum. It never uses a non-optimal primal incumbent as a lower-bound
certificate. The raw lower bound is corrected outward by
`1e-9 + 1e-9 * scale` and `nextafter` toward negative infinity, and F0 requires
the resulting bound to exceed `1e-7`. Feasibility and integrality checks both use
the frozen `1e-7` tolerances.

A non-positive relaxation lower bound is never an unsafe verdict. Its optimizer
is only a candidate input and must violate the concrete full selected-softmax
model inside the input box before the result may be labeled `UNSAFE`.

## Status contract

F0 emits only these semantic reasons:

| Reason | Meaning |
|---|---|
| `SAFE_WEIGHTED_RANGE` | every required F0 relaxation has a positive lower bound |
| `UNSAFE_FULL_FORWARD_FALLBACK` | a recovered input violates the concrete full model |
| `UNKNOWN_WEIGHTED_RELAXATION` | the range-only relaxation crosses the property boundary |
| `UNKNOWN_WEIGHTED_SOLVER_LIMIT` | feasibility, support, or property solving exhausted its budget |
| `UNKNOWN_WEIGHTED_NUMERICAL` | lowering or backend consistency prevented a sound conclusion |

`SAFE_WEIGHTED_SEGMENTED` is reserved for F1 and cannot be emitted by F0. A
relaxation optimizer is named `candidate_input` in the API to avoid treating it
as a counterexample before full-model validation.

## Tests and mutation control

`act.back_end.moe.test_weighted_top2` covers:

1. zero router margin gives exactly `lambda=0.5`;
2. equal experts give zero output difference and product;
3. fixed-positive `d`;
4. a `d` interval crossing zero;
5. an outside-expert tie without a selector binary;
6. enumeration of multiple legal top-2 pairs;
7. shared input-generator identity and private binary separation;
8. randomized concrete products satisfy all four generated McCormick rows;
9. randomized shared inputs satisfy the McCormick rows emitted by an F0 encoding;
10. a relaxation violation remains `UNKNOWN`;
11. a positive relaxed lower bound is `SAFE`;
12. reversing one McCormick inequality makes the consistency test fail.

## Frozen F0 diagnostic

The tracked config is
`act/pipeline/moe/configs/experiment1f0_bal010.json`. The runner verifies the
SHA-256 of the frozen Experiment 1C JSONL and selects only its 38 rows whose
parent reason is `UNKNOWN_GATE_SUFFICIENCY` or
`UNKNOWN_EXPERT_WITNESS_NOT_LIFTED`. Every result carries the parent artifact
hash, parent line number, parent-row hash, and a derived stable parent-row ID.
Previously decided `SAFE` and `UNSAFE` rows are not rerun.

The output directory is new and existing output is never overwritten. JSONL and
CSV are flushed after every parent row. A validated unsafe input is saved as a
separate witness artifact for independent replay. The preregistered decision is:

- resolve at least 10 of 38 rows, or add at least two non-repeated unique `SAFE`
  samples: freeze F0 and keep it as the primary fallback;
- otherwise: retain F0 as an ablation and implement the preregistered F1
  two-segment margin-correlated fallback.

Confirmatory ranks 100--199, baseline reproduction, and new training remain
paused until the fallback and its soundness audit meet the unlock conditions.

The initial launch directory `experiment1f0_bal010` is a preserved failed
engineering run: a support-metadata serialization mismatch caused eight optimized
rows to be labeled numerical. No verdict from that directory is used. The fixed
runner has a regression test for the public support-result fields and writes a
clean, non-overwriting run to `experiment1f0_bal010_r1`.

## F0 diagnostic outcome

The corrected frozen run completed 38 rows from 14 samples and resolved 31 rows:
26 `SAFE_WEIGHTED_RANGE`, five independently replayed
`UNSAFE_FULL_FORWARD_FALLBACK`, four `UNKNOWN_WEIGHTED_RELAXATION`, and three
`UNKNOWN_WEIGHTED_SOLVER_LIMIT`. There were no numerical/backend rows.

Nine non-repeated sample ranks gained a weighted F0 certificate. Because all
parents were selected specifically for prior semantic incompleteness, this is a
fallback-resolution result and not a population prevalence estimate. The
independent audit found zero soundness or artifact-linkage issues and replayed all
five saved concrete witnesses.

F0 passes both preregistered continuation criteria: 31 resolved rows exceeds 10,
and nine new unique safe samples exceeds two. The range-only configuration is
therefore frozen. F1 remains a documented but unimplemented ablation and is not
triggered by this diagnostic.

## F1 boundary

F1 may split the margin at zero or other preregistered cut points, reuse ACT's
validated sigmoid envelopes, and add at most one segment binary per pair. It must
not use an ad-hoc chord across the sigmoid inflection point. F1 requires its own
tests, design update, commit, and push before any F1 result is run.
