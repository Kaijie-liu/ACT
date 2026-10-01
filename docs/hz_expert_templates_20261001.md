# Checked expert templates with conditional HybridZ views

The [frozen single-factor design](hz_expert_template_design_20261001.md) now has an
implemented source algorithm, 18 passing control groups, an independent saved-
evidence audit and 146 passing regressions. Each expert is propagated once from
the shared input/router entry; each pair then receives its independently checked
guard view. All 15 pair bases, source coefficients and endpoint objectives match
the freshly executed per-pair reference exactly. All 42 endpoint bounds also agree.

This reduces repeated source propagation and source-proof records. It does not
reduce the number or size of output support problems, establish real-model
capacity, add certificates or demonstrate production/GPU speedup. All six goal
gates remain OPEN. The design was frozen on October 1; final documentation was
completed on October 2, 2026, Sydney time.

## What changes in HybridZ analysis

The prior path propagated both experts separately for every pair. Its registered
affine compensation and ReLU box ranges depend on the output expression, not
guard-based tightening. The new [producer](../scoped_source/hz_templates.py)
constructs a common entry retaining the complete checked router factors and
constraints while restoring the input expression. It propagates each expert
through actual sparse affine operations, exact ReLU references and checked
binary64 enclosure once, assigning distinct expert-private factors.

For each pair it still creates the original selected-versus-outsider guards,
checks their exact source and outward storage, and inserts their rows after the
common constraint prefix. The [independent checker](../scoped_source/check_hz_templates.py)
reconstructs every view coefficient, factor ID, owner and constraint. This version
refuses guards that introduce factors or change the original common rows.
It never calls the template producer, optimizer or propagation kernels.

For one legal route, both experts extend the same input/router assignment with
disjoint private factors. Adding the legal guard preserves these extensions.
Different pairs are separate problems: reusing an expert's template identity
across them does not identify variables in a combined cross-pair problem. No
pair is excluded and no property is skipped. The existing endpoint checker still
requires complete coverage and checks newly generated candidates.

This is not a new general abstract domain or a claim that existing multi-objective
matrix sharing is new. Nor is it cross-request caching. It reduces repeated
propagation and proof construction under a particular, checked source contract.
Guard-tightened propagation is outside this equivalence: inward ranges obtained
only under a pair guard cannot be reused as facts over the common entry.

## Fixed source comparison

Both arms recreate the four old declarations and generate fresh 128-step CPU
candidates. They alternate execution order by declaration. All original limits,
gate rules, properties and the `1/10000000` positive threshold remain unchanged.

| Declaration | Expert propagations old to template | Checked source steps old to template | Positive properties in both | Endpoints in each arm |
|---|---:|---:|---:|---:|
| weighted_sign | 6 → 3 | 19 → 10 | 3 / 3 | 6 |
| tied_partial_reuse | 12 → 4 | 37 → 13 | 18 / 18 | 18 |
| unsafe_tied | 6 → 3 | 19 → 10 | 0 / 6 | 6 |
| unresolved_sign | 6 → 3 | 19 → 10 | 6 / 6 | 12 |

Calls are recorded at the actual propagation dispatch, not inferred only from
final trace length. The router is propagated once in each arm. The differential
compares entry, both expert snapshots, joint source, complete LP base, every
endpoint objective, binary relaxation and gate range. It ignores only request
identity metadata; it does not accept equal positive counts as sufficient.
All eight complete mathematical packages are unchanged between R1 and R2.

The template partial-evidence control stays `UNKNOWN_MISSING_EVIDENCE`. The
complete nonpositive control stays UNKNOWN, not UNSAFE. All full declarations
still have zero nonzero storage-lift compensation; three separately bound affine,
ReLU and guard operator controls exercise view composition after nonrepresentable
reference handling. They are not additional complete network proofs.

## Cost observations and their limits

Final R2's complete suite took about 5.192 seconds, including imports/test
orchestration. The per-arm intervals below include source creation, propagation,
candidate generation, serialization and independent checking, but exclude those
suite-level imports and orchestration. These are one-off tiny controls, not a
timing confirmation or a complete hard-budget experiment.

| Declaration | Full interval old / template seconds | Construction old / template seconds | Serialized package bytes old / template |
|---|---:|---:|---:|
| weighted_sign | 0.270 / 0.110 | 0.133 / 0.043 | 313252 / 169685 |
| tied_partial_reuse | 0.350 / 0.198 | 0.141 / 0.080 | 716504 / 276894 |
| unsafe_tied | 0.170 / 0.113 | 0.065 / 0.045 | 305758 / 168344 |
| unresolved_sign | 0.213 / 0.210 | 0.083 / 0.124 | 393573 / 219557 |

The last construction interval is worse despite fewer expert calls. Do not turn
structural work reduction into a universal speed claim or infer its unique cause
from this one timing. Every pair view, joint CSR, support query, exact acceptance
and final source snapshot is still constructed/checked. No native solver, CUDA,
training, real checkpoint or sealed input ran.

## Independent recheck and retained attempts

[R1](hz_expert_templates_20261001_r1.json) is preserved before accounting hardening.
Its source-pollution test performed one extra template build and real candidate
pass before corrupting the input; that failed auxiliary call was charged only to
suite time, without its own stage record. R1 is not evidence of complete per-call
accounting. It also lacked final audit checks for finite suite time and arm order.

[Accepted R2](hz_expert_templates_20261001_r2.json) records both auxiliary calls:
an intentionally already-expired producer, and a source-mutation injection which
now returns no candidate and performs no optimizer work. Their source hashes,
dispatches, fault reach, error terminals and costs are retained. Unexpected
TimeoutError is not accepted as an ordinary corruption rejection.

The auditor checks nine source packages, three operator/view bridges, the fixed
partial omission and all 19 concrete negative queries. It reconstructs each
negative input from the bound normal package and independently observes refusal;
changing a receipt/hash alone cannot substitute a different mutation. Complete
case clocks are finite, ordered, nonoverlapping and contained in the suite clock.
The checker can be rerun without proposing a new bound, but still uses repository
imports: this version is not a new portable `python -S` distribution.

Raw records remain in `baseline_runs/hz_expert_templates_20261001_r1` and `_r2`.
They occupy 6,293,900 logical bytes in total. No evidence was removed.
Separate read-only reviews found no remaining blocker in the limited mathematical
and accounting contracts; they do not substitute for independent human review.

## Next decision

Retain the template mechanism as opt-in. Do not rerun the four fixtures for a
better count or enable it on the sealed real requests. The complete pair support
domains have not become smaller, and the known large-row, global-dimension,
operator and package-capacity limits remain. The next representation question is
how these shared/private views feed support and evidence without materializing
repeated full joint matrices. Compare against the current already shared multi-
objective backend, not a newly weakened baseline. Keep any block-native support
or capacity experiment separate from changes in precision, GPU hardware or gates.
New hard-budget/portable integration must explicitly cover the template schema
before real admission; older source supervision does not automatically cover it.

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data1/Kane/miniconda3/envs/act-py312/bin/python -B -m scripts.run_hz_templates audit /data1/Kane/MOE/baseline_runs/hz_expert_templates_20261001_r2
```
