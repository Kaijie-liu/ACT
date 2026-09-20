# Immutable validation amortization: controls-only protocol V1

This stage is separate from `plan_basis/` and the frozen real diagnostics.
No new real LP, native call, training, larger budget, production integration,
or claim of end-to-end acceleration. All old results remain unchanged.

## Single research change and matched controls

Study source binding and symbolic-plan validation amortization independently:
four control modes are repeated/repeated, cached-source/repeated-plan,
repeated-source/cached-plan, and cached/cached. Every mode constructs and fully
checks the same owned immutable source snapshot, fully admits each newly built
plan, follows the same prime schedule and numerical algorithm, and checks every
modular vector and complete exact rational residual. Only repeat frequency of
already completed validation on owned immutable payload differs. Old V1 and
old no-reuse arithmetic remain numerical differential references, not a newly
matched performance baseline.

Sources contain tuples of integer numerator/denominator pairs, not references
to caller dictionaries or mutable Fraction internals. Admitted plans contain
only nested tuples, integers and strings. Per-use source/scope, owner/generation
and receipt identity checks remain. External or deserialized plans require full
schema, checksum and source-binding validation; no imported trusted flag is
accepted. Handles cannot be used across sessions or after replacement/closure.
Each request owns at most one current admission; no global validation cache.

Original coefficient bit limits, complete index/row/source checks, plan schema
and checksum checks still run at admission. New-prime numerical sparsity, pivot
checks, modular residuals, full rational equation residuals and the unchanged
original-LP checker run as before. Structural invalidation pays for dynamic
factorization in the same budget. Time and operation counters never reset.

The immutability argument concerns the API's owned tuple data, not protection
against arbitrary replacement of Python code/private session state or hardware
memory corruption. This is not a machine-checked proof of the Python runtime.
The independently relocated original-LP checker remains the feasibility gate.

## Cost and resource discipline

Existing 4096-bit,20M-operation,128-prime,1M-plan-index and2M-live-unit caps stay.
Admission/copy/hash/check/guard costs are charged to the same caller deadline
and counter as arithmetic. Both modes include the owned source copy and any
old plus new plan storage during replacement. Live units are logical accounting,
not an exact Python RSS bound. No count is reset to manufacture a saving.

Common correctness repair in the new namespace: V1's dynamic path recorded
plan units in peak live usage but omitted them from its live-cap comparison.
Both new arms enforce the inclusive expression. This is covered by a small
analytic control, not a modification or relabeling of frozen V1 results.

Counters distinguish source freeze, source full checks, plan full checks,
per-use guards and unchanged numerical phases. Reconstruction prefix/failure
and interrupted-state fields remain. Timings from controls are descriptive,
not a separately registered real timing experiment.

## Fixed controls and acceptance

Use fixed random rational systems, the existing structured1024-dimensional
three-prime fixture, and small adversarial controls. Test all four modes for
identical complete solutions, modular computations and reconstruction outcomes.
Cover original-source/Fraction mutation, exported-plan mutation, invalid
imports/trusted flags, cross-session/stale/forged handles, immutable payloads,
new-prime cancellation/zero pivots, failed-prime recovery, operation/bit/storage
limits, deadlines at admission/use/reconstruction, exact-residual rejection and
relocated `python -I -S` original-LP checks. No acceptance threshold requires a
speedup. If fixed overhead dominates, retain that result rather than expanding
the fixture or choosing a more favorable number of primes.

Finish with retained attempt receipts, fresh source/artifact and exact saved-
solution review, documentation, commit and push. A unified outer supervisor,
partial-evidence publication protocol and any real diagnostic freeze remain
separate future stages.
