# Cross-prime symbolic plan reuse V1: controls-only protocol

Separate `plan_basis/`; frozen modular/primitive engines, supervisors, real LPs
and all previous results remain unchanged. No native solve or real diagnostic
is authorized in this stage. Existing 4096-bit, 20M-operation, 128-prime and
absolute caller-deadline caps remain; plan index storage has a separate 1M-unit
cap and is also included in live storage accounting. All failed controls stay.

The first usable prime records initial nonzero structure and ordered pivot
row/column/support/affected-row indices. Later primes recompute numerical
coefficients, inverses, elimination and back-substitution; only the symbolic
schedule is reused. Every schedule is bound to exact source rows including RHS
and request scope, with a content checksum. No process-global cache exists.
Identity/schema/integrity failures raise errors, never fallback. A changed
modular sparsity pattern or zero planned pivot invalidates a schedule and pays
for a dynamic factorization at that same prime in the same budget. No fallback
resets counters/deadline. Wrong-field results cannot be accepted: both arms
check original modular equations; all reconstructed vectors still require exact
rational original-equation residuals and subsequently the unchanged original-LP
checker. Schedules and modular vectors are not certificates by themselves.

Both arms expose map, elimination, back-substitution, modular residual, plan
binding/validation/publication, CRT and reconstruction cost buckets. Per-round
reconstruction records successful prefix, first failed coordinate/reason and
in-progress/interrupted/complete-vector state. These are diagnostics, not
verified partial coordinates. Plan work and fallback remain charged. Timing is
instrumented local arithmetic, not unified end-to-end request cost. The old
supervisor journal schema is not reused or silently extended.

Controls cover fixed random rational systems (against new no-reuse and frozen
arithmetic), initial modular cancellation, intermediate zero planned pivot,
matrix/RHS/scope and cache-content corruption, rehashed malicious plans,
deadlines/caps/partial reconstruction, premature modular aliases, singular and
bad-denominator primes, multi-prime synthetic sparsity, original-LP rejection,
and relocated isolated `python -I -S` original-LP checking. No success threshold
requires a speedup: extra binding/validation can cost more than the saved heap
work. Cost comparisons are descriptive controls, not matched real timings.

Completion requires retained control receipts, fresh source/artifact and exact
solution/check review, result documentation and branch commit/push. Production
integration, a new unified-budget supervisor and any real freeze are separate
future stages, not implied by passing these controls.
