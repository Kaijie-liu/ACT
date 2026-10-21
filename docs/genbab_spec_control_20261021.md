# Guarded GenBaB specification and reception controls

This is the first implementation step of the independent B2 baseline. It checks
that the external backend will receive the declared guarded weighted-pair
problem and that partial answers cannot become a complete positive result.
It does not test the backend, improve ACT's algorithm, or establish a network
certificate. Starting revision: `5b176f3b2a28b3eb70121cd9bbb5193dcd4afc59`.

## Implemented boundary

`act/pipeline/moe/genbab_spec.py` is standard-library-only and opt-in. It reads
expert and class counts and arbitrary linear properties. For every unordered
pair and property it emits a raw unsafe conjunction over `[F_pair, router]`:
`r_out-r_selected <= 0` for each membership atom and `qF <= -c`. Pairs and
properties form the outer disjunction. The target is strictly `qF+c>0`; ties
are included. Whole-box sufficient and guarded modes have different identities.
This first version enumerates all pairs; it has no route-pruning acceptance path.

Raw backend matrices are converted only if each coefficient is exactly
representable as binary64. Nonrepresentable coefficients are rejected, not
rounded into a different problem. Future native wiring must set the default
dtype before building `VerificationSpec` and recheck actual endpoints/matrices.
The API safe-expression DSL is not used.

`act/back_end/moe/genbab_pair.py` adds the separate graph
`E_b+sigmoid(r_a-r_b)*(E_a-E_b)` followed by all router outputs. It consumes one
shared input and leaves the historical softmax/CROWN graph unchanged. Only its
static implementation is reviewed here; torch is not imported by this batch.
Declared-real equivalence is not native floating-point equivalence.

Reception binds request and clause identities, distinguishes the configured
nonlinear engine from observed `incomplete_only`/`nonlinear_bab` stages, and
requires every pair/property to return UNSAT before reporting `POSITIVE`.
SAT requires separate full-model replay; it is not accepted as UNSAFE here.
The grade stays a numerical filter, with `formal_SAFE=false`. Backend status
and identity fields remain trusted runtime inputs until the subsequent native
bridge binds them to actual graph/configuration/execution records.

## Once-only roster and accounting

The frozen file records the exact25 tests in `scripts/genbab_spec_cases.py`:
guard direction, selected-index versus score order, ties, constants, AND/OR,
two experts, the full8-expert/10-class combinatorial directory, changed dimensions,
whole-box separation, exact and rejected casts, complete/missing/duplicate
receipts, source/spec/engine contamination, SAT, UNKNOWN/TIMEOUT/ERROR, continued
deadline, pair-directory/target tampering, bad coefficients, mutable aliases,
unstarted decisions and observed-stage retention.

The8-expert fixture checks252 clauses/3,276 atoms only. It is not the rejected
full-size HybridZ source, a capacity run or a substitute for one. Every receipt
is fixed synthetic data; there are zero native solver calls and zero real slots.
No checkpoint, dataset, external numerical library, GPU or new dependency is used.

One batch, one admission (at most60s), one existing owned-session child. The
controller deadline is entry+300s, worker deadline entry+260s, owned execution
and cleanup terminate by entry+265s/+280s; reception/publication consume the
remaining common time. Reuse `scoped_proof.owned_session` and the established
`DEV_MARGIN_20261021_V1` limits:8GiB AS per process, RSS telemetry, unchanged
mathematical gates and128MiB total artifacts including one archive copy.
Startup before the script entry timestamp and final stdout/exit are not claimed
as full cold-start timing. No performance claim is drawn from this control.

Publish fixed prefixes, worker result, owned-session receipt and terminal result
to a new exclusive directory. Expected rejection tests count as passes. The
first unexpected failure stops the fixed roster; preserve prefixes and do not
repair/retry this consumed batch. PASS requires all25 tests, identity/deadline/
AS validation, normal exit, owned cleanup and controller FD balance. Saved-only
audit may verify these records but cannot supply missing online acceptance.

## Next gate

PASS permits the separately frozen native bridge/environment compatibility
stage. It does not authorize an unregistered real cohort. The shared B1/B2
input roster still needs a schema-aware historical exclusion inventory; old
selection logic omits later diagnostic identities. Input preparation must be
charged inside each comparison request instead of reusing the old free setup.
The source versions, native dependencies, actual engine selection and complete
query roster remain necessary before executing the new external comparison.
