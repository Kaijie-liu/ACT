# Supervised unchanged-LP diagnostic V1

This is a separate opt-in diagnostic, not production verification, a cache
comparison, a new cohort, or a complete MoE proof. Freeze first; execution is a
subsequent explicit action. Old results, modules, gates and source hashes stay
unchanged. No checkpoint/data loading or network/HZ/F0 reconstruction occurs.

## Selection and interpretation

From the sealed 30 nonpositive obligations, select the first nonpositive
property in original pair/property order for each already observed input:
220 / pair {1,2} / property 0; 222 / {0,1} / 1;
230 / {0,3} / 2; 232 / {0,1} / 0.
The selection is diagnostic and outcome-aware; it is not confirmation sampling.
Four saved weighted LP exports are bound byte-for-byte to the original archive,
manifest, request, property, source HZ and canonical LP identity. Read-only
selection/reconstruction makes zero solver calls. Keep ranges and matrices
unchanged. No substitution, enlargement, repair, retries or resume.

Each obligation receives one native candidate call, preserving its raw point,
status, objective, marginal and residual records before any exact check. Solver
success is not a proof. The frozen component's internal check is retained; a
second isolated `python -I -S` checker validates the movable bundle. No
approximate feasibility tolerance or point projection is permitted.

Exact feasible U <= 0 proves that this supplied LP cannot have a strictly
positive minimum. L > threshold is a checked lower bound for this LP only.
A missing/non-exact point supplies no upper bound; a nonpositive candidate
lower bound alone leaves candidate weakness versus relaxation unresolved.
Neither conclusion is complete-network SAFE or UNSAFE. Network-to-HZ, guard,
route exclusion and F0 lowering remain trusted, unchanged upstream boundaries.

## Original clock and supervision

One worker, one CPU thread, GPU hidden. The original monotonic clock begins
before plan serialization; every phase shares it. Loading/parsing/hash binding,
imports, native proposal, retained internal check, serialization, copying,
isolated checking and terminal admission are charged. Native cap is 60 seconds;
load/proposal must finish by original second 218, reserving 80 seconds of the
298-second work window for packing/checking. The owned process tree is stopped
at its original phase/outer deadline. Termination/cleanup is charged, not a new
budget. No late success is accepted after 300 seconds, including publication.
OS cleanup/publication can overrun wall time; such a run is TIMEOUT, not a
promise that an OS process can always be killed and recorded instantaneously.

Immutable phase entries, completions, logs, native records, candidates and
outer/publication records remain even on failure. Incomplete checks never
produce a checked conclusion. ERROR stops the roster; all remaining jobs have
NOT_RUN_AFTER_ERROR records. TIMEOUT and checked-but-unresolved jobs continue.
All four selected jobs remain in the denominator. No silent retry or overwrite.
The launch requires a clean, pushed feature HEAD, frozen controls and a separate
selection reconstruction. Resource gating happens before each clock and is
reported separately; no other user's processes may be interrupted.

## Cost contract and audit

Full cost means **supplied-LP diagnostic cost**, not end-to-end MoE cost.
Historical propagation/range construction is not rerun and not silently zeroed
into a new network timing claim. Report full original-clock publication time,
exclusive load/propose/package/check windows and residual overhead. Native and
component times are nested in propose, NEVER added again. Missing durations
are null, not zero; interrupted phases include a censored observed window.
Preflight, per-job resource waiting and post-terminal/final archival audit costs
are separate overheads. The final summary reconstructs every terminal and
classification, including failures, rather than trusting aggregate counts.

The structural audit checks provenance, clocks, phase completeness and checker
agreement; it does not rerun the solver or independently establish upstream
network lowering. The movable checker itself uses only the standard library
and exact rational arithmetic. Its bundle/statement hashes must be supplied
externally. Mutation, relocation, deadline, missing evidence and denominator
controls are required before freezing. No performance claim follows from
analytic controls, and completion is not defined by finding a positive result.

## Commands (execution is separate)

Use `/data1/Kane/miniconda3/envs/act-py312/bin/python`:

```
python -m lp_diagnostic.controls
python -m lp_diagnostic.study freeze --controls docs/lp_diagnostic_controls_attemptNNN.json
python -m lp_diagnostic.study reconstruct
# Only after a later execution decision and commit/push:
python -m lp_diagnostic.study launch
python -m lp_diagnostic.study audit
```

New raw directory: `data/moe/results/lp_diagnostic_20260919_v1`.
Raw LPs/checkpoints are never committed. Freeze/selection review are compact
metadata and do not create the execution directory.
