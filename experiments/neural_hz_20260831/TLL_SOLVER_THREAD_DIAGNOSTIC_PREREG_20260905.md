# TLL controlled solver-thread diagnostic

The previous goal turn made progress: it closed C3, implemented default-off BN
graph repair, restored exact baseline provenance, and completed a current TLL
qualification that retained 17/17 formal solved rows but lost candidate 8/26.
The goal remains the original 13-family Neural-HZ improvement, not this test.

Pre-registered on 2026-09-05, before any new TLL execution. Source and assets
come from the sealed current-source V2 campaign. No HZ, witness or production
solver code will be edited. All new records go to a new exclusive directory.

Local inspection establishes that SciPy 1.17.1 uses embedded HiGHS 1.12.0, not
the separately installed highspy 1.15.0. Its default `threads=0` and
`parallel=choose` remain in effect despite OMP/BLAS/MKL=1. On this 20-CPU
affinity mask, a minimal call raises process thread count from 1 to 10.
That observation alone does not establish the cause of either TLL regression.

Two complete 32-row batches, both with four concurrent workers, are fixed:

1. `auto`: invoke the existing SciPy MILP options unchanged;
2. `one`: add only `threads=1` to every MILP call in every worker.

No seed, presolve, tolerance, gap, HZ rule, model, specification or witness
validation changes. No selected rerun, winner union, witness repair/clamping or
budget enlargement. Solver budget stays 45 seconds, process memory 16 GiB and
supervisor wall cap 240 seconds. Run auto first, then one; report that order
confound and do not treat this two-batch diagnostic as a definitive speed gate.

A process-local tracing wrapper fingerprints the exact numeric MILP objective,
integrality, variable bounds, matrix and row bounds before and after each call;
records CPU and wall time, native thread counts, returned status and bound
violation; and returns the original solver result unchanged. The thread monitor
adds one observation thread, which is reported separately. Hashing/monitoring
cost and consumed deadline are not hidden. A changed input digest rejects the
diagnostic's consistency claim. The wrapper cannot authorize any verdict.

Compare each row's matching first/second numeric MILP calls across the two
arms. Equal row/column/nnz counts alone are not a coefficient-identity proof.
Retain every call, result, log and supervisor exit. Report rejected solver
proposals separately from invalid reported ADV. Any policy improvement is
solver scheduling evidence, not HZ novelty or formal score gain. An apparent
positive must later survive uninstrumented replay and the existing promotion
gates. Formal baseline 1870/2413 and external E0 61/400 remain unchanged.
