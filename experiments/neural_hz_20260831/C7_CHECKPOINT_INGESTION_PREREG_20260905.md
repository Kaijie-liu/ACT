# C7 checkpoint portability and unchanged-backend ingestion audit

The offline C7 V1 result is complete and frozen:
f63d45dd21d7bfa6d0c415cd24b10ac99345d053fa72ca82ad7a2bc9087928ef.
Its new checkpoint SHA-256 is
6cbe92faf0d6eb5e4113f3e6e0b324a82a00d2d1a6019a73f2c2fd0f522e82df.
It passes the offline identity/work/storage checks, not native publication,
solver fidelity, a terminal solve or any score gate.

In a fresh bounded worker, verify source/checkpoint hashes before unpickling.
Rebind process-local identities only after the trusted file seal is checked,
then rerun the independent audit of every definition, original predicate,
factor graph and value map. Preserve the old coordinate-prefix reconstruction
map. Inventory all finite coefficient ranges and required auxiliary exponents.

Inspect the ACT ordinary lowering with its existing defaults (unused-factor
pruning and exact row coalescing on, no inactive-factor projection or binary
phase fixing). Use the SAME HiGHS binary imported by the installed SciPy
MILP wrapper, not an unrelated highspy installation. Record its version and
default small/large matrix thresholds; leave numerical options unchanged.
Call only passModel and getLp: NEVER run, optimize, presolve, or solve. Compare
the COMPLETE matrix, bounds and integrality actually retained after ingestion
against what ACT handed it, recording any changed/removed coefficients.
No objective/property result is inferred from a zero-cost placeholder model.

This is a read-only backend-fidelity diagnostic, not a rescue or a new
verification path. If loading drops small definition coefficients or otherwise
changes the problem, live/terminal advancement is not qualified. Any remedy
requires a new structurally triggered exact representation preregistration;
do not change solver tolerances or options to hide an unfavorable result.

One exclusive results/c7_checkpoint_ingestion_20260905_v1/, 16 GiB, CPU1,
240-second worker wall. Retain frozen source/library hashes, all 231 preceding
tests, stdout, complete audit result or failure, and exit hashes. No old file
is edited. Formal 1870/2413, E0 61/400 and all promotion gates stay unchanged.
