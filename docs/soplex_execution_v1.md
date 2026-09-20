# SoPlex finite comparison — execution addendum V1

This implements, without amending, `soplex_finite_comparison_v1.md` and its
machine freeze. User explicitly requested completion of the outer supervisor,
candidate receiver and terminal cost audit **then execution**. The original
freeze's `execution_ready=false` is a historical preparation status, not edited.
All four LPs, their order, coefficient semantics, exact settings and caps stay
unchanged. No custom elimination, repairs, retries, new sample or LP query.

## Supervision

Each sequential request starts its monotonic clock before specification I/O,
worker startup and source loading. Proposal phases (including full exact LP
export, import-only rational readback, one CLI solve and output capture) must
complete by 218s. Packaging, relocated `python -I -S` original-LP checking and
review share the original 298s deadline; publication is limited to 300s. Worker
phase timers and an outer owned-tree watchdog are independent protections.
Native solver time is the remaining proposal window, never a reset clock.

Resource gate before **each** job uses the unchanged 16GiB RAM, 5GiB disk and
load/core<=0.5 thresholds; wait observations are durable and outside request
cost. One worker/native attempt at a time, nice10, CPU only, single-thread
environment. Workers and descendants inherit 8GiB address-space and 128MiB
individual-file limits. Outer polling additionally checks 64MiB point and
512MiB aggregate output caps, including after process exit. Overshoot up to one
poll can leave partial files, never an accepted over-cap certificate. Only
PID/start-time-bound owned descendants are cleaned, including separate sessions.

`worker_result.json` is an unaccepted candidate terminal. The outer supervisor
checks identity, phase deadlines and evidence hashes. Terminal, publication
receipt and observed publication cost are retained; a late publication is
TIMEOUT regardless of candidate contents. No late success overwrites a timeout.
ERROR stops the batch and preserves later NOT_RUN_AFTER_ERROR slots; LIMIT,
TIMEOUT and checked-but-unresolved results continue. The denominator is four.
No resume entry point. New directory existence is an error, never an overwrite.

## Exact point admission and interpretation

Accept only the pinned CLI rational `Primal solution (name, value)` section
with its complete newline-terminated count footer. Canonical `xN` names must
be distinct/in-range, exactly count nonzero entries, and use integer/fraction
tokens. Omitted variables mean zero only under this complete contract. Check
token lexical length **before** integer allocation, then 4096-bit serialized
numerator/denominator caps. Rays, decimals, malformed or truncated output are
not feasible points. Missing point file/native no-point record is unresolved.
An externally interrupted writer is TIMEOUT/LIMIT, not a malformed-complete
ERROR. Uninterrupted malformed output is ERROR and stops later jobs.

Recompute the objective with the original offset and submit the full original
LP, statement and candidate to the unchanged isolated checker. Its exact box,
inequality, equality and objective checks alone grant feasible U. An inexact
point produces no U and remains unresolved; native status never grants U.
No dual certificate is exported: no new lower bound/optimality claim. A checked
U<=0 is an obstruction for this **LP relaxation**, not a MoE UNSAFE witness.
Historical checked lower bounds are context only. Network/HZ/F0 construction
is supplied, excluded from this diagnostic's budget and not independently proved.

## Complete accounting and review

Persist entered/done phase timestamps separately. Missing phases have null
costs; interrupted phases have null completed time plus observed censored time.
Top-level phase times do not overlap. Whole cost includes startup, all source
checks, construction, cross-process I/O, solving, capture, serialization,
isolated checking, review, cleanup and terminal publication. Native command
times and GNU-time peak RSS are nested diagnostics, not added a second time.
Sampled per-process RSS is not the 8GiB address-space contract, nor is the sum
of individual HWM values a simultaneous whole-tree peak. Files/output bits are
reported; missing native resource files remain null.

Post-terminal archival hashing/fresh isolated checks are separately timed and
do not retrospectively rescue a timed-out request. Audit rechecks every saved
coefficient against original LP using the separate readback parser, original
statement/LP binding, phase and total costs, publication time, fixed native
arguments and fresh original-LP feasibility for accepted points. Results and
failed/partial artifacts remain intact. Raw LP/point bundles stay out of Git;
compact receipts, manifests, costs and review go into Git.

## Readiness

Before launch: analytic exact native cases (including infeasible/no point),
candidate mutations, bit/file caps, source binding, inherited process limits,
proposal/work cutoffs, owned descendant cleanup, partial outputs, late
publication, fail-stop/continue rules, resource waiting and cost tampering.
The original 21 checker regressions remain unchanged (including four analytic
SciPy queries). Freeze all implementation/dependency/runtime hashes and obtain
a fresh saved-control review. Commit/push that preparation before starting the
four original LPs. After this finite run, audit/archive and close; unresolved
outcomes do not authorize another algorithm, longer time or larger sample.

Preparation attempt001 was NOT released: the unchanged checker's 1e-7-second
CLI timeout regression once exited 1 rather than expected 3 (20/21 pass).
The standalone regression subsequently passed unchanged. The failed full-suite
receipt `soplex_execution_checker_regressions.json` remains; no checker, timer
value, test assertion or acceptance criterion was altered. This is an observed
timeout-exit instability, not a feasibility error; the new outer path rejects
nonzero checker exit and preserves ERROR/TIMEOUT rather than accepting a bound.
Attempt002 uses a separate receipt identity. No real LP launched in attempt001.
Attempt002's full 21-test regression passed. Its freeze assembly then rejected
an erroneously listed `evidence_cohort/__init__.py` (that existing package uses
namespace packaging). The nonexistent file was removed from the dependency
inventory, not created; attempt003 binds the actual ownership module. All prior
control/regression receipts remain; this fix changes no executing solver path.
