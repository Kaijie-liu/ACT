# Experiment 1D: Applicable-Unresolved Closure

## Status and scope

Experiment 1D is a follow-up closure study on the 20 applicable but unresolved
rows from the frozen confirmatory cohort. The original confirmatory endpoint
remains failed: 56/100 rows were solved, below the preregistered 60% overall
threshold. This result is immutable and is not backfilled by Experiment 1D.

The endpoint decomposition is reported separately:

- boundary applicability: 76/100;
- conditional verification coverage: 56/76 (73.7%);
- no route boundary within the frozen 4/255 search cap: 24/100;
- applicable but unresolved: 20/76.

The 24 no-boundary rows are inapplicable to route-boundary certification. They
are not solver failures, but they remain in the original all-sample endpoint.

## Frozen D0 selection

The tracked selection manifest identifies every row by parent JSONL hash, line
number, row hash, rank, dataset index, category, and reuse mode. D0 runs all 20
rows, with no early stopping:

- 14 `UNKNOWN_WEIGHTED_SOLVER_LIMIT`;
- 2 `TIMEOUT_EXPERT_SOLVE`;
- 2 `INSTANCE_HARD_DEADLINE`;
- 1 `UNKNOWN_SOLVER_LIMIT`;
- 1 `UNKNOWN_WEIGHTED_RELAXATION`.

The checkpoint, radius, candidates, feasible unordered top-2 pairs, guarded
support configuration, F0 encoding, and numerical SAFE policy are unchanged.
F1, expanded route caps, public baselines, and training remain out of scope.

## Artifact reuse and the 300/900-second schedule

The confirmatory JSONL persisted decisions and support metadata, but not live
Python HZ/MILP objects. D0 therefore reuses every completed candidate, pair,
expert, and property decision; it deterministically rematerializes only the HZ
state needed by unresolved branches and checks its candidate/pair/support
identity against the parent artifact. It does not reselect instances or solve
already completed property rows.

The two hard-deadline rows have only fixed-radius Tier-1 progress records, so
their reuse mode is explicitly `fixed_radius_rebuild`. They are rerun at their
already-recorded epsilon without repeating boundary search or bisection.

Each row has a 900-second wall deadline. The first solver pass is bounded by the
300-second row checkpoint. If still unresolved, the identical mathematical
encoding is restarted with the remaining wall budget because SciPy/HiGHS does
not expose a serializable warm-start state through this backend. At 300 seconds
D0 records the active expert/pair/property, incumbent, dual bound, gap, and
solver status. Incumbent/dual/gap are marked not applicable for the Tier-1
feasibility formulation, rather than silently invented.

Only an optimal, outward-corrected lower bound above the frozen positive margin
can produce SAFE. A relaxation candidate can produce UNSAFE only after concrete
full selected-softmax model replay.

## Guard paired analysis

The 225 matched confirmatory branches are reported as the full paired 2x2 table
(`n00`, `n01`, `n10`, `n11`). The primary paired comparison is an exact
two-sided McNemar/binomial test of `n01` versus `n10`. The report also includes
net solved gain and median paired runtime difference. Binary-elimination versus
solve-transition association is secondary and must not be described as an
unconditional runtime speedup.

## Preregistered baseline unlock

The public baseline is unlocked only if all conditions hold:

- all 20 rows run;
- independent audit reports zero issues;
- every new UNSAFE result replays on the full model;
- at least 5 additional applicable rows are solved;
- conditional applicable coverage reaches at least 80%, i.e. at least 61/76;
- no silent numerical fallback.

Experiment 1D results are follow-up closure evidence. They never replace the
original confirmatory 56% overall solved-rate failure.

## Preserved first-launch failure

The first D0 launch under implementation HEAD `152fa8c5f` stopped after ranks
110 and 120 were explicitly rejected by the support-identity assertion. The
runner compared the expert's fast preactivation count with the post-support
expert ReLU binary count, two intentionally different statistical universes.
No solver verdict was produced. Directory `experiment1d_bal010_d0` is preserved
and permanently excluded. The corrected run uses the unchanged frozen selection
and mathematics and writes only to `experiment1d_bal010_d0_r1`.
