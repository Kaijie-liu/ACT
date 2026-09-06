# Staged-verifier production-path development closure

## Question

The seed-2 multi-model R1 worker spent one 300-second deadline on route-boundary
construction, a matched no-support solver control, retained-support solving,
and F0. The production entry point accepts a radius directly and omits the two
scientific controls. This development experiment asks one bounded question:
under unchanged candidate, Tier-1, F0, and numerical semantics, does the direct
path complete any previously unresolved full-model obligation?

It does not estimate natural prevalence, replace seed-2 R1, or provide an
interleaved runtime comparison.

## Frozen selection

The cohort is deliberately outcome-selected and therefore development-only.
It contains all 13 seed-2 R1 rows whose final reason is one of:

- `UNKNOWN_WEIGHTED_SOLVER_LIMIT` (10 rows);
- `TIMEOUT_EXPERT_SOLVE` (one row);
- `INSTANCE_HARD_DEADLINE` (two rows).

No SAFE, UNSAFE, or no-boundary row is included. Eleven radii come from the
completed result rows. The two killed workers had already persisted their
direct verification radius before entering F0; those radii and progress-file
hashes are frozen in the configuration. Checkpoint, source JSONL, sample-index
manifest, staged configuration, every rank/index/radius, and both partial
progress files are hash-bound before execution.

## Execution and endpoint

Each row runs in a fresh `act-py312` subprocess with a 300-second outer hard
deadline. The staged verifier writes progress at request acceptance, Tier-1
start/completion, F0 start/completion, and final verdict. A killed request is
therefore right-censored at its last recorded stage rather than assigned zero
F0 cost. Completed requests write immutable evidence packages; an independent
audit checks every package and fully replays every new UNSAFE witness.

The single predeclared signal is at least one new complete SAFE or replayed
UNSAFE among all 13 rows. If the signal is absent, the production separation
remains an engineering artifact and no new HZ cohort is launched. If present,
it permits preregistering a separate new cohort; it does not itself count as
confirmatory evidence.

Historical and new timings are non-interleaved and come from different
execution dates. They may be reported descriptively but never as a speedup.
No solver mathematics, tolerance, F0 relaxation, checkpoint, radius, or
historical artifact may change during this run. The AdvMoE Lagrangian holdout
remains locked.
