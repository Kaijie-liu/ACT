# C10 exact alias quotient V1 — preregistered component experiment

2026-09-08, redu-hz. Additive successor to the sealed read-only C10 census.
The preceding status-report turn added no algorithm evidence. This transaction
implements and independently checks the next available structural hypothesis.
Formal 1870/2413 and independent historical-origin E0 61/400 remain unchanged.

## Identity and uniform rule

For a continuous MAIN column j, the sole registered defining row is
`a*x_j + b*x_p = 0`, with positive power-of-two a, p<j, no binary coefficient,
zero RHS, no output occurrence, and exact float64 r=-b/a satisfying |r|<=1.
All remaining incident coefficient products must be exact and remain in the
unchanged [2^-20,2^40] coefficient window. Use the frozen census implementation
to establish these local conditions, never iid, verdict, margin or LP status.

Visit eligible columns in descending global-column order. Select each unless
blocked; block its parent. This deterministic independent frontier prohibits
selecting both a column and its parent. No repeated pass, chain-product rescue,
or per-row fallback after a failed collision is authorized in V1.

Substitute x_j=r*x_p in every surviving equality and inequality, erase exactly
the selected defining rows, and leave global column numbers/n_cont/frame_id
unchanged. Selected slots become unused; ordinary lowering can prune them.
Original-input prefix, radix and later ReLU slots are protected. Every binary
column, all other predicates and the value map are preserved. For any reduced
feasible point, setting x_j=r*x_p uniquely extends it to the original HZ; the
box of x_j follows from |r|<=1. The converse is restriction to retained slots.
This is a projection of redundant continuous definitions, not a convexization.

Merged column coefficients are summed on the exact dyadic integer grid 2^-72
(the fixed window implies all float64 input denominators fit this grid).
Conversion back to float64 must be exactly reversible and window-safe. Exact
zero is removed. Any inexact or out-of-window sum rejects the ENTIRE candidate.
Do not delegate unchecked addition to sparse sum_duplicates.

## Proof, accounting and fixed ceilings

Default off. Preserve the original HZ throughout construction and independent
audit. Return only the new HZ, four one-dimensional arrays (columns, parents,
ratios, defining rows), scalar dimensions/report and source/content seals.
Never retain the full old HZ behind a certificate. The independent oracle
uses Fraction for every changed predicate row and every erased definition,
checks all unchanged rows, boxes, binary coefficients, RHS, frame and output,
and verifies the deterministic selection separately in tests. Reconstruction
tests enumerate only small mathematical fixtures, not benchmark input search.

Require strictly fewer total coefficient nonzeros and strictly fewer unique
resident numeric bytes INCLUDING reconstruction arrays than the original
component. Also measure retained numeric entries, Python shallow metadata,
temporary construction, unchanged original digest and native matrix fidelity.
No broad cache deletion is authorized. Existing runtime roots are not changed.

Frozen component budgets: 256,000,000 logical work, 64,000,000 retained numeric
entries, 1 GiB construction (the existing measured_build gate, unchanged),
16 GiB address-space, tests 60 s, worker 240 s, one CPU, no GPU, no solver.
Work is census's existing bound plus eight input-coefficient passes, 32 per
MAIN factor, 64 per remaining selected incidence, and four times
sum(row_width * ceil(log2(max(2,row_width)))) over affected surviving rows.
Charge the sum bound before rewriting; observe it even when it rejects.

These are COMPONENT ceilings, not permission to add them to C9's existing
234,443,780-work construction and call the combination <=256M. Live acceptance
still needs one fused construction with its own preregistered whole-path work
proof, complete reachable-state accounting and the unchanged reference gates.
An offline component pass cannot authorize a terminal run, score, default,
CIFAR expansion or cohort/full replay.

## Tests, target and artifacts

First test signed/non-power-two ratios, dependency chains, sibling collisions,
exact cancellation, rejected inexact sums and windows, all predicate kinds,
binary/input/frame retention, Fraction feasible-set equivalence and extension,
default off, mutation/schema/cap guards, and complete certificate accounting.
Run all 405 inherited tests plus the new proof tests before touching the target.

One real target: the SEALED C9 Tiny143 final HZ (file SHA
841af01fb74ffa8cfdb7ac434a4f8da632739d866983f2b44ed0f34c353ed0b0), with defining
maps from the sealed live checkpoint (SHA
5bf82fc83205cd9b5f52187e164c70ce3c03abffd9d8bf352a643f38d2966a65).
Read both originals without rewriting. Exclusive output directory:
results/c10_alias_quotient_20260908_v1/. Freeze all inherited/new sources,
configuration and provenance; save tests, events, outcome, failure and exit
hashes automatically. Native ingestion is permitted only after all component
proof/storage gates; no presolve/optimization/terminal call is permitted.

On failure close V1 honestly; do not relax its limits or rerun with changed
selection. On pass proceed to separately preregistered fused live construction
on the SAME structure before any ordinary terminal capability attempt. Only
after its exactness/physical/capability gates may the same-structure CIFAR166/153
shadows, family replays and ultimately the complete 2413 replay advance.
