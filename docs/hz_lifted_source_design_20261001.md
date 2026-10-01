# Source connection through checked rowwise binary64 enclosure

This finite integration follows the checked given-reference kernel at
`ad756e4561a36ef5fa1ad3d39f9e2bd050a360ff`. It changes neither its eight-local-factor
limit nor the existing source connection's 128-factor, 128-output, 256-constraint
limit. No real model, sealed input, native optimizer, physical GPU retry or new
performance comparison is admitted. It is not a new supervision wrapper.

## Row composition contract

Old complete source controls can exceed the kernel's four-constraint limit even
when every affected row is small. For each output, equality and inequality row,
project exactly its nonzero original coefficients into a local reference using
the original factor IDs, owners and kinds. Keep at most eight local factors;
refuse a denser row rather than split it into a different relaxation. Constraint
references have one dummy zero output, which may introduce no output error.

Run the unchanged inclusion kernel separately on these rows. Lift the returned
coefficients back to a single global vector retaining **all** original factors,
including globally unused ones. Allocate a distinct globally bound error/slack
for each nonzero local compensation. Local names are not a license to share new
variables between rows or experts. All other entries of each new column are zero.
Global binary offsets are recomputed after all new continuous columns.

The independent checker reconstructs each projection and mapping from the given
reference, checks the local proof, reconstructs the entire expected global HZ and
compares every coefficient/ID/owner/row. Thus, for any original feasible assignment,
all independent row extensions coexist and preserve the complete output vector.
This is the same L1 rounding outer enclosure composed by rows, not new tightening
or a loss of shared original-input relations. Within the old kernel's small
scope, compare coefficients after deterministic renaming of new factors.

## Source and output contract

Use the four unchanged declarations in `configs/hz_source_representation_20261001.json`
and their fixed hashes. Check the input box and every original affine/ReLU
reference with the existing exact source-step checks before accepting a lift.
Affine nominals come from actual sparse affine propagation; exact compensation
is still required, then enclosed, never silently replaced by approximate equality.
For ReLU, construct the exact registered HZ equations first, preserving the old
blocked row order after explicit permutation; lift any nonrepresentable result.
Instantiate every lifted step as actual SparseHZono with exact snapshot matching.

Conditional route entries restore the same input expression, inherit router
factors and add all pair-versus-outsider inequalities. Independently check that
exact reference before lifting. It is a conditional domain, not a claim that
guards hold throughout the original box. All unordered tie-legal pairs remain
obligations. Router-box sign evidence derives gate bounds from the checked lifted
router; inherited common factors keep their owners, new expert factors are private.

Prepare fresh endpoint requests/candidates on the new terminal matrices using
the unchanged CPU 128-step candidate algorithm and `1/10000000` threshold. No old
proof is reused. Missing evidence remains UNKNOWN, negative lower bounds do not
establish unsafety. Identity and mutation checks run before final acceptance.

## Fixed controls and reporting

Eighteen groups are frozen: row kernel differential; shared global row identity;
row binding mutations; row capacity refusal; all four complete source rosters;
source-step reference mutations; guard and ownership; layer inventory; gate and
property binding; missing pair; partial endpoint evidence; stale endpoint evidence;
diagnosed affine rounding bridge; nonrepresentable ReLU bridge; guard rounding
bridge; expiry and source mutation; checker independence; cost/archive inventory.

The three rounding bridges are operator controls, not additional model requests.
They use the already diagnosed affine residual and guard difference, and a
binary64 source ReLU with center `1/2` and coefficients `1,2^-55`, whose exact
endpoints/RHS require enclosure. A sparse nine-original-factor/at-least-six-row
composition tests global identity beyond local width; an individual nine-factor
row must refuse. Existing eight kernel patterns provide the row differential.

Each full source case uses one cooperative at-most-300-second deadline covering
creation, propagation, proposal, serialization and checking. All costs and failures
are retained; imports/orchestration are separate, no production speed claim.
Report all actual endpoint counts, not a success requirement of matching old
positive counts. The fixed unsafe declaration must not receive a positive result.
Check saved complete/partial packages independently and retain attempts. Stop on
implementation failure to repair it; do not tune fixtures, gates, rows, proposal
iterations, capacities or deadlines to improve positivity. All G1–G6 remain OPEN.
