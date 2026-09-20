# Frozen property-directed range-row comparison

Status: **PREPARED ONLY; real comparison NOT STARTED**. See the freeze and
control receipt alongside this document. This is a new, finite comparison,
not a retry, extension or relabelling of `range_pipeline_v1`.

## Question and scope

Can a property-directed choice of the **same number of hidden affine rows**
yield more useful checked output bounds than the old fixed-prefix choice,
within the same complete-request budget? Saved-point diagnosis motivates this
question but does not prove a relaxation gap, a causal bottleneck or a gain.

Use only the already observed convolutional input98, the same captured input,
parameters, represented2/255 box, label0,10 classes and sole legal pair[1,2].
The model is67.06% clean accuracy; this single-route control cannot establish
high-accuracy or route-changing certification. No new sample, checkpoint,
training, output attack, old LP point or old dual is an execution input.

## Exactly one selection intervention

Both new arms independently reconstruct the input enclosure, compensated
prefix, remaining expert layers, join and **all nine new output LPs**.

| Arm | Hidden layer6 rows receiving two-sided range queries |
|---|---|
| `prefix` | Rows0,1 in each of experts1,2 |
| `property` | Top2 property scores **within each expert**, below |

Preserve the old active arm's quota: four rows total, at most eight native
range calls, each at most3s. This is **not** global top4 allocation across
experts, nor a comparison against the zero-query `range_off` arm. Keeping
two rows per expert avoids changing expert-budget allocation at the same time.
Both arms now pay the same scoring and snapshot procedure; only the selected
row set differs. Prefix results must be freshly generated here, not copied
from the prior comparison with different preparation costs.

Before hidden layer6 is transformed or any range call for that expert is
started, project its unmodified upstream shared HZ through every layer6 row.
Let its generator-box range be `[l_j,u_j]` and its final classifier weights
be `W`. For the registered clean class `y`, calculate with exact rationals:

```
gap_j   = -l_j*u_j/(u_j-l_j) if l_j < 0 < u_j else 0
harm_j  = max(0, max_{k != y}(W[k,j] - W[y,j]))
score_j = gap_j * harm_j
```

`gap` is the local unstable-ReLU triangle's maximum vertical excess; `harm`
is the largest negative coefficient magnitude among **all** required
classification margins. The product is a heuristic potential for harmful
local slack, **not** a whole-LP improvement bound, a witness or an importance
estimate from past solutions. Earlier-layer and weighted-product obstacles
are not assumed solved. Final classifier bias is bound in the identity but
does not affect this local excess coefficient.

Rank by decreasing exact score, break ties by ascending row index. Choose
two per expert, then query the chosen rows in ascending row order, lower
side before negative-upper side. Zero-score rows fill any remaining quota;
no further rows replace a failed or incomplete query. For synthetic widths
below2, quota is `min(2,width)`; the frozen real widths are64 in both experts.
Do not hardcode the previously observed rows or choose a property according
to current output-bound signs. Scores use the pre-query source, not the new
ranges or a source modified by this arm's range results.

Each expert's full score/range inventory and selected roster is published
before its first query. Bind source, request, pair/expert scope, layer,
classifier, clean class and class count. An isolated checker independently
projects the upstream source, decodes classifier coefficients and replays
selection; it never calls the producer or the solver. The terminal audit
also compares the arms' available pre-query source/score snapshots. Missing
snapshots are missing, not presumed agreement.

## Unchanged mathematics, calls and acceptance

Reuse the frozen range proposer, exact signed-dual checks, batched range
consumer/checker and complete source transitions. Both sides of a row range
must check against the same source and expression. Missing evidence causes
an explicit generator-box fallback. Invalid supplied evidence rejects the
build. Every original factor and constraint remains covered.

Keep gate[0,1], difference-range rule, McCormick construction, binary-factor
continuous relaxation[-1,1], route coverage, source semantics and output
acceptance unchanged. All nine competitors remain obligatory, in ascending
order, one output proposal each at most16s. Reconstruct output matrices and
generate fresh dual candidates; old positives or negatives cannot be attached
to new matrices. No retry, precision fallback, extra range call or query
substitution is permitted. Counts are fixed rosters/caps, not a promise that
every call finishes before the deadline; aborted and missing work stays visible.

Use existing act-py312, CPU/one thread, SciPy1.16.3, NumPy2.3.5 and bundled
HiGHS1.8.0, `method=highs`. Do not use highspy1.14.0 or install dependencies.

## Same300s per arm, including selection

The existing frozen outer supervisor is reused without modification.

| Phase | Cap, additionally bounded by real total remainder |
|---|---:|
| Input/prefix/all-layer generation, scoring, range proposals and exact prechecks | 100s |
| All output proposals | 120s |
| Seal | 5s |
| Independent complete source, selection and output checks | Remaining time |
| Terminal publication reserve | 2s |

Scoring, score snapshots, source serialization, numerical conversion/native
solves, exact prechecks, all LP construction, packing, imports/startup,
inventory hashing and publication are charged. Phase caps do not sum to a
larger allowance. Historical parameter/input/nominal-first-Conv **capture**
is excluded as before; all propagation is fresh. This is stored-source proof
pipeline cost, not time including checkpoint loading. Selection time is a
subcomponent of build, not an extra budget. Range/propagation/serialization
subcosts must not be double counted. Separate post-terminal moved rechecking
and audit are reported separately, not deducted from execution.

Build/check timeout stops that arm with partial artifacts; output proposal
timeout may seal/check published partials in the remainder. Errors or missing
obligations cannot establish positivity. Late terminal publication cannot
establish a positive result. Only owned process groups may be stopped.

Fixed prefix-then-property order; each arm independently gets300s,600s total
execution ceiling excluding separately reported audit. No repeated timing
trials, resume or effect-guided early termination. Resource gate is checked
before each arm. A not-started arm remains in the two-arm denominator. These
fixed-order, one-input observations are not a population or speedup estimate.

## Endpoint and finite interpretation

Report complete-request outcome, all9 checked-positive/nonpositive/missing
bounds, common-property bound differences, selected rows, proposed/accepted
range facts, checked tightening and ReLU-status changes, new source dimensions
and identity, complete phase costs, selection subcost, package size and audit
cost. Do not compute paired bound improvement for a missing bound as zero.

Only all required bounds checked strictly positive permit
`CHECKED_POSITIVE_DECLARED_REAL_MOE`. Otherwise retain UNKNOWN, ERROR or
TIMEOUT as appropriate. A checked nonpositive lower bound does not imply
model unsafety or an intrinsic LP gap. The declared graph/parameter identity
and checker/interpreter execution remain assumptions; native floating-point
execution and production numerical acceptance are not upgraded.

If selected ranges tighten but the complete request stays nonpositive, report
the separation. If there is no checked range improvement, report that instead.
Neither outcome licenses more rows, more seconds, a gate change or further
same-request effect search under this protocol. Compare with the new prefix
arm, not by replacing the historical range-on table.

## Execution boundary

This preparation stage runs **synthetic controls only**. The new real output
directory must remain absent at freeze. Code and captured-source hashes are
fixed in `docs/property_ranges_v1_freeze.json`; no historical outcomes are
inputs. After a subsequent execution instruction, run once, then audit:

```sh
python -m property_ranges.run
python -m property_ranges.review data/moe/results/property_ranges_conv98_20260920_v1
```

Keep both outcomes and partial artifacts, archive compact evidence and a
derived comparison without editing the old results. Do not restart CROWN,
PyRAT, gate tuning, exact elimination, training or the sealed holdouts.
