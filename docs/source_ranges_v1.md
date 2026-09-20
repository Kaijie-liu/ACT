# Scoped, independently checked ranges for new source states

## Stage and motivating evidence

This is an **optional source-step interface and finite analytic controls**,
not a rerun of input98 and not a production policy change. The all9 fresh-LP
attempt remains0positive,7checked nonpositive,2missing. The near-equality of
checked and reported objectives makes arithmetic precision a poor next target;
saved vectors also do not isolate gate-only failure. See
`full_bounds_v1_results.md`. No old output, gate or range certificate is
transplanted onto a new source.

The previous affine lift and ReLU source checker used generator-box ranges.
The new `source_ranges` interface can consume strictly scoped range facts that
use the source constraints, but only after exact independent checking. Its
first consumers are affine recentering and ReLU graphs. Gate parameterization,
McCormick segmentation, task selection, production tolerances and solve limits
are unchanged. This stage neither searches numerical backends nor introduces
a new exact linear algebra engine. The [control receipt](source_ranges_v1_controls.json)
binds implementation hashes and the analytic outcomes below.

## Range fact contract

For a supplied source HZ `H`, a property row `a` and offset `b`, reconstruct
exactly the expression `f(xi)=a^T H(xi)+b` in the **same continuous/binary factor
frame**. Export the original equality/inequality rows with the binary factors
relaxed to[-1,1]; no guard row is discarded. The source hash includes all
coefficients, constraints, factor identities and frame identity.

Each fact additionally binds the external request identifier, proof-domain
label, layer label, output-row index and canonical source-output expression.
These labels are provided by the caller/externally pinned statement, not inferred
from the numerical solver. A label does not independently establish network
semantics or domain inclusion. Cross-source reuse is rejected even if someone
believes the new domain is smaller; no new transport rule is introduced here.

Two signed-dual candidates are required:

`L <= min f(xi)` and `N <= min -f(xi)`.

The unchanged standard-library sparse LP checker evaluates both using exact
rational residual-box correction. A requested interval[l,u] is admitted only
if `L>=l`, `N>=-u`, `l<=u`, and it does not widen the original generator box.
Solver status, approximate optimality and an asserted scalar bound are not
acceptance inputs. The two full LP identities, including objective sign and
offset, are reconstructed from the current source.

Every output row has an explicit roster entry. `None` means **use the original
sound generator box**, not skip this output or claim it proved. A supplied
but invalid/incomplete fact rejects; it is not silently downgraded to fallback.
Missing rows also reject. This differs from a producer deciding before submission
to provide no fact for a row, in which case the fallback remains visible.

## Affine consumer and pointwise extension

Let the exact affine expression be `c+g*xi+h*beta`, with checked range[l,u].
Set `m=(l+u)/2`, `r=(u-l)/2`. If r>0, add one fresh factor eta in[-1,1],
output `m+r*eta`, and the exact equality

`g*xi+h*beta-r*eta = m-c`.

For every valid old assignment, choose
`eta=(c+g*xi+h*beta-m)/r`. The checked range places eta in[-1,1], and the
defining equality and output are exact. All old constraints and factors remain.
The nonzero RHS is essential when the interval is recentered; omitting it is
explicitly tested and rejected.

For r=0, no new factor is needed. The output is constant m, and a nonconstant
source expression still retains its redundant defining equality
`g*xi+h*beta=m-c`. The range proof justifies this equality on the entire source
domain. A constant expression with no terms needs no extra row. Thus zero width
does not become division by zero, an unproved reset or loss of correlation.

With every fact absent, the representation exactly matches the existing
generator-box affine lift, including factor names and equalities.

## ReLU consumer

The independent checker first verifies the same-source range of each input
coordinate. It preserves the input expression if l>=0, produces zero if u<=0,
and otherwise reconstructs the existing exact binary ReLU graph using l,u.
Zero is handled deterministically by the active branch, without excluding
any legal input. For an unstable coordinate, fresh negative/positive continuous
factors and a sign factor encode

`a_pre = l/2*(xi_negative+sign) + u/2*(1-xi_positive)`.

The output is `u/2*(1-xi_positive)`, with the same sign constraints as the
existing encoding. The checker reconstructs the graph, maps and all inherited
constraints without calling the producer. This changes the justified range,
not the intended ReLU or its tie behavior. Relaxing the binary variable later
still yields an outer approximation, not an exact continuous network.

## Controls and observed analytic outcomes

Nine new tests, plus46existing regressions, cover:

| Control | Result/acceptance requirement |
|---|---|
| Source x in[-1,1] with checked1/4<=x<=3/4 | Exact dual range[1/4,3/4] |
| Recentered affine output | m=1/2, r=1/4; defining RHS1/2; old assignments extend |
| Four-row ReLU, two proved ranges/two fallback rows | New binaries3→1; active2/inactive1/unstable1 |
| Tight unstable ReLU including x=0 | Exact assignments satisfy graph/output at endpoints and interior controls |
| No-fact path, including an existing binary source | Exact structural differential with old affine/ReLU constructors |
| Correlation x+y=0, zero-width projected range | No fresh value factors; defining equality retained |
| Two expert outputs in one shared frame, three classes | Both newly generated weighted properties checked, bounds1/5 and3/20 |
| Wrong source/request/scope/layer/row/expression or weak/invalid dual | Reject, including wrong upper-objective identity |
| Missing row/side, inward/reversed range, RHS/constraint/factor corruption | Reject |
| Moved affine and ReLU single-step bundles | `python -I -S` succeeds without model, solver or producer |
| Six hash-rebound moved semantic mutations | All reject |
| Expired checker deadline | No checker starts; no positive result |

The analytic weighted example is a check on supplied correlated sources, not
a trained MoE request or proof of route coverage. Its positive bounds do not
establish a speedup or a coverage advantage over optimally solved old LPs.
All new candidate multipliers in these controls are elementary exact vectors;
**no native optimizer or real-model query is used by these new controls**.
The broader regression suite retains its previously existing tiny solver test.

Run with act-py312:

```sh
python -m unittest source_ranges.tests source_ranges.portable_tests
```

The moved interface validates a single source step, with source enclosure and
operation-to-network correspondence still explicit assumptions. It does not
replace the complete-source checker or automatically upgrade an old full
request. Caller code must bind each source transition and ultimately construct
and check all output obligations on the resulting matrices.

## What remains before any real comparison

This interface has **not** been connected to the complete convolutional trace
or to a native range-candidate generator. It creates no real new SAFE.
The source/LP conversion is sparse, but this control implementation constructs
and parses source LPs per range obligation; no claim of full-layer scalability
or cheaper range proofs is made.

The next bounded integration must preserve the complete source trace, explicit
fallbacks and new-matrix obligation identities. Candidate range generation,
exact checking, subsequent propagation, serialization and final output checks
must share the unchanged request budget. Its layer/row roster and failure
rules need freezing before actual outcomes. Do not silently evaluate more
rows, reuse old-matrix certificates or trade unchecked ranges for speed.
First test that orchestration and total cost; only then freeze a limited real
comparison. No cohort expansion, longer old limits, retraining or new gate
search is authorized by the analytic controls.
