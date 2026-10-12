# Factor scales and complete-query capacity: analysis, not execution

2026-10-13; reviewed code `1b171c97fc5b88eab148d618f34a4ddf5b852f46`.
**Decision: do not implement/run factor-scale separation as a standalone capacity
repair.** It removes real work, but the current three-pass exact consumer still
exceeds 8M even for the already fixed two-expert synthetic source. No large
source, candidate, solver, GPU or real-model call was made for this analysis.

This follows [later-affine V2](hz_affine_references_20261013.md), not another
small correctness/timing experiment. The [ledger](hz_factor_scale_capacity_20261013.json)
binds the reviewed code and old source archive; the
[stdlib calculator](../scripts/analyze_factor_scale_capacity.py) checks identities
and finite integer/rational arithmetic without loading raw tensors or a forest.
It is not a newly checked source/output certificate or a wall-clock prediction.

## Sound candidate representation

For each clipped input interval `[l_j,u_j]`, independently reconstruct

\[
c_j=\operatorname{nearest}((l_j+u_j)/2),\quad
D_j=\operatorname{upper}(\max(c_j-l_j,u_j-c_j)).
\]

Keep the same normalized, distinct global factors `xi_j in [-1,1]`. The proposed
local bank L stores unscaled first-layer source weights; D contains their input
radii. Fresh compensation/ReLU factors keep D=1 and their original shared/private
ownership. Later reference matrix R remains strictly lower triangular:

\[
C=(I-R)^{-1}LD,\qquad
B_n=\sum_j |L_{nj}|D_j+\sum_p|R_{np}|B_p.
\]

The latter is the sound triangle radius used for ReLU, not an exact expanded L1
norm. With adjoint `v=L^T(I-R)^(-T) alpha`, support subtracts
`sum_j D_j |v_j|` **after** all contributions to each shared factor are combined.
Equal radii never identify different factors. Gate planning must use the same
weighted norm. Native forward scales its input by D; native adjoint scales its
output by D. Constraint rhs and centers are not scaled.

First-layer centers `b_i + sum_j W_ij c_j` and their compensation remain
independently checked. The checker must check all source weights, factor maps,
rows, guards and suffix bytes. Zero radius permits no division and does not erase
a nonzero center contribution. Neither raw request.radius nor a pre-clipping
center is a substitute for these reconstructed intervals.

Current producer/checker each calculate `W_ij D_j` and its coefficient roundoff
error for every first-layer weight. Keeping W and D exact separately removes
those operations on both sides. Weighted norms add work back; equal-radius
grouping can save multiplications but may not merge coordinates. Nonzero input
centers still incur Wc work. This changes the enclosure and compensation inventory,
so it needs a new schema and fresh evidence, not V2 bound reuse. Storing every
input coefficient as a node reference would exceed the 65536-reference limit;
the 256x3072 weights must remain a local/source-row bank.

## Why the standalone change is refused

The fixed captured source is the old dense recipe, SHA-256
`0745027b58f00c608a9db921204cb2feae58273600fee658d4c9ccc3d8c19dbc`:
3072 inputs, two experts, two classes, one legal pair, one property, two gate
endpoints. It is not the unrelated eight-expert/ten-class fixture.

First-layer weights number `N=(2+256+2)*3072=798720`. For the proposed local bank
and independently built weighted norms, producer visits/sums require at least
2N; checker parses/sums and compares source rows for at least 3N. Thus the
first-bank subtotal alone is 5N=3993600. This is a **conditional implementation
cost model**, not a theorem that every sound source checker must spend 5N.

The recipe's positive weight magnitudes repeat every 31 columns. Since
`3072=99*31+3`, each first-hidden-row radius lies between
`D*(99*496+3)/32` and `D*(99*496+93)/32`. Both lie strictly between 6 and 97/16.
Centers `(i-2)/16` therefore give 99 unstable rows (0..98) and 157 active rows
(99..255), with none inactive, also under the proposed exact W/D representation.
Oppositely signed router rows yield an interval crossing zero, retaining both
gate endpoints 0 and 1.

The only constraints come from the 99 unstable ReLUs; no pair guard exists with
two experts and top-2. Their references cannot reach the 157 active first-layer
nodes. At t=1 those nodes' adjoints are the final two class-row differences,
which are nonzero because the recipe assigns opposite signs to the two rows.
At t=0 both expert1 output-root adjoints are likewise fixed at +1 and -1.
Consequently each complete proposed, zero or selected-final pass, **across both
endpoints**, necessarily visits at least `157*3072+2*3072=488448` local entries:
482304 at t=1 and 6144 at t=0. Global residuals can cancel, but the current
node-wise consumer has to read those entries first.

This invariant is specific to this graph. On a general DAG, candidate duals may
cancel an objective's node adjoint, so zero-candidate cost is not a universal
lower bound on every candidate. No such general claim is made here.

| Necessary partial work under the stated implementation | Counts |
|---|---:|
| Independent first-bank construction/checking | 3,993,600 |
| Proposed + zero + final exact passes: 3 x (visit + multiply + add) x 488448 | 4,396,032 |
| Subtotal, even with free native preparation | **8,389,632** |
| Native first-bank preparation, additionally | 798,720 |
| Subtotal including that preparation | **9,188,352** |

The unchanged cap is 8,000,000. Metadata, later layers, auxiliaries, references,
gate planning, residual norms, optimization, serialization and terminal work
are omitted from these lower subtotals, not free in a real request. Counts mean
the current `Work` calls, not FLOPs or seconds. The archive's old 7202670 prefix
count is not transplanted onto the unimplemented scaled representation.

## Next bounded research question

Do not repeat the refused query, raise caps, change count definitions, remove
the zero candidate or trust native bounds. Native-buffer caching alone cannot
remove the 8389632 exact/source subtotal.

The next candidate is **multi-target exact residual consumption**, considered
together with the scale representation before any implementation admission:

1. Compute an independently checked, identity-bound objective pullback once;
   combine it with each dual-dependent pullback before weighted absolute values.
2. Traverse shared coefficient rows for bounded candidate columns, retaining
   the actual multiply/add cost of every active column; batching is not free.
3. Let the independent checker check both proposed and zero candidates and
   issue a sealed result. Terminal aggregation can consume that result rather
   than redo the identical selected candidate. A portable checker still
   reconstructs it from raw source and all raw candidate evidence.

This is not the current untrusted aggregator API and is not yet implemented.
Receipts must bind source/schema, complete pair/property/endpoint inventory,
candidate bytes, factor map, numerical policy and the same cumulative budget.
Missing, stale, cross-request, partial or deadline-exceeded evidence must not
become a positive terminal. The final lower bound must agree exactly with the
old consumer for the same representation and candidates.

The checker must construct the plan itself, bind constraint ordering and both
exact bounds/selection, and issue receipts only in its request-owned Work
session. Caller-constructed receipts/plans and cross-session reuse are rejected.
Keep at most one bounded tile of residual vectors and release them after
checking; terminal receipts contain small scalar results, not duplicated dense
matrices. Portable JSON receipts are not trusted proof inputs.

For this fixed graph, the new path's large-bank main terms would be
`5N + N + 3S + 5T = 7778304`, where S=488448 and T=99x3072=304128.
The 5T term charges one shared coefficient visit and two multiply/add pairs for
two potentially nonzero dual columns. Only 221696 remain for **all** omitted
work. This is not a complete upper bound and not an execution admission. Source
metadata, all layers, groups, gate bounds, RHS/references, residual merges,
weighted supports, receipts, serialization and terminal handling still need an
explicit conservative account. No assumption of zero proposed duals is allowed.

Splitting objective and dual work can also **increase** work when they cancel
at a node before its dense ancestors are visited. A next cost model must cover
that case and all new norm/merge/receipt/serialization costs. A partial subtotal
below 8M is not admission; a complete conservative ledger must fit, otherwise
stop that implementation path before another small-control series. Preserve
separate representation and consumer ablations if eventually admitted.

## Reproduction and scientific boundary

```
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S scripts/analyze_factor_scale_capacity.py --check-report docs/hz_factor_scale_capacity_20261013.json
```

Two separate read-only reviews covered source/scale soundness and endpoint cost;
the root independently inspected the recipe, source checker and three-pass
consumer. These are AI/code/math checks, not independent human review. The
analysis changes the next-action decision, not model performance. All G1–G6
remain OPEN; no new output certificate, GPU acceleration or external advantage.
The physical GPU refusal and all sealed requests remain closed. Disk inspection
showed 87 GiB free at 98% use; only this compact analysis is added, no raw copy or
cache experiment, and no scientific record is deleted.
