# M3 static audit: a missing gate–margin relation, with a bounded testable claim

2026-10-20, source revision `4edc5f8826f74d556879b2863a86b97fd572cd43`.
Branch `feat/moe-route-verification`; clean at entry. Root review plus two
separate read-only AI source/mathematical reviews and a primary-literature review.
No executable tests, native solver, trained checkpoint, data or GPU were used.
This is a static scientific/integration decision, not a measured result.

## Decision

**Continue to one default-off two-row representation control, not to a full
performance claim.** The exact same-domain separation survives a stronger N1
baseline; its premises can be checked without a new optimizer or supervisor.
However, four-slice N1 already proves the example, and generic nonlinear
verification already has affine sigmoid envelopes. The hypothesis is narrowly:

> On fixed guarded factor domains and fixed slice/gate/difference ranges, adding
> a checked relation between the existing lambda factor and the original router
> margin can exclude points admitted by ACT's current F0 relaxation.

This does not establish novelty over GenBaB, a full-request advantage, an H2
dominance theorem, route-changing certification or network-source closure.
M1b remains sealed as an overall failed attempt; M2 remains gated.

## Stronger baseline: exact algebra, not a new solve

Let x=m in [0,1], B(x)=9/16+x/4 and A(x)=B(x)-1. The property is
`B(x)-lambda`, with true lambda=sigma(x). The difference is exactly -1, so the
McCormick product is exact and there is no product-relaxation confound.

On the two fixed slices [0,1/2] and [1/2,1], even the tight sigmoid intervals
give the independent-gate minima:

| Slice | Tight-range old H2/F0 minimum | Sign |
| --- | --- | --- |
| [0,1/2] | 9/16 - sigma(1/2) | less than -3/80 |
| [1/2,1] | 11/16 - sigma(1) | negative |

Indeed exp(1/2)>3/2 gives sigma(1/2)>3/5, and exp(1)>5/2>11/5
gives sigma(1)>11/16. These statements do not use the weaker intervals in the
initial design.

The new relation lambda<=1/2+x/4 gives `B-lambda>=1/16` in both slices.
For the first slice, x=0 and lambda=1/2 attain 1/16 in the relaxation.
This establishes a positive union bound without claiming the true nonlinear
minimum is attained in every slice.

The old method can still prove this example with four fixed slices. For a slice
[l,l+1/4], its minimum is `f(l)=9/16+l/4-sigma(l+1/4)`.
Since sigma'<=1/4, f is nondecreasing and f(l)>=f(0)>0. An entirely rational
check of the last inequality is:

```
exp(1/4) <= 41/32 + (1/384)/(1-1/16)
         = 1849/1440 < 9/7,
sigma(1/4) < 9/16.
```

The geometric tail bound uses the first omitted term (1/4)^3/3! and
subsequent ratios at most 1/16. Therefore retain both the same-two-slice
comparison and a pre-fixed four-slice native cost reference. Do not tune the
number of slices after outcomes. This analytical example has no trained model,
binary factors or changing-route witness.
The displayed rational bound also gives the explicit four-slice margin
`9/16-1849/3289=17/52624>0`; it is not a measured native acceptance.

## Exact sign premises, before adding a relation

Reconstruct the canonical pair margin directly from the original stored router
coefficients as exact binary rationals:

```
m(z) = m0 + v*z,       |z_i| <= 1.
```

The binary factors also lie in [-1,1]. Do not first subtract router rows in
floating arithmetic and then call that subtraction exact.

The first implementation should accept only one of these finite proofs:

1. Exact factor-box sign: `m0-sum(abs(v))>=0` or `m0+sum(abs(v))<=0`.
2. An explicitly retained inequality `a*z<=b` and exact positive rational s:
   `a=-s*v, b<=s*m0` proves m>=0; `a=s*v, b<=-s*m0` proves m<=0.

The direct row must belong to the same bound source and complete retained
domain; its label is not proof. A floating N1 slice row that fails these exact
equalities cannot justify a sign even if its recorded range is nonnegative.
Cross-zero or unsupported cases keep the unmodified F0 domain and obligations.
They are not dropped, automatically split or certified. The zero case is valid
on both closed sides; no division by a zero margin coefficient is allowed.

For m>=0 the ideal rows are `-lambda<=-1/2` and
`lambda-m/4<=1/2`. For m<=0 they are `lambda<=1/2` and
`-lambda+m/4<=-1/2`. These follow from sigma(0)=1/2,
monotonicity and sigma'<=1/4. They are sound outer rows, not the exact graph.

## Factor mapping and native lowering

`weighted_top2.py` already retains shared expert factors, but its independent
lambda column has no affine link to the router margin. A frame ID alone does
not justify adding this link.

Require `shared_input`, bind the unmodified common source and explicit factor
map, and check that the router uses only the declared shared continuous/binary
prefixes. Expert-private columns must receive zero margin coefficients.
Continuous and binary maps are distinct. Reject mapping an original router onto
one of the independent-input ablation's copies.

There is an actual N1 row-order trap:

```
router: entry rows, slice rows
joint:  entry rows, A-private rows, B-private rows, slice rows
```

Thus the complete sliced router is not generally a leading constraint prefix
of the sliced joint source. Bind the unsliced common prefix and the explicit
retained slice rows separately; do not replace this check with frame equality.
Mutable arrays require content/ownership binding, not merely object identity.

Native F0 stores lambda as `C+R*eta`, rather than using lambda itself as a
factor. For `m=m0+v*z`, the two native rows must be:

| Premise | Relation row | Half-range row |
| --- | --- | --- |
| m>=0 | `-v*z/4+R*eta <= 1/2+m0/4-C` | `-R*eta <= C-1/2` |
| m<=0 | `v*z/4-R*eta <= C-1/2-m0/4` | `R*eta <= 1/2-C` |

Use the actual stored C, R and lambda slot at construction, not a separately
recomputed center/radius. The product slot has coefficient zero; R=0 needs no
division. A flattened native LP shifts binary columns by the *expanded*
continuous width, unlike the direct rational LP's explicit lambda variable.
For a claimed range embedding also check exactly `R>=0, C-R<=l, C+R>=h`.
Failure must not silently widen only the on arm or change the old producer.

For an ideal row `a*z<=b`, a stored binary64 row is an outer approximation if

```
stored_rhs >= b + sum(abs(stored_a_i - a_i)).
```

Check this with exact values of the stored floats, covering the union of
source and target nonzero columns. This includes subtraction, scaling, sparse
accumulation and underflow error; one final nextafter is not a proof.
All original F0 rows, output coefficients, bounds and factor ownership must
remain unchanged. Only the two explicitly checked inequalities may be added.

## Reuse inventory and limitations

| Existing component | Reuse | Remaining boundary |
| --- | --- | --- |
| `conditioned_difference_support.py` | Closed same-factor N1 slices | Floating projection/sign is not an independent exact sign proof |
| `check_hz_endpoints.check_pair_map` | Shared/private expert map and all original rows | Router-to-common-source binding still needed; existing bounded capacities remain |
| `rational_mccormick.build` / `check_construction` | Exact direct LP and independent reconstruction | Given source and justified rectangle only; old schema requires exactly four product rows |
| `checked_gate.propose` / `checker.check` | Independent rational sigmoid interval proof | Given margin only; fixed [-16,16], 24/40/24 bits and degree16; not a joint graph relation |
| `hz_sparse_rows` / `check_hz_sparse_rows` | Outward binary64 row check | Source <=4096 factors, c/b/ub each <=16 entries, total NNZ<=262144; target may have <=32 existing compensation columns; no implicit enlargement |
| `rowwise_bound.check_bound` | Final LP dual/residual lower-bound check | Does not establish the LP's source inclusion or a feasible counterexample |

The sigmoid source was initially missed by a narrower search; the explicit
`checked_gate/` directory was then read. No new sigmoid theorem implementation
or rerun of its historical candidates is required.

These interfaces do not yet compose into one cumulative online account:
the rational builder/checker and checked gate lack a caller Work/deadline;
rowwise_bound has a deadline only; sparse-row production and checking each
create local Work rather than continuing the common 8M ledger.
Do not present their current composition as one checked 8M request. Reuse the
existing ledger and supervisor with explicit cumulative accounting; do not build
a new measurement framework or charge the missing work as zero.
The sparse-row API accepts an owned-HZ reference, not arbitrary `(a,b)` rows:
a two-row wrapper still needs a zero dummy output and complete factor
identity/ownership. Its checker does not replace the independent final-native
diff check that the old F0 object gained only the intended two rows.

The smallest native experiment checks only the new rows and their premises.
Old gate derivation, floating F0 construction, network/guard lowering and native
solver acceptance remain explicit premises. A separate exact-LP route may
reuse rational McCormick, add independently reconstructed exact rows, and check
the final bound. Neither route automatically closes G3.

## Literature and direct-comparison contract

[GenBaB](https://arxiv.org/html/2405.21063v2) already supports general nonlinear
branching, per-subinterval affine bounds and optimized multiplication
relaxations. The official auto_LiRPA
[sigmoid](https://github.com/Verified-Intelligence/auto_LiRPA/blob/master/auto_LiRPA/operators/s_shaped.py)
and [multiplication](https://github.com/Verified-Intelligence/auto_LiRPA/blob/master/auto_LiRPA/operators/bivariate.py)
implementations support the corresponding operators. These moving code links
are literature/capability evidence, not pinned experimental configurations.
The shared-input fixed-pair graph `b+sigma(m)*d` has no identified
operator-expression obstacle from this review; complete guarded/tie-legal
end-to-end compatibility was not executed or established.

Do not claim generic solvers lose shared input or cannot refine gates.
An eventual external experiment must distinguish whole-box pair, guarded pair
and complete route aggregation, retain reasonable optimized/nonlinear-branching
configurations, and bind a revision before running. No deployment is authorized
by this audit.

For an internal single-factor control keep fixed all source matrices,
properties, slices, gate/difference ranges, binary treatment and query budget.
Use relation-row off/on with N1+F0; keep N1+H2 separately labeled, since its
checked relaxation can relax binary factors while native F0 retains them.
Test equal experts, constant gate, negative side/direction, ties, cross-zero
fallback, private factors, incorrect source/maps/signs, missing obligations,
inward rounding, deadline/Work exhaustion and unsupported premises.
Frozen counts and resource reception must precede execution. No M3 execution
attempt has been made or consumed by this audit.

## Continue / stop

Continue with minimal default-off representation implementation and independent
static review. Do not enable it in production or start M2/real/GPU requests.
Freeze a finite mechanism control only after exact sign/mapping/row reception
and the existing cumulative budget contract are connected.

Stop this implementation if its mapping or sign premise cannot be checked
without a new unbounded source system. Narrow the result if stronger native
comparators remove the complete-request advantage. No positive control, local
row check or improved exact bound closes G1–G6.
