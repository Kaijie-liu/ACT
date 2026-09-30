# Gate interval endpoint support for weighted top two

This is an elementary affine endpoint identity used to organize MoE proof
obligations, not a new general optimization theorem. It removes the product
relaxation of an **independent interval gate**, not the dependence of the true
softmax gate on the input. The current prototype checks given LPs; connecting
them to a declared network remains a separate obligation.

## Statement and proof

Fix a legal unordered pair S = {a,b} and an outer domain P of its shared-input,
guarded expert states. Shared input factors stay shared and private expert
factors stay private. For a linear property q and constant c, put

    A(z) = qᵀ E_a(z) + c
    B(z) = qᵀ E_b(z) + c
    d(z) = A(z) − B(z).

Assume that every true execution on this route embeds in P and its normalized
weight on expert a lies in a justified interval [l,h] contained in [0,1]. Then

    inf { B(z) + t d(z) : z ∈ P, t ∈ [l,h] }
      = min( inf_P [B + l d], inf_P [B + h d] ).

For fixed z, the expression is affine in t, so its minimum on the interval is
the smaller endpoint. Taking the infimum over z commutes with a finite minimum.
This proof does not require P to be convex. If P is empty, infimum is +infinity;
if an endpoint is unbounded below, the identity holds in extended-real terms.
The implemented LP check requires finite rational variable bounds.

Consequently two independently checked bounds L_l and L_h imply the valid bound
min(L_l,L_h) for this branch/property. When l=h, only one distinct endpoint is
needed. Accept a complete request only when every necessary pair/property has
both endpoint obligations discharged, or a separately checked valid reuse or
route-exclusion proof. A negative lower bound is not a network counterexample.

The endpoints need not correspond to attainable softmax values. That is safe:
the independent interval problem overapproximates the true gate relation.
Changing the pair orientation requires swapping A,B and changing the interval
to [1−h,1−l]. With [0,1], the endpoints are B and A, recovering the expert-wise
sufficient condition rather than an additional weighted verification ability.

## Comparison with range only McCormick

Suppose both methods use the **same P, same gate range and same affine forms**.
Add a valid range [d_l,d_h] and the four usual McCormick planes for w=t*d to
the comparison arm. The true independent-gate product graph is contained in
that relaxation. Therefore its exact optimum cannot exceed the endpoint
minimum above. This is a statement about exact optimization, not the quality
of finite-budget proposed duals, runtime, or coverage of an implementation.

If the McCormick arm also has nontrivial gate-to-input coupling cuts, this
comparison does not automatically apply. Likewise, it is unfair to give only
the endpoint arm a tighter P, a better gate interval or free source analysis.
Valid difference-range constraints may be included in the common P; they must
then be included in both arms. The endpoint objectives do not otherwise need
a difference-range support query, a product variable or four product planes.
They can need two support solves instead of one, which must both be paid for.

## Exact three expert separation control

Use real rational x in [−1,1], c=1/10 and scores

    r_0=x−1, r_1=0, r_2=−x−2.

Expert i has its own p_i=ReLU(x), n_i=ReLU(−x), with margins

    A = (p_0+n_0)/4 + 3x/4 + c
    B = (p_1+n_1)/4 −  x/4 + c
    C = (p_2+n_2)/4 −  x/4 + c.

Outputs are [margin,0]. P includes each private ReLU triangle

    p_i ≥ 0, p_i ≥ x, p_i ≤ (x+1)/2,
    n_i ≥ 0, n_i ≥ −x, n_i ≤ (1−x)/2,

and the relevant pair guard. Do not add equalities between different experts'
ReLU variables. In particular, A−B=x is true of the real functions but is not
an equality on this private-variable relaxation.

Pair {0,1} is legal for x≥−1/2 and has weight on expert 0 in [0,1/2]. Its two
endpoint certificates follow from nonnegative constraint combinations:

    B−c = ((p_1−x)+n_1)/4,
    (A+B)/2−c = [p_0+(n_0+x)+p_1+(n_1+x)]/8.

Pair {1,2} is legal for x≤−1/2, and the weight on expert 1 lies in [1/2,1].
Both B and C are at least c by the same individual constraint identities, hence
the endpoints B and (B+C)/2 are at least c. These gate ranges follow from the
affine score ordering and sigmoid(0)=1/2; no estimated sigmoid evaluation is used.

Pair {0,2} would require r_1≤r_2, but r_1−r_2=x+2≥1. The prototype still retains
both of its endpoint duties and supplies exact dual bounds using the conflicting
guard and input box. Their positive bounds are vacuous on an empty P, not evidence
of a reachable third route. At x=−1/2 both {0,1} and {1,2} are legal and covered;
x=−1 and x=1 give distinct unique pairs. A at the legal tie is −3/20, so the
complete output proof is not reducible to proving every expert individually safe.

For pair {0,1}, the box-derived difference range [−3/2,3/2] is valid. In the
same P and gate range, the McCormick point

    x=0, all p_i=n_i=0, t=1/4, w=−1/8

is exactly feasible and has objective c+w=−1/40. Thus this LP cannot have a
positive lower bound. This is not a real-network counterexample and −1/40 is
not claimed to be its optimal value.

Nor is this just an overly wide difference range: real points x=−1/2 and x=1
on pair {0,1} have differences −1/2 and 1. Any valid enclosing interval must
satisfy d_l≤−1/2 and d_h≥1. At the displayed relaxation point, the McCormick
lower planes are d_l/4≤−1/8 and −d_h/4≤−1/4; the upper planes are
−d_l/4≥1/8 and d_h/4≥1/4. The same negative point remains feasible even at
the narrowest interval compatible with those two witnesses. This argument
isolates a product-relaxation gap **on this constructed example only**.

## Implementation and remaining assumptions

[Builder](../../../../scoped_source/endpoint_build.py) constructs rational
objectives B+t(A−B). The [checker](../../../../scoped_source/endpoint_check.py)
independently reconstructs t*A+(1−t)*B and preserves the supplied LP constraints,
variable bounds and names. It uses the existing exact residual-compensated dual
kernel; it does not import the builder or a numerical solver. The expected request
hash is external to the received proof and fixes the complete pair/property list.

The generic result is `CHECKED_POSITIVE_GIVEN_BASE_AND_GATE`, never source-complete
or deployed-float SAFE. A hash detects drift from a supplied request; it does not
prove that P encloses a network or that a gate interval is justified. The algebra
above supplies those premises for this fixed analytic model, but the generic
source adapter is not yet implemented. A future float64 capture of 0.1 must use
its exact binary value, not silently substitute the rational 1/10 used here.

See the [control report and next gate](../../../../docs/h2_gate_endpoint_20260930.md).
