# D014 companion — Laminar factor, source gluing and closure limits

2026-09-28, `redu-hz`, HEAD `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.
Paper only; no new numerical run after D003. No baseline/default change.
This note supports [the exact transfer](D014_EXACT_TRANSFER.md), not a new
permission to replace original source relations by independent amplitudes.

## 1. Standalone guarded-amplitude factor

Let L be a laminar family of subsets of [k], integer capacities
0<=U_G<=|G|, and u_i>=0. Define

    K={(e,sigma): sigma in {0,1}^k, sum_{i in G}sigma_i<=U_G,
                 0<=e_i<=u_i*sigma_i}.

This equals e_i=u_i*sigma_i*t_i with independent LOCAL t_i in [0,1]. It does
not justify forgetting t_i if t_i is an original correlated neural source.
The capacity supports form a classical laminar matroid
([Fife--Oxley, introduction](https://arxiv.org/pdf/1606.08354)); no novelty is
claimed for this geometry or its maximum-weight independent-set problem.

The exact standalone convex hull in all (e,sigma) coordinates is obtained
by replacing sigma in {0,1}^k with 0<=sigma<=1, keeping the displayed rows.

Proof. Laminar counts have an integral flow representation: send selected
leaf flow through its ancestor sets, with integer capacities. Therefore
sigma=sum_j lambda_j sigma^j for feasible binary supports. Put
q_i=e_i/(u_i*sigma_i) if its denominator is positive and q_i=0 otherwise.
Then 0<=q_i<=1, and e_i^j=u_i*sigma_i^j*q_i gives a simultaneous decomposition
(e,sigma)=sum_j lambda_j(e^j,sigma^j) into K. Necessity is immediate.
This proof is not a runtime phase-enumeration algorithm.

For an affine row y=b+w*e let M_L(p) denote the maximum sum of nonnegative
profits over feasible supports. Then exactly on K,

    max y=b+M_L(u_i*max(w_i,0)),
    min y=b-M_L(u_i*max(-w_i,0)).

Descending-weight matroid greedy computes each maximum. With a validated
tree of depth D, a straightforward implementation costs O(k log k+kD) and
O(k+|L|) work storage, plus tree input/validation and coefficient bit costs.
m output rows require m such queries and coefficient formation. A candidate
optimizer may omit zero-profit choices; the domain may NOT delete their bits.

Control: u=(1,1,1,1), capacities1 on {1,2} and {3,4}, total capacity2,
w=(3,2,1,0). The global budget alone permits5; the laminar factor gives4,
attained at support{1,3}. With an energy-only factor sum e_i^2<=E, the analogous
support is sqrt(E*M_L((w_i^+)^2)). The minimum of separate box and energy
supports is only a sound bound for their intersection, not its exact support.

Native factor cost, before source definitions: k continuous amplitudes,
k original bits, k coupling inequalities e_i<=u_i*sigma_i and |L| count rows,
at most 2k+sum_G|G| nonzeros; nonnegativity/bounds, RHS and metadata additional.
The same rows already define an ordinary HZ formulation. There is no precision
gain over ordinary HZ supplied those same rows for a standalone linear query.

For y=c+B*e, affine/Conv changes the readout to Wc+d, WB; shared Add/Concat
combines readouts over the SAME frame. Truly independent frames allow direct
sum of the forests; arbitrary overlapping merged budgets need not be laminar.
After arbitrary neural source predicates are retained, the standalone support
is only an outer bound and its optimizer is not a concrete-network witness.

Original zero phases matter: support(e)<=U alone does NOT prove sum sigma<=U.
For g=(0,0), e=0 has empty support while two original active bits may equal1.
Budgets on original bits require proof on their full original relation.

## 2. Crossing budget failures

Groups {1,2} and {2,3}, each capacity1, with profits(2,3,2), defeat greedy:
it takes {2} with value3, but {1,3} gives4. If the third crossing group{1,3}
also has capacity1 and u_i=1, e=sigma=(1/2,1/2,1/2) satisfies every relaxed
row, whereas every feasible integer support has size<=1. Thus the relaxed
factor is not its hull. Laminarity is substantive, not a speed hint.

## 3. Even ideal individual gates and ideal factors cannot be glued freely

Let (x,t) in [-1,1]^2,

    f1=x+t+3/2, f2=-x+t+3/2,
    e_i=ReLU(-f_i), sigma_i=1-beta_i, r_i=f_i+e_i.

The exact joint relation has sigma1+sigma2<=1 and u1=u2=1/2.
Consider the fractional tuple

    x=t=0, sigma=(1/2,1/2), e=(1/4,1/4).

It lies in the ideal standalone amplitude-factor hull: mix supports(1,0)
and(0,1) with respective amplitude vectors(1/2,0) and(0,1/2).
It also lies in EACH individual gate's ideal retained-input/phase hull.
For gate1 mix source corners(-1,-1),(1,1); for gate2 mix(1,-1),(-1,1).
Each produces x=t=0, sigma_i=1/2,e_i=1/4. Their source decompositions differ.

It is not in the ideal JOINT relation hull. If an original sigma_i=1 then
f_i<=0 requires t<=-1/2. Since every original support count is<=1, a mixture
whose average count equals1 consists entirely of count-one points. It must
have mean t<=-1/2, contradicting t=0. Explicitly the valid source-binding row

    t<=1-(3/2)*(sigma1+sigma2)

excludes the tuple by1/2. This is one row/three nnz, not a new abstract-domain
theorem or a free factor composition rule.

There is also an input/amplitude projected separator without the bits:

    e1+e2=max(0,|x|-t-3/2)<=max(0,-t-1/2)<=(1-t)/4.

The final chord bound holds for t in [-1,1]. The fake point has e1+e2=1/2
whereas the bound is1/4, so the issue is not merely an arbitrary phase tuple.

For a paid comparator, the exact single-gate hull for gate1 is

    e1>=0, e1>=-x-t-3/2, e1<=sigma1/2,
    e1<=-x+1-3*sigma1/2,
    e1<=-t+1-3*sigma1/2,
    e1<=-x-t+2-7*sigma1/2.

For gate2 replace -x by +x. These follow by disaggregating the two sign
branches over the source box and eliminating their two scaled coordinates;
this is an ideal-hull proof, not an allowed runtime split procedure. Per gate
the nonzero counts are1+3+2+3+3+4=16. The two ideal gates plus count therefore
have4 continuous coordinates(x,t,e1,e2),2 bits,13 rows/34 nnz, before8 scalar
source/phase bounds. Additionally materializing r1,r2 costs2 continuous
variables and2 equalities/8 nnz. Both ideal gates still admit the fake tuple.

## 4. Independent amplitudes are not generally ReLU-closed

On x,y in [0,1] consider the two mixed nonparallel rows

    f=(2x-y-1/2, -x+2y-1/2), r=ReLU(f).

Both gates are unstable. Fix ORIGINAL output bits beta=(1,1). The source
fiber vertices are (1/2,1/2),(3/4,1),(1,1),(1,3/4); its output vertices are

    (0,0), (0,3/4), (1/2,1/2), (3/4,0).

The sums of opposite vertices differ, so this quadrilateral is not centrally
symmetric. Once ALL bits are fixed, any finite affine image of independent
amplitude boxes (with only bit-count budgets) is centrally symmetric. Adding
more independent continuous amplitudes cannot represent this fiber exactly.

Retaining relational source predicates or new coupled guards CAN express
it, as ordinary HZ can. This result rules out only the uncoupled amplitude
primitive's general exact ReLU closure, not predicate-bearing HZ or the
restricted two-guard transform in the main note. It prevents an unjustified
transition from an ideal standalone factor to a general neural domain.

## 5. Disposition

All examples and costs above are algebraic controls, not executed tests.
The standalone factor and source-binding inequalities are useful supporting
mathematics, not established innovations. A scalar source-barrier/mixing-set
extension was briefly considered: conditional caps and counts could bind
the shared source to phases, but no full neural closure or new hull theorem
was derived. Existing mixing-set/cardinality work is an obvious comparator,
not a missing constraint family that may be claimed as new without research.

The positive result retained for the next definition step is the exact,
phase-preserving disjoint-sign transformer. Its biased extension and ordinary
source prevalence must be checked without turning this into a phase-split,
spline-enumeration or full old-graph duplication implementation. No runtime
candidate is enabled and no historic evidence is modified by these notes.
