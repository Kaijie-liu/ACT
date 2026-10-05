# D014 extension — Partial sign shielding with a retained varying baseline

2026-09-28; `redu-hz`; HEAD `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.
Paper derivation and independent algebraic review only. No implementation,
numeric evaluation, test, model decode, solver or new benchmark verdict.
Formal1870/2413 and separateE061/400 unchanged. All original goal gates apply.

## 1. Result: full mutual exclusion and zero bias are unnecessary here

The main D014 theorem eliminates the output variable but needs PN=0, a strong
condition. The following weaker rule retains that variable and the four gate
rows, while reducing one predicate's dependence and strengthening the LP.
The remaining positive and negative core may overlap arbitrarily. Its baseline
may be a varying affine function of the original source, not just zero bias.

Let f=R-n with n>=0 and assume n*ReLU(R)=0 on the full original integer
source relation. Then

    ReLU(f)=ReLU(R).

If n>0, R<=0 and both sides vanish; if n=0 the arguments agree. Preserve the
ORIGINAL gate bit beta and original valid l<0<u, using

    r>=0, r>=R, r<=u*beta, r<=f-l*(1-beta).                 (1)

Compared with the original four rows, ONLY r>=f is replaced by r>=R. The LP
claim additionally requires n>=0 in the RETAINED LP relaxation, not merely
on integer source points. Section2 supplies this using nonnegative amplitude
columns and positive coefficients. Every new LP tuple then satisfies the old
rows since R-f=n>=0. All other source constraints, bounds, bits and raw f
consumers remain.

For integer beta=0, (1) forces r=0 and R<=0, hence f<=0. For beta=1 it forces
R<=r<=f=R-n, so n=0 and r=R=f>=0. Conversely, any original integer gate tuple
satisfies (1) by the value identity. In particular, R=0,n>0 has original f<0
and only beta=0; the retained old upper row forbids beta=1. A flat ReLU(R)
replacement would lose that distinction. At original f=0, the premise forces
n=R=0, so BOTH original beta choices remain legal.

No variable, bit, output generator or row is added. With canonical DISTINCT
amplitude columns, and baseline columns disjoint from those columns, removing
s_n nonzero negative terms saves s_n nnz in the lower row. If aliases/baseline
share columns, coefficients must be fully merged and actual row counts checked:
the exactness/LP theorem survives, but this nominal nnz saving need not.
This is a paid structural predicate normalization, NOT a new domain by itself.

## 2. One simultaneous coefficient/bound rule proves the premise

Write the original preactivation as

    f=s(xi)+sum_{i in P} a_i*e_i-sum_{j in N} d_j*e_j,
    a_i,d_j>0, 0<=e_i<=u_i, s(xi)<=Sbar.

Retain s(xi), its shared original source coordinates and every defining
relation. Let E be a graph of already CERTIFIED amplitude conflicts
e_i*e_j=0. Unproved edges are absent, i.e. treated as potentially co-occurring.
For each negative contributor j compute

    C_j=sum_{i in P, {i,j} not in E} a_i*u_i,
    J={j in N: Sbar+C_j<=0},
    n=sum_{j in J} d_j*e_j, R=f+n.

Extract ALL j in J in one pass. If n>0, some j in J has e_j>0. Its conflicting
positive amplitudes vanish, so s+sum positive<=Sbar+C_j<=0. R subtracts only
the remaining nonnegative negative mass and is therefore<=0. This proves
the simultaneous premise, without assuming conflicts among negative terms,
among positive terms, or within the residual core.

The rule depends only on coefficients, certified caps and support relations;
no instance identity, label, margin, solver state, phase search or LP query.
Using an unconditional Sbar may be conservative, but never authorizes freezing
s(xi) to a nominal value. Each bound must be exact or independently outward
certified; ordinary floating-point interval hashes alone are not certificates.

This is only maximal relative to this fixed sufficient test, not globally
maximal elimination. A failed inequality says the test did not certify j;
it does not prove j essential in the exact neural relation.
Since C_j>=0, this particular sufficient test requires Sbar<=0. Retaining
a varying baseline is NOT coverage of every baseline sign. A symmetric
positive-extraction rule changes output readouts and has a different cost;
it is not silently appended as a second implemented route here.

## 3. Ordinary biased mixed control and full source bill

Keep four ORIGINAL t_i in [0,1], four original bits sigma_i, exact products
e_i=sigma_i*t_i, and sigma1+sigma3<=1. Hence e1*e3=0, but e2 may coexist with
e3 and the remaining core's positive/negative terms can also coexist. Take

    f=-1/2+e1+(1/4)*e2-e3-e4.

Caps are all1. C3=1/4<=1/2, whereas C4=5/4>1/2. The uniform rule extracts
exactly n=e3 and leaves R=-1/2+e1+(1/4)*e2-e4. Tight old bounds are
l=-5/2 and u=3/4. These are attained with respectively e3=e4=1 and e1=e2=1.

Old four-row gate nnz: 1+5+2+6=14; new: 1+4+2+6=13. Each source product's
four rows has8 nnz, and the shared count adds one row/two nnz. Therefore,
with explicit product nonnegativity and before scalar t/bit bounds and RHS:

| Formulation | Continuous | Original bits | Rows | Row nnz |
| --- | ---: | ---: | ---: | ---: |
| Original four products + count + original gate | 9 | 5 | 21 | 48 |
| Same source + shielded lower row | 9 | 5 | 21 | 47 |

No hidden output readout or extra variable is used. Conflict discovery,
proof metadata, caps and comparisons are additional costs; no net speed or
byte saving follows from a single nonzero. The important algebraic difference
is a STRICT LP strengthening with no new native row.

For example, the old LP admits

    t=(1,1,1,0), sigma=e=(1/2,1,1/2,0), beta=0, r=0.

Every product and count row holds; f=-1/4 but R=1/4, so the new lower row
rejects it. A second old point with t unchanged, sigma=e=(3/4,1,1/4,0),
beta=1,r=1/4 has f=1/4,R=1/2 and is likewise excluded. Both have an integral
CURRENT gate bit; predecessor bits remain fractional. These are paper LP
controls, not concrete adversarial inputs or executed property proofs.

## 4. A compositional, zero-safe amplitude-support invariant

The preceding rule needs certified conflicts; they are not a free oracle.
A possible FORWARD inference component over the exact guarded-amplitude
element is the following. For f=s+sum_i w_i e_i, known Slo<=s<=Sbar, and an
existing query amplitude e_j, define

    U_j=Sbar+sum_{w_i>0, {i,j} not in E} w_i*u_i,
    L_j=Slo-sum_{w_i<0, {i,j} not in E} |w_i|*u_i.

On e_j>0 all conflicting terms vanish; dropping remaining negative/positive
mass proves respectively f<=U_j and f>=L_j. For r=ReLU(f) and the exact
complementary amplitude q=r-f=ReLU(-f), infer

    U_j<=0 => r*e_j=0,
    L_j>=0 => q*e_j=0,
    always r*q=0.

q can be an affine readout of the existing r and source, not a fresh native
variable. Its readout density and later coefficient fill must still be paid.
The inference stores consequences over one unchanged source relation; it
does not branch/restrict that relation or enumerate phase assignments. The
native terminal retains original integer bits, EQ/LE and the exact gate
formulation. Affine/Conv and shared Add/Concat propagate source/readout forms
and retain the same amplitude identities; ReLU introduces its original bit
and output, then soundly extends this support invariant. No general compact
exact closure or independent-domain novelty follows from these operations.

There is an important zero distinction from original-BIT dependencies.
For e1=ReLU(x), e2=ReLU(-x-1), x in [-2,1], f=e1-e2, the graph has e1*e2=0
and U_2=0 proves ReLU(f)*e2=0. At x=-1, e2=f=ReLU(f)=0, and both corresponding
original active bits may equal1. Thus a support edge does NOT authorize a
bit-count inequality or phase implication. No D012 budget is inferred from
such an edge without a separate proof including every zero choice.
Nor are edges transitively closed: e1=e3=1,e2=0 satisfies edges12 and23, not13.
For a nonnegative sum of amplitudes, certified conflict neighbors propagate
by INTERSECTION of the summands' neighbor sets, not their union. The signed
readout q=r-f must use its ReLU proof above, not that nonnegative-sum shortcut.

Given positive caps, a support edge implies r/U_r+e_j/u_j<=1, and an amplitude
clique implies sum e_i/u_i<=1. These are standard complementarity hull rows,
not new innovations, and are optional paid consequences, not selected here.

First-bank seed certificates may use a faithful shared-source upper bound
on g_i+g_j<=0 for amplitudes e_i=ReLU(g_i), e_j=ReLU(g_j). Both cannot be
strictly positive. A non-strict bound suffices for amplitudes, unlike some
original-bit budgets. Source sharing, convolution padding and BN operations
must be authenticated. Correctly independent outer boxes can weaken a
certificate without making it false; incorrectly identifying patch coordinates
or claiming shared-source cancellation across different sources is unsound.

Cost is not merely the number of pairs. For m output rows, k incoming
amplitudes and q query amplitudes, naive weighted-neighborhood work is
O(m*k*q), plus bounds/source work. With adjacency lists a paid bound is

    O(nnz(W)+sum_i nnz(W[:,i])*degree_E(i)+m*q),

plus all output edges, evidence and readout storage. Dense support graphs
can make it cubic when m=k=q. A fixed structural frontier may bound work,
but none has yet been implemented or qualified; dropping optional graph
facts must never drop original exact predicates or invent absence of overlap.

## 5. Biased arm splines: exact identity, rejected general escape route

For an at-most-one-nonzero amplitude vector, every function F obeys

    F(e)=F(0)+sum_i (F(e_i*unit_i)-F(0)).

Thus a normalized arm expression c+sum_i g_i(e_i), g_i(0)=0, has exact
biased affine and ReLU propagation. For ReLU use new center ReLU(c) and arms
ReLU(c+g_i)-ReLU(c), componentwise. Shared residuals combine the SAME arms.
Every original gate's preactivation/phase guard, not only final values, must
still be retained. With a varying source baseline F(xi,e), the arms depend
on (xi,e_i), not a single variable: freezing xi would be unsound.

Even for a fixed center, explicit arm splines are not a compact general
replacement. Let e1,e2 in [0,1], support(e)<=1, u0=e1, and repeat

    u_next=1-ReLU(2u-e2-1)-ReLU(-2u-2e2+1).

The rows (2,-1),(-2,-2) are nonparallel/nonopposite with ordinary biases.
Their preactivations sum to -3e2<=0 and are each<=1 when u,e2 in [0,1], so
u_next stays in [0,1]. On the e2=0 arm this is the tent map 1-|2u-1|.
After L blocks it has 2^L affine intervals and 2^L-1 essential interior knots,
while the original graph has2L gates. A flat spline/hinge list is exponential;
a factored expression avoids that expansion but can just recreate/copy the
old circuit. This is NOT a lower bound for every neural abstract domain.
Runtime root/subinterval enumeration is not proposed under the no-split rule.

These are known univariate PWL phenomena, not a novel definition or lower
bound claim: [Plonka--Riebe--Kolomoitsev, section4](https://arxiv.org/pdf/2207.14609)
gives recursive spline conversion and its product knot bound;
[Telgarsky, section3.3](https://proceedings.mlr.press/v49/telgarsky16.pdf)
analyzes iterated triangle-wave oscillations. Both primary texts were checked.

## 6. Comparator and decision

Partial extraction follows the classical identity
(R-n)^+=R^+-min(n,R^+) for n>=0. Keeping the original upper row remedies
the zero-phase and LP-information loss of an indiscriminate flat rewrite.
It does not establish novelty simply because it is phrased as a Neural-HZ rule.

Existing [Venus dependency analysis](https://www.doc.ic.ac.uk/~alessio/papers/20/aaai20-BKKLM.pdf)
already computes intra/inter-layer ReLU dependencies from bounds. Its search,
split and solver-callback machinery is NOT imported here. The forward support
invariant is closely related to this prior work; distinguishing amplitude
zeros and replacing a native row is not sufficient evidence of a new domain.
Ordinary HZ supplied the same valid row has the same integer/LP semantics.

Positive research result: biased, source-varying, overlapping-core rows have
an exact, phase-preserving strengthening with no new native rows or variables.
Negative result: arbitrary biased arm closure does not provide cheap general
composition. Next useful evidence is an honestly bounded, faithful first-bank
coefficient/certificate census and complete-cost comparison, NOT further
arm-spline or test-framework expansion. No census is executed or pre-certified
by this paper; source-scope notes record what has and has not been observed.
This remains supporting mathematics within the full definition-first goal,
not the completed requested Neural-HZ innovation or any formal solved gain.
