# D013 — Budget-quotient generators: exact algebra, limited terminal benefit

2026-09-28; branch `redu-hz`; HEAD
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.
Paper mathematics and primary-source/independent review only. No numerical
execution, test run, solver, model decoding or real-network admission.
Baseline: 1870/2413 (1063 CERT +807 validated ADV), separate E0 CIFAR10025 +
TinyImageNet36 =61/400. No score/default change or historical-source edit.
[Goal authority](../../GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md).

## 1. Definition-level experiment and its disposition

D012 derives phase budgets but remains a resource component over the original
HZ graph. Here we ask a stronger question: can certified budgets change the
algebra of the generators, permitting exact compositional elimination of
cross-layer phase interactions while retaining every original bit and guard?

The answer is conditionally yes. Sections2–3 give an exact quotient and neural
operators, and section5 gives a mixed-sign residual control with a strict LP
tuple separation even against ordinary native HZ plus the same count row.
But the quotient algebra itself is already an explicit published construction;
same-layer budgets do not simplify any ordinary forward path monomial; and
general terminal lowering can cost more than the original graph. Thus this
is a checked research candidate, NOT an established innovative Neural-HZ
domain, production implementation or formal verification gain.

## 2. Domain objects, original-HZ embedding and concrete semantics

Let xi in [-1,1]^p be the original continuous factors, and zeta be ALL original
free binary factors, recoded to {0,1} without changing their identities. Each
original neural gate has its own original bit beta_i. Choose a fixed structural
reference tau_i in {0,1}, and let delta_i=beta_i xor tau_i be an alias, not a
new variable. One frame must have one consistent reference per gate. A
candidate implementation must preregister the reference rule, not select it
from instance identity, final margins or observed success.

The candidate changes the constant-generator HZ language into source-affine
forms whose coefficients belong to a phase-budget quotient:

    v = sum_{S in Delta} delta_S * (c_S+Gc_S*xi+Gb_S*zeta),
    delta_S=product_{i in S} delta_i.

Here Delta is defined by a finite collection of certified UPPER budgets

    sum_{i in B_j} delta_i <= U_j,
    Delta={S subset [q]: |S intersect B_j|<=U_j for all j},
    0<=U_j<=|B_j|, U_j integer.

Delta is downward closed. This definition uses only upper budgets in one
fixed orientation; arbitrary lower budgets or independently reoriented groups
do not automatically give the same downward-closed algebra.

An element retains the shared factor/frame registry, all original EQ/LE
predicates and neural sign guards, value forms, budget certificates and the
original-input/hidden-value reconstruction map. Define the strong relation

    Gamma(N)={(original_input, ALL original bits, declared_outputs):
               original factor bounds/predicates/guards hold,
               delta=beta xor tau and the certified budgets hold,
               outputs equal the value forms}.

On an admitted element the budgets are already proved consequences of the
original strong relation, not unsupported extra restrictions on that relation.

The ordinary concrete output set is its projection, and semantic order is
inclusion of the strong relation after aligning original identities. No
computable best abstraction, join, complete lattice or complete semantic
normalizer is established. Original HZ embeds with empty neural budget data
and the affine map c+Gc*xi+Gb*(2*zeta-1), retaining its original predicates.
Subsequent exact neural transformers introduce their original gate bits.

The representation remains genuinely nonconvex: with no additional phase
budget, an element can represent the retained-input graph of ReLU, excluding the midpoint of
(-1,0) and (1,1). Budget normalization never relaxes bits to intervals or
pivots/deletes them. This is not a CZ/Zonotope replacement. For fixed bits
the fibers remain polyhedral, so no claim of greater set expressiveness than
general HZ or constrained polynomial domains is made.

## 3. Exact normalization and neural operators

Use the ideal

    I_Delta=<delta_i^2-delta_i, delta_T for T not in Delta>.

The allowed squarefree monomials form a unique basis of R[delta]/I_Delta.
Spanning follows by squarefree reduction and deletion of nonfaces. To prove
independence, evaluate them on indicator vectors 1_T for T in Delta: the
matrix has entry1[S subset T], is triangular in an inclusion-compatible order
and has diagonal1. Hence a zero function has all zero coefficients. This
is exactly the vanishing ideal of the budget-allowed Boolean supports, NOT
generally the complete vanishing ideal of the source/guard feasible set.

Multiplication is the simple exact operation

    delta_S*delta_T = delta_(S union T) if S union T in Delta, else0.

Membership checks the stored budgets; it need not enumerate phase assignments
or compute a Groebner basis. Expanding all surviving supports can nonetheless
be exponential, so this is an algebraic rule, not a free efficient oracle.
Uniqueness is a formal coefficient normal form; source constraints can make
distinct remaining forms agree on the actual feasible set.

Affine/Conv: multiply and add source-affine coefficient vectors with the
constant operator, preserving factor identities and predicates. ReLU gate i:

    r_i=NF((tau_i+(1-2*tau_i)*delta_i)*f_i),
    retain the ORIGINAL guard (2*beta_i-1)*f_i>=0.

Each product inserts one support index; the quotient equals the original
expression on every budget-allowed assignment, proving exact transfer.
Affine/Conv and this rule never multiply two source-dependent forms, so source
degree stays affine. General products, smooth activations and attention do
not inherit this proof.

Add/Concat align the SAME input, original bits and reference registry, conjoin
the retained branch predicates and combine/stack forms. Independently cloned
branches would be wrong. Budget certificates remain valid under this
strengthening. Do not multiply branch value forms merely because they merge.

Admission order matters. A budget must first be proved on the original strong
integer relation, including all zero-value bit choices. One can build a block
exactly, prove its budget from already valid relations, then normalize it and
later operators. One cannot assume a future budget to delete terms and use
the deleted graph to prove that budget. Missing evidence leaves the unchanged
representation, never an approximate equality. Witnesses retain the original
input and bits and must independently reconstruct and validate the concrete
network; a candidate symbolic point is not an ADV.

**Direct prior art:** the algebra and basis in this section are already given
in Gouveia--Parrilo--Thomas, section3.1 [P1]. We independently recovered them,
but they are not a new definition/theorem to claim as our contribution.
The remaining research question is a useful NN-specific certificate/transfer
calculus and complete-cost advantage, not the existence of this quotient.

## 4. Path-support theorem and the single-layer obstruction

For a DAG containing only constant affine/Conv maps, Add/Concat and ReLU,
every phase support appearing in a fully expanded forward VALUE form lies
within a single causal path (possibly as a proper subset). Induction: an
affine combination only unions term
lists, never multiplies terms from distinct predecessors; ReLU inserts its
own descendant gate into each incoming path term. Reference substitution
beta=tau+(1-2*tau)*delta preserves this property.

Consequently a budget confined to one ordinary feedforward layer, with U>=1,
deletes NO such monomial: a path meets at most one gate in that layer. This
does not deny D012's useful count inference or its support bounds, and does
not cover later products created by RLT, predicate multiplication or unrelated
algebra. But it disproves the hoped-for direct bridge from every same-bank
budget to generator compression.

A global budget sum delta<=U gives at most sum_{j=0}^U binomial(q,j) basis
terms, polynomial in q for fixed U. Generic overlapping local budgets give
no such guarantee. The following ordinary bounded residual construction
shows an exponential surviving expansion even with a realizable reference.

Let u0 in [0,1] and shared v in [-1,1]. For each layer use

    f_plus=3*u+2*v-4, f_minus=-3*u+2*v-1,
    u_next=ReLU(f_plus)+ReLU(f_minus).

The rows (3,2) and (-3,2) are mixed, nonparallel and not opposite. Since
f_plus+f_minus=4*v-5<=-1, all original bits obey beta_plus+beta_minus<=1,
including zeros. Both positive outputs are <=1 and cannot coexist, so
u_next remains in [0,1]. At the single nominal input (u0,v)=(1/2,0), every
layer has both bits0; thus tau=0 is a feasible reference, not an artificial
unreachable phase pattern.

On the v=1 slice the map is max(3u-2,0)+max(1-3u,0). Its two outer open
branches each map onto (0,1), so all 2^L strict branch sequences of length L
are attainable. For any fixed finite sequence an interior input with strict
margins extends continuously to v<1 nearby; this is not merely a boundary
or zero-phase artifact. The coefficient of u0 in the L-layer phase expansion
is

    3^L * product_l (delta_(l,plus)-delta_(l,minus)).

It contains 2^L distinct nonzero allowed monomials, each picking only one bit
per layer. This is a lower bound for the explicit monomial representation,
not for all possible HZ/extended/circuit representations. The original graph
still has only 2L gates. A factored circuit avoids that expansion but may
simply be the original graph, which is not the requested domain innovation.

## 5. Positive mixed-sign residual control with all original phases

For (x,t) in [-1,1]^2 let

    f1=x+t+3/2, f2=-x+t+3/2,
    r1=ReLU(f1), r2=ReLU(f2),
    h=r1-r2/2-9/4, r3=ReLU(h),
    Y=r3+r1-2*r2.

Use tau=(1,1,0), which is the actual phase at the box center (h=-3/2).
The two first-layer rows are distinct mixed rows; the second row (1,-1/2)
is genuinely mixed-sign; Y retains a shared residual consumer. Every gate
has strict active and inactive regions. This is a paper control, not a
claimed frequent motif or real-network witness.

Let delta=(1-beta1,1-beta2,beta3). The complete original relation satisfies
sum delta<=1, including zero aliases:

- f1+f2=2t+3>=1 forbids simultaneous first-layer mismatches.
- delta1=1 implies r1=0 and h<=-9/4, so delta3=0.
- delta2=1 implies f2<=0 and r2=0. Since f1=f2+2x<=2, r1<=2 and
  h<=-1/4, so delta3=0.

Define a3=f1-f2/2-9/4=(3/2)x+(1/2)t-3/2. Before normalization,

    r3=delta3*(a3-delta1*f1+(delta2*f2)/2).

Both cross-layer coefficients are nonzero. The certified budget deletes
delta1*delta3 and delta2*delta3, giving r3=beta3*a3. Similarly

    Y=f1-2*f2-delta1*f1+2*delta2*f2+delta3*a3.

For this identity, the two pairwise implications beta3<=beta1,beta3<=beta2
already suffice; the stronger three-literal count is not indispensable.
It is a clique/cardinality constraint on incompatible phase disagreements,
not an additional geometric discovery beyond the cited Boolean theory.

Full phase correspondence, rather than output equality alone, can also be
proved. If f1<=0 then both h and a3 are strictly negative. If f2<=0 then
h<=-1/4 and a3=f2/2+2x-9/4<=-1/4. Otherwise h=a3. Thus h and a3 have
identical guarded-ReLU relations, including their zero sets and both original
beta3 choices. This is related to D005's contextual quotient, now with a
mixed-sign jointly shielded edge; it is not a proof of a wholly new domain.

### 5.1 Stronger than native gates plus the count on a retained tuple

The first-layer tight bounds are [-1/2,7/2]. Tight h bounds are [-13/4,1/2]:
|r1-r2|<=|f1-f2|<=2 yields -1<=r1-r2/2<=11/4; endpoints occur at
(-1,-1/2) and (1,1). Also a3 has tight box bounds [-7/2,1/2].

The old standard four-row formulation for each gate, PLUS sum delta<=1,
admits the fractional tuple

    (x,t)=(1/2,0), beta=(3/4,1,3/4),
    (r1,r2,r3)=(17/8,1,3/16).

Indeed f=(2,1), h=-5/8. The first gate saturates
r1<=f1+(1/2)(1-beta1)=17/8; the third saturates
r3<=h+(13/4)(1-beta3)=3/16; its other upper bound is3/8.
All lower rows hold, the second gate is exact, and sum delta=1.

The normalized graph has the valid row

    r3<=a3+(7/2)(1-beta3).

Here a3=-3/4, making the RHS1/8 and excluding r3=3/16. This is a strict
LP retained-tuple separation even after giving the old native formulation
the same count row. It is NOT yet a separation in output-only projection,
not a new CERT, and not an advantage over ordinary HZ supplied the same new
valid row or stronger generic lifts.

The normalized lower row r3>=a3 is also independently nonredundant. The
tuple (x,t)=(1,1/2), beta=(1,3/4,1/2), r=(3,9/8,3/16) satisfies all old
rows, the budget and the new upper row; h=3/16 but a3=1/4. Thus it violates
the new lower row by1/16. The preceding tuple violates only the new upper
row, also by1/16. Neither added row is being counted as a useful duplicate.

### 5.2 Full terminal bill and preservation options

No hidden-node value may silently change meaning. In particular the ORIGINAL
h must still decode as r1-r2/2-9/4; it is not equal to a3 everywhere. Any
old EQ/LE or raw consumer of h remains on the original expression.

For this control, the native graph has 5 continuous variables,3 original
bits,12 gate rows/30 nnz. Adding the count gives13 rows/33 nnz. Retaining
ALL those rows and adding the two nonduplicate normalized rows

    r3>=a3,
    r3<=a3+(7/2)(1-beta3)

gives15 rows/40 nnz, with NO additional continuous or binary variable. The
other normalized rows r3>=0,r3<=beta3/2 are already present. Hence this
specific additive lowering preserves the entire old LP by construction and
strictly strengthens it on the displayed tuple, with the same integer relation.
It is a paid derived-row strengthening, not dimension reduction. Source,
bounds, row RHS, evidence, old predicates, reconstruction and equality-only
slacks are additional costs, not excluded from qualification.
Y as a readout additionally stores three coefficients; materializing it as
a native variable adds one continuous variable and one equality with four
nnz. The displayed counts do not include that optional materialization.

Replacing the old third gate by the flat a3 gate has the same gate-row count
but produces an INCOMPARABLE retained-tuple LP relaxation, even with the
budget. The old-not-flat tuple is given above. In the reverse direction,
(x,t)=(1/2,0), beta=(1,1,3/4), r=(2,1,1/8) passes the flat rows and budget,
but the old parent upper requires r3<=1/16. Retaining the old lift is
essential to the stated LP-subset guarantee. No replacement is selected or
enabled by this note, and raw original consumers cannot be silently changed.

For a GENERAL q-gate block with a global at-most-one mismatch budget, one
may express every actual preactivation/output as affine in the source and
z_i=delta_i*f_i^0, where f_i^0 is its fixed-reference preactivation. All old
guards are on the ACTUAL preactivation, not automatically f_i^0. A generic
lowering uses q product variables,4q product rows,2q sign rows and the
budget:6q+1 rows, with source/ancestor fill up to quadratic order. This is
not a gain over the ordinary p+q continuous-variable,4q-gate-row graph.
The product rows require certified bounds on f_i^0, while the original
guard rows require separately certified bounds on the ACTUAL preactivation.
The former bounds cannot silently replace the latter's provenance.
Using all delta_i*source_j products can be larger still. The small control's
two-row strengthening must not be generalized into a free terminal theorem.

## 6. Prior-art and capability boundary

- [P1: Gouveia--Parrilo--Thomas, Theta bodies for polynomial ideals, section3.1](https://optimization-online.org/wp-content/uploads/2009/01/2196.pdf)
  explicitly defines the same Boolean/nonface ideal and face-monomial basis.
  No theta-body or SDP computation is proposed here.
- [P2: Sharp-HZ, sectionIV/equation18](https://arxiv.org/html/2503.17483v2)
  already lifts Boolean monomials and continuous-source times Boolean
  monomials. Our source-affine phase generator family is not new on that basis.
- [P3: abs-normal piecewise-linear representation, equation4](https://arxiv.org/pdf/1701.00753)
  gives the finite triangular Neumann expansion underlying phase-path products.
  Its numerical solution methods are not imported into the verification path.
- [P4: mixed polynotopes](https://arxiv.org/html/2009.07387v2) already provide
  shared typed symbols and polynomial rewriting. [CPZ](https://arxiv.org/abs/2005.08849)
  can encode polynomial values and predicates; no greater set expressiveness
  than these nonconvex families has been proved. Convex CZ is not the comparator
  being adopted as a replacement.

The exact neural use of certified overlapping budgets could still be useful,
but neither a literature gap nor actual common-CNN prevalence/cost has been
established. A rediscovered algebra plus an engineered control is insufficient
for the requested PLDI-quality Neural-HZ contribution.

## 7. Result and next mathematical question

This checkpoint rejects two unsupported premises: that same-layer phase
budgets automatically reduce phase-generator degree, and that exact quotient
normalization automatically reduces the complete native query. It preserves
a mixed-sign cross-layer positive control and a rigorous additive LP gain as
supporting evidence. It does not launch a monomial-expansion implementation,
resume D010, or change the full2413/400 replay and resource gates.

Next investigate a query-relevant nonconvex invariant which uses joint phase
budgets WITHOUT expanding path monomials or merely retaining an extra copy
of every old gate. A candidate must specify a compositional object and its
finite inference/terminal cost, and beat the same-row/shared-graph comparator
in a demonstrated complete-cost/precision tradeoff on ordinary mixed blocks.
Equal expressive power or equal precision after giving HZ exactly the same
rows is EXPECTED and is not, by itself, a disqualification: general HZ can
represent bounded finite unions of polytopes. The missing contribution is a
substantive NN-oriented definition/transfer theorem with useful complete cost,
not an impossible demand for greater set expressiveness. Novelty research,
mathematical qualification and eventual real-network evaluation remain separate
obligations under the original goal; this note adds no new admission gate.
Do not call the displayed derived rows themselves the completed new domain.

All findings here are paper-level; latest numerical reference remains D003.
The full Goal stays ACTIVE and is far from achieved.
