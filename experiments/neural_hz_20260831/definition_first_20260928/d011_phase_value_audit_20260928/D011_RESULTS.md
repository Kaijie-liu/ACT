# D011: phase/value coupling audit of three proposed Neural-HZ definitions

2026-09-28; branch `redu-hz`; commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.

Status: mathematical investigation, primary-source comparisons, and independent
paper review. No candidate import, AST parse, test, model execution or solver
call. None of these three proposals establishes the requested new domain.
Formal baseline remains 1870/2413 = 1063 CERT + 807 validated ADV; separate E0
remains CIFAR100 25 / TinyImageNet 36. New formal/benchmark gain: zero.

Previous user-facing goal turn: NO PROGRESS under the goal criterion; it
reconfirmed alignment. This investigation produces counterexamples and scoped
theorems that change the next research action. The full definition-first Goal
remains ACTIVE. The negative results below are not impossibility results for
Neural-HZ innovation, and do not justify weakening any gate.

## 1. Scope and question

D009's resource calculus is a candidate reduced product, not an established
domain innovation. D010's certificate kernel and tests remain unexecuted drafts;
their owner checks and resource accounting are supporting engineering, not a
substitute for the missing mathematical contribution. See D010_DRAFT_STATUS.md.

This checkpoint asks whether three other exact semantic organizations provide
a useful nonconvex simplification rather than a renamed circuit:

1. a shared affine bank's realizable phase/zero strata;
2. differences of convex support generators, modulo common factors;
3. activation-difference generators with shared cycle consistency.

Every candidate is considered with an owned original HZ frame, continuous
factors, every original/free/ReLU binary identity, old EQ/LE predicates, and
input/output decoders. Strong concretization retains input, all these bits,
and output. Equal output functions alone do not authorize dropping a guard.
All transformations must preserve shared assignments across residual branches.

The question is not whether ordinary HZ can express the same finite PWL set:
that would be an inappropriate novelty requirement. The question is whether
the proposed organization yields a new, sound, compositional and useful
inference or simplification theorem with its entire cost paid.

## 2. Candidate A: sign-realizable affine bundles

### 2.1 Exact sign information is three-valued

For f=Ax+b, form H with rows h_0=(0,...,0,1), h_i=(A_i,b_i).
Then (1,f)=H(x,1). A circuit is a support-minimal nonzero vector c with
H^T c=0. For retained magnitudes m_i>=0 and original bits beta_i, use

    tau_0=+,
    tau_i=0 when m_i=0, otherwise tau_i=2 beta_i-1.

The circuit condition is that the nonzero products sign(c_i)*tau_i either
are absent or include BOTH signs. All such conditions characterize realizable
three-valued signs of vectors in image(H). The positive anchor coordinate
can be scaled to one. This is the standard circuit/covector orthogonality
characterization, not a new neural theorem; see Richter-Gebert and Ziegler,
section 7.2.5 of [Oriented Matroids](https://science-to-touch.com/Articles/jrg/21_OrientedMatroids.pdf).

Necessity follows immediately from c^T H z=0. For sufficiency, the standard
linear-alternative argument for the proposed strict/non-strict sign system
provides a separating vector in ker(H^T) if no realizing z exists; decomposing
it into conformal circuits gives a violating circuit. This is a paper proof,
not an implemented LP-dual or infeasibility-certificate algorithm.

For example, f_1+f_2-f_3=0 with all f_i=0 admits every original bit triple.
A strict-tope rule forbidding beta=(1,1,0) would incorrectly remove legal
zero-phase witnesses. Conversely, merely requiring opposite bit literals
does not ensure an opposite NONZERO magnitude exists.

Writing s_i=2 beta_i-1, define

    P_c=max(c_0,0) + sum_{c_i s_i>0} |c_i| m_i,
    N_c=max(-c_0,0) + sum_{c_i s_i<0} |c_i| m_i.

The sign condition is P_c=0 iff N_c=0. Actual numerical affine dependence
requires the stronger equality P_c=N_c. These are not interchangeable.
This elementary boundary distinction is a semantic check, not a proposal to
devote the main experiment budget to special zero cases.

### 2.2 Existential sign realization loses numerical coupling

The complete sign characterization says that SOME input realizes the signs.
It says nothing about whether the CURRENT magnitudes or shared input do so.
For f_1+f_2=f_3, beta=(1,1,1), magnitudes (1,1/4,1/4) pass the realizable
strict-positive sign test, but violate the numerical relation. This already
fails away from every zero boundary.

A sharper representability consequence concerns the candidate set

    S={ (m,beta): 0<=m<=1,
                   tau(m,beta) is realizable on f_1+f_2=f_3 }.

For every 0<epsilon<=1, (m,beta)=((1,epsilon,epsilon),(1,1,1)) is in S:
the sign (+,+,+) is realized, for example, by (1/4,1/4,1/2).
Its limit ((1,0,0),(1,1,1)) is not in S, since (+,0,0) is impossible.
Thus S is not closed, even with bounded m.

A finite HZ with bounded continuous variables and non-strict linear
predicates is a finite union of compact polytopes; its continuous projections
are compact and therefore closed. Such an HZ cannot represent S exactly.
Taking the closure could define a sound overapproximation, but loses the exact
stratum semantics claimed by this candidate. Restoring P_c=N_c repairs the
issue by restoring the original numerical affine relation, whose cost remains.
The exact original ReLU graph itself is closed; this obstruction concerns
the proposed sign-only magnitude lift, not the graph.

Box membership also cannot be inferred from the all-space arrangement.
Additional forms x_j-L_j and U_j-x_j can express its nonnegative sign
requirements, but enlarge the arrangement. A current shared x still requires
its numerical source equations. General HZ predicates likewise remain.

### 2.3 A relation basis is not a complete sign oracle

Take f_3=f_1+f_2 and f_4=f_1+2f_2. Their two fundamental circuits each
accept tau=(+,-,-,+): each signed relation has both positive and negative
terms. Nevertheless f_4=f_3+f_2<0 contradicts tau_4=+.
The additional circuit f_4-f_3-f_2=0 exposes the contradiction.

A generic rank-r configuration with n rows has a circuit on every r+1
subset, hence binomial(n,r+1) supports. A 64-by-27 bank with affine anchor
can have n=65,r=28, and binomial(65,29) circuits. These are algebraic class
counts, NOT a measured rank or circuit count for either benchmark network.
A fixed finite subset remains sound but incomplete.

Checking a supplied k-row circuit over d ports costs O(kd) rational arithmetic
plus coefficient-bit and source checks. Checking a supplied sign assignment
against a finite ledger costs O(sum support sizes); discovering and storing
the ledger is additional. No complete circuit enumeration is proposed.
When output rows are independent, output-only circuits are absent; adding
source coordinates merely recovers source equations. After ReLU, later banks
are not globally affine in the original inputs. Retaining shared prefix
values/guards preserves this information; explicitly constructing their
phase-conditional arrangements would introduce forbidden phase enumeration.
No alternative compact exact forward closure follows from local sign data
alone in this candidate.

Three-valued neural sign complexes also have direct prior art: Masden's
[2022 paper](https://arxiv.org/pdf/2207.07696), section 3.2, includes zero
coordinates and faces, while section 4.3/discussion identifies exponential
input-dimensional cost. Its additional genericity hypotheses are not assumed
for our verifier. No topology/enumeration algorithm from that work is adopted.

Decision: retain finite checked circuits as a possible supporting inference;
reject the complete sign-only factor as a cheap exact Neural-HZ definition.

## 3. Candidate B: support-function differences and a forward quotient

For a finite polytope P, let h_P(z)=max_{p in P} p^T z. Represent a value as
v=h_P-h_Q on the SAME latent direction z. Bias uses an appended constant one.
All original HZ predicates and original phase guards would still be retained.

Nonnegative scaling/sum of support functions corresponds to scaling/Minkowski
sum of polytopes. Negative weights exchange the P and Q roles. ReLU obeys

    ReLU(h_P-h_Q)=h_conv(P union Q)-h_Q.

This is directly the known tropical rational/signomial neural calculus, not
a new grammar: Zhang, Naitzat and Lim's [2018 paper](https://proceedings.mlr.press/v80/zhang18i/zhang18i.pdf),
Propositions 3.2, 5.1 and 5.6, includes the signed-affine recurrence and real
weights. Existing [tropical-polyhedral abstract interpretation](https://arxiv.org/html/2108.00893v2)
is also an obligatory comparator, although retaining HZ bits is not literally
the same domain.

### 3.1 A sound compositional quotient, but not new algebra

Suppose P=P_0+R and Q=Q_0+R with a supplied exact factorization. Then
h_P-h_Q=h_{P_0}-h_{Q_0}. Moreover,

    conv((P_0+R) union (Q_0+R))=conv(P_0 union Q_0)+R.

Proof: a convex combination of the two left summands combines two points of
R into a point of R; conversely use the SAME point of R in both summands.
Thus common-factor cancellation commutes with ReLU.
For signed affine mixing of values with common factors R_i, the two output
support polytopes have the same factor sum_i |w_i| R_i; cancellation commutes
there too. Shared Add/Concat follows from the same-frame operations.

This is a genuine non-enumerating forward identity, but follows from ordinary
support-function/Minkowski algebra. Discovery of a factorization is not free.
Keeping an already supplied factor certificate costs its complete DAG and
references, not zero. The canceled factor is absent from visible values, but
original guards or other consumers may still need it.

For ALL homogeneous directions, equality of differences is equivalent to
P+Q'=P'+Q, because compact convex sets have unique support functions.
For affine directions (x,1), the condition instead concerns upper hulls, or
equivalently downward-closed extensions of the lifted polytopes. Under HZ
predicates, only a restricted subset of directions is queried, so unrestricted
polytope equality is sufficient but not necessary. Complete restricted
dominance would already decide whether h_P-h_Q<=0 on the input set: it cannot
be treated as a free simplifier. Real-coefficient upper-hull dominance and
composite simplification have prior art in Kordonis and Maragos,
[Proposition 1 and section VI](https://arxiv.org/html/2306.15157).

### 3.2 Flat normalization cannot guarantee small size on dense banks

Let M be any dense invertible n-by-n matrix, and the input box contain a
neighborhood of zero. Set F(x)=sum_i ReLU((Mx)_i).
Every open sign orthant of Mx intersects that neighborhood: scale M^{-1}s
for a strict sign vector s. On the corresponding cell F has gradient M^T b,
b in {0,1}^n. Invertibility makes all 2^n gradients distinct.

Suppose on that box F=max_{a<=p} ell_a - max_{b<=q} t_b for finite affine
lists. Away from their finitely many tie hyperplanes a gradient is one of
at most pq slope differences. Every open F-cell contains such a point,
after identical affine list entries are deduplicated. Therefore

    p*q >= 2^n, hence p+q >= 2*2^(n/2).

This is a paper counting proof, not runtime enumeration of phases. It applies
even if a different DC decomposition is selected. It does NOT bound the size
of factorized or extended representations: the n input hinges themselves
are compact. Using such factors avoids the flat list, but does not by itself
reduce the paid neural factor/terminal encoding.

No assertion is made that this counting lemma is a new literature result.
It rejects the proposed general polynomial-size flat normalization guarantee
on ordinary full-rank dense banks, not just a numerical exceptional case.

Native HZ cannot treat support values as free continuous factors: their max
relations must be constrained or the original circuit lowered. All original
phase identities, guards, shared source coordinates and input reconstruction
remain payable. Ordinary HZ can use the SAME cancellation identities.
Decision: preserve the identity as supporting algebra; do not implement a
full tropical normalization pipeline as the requested innovation.

## 4. Candidate C: activation-difference factors and cycle consistency

Let d_ij=f_i-f_j and e_ij=r_i-r_j. ReLU is monotone and 1-Lipschitz, hence

    e_ij*(e_ij-d_ij)<=0.

With original integral phases, two inactive neurons have e=0, two active
neurons have e=d, and mixed phases have the directed sector between 0 and d.
Storing e as explicit edge variables requires definitions or reconstruction;
if e already means r_i-r_j, every cycle sum is an identity.

### 4.1 Exact characterization of the missing offsets

Fix f and a phase partition A/I with both groups nonempty, f_A>=0, f_I<=0.
Consider ALL pairwise within-group exact differences and mixed-group sectors,
but not the original absolute integer gate equations. Then exactly

    r_i=f_i+a for i in A,   r_j=b for j in I,
    -min_{i in A} f_i <= a-b <= -max_{j in I} f_j.          (G)

Proof: differences within each group fix values up to one scalar offset.
Each cross pair has 0<=f_i+a-b<=f_i-f_j, equivalently
-f_i<=a-b<=-f_j; their intersection is (G). Conversely (G) implies every
stated pairwise condition. Explicit edge/cycle consistency adds nothing.

An exact inactive output anchor forces b=0. Even after r>=max(f,0), the
remaining allowed offsets are 0<=a<=-max_I f. If every inactive f is strictly
negative, this interval has positive length. Similarly an active anchor
forces a=0 but leaves 0<=b<=min_A f when all active f are strictly positive.

Within the restricted repair method that adds only node absolute anchors,
one anchor from EACH strict-sign group is necessary and sufficient. This
does not prohibit other repairs: a-b=0 plus one anchor also suffices, and
zero-boundary values can force an offset. It is not a universal lower bound
on neural abstraction.

### 4.2 Ordinary mixed-bank counterexample, away from zero activations

Take x,y in [-1,1],

    f_1=x+y-1, f_2=2x-y+1, f_3=-x+2y+2.

All rows are mixed, nonparallel and distinct. At (0,0), f=(-1,1,2) and
the actual phases are beta=(0,1,1). The false output

    r=(0,5/4,9/4)

passes every pair condition, every cycle, the exact inactive anchor r_1=0,
and all r>=max(f,0). It has offsets a=1/4,b=0, permitted by (G).
True ReLU outputs are (0,1,2). Strict sign gaps and strict mixed-phase sector slack
persist in a neighborhood; this is not an epsilon/zero-boundary pathology.

The scalar bounds are [-3,1],[-2,4],[-1,5]. Their triangle upper values at
the center are (1/2,2,5/2), also admitting the false output.
Even the three individual ideal (x,y,r_i) continuous hulls admit it:
for neuron 1 use its exact center point; for neuron 2 mix the center with
weight 3/4 and the two corners (1,-1),(-1,1) with weight 1/8 each; for
neuron 3 mix the center with weight 1/2 and those corners with weight 1/4
each. Each has mean input zero and the stated respective r_i.
These are DIFFERENT mixtures, and this claim projects away their bits;
it is NOT membership in an ideal hull with beta fixed to the integral tuple.

The counterexample does NOT pass the complete original HZ integer gates.
It refutes a proposed replacement by weak differential factors, not original
HZ exactness. Retaining the original gates makes the new facts redundant
for exact concretization; they may still strengthen a query relaxation.

### 4.3 Sparse topology does not remove the issue for free

On k>=3 original nodes, if a FIXED graph must make each of the two induced
phase groups connected for EVERY nonempty bipartition, it must be complete.
For any pair i,j, choose it as the entire active group; the edge ij is then
necessary. The method needs k(k-1)/2 edges. This statement assumes all
bipartitions, not merely a particular network's feasible phase patterns.
It is limited to within-phase propagation on original nodes.

In particular adding a fixed zero reference (f_0,r_0)=(0,0) evades that graph
premise. Its k star edges give r_i(r_i-f_i)<=0; together with r_i>=0 and
r_i>=f_i the product is also nonnegative, so it must vanish. This is EXACT
ReLU with only k such constraints, but is the familiar scalar complementarity
or QC description, not a new domain or a new permitted quadratic solver.

The repeated-nonlinearity sector and its Laplacian combination are already
in Fazlyab, Morari and Pappas, [section III-C3 equations 15-17](https://arxiv.org/html/1903.01287v3);
section III-D equation 19 gives the scalar ReLU complementarity relation.
DiffPoly has cross-execution affine/difference bounds and the phase/difference
ReLU cases in [section 4, Tables 2-3](https://debangshu-banerjee.github.io/assets/pdf/RaVeN.pdf).
Its linear difference abstraction does not retain all ReLU bits as our strong
semantics requires; the local inference rules nevertheless are not new.
No backward analysis or alternative solver from these works is adopted.

Decision: the offset theorem identifies exactly what this weak factor loses.
Cycle closure alone is rejected as an exact simplification. A claimed new
domain must explain its absolute phase/value link, not just its edge ledger.

## 5. Consequence and next research boundary

These are bounded hypothesis tests, not three algorithms added to a runtime
menu. No instance identity, public result, terminal margin or solver status
selected a rule. All old files and results remain read-only.

The next mathematical object must retain quantitative coupling between phase
and amplitude, and give an explicit closed, bounded concretization or a stated
sound overapproximation. Merely restoring every original gate behind a new
factor record is not the missing contribution. Any proposed abstraction must
identify the correlations it intentionally forgets and the inference it gains;
exactness, novelty and benchmark utility remain separate obligations.

A concrete next question is an intermediate, quantitative generator relation:
for two same-frame activations define the directed phase-disagreement charge

    q_ij = r_i - beta_j f_i = (beta_i-beta_j) f_i >= 0.

It vanishes for equal phases; q_ij+q_ji=(beta_i-beta_j)(f_i-f_j)>=0.
Moreover 0<=q_ij<=q_ij+q_ji<=|f_i-f_j|: with unequal bits the two values
have opposite weak signs, and with equal bits both charges vanish. These
statements concern the original INTEGER phase semantics, not arbitrary
fractional query assignments. Unlike a bare sign or sector, the charge
retains amplitude and has a certified common-source difference bound.
These elementary identities are NOT a novelty claim (they relate to convex subgradient defects
and QC/product lifting). The next hypothesis test is whether a finite FAMILY
of such charges supports a useful forward composition/reduction across a
mixed affine/ReLU block, rather than becoming Sharp-HZ product lifting,
DiffPoly, or the original neural graph in different coordinates. First derive
or refute that theorem and compare complete costs; do not implement it solely
because the scalar identity holds. This question is an open research action,
not a smaller replacement for the active Goal.

No full replay, new CERT, validated ADV, native admission, speed claim or
default enablement follows from this checkpoint. D010 remains available as
unqualified support and is not destroyed. The next turn must not silently
resume its runner/test expansion instead of the definition question.
