# D012 — Phase budgets: a useful invariant, not yet a new Neural-HZ domain

2026-09-28; branch `redu-hz`; HEAD
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac` (dirty supporting work preserved).
Configuration: paper derivation, independent read-only review and primary-source
research only. No imports, tests, model execution, solver, benchmark or new
numerical qualification. Formal baseline remains 1870/2413 (1063 CERT + 807
validated ADV); separate E0 remains CIFAR10025/TinyImageNet36 = 61/400.
Authority: [definition-first goal](../../GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md).
All original bits, EQ/LE, shared source identity, reconstruction, replay gates,
resource caps and historical read-only custody remain unchanged.

## 1. Disposition and the definition-level question

D011's quantitative phase-disagreement question has both negative and positive
answers. Independent charge products give NO LP projection strengthening.
Bounded joint energy can give integral phase-count information, and that
information can soundly propagate through the next affine operator. An exact
projected hull theorem and two ordinary mixed-bank controls below establish
the latter with explicit comparison scope.

These results alone are a resource reduced product over the retained exact
HZ graph. Ordinary HZ supplied the same rows obtains the same improvement.
They are NOT an established new abstract domain or a benchmark gain. The
next definition question is whether certified budgets can normalize the
phase-dependent generator algebra itself, especially across successive layers,
without simply reconstructing the old graph at the terminal query.

## 2. Charge-matrix audit: why the tempting cheap route fails

For exact ReLU, let r_i=beta_i*f_i, beta_i in {0,1}, with the original sign
guards and both original bit choices at zero. Define

    q_ij = r_i - beta_j*f_i = |f_i|*|beta_i-beta_j|,
    Q = r*1^T - f*beta^T.

Q has rank at most two, but storing its factors just stores the existing
f,r,beta. Diag(Q)=0 gives r_i=beta_i*f_i. Q>=0 additionally recovers the ReLU
epigraph only if both phases occur: a column with bit0 gives r>=0, and one
with bit1 gives r>=f. All bits0 instead permits arbitrary positive f,r=0;
all bits1 permits negative f,r=f. Two fixed zero anchors of opposite phases
restore precisely the original scalar epigraph guards, not a new inference.

The charges are not distances. For f=(1,-1,1/10), q12=1 while
q13+q32=1/10; even s_ij=q_ij+q_ji violates the triangle inequality. The exact
replacement is

    q_ij-q_ik-q_kj = (beta_k-beta_j)*(f_i-f_k).

For a fixed column j, |q_ij-q_kj|<=|f_i-f_k|. Also q_ij is the generalized
Bregman divergence of ReLU using beta_j as its chosen subgradient at f_j;
that interpretation is not novel [S1].

**Independent-product no-gain theorem.** Suppose an arbitrary old LP already
has l_i<=f_i<=u_i, r_i>=max(0,f_i), and beta_j in [0,1]. Independently add
t_ij for every i!=j, its four McCormick inequalities for f_i*beta_j, and
r_i-t_ij>=0. Every old point extends by choosing

    t_ij=max(l_i*beta_j, f_i-u_i*(1-beta_j)).

Each chosen value lies in its McCormick interval, and both lower candidates
are <=max(0,f_i)<=r_i. All extensions are simultaneous because the new
products are independent. Thus the projection is unchanged, even with
arbitrary existing source/bit correlations. Extra coupled products could
help, but they are selected binary-continuous RLT lifting, a strong existing
comparator [S2], not a consequence of rank(Q)<=2.

Cost beyond the retained old graph: P=k(k-1) independent products give P
new continuous variables and 5P rows for zero LP gain. Shared source products
Y_lj=x_l*beta_j instead give dk variables, 4dk+k^2 rows and a dense nnz
upper bound 10dk+k^2(d+2), for a purely continuous d-coordinate source.
For k=64,d=27 these symbolic counts are 1728,11008,136064. They are not a
measurement on a faithful CNN source. Original free binary products, original
gates, bounds, equality-only slacks, metadata and witness reconstruction are
additional costs. The rank factorization does not eliminate them.

Finally, the proposed common-reference charge subadditivity formula cancels
its reference phase algebraically: for g=sum a_i*f_i+c it reduces to
r_g<=sum |a_i|*r_i-sum a_i^-*f_i+c^+. On nonnegative previous ReLU outputs
this is just r_g<=sum a_i^+*r_i+c^+. It is ordinary subadditivity, not new
phase-aware composition.

## 3. Exact projection of a symmetric phase-energy relation

Consider the FULL relation, not a particular affine source:

    sum g_i=c, sum g_i^2<=E,
    delta_i in {0,1}, g_i>=0 if delta_i=1, g_i<=0 otherwise,
    n=sum delta_i, T=sum ReLU(g_i), k>=1, E>=0.

For 0<n<k, Cauchy gives

    T>=max(c,0),
    T^2/n + (T-c)^2/(k-n) <= E,
    h(n) = (n*c + sqrt(n*(k-n)*(k*E-c^2)))/k.

These conditions are sufficient as well: assign T/n to each positive slot
and (c-T)/(k-n) to each negative slot. Therefore the exact fiber of T is
[max(c,0),h(n)] when it is nonempty. At n=0 the only T is0 and c<=0;
at n=k the only T is c and c>=0; both need E>=c^2/k.

If E>0 and k*E>=c^2, the feasible integer counts are the contiguous interval

    c>0: ceil(c^2/E),...,k;
    c<0: 0,...,k-ceil(c^2/E);
    c=0: 0,...,k.

E=0,c=0 forces g=T=0 while ALL original bits remain free. E=0,c!=0 or
k*E<c^2 is empty. These cases are proof boundaries, not special runtime
rescue paths.

Let [a,b] be the feasible integer interval, and H be the adjacent-integer
linear interpolation of h, including its feasible endpoints. The exact
continuous convex hull of the projection on ALL (delta,T) coordinates is

    0<=delta<=1, a<=sum delta<=b,
    max(c,0)<=T<=H(sum delta).

Proof: h is concave, so the displayed bounds are necessary. For a fractional
delta with sum s, the polytope 0<=v<=1 with floor(s)<=sum v<=ceil(s) has
integral vertices. Express delta as a mixture of those adjacent-cardinality
vertices. Mixing their maximum capacities gives H(s), and using a common
fraction between their common lower bound and maxima realizes any intermediate
T. This is a proof, not a proposed phase-enumeration algorithm.

With identical bounds L<=g_i<=U0 and L<=0<=U0, replace h by

    h_clip(n)=min(h(n), n*U0, c-(k-n)*L).

Uniform values within each sign group still prove sufficiency. The minimum
of these concave/affine functions is concave, so feasible integer counts
remain contiguous and the same interpolation proof applies. All-positive or
all-negative bounds instead reduce to their appropriate stable sign case.

Limitations: this is not a full (g,r,delta) hull, nor an exact hull after
intersecting an arbitrary source or heterogeneous bounds. General capacities
can be irrational (k=2,c=0,E=1 gives max T=1/sqrt(2)); a finite rational-row
lowering must be certified outward, not called an exact rational hull. No
SDP, new solver path or numerical admission is implied. Cauchy/perspective
and integer-cardinality convexification are existing mathematical ingredients
[S3]; no fixed-level RLT/SDP dominance claim is established here.

## 4. Count inference on ordinary, biased shared-input banks

More generally, certified sum bounds [Slo,Shi] and energy E>0 imply

    Slo>0 => sum beta >= ceil(Slo^2/E),
    Shi<0 => sum beta <= k-ceil(Shi^2/E).

The positive mass is at least Slo and its square is at most n*E; apply the
same argument to -f for the second inequality. A positive-part support upper
bound follows from a count UPPER bound, not a count lower bound. E=0 fixes
all values to zero but does not fix any original phase bit.

For f=z+A*x, |x_j|<=1, one elementary paid source certificate is

    G=A^T*A, B=tr(G)+2*sum_{j<l}|G_jl|,
    E=||z||^2+2*||A^T*z||_1+B,
    [Slo,Shi]=1^T*z +/- ||1^T*A||_1.

Expansion and |x_j|<=1 prove these statements. Original source predicates
are retained; they need not be discarded merely because these bounds ignore
some of them. Dense Gram construction costs O(k*d^2) arithmetic and up to
O(d^2) storage, plus source, bit-complexity and certificate costs. A first
3x3 RGB patch does not make every later CNN source dimension27.

### 4.1 Three-neuron paper control

Take A=((1,1),(-2,1),(1,-2))/3, z=(3/5)*1, x in [-1,1]^2.
Then c=9/5, B=2, E=77/25 and c^2/E=81/77>1, so sum beta>=2.
Each row has d_i=||A_i||_1=(2/3,1,1).

At x=0, the old fractional point beta_i=1/2, r=(z+d)/2,
m=2r-f=d survives each retained-input, retained-phase independent ideal
single-neuron hull: mix the row's maximizing box corner and its opposite.
It also survives the full specified D007 box-defect family and the full
same-source D009 nonnegative weighted Gram-mass family by the general proof
in the next control. Standard upper gate rows force the displayed half bits,
so the new count excludes its whole (x,r) projection.

This count also follows from scalar-capacity cover reasoning or rounding
pairwise phase inequalities. The natural individual perspective-energy
formulation rejects this point too. Therefore this is not evidence of new
geometry beyond those comparators.

### 4.2 Five-neuron control and exact comparison scope

Let k=5, A=I-(1/5)*11^T, z=(21/20)*1, x in [-1,1]^5.
The rows are dense, distinct and nonparallel. A^T*A=A, d_i=8/5,
||A_i||^2=4/5, A^T*z=0, B=8, c=21/4 and E=1081/80.
Since c^2/E=2205/1081>2, the original bits obey sum beta>=3.

The fractional point x=0,beta=(1/2)*1,r=(53/40)*1,m=(8/5)*1 passes:

1. Every independent ideal single-gate hull retaining full x and its bit.
   Use the corner whose i-th entry is1 and all others -1, and its opposite,
   with equal weights. These give preactivations53/20 and -11/20.
2. ALL D007 source-box defect member/anchor inequalities. For any coefficient
   vector a and anchor c0, the member row follows from
   |a_i|d_i-sum_{j!=i}|a_j|d_j<=||a^T*A||_1; the anchor row follows from
   sum |a_i|d_i>=|a^T*z|. Add the nonnegative anchor defect |a^T*z-c0|
   as in D007. No bounded selector of a is being used in this comparison.
3. ALL direct same-bank, same-box D009 Gram-mass rows with weights w>=0.
   Put h=d-z=(11/20)*1. Componentwise h_i^2<=||A_i||^2; therefore
   (sum w_i*h_i)^2<=W*sum w_i*h_i^2<=W*tr(G_w)<=W*B_w.
   Every certified outward version of the corresponding mass row accepts.

Both standard upper gate rows force beta_i=1/2 at the displayed x,r. Hence
sum beta>=3 excludes the entire shown (x,r) projection, not just one optional
bit extension. This proves a strict gap relative to these SPECIFIED continuous
families, not all transformed-coordinate, QC, perspective or RLT families.

Scalar bounds [-11/20,53/20], sum f=c and sign guards alone allow n=2 via
f=(21/8,21/8,0,0,0). This is a scalar-summary witness, NOT an actual source
point. In fact f_i-f_j=x_i-x_j<=2; with at most two active coordinates an
inactive coordinate would imply every active value<=2, contradicting c>4.
Thus pairwise-difference bounds plus integral support reasoning also recover
n>=3. This control does not prove that the energy route is indispensable.

For the direct native control: 5 source +5 ReLU continuous values,5 original
bits,20 gate rows/80 nnz; the count adds1 row/5 nnz. Equality-only HZ needs
an additional bounded slack. Bounds, sources, exact arithmetic, certificates
and reconstruction remain payable. No measured speed or real solve follows.

## 5. Exact phase-referenced residuals and sound forward resources

For a fixed, structurally chosen reference tau_i in {0,1}, use ONLY aliases
of the original bits:

    g_i=(1-2*tau_i)*f_i,
    sigma_i=beta_i if tau_i=0, else 1-beta_i,
    e_i=ReLU(g_i), r_i=tau_i*f_i+e_i.

These identities are exact, including zero values with both original phase
choices. A proved sum sigma<=U gives support(e)<=U. With e>=0, e_i<=u_i and
sum e_i^2<=Ee, where U is an integer in [0,k], u_i>=0 and Ee>=0, the next
affine map is exactly

    W*r+b = W*diag(tau)*f+b+W*e.

For a FORWARD row a, the support over the nonnegative U-sparse box alone is
sum of the U largest a_i^+*u_i. Over the nonnegative U-sparse energy ball
alone it is sqrt(Ee*sum of the U largest (a_i^+)^2). Negative parts give
lower bounds. The first maximizer selects the largest profitable capacities;
the second follows by Cauchy and is attained by a vector proportional to
the selected positive coefficients. No phase search is an implementation
requirement for these formulas; sorting suffices.

These are exact supports of those OUTER FACTORS, not of their intersection
or the full retained-source relation. The minimum of the two upper bounds
is valid but can be loose: U=Ee=1,a=(2,1),u=(1/2,2) gives both separate
bounds2, whereas the intersection's support is1. The sparse-ball expression
is the one-sided version of the existing k-support dual norm [S4].

Whole-layer resources follow without independence assumptions:

    ||W*e||^2 <= Ee * sum_top_U ||W_:i||^2;
    ||W*e+b||^2 <= 2*Ee*sum_top_U ||W_:i||^2+2*||b||^2;
    E_Add <= 2*(E1+E2); E_Concat <= E1+E2.

For the first bound use the Frobenius norm of the active column submatrix.
Forward sum bounds use a=W^T*1; subsequent count inference uses section4.
The displayed energy of W*e+b is NOT the energy of the full next preactivation:
the shared baseline W*diag(tau)*f must also be bounded and combined, for example
using the stated sound Add bound. Likewise its sum interval must be included
before applying a next-layer phase-count rule. Ignoring that term is unsound.
Explicitly, for h0=W*diag(tau)*f+b with a separately proved ||h0||^2<=E0,
one may use ||h0+W*e||^2<=2*E0+2*Ee*sum_top_U ||W_:i||^2, and add the h0
sum interval to the residual sum interval. This does not assume independence.
Shared sources must NOT be cloned into independent error factors. Keeping
the exact e=ReLU(g) relation retains exact HZ semantics; dropping it and
keeping only guards/caps/budget/energy is merely a sound overapproximation.

For control4.1 take all tau=1. Then sum sigma<=1 and
u=(1/15,2/5,2/5). The next ordinary row a=(1,1,1) has sum e<=2/5, so
sum r<=9/5+2/5=11/5, attained at x=(1,-1). Independent error boxes give
8/3 instead, and the old fractional point has sum r=67/30>11/5. This is
genuine forward propagation on a paper control. Ordinary HZ with the same
count and e_i<=u_i*sigma_i obtains the same bound.

Costs include O(k log k) work per straightforward sorted row, O(mk) to form
column norms for an m-by-k matrix, all exact identities, original predicates,
certificate storage and any terminal lowering. Neither a new domain nor a
net native reduction has been established by this resource calculus alone.

## 6. Evidence, prior art and next discriminating test

Root derived and inspected these statements. Independent reviewers checked
charge lifting, all-family control separation, full-phase zero semantics,
the projected hull, forward support formulas and their counterexamples.
These are paper proofs, not machine-checked proofs or numerical tests.

- S1: [Faust--Fawzi--Saunderson, generalized Bregman definition, section2.1](https://proceedings.mlr.press/v206/faust23a/faust23a.pdf).
- S2: [Sharp Hybrid Zonotopes, sectionIV/theorem7/equation18](https://arxiv.org/html/2503.17483v2).
- S3: [Atamturk--Gomez, Rank-one Convexification for Sparse Regression, section1.3](https://jmlr.org/papers/volume26/19-159/19-159.pdf).
  Also [Gunluk--Linderoth, Perspective Reformulations, sections1.1 and3](https://jlinderoth.github.io/papers/Gunluk-Linderoth-09-TR.pdf)
  for the established indicator/perspective construction. Neither reference
  is asserted to publish this exact neural projected-count theorem.
- S4: [Argyriou--Foygel--Srebro, Sparse Prediction with the k-Support Norm, section2.1](https://arxiv.org/pdf/1204.5043).

No claim that a specified finite RLT/SDP level contains every rounded count
row, or that this candidate dominates those methods, has been proved.

Next test: define a phase-budget quotient of D001's generator coefficients,
prove its exact operators and full terminal semantics, and determine whether
budgets actually remove interaction terms on ordinary neural paths. In
particular a within-layer count restriction may provide useful inference yet
remove NO generator monomial, because affine mixing adds paths rather than
multiplying same-layer activations. The next checkpoint must resolve this
distinction before any implementation or real-source census is selected.

D010 drafts remain frozen, unexecuted and unqualified. D003 remains the most
recent numerical reference. No prior source/result, production default, score
or admission rule is changed by this checkpoint.
