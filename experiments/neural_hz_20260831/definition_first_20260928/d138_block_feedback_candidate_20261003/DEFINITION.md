# Small feedback blocks in a shared Neural HZ fiber

This paper candidate extends the shared triangular fiber to small cyclic blocks. It keeps a single amplitude assignment for every Conv, Add and residual consumer, retains the original nonconvex HZ predicates and phases, and has an exact linear-arithmetic-cost support query for the selected fiber. A two-source neural control proves that preserving both directions of a feedback relation can stabilize two subsequent ReLUs, whereas a fixed triangular cut loses one of those relations. These cycles concern valid abstract amplitude dependencies; the original neural network is still feedforward.

This is a project-level calculus extension, not an established original optimization principle, a completed Neural-HZ, a GPU result or a benchmark gain. Its support equations are standard Bellman/M-matrix structure. The open research obligation is a useful, fully paid nonconvex neural abstraction and composition on real networks, not merely a better optimizer for a familiar convex subsystem.

## Element and concrete semantics

Retain the owned original source relation S(z,beta), all continuous source coordinates, all original signed binary identities, EQ/LE predicates and original-input decoder. The notation beta=(signed_bit+1)/2 does not change a bit's identity or discrete type. Retain one shared nonnegative amplitude vector e, additional linear mixed predicates P, and all visible readouts y=d+Az+B beta+Ce.

Replace the triangular selected query subsystem by

```text
F(z,beta) = { e >= 0 : e <= c(z,beta) + M e },
Gamma = { (z,beta,e,y) : (z,beta) in S, e in F(z,beta),
                           P(z,beta,e), y=d+Az+B beta+Ce }.
```

M is constant and nonnegative. c is affine in the retained source and original phases, with a certificate of c>=0 on the admitted source; constant caps are a special case. After any explicitly paid positive diagonal normalization, the directed graph of the ENTIRE selected M has strongly connected components of size at most two. Its diagonal is zero, and each two-coordinate component has couplings a,b>=0 with ab<1. The condensation graph supplies a block topological order. This makes M block lower triangular with spectral radius below one.

If a raw diagonal M_ii is present, require 0<=M_ii<1 and divide its entire row, including c and all off-diagonal terms, by d_i=1-M_ii. The normalized diagonal is zero and the feasible e-set is unchanged. Nonnegative coefficients, denominator positivity, normalization bit growth and the final whole graph must be checked. Normalizing only internal pair coefficients would be incorrect.

F itself is convex. Gamma remains nonconvex because its original integral phases and exact neural graph predicates remain. Empty amplitudes embed ordinary HZ exactly. After aligning the same source and labels, concrete inclusion defines precision; quotienting equivalent presentations yields a partial order, not an asserted computable best abstraction or complete lattice. The input decoder remains unchanged. A fiber optimizer need not satisfy P and is never automatically an ADV.

There is exactly one selected upper row per amplitude in F. Other old upper bounds, source predicates, on/off conditions and complete ReLU rows remain in P. The formulas below do NOT optimize the intersection of arbitrarily many upper rows for each coordinate. Reclassifying which row belongs to F must be structural and fixed before queries; it cannot be selected by property, margin, LP state or instance identity. That reclassification can weaken some old cheap bounds even while preserving exact Gamma, so no blanket zero-regression claim is made.

## Support of a nonnegative contractive feedback fiber

Fix source and phase, omit P temporarily, and consider any signed w. More generally, let M>=0, c>=0 and rho(M)<1. Then

```text
F = { e >= 0 : (I-M)e <= c },
k* = positive_part(w + M^T k*),
support_F(w) = c^T k*.
```

F contains zero and is compact because e<=(I-M)^(-1)c and the inverse is nonnegative. Choose a positive vector v and kappa<1 with M^T v<=kappa v; such a vector exists under rho(M)<1. The positive-part map is a contraction in the v-weighted supremum norm, so k* exists and is unique. Its inequalities k*>=0 and (I-M^T)k*>=w give w^T e<=c^T k* for every e in F.

Attainment does not need a runtime LP: let A be the indices where k*_i>0 and set e_A=(I-M_AA)^(-1)c_A, e_outside=0. Principal submatrices also have spectral radius below one, so this e is nonnegative. Active primal rows are tight; inactive rows are valid because c and M are nonnegative. The dual residual vanishes on A. Consequently w^T e=k*^T(I-M)e=c^T k*. Selecting A in this proof is not an algorithm enumerating original neural phase assignments.

The requirement c>=0 is essential. For c=(-1,2), M=[[0,1/2],[1/2,0]], F is {(0,2)}. Direction w=(0,-1) has support -2, whereas the displayed fixed point is zero. Negative-cap fibers are outside this candidate, not silently repaired.

With P restored this is only an upper bound for Gamma. With affine c(z,beta), k is independent of the particular source and phase value, so k^T c is a sound affine source-and-phase envelope. Optimizing that envelope over S remains a separately charged task. A known independent source-box bound is sound but may lose phase/guard information; retaining bits alone is not proof of stronger phase reasoning.

## Closed form for two coordinates and exact block composition

For one normalized block M_BB=[[0,a],[b,0]], define D=1-ab>0. For its current signed direction w_B,

```text
k1 = max(0, w1, (w1+b*w2)/D),
k2 = max(0, w2, (w2+a*w1)/D).
```

Substitute k2=max(0,w2+a*k1) into the first fixed-point equation. This gives k1=max(0,w1,w1+b*w2+ab*k1), whose unique solution is the displayed maximum; the second coordinate follows symmetrically. Zero direction, zero coupling and one-way edges are included. A normalized singleton uses k_i=max(0,w_i). The formula's maxima do not branch over original phases or create subproblems.

Process components in reverse block order. First coalesce ALL contributions from later consumers into the block's effective signed direction. Apply its singleton or pair formula, then add M_BA^T k_B to every earlier block A. Accumulate k_B^T c_B. Each fixed earlier prefix gives a nonnegative block offset c_B+sum_(A<B) M_BA e_A, so it extends through that block. The support theorem therefore applies at each elimination step and proves exact total support k^T c.

For r directions this costs O(r(n+nnz(M))) scalar arithmetic after readout formation, graph construction and admission. No dense inverse is formed. If raw rows were normalized, the corresponding certificate on raw rows is k_raw,i=k_normalized,i/d_i. Do not forget this conversion when checking original rows.

This extends D135's single-coordinate recurrence. It does not add a guarantee that arbitrary larger components have an equally cheap closed form. They are outside this proposed implementation class; they cannot be silently split after observing a query result.

## Uniform neural relation generation and exact transformers

For two authenticated ReLUs q_i=R(f_i), q_j=R(f_j) over the same current element, any structurally fixed eta>=0 and reliable bound U>=f_i-eta*f_j give

```text
q_i <= eta*q_j + max(0,U).
```

Proof: ReLU is monotone, subadditive and positively homogeneous for eta>=0. The source certificate must precede the new relation; the relation cannot certify its own cap. Both gates' complete original integer graph constraints and zero labels must remain. Then adding the relation preserves the exact current-state ReLU transformer, not merely its behavior on a few original concrete trajectories.

For a same-prefix bank, one can instead merge the difference first as

```text
f_i-eta*f_j = h(z,beta) + sum_l r_l e_l,
q_i <= eta*q_j + max(0,U_h) + sum_l max(0,r_l)e_l,
U_h >= h on the retained source.
```

All e_l here precede the new bank. Two directions within disjoint bank pairs therefore create local feedback without introducing a cycle back through previous banks. This is one constructive route to the required block graph. Pairing, eta selection, reliable coefficient binding, unmatched gates and all budgets still require one frozen structural policy before execution. This paper does not claim such a policy is already source-qualified or that its chosen pairs are useful on trained filters.

Using independent scalar bounds destroys the intended information: if both gates cross zero, U=U_i-eta*L_j gives max(0,U)>=U_i, so the new row is already implied by q_i<=U_i. Shared source coefficients must actually be combined before bounding. Their discovery and arithmetic cost cannot be omitted.

Affine/Conv compose the same visible readouts; Add/Concat coalesce identical latent coordinates. A retained identity skip simply keeps reading the same e. No amplitude, original bit or predicate is deleted. Appending another exact ReLU preserves its original bit and four graph rows. A bound-certified inactive gate may be assigned a zero selected row, preserving both zero labels and making its zero available to subsequent queries, as in D136. All unmatched additional valid rows stay in P.

This proves a compositional carrier and the scope of its selected query, not a new larger PWA set class or a complete useful discovery algorithm. It neither requires nor proves elimination of all raw residual ports. Complete actual consumers must still be represented and charged.

## An ordinary simultaneous consumer and successor control

Take x,y in [-1,1] with

```text
f1=x+y/10+1/4,       q1=R(f1),
f2=x/2+y/10+1/4,     q2=R(f2).
```

Both gates cross zero and their affine normals are nonparallel. Their tight preactivation intervals are [-17/20,27/20] and [-7/20,17/20]. Combining the common source gives f1-f2=x/2<=1/2 and f2-f1/2=y/20+1/8<=7/40. Thus

```text
c=(1/2,7/40),   M=[[0,1],[1/2,0]],
J1=q1-q2 <= 1/2,          k=(1,0),
J2=q2-q1/2 <= 7/40,       k=(0,1).
```

The bounds are global and attained; x=y=1 attains both. Consequently s1=R(J1-11/20) and s2=R(J2-1/5) are both identically zero. Their certified preactivation upper bounds are -1/20 and -1/40.

Grant the triangular-cut comparator the exact coordinate bounds u=(27/20,17/20). Ordering q1 before q2 replaces the removed M_12*q2 by its cap and gives q1<=27/20, q2<=7/40+q1/2. Its J1 support is 27/20 rather than 1/2. Reversing the ordering keeps q1<=1/2+q2 but replaces the other row by q2<=17/20; its J2 support is 17/20 rather than 7/40. These are exact supports of those specified selected fibers. Additional old P can tighten either, so the comparison is not a claim about their complete LPs.

The original independent four-row LP, with the SAME source and original gate intervals, also admits these two separately checked points:

| Source | q1 and its original phase | q2 and its original phase | Positive successor |
| --- | --- | --- | --- |
| x=1, y=-1 | 27/22, 10/11 | 13/20, 1 | s1=3/110 |
| x=0, y=1 | 7/20, 1 | 119/240, 7/12 | s2=29/240 |

For the first, both q1 upper rows equal 27/22, f2=q2=13/20 and J1=127/220. For the second, f1=q1=7/20, both q2 upper rows equal 119/240 and J2=77/240. Lower rows hold in each case. Common conservative successor intervals [-7/5,4/5] and [-7/8,13/20] permit each positive successor with its own original phase equal to one. The other successor can be zero in each point. These are distinct fractional diagnostic assignments, not a combined witness or an ADV.

Old HZ with the SAME two valid feedback rows proves both improved bounds immediately. Its integer graph already has the true result. The project's positive result is preserving and cheaply consuming both shared relations in one fixed query structure, including downstream stabilization; it is not a new source inequality principle or superiority over that exact same-evidence LP.

## Complete costs and numeric boundaries

The O(r(n+nnz(M))) statement is a query arithmetic count, not an end-to-end runtime or memory result. Pay for structural matching, eta, current-source/BN coefficient enclosures, residual merges, cap certification, whole-graph components and ordering, diagonal normalization and rational bit lengths. Also pay for all readouts/directions, affine source envelopes, original P and bits, row lowering, certificates, decoder, evidence and host/device coexistence. Sparse cross-block edges may become dense after propagation.

On physical q1,q2 columns, the control's two additional feedback inequalities contribute two rows and four nnz, no new continuous or binary columns. They remain additional to the old graph rows and any retained bounds. Counting this tiny local increment does not price real Conv construction or terminal work.

Independent blocks at one dependency level admit batched arithmetic and sparse reductions on GPU. Sequential block depth and synchronization remain. No GPU kernel, reliable device rounding, source run or timing was performed here. A componentwise upper approximation to k* need not itself satisfy the dual inequalities: increasing one coordinate also increases its neighbor's required RHS. A floating implementation therefore needs proved interval evaluation/monotonicity or an independently verified full certificate, not an unqualified reuse of an exact-rational claim. Unproved denominators or excessive arithmetic fail closed; do not create a rescue for extreme near-singularity.

## Prior art and supplementary general cycles

Kallenberg's optimal-stopping equation and its majorant LP, Section 5.3.2 equations 29 and 30, have exactly the positive-part fixed-point form after zero terminal reward and positive diagonal scaling. Our M-matrix support specialization is therefore not a new optimization principle. [Primary paper](https://link.springer.com/article/10.1007/s10479-011-1047-4)

The underlying nonnegative-matrix contraction and geometric convergence are established as well; see Toda, Section 2, Theorem 1 and Proposition 2. [Primary paper](https://arxiv.org/pdf/2310.04593)

For completeness only, general contractive cycles permit a fixed-iteration upper bound. With v>0, M^T v<=kappa*v, kappa<1, k0=0 and k_(t+1)=(w+M^T k_t)_+, let r=k_(t+1)-k_t and epsilon=max_i r_i/v_i. Then khat=k_t+epsilon*v/(1-kappa) satisfies khat>=w+M^T khat, and

```text
c^T k_t <= support_F(w) <= c^T khat.
```

This follows by monotonicity and the contraction inequality. Obtaining and checking v,kappa is not free. It is a supplementary paper scope result, NOT a second registered execution path or a runtime fallback. The proposed next implementation uses the closed small-block formulas only.

The k variables are mathematically LP dual certificates. The intended boundary is an intrinsic, predeclared query of the represented fiber, not failed-LP feedback, a generic backward neural verifier, a dual optimizer, or infeasibility repair. That distinction must remain explicit. All original restrictions on helpers and promotion remain unchanged.

## Research decision

The next step is to write a default-off same-frame reference and its new contract, then freeze both before any import or execution. It must preserve the complete D136 test population and original gates, exercise both simultaneous consumer directions and their successors, and reject invalid whole-block graphs and source certificates. It must not stop at a standalone optimizer detached from the retained nonconvex element.

Then test useful relation generation and total cost on the complete ordinary Conv/Add consumers already recorded in D137. The identity skip need not be eliminated, but it must remain part of the shared state and cost. No real coefficient census, full frontier qualification, smooth/Transformer rule or source-independent novelty result is supplied by this paper.

Formal 1870/2413 and independent CIFAR25 plus Tiny36 equal 61/400 remain unchanged, with zero new solves. The full Neural-HZ goal remains active.
