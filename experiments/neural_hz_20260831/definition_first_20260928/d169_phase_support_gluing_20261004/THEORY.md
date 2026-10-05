# Phase support and birth-shared witnesses: results and limits

Paper-only research, 2026-10-04. The objective remains a stronger nonconvex Neural-HZ definition, not another helper verifier. This note establishes a finite consequence of the existing D157 native domain and tests a birth-shared two-layer construction. Neither is promoted to a new domain or implemented.

## 1. A uniform finite consequence of a common native witness

Use the D157 carrier C in R^(p by m), original phase indicators beta in {0,1}^m, source displacement d, and s=Cd. Indicators are the existing phase labels in 0/1 notation, not replacement or additional bits. Its bare native predicate has the equivalent witness form

```text
a = C D_beta dhat,
s = C dhat,
||dhat||_2^2 <= E.
```

Let a fixed reliable constant R satisfy R >= sqrt(Emax), with Emax bounding E over the entire parent native domain and all original phase assignments. For any fixed direction w, set h=C^T w. Then

```text
|w^T a|     <= R sum_i |h_i| beta_i,
|w^T(s-a)| <= R sum_i |h_i| (1-beta_i).                 (1)
```

Proof: each |dhat_i| <= R. Expand the two sums over the active and inactive coordinates and apply the triangle inequality. Each absolute inequality yields two LE rows. No new amplitude, bit, optimization, branch enumeration, backward pass or dual repair is required. All original predicates, identities, guards, zero labels and the input decoder remain.

These inequalities hold on all integer native members and hence on their convex hull. We do not assert them for an invented fractional-phase extension of the quadratic native predicate; fractional phases belong to the finite LP outer representation, not D157 native membership.

One deterministic direction rule is exact RREF in the original fixed column order, with the first available nonzero pivot row and recorded row transformation H=TC. Use every row of T. This is fixed matrix algebra, not a choice based on a model identity, property, phase solution, terminal margin or LP state. A zero row of H yields the corresponding exact carrier dependency. Approximate rank decisions or nominal BatchNorm coefficients with unaccounted errors are not certified by this proposal.

## 2. A strict finite-outer improvement on the four-gate control

Take the complete declared consumer matrix and D157 canonical carrier

```text
B = [ -1  1  1  1
       1  0  1  0
       1  0 -1  0 ],
C = [B; 1 1 1 1].
```

Let x in [-1,1]^4, g_i=x_i+x_(i+1)/4+b_i with cyclic indices, and b=(1/4,-1/4,1/4,-1/2). Here d=g-b, |d_i|<=5/4, E=25/4 is reliable, and R=5/2. At x=0 and beta=(1,0,1,0), consider the finite-outer point

```text
s=0, a=(1/4,0,0,1/4).
(Y1,Y2,Y3,Q) = C D_beta b+a = (1/4,1/2,0,3/4).
```

The original sign guards hold. Active coordinate bounds are [-3/2,3/2], [-1/2,5/2], [-3/2,3/2], [-1/2,5/2]; all four amplitudes lie within them. Inactive bounds for coordinates Y1 and Q are [-5/2,3/4], containing -1/4; the other two inactive coordinates equal zero. Mass dominance gives Y1,Y3 in [-Q,Q] and Y2 in [0,Q], which also holds.

The six energy directions per coordinate do not reject this point. On the two nonzero-amplitude rows, ||C_j||^2=4 and t=5/4. The largest opposite-direction left side is 4t(1/4)=5/4, below E+4t^2=25/2. The largest single-branch left side is 5/8, below 75/8. The other coordinates have zero a_j and s_j, so their energy left sides are zero. Thus this point passes the old finite rows, although it is not a common-witness native member.

The stated RREF rule produces

```text
T = [  0  1/2  1/2  0
       1    0    1  0
       0  1/2 -1/2  0
      -1   -1   -1  1 ],
H = [ 1 0 0 0
      0 1 0 1
      0 0 1 0
      0 0 0 0 ].
```

In particular w=(1,0,1,0) has C^T w=(0,1,0,1). Equation (1) supplies

```text
|a_Y1+a_Y3| <= (5/2)(beta_2+beta_4).
```

The false point violates this inequality. On the stated integer chart it enforces a_Y1+a_Y3=0. The actual consumer J=Y1+Y3=q2+q4 has zero bias contribution on that chart, hence J=0. The next preactivation J+x4/10-1/8 is at most -1/40, whereas the old finite-outer point at x=0 gives ReLU value 1/8.

This is a fixed-chart mathematical control, not a whole-box CERT, an ADV, a benchmark gain or proof of superiority over full HZ. The old exact four-gate HZ already obtains q2=q4=0 and J=0 on this chart.

## 3. Complete cost and semantic classification

For rank r, exact row reduction with recorded T costs approximately O(r p (m+p)) arithmetic operations, plus rational bit growth, pivot checks and proof metadata. T can occupy p^2 entries and H can occupy pm entries. Fill is real cost; no wide CNN or GPU feasibility is established.

There are at most 4p additional LE rows. For a direction w, let omega=nnz(w), sigma=nnz(C^T w), and nu=nnz(w^T s) in the actual parent-coordinate basis. The four rows have at most 4 omega+4 sigma+2 nu coefficient occurrences before collisions; source expansion, energy-bound derivation, RHS storage, identities, evidence and terminal consumption remain payable.

In this control, the D157 count is 2m+10p+2 rows(B)=54 rows; adding every direction gives at most 70, with four packet amplitudes unchanged. The old exact four-gate encoding has 16 gate rows. These counts exclude common inherited predicates and readouts, which must be paid in either implementation. There is no demonstrated total-cost improvement.

On a common D157 native fiber, (1) is already implied. It strengthens its finite query/compiler but does not change native concretization, remove its legitimate kernel witnesses or create a new foundational domain. On a domain with independent local proxies, imposing common compatibility at birth would instead restrict concretization; it cannot be mislabeled a lossless post-hoc consequence of that old parent.

Replacing R in (1) by actual per-gate displacement bounds is not automatically a native consequence: dhat need not satisfy those bounds coordinatewise. Such tighter rows would need a separate true-image soundness argument as additional birth-time strengthening. D157 already distinguishes such true-image branch rows from its bare fiber consequences.

Decision: retain (1) as a supporting native-to-finite bridge. Do not implement dense RREF or claim a new Neural-HZ definition on this evidence.

## 4. Sharing a witness from birth is sound but not sufficient

Let a parent retain source xi, its entire predicate P and decoder. Introduce bounded eta with 0 in its allowed set and set xihat=xi+N eta. For a declared two-layer block define

```text
qhat = D_alpha(A xihat+b),
rhat = D_gamma(V qhat+c),
z = K qhat+U rhat.
```

All internal guards, declared consumers and uses of these amplitudes use the same (xi,eta,alpha,gamma). A raw-source skip, if present, retains its actual xi readout; the displayed formula does not silently replace it or omit live consumers. The original source and input decoder are retained. Optional original-source first-layer guards must also be soundly justified.

For every member of the entire parent domain, eta=0 and the actual new gate values provide a common extension. Both labels at zero remain legal. Thus this construction avoids the error of independently choosing proxies and then declaring them correlated after the fact.

However, its source coefficient for a fixed pair of phases is

```text
F_alpha,gamma = (K+U D_gamma V) D_alpha A.
```

For an unconstrained centered unit source box and fixed phases, directional source support is ||F_alpha,gamma^T w||_1; biases, non-unit widths, eta and source skips add their actual terms. With guards this free-box formula is an upper bound, not automatically exact. Unknown phases produce genuine alpha_i gamma_j interactions. D161 already rules out general degree-one exact closure on an ordinary open-chart control. Retaining the complete phase circuit is an old representation route, not a new low-cost query theorem.

### A conditional LP containment theorem

Assume the new lowering uses the same per-gate four-row relaxation with the same or weaker bounds and the same parent LP. Also assume every additional constraint permits eta=0 for every old LP point, not merely for each native member. Then every old LP point extends to the new one with eta=0 and its old amplitudes. Consequently

```text
projection(P_new) contains projection(P_old).
```

Exact auxiliary elimination does not change that containment. Merely adding the shared proxy and using that lowering cannot improve LP precision. Native eta=0 soundness alone does not establish the stronger LP extension premise. The theorem does not cover joint native-image compilation or additional native-valid consequences that exclude old LP points.

The ordinary lowering retains all m+n original bits, introduces dim(eta) new continuous variables, and restores m+n gate amplitudes with about 4(m+n) gate rows, plus all guards, source and complete-consumer dependencies. A shared circuit avoids explicit monomial expansion but does not remove these terminal costs.

Decision: no new implementation. The missing result is a paid, directly consumable joint-image representation with non-redundant strength, not another syntax for sharing witnesses.

## 5. Supplemental independent-bank correlation control

This small control isolates why birth binding and post-hoc correlation differ. Use two sequential banks, both reading g=x+(1/4,-1/4,1/4), x in [-1,1]^3, B=(1,-1,1/2), C=[B;ones], and k=(3,1,-4) in ker(C). Take x=k/64 and beta=(1,0,1) in both banks. These are equal assignments to two distinct original bit lists; their identities are not merged. Both g and its phase signs are ordinary nonzero values.

The true active displacement has first packet component 1/64 in each bank, so their product is 1/4096. Independent bank proxies dhat_1=k/32 and dhat_2=-k/32 satisfy C dhat_j=Cx=0 and ||dhat_j||^2=13/512<3. Their packet displacements are respectively (1,-1)/32 and (-1,1)/32; both satisfy the declared coordinate branch and mass rows. The outputs are (Y,Q)=(13,15)/32 and (11,17)/32. Their first displacement product is -1/1024.

The second bank is a descendant retaining the first, not an illegal sibling merge. Forcing the two amplitudes to match their true common trajectory would remove a legal independent-bank parent assignment. It can be a separately proved stronger birth transformer, but not an exact rewrite of that already enlarged parent domain. This is not a network counterexample or a new candidate.

## 6. Exact separator gluing: known premise, unpaid closure

D059 already proves the basic relation identity, for disjoint private variables u,v and the same complete separator assignment (s,b),

```text
exists u,v: P_A(u,s,b) and P_B(v,s,b)
iff (exists u: P_A(u,s,b)) and (exists v: P_B(v,s,b)).
```

Running-intersection trees permit leafwise witness gluing when compatible complete separator assignments are provided and all cross-consumer, predicate and decoder uses are included. Local nonemptiness alone does not establish that compatibility. Matching source/phase names or means is not matching complete relations; convex-hull gluing in general needs a joint separator distribution, not only a few moments.

For an ordinary row-wise k by k convolution layout with stride s<k, a shared input strip can have approximately (k-s) W C_in coordinates away from boundaries. This is a count for that layout, not a universal lower bound. Fixed small kernels do not by themselves make that interface constant-sized. Exact projection, coefficient fill, all original bits and terminal costs still require accounting. Keeping all private witnesses avoids projection work by retaining the old graph; it does not solve the stated research problem. No junction-tree verifier, phase table or split procedure is proposed.

## 7. Research disposition

The positive result is a uniform valid family that strictly strengthens one finite outer approximation. The negative results prevent us from presenting it, shared proxy syntax, or known separator gluing as a new powerful abstract domain. They do not prove impossibility of Neural-HZ.

Next definition work must exhibit a complete common-image relation whose information survives mixed affine consumers and the next activation, with a direct paid query and a demonstrable advantage over the strongest same-information old path. Preserve nonconvex original phases, source identity, all predicates and witness validation. Do not initiate implementation for the two unqualified proposals in this note.

No new mathematical component qualification, model run, GPU result, formal CERT or validated ADV was produced. Formal baseline remains 1870/2413; independent E0 remains CIFAR100 25 and TinyImageNet 36, totaling 61/400. Their gains are both zero and their totals are never added.
