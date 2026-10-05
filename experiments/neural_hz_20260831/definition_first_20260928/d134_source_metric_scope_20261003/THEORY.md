# Source restricted ReLU relations and the missing Neural-HZ construction

This checkpoint records a useful mathematical distinction, not a completed new domain. Common affine sources permit cross-channel ReLU relations unavailable on an unrestricted preactivation space. One ordinary four-gate control separates such a relation from a specified two-copy ordered-HZ LP relaxation. However, the relation is quadratic and pairwise: there is no admitted GPU query implementation, multilayer transformer or end-to-end capability result. The user's instruction to focus on a powerful complete Neural-HZ takes precedence over expanding this metric subproblem. We retain the results as supporting research and do not start a metric implementation or numerical replay.

Date 2026-10-03 Australia/Sydney; branch `redu-hz`; commit `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`; pre-existing tracked binary diff SHA256 `29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5`. Configuration: paper derivation, independent paper review, read-only project inspection and primary-source abstract review. No candidate imports, model evaluations, solver calls, numerical tests or GPU jobs were performed for this checkpoint.

The [goal authority](../../GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md) and active goal remain unchanged. Formal baseline is 1870/2413; independent E0 is 61/400. Neither has a new gain. No production default, old evidence or frozen source was changed.

## Semantic scope

An eventual Neural-HZ element must retain its exact common source relation, continuous factors, every original signed binary identity and legal zero label, EQ/LE predicates, shared frame and input decoder. The following propositions are consequences of ReLU on that source. They do not authorize replacing the nonconvex carrier by an ellipsoid, deleting bits, or adding an optimization helper.

Parameterize the actual feasible affine hull before applying an open-set theorem. Let V be a nonempty convex open subset of R^r, g(z)=Az+b, q(z)=ReLU(g(z)), and M=M^T be positive semidefinite. An original source x=x0+Bz is incorporated by replacing its preactivation matrix with A_original B. Assume no g_i is identically zero on V. A constant-zero coordinate can instead be handled analytically as such, without deleting its stored phase identity or either legal zero label.

Write d=g(z)-g(w) and e=q(z)-q(w). The relation under investigation is

```text
e^T M (d-e) >= 0.                                      (F)
```

This is an incremental, two-source relation. It is not automatically a one-state predicate on an ordinary output element.

## Local source conditions are sufficient as well as necessary

For a strict reachable derivative mask D=diag(1_{g_i>0}), define

```text
S_D = (D M + M D)/2 - D M D.
```

Relation (F) holds for every pair z,w in V if and only if

```text
A^T S_D A is positive semidefinite
```

for every strict mask attained at an interior source point.

Necessity follows by taking arbitrary sufficiently small source increments h inside such a cell. There e=DAh and d=Ah, so the quadratic form in h must be nonnegative in every direction.

For sufficiency, first choose a segment not contained in any gate's zero hyperplane. With Theta the average of its derivative masks, e=Theta d. Direct expansion gives

```text
S_Theta - average(S_D)
  = average((D-Theta) M (D-Theta)) >= 0.
```

Pulling back by A proves (F). A segment contained in a zero hyperplane is obtained as a limit of generic segments with endpoints in V; ReLU is continuous. Convexity ensures the whole segment stays in the region where the local conditions were certified. M being positive semidefinite is essential to the displayed sufficiency argument.

This extends [D133's source-local necessary condition](../d133_query_alignment_boundary_20261003/METRIC_BOUNDARY.md). It does not supply an efficient algorithm for identifying every reachable mask. Proof quantification over masks is not permission to enumerate or split them in a candidate.

The convexity assumption cannot simply be discarded. Let g(x)=(x+1,x-1), V=(-3,-3/2) union (3/2,3), and M=[[1,-1],[-1,1]]. The only strict masks are 00 and 11, whose local forms vanish. Yet x=-2 and x=2 give d=(4,4), e=(3,1), and e^T M(d-e)=-4. For a nonconvex upstream HZ relation, a certified convex outer source region can prove sufficiency; it does not justify asserting necessity on that outer region.

## A constructive family that adds no strength to scalar sectors

Let T be diagonal and nonnegative, L positive semidefinite with LA=0, and M=T-L positive semidefinite. Then

```text
e^T M(d-e) = e^T T(d-e) + e^T L e >= 0.
```

Each scalar ReLU incremental sector makes its term in the first sum nonnegative; Ld=0 gives the equality. Therefore this family is valid, but its inequality follows already from scalar incremental sectors and the exact common-source difference equation. Non-diagonal coefficients alone do not establish a stronger abstraction.

For nonzero n with n^T A=0, L=rho n n^T supplies an explicit certificate. If T is positive diagonal, 0<=rho<=1/(n^T T^-1 n) ensures M is positive semidefinite. A rank-one action costs O(m) arithmetic and stored coefficients, but certifying and maintaining n^T A=0, source ownership and all original predicates remain payable. General L=BB^T costs O(mk) for k columns plus a certificate of B^T A=0.

The comparator here is scalar INCREMENTAL sectors with the same source binding. It is not a claim that every ordinary one-copy triangle LP contains those quadratic relations.

There is also a direct mixed-readout limitation. Let P_perp project orthogonally away from the image of A, P_V=I-P_perp, and M=I-kappa P_perp with 0<kappa<1. Given a source-difference bound norm(d)<=R, (F) implies norm(e)_M<=norm(d)_M=norm(d)<=R. At this unchanged source energy, the metric ball alone supplies

```text
abs(c^T e) <= R sqrt(norm(P_V c)^2
                        + norm(P_perp c)^2/(1-kappa)).
```

This cannot improve the corresponding Euclidean-ball support R norm(c), and is strictly worse when P_perp c is nonzero. At kappa=1 the metric alone cannot bound that readout component. Other retained predicates can avoid the loss; this calculation does not prove they yield a gain.

## Two precise limits on strengthening this family

First suppose T is diagonal nonnegative, M>=T in the positive semidefinite order, and A^T M A=A^T T A. Set N=M-T. Then N>=0 and NA=0. Within any strict phase cell, the T firm gap vanishes, so validity of M forces

```text
N^(1/2) D A = 0.
```

If every row a_i of A is nonzero and each gate has an isolated reachable crossing, two adjacent strict masks differ only in coordinate i. Subtracting the two equations gives N^(1/2) e_i a_i^T=0 and thus N^(1/2) e_i=0. Hence N=0. Under these hypotheses there is no positive-semidefinite strengthening at unchanged source energy. This does not exclude incomparable metrics or a different domain construction.

Without the nonzero-row assumption only the corresponding columns are forced to vanish. For example A=(1,0)^T, g=(x,1), T=I and N=diag(0,1) satisfy the unchanged-source-energy condition and (F), despite N not being zero.

Second, consider m>=3, rank(A)=m-1, ker(A^T)=span(n) with every n_i nonzero. Suppose at least m-1 singleton-active strict cells are reachable with locally unrestricted source directions. Every valid fixed M>=0 then has the form

```text
M = T - rho n n^T,    T diagonal nonnegative, rho>=0.
```

To prove this, in a singleton cell i the local condition is

```text
(a_i^T h) (sum_(j!=i) M_ij a_j^T h) >= 0 for every h.
```

The first linear form is nonzero. The second must be a nonnegative multiple of it, including the possible zero multiple: sum_(j!=i) M_ij a_j=lambda_i a_i, lambda_i>=0. The unique row dependency gives M_ij=alpha_i n_j for j!=i and -lambda_i=alpha_i n_i. Symmetry between the available singleton rows forces alpha_i/n_i to be one common nonpositive constant, -rho. There are at least two such rows, and their entries also determine every off-diagonal coefficient of the one possibly missing row. Set T_ii=M_ii+rho n_i^2>=0. The preceding constructive theorem proves the converse whenever M>=0.

A bounded ordinary control satisfies the premises: on (-1,1)^2 take g=(x+1/5,y-1/7,x+y-1/2), n=(-1,-1,1). Sources (0,-1/2) and (-1/2,1/2) realize singleton cells 100 and 010. All three affine normals are nonparallel and all gates cross zero. For this control all fixed metrics of form (F) reduce to the scalar-sector construction above. No general conclusion about other bounded sources follows.

## A bounded ordered source escapes the redundant family

Let (x,y) lie in (-2,2) times (-1/10,1/10), with

```text
A = [1  0; 1  1; 1 -1; 1  2],
b = (3/2, 1/2, -1/2, -3/2),
M = [10 1 1  0;
      1 10 0 3;
      1 0 10 -1;
      0 3 -1 10].
```

Strict diagonal dominance makes M positive definite. All four normals are nonparallel, all gates cross zero, and g1-g2=1-y, g2-g3=1+2y, g3-g4=1-3y are strictly positive throughout the source box. Only the five prefix masks are reachable, and each occurs by taking y=0 and varying x. This is a direct structural proof, not runtime phase search.

The three nontrivial pulled-back matrices A^T S_D A are respectively

```text
D={1}:       2 a1 a1^T,
D={1,2}:     [4 4; 4 6],
D={1,2,3}:   2 a4 a4^T.
```

All are positive semidefinite; empty and full masks give zero. The local-to-global theorem therefore proves (F). The unreachable singleton mask {2} would give [[4,5],[5,6]], whose determinant is -1. The valid relation genuinely uses the bounded common source.

Row 2 of MA is (14,16), which is not proportional to row 2 of A. Hence M is not T-L for ANY diagonal T with LA=0. This escapes the redundant family without contradicting the corank-one result or D133's full-dimensional result.

## Separation from an explicitly matched ordered LP comparison

Use the same two source endpoints in both arms:

```text
u=(0,0),                  v=(1/10,-2/25),
g(u)=(3/2,1/2,-1/2,-3/2),
g(v)=(8/5,13/25,-8/25,-39/25),
d=(1/10,1/50,9/50,-3/50).
```

The tight closure bounds are l=(-1/2,-8/5,-13/5,-37/10) and h=(7/2,13/5,8/5,7/10). Compare TWO copies of the four original relaxed-bit ReLU gates, with the D018 chain replacing their lower rows, and additionally grant the comparator all four scalar incremental sector inequalities. Original upper rows in each copy are Q_i<=h_i beta_i and Q_i<=g_i-l_i(1-beta_i). D018 lower rows are Q4>=0, Q1>=g1 and 0<=Qi-Q_(i+1)<=gi-g_(i+1).

The following fictitious relaxed outputs are feasible in that comparator:

```text
Q(u)=(33/20,9/10,1/2,1/5),
Q(v)=(33/20,23/25,1/2,1/5),
beta(u)=beta(v)=(1/2,1/2,1/2,1/2).
```

At u their output gaps are (3/4,2/5,3/10), below preactivation gaps (1,1,1). At v their output gaps are (73/100,21/50,3/10), below (27/25,21/25,31/25). Both absolute anchors hold.

The first upper rows at either endpoint give (7/4,13/10,4/5,7/20). The second upper rows give that same vector at u and (37/20,33/25,49/50,29/100) at v. The proposed Q values satisfy every component of both upper bounds. This also verifies the continuous phase variables explicitly; they are not secretly absent from the comparator.

Their increment e=(0,1/50,0,0) satisfies all scalar sectors with equality, yet

```text
e^T M(d-e) = (1/50)(1/10-9/50) = -1/625.
```

Thus the metric relation separates this same-source two-copy D018 relaxation PLUS scalar sectors. This is stronger evidence than comparing only against unconstrained scalar sectors. It remains a relation-level paper separation, not a property solved on a neural benchmark.

Both Q endpoints are fictitious; true outputs are (3/2,1/2,0,0) and (8/5,13/25,0,0). Full exact HZ already rejects the fictitious points. No advantage over the complete source joint hull in [D018's comparison](../d018_order_relations_20260930/COMMON_SOURCE_COMPARISON.md), a fixed-concrete-anchor query, or an unrestricted set of stronger cuts is established. Relaxed bits occur only in the comparison query; original binary carrier semantics are not changed. This is neither a concrete adversarial witness nor an invalid-ADV finding.

## Why this does not yet deliver the requested domain

Appending (F) to old HZ is a valid semantic consequence, not by itself a new domain definition. It needs two authenticated source/output copies. Dense M costs m*m coefficients, and evaluating a quadratic constraint on the GPU does not give an optimization procedure for an output bound. Sending it to a new QP/SDP solver, lifting all products for free, or enumerating phase regions would not meet the present research boundary.

An affine or convolutional consumer can use the same source relation, but after mixed consumers and another ReLU the preactivation is generally no longer affine in the ORIGINAL source. The theorem supplies no new complete multilayer transformer in that case. Arbitrary nonconvex sources cannot inherit its necessity statement. No smooth/softmax/attention theorem is supplied here. There is no demonstrated source-discovery cost, stable GPU representation, witness implementation, full physical storage result, speed result or retained-solve result.

This is why we stop expanding this side investigation now. A subsequent candidate must explain, together, its nonconvex element and concretization, sound compositional transforms, GPU-suitable forward/query mechanism with complete cost, and a meaningful same-backend capability discriminator. Those are the existing goal's deliverables, not new permission gates or a reduction of the objective. The mathematical results above may support such a candidate; they do not replace it.

## Relation to prior work and evidence

[D018](../d018_order_relations_20260930/D018_THEORY.md) already provides order-certified lower-row replacement, mixed-sign forward consequences and full symbolic costs. Its order machinery is not new here. [D133](../d133_query_alignment_boundary_20261003/METRIC_BOUNDARY.md) establishes the full-dimensional fixed-metric boundary. Earlier circuit, shared-secant and source-decoder work remains archived and is not relabeled as a new domain.

Incremental quadratic neural constraints are an existing research area. The primary publication page for [Hashemi, Ruths and Fazlyab, 2021](https://proceedings.mlr.press/v144/hashemi21a.html) describes LMI certificates of neural input/output quadratic relations and SDP applications. Only that page and abstract were reviewed for this checkpoint; it supports this prior-art context, not novelty of our specific formulas or a complete literature comparison. All propositions and rational controls above are paper derivations independently reviewed within this project, not externally established novelty or machine-checked proofs.

The latest helper audit confirmed that removing explicit attacks and sampling is not the same as demonstrating a definition-only gain. Current scores remain as recorded in the [independent candidate review](../../claude_review_20261003/README.md). Another session's N109 replay was observed running during that audit; it is not a D134 job and its continuation does not qualify the withdrawn candidate. We neither restarted nor stopped it in this checkpoint, and do not use its outputs here.

No test population, mathematical gate, numerical tolerance, original input semantics or baseline retention requirement has been weakened. In particular the inherited 3953-test/207-file population, ordinary terminal-query boundary, resource gates and required full same-path replays remain applicable to any future implementation. D134 is supporting paper progress; the complete Neural-HZ goal is active and unfinished.
