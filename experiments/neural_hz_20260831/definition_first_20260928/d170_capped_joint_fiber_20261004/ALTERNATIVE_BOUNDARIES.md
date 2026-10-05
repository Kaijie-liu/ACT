# Box projections and cross layer potential limits

These paper results record the alternative definitions examined alongside the common boxed-energy fiber. They close specific insufficient proposals, not the full Neural-HZ goal. No solver, model or numerical candidate was executed.

## Exact phase box projection

For fixed reliable intervals and original integer beta, define

```text
Z_A = C D_beta [ell,u],
Z_I = C(I-D_beta)[ell,u],
K_beta(s) = {C D_beta t : C t=s, ell<=t<=u}.
```

Then K_beta(s)=Z_A intersect (s-Z_I). Active and inactive coordinates are disjoint, so two image witnesses concatenate into one t. This is a genuine common box witness, unlike choosing independent full proxies for different consumers. Branch-specific intervals obey the same argument.

For C=c^T, let A_minus/A_plus be the sum over active indices of min/max(c_i ell_i,c_i u_i), and I_minus/I_plus the inactive sums. The exact interval is

```text
max(A_minus,s-I_plus) <= a <= min(A_plus,s-I_minus).
```

Four phase-affine rows suffice. Construction and pointwise proxy reconstruction are O(m): fill the required weighted mass from interval endpoints in a fixed order separately in each group, then concatenate. This reconstructs a proxy, not the original network trajectory.

For a full-row-rank two-dimensional C, every nonzero column direction supplies a perpendicular normal and its opposite. These fixed normals describe every masked box image, including its degenerate faces; the full column fan refines each subset's fan. With N<=m distinct directions, the two images need at most 4N rows, but generally O(m^2) phase coefficients. High-dimensional explicit facet templates can be combinatorial. This uses classical zonotope face geometry, not a proposal to replace nonconvex HZ by a convex domain. [Guibas, Nguyen and Zhang, introduction and section 3](https://graphics.stanford.edu/~anguyen/papers/zonotope.pdf) describe the generator/normal-arrangement correspondence.

For crossing ReLU bounds L<=0<=U, the old four-row LP has q in [0,U beta] and g-q in [L(1-beta),0]. Their C images satisfy exactly the new scaled box-image support rows and sum to Cg. Therefore every old LP point projects into that box-only interface. The new projection loses the old coordinatewise equation q+(g-q)=g in ker(C), rather than improving it. This comparison excludes new joint-energy restrictions; it does not refute the strict refinement in the companion note.

The D157 three-gate kernel proxy already lies within the branch boxes, so exact box geometry alone does not repair its source binding. Old D019 and D155 explain the same general projection issue; D102 supplies a related finite-fan construction for a budget interface. Reusing this geometry as a new foundational domain would be a relabeling. There is no implementation admission for the box-only proposal.

## Separate convex potentials do not compose automatically

Let R denote ReLU and define

```text
a=(1,-1/2), c=(1/3,1),
T_a(x)=x+a R(a^T x+1/4),
T_c(y)=y+c R(c^T y+1/4).
```

Each is the gradient of the convex potential ||x||^2/2+R(a^T x+1/4)^2/2, with the corresponding vector in the second layer. These are biased mixed-coordinate residual maps with their original bits retained. At x=0, the two preactivations are 1/4 and 5/24, so a strict open all-active neighborhood exists. In it,

```text
J_(T_c o T_a) = (I+cc^T)(I+aa^T)
              = [[37/18,-5/36],[-1/3,7/3]].
```

The Jacobian is not symmetric, so the composition is not a scalar-potential gradient. For two symmetric layer Jacobians H1,H2, their product is symmetric precisely when they commute. Thus even individually favorable layers do not give the needed generic cross-layer closure.

## A fixed source metric cannot repair a mixed residual block

On x in [-1,1]^2, take

```text
A=[[1,1/4],[-1/4,1]], b=(1/4,1/4),
V=[[1/2,-1],[1,1/2]], c=(1/4,-1/4),
q=R(Ax+b), r=R(Vq+c), z=q+r.
```

All four gates cross zero in the box. At x=0 all are strictly active, and

```text
J_z=(I+V)A=[[7/4,-5/8],[5/8,7/4]],
eigenvalues=7/4 +/- (5/8)i.
```

If Mz were a scalar gradient for a fixed symmetric positive definite M, then MJ_z would be symmetric. Hence M^(1/2) J_z M^(-1/2) would be symmetric and have only real eigenvalues, a contradiction. A convex proximal map in metric M has the same necessary weighted-symmetry condition, from its envelope potential, and is also excluded. This statement is about representing the actual output itself, not D131's different sufficient statistic plus a nonlinear decoder.

Jacobian symmetry as an integrability requirement is established prior mathematics, not a claimed new theorem; [Reehorst and Schniter, Theorem 1](https://phil-schniter.web.app/pdf/tci19_red.pdf) discusses the corresponding failure in neural denoiser regularization. The concrete fixed-metric block calculation here is a project-specific scope check. No denoising optimizer is adopted.

## Positive activation squares do not supply the missing convex potential

Use the same A,b,V,c and define

```text
x_star=(-1/17,-9/34), h=(-4/17,16/17),
x(t)=x_star+t h, |t|<1/32.
```

Here A h=(0,1), so

```text
q=(1/8,max(t,0)),
r=(5/16-max(t,0),0).
```

Only the second first-layer gate crosses; the other gates have strict nonzero margins. For arbitrary positive weights u_i,v_j and any C1 source regularizer F0, consider

```text
Phi(x)=F0(x)+(1/2)sum_i u_i q_i^2+(1/2)sum_j v_j r_j^2.
```

Its right derivative minus left derivative along this line at zero is -5v_1/16<0. A convex univariate restriction cannot have a downward derivative jump. No size of smooth source regularization or positive first-layer square weight fixes that jump. These are after-activation squares; this does not cover a different nonsmooth potential or the full propagation-gap relations in earlier work.

There is a limited positive scope: a continuous PWA map on a convex source domain whose cell Jacobians are all symmetric positive semidefinite is a convex-potential gradient. Pairwise commuting symmetric positive semidefinite layer Jacobians provide a sufficient route. Checking these global premises and querying the resulting image are not free, and ordinary untied mixed-weight networks do not inherit them by declaration.

Decision: do not pursue a generic shared scalar-potential or proximal-output wrapper. A future candidate must retain nonconservative mixed-source dependencies, or exhibit a different directly consumable statistic with all decoder and source-binding costs paid. These counterexamples are not a proof that all stronger nonconvex Neural-HZ domains are impossible.
