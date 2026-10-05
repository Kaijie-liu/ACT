# Joint forward support and source quotient limits

This work completes a forward support operation for the nonconvex D157 consumer-fiber candidate. It is not a new set definition, external helper or novelty claim. The useful change is operational: a uniform algebraic rule can automatically retain a common-source bound through a mixed consumer and skip into the next ReLU. A separate source-directed definition was investigated and rejected for this implementation because its advantage disappears even in the complete fixed-phase convex hull.

The controlling goal remains a powerful definition-first Neural-HZ with actual family-wide capability gains, not growth of infrastructure. D157 was progress: its frozen 4020-test component exposed both expressible joint information and a weak forward query. All D157 sources and evidence remain read-only.

## Protected domain and the automatic query

Use D157's original source coordinates, binary phase identities, EQ/LE predicates, shared amplitudes, native two-branch energy relations and source decoder. In a bank, C contains all declared B rows and a shared mass row, with exact mass-row reuse. Write

```text
r = Cq = C diag(beta) gbar + a,
Q = sum(q), s = C(g-gbar),
E = certified phase-affine parent energy,
m = number of original gates, u_i = max(U_i,0).
```

Neither the query nor its proof reconstructs separate hidden q coordinates. D157's installed coordinate caps and mass-dominance rows provide the following relations on the retained packet:

```text
L_Q = sum_i max(L_i,0) beta_i <= Q <= sum_i u_i beta_i,
min_i C_ji * Q <= r_j <= max_i C_ji * Q.
```

For the mass row, the last relation is the identity Q=Q. These are valid even for D157's finite outer polyhedron; they do not assume that every outer packet has one true vector q.

Let t be the same fixed mass scale used to emit the D157 opposite-direction energy row: sqrt_upper(Emax/m) if Emax>0, and 1 otherwise. That row is equivalent to

```text
Q <= s_mass/2 + E/(4t) + tm/4 + sum_i gbar_i beta_i.
```

If every u_i=0, the existing mass caps imply Q=0. Otherwise choose, by one fixed structural formula,

```text
rho = min_(u_i>0) abs(gbar_i)/u_i,
U_Q = [s_mass/2 + E/(4t) + tm/4
       + sum_i (gbar_i+rho*u_i) beta_i] / (1+rho).
```

Multiply the upper mass cap by rho and add the energy row. Since rho>=0, this proves Q<=U_Q. No optimization, phase enumeration, certificate search, terminal status or margin enters the choice. Mixed and nonuniform references are supported. Negative phase coefficients must remain signed until all shared terms are combined; replacing them prematurely by a nonnegative constant loses information.

For coefficients v of this bank's shared amplitudes, compute

```text
kappa = sum_(v_j>=0) v_j max_i C_ji
      + sum_(v_j<0)  v_j min_i C_ji,
kappa_plus=max(kappa,0), kappa_minus=min(kappa,0).

v^T a <= kappa_plus U_Q + kappa_minus L_Q
         - sum_i (C^T v)_i gbar_i beta_i.
```

The last subtraction is essential: the affine readout stores a, not r=Cq. The sign of kappa only determines which certified mass endpoint multiplies it, as in ordinary interval linear algebra. All coefficients are merged with the readout's existing source and phase coefficients before taking any source-box support.

Apply this substitution once per bank, newest to oldest. Each s contains only its parent amplitudes and E only parent phases, so substitution never reintroduces a removed newer bank. This proves termination after K banks and soundness over the full current abstract state, including non-concrete members. It is a local readout calculation on stored forward state, not a network backward pass. It does not call support recursively or run a solver.

After all amplitudes disappear, maximize each source coefficient over its physical source interval and return the resulting PhaseAffine certificate. Keep D157's existing norm certificate simultaneously. The scalar support is the minimum of the scalar upper bounds from both certificates, never a coefficientwise minimum of their phase-affine expressions. Both are always computed under the same rule, not selected after one verifier fails. This preserves useful norm cancellations such as C^T v=0 that the mass estimate alone can lose.

The subclass must persist through every extension. Bounds for newly created ReLU packets may become tighter, but the native relation, original factors, packet construction and finite-row formula remain D157's. Original zero labels are not fixed or removed. A numeric/work failure fails closed; the old norm certificate is not a rescue path after failure of the joint calculation.

## A whole-box result obtained automatically

Use D157's full 16-input box, A=I-11^T/8, g=Ax-(1/2)1 and B=(1,...,1,-1/4). The fixed carrier is C=[B;1^T], E=16, t=1, U_i=9/4. The uniform formula gives rho=2/9 and

```text
Q <= 72/11 - (9/22) sum(x),
J = Bq + (9/22) sum(x) <= 72/11,
J-33/5 <= -3/55.
```

The joint support query obtains this after actual coefficient cancellation, with no caller-supplied row weights. The next original ReLU therefore has a certified negative upper bound; its generated upper mass cap forces output zero throughout the finite outer domain too. The new original bit is retained and its incompatible active label rejected by guards, not deleted.

Changing the last reference to -3/4 leaves rho=2/9. The joint phase certificate additionally has coefficient -1/4 on that gate's original bit and the same scalar upper bound. Thus the rule is not tied to equal biases. A four-gate identity-source control with references (1/4,-1/4,1/4,-1/4) gives, for Q-(5/12)sum(x), constant 5/3 and phase coefficients (5/12,-1/12,5/12,-1/12). These explicit controls check the retention of signed phase information.

This advances D157 from a manually combined relation proof to an automatic forward query. It does not invent the energy inequality, dominate exact HZ, repair D157's three-gate mixed-phase loss, or establish actual network gains. The old false abstract member must remain legal under the new sound query. History-dependent energy growth is also not solved.

## Why a fixed source quotient cannot generally be exact

This is a scoped obstruction, not a ban on approximate domains. Fix an allowed phase family and assume that each considered phase permits a full-dimensional open set of d. For a specified beta, recovery of B diag(beta)d using only Cd is possible, even with an arbitrary nonlinear recovery function, exactly when

```text
ker(C) is contained in ker(B diag(beta)),
equivalently row(B diag(beta)) is contained in row(C).
```

Necessity follows by comparing d and d+epsilon*h inside the phase's open set for h in ker(C). Sufficiency is linear factorization B diag(beta)=K_beta C. If all singleton masks are available and every B column is nonzero, the condition requires every coordinate row and hence rank(C)=m. Ordered prefix masks have the same consequence by subtracting adjacent prefix matrices. No runtime phase enumeration is authorized by this paper proof.

Only when the entire C packet must close under all masks does the condition become invariance of row(C). For phase-signature groups I, the corresponding minimal invariant closure containing row(B) has dimension sum_I rank(B[:,I]); independent signatures commonly eliminate compression. Strictly stable all-on/off blocks are ordinary positive exceptions. If d=d0+A*xi, the correct condition is ker(CA) contained in ker(B diag(beta) A), restricted to reachable source-space interiors. Zero labels do not create fictitious full-dimensional phases. A method retaining the complete source or a phase-dependent source map is outside the quotient-only hypothesis, so this is not a lower bound on every possible Neural-HZ.

Observable-preserving invariant-subspace reduction is established mathematics. [CLUE, section 2 and Algorithms 1–2](https://arxiv.org/pdf/2004.11961) develops minimal invariant row-space closure for polynomial ODE reduction. The diagonal-mask statement above is an elementary specialization/inference, not a new foundation theorem or a result about neural verification supplied by that paper.

## Interior exactness of a convex phase slice

An elementary convexity fact sharpens the source-binding issue. Let D be a convex source set, d0 in its relative interior, and the complete true map on D be the affine function Md+b. Let F be a convex subset of D times the output space that contains this entire graph. If the fiber of F at d0 contains only Md0+b, then F equals the true graph on all of D.

To prove this, suppose (d,a) in F has error e=a-Md-b not zero. Choose a small 0<lambda<1 so z=(d0-lambda*d)/(1-lambda) belongs to D; relative interior permits this. The true point (z,Mz+b) is in F. Their convex combination has source d0 and output Md0+b+lambda*e, contradicting the singleton fiber. No closedness, boundedness or ambient full dimension is needed. Equivalently, a convex enlargement with any false amplitude has false amplitudes at every relative-interior source point.

This is a scoped geometric fact, not a novelty claim or a prohibition on Neural-HZ. Bits are fixed throughout the argument; keeping their integer labels does not evade it. A nonconvex fixed-phase fiber is outside its hypothesis. For deeper layers whose abstract source projection exceeds the concrete phase domain, apply the fact only to the intersection with a verified convex D on which the complete true map is affine. It does not say that useful whole-box property bounds are impossible, nor that all false abstract members must be eliminated before a candidate can gain capability.

## Rejected source-directed kernel definition

An investigated strengthening used P=C^T(CC^T)^dagger C, N=I-P and one shared theta in [0,1]:

```text
dhat=Pd+theta*Nd,
a=C diag(beta) dhat.
```

Keep the original source, phase guards and all D157 caps. The true image has theta=1; C*dhat=C*d and ||dhat||^2<=||d||^2<=E. Thus this is a sound subrelation of D157. At d=0 it enforces a=0 and repairs the three-gate fake point in native semantics. It is not selected for implementation.

The repair is lost even by its complete fixed-phase convex hull. In D157's three-gate example, C has rows (1,-1,1/2) and (1,1,1), gbar=(1/4,-1/4,1/4), k=(3,1,-4), beta=(1,0,1) and v=C diag(beta)k=(1,-1). Take two members:

```text
A: d=(3/50)k, theta=1,    a=(3/50)v,
   g=(43/100,-19/100,1/100).
B: d=-(3/40)k, theta=1/16, a=-(3/640)v,
   g=(1/40,-13/40,11/20).
```

Both sources are strictly interior and both original phases are strictly (+,-,+). Their proxy energies are 117/1250 and 117/204800, below 3; their branch caps and mass rows hold. Combining A with weight 5/9 and B with weight 4/9 yields d=0 and a=v/32, precisely the old false output Bq=13/32, Q=15/32. The next ReLU(Bq-25/64) again admits 1/64 at this source although its concrete output is zero.

Every globally valid linear inequality at those same integer bits accepts this convex combination. Continuous linear auxiliary variables do not change that fact, since their projection is convex. Therefore no ordinary LP lowering, even the complete fixed-phase convex hull, preserves this specific native improvement. The candidate would need paid continuous nonconvex semantics or a different relation; a point membership test is not a global solver. This is a regular source/phase example, not an extreme numeric exception, and does not rule out all source-bound domains.

## Complete cost and admission boundary

The implemented operation adds no domain variables, bits, banks or rows. It does add work: all bank C scans, C^T v, row extrema, certified scale and ratio arithmetic, parent-form substitution, source-box support, the old norm query and both certificate objects. A dense bookkeeping bound is O(sum_k p_k*m_k + K*(n_sources+n_phases+n_amplitudes)), in addition to inherited norm-query costs and arbitrary-precision arithmetic. Temporary forms, outward square roots, rational bit lengths and the shared Work counter are not free. No cache or new physical-memory claim is introduced.

Retain the inherited rational, population and resource caps. Neither this formula nor the rejected kernel relation supplies a GPU implementation. Sparse matrix operations and reductions are potential device work only; outward device arithmetic, transport, terminal costs, smooth/Transformer operators and full models remain unqualified.

The preregistered component is default-off and keeps all 4020 inherited tests/211 files before adding twelve targeted tests. There is no registered network worker, new solver, GPU initialization, shadow or family replay. Formal 1870/2413 and independent E0 61/400 remain unchanged; no theoretical or component success can update them.

Provenance: 2026-10-04 Australia/Sydney; branch redu-hz; HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac; tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5. Dependencies are D156/D157, their inherited algebra and the cited primary paper. All new files are isolated in this experiment tree. Mathematical proofs above were checked on paper by the main agent and independent agents; they are not machine proofs or measured model results.
