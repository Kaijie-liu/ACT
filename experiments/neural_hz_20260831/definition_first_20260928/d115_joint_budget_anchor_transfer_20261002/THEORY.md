# Joint amplitude budgets across a change of phase anchor

This record gives a positive, bounded-size relation for the definition-first Neural-HZ investigation. Several nonnegative neural observations can share a certified total-amplitude budget. Carrying that budget jointly from an old original phase to a new original phase is strictly stronger than carrying every observation and its total through separate two-phase box interfaces. The precise local hull, a separating control and complete local lowering counts are given below.

This is a specialization of known disjunctive perspective and RLT mechanisms, not a claim of a newly discovered convexification principle or a finished Neural-HZ domain. The contribution to the project is identifying the missing joint capacity, proving its finite transfer interface, and distinguishing it from the previously derived box interface. No implementation, numerical experiment, trained network or GPU ran for this record.

## Nonconvex carrier and shared observations

Retain the full original HZ relation H(theta): bounded continuous factors, every original signed binary factor, EQ/LE predicates, original gate guards, shared latent/frame identity and concrete input decoder. Let alpha and beta be authenticated active-bit views of two distinct original gates. They remain binary in the domain; writing them in [0,1] below describes only a query relaxation for comparison. Both original labels at a zero preactivation remain legal.

A budget bundle consists of nonnegative observations v_1,...,v_k of the SAME theta and a certified finite B>0 such that sum(v_i)<=B. An observation may be a scaled original ReLU output or another certified nonnegative readout. It is not a fresh independent source. Add named canonical observations

```text
t_i = alpha*v_i
u_i = beta*v_i
delta = alpha*beta
s_i = alpha*beta*v_i.
```

The candidate relation fragment is (H, named observations, certified budget bundles, shared anchor interfaces). Its concrete interpretation uses one theta for all observations and retains H. Empty bundles embed H; a valid bundle and its canonical extension preserve the exact projection to original factors and outputs. Semantic inclusion gives a preorder, and semantic equivalence can be quotiented to obtain a partial order. No computable best abstraction or full lattice is claimed.

The additional relation language is useful only if it supports a valuable precision/cost tradeoff. The container alone is a reduced product over H, not greater set expressiveness than HZ or a new name for the whole neural graph. Binary factors are neither removed nor replaced by a CZ/zonotope. B=0 is outside this normalization contract; it does not authorize fixing original phase labels or a special solver rescue.

## Complete local hull over a simplex source

For this theorem only, take the local set

```text
K = {v>=0, sum(v)<=B, alpha,beta in {0,1},
     t=alpha*v, u=beta*v, delta=alpha*beta}.
```

Write V=sum(v), T=sum(t), U=sum(u), and

```text
omega11=delta, omega10=alpha-delta,
omega01=beta-delta, omega00=1-alpha-beta+delta.
```

Assume 0<=alpha,beta<=1, omega>=0, and the two complete single-phase simplex interfaces:

```text
0<=t_i<=v_i, 0<=u_i<=v_i
T<=B*alpha, V-T<=B*(1-alpha)
U<=B*beta,  V-U<=B*(1-beta).
```

Then conv(K) is described exactly by these conditions and four joint budgets, where a_plus=max(0,a):

```text
sum_i (t_i+u_i-v_i)_plus <= B*omega11
sum_i (v_i-t_i-u_i)_plus <= B*omega00
sum_i (t_i-u_i)_plus     <= B*omega10
sum_i (u_i-t_i)_plus     <= B*omega01.                 (J)
```

These are four convex piecewise-linear inequalities, NOT four ordinary LP rows. Their lowering cost is accounted for below. The theorem describes a local simplex source with two bits. Intersecting the resulting factor with arbitrary actual source predicates and guards remains sound but need not give the complete neural graph hull.

## Constructive proof and zero masses

Introduce s_i and the four cell vectors

```text
w11=s, w10=t-s, w01=u-s, w00=v-t-u+s.
```

Require each vector to be nonnegative and sum(w_ab)<=B*omega_ab. Every point of K has the canonical extension s=alpha*beta*v, so necessity is immediate. Conversely, a feasible fractional point has a common mixture: for each positive omega_ab, divide the entire vector w_ab by omega_ab to obtain one simplex source vector and attach the original bit pair (a,b). Its four-cell mixture recovers v,t,u,delta simultaneously. A zero-mass cell has zero total nonnegative amplitude and thus is the zero vector; choose any simplex source there. No phase case is discarded.

Eliminating s gives (J). Coordinate feasibility requires

```text
l_i=max(0,t_i+u_i-v_i) <= s_i <= h_i=min(t_i,u_i).
```

The four total capacities require the sum of s to belong to

```text
[L,H], where
L=max(T-B*omega10, U-B*omega01)
H=min(B*omega11, B*omega00-V+T+U).
```

Coordinate bounds give l_i<=h_i. The four single-phase total bounds give L<=H: expand each of the two lower candidates against each of the two upper candidates. The first two inequalities in (J) are exactly sum(l)<=H, because sum(v-t-u)_plus=V-T-U+sum(l). The last two are exactly L<=sum(h), because sum(t-u)_plus=T-sum(h), and analogously for u-t. Hence [sum(l),sum(h)] intersects [L,H].

Choose S=max(sum(l),L). Starting from s=l, distribute S-sum(l) across the capacities h_i-l_i in canonical observation order. This constructs a joint witness in O(k) arithmetic operations; it is not a max-flow oracle or an input/phase search. The proof establishes a local convex-mixture witness, NOT an original-network adversarial input. Concrete ADV validation still uses the retained original decoder and network.

## Strict comparison with the complete box interface and total observation

Set B=1 and

```text
alpha=beta=1/2, delta=1/4
v=(1/4,1/4,3/10)
t=u=(1/5,1/5,1/20).
```

The point satisfies the complete single-phase simplex hull for both anchors. For alpha, use equal weights on the positive source vectors 2t=(2/5,2/5,1/10) and 2(v-t)=(1/10,1/10,1/2); their totals are 9/10 and 7/10, strictly below one. The same construction works for beta.

It also satisfies every coordinate's complete two-phase box hull from [D035](../d035_cross_phase_source_20260930/THEORY.md), with the SAME delta. Explicit coordinate cell amounts in order 11,10,01,00 are

```text
coordinate 1: (7,1,1,1)/40
coordinate 2: (7,1,1,1)/40
coordinate 3: (1,1,1,9)/40.
```

Every cell amount is strictly between zero and its mass 1/4. The aggregate observation V=4/5, with T=U=9/20 and valid range [0,1], also has a complete D035 witness: its four amounts can be (9,9,9,5)/40. This aggregate witness is feasible separately, but cannot be the sum of feasible coordinate witnesses with a common simplex capacity.

Indeed, the first joint budget requires

```text
(2/5-1/4) + (2/5-1/4) + max(0,1/10-3/10)
= 3/10 <= 1/4,
```

which fails by 1/20. Thus the new factor is strictly stronger than the specified coordinate-plus-total reference, even with two ideal single-phase simplex interfaces. All four masses are positive and the single-phase source witnesses have positive slack. The separation is not a zero-mass or numerical exception.

This point is an auxiliary local relaxation point, not a concrete network state, invalid ADV, trained-model certificate, or proof that the same gap survives all original neural constraints. In fact it cannot extend to the companion neural example with faithful old observations: there alpha is q1's own phase and v1 is its scaled output, so t1=v1 is an original identity, whereas this point has 1/5!=1/4. The companion example establishes budget and anchor-change premises, not neural-query separation.

## Exact strong references and lowering costs

The compact fair reference is the s-variable formulation from the proof itself. It is a known four-cell simplex perspective/RLT extension, and the candidate is exactly equivalent to it. No precision or speed superiority over that reference is claimed.

Another exact description uses D035 for EVERY subset aggregate v_J=sum_{i in J}v_i, with range [0,B] and its t_J,u_J. Its four nontrivial triangle rows, for all subsets J, are exactly the linear expansions of (J). Thus the gain above does not survive a reference already containing all subset aggregates. A literal no-auxiliary expansion has up to 4*2^k rows before redundant rows are removed. It is not a proposed dynamic cut generator.

With v,t,u,alpha,beta,delta already available as physical coordinates, append k coordinates s and:

```text
s_i>=0, t_i-s_i>=0, u_i-s_i>=0, v_i-t_i-u_i+s_i>=0
sum(s)<=B*delta
sum(t-s)<=B*(alpha-delta)
sum(u-s)<=B*(beta-delta)
sum(v-t-u+s)<=B*(1-alpha-beta+delta).
```

The coordinate rows use 4k LE and 9k nnz. The four total rows use 4 LE and 9k+8 nnz. Total increment is k continuous coordinates, 4k+4 LE, 18k+8 nnz. At k=3 this is 3 coordinates, 16 LE, 62 nnz.

For a fresh constructor starting with old t but no u or delta, creating u,s,delta and the four delta McCormick rows costs 2k+1 coordinates, 4k+8 LE, 18k+16 nnz. The factor already implies the new single-phase u interface, so those rows need not be created twice. At k=3 this is 7 coordinates, 20 LE, 70 nnz. This is NOT permission to delete previously installed or original rows; appending to an existing full reference uses the preceding incremental account.

None of these counts include obtaining old t, readout definitions, expansion into latent coordinates, original HZ rows, signed-bit normalization, bounded-slack conversion, finite bounds, exact arithmetic, outward rounding, provenance and reconstruction evidence, host/device coexistence or terminal solve work. k is the actual retained observation count, not automatically three or independent of network width. Repeated interfaces accumulate unless a separately proved projection removes their consumers safely.

## Forward use and exactness limits

The new u observations provide a genuine beta-anchored interface without pretending that alpha and beta describe nested regions. For a complete affine consumer E=c+sum_i a_i*v_i+r, its beta-active observation is

```text
beta*E = c*beta + sum_i a_i*u_i + beta*r.
```

The last term cannot be omitted or called independent noise: r is the actual remaining same-frame readout and needs a canonical observation or a sound paid bound. Affine/Conv/Add use the same actual coefficients and observation identities; Concat preserves them. For a subsequent original ReLU, a certified beta-conditional preactivation interval supports the same perspective secant rule as [D091](../d091_joint_realization_relay_20261001/THEORY.md). Its original new bit and graph rows remain, and output observations used farther downstream must be represented and paid for.

For any finite sequence of these operations, canonical observations defined from the same theta give a simultaneous extension. This proves sound composition, not complete preservation of all fractional auxiliary states, exact conditional distributions, a global ideal hull or a uniformly small interface at arbitrary depth. All original continuous and binary factors, predicates and decoder remain. Other omitted source relations can still cause relaxation loss.

The constructor consists of nonnegative cell differences and fixed reductions. It has a plausible batched GPU form but no GPU kernel or reliable device rounding has been implemented here. A device implementation must not quietly evaluate nonlinear positive-part constraints and claim to have installed four LP rows. Ordinary terminal LP/MILP and independent concrete validation retain their original roles.

## Prior work and research status

[Balas, Disjunctive Programming, introduction page 284](https://lara.epfl.ch/w/_media/projects/disjunctive_programming.pdf) describes the scaled-copy extended formulation for the convex hull of a union of polyhedra. The four simplex cells above are a direct specialization. No branching or solver-state cut procedure from that literature is adopted.

[Sharp Hybrid Zonotopes, Theorem 7 and equation 18](https://arxiv.org/html/2503.17483v2#S4) supplies HZ-compatible Boolean and Boolean-continuous product lifting. The products alpha*beta*v here belong to that established family. Retaining the original bits preserves nonconvex semantics; we do not substitute its CZ relaxation for H.

Independent paper reviewers checked the local hull, strict control, ordinary neural premises and counts. The next substantive question is whether this exact joint-budget interface improves a real downstream neural query against the archived fixed/fresh-anchor baselines, with all original identities, residuals and costs present. The small same-information perspective reference must agree, not be defeated in precision. The proof is not machine checked, external novelty is unproved, and there is no candidate execution or formal gain.
