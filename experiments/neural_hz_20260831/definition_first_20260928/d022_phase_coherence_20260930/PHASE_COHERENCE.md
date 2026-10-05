# Shared phase consistency across activation relations

This investigation makes two concrete advances beyond the preceding literature review. First, the retained-phase difference factor can strictly improve a real scalar residual output, not merely exclude one optional fractional bit assignment. Second, projecting each edge's phase independently loses cross-edge information: one original phase is a compact common variable that connects the positive and negative amplitudes seen by all its incident relations. A three-neuron interior example proves that loss.

These are paper results about a candidate nonconvex relational block. They do not establish a new Neural-HZ domain, novelty over all formulations, GPU performance, or benchmark improvement. Ordinary HZ with exactly the same predicates reproduces the same semantics and LP. All original bits remain in the candidate; the projections below are mathematical analyses, not instructions to delete or relax bits in the represented domain.

## A phase coherent block definition

Let S be the retained HZ source relation, with its original continuous and binary factors, EQ/LE predicates, source identities and input decoder. Preactivations f are affine readouts of S. For each current gate retain its original bit beta_i and output r_i. Define the alias n_i=r_i-f_i; at concrete points it is ReLU(-f_i).

Assume certified crossing bounds -a_i<=f_i<=u_i with a_i,u_i>0. A fixed structural undirected graph selects pairs. For each oriented observation i to j use certified bounds

```text
-a_ij <= f_i-f_j <= u_ij,     a_ij,u_ij>0.
```

Reversing an edge exchanges a and u. Define, without introducing independent source copies,

```text
P_i = max(0, r_i/u_i,
             (r_i-r_j)/u_ij for all neighbors j),
N_i = max(0, n_i/a_i,
             (n_i-n_j)/a_ij for all neighbors j).
```

The block relation is

```text
S, original f readouts and bounds,
r_i>=0, n_i=r_i-f_i>=0,
P_i<=beta_i, N_i<=1-beta_i, beta_i in {0,1}.
```

The max notation means a finite family of linear inequalities. It is not a new max activation, a nonlinear terminal solver, a runtime branch, or an uncharged auxiliary variable. Each edge contributes exactly the four preceding [phase-difference rows](../d020_phase_difference_20260930/D020_PHASE_DIFFERENCE.md). The original scalar gate rows are still present through the terms r_i/u_i and n_i/a_i. The graph and bound method must be fixed structurally before any experiment; no selection by solver state or property margin is proposed.

### Exact integer semantics

If beta_i=0, the gate term forces r_i=0 and f_i<=0. If beta_i=1, it forces n_i=0 and r_i=f_i>=0. Thus every block point satisfies the original gate graph. Conversely, an exact gate point obeys every edge row by the phase-difference validity theorem, so no old point is lost. All S predicates remain, and the same input decoder applies.

At f_i=0, both r_i and n_i are zero. Neighbor outputs and negative amplitudes are nonnegative, so all incident differences in P_i,N_i are nonpositive. Both original bit choices remain legal independently. No phase ordering at zero is inferred.

This defines an exact source-sharing nonconvex block with a stronger available relaxation. It is still an extended predicate organization of old HZ, not by itself evidence of definition-level novelty.

## What one shared phase preserves

Temporarily consider the continuous relaxation for projection analysis only. At a fixed assignment to all nonphase coordinates, suppose beta_i occurs only in the displayed bounds. A compatible beta_i in [0,1] exists exactly when

```text
P_i + N_i <= 1.
```

Proof: its feasible interval is [P_i,1-N_i]. Nonnegative P_i,N_i make interval nonemptiness equivalent to the displayed inequality, including the bit box. This is ordinary linear projection, not a new convexification theorem.

Expanding the two maxima pairs every lower demand with every upper demand. In particular two possibly different neighbors j,k imply

```text
(r_i-r_j)/u_ij + (n_i-n_k)/a_ik <= 1.          (1)
```

The original scalar terms produce additional pairings with r_i/u_i and n_i/a_i. Edgewise projection keeps pairings within each edge, but can miss j!=k. This identifies a concrete reason to retain a single original phase identity across relations.

For degree k_i, the phase form has 2(k_i+1) amplitude inequalities plus its box, while literal expansion has at most (k_i+1)^2 comparisons after dropping the zero terms, which the nonnegative scalar gate terms already dominate. This is an upper count for one direct expansion, not a facet lower bound; many rows may be redundant. With bounded graph degree both totals remain linear in network size. No unconditional asymptotic speedup follows.

If other retained predicates couple beta_i to another bit or source variable, the interval criterion characterizes only this submodule. It cannot independently eliminate that bit from the full LP. The candidate never deletes it in either case.

## A three neuron interior witness for cross edge consistency

Use the ordinary shared input box x,y in [-1,1] and

```text
f0=x,     f1=x-y/2,     f2=x+y/2.
```

The scalar bounds are [-1,1], [-3/2,3/2], [-3/2,3/2]. Select only edges (0,1),(0,2); both differences have bounds [-1/2,1/2]. Consider

```text
(x,y)=(1/10,3/5),
(f0,f1,f2)=(1/10,-1/5,2/5),
(r0,r1,r2)=(2/5,1/8,17/40),
(n0,n1,n2)=(3/10,13/40,1/40).
```

Edge (0,1) has delta=3/10, d=11/40, e=-1/40 and requires beta0>=11/20. Edge (0,2) has delta=-3/10, d=-1/40, e=11/40 and requires beta0<=9/20. Shared-phase composition therefore rejects this tuple: (1) has left side 11/10.

Each edge alone admits the tuple, even when accompanied by full retained-source single-gate hulls:

- Set beta1=beta2=1/2. The two source points (11/20,3/5) and (-7/20,3/5), mixed equally, produce r1=1/8 and r2=17/40 in their respective exact gate graphs. Their preactivations are 1/4,-13/20 for gate 1 and 17/20,-1/20 for gate 2.
- For edge (0,1), use beta0=11/20. Mix center-gate inputs x=8/11 and x=-2/3 with those phase weights, keeping y=3/5. They produce mean x=1/10 and r0=2/5.
- For edge (0,2), use beta0=9/20. Mix x=8/9 and x=-6/11 instead, again keeping y=3/5. The means are unchanged.

All constituent source points are strictly inside the box and their relevant gate preactivations are nonzero. At beta1=beta2=1/2 the other endpoint inequalities also pass. The phase-free difference intervals are [-1/10,2/5] for edge (0,1), and [-2/5,1/10] for edge (0,2); the displayed d values pass both.

The individual full-source gate hulls also admit a common displayed bit vector with all three bits equal to 1/2: for the center, mix (x,y)=(4/5,3/5) and (-3/5,3/5) equally. This gives the same x,y,r0 means. The edgewise projections allow their separate center-bit witnesses, whereas the shared edge formulation admits no single center bit at these source/output coordinates.

Thus the intersection of independently phase-projected edge formulations is strictly weaker than the projection of the shared-phase formulation. It is not a claim against formulations that already keep the same global beta0. Nor is the shared formulation asserted to be the complete joint source hull.

## A strict residual output bound from one pair

Take x,y in [-1,1], f=x+y/4, h=x-y/4, r=ReLU(f), p=ReLU(h). Their bounds are [-5/4,5/4] and delta=f-h=y/2 has bounds [-1/2,1/2]. Write d=r-p and e=d-delta.

The original row r<=(5/4)beta and the phase-difference row e<=(1/2)(1-beta) imply, without changing any represented bit,

```text
(7/5)r - p - delta <= 1/2.                    (2)
```

This is obtained by adding r/(5/4)<=beta and e/(1/2)<=1-beta. It is valid for the concrete graph and for the strengthened LP.

The weaker comparison is strong: full retained-input ideal hulls for each individual gate, combined with the exact phase-free difference hull. It nevertheless admits

```text
(x,y)=(1/2,0), r=3/4, p=13/25,
beta=3/4, eta=4/5, delta=0, d=23/100.
```

For the f gate, mix (x,y)=(19/20,1/5), active with f=1, with (-17/20,-3/5), inactive with f=-1, using weights 3/4 and 1/4. For the h gate, mix (13/20,0), active, and (-1/10,0), inactive, using weights 4/5 and 1/5. Each pair has the required source mean and output. All constituent points are interior and strictly signed. At delta=0 the phase-free hull permits d in [-1/4,1/4], so d=23/100 passes.

But the left side of (2) is 53/100. No alternative bits rescue these same source/output coordinates: the two relevant rows require beta>=3/5 and beta<=27/50.

Now define an actual mixed affine residual output z=(7/5)r-p-y/2. Its true maximum is 1/2: (2) proves the upper bound, and input (1,1) attains it. The property z<=51/100 therefore has a positive safety margin of 1/100, whereas the weaker relaxation permits z=53/100. This is a strict terminal scalar-output separation, not just a labeled tuple separation. Projecting only to (r,p), without the source residual, would not establish it.

The example is a paper control, not an executed CERT, a real-network prevalence result or a new formal solve. It does not prove novelty of (2) against every multi-neuron formulation.

## Complete pair projection for comparison

For an isolated pair with f in [-a_f,u_f], h in [-a_h,u_h], delta in [-a,u], all capacities positive, set d=r-p and e=d-delta. Projecting the relaxed bits beta,eta in [0,1] out of ordinary gate rows plus the four phase-difference rows gives:

- the original scalar bounds, epigraph rows and single-gate triangle projections;
- the two phase-free difference hull rows;
- four additional cross rows:

```text
a*r + u_f*e <= a*u_f,
a_f*d + u*(r-f) <= a_f*u,
u*p - u_h*e <= u*u_h,
-a_h*d + a*(p-h) <= a*a_h.
```

Proof: beta lies between max(r/u_f,d/u) and min(1-(r-f)/a_f,1-e/a); eta lies between max(p/u_h,-d/a) and min(1-(p-h)/a_h,1+e/u). The epigraph rows make the explicit 0/1 endpoints redundant in these intervals. Pairing each lower and upper bound yields exactly the listed rows, and conversely proves both intervals nonempty. Scalar bounds and delta bounds remain present.

This exact projection statement assumes the bits have no other occurrences. With full-source ideal single-gate rows, other bit relations, or additional incident edges, the four displayed rows remain valid but need not describe the full projection. The three-neuron example shows why independent edge projections cannot generally substitute for shared bits.

## Cost and novelty disposition

With retained f,r columns, n is an alias. Each edge adds four rows and ordinarily 16 coefficient nnz; original gates, source equalities and bounds are additional. Substituting a source expression for delta instead gives 12+2*s_delta nnz when coordinates are distinct. Necessary difference-bound rows, edge metadata, multiplier buffers and terminal conversion remain chargeable. Computing P,N uses gathers and per-node maxima but is only submodule evaluation, not a complete verifier or permission for a new solver path.

The companion [composition analysis](COMPOSITION_AND_CONTROLS.md) supplies the exact affine interface condition, CNN boundary handling, GPU operator and a known source-conditioned two-layer control. No kernel was executed.

The capacity notation itself is an organization of known linear inequalities. [DiffPoly](https://ggndpsngh.github.io/files/raven.pdf) is prior art for difference propagation; [Sharp HZ](https://arxiv.org/pdf/2503.17483), especially its distinction between a set and its relaxation, is prior art for strengthening retained-bit representations. Standard linear projection explains the interval elimination theorem. Neither the star notation nor an exact rewritten graph alone is a new abstract domain.

What is now established is a specific semantic role for shared original phases, and a strict output control worth testing in a larger definition. What remains missing is a compositional rule or complete-cost advantage beyond ordinary HZ supplied the same facts, and evidence on real structures. This candidate component is not promoted or implemented as a default path.

## Evidence status

Date 2026-09-30, branch redu-hz, commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac, pre-existing dirty worktree preserved. Configuration: paper derivations, primary-source comparison and independent read-only audits; no executable mathematics tests, model runs, numerical solvers, GPU work or replay. The two strict examples are mathematical witnesses, not measured experimental outcomes. Formal 1870/2413 and independent external 61/400 remain unchanged.
