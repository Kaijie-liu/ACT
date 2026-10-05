# A shared causal fiber candidate for Neural-HZ

This is a complete mathematical PWA carrier and query proposal, not a completed Neural-HZ implementation or an established novel domain. It addresses the shared-consumer failure left by D124 and D125: all current affine consumers use the SAME amplitude vector, rather than independently chosen scalar envelopes. A sparse causal constraint subsystem admits a closed-form support computation, and a uniform forward ReLU rule creates such constraints. An ordinary global residual control distinguishes the resulting query from the specified old triangle LP. No claim of dominance over old HZ supplied with the same relations is made.

The main research question is whether this common-fiber organization can deliver a practical, genuinely useful nonconvex Neural-HZ on the full live frontier. It is not permission to substitute solver portfolios, generic backward rescue, or a larger test framework for that work. D134's metric side branch remains closed as a standalone implementation direction.

Date 2026-10-03 Australia/Sydney; branch redu-hz; HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac; tracked binary diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5. This checkpoint contains paper mathematics and read-only source review only. No candidate, tests, model, solver, GPU job or replay was executed. The [goal authority](../../GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md) is unchanged.

## The current source problem

[D125's actual-consumer record](../d125_signed_phase_component_20261002/NEXT_REAL_STRUCTURE.md) identifies an identity shortcut in CIFAR large, projected shortcuts in medium and Tiny, and further downstream skips in the two inspected CIFAR graphs. Its scalar component already exists; another scalar adapter is not the missing deliverable. A local consumer projection cannot discard a parent still used on those paths.

[D126](../d126_joint_phase_closure_20261002/THEORY.md) also proves that separate output bounds can admit incompatible outputs despite retained source and phase identities. The new carrier therefore keeps one joint amplitude fiber visible to every consumer. It does NOT promise a reduction in amplitude dimension, which the live identity skip already obstructs for simple linear factorization.

## Element and concretization

An element E consists of:

- An owned bounded source relation S on continuous source coordinates z and ALL original signed binary identities. Use beta=(signed_bit+1)/2 only as a notation-preserving 0/1 view. S retains its EQ/LE predicates, frame and original-input decoder.
- One owned amplitude vector e of dimension n and a constant nonnegative strictly lower-triangular matrix L, under a fixed structural order.
- An affine nonnegative source-and-phase vector c(z,beta), certified nonnegative on S, defining the fiber F(z,beta).
- Additional linear EQ/LE predicates P involving source, original phases and e. Exact neural gates and constraints not belonging to F remain here.
- All visible tensor readouts y=d+Az+B beta+C e, with shared coordinate identities rather than independent per-output copies.

The concretization, including labeled latent state when comparing elements, is

```text
Gamma(E) = { (z,beta,e,y) : (z,beta) in S,
              beta retains its integral type,
              0 <= e <= c(z,beta)+L e,
              P(z,beta,e),
              y=d+Az+B beta+C e }.
```

All expressions are interpreted simultaneously. c>=0 is required on the admitted source region, not guessed from a few samples. Strict triangularity ensures each fiber is nonempty and bounded. A valid source-wide upper vector cbar gives the finite forward bound t=cbar+L t. This is a triangular substitution, not a materialized inverse. Nonzero t_i permits e_i=t_i(1+xi_i)/2 with xi_i in [-1,1]; t_i=0 fixes that amplitude without deleting any original binary identity.

Ordinary HZ embeds exactly by taking n=0, placing its original mixed linear system in S and using its unchanged affine readout and decoder. The carrier remains nonconvex: an exact ReLU graph encoded below includes (-1,0) and (1,1), but excludes their midpoint. Bits are active constraints on amplitudes, not discarded or relaxed metadata. Its PWA denotations are still expressible by ordinary HZ. The proposed difference is structured factors and their transfer/query calculus, not a larger PWA set class. No best abstraction, complete lattice or external novelty is asserted.

After aligning source and original phase identities, define semantic precision by inclusion of the same visible-and-labeled projection of Gamma; quotienting equivalent presentations makes this a partial order. Auxiliary coordinates are existentially projected for that comparison, not silently deleted from an implementation. A concrete candidate is decoded with S's original input map and must pass the retained predicates, original input/property semantics and concrete network validation before any ADV credit. A purely relaxed fiber tuple is insufficient.

## Exact common-fiber support and its limited scope

Fix (z,beta), omit P temporarily, and let w be any signed coefficient vector. Define, in reverse structural order,

```text
k_i = max(0, w_i + sum_(j>i) L_ji k_j).
```

Then the exact support of F is

```text
max_(e in F(z,beta)) w^T e = sum_i k_i c_i(z,beta).
```

Proof: eliminate the last amplitude. Its interval is [0,c_n+sum_(j<n)L_nj e_j], a nonnegative interval for every feasible prefix. A positive current coefficient takes the upper endpoint; a nonpositive coefficient takes zero. Substitute that contribution into the preceding coefficients and repeat. This derives the recurrence and the objective constant. Reconstructing choices in forward order yields a single optimizer of F, not separate optimizers for different consumers.

Equivalently k>=0 and (I-L^T)k>=w supply an LP dual certificate. This fact must be named honestly; it is a closed-form certificate for the element's own specified fiber LP, not an external dual optimization, failed-LP repair, or license to run a backward network verifier. No query status, margin, label, instance or family selects a new path.

With P restored, the same result gives a sound affine envelope

```text
w^T e <= k^T c(z,beta),
```

but generally NOT exact support of Gamma(E). Source/phase optimization is also a separate remaining task; maximizing each beta independently may lose retained guard correlations. A fiber optimizer need not satisfy P or the original network, and is never automatically an ADV.

For a visible direction a, first form w=C^T a. The resulting source-and-phase envelope is a^T(d+Az+B beta)+k^T c. Lower bounds use -a. All directions share the same e and source; support values or witnesses from different directions are not concatenated into a fake joint input.

## Exact PWA operators and a uniform forward constructor

Affine and convolutional operators compose the visible readout with their linear map and bias. Add and Concat take a natural join over matching source, phase and amplitude identities. Shared ancestors are not copied into independent factors; unrelated branches may be joined in a common topological order. Intersecting a property appends its linear rows to P. No amplitude is retired while an unhandled consumer or predicate still uses it.

For a new preactivation g=h(z,beta)+sum_j a_j e_j, obtain valid bounds l<=g<=u on the CURRENT element and extend them to l<=0<=u. Keep the original gate identity alpha and append amplitude q. Set its causal row to

```text
0 <= q <= b + sum_j max(a_j,0) e_j,
b = max(0,U_h),     U_h >= h on the admitted source.
```

Put the remaining exact gate rows in P:

```text
q >= g,
q <= u alpha,
q <= g-l(1-alpha).
```

The original four-row graph is now present, with q>=0 supplied by F. Thus the transformer is EXACT for ReLU of the current abstract state. Its added causal upper row is redundant in the integer semantics because

```text
g <= b + sum_j max(a_j,0)e_j,
b + sum_j max(a_j,0)e_j >= 0
  => ReLU(g) <= b + sum_j max(a_j,0)e_j.
```

This implication holds for every current abstract state, not just true trajectories. New rows remain strictly lower triangular by creation order. At g=0 both original labels remain legal. For stable gates, the original identity still remains stored with the appropriate exact constraints; it is not deleted.

When source-affine h cannot be bounded cheaply with its full constraints, a certified source-box bound is sound but may be loose. No LP or optimization oracle is silently treated as free. For an imported signed continuous factor not represented as a nonnegative e, its coefficient remains in h and its actual source identity/bounds are retained.

This constructor proves compositional PWA coverage, including residuals and multiple consumers. It does not prove precision improvement at every layer. Keeping all exact rows makes it a structured exact-HZ presentation with additional query information; this alone is NOT the requested completed definition innovation. Any later lossy block replacement must state its new concretization and prove simultaneous inclusion for all consumers before dropping rows.

## Connecting the earlier phase-conditioned error certificates

D126 gives, on the same original trajectory and phases,

```text
0<=e<=u,
e<=a+B(1-beta)+M e,       M>=0, a>=0, B>=0.
```

Its M may contain cycles. Fix a structural order, write M=L+R with L strictly lower triangular and R containing every other coefficient, and set

```text
c(z,beta)=a+B(1-beta)+R u.
```

The original errors lie in the resulting causal fiber, by M e<=L e+R u. A previously proved nonnegative affine H_K(beta)>=e can replace u on removed edges, with its full generation, fill and storage costs paid. This is sound outer approximation, not exact removal of a cycle or amplitude reduction. It has no runtime phase partition.

If these rows are only consequences of the original exact block, they cannot automatically support repeated reabstraction of already enlarged states: either keep the original relation for that reduction step or prove the row on the current abstract state. Every original parent phase guard, on/off amplitude restriction, untouched predicate and raw consumer remains part of the contract. Renaming the old amplitudes e does not make them disappear.

For D126's two-error control, using H2 and beta1=beta2=1 gives

```text
e1<=9/16,
e2<=1/4+e1/2.
```

Support in direction (-1,3) is 33/32. Independent coordinate intervals for the SAME fiber give 51/32. This is a conditional fiber comparison only, not a global neural property. The full phase-dependent c is

```text
c1=9/16+(15/16)(1-beta1)+(3/8)(1-beta2),
c2=1/4+(3/4)(1-beta2).
```

Repeated readouts represented by the same row C remain identically equal, fixing the inconsistency permitted by independent scalar copies. This does not magically recover the true parent amplitudes or make the enlarged fiber exact.

## A global residual and next-ReLU precision control

Use the ordinary two-input, biased, nonparallel-source network

```text
x,y in [-1,1],
q1=ReLU(x+y/10+1/4),
g2=y/4-1/8+q1/2,
q2=ReLU(g2),
J=-q1+2q2,
s=ReLU(J-1/2).
```

Both first gates genuinely cross zero. The exact shared bounds are f1 in [-17/20,27/20], q1 in [0,27/20], and g2 in [-3/8,4/5]. The new causal cap is q2<=1/8+q1/2. It immediately yields the GLOBAL bound J<=1/4, attained for every source with y=1. Consequently the next preactivation is <=-1/4 and s=0 for every input. This conclusion uses a direct forward relation and addition, with no reverse-query algorithm, LP, attack, phase split or helper.

The same result follows from fiber support with c=(27/20,1/8), L_21=1/2 and w=(-1,2): k2=2, k1=0, support=1/4.

The old full independent four-row ReLU LP, with the SAME source and bounds, permits

```text
x=-1/2, y=1,
f1=-3/20, q1=0, beta1=0,
g2=1/8, q2=16/47, beta2=20/47.
```

For q2, both upper rows equal 16/47: (4/5)(20/47)=16/47 and 1/8+(3/8)(27/47)=16/47. The lower rows also hold. Thus J=32/47 and J-1/2=17/94>0. Using the common conservative next bounds [-37/20,11/10], the choice beta3=1,s=17/94 extends this to a valid old-LP point. Fractional phases appear only in this diagnostic relaxation; the actual domain keeps them integral. This is not an adversarial input.

This is a global next-ReLU discriminator, stronger in scope than the preceding fixed-phase fiber example. It is still only a mathematical control, not a benchmark gain. Old HZ with the same cap also proves the new bound, and its integer graph already has the true result. Input-aware or symbolic bounds may also recover it. External novelty and superiority over those strong comparators are unproved.

## GPU execution shape and all paid costs

For n amplitudes, s nonzeros in L and r signed directions, fiber supports require O(r(n+s)) arithmetic after forming the directions. Gates at the same dependency depth may be batched; dependency depth remains a synchronization cost. It is not valid to claim constant-depth GPU work for a long causal chain. Sparse transposed actions, maxima and dot products are the required kernel operations, not a new solver portfolio.

If c has p+q source/phase coefficients, forming all affine envelopes can cost O(r*nnz(c)) and may fill them. Computing C^T a, materializing visible convolutional readouts, source-bound construction, topological metadata, P, all phase bounds, lower/upper matrices, certificates, decoder, evidence and host/device coexistence all count. The dimension remains n; no reduction of live amplitude columns is established.

The exact forward constructor has the four ordinary gate inequalities plus ONE extra causal upper inequality when nonredundant in the stored representation. A sparse coefficient pattern may save query work, but it does not reduce terminal rows for free. Flattening Gamma into the ordinary LP/MILP backend retains S, P, nonnegative fiber rows and all original bits. Any lazy omission must preserve its proof and charge later lowering. Holding old and new matrices simultaneously incurs both costs.

A sound floating implementation may use any certified nonnegative khat satisfying (I-L^T)khat>=w, giving w^T e<=khat^T c. A descending outward evaluation can construct such a certificate. This is a proposed numeric contract, NOT a tested CUDA rounding implementation: reduction order, coefficient enclosures, underflow/overflow, sparse accumulation and source-plane rounding still need explicit bounds. Uncertified arithmetic fails closed. The inherited physical and numerical gates are unchanged.

## Prior art and the research decision

The authoritative [DeepPoly publication page](https://www.sri.inf.ethz.ch/publications/singh2019domain) already describes a domain combining polyhedra and intervals with neural-specific transformers. The authors' institution records [octatopes](https://researchconnect.stonybrook.edu/en/publications/the-octatope-abstract-domain-forverification-ofneural-networks/) as affine images of octagons with specialized optimization procedures. These are direct reminders that changing the latent constraint family and providing a special optimizer is not automatically novel. The publication pages were read; fetching both PDFs failed in this turn, so no detailed algorithm-level equivalence or literature-completeness claim is made.

Compared with ordinary HZ, the present exact carrier is a selected causal subsystem plus its remaining linear mixed predicates. The query recurrence is structured linear optimization, not a newly discovered optimization theorem. Compared with D124, its useful distinction is joint amplitude ownership and a compositional query rule. Compared with ImageStar or a convex zonotope, its original mixed phases and nonconvex gate relations remain. This does not establish a PLDI-level contribution.

Decision: keep this as ONE integrated PWA candidate hypothesis for the next targeted construction/comparison, not as a completed new domain and not as another open-ended theorem branch. The next substantive task is to bind the shared fiber to an entire existing live Conv/Add frontier, generate the same causal relations structurally for every member, and compare original HZ, original HZ plus those relations, and the candidate query under identical source information and cost accounting. Before numerical execution, make a new opt-in implementation/preregistration preserving the inherited population and gates; do not rerun or amend frozen D125/D130 experiments.

Smooth activations and Transformer behavior remain required parts of the full goal. This checkpoint gives no new native smooth/attention transformer; attaching the old independently studied attention primitive is not evidence that the combination works. A field for arbitrary function graphs alone would repeat the unmodified-graph problem. Do not present PWA closure as completion of that wider scope.

Formal baseline remains 1870/2413, independent E0 61/400, both gains zero. All 13 families, CIFAR/Tiny, GPU, smooth/Transformer and new-family goals remain active. No legacy code, evidence, results, production defaults, commits or pushes changed. The document skill only organized new isolated prose and kept proof, implementation and capability claims separate.
