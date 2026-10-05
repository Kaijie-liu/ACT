# Reference observed Neural HZ mathematical contract

This default-off component implements the definition proposed in the frozen D178 THEORY.md. It changes concretization, not merely storage: a ReLU packet retains observations of one common continuous witness under both the complete consumer map and its reference-phase mask. This document states the mathematical claims to test. It does not confer novelty, real-model, GPU, physical-memory or benchmark qualification.

The full objective remains a strong nonconvex Neural-HZ with reproducible family-wide gains, including GPU execution and eventual smooth/Transformer coverage. The present ReLU component is one necessary intermediate experiment, not a replacement objective. D179 found the relevant preterminal structure in the three frozen model graphs, but established shapes only, not the coefficients or soundness of an actual model binding.

## Domain and concretization

Keep the original continuous source frame, signed integer binary identities, EQ/LE predicates, shared latent identities, and original source decoder. Source constraints and all historical banks apply to the same global assignment. Different consumers of a bank share its coordinates and a single witness. Distinct banks have distinct witnesses; this is not a claim that all layers share one witness vector.

For a current parent preactivation g of width m, let b be the fixed nominal affine readout at the source midpoint, old reference bits and zero canonical residuals. It need not be an attainable parent point. Set d=g-b and tau_i=1 exactly when b_i>0. All actual phase bits beta remain variables, with both legal labels at zero preserved. No search or observed verification result selects tau.

B must contain every live linear consumer of the new ReLU output, including uses in predicates and reconstruction. C is B with an all-ones mass row appended unless that exact row is already present. Let p be its row count and H=C D_tau. A skip is a readout of the same owned parent and may contain existing sources, phases and residuals; an actual identity consumer of the new ReLU value cannot be omitted to make p smaller. The algebra API requires this complete-consumer premise but does not itself certify graph completeness.

Each bank stores p continuous canonical coordinates delta. With E certified to bound the squared norm of d on the entire native parent domain, its additional concretization relation is

```text
exists t:
    C t = C d
    H t = H d
    ||t||^2 <= E
    delta = C (D_beta - D_tau) t

packet = C D_beta b + H d + delta
```

Original phase guards constrain the actual g, not a surrogate. All inherited linear predicates remain, and the former branch caps, mass relations and fixed energy inequalities are retained with the explicit substitution a=Hd+delta. These strengthening predicates are part of the definition; they cannot be silently inferred from the bare ball relation.

With no banks, HZ embedding preserves continuous and signed binary factors, source predicates, output readouts and the input decoder. For one gate with C=1, Ct=Cd forces the actual nonconvex ReLU graph, not a convex set with decorative bits. General banks are still an overapproximation of the exact network graph.

## Sound extension and exact conditional slices

For any native parent member and any legal new phase assignment, choose t=d and delta=C(D_beta-D_tau)d. The packet is then C ReLU(g). Guards, caps and fixed inequalities hold by their stated bounds and square certificates. This proves extension over the whole abstract parent, including members not reachable in the original network. Requiring only concrete-network trajectories would be insufficient for recursive soundness.

The common witness satisfies t-d in ker(C) intersect ker(H). Consequently the packet is exact when beta=tau, beta=1-tau, all bits are active, or all bits are inactive. These are implications inside one domain element, not enumerated cases or split subproblems. The tests must also exhibit a legal nonreference member whose packet differs from the actual ReLU packet: these exact slices do not make the entire domain exact. The single-bank relation refines D157 only when comparing the same parent frontier, E, complete consumer map and corresponding strengthening constraints; it is not a full recursive-program dominance claim.

Canonical packet readout retains the shared source term C D_tau g. Affine maps and same-frame Add/Concat combine source and residual coefficients before bounding. From an exact input frontier, if all historical actual bank bits equal their reference bits, every delta is zero and the represented outputs equal the true affine network on that feasible phase slice. Starting from an already lossy frontier does not recover previously lost information.

The squared norm certificate E itself need not vanish on a reference slice. Only the residual flip contribution below vanishes there. Any remaining source variation must still be bounded. Exact cancellation such as q-x-1/2 can remove that variation; simply consuming q cannot.

## Recursive energy and fixed forward queries

For each fixed consumer matrix M and each bank, use the whole-parent maximum Emax to obtain

```text
||M delta||^2 <= Emax * sum_i ||M C_i||^2 * |beta_i-tau_i|.
```

This follows from the Frobenius bound on M C(D_beta-D_tau). For integer bits the flip is beta_i when tau_i=0 and 1-beta_i when tau_i=1. Negative phase coefficients are essential. No phase relaxation, cross-bank phase product, caller-supplied energy, LP status or ray is used.

For a later preactivation, its deviation splits into the shared source term, each old phase column relative to its reference, and each historical bank residual contribution. Apply the inherited source norm bound, the exact phase-column square and the displayed residual bound, combined by fixed equal-weight Cauchy over all nonzero terms. Replacing E by Emax and this Cauchy decomposition can lose precision. There is no general theorem that deeper D180 propagation dominates D158.

Every support query unconditionally computes three sound scalar certificates from the same state: a canonical residual norm bound, the former norm bound after reverse-order delta=a-Hd substitution, and the fixed D158 mass calculus with the same substitution. Earlier residuals introduced by Hd are processed in turn. Source box bounds are applied only after the required substitutions. Taking the minimum of complete scalar upper bounds is the single fixed query rule; no failure or terminal result activates another verifier. Unsupported arithmetic or exhausted work propagates failure.

This fixed algebra does not call an optimizer, choose phases, use a backward network pass, or repair a failed verification. An ordinary terminal solver and independent original-network witness validation remain later stages under the original project boundary, not implemented by this component.

## Finite inequalities and native membership

Keep six old directions per carrier row and add six reference-shifted directions. For each old scalar pair (l,r) on row c, the latter is exactly

```text
2 * [(l-r) delta_j + r (Cd)_j]
 <= E + sum_i c_i^2 * {
      beta_i * [l-(l-r)tau_i]^2
      + (1-beta_i) * [r-(l-r)tau_i]^2 }
```

There is no additional squared-shift term. The six pairs are (s,0), (-s,0), (0,s), (0,-s), (s,-s), (-s,s), where s=sqrt_upper(Emax/||c||^2), or 1 for a zero case. The exact coefficient oracle tests this formula, rather than only checking true points against potentially overlarge rows.

Add two range rows per carrier coordinate with R equal to the maximum of the outward square root of Emax and all absolute certified endpoint deviations from b:

```text
|delta_j| <= R * sum_i |C_ji| * |beta_i-tau_i|.
```

This finite system is only an outer approximation. Supplied-point native membership additionally checks one common witness for each bank. With K=[C;H;C D_beta] and z=[Cd;Hd;Hd+delta], require z in range(K) and minimum witness squared norm at most E. The existing exact rational Gram routine performs this check for one supplied original integer assignment. It is not an unknown-phase solver, a global optimizer, or an ADV validator. All inherited banks are checked; a feasible last bank cannot excuse an infeasible parent.

The tests must retain an explicit finite-feasible/native-infeasible point. The three support certificates and the next-layer E are sound over native members, not necessarily over every point in the larger finite polyhedron. Such finite-only points are not legal abstract parents. A decoded source from any abstract or finite candidate still requires original-network and property validation before it can count as ADV.

The supplied-point membership computation is exact when the registered rational calculation completes. Unsupported input, bit growth or exhausted resources also returns nonacceptance; that return does not prove mathematical infeasibility, domain emptiness or a CERT.

## Complete cost and current limits

Each new bank retains m original bits and p canonical continuous coordinates, and adds exactly 2m+18p+2*rows(B) finite rows. Count complete C, Cd, Hd, the source/phase expansion, all inherited predicates and banks, readouts, decoder and evidence. Logical coefficient counts are not physical resident-memory certification. The five-gate test has 50 fresh rows, compared with 20 rows for a four-row-per-gate exact graph, so it is not a compression win.

The native Gram has at most 3p rows. Straightforward formation costs O(mp^2), the current matrix O(p^2), and elimination O(p^3), plus rational bit growth, source storage and history. Testing a supplied assignment does not establish the complexity of solving for unknown phases.

All inherited rational and resource limits remain: 512-bit values, 32 sources, 64 phases, 128 errors, 64 outputs, 16 banks, 128 groups, 512 predicates, 8192 matrix entries and 20000000 algebra-work units. These are small reference-component limits, not an admission claim for the actual thousands-of-gates CNN structures. No increase of those limits is authorized here.

The candidate remains lossy away from the exact slices. The reference-slice property, ball geometry and finite square certificates are not claimed as individually new principles, nor as stronger than exact HZ. A publishable contribution requires stronger structural and cost evidence against the complete old path. Model coefficient binding, whole-domain precision on genuine bundles, terminal lowering, GPU soundness, concurrent performance, smooth/Transformer closure and all benchmark promotion gates remain open.

## Provenance and nonpromotion

Date 2026-10-04 Australia/Sydney; branch redu-hz; commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac. Expected tracked production diff SHA256 is 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5 and must be authenticated by the runner. No production source is edited or integrated by this experiment.

Formal baseline is 1870/2413, comprising 1063 CERT and 807 validated ADV. Separate E0 is CIFAR100 25 plus TinyImageNet 36, or 61/400. Mathematical success adds zero to either account. All historical sources, models, archives and result directories remain read-only.
