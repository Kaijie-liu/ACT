# Quantified loss in finite neural observation interfaces

This paper result quantifies a particular loss of information: replacing a common nonlinear source by finitely many feature means. Its worst downstream ambiguity is exactly twice a best uniform approximation error. It gives an explicit bounded-error contract rather than demanding exact closure at every depth. It is an application of classical approximation and moment duality, not an externally novel abstract domain or a benchmark improvement.

The [earlier interface audit](../d051_interface_sufficiency_20260930/PROOFS.md) already proved that finitely many continuous features and phase marginals need not determine a new biased ReLU mean. That qualitative obstruction is not a new result here. The addition is the exact quantitative ambiguity and recovery criterion, with a small original-phase example.

## Semantic scope

Let K be a nonempty compact set of admissible common source states. It can be a bounded closed nonconvex HZ relation, including continuous factors, every original signed phase, EQ/LE, a shared frame and the input decoder. For notation, an original signed phase may be viewed as beta in {0,1}; the underlying original phase is retained. Both legal labels at a zero preactivation belong to K. Features are continuous in the relative topology of K, which includes its discrete phase coordinates.

Choose continuous real functions phi_1,...,phi_k and a continuous consumer psi on K. Let V=span{1,phi_1,...,phi_k}. For a probability measure mu on K, the limited interface knows the vector m(mu)=(E_mu phi_i). These measures are mathematical descriptions of convex mixtures, not a distributional assumption on network inputs, random sampling, attacks or a probabilistic verification guarantee.

Crucially, the result concerns recovery of E_mu psi from m(mu), not recovery of psi(theta) from a particular theta. Even the injective point feature phi(z)=z need not determine E ReLU(z) from E z. A nonconvex representation retaining the complete common state is not restricted to this mean interface. Additional retained HZ predicates, observations or joint information must be included before applying any lower-bound conclusion to a verifier.

## Exact ambiguity theorem

Define

```text
d = inf_{g in V} max_{theta in K} |psi(theta)-g(theta)|
Delta = sup_{mu,nu probability measures on K; m(mu)=m(nu)}
            |E_mu psi-E_nu psi|.
```

Then Delta=2d. This is a maximum over all realizable summary values; it does not assert that every fixed summary has that diameter.

For the upper bound, matching feature means imply E_mu g=E_nu g for any g in V. Hence the difference in consumer means is at most 2 max_K|psi-g|. Taking the infimum gives Delta<=2d.

For the reverse bound when d>0, V is a closed finite-dimensional subspace of C(K). Hahn–Banach separation in C(K)/V gives a norm-one bounded linear functional L annihilating V with L(psi)=d. Riesz representation gives a finite signed measure sigma with total variation 1, integral of psi equal to d, and integral of every g in V equal to zero. Since 1 belongs to V, sigma has total mass zero. Its positive and negative Jordan parts consequently both have mass 1/2. Set mu=2*sigma_positive and nu=2*sigma_negative. They are probability measures with matching features and consumer difference 2d. The d=0 case follows from the upper bound. This proves the theorem.

For completeness, the witnesses need not require diffuse measures: apply finite-dimensional convex-hull representation to (phi_1,...,phi_k,psi) to replace each witness by a finite atomic measure preserving these integrals. This is a proof of existence, not an algorithm to enumerate input or phase subproblems.

There is also an exact recovery statement. Among all decoders D of the summary vector, including nonlinear decoders,

```text
inf_D sup_mu |D(m(mu))-E_mu psi| = d.
```

The lower bound follows because the two matching witnesses cannot both be approximated by a single decoder value with error below half their separation. A best g=a0+sum_i a_i phi_i exists in the finite-dimensional space V and yields the affine decoder D(m)=a0+sum_i a_i m_i with error at most d. Thus allowing a complicated nonlinear decoder does not overcome this specific worst-case mean-information loss.

The proof uses functional-analytic duality only on paper. No LP dual, marginal, infeasibility ray, backward rescue or optimization routine is introduced into the verification path.

## Ordinary ReLU example retaining the original bit

Consider the actual original gate psi(z)=ReLU(z-1/2), with z in [0,1], and its original active view beta. Its source set includes beta=0 for z<=1/2 and beta=1 for z>=1/2, with both choices at z=1/2. Let the summary retain 1,z,beta.

The explicit affine approximation g=z/2-1/8 has uniform error 1/8. On the beta=0 part, psi-g=1/8-z/2; on the beta=1 part, psi-g=z/2-3/8. Both lie in [-1/8,1/8].

Take the following finite mixtures of genuine gate states:

```text
mu = (1/2) point(z=0,beta=0) + (1/2) point(z=1,beta=1)
nu = (1/2) point(z=1/2,beta=0) + (1/2) point(z=1/2,beta=1).
```

Both have summary (1,1/2,1/2). Their consumer means are 1/4 and 0. Therefore d>=1/8 even when the approximation is allowed a beta coefficient; the explicit g attains this lower bound. Delta=1/4 exactly. No original bit has been deleted or merged.

The information gap is not dependent on zero-gate ambiguity. Replacing the first mixture by equal masses at z=1/8 and 7/8, and the second by equal masses at z=3/8 and 5/8, gives the same feature means and legal phase masses. All four preactivations are strictly nonzero and all four inputs are strictly interior. The consumer means are 3/16 and 1/16, differing by 1/8. Adding any common affine residual az+b changes both expectations equally and does not remove the discrepancy.

These are analytic mixture witnesses, not adversarial examples, sampled inputs or a new strong-control separation from a full HZ system. For this particular known consumer, adding the actual mixed feature beta*z makes psi=beta*z-beta/2 exactly representable. That remedy is a known product relation, not a new-domain contribution.

## Usable forward contract without an optimality oracle

A future implementation need not compute d or solve for the signed measure. Given an explicitly constructed g=a0+sum_i a_i phi_i and a sound pointwise certificate epsilon>=max_K|psi-g|, it may use

```text
|E_mu psi-a0-sum_i a_i E_mu phi_i| <= epsilon.
```

For a nonnegative conditional mixture of mass m, the corresponding statement is

```text
|integral psi-a0*m-sum_i a_i integral phi_i| <= m*epsilon.
```

At the actual common-state level, if pi>=0 is a retained original phase indicator, the pointwise relation |pi*(psi-g)|<=pi*epsilon follows directly. Every observation must still refer to the same theta. This proof does not authorize replacing shared error identities with independent noise and then claiming joint exactness.

For a fixed vector of actual consumers Psi=(psi_j), the scalar theorem applies to each structurally declared direction w: its mean ambiguity is 2*dist_infinity(w dot Psi,V). Using only separate coordinate errors can lose cancellations; the [earlier envelope audit](../d100_envelope_factorization_20261001/THEORY.md) already addressed that distinct problem. No runtime search over w, labels, margins or solver states is implied.

This permits certified inexact interfaces rather than insisting on universal exact closure. However, computing a useful pointwise certificate, constructing shared observations, bounding arithmetic and preserving the complete decoder are still real obligations. The infimum over functions is not a free forward operator. Without a concrete structural certificate and full-cost comparison, this contract is not an implementable new Neural-HZ candidate.

## Prior work and research consequence

[Han, Jiao and Weissman, COLT 2018, Appendix B.1, Lemma 25](https://proceedings.mlr.press/v75/han18b/han18b.pdf) state the factor-two relation between moment matching and best uniform polynomial approximation on an interval. The finite-feature version above is a standard functional-analytic extension, proved here to make its neural-interface assumptions explicit. We do not import the paper's statistical estimator, sampling model or inference procedure.

Unlike D051's qualitative warning, the quantitative criterion states the exact worst-case penalty of a specified mean interface and the best possible recovery error. Unlike D100's chosen box-envelope comparison, it ranges over all mixtures on the declared K and all decoders of the declared features. It still does not establish a new domain or explain D116's zero query gain: that experiment retained additional predicates and considered only 14 fixed directions.

The next representation hypothesis may use an explicit nonzero error budget or retain an additional genuinely needed common relation. It must not assume a generic finite moment interface is exact, solve an unbudgeted best-approximation problem, or relabel a known beta*z lifting as innovation. Original nonconvex HZ semantics and all promotion requirements remain unchanged.
