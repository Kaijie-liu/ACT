# Quadratic relation boundary for a phase preserving ReLU bank

This is a paper result for the definition-first Neural-HZ project. It tests a specific proposed source of innovation: whether a degree-two vanishing-relation component can discover genuinely new cross-neuron equalities on an ordinary affine-to-ReLU bank, while retaining its original binary phases. Under the explicit crossing hypotheses below, the complete answer is negative: every such identity is a constant linear combination of four familiar single-neuron identities. The result is not a new abstract domain, a benchmark result, or a theorem that relational verification cannot improve.

Two independent mathematical reviewers checked both the output-only argument and the strengthened version including all current-layer original phase variables. This is not a machine-checked theorem. No numerical program, candidate import, solver, model or GPU was run.

## The candidate that this question would support

The contemplated element was D=(H,J2,L,decoder). H is the complete original nonconvex HZ, with bounded continuous factors, all original signed binary factors, EQ/LE predicates, shared latent/frame identity and concrete-input reconstruction. J2 is a finite certified collection of total-degree-at-most-two identities among common-source coordinates, current ReLU outputs and current original bits. L is the joint retained output readout.

Its exact concrete meaning intersects the original HZ relation with those identities and projects only internal continuous coordinates. Valid J2 entries are consequences of actual network semantics; they do not authorize deleting original phase guards or original constraints. Empty J2 embeds H exactly. Semantic inclusion defines a preorder; no efficient best abstraction, full lattice or canonical complete ideal computation is assumed. A linear product-envelope lowering would be an additional approximation whose precision and cost need separate evaluation.

Such a container alone is already within established polynomial/mixed-symbolic representational ideas. Its possible contribution would have to be a useful neural structural inference and a paid compositional advantage. The theorem below evaluates the narrower hypothesis that complete quadratic identity discovery on one ordinary bank supplies new cross-neuron equalities merely because neurons share a low-dimensional source. It does not evaluate all inequality-based or cross-layer components.

## Exact graph and geometric hypotheses

Let x be d independent effective real source coordinates, d>=2. Write g_i(x)=a_i dot x+c_i with a_i nonzero, i=1,...,m. Let U be a nonempty open subset of the actual feasible source region. Neither convexity nor connectedness of U is required. Each H_i={x:g_i(x)=0} must cross U.

For every pair i<j, require a point p_ij in U satisfying all of the following:

- g_i(p_ij)=g_j(p_ij)=0;
- a_i and a_j are linearly independent;
- g_k(p_ij) is nonzero for every other current neuron k.

These are local reachability hypotheses, not a conclusion inferred from interval instability or ordinary matrix rank. They provide a small source ball around p_ij in which only the two selected gates change sign and all four sign combinations contain open subsets. Using these four subsets in a proof is not a runtime input or phase splitting algorithm.

The exact graph has q_i=max(0,g_i(x)) and original active-bit views beta_i in {0,1}. When g_i>0, beta_i=1; when g_i<0, beta_i=0; at g_i=0, both original labels are legal. A beta view is an authenticated affine view of the original signed bit, not a new factor or a relaxation. Its orientation must follow the original gate contract.

The theorem concerns polynomials only in these x coordinates, current q coordinates and current beta coordinates. If preactivations are retained as separate g variables, their affine defining equations and polynomial multiples must first be accounted for by substitution. Ancestor bits, arbitrary additional old-layer outputs and nonlinear source parametrizations are not silently treated as independent x coordinates. Extra source predicates are allowed only when the required feasible open neighborhoods really remain; no predicate may be dropped to fabricate them.

## Complete quadratic identity statement

Let I2 be the real vector space of all polynomials P(x,q,beta) of total degree at most two that vanish on the exact graph above. Then

    I2 = span over real constants of
         q_i*(q_i-g_i),
         q_i*(beta_i-1),
         q_i-beta_i*g_i,
         beta_i^2-beta_i,
         for i=1,...,m.

The 4m displayed polynomials are linearly independent in the free polynomial ring, so dim(I2)=4m. I2 itself is a bounded-degree vector space, not an ideal closed under arbitrary multiplication. In particular, source rank below the number of neurons does not by itself produce additional quadratic cross-neuron identities.

For an original signed bit sigma_i=2*beta_i-1, the same span has generators

    q_i*(q_i-g_i),
    q_i*(sigma_i-1),
    2*q_i-(1+sigma_i)*g_i,
    sigma_i^2-1.

This affine change preserves total degree and every original zero-phase label. It is notation only; no bit is pivoted, merged, deleted or continuously relaxed.

## Proof of necessity

Every degree-two P can be written uniquely using a pure-source polynomial P0 of degree at most two, affine forms l_i(x), m_i(x), and constants as

    P = P0(x)
      + sum_i [q_i*l_i(x) + beta_i*m_i(x)
               + A_i*q_i^2 + C_i*beta_i*q_i + D_i*beta_i^2]
      + sum_{i<j} [a_ij*q_i*q_j + b_ij*q_i*beta_j
                    + c_ij*beta_i*q_j + d_ij*beta_i*beta_j].

Fix i<j and take the certified small ball around p_ij. On each of its four open sign regions, beta_i and beta_j are constants 0 or 1, q_i is respectively 0 or g_i, and q_j is respectively 0 or g_j. Every other gate has a fixed sign in the entire ball. Substituting these expressions produces four polynomials in x. Each is zero on a nonempty open set, so each is identically zero as a polynomial, even though the four open sets differ.

Their mixed difference, with signs +,-,-,+ for 11,10,01,00, cancels every term except

    a_ij*g_i*g_j + b_ij*g_i + c_ij*g_j + d_ij = 0.

The affine map x -> (g_i,g_j) has rank two. Thus the expression is a zero polynomial in two independent affine coordinates, forcing all four coefficients to be zero. Repeating this argument for each pair removes every cross-neuron quadratic term. It does not enumerate complete phase assignments.

Now cross H_i at a feasible ordinary point not on any other current hyperplane. Such a point exists under the stated hypotheses. Subtracting the two resulting zero polynomials gives

    g_i*[l_i + A_i*g_i + C_i] + m_i + D_i = 0.

The bracket is affine. If its linear part were nonzero, multiplying it by the nonzero linear part of g_i would create a nonzero homogeneous quadratic polynomial, which cannot cancel the affine m_i+D_i. The real polynomial ring has no zero divisors, so the bracket is a constant K_i. Consequently

    l_i = -A_i*g_i + K_i-C_i,
    m_i = -K_i*g_i-D_i.

The whole i-th contribution is therefore exactly

    A_i*q_i*(q_i-g_i)
    + C_i*q_i*(beta_i-1)
    + K_i*(q_i-beta_i*g_i)
    + D_i*(beta_i^2-beta_i).

After this substitution, the remaining P0 vanishes on U and is the zero polynomial. This proves that no other degree-two identity exists in the stated language.

## Sufficiency and legal zero labels

If g_i<0, q_i=beta_i=0 and all four generators vanish. If g_i>0, q_i=g_i and beta_i=1, with the same conclusion. If g_i=q_i=0, both beta_i=0 and beta_i=1 make all four vanish. Thus every listed identity holds on the complete original graph, including its zero labels.

For independence, the coefficients of q_i^2, beta_i*q_i and beta_i^2 first determine A_i, C_i and D_i in any zero linear combination. The remaining beta_i*g_i terms force K_i=0 since g_i is nonconstant and has its own beta_i coordinate. No cross-neuron cancellation can remove them. Hence the dimension is exactly 4m.

The generator equations alone still do not encode the signs. For example beta=0, q=0, g>0 satisfies all four equations but is not a ReLU graph state. The retained nonnegativity/sign guards and original HZ remain essential; this theorem does not authorize replacing them with a polynomial equality system.

## Ordinary overcomplete control and hypothesis boundary

Take the source box [-1,1]^2 and its interior U, with three biased gates

    g1=x+1/5,
    g2=y-1/7,
    g3=x+y-1/2.

Their normals have rank two, less than the three outputs. Yet every pair has a transverse intersection strictly inside the box, away from the remaining gate:

| Pair | Source intersection | Remaining preactivation |
| --- | --- | --- |
| 1 and 2 | (-1/5,1/7) | g3=-39/70 |
| 1 and 3 | (-1/5,7/10) | g2=39/70 |
| 2 and 3 | (5/14,1/7) | g1=39/70 |

All biases are nonzero and all gates cross zero. The determinant, interior and nonzero-third-value conditions persist for sufficiently small parameter perturbations; no numerical perturbation radius has been certified. The 45-dimensional space of degree-at-most-two polynomials in the eight coordinates (x,y,q1,q2,q3,beta1,beta2,beta3) therefore has exactly the twelve listed vanishing directions. These are paper calculations, not matrix-rank software output or a network verification result.

Domain reachability matters. For g1=x and g2=x+y/10+1/2 on [-1,1]^2, both gates are unstable and their normals are independent, but their intersection (0,-5) lies outside the source box. The valid extra identity is q1*(q2-g2)=0: whenever q1>0, g2>2/5, so q2=g2; otherwise q1=0. This is precisely the kind of bounded-source amplitude conflict that the hypotheses exclude and that the earlier support-invariant work already studies. It is not an exotic numerical exception and not a reason to discard that earlier work.

## Implication for domain research

This closes only the premise that generic quadratic equality completion on these banks automatically yields a new cross-neuron invariant. It does not say that adding the known relations to a weak linear relaxation cannot improve its bound. Product envelopes, inequality certificates, source-dependent conflicts, higher-degree relations and relations spanning old and new layers can still carry useful information. They must be compared with the same known information at the same complete cost.

Affine/Conv/Add readouts do not invalidate the theorem: any quadratic identity purely in an affine readout pulls back to one in the displayed coordinates. That does not prove that discovering a projected relation is algorithmically free or that a lower-dimensional readout representation cannot be useful. It prevents attributing new semantic content to the quadratic identity container alone. Another ReLU introduces new phases and a generally piecewise-affine source, so its relation space is not covered without a new proof.

The result supplies no real-model applicability census. Proving the required source neighborhoods on arbitrary constrained HZ factors may itself be expensive; no hidden feasibility oracle, phase search, input split or solver rescue is proposed. A GPU monomial tensor would not change this semantic boundary. No such tensor implementation or new test framework is warranted by this candidate alone.

## Prior work and novelty limits

The complementarity equation q*(q-g)=0 with q>=0 and q-g>=0 is established; [Aydinoglu et al., Lemma 1 and Lemma 2](https://arxiv.org/pdf/2011.07626) give ReLU and network complementarity representations. Their solver/stability machinery is not adopted here. The binary and gate-product equations above are standard consequences of the original gate semantics, not new discoveries.

[Sharp Hybrid Zonotopes, Theorem 7 and equation 18](https://arxiv.org/html/2503.17483v2#S4) already treats Boolean products and phase-conditioned continuous products through RLT within HZ. The present proof does not establish a new alternative to that hierarchy, nor authorize its binary relaxation or branching strategies.

Project [D009 section A](../d009_bounded_phase_energy_20260928/D009_PROOFS_AND_LIMITS.md) already used independent hinge crossings to rule out new affine identities. This record extends that project-level classification to all degree-two polynomials including current-layer bits, under stronger pair-crossing premises. A limited primary-source search did not establish whether this exact classification has already appeared elsewhere. No external novelty, publication-level contribution or formal capability gain is claimed.
