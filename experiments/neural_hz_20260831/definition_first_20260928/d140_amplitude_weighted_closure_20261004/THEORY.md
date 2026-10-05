# Shared amplitude relations across neural layers

This paper study supplies a constructive forward relation system, not a completed new Neural-HZ domain. A fixed original activation supplies a common nonnegative weight. Weighted source, phase and output observations propagate through affine maps, mixed weights, biases and live residuals. Reusing a weighted phase observation in the original descendant output equation gives a strict two-layer control: the new rows prove a successor identically zero, whereas the intersection of the specified complete individual child hulls does not. The positive control has strictly interior, nonzero-sign witnesses and a nonzero bias interval.

Two strong comparisons prevent selecting this system for implementation. Weighted copies alone are dominated by the corresponding complete single-phase lift. Adding the output identity gives the positive comparison above, but the archived D052 rule already solves that entire control more tightly with one row and no auxiliary quantity. Both ingredients come from established RLT and original ReLU algebra. New abstract-domain novelty, useful trained-network coverage, GPU performance and formal verification gains are not established.

## Protected semantics and common observations

Let H retain the original continuous factors, every original signed binary factor, EQ and LE predicates, common source frame and input decoder. Write an original active indicator as alpha=(b+1)/2; this changes notation, not its identity or discrete domain. Let theta collect the retained physical coordinates and these indicator views. Every view must have a certified readout in H.

Choose structurally an actual original activation q=ReLU(f) with a certified 0<=q<=U and U>0. Put a=q/U. A finite observation environment assigns each selected quantity v its single canonical observation v_a=a*v. Multiple consumers use the same observation, not independent copies. An old predicate row L*theta<=d has the valid lifted rows

```text
L*theta_a <= d*a
L*(theta-theta_a) <= d*(1-a).
```

An EQ needs only one additional weighted EQ; its complement follows from the old EQ. Finite scalar bounds generate the usual four product-envelope rows. Products with original binary indicators can be encoded exactly with their four envelopes; continuous products generally cannot.

The canonical interpretation and the implemented linear outer relation are different. For every original integer state theta, the same simultaneous assignment theta_a=a(theta)*theta satisfies all proposed rows. The stored mixed linear system may also admit noncanonical continuous-product assignments. Retaining all of H proves that its projection onto the original states is exactly H, not that every stored auxiliary assignment is a true product. Empty observations embed H. Semantic inclusion is inclusion of concretizations in a common frame; no complete lattice or best abstraction is asserted.

This preserves all legal zero labels: q=0 implies a=0 under either original label, and the weighted observations are all zero. No binary factor is pivoted, deleted or relaxed in the actual domain. Fractional indicators below describe a query relaxation only. The original decoder remains authoritative; an auxiliary solution is never a validated ADV by itself.

## Forward propagation and an output quotient

For an actual affine or convolutional readout v=W*u+d,

```text
v_a = W*u_a + d*a.
```

The bias scales with a. Add and Concat preserve the common identities; a live skip uses its existing weighted readout. For an actual ReLU r=ReLU(g), positive homogeneity gives a*r=ReLU(a*g), with the SAME original phase label. At a=0 both weighted quantities vanish, but this does not erase the original unweighted guard. Original gate rows multiplied by a and 1-a give a sound forward linear representation. Repeating this operation does not introduce products of observations with one another.

At the anchor itself, a*alpha=a and a*q=a*f. The latter follows from q*(q-f)=0; it also follows by combining the weighted original upper and lower rows with the former identity. These are established ReLU identities, not a new complementarity theorem.

The additional rule uses the actual descendant equation, not a reference network. Suppose

```text
g = c*q + v,    r = ReLU(g),    original bit beta,
B = q*beta = U*beta_a.
```

Then r-c*B=beta*v for every original state. For a certified bound L<=v<=V, install

```text
L*beta <= r-c*B <= V*beta
v-V*(1-beta) <= r-c*B <= v-L*(1-beta).
```

This uses four LE rows and no additional variable if B is already present. It works for either sign of c and preserves the original gate and bit. It is the existing exact product substitution of D035, now consumed by a common amplitude-weighted forward system. The residual v includes every other input and skip; its bounds and readout expansion are payable. Setting a missing residual to zero is not allowed. The rule is chosen from source/readout structure, never from a property margin, model identity or an LP solution.

At fixed anchor count k, products have degree at most two in retained physical coordinates. This is NOT a bounded polynomial-degree statement after expanding the whole network into original inputs, nor a claim that network coordinates or old predicates disappear.

## Why the plain weighted copy does not suffice

Consider the stronger comparison that already contains the complete single-phase lift z of a bounded linear relaxation P={theta:A*theta<=d} around the anchor's original alpha:

```text
A*z <= alpha*d
A*(theta-z) <= (1-alpha)*d
z_alpha=alpha,  z_q=q,  0<=a=q/U<=alpha.
```

All source bounds, EQ and scalar bounds are included. This is a statement about a precisely specified lift; it does not assert that the current production path constructs it.

Every feasible point in this comparison extends to the plain amplitude-weighted system. For alpha>0, set lambda=a/alpha and w=lambda*z. Then w_alpha=a and A*w<=a*d. Moreover

```text
theta-w = (1-lambda)*theta + lambda*(theta-z),
```

so its row bound is (1-lambda)*d+lambda*(1-alpha)*d=(1-a)*d. The weighted anchor definition is consistent, and weighted scalar bounds give all product envelopes. When alpha=0, a=q=0 and w=0 works. The diagonal value w_a=a*a/alpha lies between a*a and a, so adding ordinary lower tangents of the square alone does not defeat this construction.

Thus plain amplitude weighting cannot claim stronger projection than this phase lift. The proof does not cover the descendant output-quotient rows above: their physical r term is not rescaled along with w, so independent rescaling need not satisfy them. A bank of independent anchors is covered separately; shared cross-anchor product identities would require a new argument. No such bank is selected here.

## An ordinary biased control with a live residual

Take x in [-1,1/2], y in [-1/2,1/2] and any fixed epsilon in [1/200,1/100]. Define the actual network

```text
q = ReLU(x+1/2),                    original alpha
r = ReLU(q+y-1/2-epsilon),          original beta
s = ReLU(q-y-1/2-epsilon),          original gamma
J = r+s-q
h = ReLU(J-3/8),                   original eta.
```

All three earlier gates have ordinary crossing ranges. The two child rows have mixed signs and a nonzero bias; q is a live negative skip in J. Common certified bounds are 0<=q<=1 and child preactivations in [-1-epsilon,1-epsilon]. No phase subset or source subproblem is enumerated at runtime.

Use the anchor a=q and shared observations

```text
X=q*x, Y=q*y, Q=q*q, R=q*r, S=q*s, B=q*beta, C=q*gamma.
```

The existing q*alpha observation is exactly q and needs no separate variable. Retain original gates and source rows, product envelopes, their q and 1-q weighted rows, and the two descendant quotient systems. For the following certificate abbreviate p=r+s, b=beta+gamma, z=B+C and H=R+S. These abbreviations need no new coordinates.

The complements of r<=(1-epsilon)*beta and s<=(1-epsilon)*gamma give

```text
H >= p-(1-epsilon)*b+(1-epsilon)*z.
```

The other original upper rows are r<=q+y+1/2-(1+epsilon)*beta and s<=q-y+1/2-(1+epsilon)*gamma. Weighting them by q, and adding BEFORE bounding the common Y, gives

```text
H <= 2*Q+q-(1+epsilon)*z,     Q<=q.
```

The two quotient residuals y-1/2-epsilon and -y-1/2-epsilon lie in [-1-epsilon,-epsilon]. Their upper envelope rows imply

```text
p <= z-epsilon*b
p <= z+1-(1+epsilon)*b.
```

Combining these inequalities yields the explicit forward certificate

```text
3*p-1+3*epsilon*b <= p-(1-epsilon)*b+2*z <= 3*q
J <= 1/3-epsilon*b <= 1/3.
```

Consequently the new relation system proves h=0 with upper preactivation margin -1/24. This is a sound upper bound; no claim that 1/3 is the exact support optimum of the whole new relaxation is needed or made.

## A fully specified stronger old comparison

Compare against the original source and parent graph, the complete parent-alpha lift above, and the intersection of each child's COMPLETE source-labelled hull separately. Each individual hull includes x,y,q,alpha, its own child output and its own bit. It does not include the joint graph of BOTH children, nor complete cross-layer hulls involving h. Both sides may use the common conservative scalar preactivation interval [-11/8,13/8-2*epsilon] for h, derived from the displayed coordinate bounds. The old tuple below satisfies the exact h graph. We do NOT give the old side the globally exact true J or h interval: that would already prove the property, and is precisely additional joint information absent from this comparison.

This comparison admits

```text
x=y=0, q=1/2, alpha=1,
r=s=9/20-epsilon/2, beta=gamma=1/2,
J=2/5-epsilon, h=1/40-epsilon, eta=1.
```

For r, take equal weights on the two real source points

```text
(x,y,q) = (9/20,9/20,19/20),  r=9/10-epsilon, beta=1,
(x,y,q) = (-9/20,-9/20,1/20), r=0,             beta=0.
```

For s reverse the two y signs and use the analogous outputs. Every witness is strictly inside its source box, the parent preactivation is strictly positive, and each selected child's preactivation is strictly nonzero. The source and parent coordinate means agree across the two hulls, but their conditional q*y observations do not have to agree. Because alpha=1, the complete parent-alpha lift adds no restriction: take z=theta. The displayed false children's preactivations are -epsilon, also nonzero. The successor takes its exact graph value h>0. No point in this proof is a concrete adversarial example.

The new row J<=1/3 excludes the old J, which is at least 39/100>1/3. The old positive h lies in [3/200,1/50], while the new system proves h=0. This gives a strict downstream consequence, not only a different auxiliary assignment.

The true network has the still stronger relation J<=0. Indeed, writing A=q+y-1/2-epsilon and D=q-y-1/2-epsilon, ReLU(A)+ReLU(D)=max(0,A,D,A+D), and each of these four terms is at most q. Hence a full joint child hull, or ordinary HZ supplied this exact structural consequence, already proves h=0. Ordinary HZ supplied ALL of the new weighted and quotient rows is exactly equal to the candidate's linear query system. These are mandatory strong comparisons; this control does not establish new expressive power or an advantage over all old relational methods.

## The existing project rule already solves this control

The decisive comparison is not merely a hypothetical full hull. Apply the archived [D052 fixed coefficient matching rule](../d052_static_source_certificates_20260930/THEORY.md) to the actual common internal frontier

```text
u=q in [0,1], t=y+1/2 in [0,1],
f=u+t-1-epsilon, g=u-t-epsilon.
```

Its prescribed matching has P=u, N=t, kappa0=kappa1=U_bound=-epsilon and Delta=0. The rule therefore directly installs

```text
r+s-q <= -epsilon*(beta+gamma) <= 0.
```

This is one LE with five physical-coordinate nonzeros for epsilon>0 and zero new auxiliary or binary quantities. The coefficient scan, source bindings, readout expansion and evidence still cost work; neither comparison gets those for free. D052 explicitly permits a certified internal physical frontier and negative certificate constants, so there is no missing premise that would exclude this example.

Consequently the whole perturbation family is already covered by a stronger, cheaper, structurally generated project rule. The new combination is a valid mathematical formulation, but the control is NOT evidence of an incremental project capability. Do not preregister an implementation on the strength of this example or replace the old rule with the larger system.

## Complete local costs and forward limits

For the prefix q,r,s above, without duplicate-row elimination and using physical 0/1 bit views, one conservative explicit bill is:

| Increment | Continuous quantities | LE rows | Coefficient nonzeros |
| --- | ---: | ---: | ---: |
| Five continuous product envelopes | 5 | 20 | 43 |
| Two original-bit product envelopes | 2 | 8 | 16 |
| q and 1-q times the twelve original gate rows | 0 | 24 | 87 |
| Two descendant output quotients | 0 | 8 | 28 |
| Explicit auxiliary scalar bounds if stored as rows | 0 | 14 | 14 |
| Total increment | 7 | 74 | 188 |

There are zero new binary factors. Omitting the separately stored scalar-bound rows gives 60 LE and 174 nonzeros, not a claim that those bounds cost nothing. At epsilon=0 two coefficients in each quotient vanish, but that smaller count is not used for the positive interval above. Actual HZ signed-bit conversion, factor normalization, RHS, LE-to-EQ slack if required, original H, source expansion, certificates, aliases, host/device copies, terminal work and decoding are additional.

The count stops at q,r,s. The original h and its bit are retained in both comparisons; the certified J bound proves stability directly. Continuing the same observation through h would require q*h, q*eta and their corresponding rows, or an independently certified exact stable rewrite. Those costs are not in the table.

With k fixed anchors and n retained quantities, R LE rows, E EQ rows and s coefficient occurrences, a dense-in-scope version adds at most k*n observations, 2*k*R weighted LE rows, k*E weighted EQ rows, and O(k*n) envelope rows; weighted coefficient occurrences are O(k*(s+R+E+n)). Descendant quotient rows add four per selected anchor/gate occurrence plus the full residual readout support. All scopes must include actual consumers. Fixed k prevents exponential observation-degree growth, but this is linear MULTIPLICATION of the old global cost, not compression or a speed claim.

Affine/Conv weighted propagation has an obvious batched matrix-operation form. This observation neither certifies floating-point arithmetic nor puts the remaining terminal solver on GPU. No GPU or candidate ran. Smooth activations lack the ReLU identity r=beta*g; their homogeneous perspective a*phi(g) is only a semantic observation here, not an implemented or proved cheap smooth/Transformer transformer.

## Prior art and present research decision

[D035](../d035_cross_phase_source_20260930/THEORY.md) already gives the native output product substitution; [D091](../d091_joint_realization_relay_20261001/THEORY.md) gives the common canonical realization principle; [D114](../d114_quadratic_relation_boundary_20261002/THEORY.md) prevents claiming the anchor's complementarity identity as new. [Sharp Hybrid Zonotopes, Section IV, Theorem 7 and equation 18](https://arxiv.org/html/2503.17483v2#S4) explicitly develops RLT lifting within HZ. Its displayed hierarchy uses Boolean and Boolean-continuous products; that does not make the continuous weighting used here a newly invented general RLT principle.

The project-level result is the specified amplitude-weighted forward system, the dominance audit that isolates its insufficient part, and the stronger-reference audit that rejects its current capability justification. The 74-row local cost and D052's one-row coverage make this control unsuitable for implementation admission. A genuinely new positive case must first survive the existing native quotient, phase-conditioned and source-matching rules under a fixed structural policy and full-cost comparison. The present record selects no runtime candidate. Do not turn this result into a new independent solver or a growing helper framework.

All algebra is exact-real paper reasoning and independent manual review. It is not machine-checked, a numerical admission, trained-model evidence, or formal score improvement.
