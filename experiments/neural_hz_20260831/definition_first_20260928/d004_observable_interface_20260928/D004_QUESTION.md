# D004 question — guard-complete observable interfaces

2026-09-28, `redu-hz`, commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.
Paper-only next question after D003; no implementation, target scan or numerical
run. Not a newly established abstract domain, novelty result or measured gain.

## The question worth falsifying

For a normal mixed affine/ReLU/residual block with Q ReLU gates, can guard-aware
identities expose an interface of k<Q retained nonlinear values such that EVERY
later preactivation, declared output and original EQ/LE consumer has an exact
constant-affine representation in the original input/free bits and available
retained values? All original gate bits must remain. No phase enumeration,
split, numerical identity guessing, dual rescue or instance selection.

The target is the complete relation Gamma=(original input, ALL original bits,
declared outputs), with all original predicates and exact reconstruction of
hidden values. This is D002's observable relation. It does **not** require a
constant-affine native decoder for every hidden node, which D003's stronger
reference deliberately supplies. D003's pass remains unchanged; a future
observable-interface candidate must explicitly prove this distinction, not
reuse its pass as evidence that a changed decoder is qualified.

## Conditional exact terminal theorem

Order gates topologically. Before deciding gate j, suppose a certificate proves
that its actual preactivation equals an affine form fhat_j of original inputs,
original free bits and already retained gate values, on EVERY feasible original
prefix with the SAME prefix bits. The proof may use already proved prefix
identities, but not the very removed definition it is supposed to establish,
nor assumptions supplied only by future gates. The analogous output/predicate
identities must hold on all applicable complete/prefix feasible assignments.

For retained gates, emit the ordinary four exact rows. For a removed gate keep
its bit beta_j and emit only its complete sign guard:

    l_j*(1-beta_j) <= fhat_j <= u_j*beta_j.

Here l_j,u_j are independently sound bounds for fhat_j on the matching prefix.
Reconstruct its value as beta_j*fhat_j when materializing a concrete witness.
This reconstruction may be nonlinear in native variables; it is exact and
does not discard the bit or a predicate. At zero, both phases remain legal.

Proof by induction: fix the original input/free bits and complete gate bits.
Assume the retained native prefix corresponds to an original feasible prefix.
The certified affine identity makes fhat_j the original f_j. The two sign rows
are equivalent to the full sign guard within the sound bounds. A removed gate
therefore reconstructs its unique ReLU value beta_j*fhat_j; a retained gate's
four rows force that same value. Extend the matched prefix and repeat. The
converse maps any original feasible prefix to the retained values and satisfies
the same rows. Certified original predicates/outputs then give equality of
Gamma, not merely one-way inclusion or output projection.

This is conditional on a NON-CIRCULAR certificate. The bare assertion
"the original full network implies a replacement identity" is insufficient
if checking that identity presupposes deleted constraints in the new system.

Ignoring only unrelated input/free-bit bounds (identical in both versions),
the gate bill becomes p+k continuous variables, all Q gate bits, and
4k+2(Q-k)=2Q+2k rows, instead of p+Q and 4Q. Original predicates, all coefficient
fill-in, sound-bound computation, certificate construction/checking, witness
reconstruction and saved evidence are additional real costs. Fewer rows or
variables alone is not a net-benefit or admissibility result.

## Strong comparator and a useful negative control

Give the ordinary HZ/shared-circuit comparator the SAME certified identities.
Allow it exact common-subexpression sharing, stable/signed/co-sign folding,
affine elimination and ordinary optimized piecewise-linear formulations. If
it obtains the same interface and terminal bill, this conditional theorem is
not a domain-specific advantage; a domain innovation needs another substantive
property. No novelty is claimed for the theorem itself.

An ordinary two-input mixing example prevents a cheap "small output implies
small nonlinear interface" argument:

    r1 = ReLU(x1+x2)
    r2 = ReLU(x1-x2)
    g  = r1 - 2*r2 + x2
    y  = x1 + ReLU(g)

On a box with interior around zero, the two crossing kink lines are distinct.
If g were affine in (x1,x2,r1), rearrangement would express r2 as such an affine
form, impossible near a point of x1=x2 away from x1=-x2: the right-hand side is
locally affine there while r2 has a kink. The symmetric argument rules out an
affine form in (x1,x2,r2). Thus either original hidden gate alone does not meet
the prefix-interface premise. This does NOT rule out every alternative lift,
new combined generator, multi-bit formulation or more general exact domain.

## Decision boundary for the next exploration

First seek a substantive certificate/definition property on ordinary repeated
mixing/residual structures and compare it against the strong baseline above.
If only exact duplicate neurons, stable gates or scalar threshold chains meet
the premise, record that negative result and do not promote this as the main
large-CNN route. No rare-structure or extreme-number search is justified.
An implementation or real-structure census needs a fresh scoped preregistration;
this paper question authorizes neither. Formal1870/2413 and separateE0=61/400
are unchanged, and the full research goal remains ACTIVE.
