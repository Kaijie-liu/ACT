# Shared source certificates for phase preserving group contraction

This paper record gives a constructive group certificate for the existing phase-compatible contextual quotient, including an optimal discovery rule within one explicitly bounded certificate family. It changes the question from independent omitted-term bounds to a shared-source amplitude budget. It is supporting mathematics, not a completed Neural-HZ domain, a new kind of convexification, a numerical qualification or a solved benchmark.

The full goal remains definition-first nonconvex Neural-HZ for ordinary neural structures, GPU, smooth activations and Transformers, all 13 families and independent external CNNs. The predecessor [contextual quotient](../d005_contextual_congruence_20260928/D005_DEFINITION_DRAFT.md) is an explicit comparator, not a result being rediscovered under a new name. All proofs here are paper derivations independently reviewed by agents and the root; none is machine-checked.

## Exact semantic interface

Let S be the same retained source relation, with continuous coordinates, every original binary identity, EQ/LE predicates, shared frame and input decoder. Certificates may use an outward-safe enclosing box, but no source predicate is removed from S. They must not depend on the new target guard or on a constraint being replaced. Write R(t)=max(0,t), and retain the original child phases alpha_i and target phase beta.

For a preactivation z, its full labeled relation is

```text
R_z(s) = { (beta,r) : beta in {0,1},
                     (2 beta-1) z(s) >= 0, r=beta z(s) }.
```

Thus equality of R_z and R_h requires either z,h both strictly negative, or z=h>=0, at every same source state. Equality of rectified values alone is insufficient at zero. The proposed rewrite is local to this typed ReLU consumer, not an equality available to arbitrary raw-value predicates.

## A group certificate using one common remainder

Consider a pre-registered structural group of positive contributions

```text
q_i = R(f_i),     a_i > 0,
g = h + sum_i a_i q_i.
```

The remainder h includes every unselected contribution, bias and residual. Suppose same-source certificates establish

```text
h + kappa_i f_i <= 0   for every i,
kappa_i > 0,
T = sum_i a_i/kappa_i < 1.
```

Then R_g=R_h, including the target's original zero-phase labels. In particular, the whole selected positive group can be removed from this consumer at once.

Proof without a runtime phase partition: if h>=0, every f_i<=-h/kappa_i<=0, hence every q_i=0 and g=h. If h<0, then 0<=q_i<=-h/kappa_i and

```text
h <= g <= (1-T) h < 0.
```

Therefore the complete target relation agrees in all source states. Every original alpha_i remains related to its original f_i; no binary factor is merged, pivoted or removed. If H>=h is a certified same-source affine upper bound, certificates H+kappa_i f_i<=0 also suffice, but the new target is R(h), not R(H).

The strict budget is a sufficient guard-preserving condition, not a necessary condition for every valid rewrite. Replacing it unconditionally with T<=1 is incorrect: h=-1, f1=f2=1, a1=a2=1/2 and kappa1=kappa2=1 give g=0 whereas h<0. Rectified values agree but the original target label beta=1 is lost. No special numeric repair is proposed.

This is a common negative-amplitude cone and a strict contraction of that amplitude at the target. It follows from elementary order/convexity; the vocabulary does not confer novelty. Its useful difference from the old omitted-remainder rule is the shared-source certificate and the joint, checkable budget.

## Constructing the best budget within an explicit source box family

For this subsection only, normalize the certified common source box to xi in [-1,1]^d, and assume fixed exact affine coefficients

```text
H = H0 + sum_j H_j xi_j,
f_i = f_i0 + sum_j F_ij xi_j,
B_i(kappa) = H0 + kappa f_i0 + sum_j abs(H_j+kappa F_ij).
```

This B_i is the exact support on that box, and a valid upper bound on S. It is not the exact support of an arbitrary constrained HZ. Assume each f_i has a positive box upper bound. Then B_i is convex piecewise linear and its eventual slope is f_i0+sum_j abs(F_ij)>0. Define

```text
C_i = { kappa>0 : B_i(kappa)<=0 }.
```

If C_i is nonempty, it has a finite attained maximum K_i>0. There exists a certificate of the displayed group family if and only if all C_i are nonempty and

```text
sum_i a_i/K_i < 1.
```

Necessity follows from kappa_i<=K_i and positivity of a_i. Sufficiency chooses kappa_i=K_i. This is completeness only for this fixed H, box and positive-multiplier family; failure is not proof that the semantic contraction is impossible.

Each K_i is constructible without a solver. Sort the positive breakpoints -H_j/F_ij for F_ij nonzero, merge coincident events, initialize B_i and its right derivative at zero, and scan its affine pieces. A derivative jumps by 2 abs(F_ij) at its event. The last crossing of level zero gives the right endpoint; the final positive slope guarantees finiteness. Reject an empty positive feasible set, including the case where only kappa=0 is feasible. No source-domain or phase split is performed: the scan is over a static certificate coefficient, not neural states.

For d_i the support union actually processed for group member i, the scan needs O(sum_i d_i log d_i) arithmetic/comparison operations and O(sum_i d_i) stored events if processed together. Shared-source matching, expansion of H, exact rational bit lengths, evidence, outward rounding for non-rational coefficients and reconstruction remain chargeable. This statement provides neither a GPU implementation nor a claim that arbitrary deep-source H is cheap to obtain.

## A noncollinear mixed residual control

Use x,y in [-1,1] and

```text
f1 = x+y/10,       f2 = x-y/10,
h  = -2x+y/20-3/10,
g  = h + R(f1)/2 + R(f2)/2,
r  = R(g).
```

The two child normals are independent, both child gates cross zero, the residual has mixed signs and nonzero bias, and the target crosses zero. For example g(-1/4,0)=1/5 and g(1/4,0)=-11/20. At (-3/20,0), both children are strictly inactive and g=h=0, retaining both target labels.

With both kappa_i=2, B1=-1/20, B2=-3/20 and T=1/2. The optimal right endpoints are

```text
K1=45/22, K2=47/22,
T_min=11/45+11/47=1012/2115 < 1.
```

Thus r=R(h) with the full original phase relation, although h has positive values. The old globally-negative omitted-remainder condition fails: at (-1/4,0), each remainder h+R(f_j)/2 equals 1/5>0. This separates the sufficient certificate families, not ordinary HZ with the same certificate.

Exact bounds are f_i in [-11/10,11/10], h in [-47/20,7/4], and g in [-27/20,7/4]. The latter follows because g's slopes in x are -2, -3/2 or -1; evaluate the two x endpoints and then the y endpoints.

For the isolated motif with no other raw child consumers, the standard four-row gates give the following symbolic cost. Source x,y and every original bit are retained.

| Item | Original motif | Contracted motif |
| --- | ---: | ---: |
| Continuous coordinates | 5 | 3 |
| Original binary coordinates | 3 | 3 |
| Gate inequalities and RHS entries | 12 | 8 |
| Predicate coefficient nnz | 34 | 22 |

Each original child contributes 10 nnz, the original target 14. Each retained child phase guard pair contributes 6, and the new target contributes 10. These counts exclude the original source system and are not native-HZ physical qualification. Continuous bounds change from ten to six endpoints; six binary-bound endpoints remain. Source, certificate, decoder, equality-format slack, backend conversion, host/device coexistence and terminal work must still be paid.

## Integer equivalence does not preserve the old LP

Use the exact bounds above and the four-row formulations. The retained-coordinate LP relaxations are incomparable.

New-only tuple: x=y=0, alpha1=alpha2=0, beta=4/5, r=1/10. New child guards are legal and the rewritten target upper row allows r<=17/100. The old child rows force q1=q2=0, so g=-3/10 and the old target upper row would require r<=-3/100. No old lift exists for this retained tuple.

Old-only tuple: x=y=0, alpha1=alpha2=1/2, q1=q2=11/20, beta=1, r=1/4. Every old gate row holds and g=1/4. The new h=-3/10 is incompatible with beta=1. These are fractional diagnostic tuples including original bits, not a proof of output-only projection incomparability and not network adversarial examples.

Consequently neither the exact integer theorem nor the smaller row count proves old CERT retention or faster MILP. There is no authorized fallback menu, changed score or default activation. Keeping the old lift as well would require charging it and would abandon the displayed deletion count.

## Single edge discovery and simultaneous deletion boundaries

Another sufficient rule uses the complete original remainder h_i=g-a_i R(f_i). For a_i>0, a certificate h_i+kappa_i f_i<=0 with kappa_i>a_i permits deleting that edge. When f_i>0 both h_i and g are strictly negative; otherwise g=h_i. If every selected positive edge has such a certificate against the original g, all can be deleted simultaneously: any positive selected q_i forces g<0, and deleting positive contributions can only decrease it.

For affine h_i on the same box, again require f_i to have a strictly positive box upper bound. Under that explicit premise, existence in this one-parameter family is decidable by the same breakpoint sweep on kappa>a_i. If B(a_i)<0, set L=abs(f_i0)+sum abs(F_ij), delta=-B(a_i)/(2(L+1)), and use kappa=a_i+delta. Lipschitz continuity gives B(kappa)<0. Otherwise a feasible point exists only if a breakpoint strictly greater than a_i has B<=0, since the eventual slope is positive. This is a constructive family-completeness result for the stated scope, not complete contextual-equivalence detection.

Independent checks against a common h cannot replace the group budget. For f1=x+y/10, f2=x-y/10, h=-3x/2-3/20, a1=a2=1 and kappa1=kappa2=3/2, both h+kappa_i f_i<=0 hold on the box, but g(1/2,0)=1/10>0 while h=-9/10. Here T=4/3, and the group rule correctly rejects.

Negative edges cannot be simultaneously deleted using independent original-remainder certificates. A fully ordinary counterexample is

```text
q0=R(-x), q1=R(x+y/10), q2=R(x-y/10),
h=(7/4)q0+x-9/16,
g=h-q1-q2,               x,y in [-1,1].
```

For each negative edge, kappa=3/8 gives h-q_j+kappa f_i<=-1/20, using R(x+/-y/10)>=R(x)-abs(y)/10. Hence either single deletion preserves its full target relation. But at (3/4,0), g=-21/16 and h=3/16; deleting both changes the output. At (-7/8,0), g=3/32>0, so the target is not a dead gate. All source gates cross zero. This closes an invalid mixed-sign generalization, not an extreme-case optimization project.

A safe mixed-sign extension can instead use one upper envelope. For the whole selected set, let U=h+sum_i max(a_i,0)q_i. If every selected i has a certificate

```text
U-max(a_i,0)q_i+kappa_i f_i <= 0,
kappa_i > max(a_i,0),
```

then every subset of selected edges can be deleted safely. Whenever any selected q_i>0, U<=-(kappa_i-max(a_i,0))f_i<0, and every deletion version is <=U. When all selected q_i=0, every version equals h. This also works with a certified upper bound on h in U, without authorizing deletion of unselected terms in the actual network. Discovery of these generally nonlinear envelope certificates is not supplied by the affine-box sweep automatically.

An ordinary mixed-sign positive control is f1=x+y/10, f2=x-y/10, h=-2x-3/4, g=h+q1-q2/2 on the same box. Take U=h+q1, kappa1=2 and kappa2=3/2. The first certificate is -3/4+y/5<=-11/20; the second is abs(x+y/10)/2-y/10-3/4<=-1/10. All three gates cross zero and every selected subset deletion is safe. This is one mathematical sufficient-rule extension, not a second runtime path or a new-domain claim.

## Composition and research attribution

Certified replacements preserve subsequent Affine/Conv/ReLU/Add/Concat on the same frame because their inputs agree exactly. This does not make unequal negative preactivations equivalent under arbitrary raw affine contexts. Every raw consumer, residual, side predicate and output involving a retired child or old g must remain representable and be checked. A child amplitude can be removed only after all consumers allow retirement; its original sign guards and bit remain. Hidden reconstruction evaluates R(f_i) from the retained source. No constant-affine decoder for retired values is assumed.

This adds a constructive, group-level sufficient certificate to D005, not a new abstract-domain definition. D014/D039 already contain conditional amplitude and perspective mechanisms. Botoeva et al. study [neural dependency analysis](https://ojs.aaai.org/index.php/AAAI/article/view/5729/5585), including consecutive-layer bounds in Section 3; their branching/splitting machinery is not imported. Giving ordinary HZ the same certified rewrite gives the same result. A limited archive/source review does not establish external novelty of the group theorem or coefficient scan.

The actionable result is therefore narrower and honest: a source-dependent group contraction can pass a symbolic row/nnz reduction where independent omitted-term bounds fail, but can weaken the terminal relaxation. Do not select engineering or a new test framework solely from this toy count. A main-domain contribution still needs a paid compositional representation and real-network evidence under the original gates.

Provenance: 2026-10-02 Australia/Sydney; redu-hz; HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac. Configuration paper mathematics and read-only source/primary literature. Formal 1870/2413 and independent E0 61/400, both gains zero. No candidate import, numerical execution, model run, new solver, GPU, replay or production edit.
