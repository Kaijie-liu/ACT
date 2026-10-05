# Ordered probability flow definition and attribution control

This default-off experiment asks whether a shared cross-query probability relation has a compositional advantage over explicit known relations. It is a definition candidate and a falsifiable comparison, not a claimed new Neural-HZ domain. The preceding user-facing turn was a status audit with no research execution. The strongest registered control below can refute the novelty interpretation even when the candidate tightens the production component substantially.

## Carrier and concrete meaning

Write a relation-bearing element as D=(H,R,L,decoder). H retains the complete original HZ continuous factors, signed binary factors, EQ/LE predicates and shared source identity. R contains semantic relations between readouts of that one assignment. L is the joint output readout, and decoder retains the original concrete-input reconstruction. Its exact concrete meaning is the set of L(w,u) for original assignments w satisfying H and auxiliaries u satisfying R. Semantic inclusion is the proposed order; no computable best abstraction, complete lattice, or novel order structure is claimed. Empty R and the original affine readout embed H exactly.

For the relation studied here, the caller must bind actual p=softmax(s), q=softmax(t), a common temperature and token support, and the same value bank V at the same original input. The exact relation includes F_k=sum_{i<=k}(q_i-p_i) and product coordinates T_kr=F_k*(V_{k+1,r}-V_{k,r}). The implementation lowers these products to sound McCormick envelopes, not exact nonlinear equations. Therefore its executable concretization is an outer approximation of the semantic relation, not an exact encoding of Softmax in finite HZ. Matching integer frame labels is not proof of native model binding.

Original nonconvex phase semantics are retained, not replaced with continuous phase factors. Empty R preserves every original HZ, including nonconvex examples; this does not mean every possible element is nonconvex. Appending valid neural relations can remove spurious points of an existing relaxation, so it is not generally an exact simplification of that relaxation. The soundness claim concerns retention of actual common-source network states.

Affine and convolution readouts can compose exactly with the joint L while retaining the same relation assignment. Add and Concat are exact only for already certified shared assignments, not by equating unrelated frame labels. Standard bounded ReLU graph extension can retain original signed phases and add new ones, but this prototype does not implement or qualify that native transformer. Smooth relation closure, full QK source binding, native floating lowering and concrete witness admission remain unproved here. Merely packaging H with R is also possible in existing mixed-symbolic and lifted relation representations; it is not itself the sought innovation.

## Ordered flow soundness

Let d=s-t and certify one order d_1<=...<=d_N throughout the entire retained source domain. At each actual state p_i/q_i=exp(d_i)/sum_j(q_j exp(d_j)). Thus p_i*q_j<=q_i*p_j for i in any prefix and j outside it. Summing gives P_A*(1-Q_A)<=Q_A*(1-P_A), so F_k=Q_A-P_A>=0. This is a finite instance of the known likelihood-ratio-order implication to stochastic order, not a new probability theorem; see Theorem 1 and Remark 4 of [Dumbgen and Mosching](https://arxiv.org/pdf/2209.07868).

With F_0=F_N=0, z_i=p_i-q_i=F_{i-1}-F_i. Telescoping gives DeltaY_r=sum_{k=1}^{N-1} F_k*(V_{k+1,r}-V_{k,r}). This identity holds for dynamic V and for all channels simultaneously at a common assignment. A shared value translation cancels. Neither telescoping nor shared prefix coordinates are claimed novel; related prefix/Abel ideas are already in the project archives.

The compiler proposes a deterministic order from the centers of certified affine boxes, then certifies every adjacent d_i-d_j upper bound is nonpositive using all shared coefficients. Center sorting alone never admits the relation. If any whole-source comparison fails, it rejects without input/phase splitting, sampling, a solver call, or an identity-based alternative path. Equal score changes may be ordered arbitrarily. Probability boxes are caller-certified positive bounds.

For each prefix, the installed flow upper bound is the minimum of 1, the sum of prefix (q_upper-p_lower), and the sum of complementary (p_upper-q_lower). These follow respectively from simplex mass, prefix boxes and the complementary mass identity. Adjacent conservation equalities and two simplex equalities connect the flows to the original probabilities. All old columns, bit indices and EQ/LE rows remain literal prefixes. Actual products extend any real network state to every McCormick row, establishing conditional soundness. The original input decoder needs no inverse flow elimination; a relaxed terminal assignment still requires independent concrete-network validation before it can be an ADV.

## What the relation does not replace

The exact rational point d=(-3/4,-1/4,1/4,3/4), q=(1/4,1/4,1/4,1/4), p=(19,22,18,21)/80, c=0 and lambda_i=1/8 satisfies the specified preceding endpoint relaxation with probability/lambda boxes [1/512,511/512]. It also has d dot z=1/160 and squared norm z=1/640, satisfying the listed cocoercivity and coordinate increment conditions. But F_2=-1/80, so ordered flow excludes it. This is a non-implication example for those finite relaxation rows, not a true network counterexample or a separation from every valid IQC, interpolation method or Taylor relaxation.

Conversely, nonnegative flow alone loses magnitude information: with two tokens d=(0,epsilon), q=(2/3,1/3), p=(1/3,2/3), the prefix flow is nonnegative independently of epsilon. The endpoint relaxation with lambda upper 2/3 and c in [0,epsilon] forces z_2<=epsilon/3 and excludes this point when epsilon<1. Therefore the new block retains the entire preceding endpoint reference instead of removing its auxiliaries for a misleading size saving.

Common positive denominator lifting is not pursued as an alternative new candidate: the project already recorded it in the [radial prior-art analysis](../d104_radial_source_relations_20261002/PRIOR_ART_AND_DECISION.md). Two queries generally have different normalizers even when their exponentials share a source; a single Charnes-Cooper change does not make both readouts jointly affine. Product predicates also have established representational precedents such as [constrained polynomial zonotopes](https://arxiv.org/abs/2005.08849). These observations reject a naming-based novelty claim; they do not authorize replacing HZ with a polynomial or convex domain.

## Fixed production control and strong reference

Use four logits t=x with x in [-9/4,9/4]^4, s=x+d for the fixed centered d above, and the common dynamic value bank V=(v,v,0,0), v in [1,2]. The source contains five independent continuous coordinates, eight probability coordinates and two fused outputs, initially 15 columns. No trained model is executed. The unchanged public sparse_hz_softmax_value_relaxation runs, including its existing internal LP, alongside original probability birth, simplex and ratio constraints. The same explicit probability/value bindings are added to every arm; A is not the bare production verifier.

Five systems maximize the same DeltaY objective:

| Arm | Retained mathematical information |
| --- | --- |
| A | Complete selected production wrapper and common probability/value bindings |
| B | A plus the frozen endpoint block and fixed known incremental sector rows |
| B_prefix | B plus three known prefix inequalities on the original probabilities |
| B_group | B_prefix plus the known grouped-value identity and its single product envelope |
| C | B plus the shared ordered-flow block |

For the sector reference, let mu be the minimum endpoint probability lower bound and L=1/2. Concavity of each log-softmax coordinate implies the probability on the line segment between s and t is at least mu; this segment need not itself be realizable as a network input path. On the zero-sum subspace, the Jacobian obeys mu*I<=J<=L*I: its quadratic form is a weighted variance, bounded below by mu times the unweighted centered norm, and the upper eigenvalue bound follows from its row absolute sum. Averaging along the segment gives z=J_average*d. Since the fixed d is centered, the necessary sector inequality is norm(z)^2-(L+mu)*d dot z+mu*L*norm(d)^2<=0. For noncentered d the projected norm is required; this fixture formula is not a generic compiler.

Replace norm(z)^2 by each supporting plane 2*a dot z-norm(a)^2 for the 81 fixed a in {-1/4,0,1/4}^4. Add d dot z>=0 and the eight coordinate increment rows abs(z_i)<=span(d)/4. These are fixed necessary linear cuts, not an exact IQC oracle, numerical search, nonlinear solver, or input/phase partition. They are not selected from any terminal LP result.

The strongest attribution reference uses delta=z_1+z_2<=0. For this value bank, the elementary grouping identity is DeltaY=delta*v. Its lower bound is minus the same F_2 upper bound used by C. A single McCormick product on delta in [-U_F,0] and v in [1,2] proves DeltaY<=0. C gives precisely DeltaY=-F_2*v; the product envelopes match under F_2=-delta. The other flows have zero value differences and only reproduce existing prefix constraints and box-derived bounds. Thus C and B_group are expected to have the same projection onto original coordinates on this fixture. A benefit over B_prefix alone cannot establish a novel domain contribution. A difference exceeding numerical tolerance against B_group demands diagnosis, not a novelty announcement.

## Full costs and unqualified stages

For N tokens and C channels the literal compiler adds at most (N-1)*(1+C) continuous columns, N+1+C EQ rows, and 4N+8*(N-1)*C LE rows. Point products reduce the actual counts. Adjacent flow equalities have 4N-5 coefficient occurrences when p and q are explicit columns; substituting every prefix could instead produce quadratic support. Ordering requires O(N log N) comparisons plus exact source-wide difference bounds. Source gathering, prior endpoint rows, base value products and all reference products remain part of the bill.

The retained-entry preflight bounds possible row support by the complete final column count and performs a final exact count. It does not certify transient Python allocations, Fraction object storage, GPU transfer/rounding, terminal copies, solver work or full physical memory. The original 512-bit rational and 64M retained-entry bounds remain; whole work 256M, branch work 200M and evidence 40M still require later full qualification. No reduction from NC to (N-1)C is claimed against the already centered preceding implementation.

Prefix scans and channel products suggest a parallel GPU layout, but this Fraction prototype is CPU only. No GPU speed, native precision, full network binding, model coverage or full resource qualification follows from this comparison. All five terminal LP optima are floating diagnostics, not exact optimality certificates.

## Provenance and accounting

Date 2026-10-02 Australia/Sydney; branch redu-hz; commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac; tracked binary diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5. All additions are in this isolated experiment tree. Production, frozen sources, historical models and /data1/Kane/HyZor are not write targets.

Formal 1870/2413 and independent CIFAR100 25 plus TinyImageNet 36, 61/400, cannot change here. Even a component pass does not establish 13-family zero regression or enable a default. The complete definition-first Goal remains active; this experiment is allowed to close a hypothesis, not to redefine success around one positive bound.
