# Cross-domain literature review for definition-first Neural-HZ

Date: 2026-09-29. Branch: `redu-hz`.
Commit: `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac` (pre-existing dirty worktree).
Mode/configuration: primary-source literature review and paper derivations only;
no implementation, network execution, solver experiment, default change, or promotion.
This is a bounded mechanism-oriented review, not an exhaustive novelty search.

## Decision

Prioritize **shared-source relations across unstable nonlinearities**, drawing
from static analysis, control theory, and mathematical programming. Do not limit
the search to HZ papers or continue treating convolution storage reduction as
the main domain contribution. The candidate direction is a compact nonconvex
representation of source-coupled phase/amplitude relations with compositional
forward operators. It is a research hypothesis, not an established new domain.

Earlier D007/D009/D012/D013 notes already compared abs-normal forms, PRIMA,
reduced products and lifting. This review consolidates those connections and
adds concrete comparators and rejection criteria; it does not claim these
connections were never considered before.

## Local evidence and provenance

Authority remains the existing
[definition-first amendment](../GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md)
(SHA256 `0fd93920413f63ad6b3b65849ea4d857c416c6bf4ab56263c63ff73416ebe80c`).
Local evidence is the frozen
[D015 v2 checkpoint](../definition_first_20260928/CHECKPOINT_D015_V2_20260928.md)
(SHA256 `cb07722c5be727ad99725149c246c37c407c3ad17431b140c5dd283d56086003`).

D015's entire three-model pilot did not qualify: it stopped at its work cap.
In its completed CIFAR-large local portion, all 149 rows containing extracted
negative terms were already inactive under that diagnostic's ordinary interval
bound. Neither of its two interval-crossing targets benefited; both retained a
baseline bound crossing zero. This is not a comparison with a rerun of ACT's
historical native bounds. Medium results were incomplete; Tiny was not reached.
The evidence motivates investigating baseline/amplitude correlation, but proves
neither its prevalence on the full population nor future property gain.

Formal 1870/2413 and independent external CIFAR25 + Tiny36 = 61/400 remain
unchanged. This review provides no new CERT or validated ADV.

## Primary-source mechanism map

### 1. ImageStars: shared parameters survive affine image operators

Tran, Bak, Xiang, Johnson, *Verification of Deep Convolutional Neural Networks
Using ImageStars*, CAV 2020, §§3–4.1.
[Author paper](https://assured-autonomy.isis.vanderbilt.edu/files/TranCAV2020ImageStars.pdf).

Image-valued generators share predicate variables; convolution transforms the
anchor and generators without changing those predicates. This supports source
alignment before bounding and tensor-native affine execution. A single ImageStar
is convex, and exact nonlinear reachability uses multiple stars. Neither a convex
replacement nor that splitting procedure is our route. Tensorization and shared
parameters alone are established mechanisms, not our definition-level novelty.

### 2. DeepPoly: relational expressions before scalar bounds

Singh, Gehr, Püschel, Vechev, *An Abstract Domain for Certifying Neural Networks*,
POPL 2019 / PACMPL 3, article 41, §§2–4.
[Author paper](https://ggndpsngh.github.io/files/DeepPoly.pdf).

Symbolic lower/upper expressions retain dependencies lost by separate intervals.
The lesson is to combine coefficients in a common source frame before bounding.
Its backsubstitution and refinement procedures are not imported into this goal.
One recovered affine cancellation is a known transfer improvement, not a new
nonconvex domain. Any forward relational substitute must account for expression
growth as well as precision.

### 3. k-ReLU and PRIMA: relationships between nonlinearities

Singh, Ganvir, Püschel, Vechev, *Beyond the Single Neuron Convex Barrier for
Neural Network Certification*, NeurIPS 2019, §1 and construction sections.
[Official paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/0a9fdbb17feb6ccb7ec405cfb85222c4-Paper.pdf).

Joint input/output constraints capture more than per-neuron triangles or pairwise
incompatibility. A useful known control follows from
`|h1| + |h2| = max(|h1+h2|, |h1-h2|)`: certified shared-source bounds of M on both
right-hand expressions imply `2(r1+r2)-h1-h2 <= M`. This identity needs no runtime
phase enumeration. General k-ReLU construction does enumerate phase regions;
that procedure is not adopted.

Müller, Makarchuk, Singh, Püschel, Vechev, *PRIMA: General and Precise Neural
Network Certification via Scalable Convex Hull Approximations*, POPL 2022 /
PACMPL 6, article 43, §§3, 5, 7.5.
[Author paper](https://ggndpsngh.github.io/files/PRIMA.pdf).

Overlapping small neuron groups offer a useful precision/cost organization.
However, Split-Bound-Lift explicitly constructs regions along activation
hyperplanes; this is not merely a harmless name containing “split.” We borrow
the question of which joint relations to retain, not its general region-building
algorithm. Adding existing multi-neuron inequalities to HZ is a required
comparator, not sufficient evidence of a new abstract domain.

### 4. Control theory and complementarity: source-coupled nonlinear relations

Aydinoglu et al., *Stability Analysis of Complementarity Systems with Neural
Network Controllers*, 2020 preprint / 2021 publication, §3.1, Lemma 2.
[Original paper](https://arxiv.org/pdf/2011.07626).

ReLU networks admit exact complementary-variable encodings with layer-triangular
structure. This encourages keeping amplitude/source relations rather than
independent amplitude boxes. ReLU-to-complementarity is already known; changing
names is not novelty. Complementarity alone also does not retain explicit
original binary identities at zero. Keep our original guards and bits; import
no LCP/PATH/LMI rescue solver.

Zhang and Khan, *New complementarity formulations for root-finding and
optimization of piecewise-affine functions in abs-normal form*, 2025 preprint,
§§2–3. [Original paper](https://arxiv.org/html/2501.18503v1).

Abs-normal structure keeps the direct affine source and nested absolute-value
terms together. It is a useful reference representation, not a novel Neural-HZ
definition. Algebraic elimination may fill in matrices; a compact formula does
not establish cheap terminal lowering. The paper's root-finding/optimization
algorithms are outside this proposal.

Fazlyab, Morari, Pappas, *Safety Verification and Robustness Analysis of Neural
Networks via Quadratic Constraints and Semidefinite Programming*, TAC 2022
(2019 preprint), §III-C(3), equations (15)–(17).
[Author paper](https://www.georgejpappas.org/wp-content/uploads/2022/01/Safety_Verification_and_Robustness_Analysis_of_Neural_Networks_via_Quadratic_Constraints_and_Semidefinite_Programming.pdf);
[preprint](https://arxiv.org/abs/1903.01287).

Repeated-activation slope relations connect different neurons even when both
cross zero. For ReLU, writing d = r_i-r_j and delta = g_i-g_j gives
`d(d-delta) <= 0`. A certified nonnegative delta yields the linear consequence
`0 <= d <= delta`. This known fact is a control, not our contribution. No SDP
terminal, all-pairs relation explosion, or deletion of original phases is
authorized by it. The scalar consequence does not require choosing an SDP
multiplier family.

### 5. Mathematical programming: grouping sums is not splitting the input set

Tsay, Kronqvist, Thebelt, Misener, *Partition-Based Formulations for Mixed-Integer
Optimization of Trained ReLU Neural Networks*, NeurIPS 2021, §3.
[Official paper](https://papers.nips.cc/paper/2021/file/17f98ddf040204eda0af36a108cbdea4-Paper.pdf).

Here “partition” groups summands feeding a neuron. Shared-phase lifted variables
trade formulation size for relaxation strength; singleton groups recover the
node hull over input bounds. Algebraic lifting into one mixed-integer formula is
not input-region or phase-subproblem search. Its auxiliary variables and rows
must nevertheless be charged. We do not import its OBBT, callbacks or search
workflow. Directly adopting the known formulation is not domain novelty.

Anderson, Huchette, Ma, Tjandraatmadja, Vielma, *Strong mixed-integer programming
formulations for trained neural networks*, Mathematical Programming 183 (2020),
3–39. [Author/institution publication](https://research.google/pubs/strong-mixed-integer-programming-formulations-for-trained-neural-networks/).

Strong neural MIP formulations are essential prior art when judging whether a
proposed lift or predicate is new. Bibliographic/abstract verification only in
this review; exact relevant formulation comparison is a remaining task before
any novelty claim. No new cutting-plane or separation loop is proposed here.

### 6. Static analysis: specialized relations and bounded variable packs

Blanchet et al., *A Static Analyzer for Large Safety-Critical Software*, PLDI
2003, §§3.1, 6.2.3–6.2.4, 7.2.1–7.2.2.
[Original paper](https://arxiv.org/pdf/cs/0701193).

Reduced products let specialized components exchange information. Syntactically
chosen overlapping variable packs limit relational scope. For us, shared sources,
receptive fields and residual joins can define packs uniformly before analysis.
Do not copy its optional selection from previously useful packs: our current
rules prohibit historical outcome menus. A relational cache attached to old HZ
is not automatically a new domain; its language and neural reductions need a
substantive precision/cost theorem.

Rival and Mauborgne, *The Trace Partitioning Abstract Domain*, TOPLAS 2007,
§§3–5. [Author paper](https://software.imdea.org/~mauborgn/publi/toplas29.pdf).

Useful diagnostic: losing the link between conditions and numerical values can
destroy precision even if each is separately represented. Do not adopt the
partitioned analysis algorithm. A single symbolic guarded relation is admissible;
enumerating phases and calling each branch a “factor” is not.

### 7. Elimination and compilation: honest cost and certified rewrites

Dechter, *Bucket Elimination: A Unifying Framework for Reasoning*, Artificial
Intelligence 1999, §§2.1–2.4.
[Author paper](https://ics.uci.edu/~csp/r48b.pdf).

Study interfaces and fill before eliminating private continuous variables.
Crucially, §2.3 explicitly excludes Fourier elimination from a complexity bound
depending exponentially only on induced width: inequality counts are another
source of growth. Local convolution does not imply cheap global exact inference.
Conditioning creates subproblems and is not our permitted route. Count interface
dimension, retained bits, rows/nnz, coefficient size and reconstruction together.

Willsey et al., *egg: Fast and Extensible Equality Saturation*, POPL 2021.
[Original paper](https://arxiv.org/abs/2004.03082).

E-class analyses suggest organizing side conditions for equivalence-preserving
rewrites. This is supporting infrastructure, not the new abstract domain.
Our equivalence must preserve shared sources and all original phase choices,
not merely a floating-point output expression. Extraction must include sharing,
predicates and terminal cost, not only count graph nodes. No implementation of
equality saturation is proposed in this review.

## A concrete cross-zero control (paper derivation, not an experiment)

Let x be in [-1,1], g1 = x+1/4, g2 = x-1/4, and r_i = ReLU(g_i).
Both gates cross zero, but monotonicity and the unit slope bound imply
`0 <= r1-r2 <= 1/2`. This is an elementary instance of the known repeated-ReLU
relation above, not a new theorem.

At x=0, the per-gate big-M continuous relaxation permits
`r1=1/4, r2=3/8, beta1=beta2=1/2` using exact intervals
`g1 in [-3/4,5/4]`, `g2 in [-5/4,3/4]`.
The second output reaches its allowed upper bound 3/8, while the first also
satisfies its four inequalities. This relaxed point violates r1 >= r2.
The actual outputs are (1/4,0). Thus cross-zero relational precision is a real
question even without a stable gate. No network prevalence is inferred.

Do not generally infer beta_i >= beta_j from g_i >= g_j: when both are zero,
that extra condition could discard legal original binary choices. Output
relations and binary-identity preservation are separate obligations.

## Candidate definition question and novelty boundary

Research hypothesis: ordinary Conv–ReLU and residual blocks may admit enough
small, shared-source phase/amplitude relations to support a useful nonconvex
Neural-HZ representation without enumerating phases.

The proposed primitive to investigate is a **source-coupled relational block**:
one shared latent frame, all original binary guards, bounded-size joint amplitude
relations, and explicit external interfaces. This phrase is a design question,
not a completed definition. Relation scope is determined by static structure;
coefficient-based validity checks certify each rule, not choose model identities.
The baseline affine term must participate in the relation instead of being
bounded independently of every amplitude.

Required work before implementation:

1. Define elements, concretization, refinement/order, and embedding of original
   HZ, including every original bit choice at zero and exact input reconstruction.
2. Give forward Affine/Conv, ReLU, Add and Concat operators. Shared-source branches
   must remain aligned. Distinguish exact normalization from sound relaxation.
   If using new primitive notation, provide faithful EQ/LE mixed-binary lowering.
3. Prove one useful composition result across a normal multi-operator block, not
   only one tight inequality. Quantify construction, representation, lowering,
   query and witness costs, including fill and auxiliary variables.
4. Compare ordinary HZ, HZ plus the same known relations, and the proposed domain
   under equal budgets. Add relevant group-lift/strong-formulation controls.
   If the candidate is merely an existing formulation renamed, report that.

Exact HZ can already encode these piecewise-affine reachable sets. Therefore the
claim cannot be “a truer reachable set than exact HZ.” A meaningful contribution
would instead establish neural-specific compositional invariants and a better
precision/cost trade-off for the available queries/relaxations, or an exact
structural simplification unavailable at the same cost in the controls.

Two hypotheses merit paper comparison, not a runtime menu: (A) source-difference
relations preserved through local nonlinear blocks; (B) shared-phase source-sum
lifting with a stronger compositional compression theorem than known grouping.
Select at most one candidate after proof/cost comparison. The first is currently
the priority because it directly challenges D015's independent-bound obstacle.

## Next bounded evidence and rejection criteria

First use ordinary mixed-sign shared-input examples with both activations
unstable and the varying baseline crossing zero. Avoid accumulating already
stable rows as capability evidence. Paper counterexamples should catch lost
shared-source cancellation and the false inference from local to global
consistency; they are basic correctness controls, not an extreme-case research
detour.

Only after the definition/theorem comparison should a new preregistered pilot
measure structural prevalence on original sources. Freeze structural pairing or
grouping, degree/support limits and complete cost accounting before execution;
do not select by instance ID, known verdict, margin or terminal LP status. The
incomplete D015 run cannot supply a qualified prevalence denominator.

Reject or reclassify a candidate if it only rediscovers stable pruning, equals
HZ plus known predicates, needs all-pairs explosion, loses phase/source identity,
relies on splitting/backward/dual rescue, or pays more full terminal cost than
its measured benefit. These criteria prevent prolonged implementation of a
known reformulation; they do not assert that relational research is exhausted.

No restrictions are relaxed by this review. Continuous-factor semantics, shared
source identities, all original binary choices, EQ/LE, fail-closed behavior and
validated-witness obligations remain. Exact elimination of private continuous
auxiliaries still requires its equivalence and reconstruction proof. Any
implementation remains isolated and opt-in; mathematical tests,
real-structure tests, shadow replays and the complete 2413-case retention/promotion
gate remain required. The independent 400-case external ledger stays separate.
The 1870 baseline and every family's existing successes cannot be traded away.

## Archive boundary

Only this newly created review directory is written for this task. Production
code, old mathematical notes, frozen models/specifications, logs and score tables
are not changed. No new numerical worker is launched, and no formal result is
claimed. Sources were reviewed for the mechanisms stated above, with the limited
Anderson-paper reading scope explicitly identified. A full novelty survey and
the proposed domain proofs remain open work.
