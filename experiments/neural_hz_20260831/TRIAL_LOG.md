# Neural-HZ trial log

This is an append-only narrative for the isolated Neural-HZ experiments. Raw
records live in `results/` and carry the branch, base commit, candidate source
hash, timeout, and memory cap. Formal 13-family accounting remains unchanged at
1,870 / 2,413.

## 2026-08-30: baseline infrastructure, not the paper contribution

The updated mainline had lost exact unused-factor pruning and exact
parallel/antiparallel predicate-row coalescing that were already present in the
frozen 13-family implementation. They were restored and tested, but are treated
as baseline recovery rather than Neural-HZ novelty.

On VNNLIB 2.0 ViT iid 101 (`ibp_3_3_8_9191`), final-state pruning plus release
of 126 intermediate sparse HZ states changed a 16 GB `MemoryError` into
`CERTIFIED`: 10,802 to 5,400 continuous factors, all 40 binary factors retained,
and 18,112 to 399 predicate rows. See
`docs/HYBRIDZ_IMAGESTAR_SIMPLIFICATION_TRIAL_20260830.md` for the full record.

An optional softmax numerical fallback was also tried. It improved one PGD
failure mode but regressed the retained IBP certificate to `UNKNOWN`; the code
and its test were rolled back completely. This is a rejected attempt, not part
of the candidate.

## 2026-08-31: Trial 1 — fill-reducing inactive-factor projection

### Hypothesis and exactness boundary

Target repeated structure: after projected neural nonlinearities, a continuous
factor no longer occurs in the value basis and has a unique defining equality
`a*x + q(u) = b`. Eliminate `x`, substitute it in other predicates, and replace
its latent box by the equivalent predicate
`b-|a| <= q(u) <= b+|a|`.

Acceptance rules:

- constraint nnz must strictly decrease;
- wide/fill-increasing substitutions are rejected;
- no binary factor is pivoted or removed;
- eliminated factors are reconstructed in reverse pivot order for witnesses;
- the feature is opt-in and disabled by default.

The initial implementation passed 15 focused tests covering chained pivots,
binary-preserving projection, input reconstruction, fill rejection, exact row
coalescing, unused-factor pruning, and cache lifetime.

### Pilot v1 (`candidate_sha256=6fae0d3fc498...`)

Paired 15-second, 16 GB SAT-ReLU shadows:

| iid | baseline status | Neural status | continuous factors | solver stage baseline | solver stage Neural | formal gain |
|---:|---|---|---:|---:|---:|---:|
| 0 | UNKNOWN | UNKNOWN | 166 -> 136 (-30) | 0.0273 s | 0.0419 s | 0 |
| 16 | UNKNOWN | UNKNOWN | 659 -> 608 (-51) | 0.3212 s | 0.2302 s | 0 |
| 90 | UNKNOWN | UNKNOWN | 656 -> 592 (-64) | 0.2601 s | 0.2264 s | 0 |

All binary counts were identical (68, 304, and 296 respectively). The structure
is real and repeats, but the column-wise prototype penalized the small case.

### Batched leaf projection v2 (`candidate_sha256=8157d5c50299...`)

Independent leaf factors were changed from repeated whole-matrix rebuilds to a
single sparse transaction. The factor reductions and all verdicts were
unchanged. Overall solver-stage measurements were 0.0451 s, 0.2401 s, and
0.2309 s for iid 0, 16, and 90. The remaining small-case cost motivated a
uniform workload gate; this gate is data-informed after the pilot and is not
claimed as pre-registered.

### Cost-gated v3 (`candidate_sha256=f126e5bb9591...`)

The projection now runs only when `predicate_rows + latent_variables >= 512`.
This is one structural cost rule, not an instance list. Focused tests increased
to 16/16.

- SAT-ReLU iid 0: the gate uniformly skipped projection (`projected_cont=0`),
  preserving `UNKNOWN`; concurrent timings are noisy and are not used as a
  speed claim.
- ACAS Xu iid 100: no matching factor was found (`projected_cont=0`); both arms
  remained `UNKNOWN` at the 15-second limit.
- ACAS Xu iid 102: no matching factor was found; both arms retained `CERTIFIED`
  in about 11.1 seconds.

### Final shadow cohort and decision

SafeNLP iid 21 did not contain the target structure (`projected_cont=0`). Both
arms retained `CERTIFIED` at essentially the same 15-second boundary: 14.7003 s
baseline versus 14.7054 s projection.

The SAT-ReLU retained-CERT cohort did contain the structure. Each projection arm
removed exactly 64 continuous factors and preserved `CERTIFIED`, but every one
was slower in the solver stage:

| iid | continuous factors | baseline | projection | verdict regression |
|---:|---:|---:|---:|---:|
| 3 | 569 -> 505 | 0.6269 s | 1.4503 s | 0 |
| 5 | 707 -> 643 | 1.1312 s | 1.9272 s | 0 |
| 91 | 660 -> 596 | 1.0365 s | 1.7299 s | 0 |
| 97 | 669 -> 605 | 0.6400 s | 0.9038 s | 0 |

A separate sequential rerun of iid 5 confirmed that this was not merely
concurrent-run noise: 1.0929 s baseline versus 1.8837 s projection, with both
arms `CERTIFIED`.

Trial 1 is therefore closed as a standalone promotion candidate. It establishes
an exact, repeating Neural-HZ compiler transform and helped two feasible/UNKNOWN
SAT-ReLU pilots, but it penalizes the retained infeasible/CERT proof cohort and
has formal gain 0. The code remains default-off for composition/ablation; it is
not part of the 1,870 baseline and will not enter a family replay by itself.

## 2026-08-31: Trial 2 — predicate-implied binary phase fixing

### Pre-registration and exactness boundary

Target repeated structure: a binary ReLU phase remains syntactically live in
the final HZ, while the accumulated forward predicate system already implies
one phase. For each binary value, independently relax all other continuous
latents to `[-1,1]` and all other lowered binaries to `[0,1]`. A value is ruled
out only if this superset cannot intersect an existing predicate row. If exactly
one value survives, substitute it once and continue so consequences may cascade.

Rules fixed before real-network inspection:

- the row enclosure includes a sparse-dot rounding-error pad and outward
  `nextafter` endpoints;
- binaries for which both values survive remain nonconvex and unchanged;
- fixed original latent IDs are retained for concrete witness reconstruction;
- no attack, branching, split, backward/dual pass, or solver-status rescue;
- the existing uniform `rows + variables >= 512` cost gate applies;
- maximum 128 cascading fixes per final HZ solve;
- opt-in only; default and formal baseline remain unchanged.

The first implementation passes 22/22 focused Neural-HZ/baseline tests. New
tests cover z=0 and z=1 implications, cascading implications, preservation when
both phases survive, explicit integer-empty representation with fail-closed
verdict semantics, output substitution, and reconstruction of a fixed input
binary. This is the mathematical gate only, not a performance or score claim.

Pre-registered structural census: paired baseline/phase shadows on binary-heavy
ACAS Xu, ReluSplitter, and TLL instances. A real fix count is required before a
larger cohort; verdict regression closes the trial immediately. Formal score
remains 1,870 / 2,413 until every later gate passes.

### Structural census and decision

The full-budget pilot covered ACAS Xu iid 100/102 and TLL iid 0/2. It preserved
the sampled CERT/UNKNOWN verdicts but fixed no binary. A 1 ms solve census then
sampled eight additional ACAS Xu structures and the remaining two N=8 TLL
structures. Across 10 ACAS Xu plus all four N=8 TLL final HZs, 2,895 binary
factors were inspected and `fixed_bin=0` throughout. ReluSplitter iid 2 lost
the HZ before final lowering, so this pass could not apply.

Trial 2 is closed as a structural miss. The necessary-condition rule and its
tests remain default-off as a documented exact negative result; no family
replay was launched and formal gain is 0.

## 2026-08-31: Trial 3 — fill-aware exact ReLU graph quotient

### Pre-registration and exactness boundary

The extended exact ReLU graph retains two continuous auxiliaries, one binary,
one linking equality, and two inequalities. Algebraically projecting the
predicate-only branch auxiliary gives the equivalent graph

`0 <= y <= u`, `y <= u*delta`, `x <= y`,
`y <= x-l*(1-delta)`, `delta in {0,1}`,

where the output box is intrinsic to one continuous HZ factor. Thus the graph
uses one continuous factor, the same binary nonconvex phase, and three
inequalities. Shared preactivation latent IDs and all earlier predicates remain
unchanged. This is neither a Zonotope/CZ relaxation nor a split/backward/dual
solver path.

For preactivation support `p`, the extended and quotient graphs add `p+7` and
`2p+5` predicate nonzeros respectively. The current selector was derived from
paired failures:

- strict fill decrease is always accepted;
- equal fill (`p=2` on average) is accepted only if post-quotient latent width
  is at most 512;
- all other layers retain the existing extended equality graph;
- the rule reads only current HZ structure and workload size, never benchmark
  or iid;
- default remains off.

Dense pointwise extremization over both binary values, stable-case tests,
latent-ID tests, and dense/sparse lowered-matrix equality tests bring the
focused suite to 28/28 passing.

### Always-quotient ablation: useful but not promotable

The raw quotient nearly halved continuous dimension while preserving binary
and row counts: ACAS Xu iid 0 `515 -> 260`, TLL iid 0 `438 -> 220`. It did not
recover ReluSplitter iid 2 from its earlier propagation drop.

Solver behavior exposed the formulation boundary. TLL iid 2 retained CERT and
improved from about 3.65 s to 2.24 s. ACAS Xu iid 102 regressed from an 11.1 s
CERT to UNKNOWN at 15 s because dense preactivation support increased predicate
nnz to 11,556. Unconditional quotient is therefore rejected.

### Non-increasing-fill selector v1 and large-block counterexample

The first fill-aware rule accepted `p<=2`. ACAS Xu selected zero factors and
retained iid 102 CERT at 11.12/11.13 s paired. Dense TLL iid 2 selected 96
factors (`442 -> 346` continuous, both 2,596 predicate nnz) and retained CERT,
3.63 s baseline versus 2.42 s candidate.

The sparse implementation then reserved one rather than two continuous frame
slots. N=16 iid 4 selected 374 factors (`1766 -> 1392`) and iid 7 selected 360
(`1738 -> 1378`), with binary count, total rows, and predicate nnz unchanged.
At 15 s both arms were UNKNOWN. The required longer retained-CERT gate found a
decisive regression on iid 4: baseline CERT at 31.70 s, quotient still UNKNOWN
at 60 s. Equal-nnz removal of a large equality block harms MILP presolve even
though the integer set and LP projection are equivalent. This v1 selector is
rejected.

### Strict-fill probe and budgeted selector v3

Requiring strict fill decrease alone selected zero factors on TLL N=8 and N=16,
so it was safe but discarded the measured N=8 benefit. The current rule allows
equal fill only when post-transform latent width is at most 512. This restored
exactly 96 sparse quotient factors on N=8 iid 2 and selected zero on N=16 iid
4/iid 7.

On the final sparse paired run for iid 2, binary count and predicate nnz were
identical, continuous factors changed `442 -> 346`, and both arms retained
CERT: 3.23 s baseline versus 2.56 s candidate. N=16 is structurally identical
to baseline under this rule, so the earlier retained-CERT regression is outside
the candidate boundary.

### Current decision

Trial 3 v3 is retained as a default-off candidate component. It is a real exact
Neural-HZ representation improvement with dense and sparse implementations,
unit equivalence evidence, a repeated structural hit, and a retained-CERT
speed positive. It has not produced a new formal solve, so the score remains
1,870 / 2,413 and no full replay/default promotion is authorized. The next
step is a read-only census of formal UNKNOWN instances in other families for
the same low-support ReLU structure before defining a new structural class.

## 2026-08-31: Trial 4 — exact shared ReLU graph

### Pre-registration and exact boundary

The new repeated structure is a set of unstable ReLU inputs whose complete
affine rows are identical in one sparse HZ frame: center, continuous column
indices and values, and binary column indices and values all match byte for
byte. Such rows denote the same latent expression. Applying ReLU once and
copying the resulting latent row is exact, so one extended ReLU graph can be
used for the whole group.

The implementation keys groups by the bytes themselves, not by a digest or a
tolerance. Bounds for a group use the minimum valid lower and maximum valid
upper endpoint. It retains the existing two-continuous/one-binary extended
graph, equality predicate, inequalities, shared frame identity, and fail-closed
size limit. No approximate duplicate, convex relaxation, split, attack,
backward pass, or instance identifier participates in the rule. The arm is
explicit opt-in (`share_relu`).

An LP-extremization test fixes the source coordinate across both binary phases
and proves that both copied outputs have the unique true ReLU value, including
when group members carry different valid bounds. The focused suite is 29/29.

### TLL structural and retained-result evidence

The exact duplicates recur throughout TLL:

- iid 2: `442c/220b/660 rows/2596 nnz` becomes
  `314c/156b/468 rows/1908 nnz`; 50 rows are shared and CERT is retained
  at 2.45 seconds;
- retained iid 4: `1766c/882b/2646 rows/10466 nnz` becomes
  `970c/484b/1452 rows/6132 nnz`; 316 rows are shared. The candidate is
  UNKNOWN at 15 seconds but retains CERT at 42.26 seconds under a 60-second
  cap, versus baseline CERT at 31.70 seconds;
- formal UNKNOWN iid 7: `1738c/868b/2604 rows/10292 nnz` becomes
  `918c/458b/1374 rows/5866 nnz`, with 318 shared rows, but remains UNKNOWN
  at both 15 and 60 seconds.

All 15 formal TLL UNKNOWN instances hit the structure. Their final shared HZs
range from iid 7's 918 continuous/458 binary factors to iid 31's
10,730 continuous/5,364 binary factors. The largest observed exact group has
476 members. Short full solves on iid 8, 9, 12, 14, and 15 remain UNKNOWN.
iid 9 and iid 15 reach the unsafe-region subproblem, but iid 9 remains UNKNOWN
after 60 seconds with one explored node.

### Current decision and next same-class hypothesis

Exact duplicate sharing is a genuine HZ structural simplification and retains
the sampled certificates, but it is not a speed win on the larger retained
case and has produced no new formal solve. It stays default-off; headline score
remains 1,870 / 2,413.

The next census stays in the same algebraic class. For affine rows proven
exactly related by a positive scalar, `x_j = lambda*x_i`, positive homogeneity
gives `ReLU(x_j) = lambda*ReLU(x_i)`. Trial 4 may therefore be extended to
strict positive-proportional groups, but only with an exact component-wise
proof of the relation and no tolerance matching. This hypothesis must first
show additional real structural hits before an implementation is allowed to
enter solver gates.

### Signed relation census

Exact positive-proportional census on iid 7, 9, and 15 produced exactly the
same group counts as byte-identical duplicate census, so it added no usable
structure and was closed without a solver transform. Removing only the pivot
sign from the exact rational signature produced much larger groups. A stricter
second census proved that every extra relation was byte-exact negation, not a
general scalar:

- iid 7: duplicate removable rows 242, signed removable rows 555;
- iid 9: 554 versus 1,294;
- iid 15: 1,082 versus 2,304.

The implemented boundary therefore admits only `x_j = x_i` or `x_j = -x_i`.
It canonicalizes the first nonzero sign and compares the complete finite
binary64 coefficient bytes with matching sparse column indices. Signed zero is
canonicalized to real zero; nextafter-different values remain separate.

For a negative member it uses the exact identity
`ReLU(-x) = ReLU(x) - x`. The shared extended graph retains two continuous
factors, one binary phase factor, one equality, and two inequalities per signed
group. A compact quotient retains one continuous factor, the same binary phase,
and three inequalities. Neither form drops the original affine latent row.
LP extremization across thirteen source values and both binary values proves
both `ReLU(x)` and `ReLU(-x)` outputs uniquely in both formulations. The
focused suite is 33/33 passing.

### TLL capability result

Signed sharing changes the final models by an order of magnitude. Examples:

- iid 2: `442c/220b/2596 nnz` becomes `142c/70b/746 nnz` in the extended
  signed form and retains CERT in 0.17 seconds;
- iid 4: `1766c/882b/10466 nnz` becomes `454c/226b/2482 nnz` extended, or
  `228c/226b/2930 nnz` compact; compact retains CERT in 0.72 seconds;
- iid 7: compact final model is `213c/211b/2731 nnz` and changes the formal
  result from UNKNOWN to CERT;
- iid 18: compact final model is `1202c/1200b/16850 nnz` and changes UNKNOWN
  to CERT.

All five old TLL certificates (iid 0, 1, 2, 4, 6) are retained by the compact
arm. Repeated isolated runs produced five new certificates:

- iid 7: CERT at about 2.7--3.1 seconds;
- iid 12: CERT at about 4.7 seconds compact (14.4 seconds extended);
- iid 14: CERT at about 17.6 seconds extended and 26.6 seconds compact;
- iid 18: CERT twice at about 18.8 seconds compact;
- iid 22: CERT twice at about 16.6--16.7 seconds compact.

The compact arm also recovers five concrete witnesses from the exact shared
HZ frame. The isolated worker rejects any reconstructed point outside the
input box or failing a concrete PyTorch forward pass. Independent ONNX Runtime
checks on the original models then confirmed:

- iid 19: input `[-0.76232576, 1.87528753]`, output `-2.95143795`, unsafe
  `Y <= -1.67281347`, slack about 1.27862;
- iid 21: input `[0.48274392, 0.10813932]`, output `-0.97488809`, unsafe
  `Y >= -1.01609644`, slack about 0.04121;
- iid 26: input `[1.99282408, -1.50796473]`, output `-3.86822224`, unsafe
  `Y <= -3.49615862`, slack about 0.37206;
- iid 28: input `[0.79883474, -0.80744892]`, output `-1.67825508`, unsafe
  `Y <= -1.60748959`, slack about 0.07077;
- iid 31: input `[-0.08583260, 0.07803351]`, output `-1.65809441`, unsafe
  `Y >= -2.42284633`, slack about 0.76475.

All five witnesses reproduce in a second HZ run. No invalid ADV is retained.
The five remaining formal TLL UNKNOWNs are iid 8, 9, 15, 16, and 24;
selected 30/60-second extensions did not change them.

### Trial 4 gate status

TLL candidate coverage changes from `5 CERT + 12 ADV = 17/32` to
`10 CERT + 17 ADV = 27/32`, a net `+5 CERT +5 validated ADV`. The corresponding
cross-family candidate score is 1,880 / 2,413. This is not yet the formal
headline: 1,870 remains frozen until cross-family retained-result shadows and
the complete 2,413-instance gate pass. The extended and compact signed-share
arms remain explicit opt-in and the historical HyZor archive remains untouched.

### Compact trigger correction and guarded replay

The first compact implementation also quotient-encoded singleton groups. That
was mathematically exact, but it made a nominal zero-hit family inherit the
presolve risk measured in Trial 3. Before cross-family promotion, the selector
was corrected: compact form is used for a layer only if the exact signed
partition contains at least one member with orientation opposite to its
representative. Otherwise the layer uses the established extended graph. An
integration test verifies that two unrelated singleton ReLUs allocate the
baseline two continuous factors each and record zero compact layers.

All ten new TLL results were rerun after this correction. Every result is
unchanged, each actually triggers 8--12 signed compact layers, and all five old
TLL certificates are retained. Focused tests are 35/35 passing.

Guarded baseline/candidate pairs on ACAS Xu iid 0 and Cersyve iid 11 are
structurally identical: respectively `515c/255b/12076 nnz` and
`348c/172b/4216 nnz`, with zero shared rows and zero compact layers. Across
representative CERT and UNKNOWN/TIMEOUT probes in the other non-TLL families,
no signed member was observed; some convolutional/transformer paths lose the
HZ before this candidate can execute. ViT iid 101 records zero signed hits and
then reaches the isolated worker's known 16 GB lowering limit because that
worker retains intermediate caches. The previously validated production
cache-release CERT remains the controlling ViT evidence. Full 2,413 replay is
still pending, so 1,880 remains a candidate score rather than the headline.

## Trial 5: exact dead signed-ReLU graph elimination

### Pre-registered rule and first capability result

For a compact signed group consumed by exactly one Dense layer, the shared
nonlinear term is dead when every successor output row has exact-real zero
sum over that group's stored binary64 weights. The implementation proves zero
with integer dyadic ratios, removes exactly the group's three local
inequalities and its private continuous/binary columns after the Dense, and
fails closed on any unexpected retained coupling. It performs no tolerance
merge, search, split, backward pass, or instance-dependent selection.

The all-cancel arm solved the remaining TLL iid 24 as CERT twice, at about
36.5--36.9 seconds. Its lowered model has `923c/921b/2763 rows/13451 nnz`
after eliminating 919 local graphs. All eleven old and candidate certificates
remain CERT. However, the retained-ADV gate exposed a real optimization
regression: iid 21, 27, and 29 remain UNKNOWN at both 45 seconds and isolated
60 seconds, although the Trial 4 compact arm finds their validated witnesses.
Therefore the all-cancel arm is rejected for promotion despite its exactness;
iid 24 is capability evidence only until the regression is removed.

### Pre-registration for the mixed-orientation refinement

The next and only refinement in this sub-trial is structural: eliminate a
dead graph only when its exact equivalence class actually contains both an
`x` row and a `-x` row. Exact-zero groups containing only same-orientation
duplicates retain their redundant lift, which may preserve useful MIP branch
geometry. The selector is uniform across all instances and layers and is
strictly narrower than the already-proved all-cancel rule. The gate order is:
iid 24 must remain CERT; iid 21/27/29 must recover concrete-valid ADV; then all
remaining TLL solved cases are replayed. Failure of either first gate closes
the refinement without changing the 1,870 formal headline.

The refinement has now failed its first retention check. It deletes the same
862 groups on iid 21 as the all-cancel arm and remains UNKNOWN at 45 seconds;
all of that instance's cancelable groups are already mixed-orientation. The
mixed-only arm is therefore closed without running iid 27/29 and is not a
candidate promotion.

Before another selector is proposed, the next action is a propagation-only
layer profile on iid 24 and the three ADV regressions. This census may inspect
topological location and exact group cardinalities, but may not choose a
policy. Any subsequent retention rule must be written here before its solver
results are run, must be uniform across the four structures, and must keep iid
24 CERT plus all three concrete-valid ADV before the broader family gate.

The profile separates a common structural class. Total cancelable groups split
into exact two-row groups versus higher-multiplicity groups as follows: iid 21
`862 = 689 + 173`, iid 24 `919 = 728 + 191`, iid 27 `986 = 794 + 192`, and iid
29 `1304 = 1050 + 254`. Higher-multiplicity groups occur only in the first five
aggregation layers (maximum cardinality 556--1098 across these examples); the
last seven layers are entirely irreducible `x/-x` pairs.

The next pre-registered rule is therefore **pair-only cancellation**: delete a
proven dead compact graph only when its exact signed equivalence class has
cardinality two, while retaining every higher-multiplicity graph as a
redundant aggregate lift. This retains the large early aggregation cuts while
still deleting about 79--81% of all dead graphs. It uses group cardinality
only, never instance identity or a numerical tolerance. The ordered gate is
iid 24 CERT, then iid 21/27/29 concrete-valid ADV, then the complete TLL solved
set. A miss at either targeted gate rejects pair-only cancellation.

Pair-only cancellation passes the targeted gate. iid 24 remains CERT in 27.4
seconds with `1114c/1112b/3336 rows/15310 nnz` after 728 eliminations. The
three previously regressed cases are again concrete-valid ADV: iid 21 in 32.8
seconds after 689 eliminations, iid 27 in 4.8 seconds after 794, and iid 29 in
9.3 seconds after 1050. Invalid ADV remains zero. This is not yet a promoted
family result: the next pre-registered action is a complete replay of all 11
candidate CERT plus all 17 retained/candidate ADV under the same pair-only
arm. All 28 must retain their result before the remaining UNKNOWNs are tried.

The complete solved-set gate retained 27/28 cases. The sole regression is iid
31, which remains UNKNOWN at 45 seconds; all eleven certificates and the other
sixteen concrete ADV are retained with zero invalid witnesses. Thus pair-only
is not promoted. A structural histogram shows iid 31 contains 110 dead groups
of cardinality four versus 89 in iid 29, while their cardinality-six-and-above
tails are otherwise similar. The smallest pre-registered extension is
**two-pair cancellation**: eliminate exact dead groups of cardinality at most
four and retain every group of cardinality six or more. This is a single
uniform bound, not a threshold search. It must first recover iid 31 ADV, then
retain iid 21/27/29 ADV and iid 24 CERT, before another solved-set replay.

Two-pair cancellation passes the targeted gate. iid 31 is concrete-valid ADV
in 8.3 seconds after 1,149 eliminations. iid 21/27/29 remain concrete-valid
ADV in 4.5/12.6/36.5 seconds, and iid 24 remains CERT in 23.0 seconds after
786 eliminations. Invalid ADV is zero. The complete 28-case TLL solved-set
replay is now the controlling promotion gate; no remaining UNKNOWN may be
attempted until all 28 are retained under the same candidate hash.

The two-pair solved-set replay retains 26/28 but regresses iid 10 and iid 26 to
UNKNOWN, so it is also not promoted. Their first signed-ReLU source widths are
1,152 and 6,272, compared with 8,192 for iid 31; subsequent source widths halve
monotonically. The final Trial 5 refinement is pre-registered as a local
**width-adaptive lift budget**: cardinality-two dead graphs are always removed,
but cardinality-four graphs are removed only at a ReLU whose source width is
at least 8,192. The bound is a power-of-two representation-size boundary, is
evaluated locally in one pass, and cannot observe instance identity, solver
state, or future layers. It makes iid 10/26 representation-identical to the
passing pair-only arm while changing only the widest layer of iid 31. Failure
to recover iid 31 closes Trial 5; success requires iid 10/26 plus the earlier
five-case guard before a final 28-case replay.

The width-adaptive rule passes its seven-case sensitive gate. iid 10, 21, 26,
27, 29, and 31 are all concrete-valid ADV, and iid 24 remains CERT. Invalid
ADV is zero. Small/medium-width cases are representation-identical to the
passing pair-only arm; iid 31 removes four-row groups only at its 8,192-row
first layer. The final 28-case solved-set replay is now pre-registered under a
single candidate hash. Only 28/28 retention can promote iid 24 as the eleventh
TLL gain and authorize attempts on the four remaining UNKNOWNs.

The final solved-set gate passes 28/28 under candidate hash
`0591e23a3e9bae368d9192a06783dbdad2777b068080ab71b632773190d1770d`:
all eleven certificates and all seventeen concrete-valid ADV are retained,
with zero invalid witnesses and zero errors. iid 24 is therefore a qualified
family-level candidate gain. TLL moves from frozen `5 CERT + 12 ADV = 17/32`
to candidate `11 CERT + 17 ADV = 28/32`, net `+11`; the cross-family candidate
becomes 1,881 / 2,413 while the formal headline remains 1,870. The remaining
iid 8/9/15/16 are now authorized for one 45-second run each. Any new result
must reproduce; a new ADV must also pass concrete-network validation.

iid 8 became CERT twice at about 15.9 seconds; iid 9/15/16 remain UNKNOWN.
The controlling TLL candidate is therefore `12 CERT + 17 ADV = 29/32`, net
`+12`, and the provisional cross-family union is 1,882 / 2,413.

## Trial 6: sparsity-priced exact CNN HZ propagation

### Pre-registration for the first convolution barrier

On CIFAR100-large iid 166 and TinyImageNet iid 143, the input sparse HZ exists
with 3,072 and 9,408 continuous factors, but disappears at the first Conv2D.
The Conv2D sparse linear operator already exists and is exact. The loss occurs
before it runs because the generic resource guard charges the dense envelope
`output_rows * latent_columns`: about `65,536 * 3,072` and
`200,704 * 9,408`, despite each convolution row having only a local 3x3
support.

The pre-registered first rule is **sparse-affine nnz pricing**. When explicitly
enabled, Conv2D, ConvTranspose2D, and AvgPool2D may bypass only that dense
pre-check. The established exact sparse linear map is then executed and its
actual stored center, generator, equality, and inequality entries are charged
against the same 64M budget. Exceeding the actual-storage budget still drops
the HZ fail-closed. No nonlinear graph rule or solver setting changes. The
capability gate is deliberately narrow: iid 166 and iid 143 must retain an
exact HZ after the first convolution; the next structural loss is then
measured before any further rule is proposed.

The sparse-affine gate succeeds: CIFAR iid 166 retains an exact 65,536-row HZ
and Tiny iid 143 an exact 46,656-row HZ after the first convolution. Both then
hit `sparse_size_limit:RELU`. Tight-bound census shows only 691 and 512 truly
unstable rows; 64,845 and 46,144 rows are phase-stable. The unstable generator
supports contain only 18,342 and 13,824 nnz.

The next pre-registered rule is **unstable-support nnz pricing for ReLU**. It
may bypass the dense pre-check only when the sparse-affine arm is enabled. The
pre-allocation charge is the current actual HZ storage plus eight times the
selected unstable generator support and 32 entries per new graph; after the
established exact ReLU transform, actual sparse storage must remain below the
same 64M cap. Stable rows remain exact zero/identity rows and every unstable
row retains continuous, binary, and predicate factors. The capability gate is
to retain an exact HZ past the first ReLU on iid 166 and 143 and identify the
next common loss.

The first run of that gate exposed an earlier common loss introduced by the
current frontend decomposition: both networks retain their exact sparse HZ
after Conv2D, then independently drop it at elementwise `SCALE` and `BIAS`
nodes under the same dense-envelope pre-check. These nodes already have exact
sparse HZ transforms and neither creates latent columns nor predicate rows.
Before rerunning the ReLU gate, the sparse-affine rule is therefore
pre-registered to cover `SCALE`, `BIAS`, and fused `BN` as well as the three
spatial affine operators. Every covered operation still executes its existing
exact transform and must pass the identical 64M actual-storage post-check.
This extension is keyed only by affine operator kind, is shared by CIFAR100
and TinyImageNet, and cannot inspect an instance id. The immediate gate is
that both elementwise affine paths retain HZ state and expose the first ReLU
to the already pre-registered unstable-support rule.

That gate passes substantially beyond its target. CIFAR100 iid 166 retains
exact HZ state through two ReLUs and three convolutions; TinyImageNet iid 143
retains it through two ReLUs and four convolutions. Their next common loss is
the first residual `ADD`, again solely at the dense-envelope pre-check. Both
operands are cached in the same non-null latent frame, and the existing
`sparse_hz_add_same_frame` operation pads shared continuous/binary columns,
adds value maps, and merges only exact predicate prefixes. It rejects frame or
shape mismatch.

The next pre-registered extension is **same-frame residual nnz pricing**.
`ADD` and its algebraic twin `SUB` may bypass the dense-envelope pre-check only
under the same sparse-affine opt-in. The established exact same-frame merge
must execute successfully, and the result must pass the unchanged 64M actual
storage post-check. Any missing branch, frame mismatch, shape mismatch, or
post-check excess remains fail-closed. The capability gate is to retain exact
HZ state after the first residual join in both networks and identify their
next structural loss.

An unrestricted first run crossed the residual join but continued into a
later sparse-convolution construction for more than three minutes, reaching
approximately 5.4--7.8 GB RSS per process before it was stopped. Those two
runs produced no result file because the worker writes exclusively at normal
completion. To preserve completed layer evidence without turning census into
an unbounded full-network run, the isolated worker now has a census-only
`--stop-after-layer` option. It stops only between completed ACT layers and
writes the ordinary provenance-bearing JSON; it does not alter propagation,
the HZ representation, solver status, or any production configuration.

The bounded records prove the first residual join exact in both families.
CIFAR100 retains `65,536` outputs, `9,882c/3,405b`, 8,954 inequality rows,
and 11.92M stored entries; TinyImageNet retains `25,088` outputs,
`12,160c/1,376b`, 3,880 inequality rows, and 16.31M entries. Both use frame 1.
The following convolution also remains exact but raises propagation time to
84.3 and 75.8 seconds. Source audit identifies a representation-independent
cost: the sparse Conv2D matrix builder appends roughly 36 million triplets
through six nested Python loops for a 64-to-64 3x3 layer.

The next pre-registered component is an opt-in **row-native exact CSR
convolution builder**. It computes the same valid receptive-field columns in
the same `output-channel, output-row, output-column, input-channel, kernel-row,
kernel-column` order, but writes CSR `indptr`, `indices`, and `data` arrays
directly. It may not change a coefficient, omit an explicit zero, reorder a
row, change bias broadcast, or alter grouped/strided/padded/dilated semantics.
The component gate is exact CSR-array equality against the untouched loop
builder over ordinary, grouped, depthwise, strided, padded, and dilated unit
cases. Only then may it run on the two real post-residual layers. It is a
performance component, not a score gain, and stays default-off.

The component gate passes. All CSR arrays and biases are exactly equal in 19
focused tests. On the real post-residual boundary, propagation falls from
84.28 to 26.02 seconds for CIFAR100 (3.24x) and 75.76 to 45.03 seconds for
TinyImageNet (1.68x), with identical per-layer HZ dimensions and storage.
Using this payment, the second residual joins remain exact at 32.56M and
61.59M entries. The next CIFAR convolution remains exact at 23.40M; the next
Tiny convolution correctly fails the unchanged actual-storage gate.

The Tiny profile localizes the growth: the second-block ReLU output occupies
10.18M entries, its following convolution 61.42M, and the residual join adds
only 0.17M. The next pre-registered representation arm therefore combines the
row-native CSR payment with Trial 3's already proven **exact compact ReLU
quotient**. Every unstable ReLU retains its binary phase and all three exact
inequalities but introduces one, rather than two, continuous auxiliary
factors. No factor present before that ReLU is removed. The capability gate is
whether this uniform quotient retains Tiny iid 143 through layer 25, where the
extended exact graph hits `sparse_storage_limit:CONV2D`; CIFAR iid 166 is the
same-structure guard. This arm remains default-off and earns no score until a
terminal result plus family replay exists.

The compact quotient capability gate is negative. It reduces Tiny's
continuous width from 13,372 to 11,390, but replaces exact equality structure
with denser inequalities: second-residual storage rises from 61.59M to 62.75M
and layer 25 still fails `sparse_storage_limit:CONV2D`. At layer 24, 60.35M
entries are already in `Gc`; residual predicate merging is only a secondary
cost. CIFAR retains exact state under the arm, so this is a capability miss,
not a soundness regression. The arm stays off and is not extended.

Before attempting a lazy representation, an interval-only phase census is
pre-registered. It records stable-positive, stable-negative, and unstable
counts at every ReLU even when sparse HZ state has already been dropped. A
local deferred Conv/affine/ReLU materialization is justified only if the ReLU
immediately following Tiny layer 25 would discard enough stable-negative rows
to bring retained exact value support plausibly below 64M. Otherwise the
direction is closed without implementation.

The phase census is positive. Immediately after the rejected Tiny
convolution, 17,021 / 25,088 rows (67.8%) are interval-proven stable negative;
only 8,067 rows can survive its ReLU. The homologous CIFAR layer has 19,228 /
32,768 (58.7%) stable-negative rows. This exceeds the minimum needed to make
row-selective materialization plausible.

The next pre-registered innovation is a **phase-filtered affine island**. It
triggers only when a Conv2D has exactly one live path consisting solely of
elementwise `SCALE`/`BIAS`/`BN` nodes into one ReLU; any other direct successor
must be a dead leaf. Using only forward interval facts, the island omits
generator rows whose final pre-ReLU upper bound is nonpositive, applies the
exact affine chain to every retained row, and constructs the ordinary exact
HZ ReLU graph at the target. The partial intermediate object is never cached,
solved, merged, or exposed to another live path. At the target, the actual
interval bounds must bitwise match the precomputed bounds; every omitted row
must still have `ub <= 0`; and the completed exact HZ must pass the unchanged
64M actual-storage gate. Any topology, bound, shape, or resource mismatch
fails closed. The arm is default-off. Unit gates compare the final HZ arrays
against full materialization on a controlled stable-negative example and
prove masked CSR rows preserve all retained coefficients. The real capability
gate is Tiny iid 143 layer 28 exactness, followed by the CIFAR homolog.

The affine-island gate passes. Tiny retains an exact frame-1 HZ at layer 28
with `13,454c/2,023b`, 4,527 equality and 9,054 inequality rows, and 20.55M
stored entries; the prior arm lost HZ at layer 25. CIFAR retains its homolog at
7.79M. The target bounds also match bitwise. Earlier exact HZ states tighten
the subsequent forward interval facts, reducing Tiny's layer-28 ambiguous
count from 3,219 in interval-only propagation to 531 without any backward
pass. This is a genuine representation capability crossing.

The next Tiny loss is Conv2D layer 29 before residual ADD 32, so the one-path
island correctly declines to trigger. The downstream ReLU 36 has 11,912 /
25,088 stable-negative rows even under interval-only propagation; CIFAR has
20,736 / 32,768 and no ambiguous rows. This pre-registers **lazy affine-DAG
HZ**. A lazy expression is an exact sum of same-frame HZ sources, each followed
by a chain of sparse affine operators, plus a constant. Conv2D appends an
operator; elementwise affine nodes update the chain and constant; residual ADD
unions terms only when every source shares one non-null frame. No latent or
predicate is copied, relaxed, or deleted. At ReLU, the outer operator is first
row-masked only for forward-interval-proven stable-negative outputs, operator
chains are composed, source HZs are transformed, and their predicates are
merged with the existing exact prefix rule. The ordinary exact ReLU graph is
then constructed. Both every composed operator and the completed HZ must stay
within 64M actual entries. Unsupported topology, frame mismatch, non-exact
source, bound mismatch, or resource excess fails closed. The cache and config
are default-off; no lazy object is accepted by the solver or terminal layer.
Unit promotion requires equality with eager full materialization on a
two-branch residual example before real layer 36 census.

The unit residual equivalence gate passes (22 focused tests). On Tiny's first
real residual DAG, two lazy terms merge at ADD 16 and materialize at ReLU 20
to the same exact `13,372c/1,982b`, 10.18M-entry HZ as eager propagation,
reducing propagation from about 90.6 to 25.5 seconds. CIFAR exposes a
pre-materialization ordering issue: its first stored Conv operator has 36.26M
entries, so appending the full outer Conv14 operator exceeds 64M before the
known ReLU17 stable-negative mask is applied. The pre-registered correction is
to recognize the same isolated outer Conv/affine/ReLU suffix for a lazy input,
row-mask that outer operator first, and only then charge/combine it. No cap or
semantic rule changes; failure to recover CIFAR ReLU17 closes the lazy arm.

Outer-row filtering recovers CIFAR ReLU17 exactly at 11.51M entries and
retains Tiny ReLU20 exactly. The deeper ReLU36 gate then exposes the same
operator-union limit at different residual joins: CIFAR ADD21 and Tiny ADD32.
Both have a saved skip expression whose ordinary HZ form was independently
shown below 64M (about 11.9M and 61.6M). The next pre-registered refinement is
an **exact lazy checkpoint**. Only when expression operator union exceeds 64M,
an operand may be fully materialized; that ordinary exact HZ must itself pass
the unchanged 64M actual-storage gate, after which it re-enters the expression
as an identity term with zero stored operators. Operands are tried in
descending operator-storage order; failed checkpoints are discarded. This is
representation scheduling only. ReLU36 exactness in both families remains the
unchanged capability gate.

Exact checkpointing recovers ReLU36 in both families: CIFAR retains 22.97M
entries after two checkpoints and Tiny 19.96M after one. The unchanged arm
then reaches ReLU44 exactly at 16.53M and 35.32M entries. The remaining CNN
body repeats the same residual motif. For terminal closure, `FLATTEN` already
preserves a lazy expression row order; `DENSE` is pre-registered as one more
exact sparse linear operator using its existing weight and bias, under the
same operator-entry cap. The following ReLU must materialize an ordinary exact
HZ, and the final Dense must use the established eager HZ transform. A lazy
expression reaching ASSERT remains an error. With this closure, the next gate
is one complete propagation of iid 166 and iid 143 to output HZ; solver time is
kept at 1 ms so this remains a representation capability test.

The first complete Tiny run reaches ReLU55 exactly at 45.89M entries, then
stops at ADD59 because logical operator-union accounting sums ADD51's already
resident 59.15M operators with the already resident 5.92M main operator. ADD
itself allocates neither matrix: it creates term references and a bias vector.
Charging the referenced logical union repeats the original dense-envelope
error at the lazy level. The pre-registered accounting correction is
**allocation-native operator pricing**: every newly allocated base operator,
every composed per-term operator, and every materialized HZ remains capped at
64M entries; an ADD is charged only for new bias/term metadata, while its
referenced operators retain their individual prior charges. Materialization
continues term-by-term and fails if any composed term crosses 64M. The process
address-space cap remains 16 GB. No matrix, factor, or predicate changes. The
unchanged capability gate is Tiny ReLU63 exactness; CIFAR terminal exactness
is the zero-regression guard.

Allocation-native pricing retains CIFAR terminal exactness with the same
10,852 continuous and 3,890 binary factors and writes a 585 MB atomic HZ
checkpoint. Tiny now crosses ADD59, but its outer Conv60/ReLU63 island fails
while summing seven already row-filtered materialized terms. Several terms
reference the identical source HZ through different affine paths. The next
pre-registered exact simplification is **source-coalesced lazy terms**: before
touching a source HZ, compose each term's operator, group terms only by Python
object identity of their source, add those sparse operators exactly, eliminate
exact zeros, then transform that source once. No cross-source equivalence is
inferred. This carries a source's predicate set once and exposes affine-path
cancellation before the intermediate-storage check. Every coalesced operator
and partial/final HZ remains capped at 64M. The existing same-source residual
unit case must remain array-identical; the real gate remains Tiny ReLU63.

Source coalescing is a real-network negative: ADD59's seven terms reference
seven distinct HZ state objects, so no legal group forms and Conv60 still
fails at the cumulative pre-ReLU HZ check. The failure object is nevertheless
strictly transient: it exists only inside the materializer and is immediately
consumed by the already-proven exact ReLU transform; it is never cached,
merged downstream, lowered, or solved. The next pre-registered rule is
**transient phase-consumer accounting**. Every base/composed operator and
every single-source materialized part remains below 64M. Their exact sum may
temporarily exceed 64M only on the stack immediately before ReLU, under the
unchanged 16 GB process cap. ReLU slot pre-allocation may likewise defer its
persistent-storage check. The completed ordinary exact ReLU HZ must still be
at most 64M or the state is discarded. No other operator or checkpoint gets
this exception. Tiny ReLU63 exactness remains the gate.

Transient phase-consumer accounting is locally sound but misses that terminal
gate. The focused guard record
`results/trial6_transient_guard__tinyimagenet_2024__iid143__relu20_v1.json`
retains the ordinary exact layer-20 HZ at 10,182,159 stored entries
(`13,372c/1,982b`, 2,546 equality and 5,092 inequality rows). The full record
`results/trial6_lazy_dag__tinyimagenet_2024__iid143__full_transient_v2.json`
reaches the prior layer-55 HZ (`13,700c/2,146b`, 45,893,946 entries), but
Conv60 is discarded with `deferred_lazy_relu_storage_limit`; ReLU63 has no HZ.
Propagation takes 995.743816 seconds. The JSON contains a requested checkpoint
path, but no `hz_checkpoint_written` field exists and the named checkpoint file
does not exist. This is a negative capability record, not a score gain.

CIFAR100 iid166 supplies the complementary positive representation result.
`results/trial6_lazy_dag__cifar100_2024__iid166__full_v2.json` completes exact
HZ propagation to Dense layer 82 and atomically writes
`checkpoints/cifar100_2024__iid166__lazy_dag_v2.pkl`. The terminal HZ has 100
outputs, `10,852c/3,890b`, 598,440 equality rows, 1,196,880 inequality rows,
and 50,485,996 stored entries (`Gc` 696,200 nnz; `Ac` 45,002,176 nnz).
Propagation takes 924.065577 seconds. The checkpoint is 613,102,026 bytes
(584.700 MiB), with SHA-256
`61bc8092b18b21ccf3264e61270ae309b5774c460d0273bd9d31105407be66ec`.
The independent 45-second reload in
`results/trial6_checkpoint_solver__cifar100_2024__iid166__45s_v1.json`
returns `UNKNOWN` (`reason=base_unknown`). Terminal exactness is therefore a
representation capability result only; it adds no CERT or validated ADV.

## Trial 7: exact predicate sharing and Star-like frontier rebasing

### Exact common-prefix predicate sharing

The same-frame residual merge previously copied identical predicate history
once per branch. The exact correction keeps one byte-identical common prefix
and retains every divergent suffix. TinyImageNet iid143 shows unchanged value
widths and binary semantics while duplicate predicate storage falls:

- ReLU36: 9,091 to 2,101 equality rows, 18,182 to 4,202 inequalities, and
  19,957,512 to 18,553,208 stored entries (-7.04%).
- ReLU44: 18,147 to 2,144 equalities, 36,294 to 4,288 inequalities, and
  35,323,208 to 31,342,037 entries (-11.27%).
- ReLU55: 36,253 to 2,146 equalities, 72,506 to 4,292 inequalities, and
  45,893,946 to 36,234,298 entries (-21.05%).

The authoritative positive record is
`results/trial7_predicate_sharing__tinyimagenet_2024__iid143__relu63_v2.json`.
It crosses Trial 6's Conv60/ReLU63 boundary and retains an exact layer-63 HZ.
One exact image rebase changes `Gc` from 40,575,778 nnz to 3,626 nnz while
adding 3,626 linked continuous image coordinates: `n_cont` 13,702 to 17,328,
`n_bin=2,147` unchanged, and total storage 42,673,985 to 42,684,863
(+10,878). The resulting state has 5,773 equality and 4,294 inequality rows;
propagation takes 821.881203 seconds.

The full follow-up
`results/trial7_predicate_sharing__tinyimagenet_2024__iid143__full_v1.json`
preserves that state and advances to the eight-term ADD67 lazy expression
(`lazy_operator_entries=125,016,192`), but Conv68/ReLU71 fails closed with
`deferred_lazy_relu_storage_limit` after 1,212.432383 seconds. Its named
checkpoint is absent. This is a genuine new exact-propagation boundary, not
terminal solving.

### Closed materialization and trigger ablations

Whole-ADD materialization is negative. In
`results/trial7_predicate_sharing__tinyimagenet_2024__iid143__add40_wide_v1.json`,
the first five-term ADD40 attempt fails with
`MemoryError:lazy materialized term storage limit`; the ADD51 record likewise
performs zero rebases and retains a six-term lazy expression. This path is
closed.

Unrestricted early image rebasing is also negative. The ReLU36 v2 ablation
rebases layers 9, 20, ADD24, and ReLU28, inflates the last exact state to
71,292 continuous and 6,250 binary factors, and does not recover ReLU36. A
narrower pre-sharing layer-55 rebase reduces `Gc` sharply but the subsequent
Conv60 allocation still fails, motivating predicate sharing instead of a cap
exception.

A structural residual-count ablation then triggered only before residual joins
with at least five lazy sources. The bounded record
`results/trial7_residual_guided_rebase__tinyimagenet_2024__iid143__relu36_v1.json`
correctly triggers once at ReLU36 (`residual_terms=5`), reducing `Gc`
16,830,814 to 4,440 with `n_bin=2,101` unchanged. The ReLU44 continuation
correctly triggers again at six sources. Both transforms are exact, but the
early dense link persists: relative to the non-rebased predicate-sharing
state, ReLU44 storage rises from 31,342,037 to 42,779,511 before its second
rebase, and binary factors rise from 2,144 to 2,484 because forward interval
propagation cannot exploit the new link equalities. Residual count alone is
therefore rejected as an acceptance rule. The implementation has been
returned to the fixed representation-wide storage-pressure gate; residual
source count is diagnostic only.

### Accounting boundary

Every completed Trial 7 record remains `UNKNOWN`, with no new concrete
validation and no terminal TinyImageNet output HZ. No family shadow or full
2,413-case promotion replay has run. CIFAR's terminal checkpoint also solves
`UNKNOWN`. These results contribute zero formal gain: the frozen headline is
still exactly 1,870/2,413 (1,063 CERT + 807 validated ADV), and no Trial 6/7
candidate may be default-enabled. Records carry branch, base commit, and a
candidate source SHA-256; while the implementation is uncommitted, the base
commit alone is not sufficient reproduction provenance.

### ReLU71 capacity diagnosis

The isolated 96M-entry capacity record
`results/trial7_capacity_probe96m__tinyimagenet_2024__iid143__relu71_v1.json`
crosses the former Conv68/ReLU71 boundary exactly. It stops intentionally at
layer 71 after 1,099.196570 seconds. The completed layer-71 HZ has 6,272
outputs, 21,227 continuous and 2,167 binary factors, 9,652 equality and 4,334
inequality rows, and 90,788,559 stored entries. Its interval phase census is
2,413 stable-negative, 3,826 stable-positive, and only 33 unstable neurons.
The second exact image rebase reduces the value map from 47,852,959 to 3,859
nnz but leaves a 90,753,607-nnz equality matrix. Thus the default 64M failure
is localized: 3,826 affine stable-positive rows are forced into the completed
HZ solely because 33 rows require a nonconvex ReLU graph. The capacity change
is diagnostic only; the record is `UNKNOWN`, creates no checkpoint or
concrete validation, and contributes zero formal gain.

## Trial 8 pre-registration: phase-separated exact lazy ReLU

The next candidate is one uniform, LP-independent structural rule. For a
lazy affine preactivation `x` with interval phase partition `(N, P, U)`, store

`ReLU(x)_N = 0`, `ReLU(x)_P = x_P`, and
`ReLU(x)_U = exact_binary_HZ(x_U)`.

Stable-positive rows remain an exact row-masked affine DAG; stable-negative
rows are zero; only unstable rows are materialized into the existing exact
binary HZ graph. The two components retain the same frame and shared latent
identities. All continuous and binary factors and all equality/inequality
predicates required by the unstable core are retained; binary factors are
never pivoted or deleted. Materializing the separated expression must recover
the ordinary exact ReLU result, including a concrete-input witness path.

The candidate is default-off and has two deliberately separate gates:

1. **Capability gate.** With the normal 64M completed-HZ limit, TinyImageNet
   iid143 must cross ReLU71 exactly. The phase core itself, each newly allocated
   operator, and every materialized HZ must remain below 64M. Failure closes the
   candidate without raising the cap.
2. **Representation gate.** Before any formal claim, hash-cons the reachable
   operator/source DAG, garbage-collect unreachable nodes, and charge unique
   physical operator, value-map, predicate, and metadata storage. The accepted
   representation must be strictly smaller than the existing reachable state.
   In particular, merely retaining ADD67's currently measured 125,016,192
   expanded operator entries behind references does not pass this gate. An
   implicit convolution stencil (`kernel + geometry + row mask`) or an
   equivalently exact compact operator is required if expanded CSR dominates.

Focused proof tests must cover stable-negative, stable-positive, and unstable
rows together; equality of materialized outputs; preservation of frame,
factor counts, predicates, and feasibility; rejection without a strict
storage decrease; and fail-closed terminal materialization. Only after those
tests may the arm run Tiny iid143 to ReLU71, then full Tiny propagation, then
CIFAR100 iid166 as the established zero-regression guard. A subsequent
same-structure shadow set, all 13 family replays, and the complete 2,413-case
replay remain mandatory before promotion. Until that sequence succeeds, the
formal score remains exactly 1,870/2,413.

### Staged implementation boundary

Trial 8 is deliberately split into two implementation stages. Version 1 first
constructs the already-audited ordinary exact ReLU HZ and then applies the row
partition. This minimizes semantic novelty and is the first real-network
equivalence/capability gate. The 96M raw (pre-rebase) ReLU71 object has
90,776,982 entries: `Gc=47,852,959`, `Ac=42,896,789`, and only 20 HZ-tightened
unstable rows (the interval census reports 33). Removing the stable-active
value rows yields a 42,924,043-entry nonlinear core before the small row-mask
and bias metadata are charged.

Only if version 1 passes is version 2 allowed to change construction order.
It materializes the authoritative interval-unstable rows first, applies the
same HZ fast-bound tightening on those rows, builds the existing exact binary
ReLU graph only for the remaining HZ-unstable rows, and keeps every
interval-stable-positive row lazy. The completed version-1 result and a full
materialization of version 2 must be byte-identical in value maps, latent
widths, equality/inequality predicates, and right-hand sides. Version 2 may
not weaken bounds, allocate fewer required binary slots, or use an interval
phase that differs from the existing `lb >= 0`, `lb < 0 < ub` encoding. Its
purpose is solely to avoid first constructing the 90.8M-entry transient.

Neither stage satisfies the global representation gate merely by crossing a
64M frontier. The phase expression can still reach ADD67's 125,016,192
expanded operator entries. Formal simplification requires exact implicit
convolution stencils or equivalent compact operators, hash-consed reachable
object accounting, consumer-aware garbage collection, and a strict decrease
in unique physical storage before family replay.

### Version-1 ReLU63 capability result

The authoritative default-64M record is
`results/trial8_phase_split__tinyimagenet_2024__iid143__relu63_v1.json`, with
candidate SHA-256
`06f3593e81f82bf1a76e457bed214d27d122b9be56e68114a9dca6ca493c0bfc`.
TinyImageNet iid143 reaches layer 63 in 681.002680 seconds and accepts one
phase-separated exact ReLU. The ordinary completed object has 42,673,985
entries. HZ tightening partitions it into 2,650 negative, 3,621 positive, and
one unstable row; the interval-only census is 2,646/3,618/8. The nonlinear
core retains `13,702c/2,147b`, 2,147 equality and 4,294 inequality rows, but
uses only 2,098,208 entries. Removing 40,575,777 stable-positive value nnz and
adding 9,893 mask/bias entries produces a 2,108,101-entry local persistent
frontier. No image rebase fires.

The output at layer 63 is an exact eight-term lazy HZ expression, not a
terminal HZ. Its reachable expanded operators still total 122,525,605 entries,
so the global representation gate remains open. The record is `UNKNOWN`, has
no concrete validation, and contributes zero formal gain. It is nevertheless
a positive real-network proof of the row-partition identity and authorizes
the pre-registered selective-materialization version 2.

The version-1 ReLU71 continuation is a representation-construction negative.
`results/trial8_phase_split__tinyimagenet_2024__iid143__relu71_v1.json`
(result SHA-256
`acf21b2c6e400d2c12d1d085319c534271b0535b760fc4e7fa126f1b1b09e2d1`)
retains the same layer-63 phase split but expands ADD67 to a 15-term lazy
expression with 128,440,229 referenced operator entries. Conv68 then fails
closed before ReLU71 while attempting to allocate an 86,495,464-element
float64 temporary (about 660 MiB) under the unchanged 16 GiB address-space
limit. The interval phase at ReLU71 is 2,415 negative, 3,831 positive, and 26
unstable rows. Propagation takes 1,835.642069 seconds; no second phase profile
exists. This closes version 1 as a scalable implementation and strengthens the
case for version 2: row filtering must occur before operator composition, not
after the P+U materialization has already been attempted.

### Post-capability residual guard queue

No broad replay is authorized until selective materialization and implicit
operators cross Tiny iid143 and preserve the CIFAR100-large iid166 terminal
checkpoint. The smallest subsequent structure-balanced guard is fixed in
advance as S6:

- CIFAR100-medium iid2 (retained ADV) and iid29 (retained CERT);
- CIFAR100-large iid118 (retained ADV) and iid113 (SAFE-side diagnostic);
- TinyImageNet-medium iid6 (retained ADV) and iid17 (SAFE-side diagnostic).

The isolated HZ arm may return `UNKNOWN` on the SAFE-side diagnostics, but it
must never return an invalid ADV or ERROR; the integrated verifier must retain
all six old solved outcomes. Only after S6 may the public-SAT UNKNOWN queue
expand in this order: Tiny iid153 and CIFAR100-large iid110, the official-UNSAT
large iid153 as a mandatory near-zero negative control, then large iid160,
iid161, and iid114. Tiny iid93 remains the final stop-loss target. This queue
does not alter either frozen ledger: 13-family formal remains 1,870/2,413 and
large-classification formal remains 59/400 until their respective full replay
gates pass.

## Goal lock: structure-by-structure Neural-HZ improvement

The active project goal is not a best-effort speed experiment. The sole formal
baseline is the previously reproduced 13-family result of **1,870/2,413**:
1,063 CERT results plus 807 concretely validated ADV results. Every candidate
is charged against the frozen per-family solved vector, not only the aggregate.
The headline may increase only when a complete 2,413-case replay preserves
every old CERT/ADV, has zero invalid ADV, has no family-level regression, and
adds at least one new CERT or validated ADV. UNKNOWN and ERROR never count as
gain. Disconnected probes, including the separate 59/400 classification
ledger, remain capability evidence only until their own promotion gate passes.

The research unit is one repeated network structure at a time. A candidate
must be a uniform rule selected from graph structure, exact HZ state, or LP
status; instance identifiers and per-case menus are forbidden. Each structure
class follows the same transaction:

1. state the exact set identity, trigger, refusal condition, and physical
   storage accounting before running the target;
2. implement it default-off and fail closed;
3. prove it on synthetic equality/feasibility/witness tests;
4. cross the pre-registered target boundary;
5. replay the same structural rule on a cross-family shadow cohort; and
6. replay each family and finally all 2,413 cases before promotion.

The representation must remain a genuine nonconvex Hybrid Zonotope throughout:
continuous factors, binary factors, equality and inequality predicates, shared
latent identities, and concrete-witness reconstruction are preserved. Binary
factors are not pivoted or silently deleted. Replacing the state with a
Zonotope, Constrained Zonotope, interval box, or another convex relaxation is
out of scope. Attack/PGD, BaB, input splitting, and backward/dual rescue cannot
be credited as a Neural-HZ representation improvement. Historical data under
`/data1/Kane/HyZor` is read-only; new records are written only below this
experiment directory with branch, commit, configuration, and candidate hash.

The current targeted sequence is fixed as follows:

- **Structure A -- sparse nonlinear frontier.** Separate stable-negative,
  stable-positive, and genuinely unstable ReLU rows, and materialize only the
  nonlinear frontier while retaining the exact affine positive path. The
  current gates are TinyImageNet iid143 at ReLU63, ReLU71, and the terminal
  output, followed by CIFAR100-large iid166/iid153 and S6.
- **Structure B -- implicit exact affine operators.** Replace expanded Conv
  CSR storage in the reachable HZ DAG by exact kernel/geometry/mask operators,
  with hash-consing, consumer-aware garbage collection, and unique physical
  byte accounting. It is integrated only after standalone dense/CSR equality
  tests pass and only when the entire reachable representation strictly
  shrinks.
- Later structures are selected from the first repeated, measured blocker left
  by the preceding class. No new class is allowed merely because one isolated
  instance is difficult.

The intended endpoint is a sequence of independently proven, reusable
Neural-HZ innovations that produces a material CIFAR100 and TinyImageNet gain
while preserving all 13 families. The aspirational score is 2,413/2,413, but
the formal score remains exactly 1,870 until the complete promotion criteria
above are met.

## Structure B, stage 1: exact implicit affine-operator foundation

The first standalone component for the phase-sliced implicit-stencil HZ DAG is
implemented in `act/back_end/hybridz_tf/exact_linear_op.py`. It is deliberately
not connected to propagation yet. `CSRLinearOp`, `DiagonalLinearOp`, and
`ImplicitConv2DOp` expose shape, resident and logical storage metrics, exact
matrix-vector action, and a capped `Q @ W` left composition. The convolution
descriptor stores the original kernel, NCHW geometry, grouped/depthwise
semantics, and an optional full-width row mask. Its normal matvec and
left-composition paths do not construct the full convolution CSR.

Seventeen focused tests compare ordinary, grouped, depthwise, batched,
strided, padded, dilated, and row-masked descriptors with the existing native
CSR builder. Dyadic cases require array-identical CSR, matvec, and composition
results; non-dyadic cases use `1e-13` absolute and relative tolerance. Invalid
geometry, non-finite inputs/results, and output nnz beyond the caller's cap
fail closed. A monkeypatch test forbids use of the reference expansion from
`left_compose`. The standalone suite passes 17/17.

For a representative `128 x 128`, `3 x 3`, `14 x 14` convolution, the
descriptor keeps 147,456 kernel coefficients (1,179,648 float64 bytes) while
representing 26,214,400 logical expanded nnz, a coefficient compression of
about 177.78x. These numbers are component evidence only: object/geometry
metadata, row masks, every reachable source/value/predicate object, and all
composed materializations must be charged by the eventual representation-wide
ledger.

Stage 2 is pre-registered to use a generic lazy-operator protocol. Appending an
operator propagates the bias with `op.matvec`; materialization starts from the
requested output-row mask and repeatedly calls `inner.left_compose(Q, cap)` in
reverse operator order. The opt-in deferred Conv-to-ReLU path must construct an
implicit convolution descriptor rather than a `P union U` CSR; otherwise the
existing Conv68 allocation happens before selective-U can act. Default/eager
CSR behavior stays unchanged. Promotion additionally requires content
hash-consing, reachability-based garbage collection, and separate unique
resident-byte versus logical-expanded-nnz ledgers. No integration or real
network claim has occurred at this stage, and formal gain is zero.

## Trial 8 version-2 ReLU63 capability result

The authoritative selective-materialization record is
`results/trial8_phase_selective__tinyimagenet_2024__iid143__relu63_v1.json`.
Its result SHA-256 is
`f1f6abcdc87d96e9d8898d0f667d8087b2a72a9898f44d61b4e123426d45b274`
and its candidate-source SHA-256 is
`9a040bfd2d8f61d3083c93c563ff559d39512e3403cb3c3b1ca6a115d2464b39`.
Under the unchanged 64M per-materialization and 16 GiB address-space limits,
TinyImageNet iid143 reaches the intentional layer-63 census stop in 272.133510
seconds. The record is `UNKNOWN`, as required for a non-terminal capability
stop.

Selective-U triggers uniformly at ReLUs 36, 44, 55, and 63; no frontier image
rebase or version-1 phase split fires. The exact layer-63 core retains
`13,702c/2,147b`, 2,147 equality rows, and 4,294 inequality rows. Its interval
partition is 2,646 stable-negative, 3,616 stable-positive, and 10 unstable
rows. Eight deterministic positive probe rows plus U prove a strict local
saving of 79,691 entries (`q=89,579`, mask/bias charge 9,888), and the resulting
nonlinear core occupies 2,154,136 entries. The earlier selective profiles are:

- ReLU36: N/P/U = 20,648/2,268/2,172, core 5,131,207;
- ReLU44: N/P/U = 1,972/3,032/1,268, core 6,349,836; and
- ReLU55: N/P/U = 3,137/3,084/51, core 2,383,916.

Propagation is about 2.50x faster than version 1's 681.002680-second ReLU63
record and about 3.02x faster than the earlier 821.881203-second
predicate-sharing boundary. This speedup is secondary; the important result is
that construction order now materializes only the exact nonlinear frontier
without changing factor widths, predicates, global neuron slot identity, or
witness semantics.

The global representation gate is still open. The layer-63 expression has 40
terms and 134,561,120 logical expanded operator entries. Moreover, the current
deferred Conv-to-ReLU implementation still constructs a `P union U` Conv CSR
before selective-U runs, which is the already measured ReLU71/Conv68 failure
shape. Therefore no redundant ReLU71 run is launched with this source hash.
Version 2 authorizes Structure B stage-2 integration: retain the same exact
phase rule while replacing those expanded Conv operators with tested implicit
kernel/geometry/mask descriptors and charging the full reachable physical
representation. Formal gain remains zero.

## Structure B, stage 2 implementation gate

The implicit operator is now connected behind the default-off
`sparse_implicit_conv_dag` flag. All three lazy Conv construction sites use the
same flag-aware helper: starting a lazy DAG from an exact HZ, appending Conv to
an existing affine expression, and the deferred Conv-to-ReLU expression path.
The eager/direct-HZ Conv path is unchanged. Deferred construction stores a
full-width row mask on the implicit Conv but retains the full bias vector;
stable-negative output is still removed only by the established exact ReLU
rule.

Lazy materialization now applies the exact generic composition
`D_keep @ W_k @ ... @ W_1`. SciPy CSR operators retain their native sparse
matrix multiplication, while an implicit descriptor receives the current left
factor through capped `left_compose`. Bias propagation uses exact operator
matvec. Descriptor construction is charged by resident payload, but every
composed CSR and final SparseHZ remains subject to the unchanged 64M nnz/storage
limit. The reference Conv expansion is never called by normal implicit matvec
or composition.

Content-identical descriptors are interned in a weak-value arena. The arena is
not a reachability root. A default-off consumer ledger removes sparse/lazy
cache keys only after their final graph consumer; INPUT/INPUT_SPEC states stay
pinned for witness decoding and the direct predecessor of ASSERT stays pinned
for terminal solving. Multi-consumer residual sources are retained until all
branches have consumed them. The isolated worker records both per-expression
logical-expanded and resident storage and a unique-live-cache upper-bound
ledger covering operators, expression biases, HZ value maps, predicates, and
phase bounds. Allocator/Python-object overhead is explicitly excluded and the
ledger is not yet called a complete RSS model.

Repeated downstream operator suffixes are composed once per materialization
through a strictly bounded reverse-prefix memo. Only prefixes used by at least
two terms are cached and total cached CSR nnz cannot exceed the ordinary
materialization limit; beyond that budget the path recomputes rather than
retaining additional memory. A `128x128`, `3x3`, `14x14` two-Conv microcase
with 40 identical term suffixes completes in 3.394 seconds, matching one
expensive suffix composition rather than forty.

The combined proof gate is 95/95 focused tests. It covers exact operators,
flag-off identity, all three flag-on lazy Conv sites, grouped/depthwise and
NCHW geometry, nonzero bias, row masks, full materialization equality, global
ReLU slots/predicates, weak interning, suffix-memo reuse/cap, fail-closed
capacity/error paths, phase Bounds plumbing, residual consumer GC, and the
live-cache ledger. No real-network result or score gain is claimed by this
implementation gate. The next pre-registered run is Tiny iid143 stopped at
ReLU36, followed only on success by ReLU63 and ReLU71.

## 2026-08-31 goal reset: structure-by-structure campaign

The active research goal is now explicitly governed by `GOAL_CHARTER.md`.
The formal baseline remains exactly `1,870/2,413` (`1,063 CERT + 807 validated
ADV`) and every one of the 13 per-family solved counts is a non-regression
constraint. CIFAR100, TinyImageNet, and every other non-13 family use independent
ledgers and cannot be added to the formal score.

The single active structure is S0: phase-sliced exact ReLU propagation plus an
implicit Conv DAG for repeated residual large-CNN blocks. Its measurement order
is TinyImageNet ReLU36 -> ReLU63 -> ReLU71 -> terminal, then the registered
CIFAR100 targets and cross-family residual/plain-CNN shadows. This order is a
measurement protocol, not an instance trigger; the candidate rule may inspect
only phase, operator, support, liveness, sharing, and proved resource state.
The earlier phrase permitting selection from "LP status" is superseded by the
charter: LP/MILP remains available as the ordinary terminal decision procedure
or post-run diagnostic, but its status, marginals, and dual ray cannot select
or repair a Neural-HZ representation branch.

The ReLU36 stage-2 smoke run remains active in the background under PID 251298
with output path
`results/trial9_phase_implicit__tinyimagenet_2024__iid143__relu36_smoke_v1.json`.
Its source-provenance files are frozen until the process exits. The output is
exclusive-created and will be checksummed and interpreted only after exit.
At the time of this goal reset it had not emitted a result; therefore it changes
neither a capability ledger nor the formal score.

### Large-CNN universe identity freeze

A deterministic generator now freezes all original referenced source assets
for the current vnncomp2025-root CIFAR100 and TinyImageNet families. The two
immutable manifests contain 200 rows each, use content-derived instance keys,
have zero duplicate key, and reproduce the registered `instances.csv` hashes.
Their file and ordered-universe hashes are recorded in `BASELINE_LOCK.md` and
`manifests/SHA256SUMS`.

This is deliberately not a baseline-verdict promotion. Every one of the 400
verdict fields is null and the status is `UNFROZEN`, because no admissible full
per-instance vector has yet been located. The historical 59/400 aggregate,
older ACT-copy iids, the partial eight converted v2 specifications, and
disconnected trials were not imported. Generator tests pass 11/11, including
content-change identity, path/symlink containment, invalid timeout, malformed
CSV, deterministic rendering, and atomic no-overwrite publication. Formal gain
remains zero.

The follow-up full conversion gate is now complete. `vnnlib_v2_full_v1`
contains exactly 200 converted specifications per family and publishes each
family only after all source hashes and exact token round trips pass. The
conversion manifests have file SHA-256 values
`7e0567d32799dddde3236b00eb862d1817e496ed80ec4a93dbaf99651cac365b`
(CIFAR) and
`eb012c46b3a4437847b24125c41907255c9bf452d4d3cddabe2305a823ee7539`
(Tiny). All eight earlier target conversions are byte-identical to the new
full-set files.

The first all-file parser invocation failed before parsing because the
standalone verifier did not put the ACT repository root on `sys.path`. It
changed no artifact. The verifier was fixed to require and validate an explicit
ACT root, and a regression test covers the failure. The combined manifest,
conversion, and verification suites now pass 31/31. The repeated read-only
closure run then verified 202/201 original assets, 200/200 converted specs, and
200/200 successfully parsed ACT queries for CIFAR/Tiny respectively.

A separate read-only provenance search found no defensible 59/400 row vector.
The closest archive is 61/375 (25 CIFAR and 36 Tiny concrete sidecars), and it
includes CIFAR iid166 plus Tiny iid153 that cannot be silently subtracted.
Exact evidence hashes and the non-import rule are now in `BASELINE_LOCK.md`.
Therefore this step closes source/spec identity but intentionally leaves the
current external baseline verdict vector `UNFROZEN`; formal gain is zero.

### Independent historical-witness evidence baseline E0

A fresh validator maps evidence only by exact model/spec content, recomputes
every winning witness through current single-thread CPU ONNX Runtime, and
evaluates the original VNNLIB assertion tree with a new evaluator that imports
neither ACT nor the historical SATSidecar. The evaluator supports strict and
nonstrict comparisons, Boolean composition, and basic arithmetic at literal
zero tolerance. A toy ONNX end-to-end plus malformed syntax, strict-boundary,
input-domain, content-tamper, count-gate, and ledger-closure tests pass 15/15.

The first CIFAR attempt stopped before publication because the historical
`x_star_sha256` is the contiguous NumPy payload hash, not the `.npy` container
hash. The validator was corrected to require both hashes separately and a
regression test covers the distinction. It then replayed all 25 CIFAR and 36
Tiny witnesses successfully. The first successful v1 ledgers were retained but
superseded because they did not record imported validator dependency hashes.
Version 2 repeated all 61 replays and records the complete three-file source
closure and runtime/session provenance.

The authoritative result is E0 = `0 CERT + 61 independently validated
historical-origin ADV + 339 UNKNOWN = 61/400`, with zero invalid ADV. All
stored historical logits are element-wise equal to current ORT outputs, and
the minimum UNSAFE margins are `9.352564811706543e-4` (CIFAR) and
`1.4238357543945312e-3` (Tiny). Exact ledger hashes and candidate comparison
semantics are in `BASELINE_LOCK.md`. Every historical-origin ADV explicitly
has `neural_hz_gain_credit=false`; E0 is neither historical 59/400 nor a
Neural-HZ score. The formal 13-family score remains 1,870/2,413.

### Goal/E0 governance closure and exhaustive formal structure map

The original goal charter and its hash were preserved. A dated E0 amendment
(SHA-256
`7c03d276a3ba9e41c8d00d87831888adea18914c19bc426e421c978b32c5085b`)
now states the distinction that was already present in `BASELINE_LOCK.md`: the
two 200-row source manifests still have null verdicts and `UNFROZEN` status,
while the separate v2 E0 evidence vector is frozen at 61 validated ADV plus 339
UNKNOWN. E0 is a retention anchor only and remains arithmetically separate
from the formal 1,870.

The E0 closure verifier was strengthened without changing either ledger. It
now recomputes every problem content key and checks the exact namespace, nested
key, legacy evidence origin, exact-unique match, zero gain credit and
verdict/historical-status relationship. Self-rehashed semantic tamper tests
cover each field. The updated verifier SHA-256 is
`e0b3490c409360bcf602f5df144da95bc09f69fcbf8420f4bad51d0d24e1731d`;
both real ledgers and all checksum manifests still verify.

The composite 13-family authority was then rebuilt into an exhaustive
content-addressed structure map. The published manifest
`manifests/formal_unsolved_structure_manifest_v1.json` has file SHA-256
`393590e26d3ae4edd50c9a7d2df48980c8b3701237b61c030bfbe4ad36947d8d`
and payload SHA-256
`dd7fd5e63b0b5a4a1b72911424326f338831b07eff0fb64192e18ab82c9dbef9`.
Its deterministic byte rebuild starts from all 2,413 rows, excludes the frozen
1,870 solved rows, and assigns all 543 remaining rows exactly once:

- A Conv/ConvTranspose--ReLU: 79 UNKNOWN + 49 TIMEOUT = 128;
- B TLL signed/symmetric ReLU: 15 + 0 = 15;
- C plain Dense FC--ReLU: 95 + 168 = 263;
- D shared Add/Concat/skip: 19 + 2 = 21;
- E attention/residual: 59 + 53 = 112; and
- F smooth tail: 2 + 2 = 4.

All 543 row labels and row-content identities are unique, and every family
vector matches `BASELINE_LOCK.md`. The audit corrects one earlier coarse
classification: LinearizeNN contains no Conv and belongs to D. A's current
ordinary `ImplicitConv2D` direct reach is 124 (MetaRoom 5 plus
ReluSplitter-CNN 119); cGAN's four large-image rows use ConvTranspose and are a
separate exact-operator extension. The target order and per-class HZ identities
are frozen in `STRUCTURE_CAMPAIGN_543_V1.md`, SHA-256
`6d1c93fca2af90abf2e2154dd45489c8336bdc1872d0700c4a822eb8f8747fce`.

### S0-C1 comparator and group-contraction audit

Before any production composed-stencil run, the representation comparator was
made explicit in a versioned amendment, SHA-256
`5b163a381c5be1f69d6421cc69be0cecd563c57c5921535bf9b21e98305b99d0`.
The combined phase-sliced/implicit/composed candidate must strictly reduce the
whole registered reachable physical state relative to the same phase-sliced
expanded-Conv path. Trial 9's implicit-unfused path is only an implementation
performance comparator; a faster composition against it is not by itself HZ
simplification.

A read-only production audit (SHA-256
`ae68e24cbd7d8064752afc1b3f66265530c1387d526656185ece3e0f77242905`)
found that the isolated prototype's dense full-channel tap multiplication
under-reports actual grouped/depthwise work by as much as `g_inner*g_outer`.
It also identified the unused transaction-contraction field, `repr`-based
content identity and an emission upper bound incorrectly named logical nnz.
These are migration blockers, not accepted production behavior.

The corrected isolated group-intersection oracle has SHA-256
`3fd9d8d91d8eda404c8d59967e08a375a29b821296b660e1989ca23128f2dbb0`.
It uses canonical ascending-middle-channel rank-one accumulation and asserts at
runtime that actual scalar products, the sum over group intersections and the
registered gate formula are identical. Ordinary, aligned/misaligned groups,
inner/outer depthwise multiplier, zero/negative scales, dyadic exactness and
non-dyadic determinism pass 11/11 tests without allocating full-channel tap
tensors.

The combined isolated manifest/evidence/composed/oracle suites pass 104/104.
No production source was changed during this work because Trial 9 still owns
the source freeze. At this record point PID 251298 had run for about 80 minutes
at full CPU, used about 1.16 GB RSS, and had not emitted its exclusive JSON;
it remains authorized to finish and retain its result automatically. Formal
gain remains zero and the headline remains exactly 1,870/2,413.

### S0-C1 isolated V2 hardening and Trial 9 exit custody

The first production-shaped V2 was kept experiment-only and subjected to a
second adversarial audit before any production integration. The original
source SHA-256 was
`c303f15ccd778379daa66af0462d603d2b4d05c132512267c661b4af6fd883a8`.
Its group-intersection algebra and NCHW geometry passed the original 23 focused
tests plus two independent seeded CSR sweeps, but five implementation blockers
were reproduced: sparse left multiplication checked its cap only after
allocation; interleaved reservations could corrupt transient-peak rollback;
`KeyboardInterrupt`/`SystemExit` could leak an open reservation; cache identity
trusted a constructor-time key for writeable Conv payloads; and a partial
commit was not fully removed by rollback.

The hardened isolated source now checks a conservative contribution bound
before sparse multiplication, permits one owner-bound open reservation,
prevalidates commit state, removes partial publication on rollback, and uses an
unconditional `finally` for all unclosed reservations. It snapshots current
Conv kernel/mask semantics, keys and compiles the same snapshots, and charges
their numeric buffers to controlled transient memory. A regression mutates a
Conv weight while its old source key remains unchanged and proves that V2
compiles a new exact descriptor instead of returning the stale cache.

The hardened source SHA-256 is
`7eb40c207bcc95be4c0109ef82ea4809d3f52f08dfa89366fad7a6a9fd06c6b0`;
the new adversarial test SHA-256 is
`1e87b63e23675d5adbb2de1b1ce752133f26dac89e3534a0a939b1322f03db32`.
The deterministic in-suite sweep checks exactly 58 nonempty valid-group
geometries. Original plus adversarial V2 tests pass 28/28, and every isolated
experiment test passes 154/154. A preliminary call through the standalone
`pytest` executable used the wrong interpreter and failed collection with no
test execution or file change; the authoritative runs use `python -m pytest`.
Full findings and non-claims are in
`S0_C1_ISOLATED_V2_ADVERSARIAL_AUDIT_20260831.md`, whose SHA-256 is
`eed6c2fe2965ec842464374ba1562f6bdab0a52ad39fb8458712d487c5d5eb28`.
The consolidated closure is `S0_C1_ISOLATED_V2_SHA256SUMS`.

Two limits remain deliberately visible. `exact_*_logical_nnz` is an exact
structural post-collision count and a safe upper bound, not canonical numerical
CSR nnz after zero/cancellation elimination. The controlled buffer ledger also
does not pretend to bound Python allocator overhead; real-network promotion
must separately record peak RSS and the entire unique reachable physical
state. The 64 MiB resident bound is per descriptor, so a future production
caller must include all already interned descriptors in
`reachable_after_other_bytes` or reject as physically unproven.

Trial 9 still owns the nine-file source freeze. Its worker PID 251298, bound to
start ticks 698719726 and start time 11:14:30 local, remained at full CPU after
about 107 minutes with RSS 2,006,984 KiB, below the 16 GiB gate. It had emitted
only its 888-byte load log and no result JSON. The first background sealer
launch was immediately reaped and left an honest zero-byte
`*.exit_seal_v1.monitor.log`; it touched neither result nor sidecar and is not
deleted. A second low-priority sealer, PID 1820591 started at 12:45:01, remains
bound to the original PID/start ticks and will publish an exclusive,
crash-atomic `*.exit_seal_v1.json` only after worker exit or PID reuse. Missing,
truncated, metadata-mismatched and valid results have distinct classifications,
and every classification carries formal gain zero.

No production source was edited in this checkpoint, no historical result was
overwritten, and no V2 real-network verdict exists yet. S0-C1 remains at the
mathematical/resource unit gate; production integration, Tiny ReLU36 and all
later shadows remain pending. The formal headline remains 1,870/2,413 and E0
remains 61/400 with zero Neural-HZ gain credit.

## 2026-09-05: current-state recovery, C3 zero-hit and TLL requalification

This entry supersedes the stale progress state above without rewriting any
prior attempt. The goal is active and incomplete. The formal baseline remains
1870/2413 and E0 remains 61/400; full 13-family retention has not been replayed.

Trial9 has naturally completed with its exclusive UNKNOWN JSON and validated
exit sidecar. Both hashes and all nine frozen sources still verify; its stop
and storage figures are recorded in `TRIAL9_RELU36_EXIT_AUDIT_20260831.md`.
The final isolated runtime adapter present in the worktree passes 264 tests.
It is not production integrated or a mathematical/verdict promotion.

A new generic necessary-condition preflight rebuilt the pinned Tiny143 graph
and BN clone. Of all four affine paths into ReLU36, three retain nested ADDs
even when every earlier RELU is granted a fresh source. The literal C3 grammar
therefore closes with zero hits, without an expensive HZ run or a jump to a new
target. The graph preflight tests pass 9/9. C1/C2 closures remain unchanged.

The loader now has `repair_batchnorm_producer_graph=False`, with no repair call
on its default path. The opt-in repair accepts only complete marked BN sibling
pairs, preserves all numeric payloads, validates variable flow and publishes
private edge maps only after every check. Its 13 tests include full affine
coefficient equality and preserved nonconvex HZ predicates/binary witnesses on
a dyadic residual model. Real Tiny143 repairs exactly 19 edges and has all 81
layer outputs byte-identical to the independent clone. This correctness work
earns zero HZ gain and has no default enablement or full-family result.

The complete isolated test run reports 734 passed, one failure. The live
Overleaf verdict table changed only revision coloring/whitespace. Its exact
original bytes were recovered from read-only Git commit
`01923a5bc896667bd0d0a19b310ee52e88bb2b70`, matching the original pinned hash.
The other four authority files match, and a separate input-adapted replay of
the untouched generator reconstructs the entire frozen manifest byte for byte.
The original live-path test is preserved and continues to reject changed table
bytes. Details and immutable evidence are in the C3/loader audit.

After C3 closure, the existing TLL signed/dead-ReLU candidate was requalified
over all 32 rows under one fixed current source, sparse representation,
45-second solver limit, 16 GiB per worker and four-way concurrency. V1 failed
all 32 loads because the converter now stores TLL weights only as buffers.
The original worker assumed at least one parameter. A separate V2 worker fixes
only dtype discovery and source provenance; all four dtype tests pass and all
V1 bytes/results remain preserved. V2 then finished every row with no drift.

V2 raw results are 11 CERT, 16 concrete-valid ADV and 5 UNKNOWN: 27/32, retaining
all 17 formal solved rows but losing prior candidate iid8 and iid26. iid8 has
unchanged lowered size but reaches the 45-second budget. iid26's proposed
input is one floating step below its box lower bound, so the worker rejects
it as UNKNOWN. The supervisor's `invalid_adv=1` counts this rejected proposal,
not an invalid reported ADV; the latter count is zero. All 16 actually reported
ADV pass independent CPU ONNX and original-VNNLIB replay at zero tolerance.
Qualification fails; no union with prior positive records is credited.

The complete record and remaining controlled diagnosis are in
`TLL_CURRENT_SOURCE_REPLAY_AUDIT_20260905.md`. There are no unfinished jobs at
this checkpoint. New sources and evidence are sealed by
`CHECKPOINT_20260905_SHA256SUMS`; the earlier consolidated manifest is preserved
as historical rather than silently rewritten.
