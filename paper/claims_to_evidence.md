# Path-Conditioned Verification for Routed Mixture-of-Experts

## Abstract

Formal verification of routed mixture-of-experts models is often reduced to a
router-invariance check, even though route stability is sufficient rather than
necessary for output robustness. Across 20 exact official full-model router
initializations, route-invariance applicability at \(8/255\) is empty for 18
seeds and leaves at most two of 10,000 test inputs for either remaining seed.
We present a staged path-conditioned verifier
that preserves tie-inclusive route conditions in a shared abstract frame,
tightens downstream expert supports, and checks every feasible route branch. A
cheap first tier eliminates normalized output gates by convexity. When that
sufficient condition is inconclusive, a second tier uses a property-directed
expert-disagreement decomposition and range-only McCormick relaxation without
symbolically encoding exponentiation, division, or sigmoid segments. On an
independent 100-sample route-boundary cohort of a verification-scale CIFAR-10
weighted top-2 MoE, the method certifies 36 route-changing samples, while exact
route analysis reduces ordinary-zonotope candidates on 75/86 route-unstable
rows and route conditioning yields a median structural-width ratio of 0.430.
The immutable preregistered overall solved endpoint is 56/100 and fails its 60%
gate; boundary applicability is 76/100 and conditional coverage is 56/76. We
separately report follow-up closure and engineering results without rewriting
that endpoint. Official-scale expert certification remains pending while the
official-code reproduction runs. The follow-up explicit end-to-end
route-invariance comparison is complete and independently audited.

## Draft status and evidence boundary

This is a claims-to-evidence paper draft, not a new experimental protocol. It
is grounded in the repository state at
`7ef324e18`. The frozen scientific endpoint,
follow-up closure experiments, and engineering reruns are reported separately.
No pending result is inferred from an implementation, a protocol, or a smoke
test.

The current empirical model is a verification-scale CIFAR-10 weighted top-2
MoE with a `3072 -> 128 -> ReLU -> 8` router, selected-softmax gating, and eight
MLP experts. Its frozen checkpoint is
`data/moe/checkpoints/cifar10_top2_e8_seed0_bal010.pt` with SHA-256
`fbaa7c871d28763ac5acb29a9502dc5d146e1d5af0b4a03e9911899251bd43f7`.
Its test accuracy is 49.26%. It is a controlled verification benchmark, not a
claim about modern production-scale sparse MoEs.

## Contributions supported by the current artifact

1. A tie-inclusive, path-conditioned verification architecture that retains
   route guards in the shared abstract factor frame and enforces conservative
   three-valued verdict semantics.
2. A normalized top-k decomposition requiring \(k-1\) property-directed
   gate--expert products, with selected-softmax top-2 F0 as the evaluated
   instance.
3. A guarded-support mechanism and a retained-affine-margin refinement, with
   explicit fallback rules that preserve soundness under solver limits.
4. An independent verification-scale evaluation of candidate reduction,
   structural width separation, route-changing certificates, F0 attribution,
   and guard-aware paired coverage.
5. A tie-inclusive implication counterexample and a sound eta-shifted compiler,
   plus a partial primary-source artifact survey whose retrieval limitations are
   reported rather than converted into prevalence claims.
6. A 200,000-decision official-construction census showing that hard-route
   certificate applicability is nearly empty at \(8/255\) and varies by an
   order of magnitude across unseeded initializations at \(2/255\).

## Problem statement

Router invariance is a convenient sufficient condition for verifying a routed
model: if one route is fixed throughout the perturbation region, verification
reduces to the selected expert. It is not a necessary condition for output
robustness. A perturbation set may intersect multiple legal routing regions
while every reachable routed output still satisfies the safety property.
Rejecting such a region at the router boundary leaves valid output certificates
unproved.

We study output-level MoEs under tie-inclusive routing. The verifier must cover
every legal unordered top-k route set, retain the corresponding route condition,
and prove the output property on every feasible branch. For weighted normalized
top-k gates, the method first tries a cheap expert-wise convex certificate and
then models only the property-directed expert disagreement when that sufficient
condition is inconclusive.

The resulting system is best described as a staged, path-conditioned analysis
for data-dependent dispatch:

1. compute a correlation-preserving reachable router abstraction and exact
   tie-inclusive feasible route sets when the router HZ remains unrelaxed;
2. retain each route guard in the shared input factor frame;
3. use guarded support to tighten expert propagation;
4. apply expert-wise gate elimination as Tier 1;
5. invoke a normalized-gate, property-directed McCormick fallback as Tier 2;
6. report `UNSAFE` only after a concrete input violates the full routed model.

## Mechanism and soundness argument

### Retained route conditions

For a feasible unordered top-k set \(S\), the verifier forms a guarded domain
\(X_S\) by intersecting the input abstraction with all weak inequalities needed
for membership in a legal top-k set. Weak inequalities deliberately include
ties. Both router and expert propagation refer to the same factor frame, so
input generators and router constraints remain shared while expert-local ReLU
binaries remain disjoint.

Candidate exactness is a conditional label. It is used only when the reachable
router HZ has not undergone relaxation. Coordinate-wise IBP and ordinary
zonotope candidates are sound upper sets, but they may lose the correlations
needed to exclude infeasible experts or route sets.

### Tier 1: convex gate elimination

For normalized non-negative weights, if every feasible selected expert satisfies
a convex linear output property throughout its guarded branch, their convex
combination satisfies it as well. This removes the gate value from the checked
obligation, but it is only sufficient: failure of one expert obligation is
`UNKNOWN`, not `UNSAFE` for the weighted model.

### Tier 2: normalized top-k disagreement decomposition

For a canonical anchor \(b\in S\), normalized non-negative weights give

\[
F(x)=E_b(x)+\sum_{i\in S\setminus\{b\}}
\lambda_i(x)\bigl(E_i(x)-E_b(x)\bigr).
\]

For one linear safety row \(q^T F+c\ge 0\), the encoding therefore needs
exactly \(|S|-1\) property-directed products
\(w_i=\lambda_i d_i\), where
\(d_i=q^T(E_i-E_b)\). Sound gate boxes, expert-difference supports, McCormick
hulls, and the simplex-intersection constraint on the omitted anchor form an
outer relaxation. A strictly positive, outward-corrected lower bound proves
safety. A non-positive relaxation optimum is only a candidate and cannot prove
unsafety.

Selected softmax and normalized sigmoid satisfy this gate-family contract;
hard top-1 is the zero-product special case. Unnormalized `switch_prob` is
rejected because it needs an additional scale product. The proof and gate-family
scope are frozen at commit
`a054ba382bb2cd02a6e4ed297944da9d2fbd98a3` in
`act/back_end/moe/proofs/normalized_topk_decomposition.md`.

The frozen five-sample N2 cohort executes this path on an alternate
normalized-sigmoid top-3 model. It exactly enumerates eight tie-inclusive route
sets and evaluates 55 fallback property rows, every one with exactly two
property-directed products. Sample outcomes are one SAFE, two independently
replayed UNSAFE, and two solver-limit UNKNOWN. An external audit rebuilds all
five router HZs, reproduces all eight route sets, checks the runner manifest,
and reports zero issues. This is mechanism-plus-engineering evidence for
non-top2 and alternate-gate execution, not a prevalence or scalability claim.
The artifact is
`act/pipeline/moe/results/experiment1n2_top3_seed0_r2_20260830.json`, committed
by `a44f64867f450d471860c4575e4e6bc084af7d71`.

For selected-softmax top-2, Tier 2 reduces to

\[
F=E_b+\lambda(E_a-E_b),\qquad
\lambda\in[\sigma(\underline m),\sigma(\overline m)],
\]

where \(m=r_a-r_b\) is bounded under the pair guard. F0 uses only this numeric
gate interval and one property-directed McCormick product. It does not encode
exponentiation, division, sigmoid segments, or the symbolic nonlinear gate
function.

### Guard-aware support

Adding a retained path condition produces a subset of the original domain.
Consequently, its downstream lower support cannot decrease and its upper support
cannot increase. ReLU binaries are allocated only after support is recomputed
on this guarded domain. Unknown or timed-out support calls fall back to sound
unconditioned bounds; they cannot create a certificate. The formal monotonicity
and closed-interval coverage argument is in
`act/back_end/moe/proofs/conditioned_support_monotonicity.md` at commit
`f2500b935bd9527a025c5f7e126a8c504b8b6bea`.

N1 applies the same principle after partitioning an affine router margin into
closed intervals. Adjacent intervals overlap at cuts, so ties remain covered.
N1 does not segment or encode sigmoid; every active segment must be safe.

### Verdict discipline

`SAFE` requires every feasible route set and every property row to have a
solver-optimal, finite, outward-corrected lower bound strictly above `1e-7`.
The registered feasibility and integrality tolerances are `1e-7`; the bound is
corrected by an absolute-plus-relative `1e-9` term and `nextafter` toward
negative infinity. A primal incumbent is never a safety certificate.

For weighted top-k, an expert violation or a negative McCormick relaxation is
`UNKNOWN`. `UNSAFE` requires recovery of a concrete input inside the perturbation
set and replay against the complete selected-softmax model. The confirmatory
audit replayed 20/20 unsafe witnesses; Experiment 1D replayed 2/2 new witnesses;
the N1 engineering audit replayed 3/3.

## Confirmatory evaluation

### Cohort and endpoints

The independent confirmatory cohort is the 100 deterministic clean-correct
CIFAR-10 ranks 100--199. Ranks 0--99 were used for development and are not
confirmatory. The fixed-radius census evaluates
\(\epsilon\in\{0.25,0.5,1,2\}/255\) without output-property solving. The
route-boundary endpoint evaluates one predeclared radius per sample,
`1.05 * certified route-unstable upper endpoint`, with a route search capped at
`4/255`.

The audited confirmatory implementation HEAD is
`1a67922c43f4e21f526e3aa12ef7b2f4e3242cba`; its result manifest is
`act/pipeline/moe/configs/experiment1_confirmatory_protocol_manifest_r1.json`,
committed by `45375d287162f17e6c1bb1168bfc16e6dd10d9b3`.

### Candidate reduction

Among 86 route-unstable fixed-radius rows, the exact, unrelaxed router HZ
candidate set was strictly smaller than IBP on 83 rows (96.5%; sample-cluster
bootstrap 95% interval 91.3%--100%) and smaller than the ordinary zonotope set
on 75 rows (87.2%; 77.8%--95.6%). This supports correlation-preserving route
analysis on the tested nonlinear router. It does not claim that arbitrary HZ
pipelines remain exact after relaxed operations.

### Binary-width separation

On the same 86 route-unstable rows, the ratio between the maximum
route-conditioned branch width and candidate-pruned structural monolithic width
had median 0.430, IQR 0.386--0.473, and 90th percentile 0.530. This supports the
structural consequence of decomposition when multiple experts are feasible. It
is not a monolithic runtime comparison: a true monolithic router-dispatch-expert
solver has not yet been executed.

### Route-changing certificates and the immutable endpoint

The route-boundary experiment certified 36/100 samples as `SAFE` despite failure
of the exact route-invariance precondition (36.0%; Wilson 95% interval
27.3%--45.8%). Five certificates came from Tier 1 gate elimination and 31 from
F0. The denominator is the full predeclared route-boundary cohort; this is a
route-boundary certification yield, not natural-input prevalence.

The preregistered overall solved-rate endpoint was **56/100 = 56%**, below its
60% threshold. This failure is immutable. Its composition is 36 safe, 20
full-forward-validated unsafe, 40 unknown, and four timeouts. Separately, a
route boundary was established within the frozen search cap for 76/100 samples,
so confirmatory conditional verification coverage is **56/76 = 73.7%**. The 24
samples with no boundary through `4/255` are boundary-inapplicable, not solver
failures, but they remain in the original denominator.

### F0 incremental contribution

Tier 1 left 60 semantic-incompleteness rows. F0 resolved 43/60 (71.7%): 31
additional safe certificates and 12 concrete full-model unsafe witnesses. Its
paired runtime overhead had median 28.1 seconds, IQR 7.4--63.5, and p90 115.1
seconds. This result establishes F0 as a core second tier, but does not establish
an end-to-end speedup.

### Guard-aware support

In the confirmatory execution, guarded support reduced expert binaries from
10,076 to 8,466: 1,610 eliminated (16.0%) across 356/903 branches. Its accounting
identity closes as 1,183 LP-support eliminations + 380 MILP-support eliminations
+ 47 structural/propagation eliminations. The full reduction must not be
attributed to support optimization alone.

The later matched Experiment 1D table over 225 branches is
`n00=21, n01=17, n10=3, n11=184`, where `n01` is support-only solved and `n10`
is no-support-only solved. The net gain is 14 branches and the exact two-sided
McNemar/binomial p-value is 0.00258. Median paired support-minus-no-support solve
time is -0.069 seconds. This supports a coverage benefit and an association with
binary elimination; it is not an unconditional runtime-speedup claim.

## Follow-up applicable-unresolved closure

Experiment 1D froze all 20 applicable but unresolved confirmatory rows and reran
only unresolved property branches under the unchanged encoding and radius with
a 900-second deadline. The audited clean run at implementation HEAD
`5f1b15ad202bb2b55b094390a00b8b63aaaf08b1` ran all 20 and resolved 12: ten safe
and two full-forward-validated unsafe. Seven remained unknown and one timed out.

This raises applicable conditional coverage only as a separately labeled
follow-up from 56/76 to **68/76 = 89.5%**. It does not backfill or replace the
failed 56/100 confirmatory endpoint. The frozen manifest is
`act/pipeline/moe/configs/experiment1d_bal010_manifest_r2.json`, committed by
`33e404104699b89ac3a942c835e8bd9512034a5f`.

## Engineering reruns and attribution

### N1 retained-margin conditioning

The N1 experiment is an
`engineering_performance_rerun_not_confirmatory_overwrite` on the same frozen
20 applicable-unresolved rows. The unsegmented D0 baseline solved 12/20; N1
solved 14/20. The paired table is `n11=12, n10=0, n01=2, n00=6`, with exact
two-sided McNemar/binomial p=0.5. N1 added one safe result and one replayed unsafe
result. Eleven of 157 evaluated properties had at least one strictly tightened
expert-difference support.

Median paired runtime increased by 15.91 seconds (median ratio 1.212). These
data support a small observed coverage gain and the retained-condition mechanism,
but neither statistical superiority nor a runtime improvement. The immutable
summary is `act/pipeline/moe/results/experiment1n1_engineering_20260829.json`,
committed by `747c3e1d11bf6069e2c2c63715aac303bc6c8e5e`.

### Incremental HiGHS guarded-box construction

The paired guarded coordinate-box benchmark contains 43 exact feasible route
branches from the frozen 20-sample selection. Both incremental HiGHS and the
SciPy reference solved 264,192 coordinate objectives with zero fallback sides;
all 43 paired hulls agreed within `1e-8`, with recorded maximum difference zero.
Incremental HiGHS used 43 model builds and took 230.83 seconds. SciPy used
264,192 model builds and took 3,491.51 seconds. The paired branch speed ratio has
median **15.03x**, and the aggregate wall ratio is 15.13x.

This is a descriptive engineering speed result for guarded coordinate-hull
construction. It is not an end-to-end verification speedup and does not claim
solver-internal warm-start behavior. The result is
`act/pipeline/moe/results/guarded_box_hull_benchmark_20260829.json`, committed by
`3943450fddcaa416b0fbe76779fdbcad93c3cb14`.

### Incremental HiGHS end-to-end engineering rerun

The same incremental backend was then connected to guarded support, expert
properties, and F0 and run on all 20 frozen D0 rows. The dedicated backend
audit reports zero issues and independently replays both UNSAFE witnesses. The
paired result is negative for performance: D0 solved 12/20 and incremental
HiGHS solved 13/20, with one `UNKNOWN->SAFE`, no solved regression, and one
`UNKNOWN->TIMEOUT`. Total time increased from 5,151.10 to 5,674.07 seconds
(1.102x); the paired median delta was +9.88 seconds, with 6/20 faster and 14/20
slower.

Telemetry records 292 sessions/builds, 3,402 solves, 12,348 row updates, 47
budget extensions, zero build failures, and exact agreement between 75 accepted
time-limit warnings and 75 `kTimeLimit` model statuses. This shows that model
reuse is real and soundly controlled, but that the 15.03x coordinate-hull
microbenchmark does not transfer to end-to-end expert/F0 solving. Incremental
HiGHS remains opt-in; no end-to-end speedup is claimed. The tracked artifact is
`act/pipeline/moe/results/experiment1_highspy_incremental_engineering_r4_20260830.json`,
committed by `235b19aff348f156721deafed34471d3b37ad498`.

## A tie-inclusive backend pitfall

For hard top-1 branch \(i\), let

\[
g_i(x)=\max_{j\ne i}(r_j-r_i),\qquad
s_i(x)=\min_k(C_kE_i+d_k).
\]

Under tie-inclusive semantics, branch \(i\) is legal when \(g_i\le0\). The
apparently natural reduction `max(g_i,s_i) >= 0` is unsound: at a legal tie,
`g_i=0` makes the reduction pass even when `s_i<0`. Repeating this reduction for
every member does not repair the omission.

For any \(\eta>0\), `max(g_i-eta,s_i) >= 0` is sound. Its exact additional
obligation domain is \(0<g_i<\eta\); \(g_i=0\) is a real legal obligation, not
an incompleteness case. The proof is in
`act/back_end/moe/proofs/tie_safe_eta_implication.md` at commit
`7bf8a388427035e6b51cb0ec56ad3bd9f3640861`.

Pinned auto_LiRPA 0.7.2 toy conformance checked four analytically constant cases,
including the unsafe legal tie, and passed all four. This validates graph
lowering and tie semantics only; it is not an official-model certificate or a
general numerical-soundness validation of CROWN. The artifact is
`act/pipeline/moe/results/crown/tie_safe_toy_conformance_20260829.json`, committed
by `6b4f2627eed465090894da851e013c8a950da4dc`.

## Audited certification-gap case series

The frozen survey protocol produced a reconciled partial corpus and a
primary-source evidence matrix, but source-native retrieval could not be
completed. We therefore close it as an audited case series rather than promote
it to a prevalence study. Two reviewers screened the same 321-record partial
corpus; nine families received full-text review and eight were retained for
six-dimension artifact extraction. All 48 cells have a primary-source URL and
locator, and the independent completeness audit found zero issues. One-hop
snowballing added 13 non-seed candidates and no new included family under the
frozen criteria.

These counts support only an artifact-centered qualitative account of the eight
already adjudicated families. They do **not** support ecosystem prevalence,
artifact-availability prevalence, certification-practice prevalence, search
recall, or a claim that no other eligible work exists. Source-native exports or
citation membership remain blocked or incomplete for ACM, IEEE metadata, PMLR,
CVF, USENIX, Springer metadata, OpenReview, and Semantic Scholar. Zero structured
citation counts are not evidence of no citations. No authors were contacted.

The partial matrix and snowball artifacts are:

- `act/pipeline/moe/results/survey/evidence_matrix_partial_20260829.json`, commit
  `16ea290f3d85be21d5491ae7e0e0e204445d5d81`;
- `act/pipeline/moe/results/survey/snowball_partial_20260829.json`, commit
  `bcaee278e76a5d4970557d14f9d43df3123eb079`.

The source artifacts retain their `PARTIAL_RETRIEVAL_NO_PREVALENCE` labels; the
reporting decision is frozen in
`act/pipeline/moe/results/survey/case_series_closure_20260830.json` as
`AUDITED_CASE_SERIES_CLOSED_NO_PREVALENCE`. No author contact has been sent.

## Executed comparisons and pending evaluation slots

The remaining empty positions are kept explicit rather than inferred from
mechanism tests.

### P0a: explicit end-to-end route-invariance baseline — complete

P0a ran on all confirmatory ranks 100--199. Exact tie-inclusive feasible
unordered top-2 set uniqueness held for 24/100 samples and failed for 76/100.
The explicit route-invariance baseline solved 12/100 (2 SAFE and 10
full-forward UNSAFE), while staged Route A solved 68/100 on the same endpoints
(38 SAFE and 30 full-forward UNSAFE), a paired coverage gain of 56 samples.
All 36 route-changing SAFE samples are Route A-only. The independent audit
reports zero issues and replays all 30 Route A UNSAFE witnesses through the
complete selected-softmax model.

Accounted time is 1,956.5 seconds for route invariance and 8,836.5 seconds for
Route A. This is a coverage-cost result, not a Route A speedup: the baseline is
cheaper because it abandons the 76 route-unstable endpoints. The comparison
artifact is
`act/pipeline/moe/results/route_invariance_baseline_confirmatory_20260829.json`
at commit `2172a381e`. It does not overwrite the immutable confirmatory
`56/100` endpoint.

### P0b: four-adapter consistency cohort — complete engineering result

P0b is frozen to 43 bal010 route branches and four configurations:

1. HZ with the retained guard;
2. guarded coordinate box hull passed to CROWN;
3. original input box passed to CROWN;
4. tie-safe eta-reduction passed to CROWN.

All 43 branches and 86 expert obligations completed without a branch error;
the runner audit and a separate raw-JSONL audit both report zero issues.
Guarded-box and original-box CROWN certified the same four pairs and the same
29 expert obligations. Across 774 property rows the paired lower-bound
difference is only micro-scale (median (-2.38\times10^{-7}), range
([-4.77\times10^{-6},2.62\times10^{-6}])) and is not outward-rounded. We
therefore report no guard-box tightening or runtime-speedup claim.

For one route halfspace intersected with a box, the coordinate-hull proposition
in `act/back_end/moe/proofs/guarded_box_coordinate_hull.md` gives the exact
condition under which each coordinate face survives: the adverse contribution
of that coordinate must be absorbable by the best compensation from all other
coordinates. Hence an oblique guard may remove a large part of the box while
leaving its coordinate hull unchanged. Multiple guards still require a joint
LP; independent face tests are not composed unsoundly. The proposition explains
the P0b geometry but does not claim that high-dimensional hulls are always
unchanged.

The tie-safe eta reduction certified no pair. Retained-guard exact HZ also
certified no complete pair at the frozen budget because 71/86 expert
obligations were solver-incomplete; only 15 completed exactly. P0b therefore
does not establish a complete backend ordering. It shows instead that a
coordinate-box adapter erased the route guard's certificate value on this
cohort, while the sound eta output reduction was too loose under CROWN. The
artifact is
`act/pipeline/moe/results/crown/crown_adapter_consistency_bal010_43_r2_20260829.json`.

### B1: official-code RT-ER reproduction — training running, endpoint pending

Before final expert checkpoints are available, the frozen affine router admits
an exact initialization-distribution study. For seeds 0--19, the census builds
the complete official four-ResNet18 model before reading the router, thereby
preserving the official random-number consumption order. Seed 0 matches the
epoch-10 training checkpoint router bit for bit. Across 200,000 test-input/seed
pairs, exact five-radius classification has one undecided bracket at \(0.5/255\)
and none at the four larger radii; the independent audit reports zero issues.

At \(2/255\), formal route-invariance applicability averages 5.038% and ranges
from 1.25% to 12.98%, a 10.384x initialization spread. At \(8/255\), 18/20
initializations have an empty applicability set; seeds 4 and 9 retain one and
two of 10,000 inputs. The 20-seed global maximum route-boundary radius is
9.754/255 at seed 9, sample 3444, whereas seed 0 reaches only 6.252/255. Thus
the correct conclusion is *near-empty applicability*, not that every observed
boundary lies below \(8/255\).

Because the released scripts do not seed or optimize the hard router, the
10.384x spread is an initialization-lottery result for certificate
applicability. It is not an output-robustness or trained-expert certificate.
For the released hard-argmax artifact reading, the theorem-applicability tree
is therefore evaluated by radius: at \(8/255\) the route-stability premise is
not established for all but three sample-seed pairs; at \(2/255\), constants
providers remain relevant only on the roughly 5% applicable subset. This does
not silently transfer the hard-argmax census to a distinct continuous-gate
semantics reading.

The audited census and paper-figure records are
`act/pipeline/moe/results/icml2025_rt_er/router_init_census_k20_20260830.json`
and
`act/pipeline/moe/results/icml2025_rt_er/router_init_figures_k20_20260830.json`.

The same official-construction census was independently repeated on the
released TinyImageNet `MOE_ViT` pipeline, using its 150,528-feature hard router
and all 10,000 validation images without labels. Released-runtime float16
resize centers are materialized in the pinned Blackwell environment before ACT
performs real-affine support calculations. Across 200,000 seed-input pairs,
the primary post-resize route-stable set contains 704 pairs at 0.5/255, one at
1/255, and none at 2/255, 4/255, or 8/255. Independent audit reports zero
issues. This extends the applicability-collapse observation across two datasets
and two expert architecture families; it remains router geometry, not an
expert or output certificate.

The secondary composition to raw 64x64 pixels is reported separately because
the input metric changes: its mean stable fractions are 45.146%, 17.1955%,
1.98%, 0.0335%, and 0% across the five radii. Forty-two of 200,000 literal
float16-normalization clean routes differ from the real-affine abstraction,
but none intersects any reported formal-stable endpoint. Two earlier runs are
retained and excluded: one used the wrong real-resize center, and one exposed a
PyTorch 2.9.1 versus 2.11.0 float16-resize kernel difference before producing a
census row. The final evidence is
`act/pipeline/moe/results/icml2025_rt_er/tinyimagenet_router_census_k20_20260830_r2.json`
and
`act/pipeline/moe/results/icml2025_rt_er/cross_dataset_router_census_figure2_20260830_r2.json`.

The cross-dataset shift is also tested with a separate quantity whose scope
matches the initialization argument: the unbounded local affine radius in each
router's normalized input coordinates.  The dimension-only prediction is

\[
\sqrt{150528/3072}=7.000.
\]

Because the clean-margin scale also depends on the normalized input second
moment, substituting the measured moments (1.555 for CIFAR-10 and 0.928 for
TinyImageNet at the released literal centres) predicts a 9.063x CIFAR-to-Tiny
median shift.  The observed aggregate median shift is 8.966x, 0.9894 of that
prediction.  An independent audit reports zero issues and a maximum error of
`6.23e-17` over 160 scalar formula replays.  This supports a standard-init
order law with two empirical points; it is not a universal theorem, an exact
box-capped radius, or an output certificate.  The raw-64 fold is the same
function under another metric and is not counted as a third dimension point.
The audit is
`act/pipeline/moe/results/icml2025_rt_er/router_dimension_law_20260830.json`.

The excluded preprocessing runs establish a separate artifact-semantics
finding.  Replacing the released float16 resize by a real-arithmetic centre
changed 111/200,000 clean routes, and nominally identical float16 antialiased
resize transforms under PyTorch 2.9.1 and 2.11.0 had different range behavior
(the former reached 255.125).  B3 consequently binds exact runtime versions,
source/checkpoint/router/data and input-order hashes, preprocessing
graph/order/dtypes/constants/domain, and solver/outward policy into every
certificate identity.  This is an executable fail-closed manifest requirement,
not merely a reporting recommendation.  Its paper-facing scope is in
`paper/certified_artifact_identity.md`.

The official repository is frozen at
`30ef94d77b5451595b82e739aa8938e1f4c4521f`. Its exact author-pinned environment
imports, but PyTorch 2.4.0+cu121 supports CUDA architectures only through
`sm_90`, while the available Blackwell GPU is `sm_120`; the first CUDA tensor
kernel fails. That exact-pin incompatibility remains an artifact-rot result.
An isolated Blackwell-compatible environment separately passed a deterministic
synthetic FFCV forward/backward smoke on the official MoE-ResNet18 architecture;
it requires an explicitly recorded JPEG-library preload and is labeled
`official-code, Blackwell-compatible deps + FFCV`. The seed-0 130-epoch
official-code reproduction is running; epochs 10, 20, and 30, their checkpoints,
and drift-guard telemetry are complete.  Epoch 30 validation is 37.40% clean
and 28.32% robust accuracy; the exact route geometry remains unchanged and its
100-sample reference replay passes with maximum radius error `1.21e-16`.
These are interim values, not a final reproduction endpoint.  Theorem
instantiation and official-scale
expert verification have not started. The exact-pin artifact is
`act/pipeline/moe/results/baseline/icml2025_rt_er_author_pin_probe_20260829.json`
at commit `291c6dfee9d68c6f025e240013f95b7ecbb1ab8e`; the compatibility smoke is
`act/pipeline/moe/results/environments/rt_er_blackwell_compatibility_smoke_20260829.json`
at commit `5ba952d1d`.

Any later Blackwell-compatible run must be labeled separately from the exact-pin
probe. The official project releases training/model code but no checkpoint and
no certificate implementation, so future models must be labeled
**official-code, paper-config reproduction**, not author checkpoints. No final
B1 accuracy, expert certificate, or runtime conclusion is claimed here.

## Learned routing in other official pipelines

A pinned source-to-loss-to-optimizer audit prevents the RT-ER artifact finding
from becoming an unsupported field-wide statement. At official robust-moe-cnn
commit `c50796fb8284512b6f6ad8e843f95182cec527cf`, the released trainer uses a
separate router optimizer, supervised router cross-entropy, adversarial router
KL, and an explicit straight-through backward for hard top-1 selection. At
official V-MoE commit `c07681241f81ba11421ba98e523e1499b2738a79`, the published
E=8, K=2 configuration creates Dense gates, uses selected gate values to combine
expert outputs, adds positive importance/load losses, and differentiates the
complete parameter tree without a router freeze rule.

Thus two of the three audited official pipelines expose learned router paths;
the pinned RT-ER release is the static case. This strengthens external validity
for learned routing while narrowing the criticism to the RT-ER artifact rather
than static routing or MoE releases generally. The mechanisms are not
interchangeable: robust-moe-cnn is a shared convolutional hard top-1 model and
V-MoE is hidden-layer weighted token routing, whereas Route A's main evaluation
is output-level weighted top-k.

The independent audit has zero issues across 26 hashed source anchors:
`act/pipeline/moe/results/published_moe_router_gradient_audit_20260830.json`.
Only RT-ER has dynamic tensor and optimizer-state evidence in this project. No
accuracy, robustness, checkpoint, or certificate result is inferred for the
other pipelines; robust-moe-cnn source is not copied because no license was
located.

### AdvMoE is a deep-path target, not a hidden-state router

A follow-up architecture and optimizer-schedule audit corrects a planning
assumption that would otherwise overstate generality. In the official CIFAR-10
ResNet-18 configuration, AdvMoE's convolutional router consumes the image
tensor before the dense ResNet stem. One PyTorch hard-argmax top-1 decision is
shared by 16 hidden MoE convolutions across eight BasicBlocks. The resulting
verification object is a full route-specialized deep pathway, not a prefix-HZ
followed by a hidden-state router.

The router is learned, but its update path is also narrower than a casual STE
reading suggests. The main optimizer is created before router attachment and
contains no router parameter. Although classification backpropagation creates
nonzero STE gradients on all 59 router parameter tensors, the main optimizer
changes none and the subsequent router `zero_grad` clears those gradients. The
separate supervised/robust router objective then changes all 59 tensors in the
synthetic schedule control. Thus the paper may call AdvMoE an explicitly
learned hard router; it must not claim that classification STE performs the
released router update.

The independent replay reports zero issues over 34 hashed source anchors and
confirms router input/output shapes, 16 routed layers, one shared router, and
literal first-max tie behavior:
`act/pipeline/moe/results/advmoe_architecture_audit_20260830.json`. This is an
architecture/training-semantics result only. Training, sampled input-space
route census, and deep-path certificates remain pending after RT-ER B3.

### Two dependency failure modes are distinguished

The official RT-ER reproduction is `pinned-but-rotted`: the released dependency
pin is concrete but cannot execute its first CUDA kernel on the registered
Blackwell device. AdvMoE is `unpinned`: its requirements do not specify Python,
PyTorch, torchvision, or CUDA, the README names a missing requirements file,
and the training entry point stops at undeclared packages. These labels describe
artifact reproducibility failure modes, not method accuracy or quality. No
dependency was installed and no AdvMoE training environment was created in the
current stage.

### AdvMoE init bracket remains fully undecided after stronger tools

The first frozen 20-input engineering pilot established only that weak PGD and
IBP could not close the route bracket. The second pilot keeps the same inputs
and five radii but uses 100-step, 10-restart margin-directed PGD and
resource-gated sparse backward CROWN. The literal router remains rejected;
the fixed-shape adapter is bit-exact and the accepted CROWN worker peaks at
20.98 GiB while B1 remains alive.

Strong PGD finds 0/20 flips at every radius. Median margin compression grows
from 0.735% at 0.5/255 to 11.324% at 8/255, with a maximum of 13.048% at
8/255. The clean margin median is `0.3087212`, and the attacked 8/255 margin
median remains `0.2731661`. Median sparse-CROWN lower bounds range from
`-3.7447e8` at 0.5/255 to `-3.7329e9` at 8/255. This reduces median bound
magnitude by 5.20x--5.37x relative to IBP, but no numerical lower bound is
positive and every one of the 100 sample-radius rows remains undecided.

The independent audit reports zero issues and replays archive, model, adapter,
hash, partition, all attack endpoints, BN deployment state, resource limits,
and CROWN/accounting identities:
`act/pipeline/moe/results/advmoe_router_bracket_init20_20260830_r7_strong_crown.json`.
The paper-safe conclusion is that a materially stronger two-sided engineering
bracket remains unresolved at initialization. Zero attack flips are not formal
stability, and large negative relaxation bounds are not evidence of intrinsic
router difficulty. Alpha-CROWN, beta-CROWN/BaB, the trained checkpoint, census,
and deep-path certificate coverage remain pending.

### The init diagnostics expose two router-architecture regimes

The L1 diagnostic is the gradient of the selected-vs-competing route margin
with respect to the unit-pixel input. Its per-input first-order boundary
estimate has median `67.850/255`; independently, extrapolating the achieved
8/255 PGD margin compression gives `70.644/255`. Across the 20 frozen inputs,
Pearson and Spearman agreement are `0.926` and `0.910`; 16/20 estimates agree
within 5% and 19/20 within 10%. Thus the agreement is pointwise rather than a
quotient-of-medians coincidence.

The K=20 RT-ER exact pixel-box aggregate median is `0.5324/255`, making the two
scale ratios `127.4x` and `132.7x`. The bounded paper wording is therefore
“approximately 130x larger empirical route-boundary scale at initialization.”
This contrasts a deep convolutional router with a standard affine router and
limits the affine `1/sqrt(d)` observation: it must not be extrapolated to deep
routers. The evidence does not isolate weight sharing, pooling, depth, or any
other architectural component as the causal mechanism, and the AdvMoE values
are not exact boundaries or certificates.

This regime also motivates a dependency inversion in the staged design. For
the official AdvMoE `E=2` shared-route model, sound verification can, in
principle, enumerate both static specialized pathways over the entire input
box without requiring a router-stability proof; router bounds become optional
pruning information. This is an implemented specialization identity but not
yet an official-checkpoint certificate result. The route-invariance baseline,
by contrast, still requires a positive router-margin certificate before it can
delegate one path to the same downstream backend.

The initial proposal to visualize a sound CROWN reach near `1e-12` does not
survive numerical audit. Although linear extrapolation gives median
`1.609e-12`, a five-input float32 CROWN bisection changes sign between median
requested epsilons `1.856e-9` and `1.868e-9`. All five transitions coincide
with expansion to the next representable float32 input box. These values are
frontend/relaxation diagnostics, not real-domain certified radii. Accordingly,
the paper must not draw an eleven-order sound certificate-gap figure from this
backend. A formally labelled reach requires outward-rounded or otherwise
validated numerical semantics.

## Threats to validity

### Construct validity

- The central 36/100 number is route-boundary certification yield on a
  deliberately constructed boundary cohort, not natural-input prevalence.
- Candidate exactness refers to an unrelaxed reachable-router HZ, not every HZ
  computation in ACT.
- The width result is structural binary width, not a measured monolithic runtime
  advantage.
- Guard support's paired result is coverage; the evidence does not justify an
  unconditional speedup claim.
- F0 uses a sound outer relaxation. Its unresolved cases measure verifier
  incompleteness, not model safety or unsafety.

### Internal validity

- The confirmatory endpoint is immutable at 56/100 even though 1D later reaches
  68/76 applicable coverage. The closure is not backfilled.
- The N1 and incremental-HiGHS measurements are explicitly engineering reruns;
  neither changes confirmatory outcomes.
- All weighted unsafe verdicts require full-model replay. Independent audits
  reported zero issues for the confirmatory, 1D, N1, and guarded-hull artifacts.
- Earlier failed or partial launch directories are preserved and excluded rather
  than overwritten.

### External validity

- The main evidence uses one seed of a 49.26%-accuracy, MLP-expert CIFAR-10 MoE.
  It cannot establish scaling to ResNet experts, token routing, intermediate
  MoE layers, larger expert counts, or other datasets.
- The official-construction K=20 result studies exact affine-router geometry
  only. It strengthens artifact applicability evidence but does not substitute
  for B1/B3 expert verification.
- Exact route-set enumeration is combinatorial in \(E\) and \(k\); the present
  `E=8,k=2` implementation is not an expert-count scalability result.
- P0a and P0b are complete. A true monolithic solver comparison and official
  RT-ER B3 expert verification remain pending before official-scale certificate
  claims.
- The AdvMoE init pilot still has a 100/100 sample-radius undecided band after
  100-step, 10-restart PGD and sparse backward CROWN. This is an init engineering
  result, not nonlinear-router applicability, stability, prevalence, or
  third-party deep-path certificate coverage. Alpha-CROWN and beta-CROWN/BaB
  remain unexecuted closure tiers.

### Statistical conclusion validity

- Candidate intervals use sample-cluster bootstrap rather than treating four
  radii from one sample as independent.
- The confirmatory unique-safe result reports a Wilson interval on the full
  predefined denominator. Solved-only denominators are not used.
- N1 has only two discordant improvements and p=0.5; it is an observed mechanism
  result, not evidence of statistically reliable superiority.
- The survey is partial and cannot support prevalence inference.

### Reproducibility and artifact validity

- Every paper number must resolve through the paths and commits in
  `paper/evidence_table.md`; raw artifacts remain separate from tracked summary
  manifests.
- The exact author-pin B1 failure is a hardware/software compatibility result,
  not evidence about the official method's accuracy or robustness.
- Tool adapters must preserve tie-inclusive semantics and must not promote
  relaxation candidates or numerical fallbacks to semantic verdicts.

## Paper-level conclusion supported today

The current evidence supports the following bounded conclusion:

> On an independent, verification-scale weighted top-2 MoE cohort,
> correlation-preserving route analysis reduces candidate sets, route
> conditioning separates structural binary width when multiple experts are
> feasible, and a staged gate-elimination plus gate-range McCormick verifier
> certifies route-changing regions that a route-invariance precondition rejects.
> Guard-aware support improves paired branch coverage. These results do not yet
> establish official-scale expert verification, a monolithic runtime advantage,
> or ecosystem prevalence. Separately, exact official-construction router
> geometry shows that route-invariance applicability is nearly empty at
> \(8/255\) and varies 10.384x across initializations at \(2/255\); this is an
> applicability finding, not an output certificate.  Across the two official
> construction families, the observed normalized-coordinate local-radius shift
> (8.966x) agrees with the dimension-and-input-second-moment prediction (9.063x),
> while the finite-precision preprocessing audit shows that exact runtime and
> dtype/order must be bound into the identity of the certified artifact.
