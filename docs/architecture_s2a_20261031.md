# S2 direct test: route-envelope certified training

## User decision and current evidence

2026-10-31, **第八次修订补充：S2 直接检验**. Starting source
`bb2f23e49c2d21f22f4c623fb529ad53654a446b`, clean,
`feat/moe-route-verification`. The user explicitly authorizes this finite test
of C3 after S1 stopped. S1 remains `STOP_S1_RETURN_TO_REDUCTION_PAPER`;
this is a new PI decision, not a retrospective change of that gate.

Decision rationale: the S1 gate screened a certified-training hypothesis using
ordinary-trained models and complete verification. Its failure does not settle
C3. S1's incomplete result, 23/3000 versus 2/3000 (input-block descriptive
interval [0.40,1.07] pp), and post-hoc clean-correct counts, 114/288 versus
115/325 and 116/328, motivate a direct test; they do **not** pass any gate.
Those conditional populations differ, and neither the balance term nor width
has been isolated as the cause. Narrow paths do not guarantee better trained
certification. No power claim follows from three observed seeds alone.

The [2026-10-31 Claude review](../../survey/drafts/act_icml_review_20261031.md)
is internal, non-independent AI feedback, not human technical review. The
reviewer helped design earlier directions. PI-managed review by a human
certified-training expert remains required on the eventual new manuscript.
Old verifier mechanisms remain stopped; no new M-B/M-C/B4 work is authorized.

## Bounded literature check (2026-10-31)

Scope: IBP, CROWN-IBP, SABR, TAPS, MTL-IBP crossed with mixture-of-experts,
gating, conditional computation and top-k routing. Searches used these exact
phrases and variants on arXiv, OpenReview, proceedings and author repositories;
primary pages below were reopened. This bounded search did not locate the
same deterministic all-tie-route envelope training and capacity-matched
experiment. This is **not** an exhaustive novelty proof or a priority claim.
Discovery of equivalent work would stop the stage for user review.

| Closest MoE work | Established overlap | Difference from the proposed test |
| --- | --- | --- |
| [On the Adversarial Robustness of MoE, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3effb91593c4fb42b1da1528328eff49-Abstract-Conference.html) | Sparse/dense robustness and computational-capacity comparison; conditional Lipschitz theory | Attacked ImageNet models, not this all-route IBP certified-training comparison. Architecture robustness is already an explicit research question. |
| [RT-ER/JTDMoE, ICML 2025](https://proceedings.mlr.press/v267/zhang25cj.html) | Expert-focused training, accuracy/robustness trade-offs, theoretical robustness bounds | Empirical attack evaluation and a dual model; not the same route-envelope IBP objective. Do not describe it as lacking any theoretical guarantee. |
| [MetaMoE, SAIV 2026](https://link.springer.com/chapter/10.1007/978-3-032-32357-6_8) | Compositional router/expert verification | Heterogeneous hard top-1 and route-preserving guarantee, rather than homogeneous weighted top-2 envelope training. |
| [Dual RS, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cd9664c7094d90e512ce27f2fd58198b-Abstract-Conference.html) | Trained selector and certified routing to experts | Smoothed classifiers, statistical guarantees and L2 radii; not the same deterministic Linf function or bound. |
| [CertMoE, October 2026](https://arxiv.org/html/2610.29395v1) | Routing containment and induced-subnetwork certification | Probabilistic smoothing for language MoE; not this vision IBP training objective. Its abstract's priority language is not adopted. |

| Certified-training reference | What is reused / what must not be claimed |
| --- | --- |
| [IBP / Scalable Verified Training](https://arxiv.org/abs/1810.12715) | Interval propagation, final property projection, warmup/radius scheduling and training for tighter bounds are established. |
| [CROWN-IBP](https://arxiv.org/abs/1906.06316) | IBP forward bounds plus a CROWN backward bound; distinct from the pure IBP training objective registered here. |
| [SABR, ICLR 2023](https://arxiv.org/abs/2210.04871) | Small attack-selected training boxes; their training loss is not a bound over the entire target region. Verification is separate. |
| [TAPS, NeurIPS 2023](https://arxiv.org/abs/2305.04574) | IBP/PGD surrogate training and independent full-domain verification; not an all-domain sound training loss. |
| [MTL-IBP / Expressive Losses, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/0b7dd2703eca7f58bae0bf455811b27e-Abstract-Conference.html) | Convex blends of attack and bound losses; approximate loss tightness alone does not predict trained performance. |

[Understanding Certified Training with IBP, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3a2e5889b4bbef997ddb13b55d5acf77-Abstract-Conference.html)
studies width/depth, initialization and training-induced tightness. This limits
the simple explanation “narrower path implies better certification”: it is a
hypothesis to measure, not an established explanation of S1. The
[CTBench author repository](https://github.com/eth-sri/CTBench) and
[MTL-IBP author repository](https://github.com/alessandrodepalma/expressive-losses)
identify reproducible dense training references. No package was installed and
no author experiment was rerun during this literature check.

## Mathematical contract

This stage supports a ReLU MLP router and E ReLU MLP experts at a single output
layer, selected-softmax top-2, common input box X, shared class space. Ordinary
IBP produces router bounds `[l,u]` and bounds on expert values. Retain pair
`S={a,b}` iff `u_a >= max_{c outside S} l_c` and
`u_b >= max_{c outside S} l_c`. Equality is retained. This condition is exact
for compatibility with the **independent score box**, only necessary for an
actual input to realize the pair. Thus every tie-legal route is covered.

For canonical `a<b`, `lambda` is a's weight and lies in
`[sigmoid(l_a-u_b), sigmoid(u_a-l_b)]`. For property `(q,c)` let
`[L_e,U_e]` enclose `q^T E_e+c`, obtained by projecting q into the final
affine layer before interval evaluation. A valid pair bound is

```
L_S = min_{t in {lambda_lo,lambda_hi}} t*L_a + (1-t)*L_b
U_S = max_{t in {lambda_lo,lambda_hi}} t*U_a + (1-t)*U_b.
L = min_{retained S} L_S; U = max_{retained S} U_S.
```

Proof: every real execution has a retained pair and gate in its interval.
Nonnegative convex weights preserve the expert bounds; extrema of the affine
endpoint expression occur at its endpoints. Taking the pair hull covers the
union. Projection includes c exactly once because the two weights sum to one.
There is no requirement that all inputs use the same pair. There is no claim
that independent expert intervals preserve their joint dependence.

The explicit E=1 control uses top-1 with weight one and equals dense IBP; top-2
on one expert is rejected. All dimensions/classes remain present. Unsupported
operations (including BN) and malformed intervals are rejected, not silently
converted. Training and evaluation use `model_bounds` in
[route_envelope_ibp.py](../act/pipeline/moe/route_envelope_ibp.py).

The real-arithmetic theorem is distinct from float32 PyTorch evaluation, which
is a numerical policy without outward rounding. Unit containment tests are not
complete floating-point proofs. The historical source gaps remain unclosed.
Primary positives require clean correctness and **all** lower margins `>0`.
No positive tolerance or skipped pair can turn an unresolved request positive.

Regularizer: batch mean of the sum over unselected experts of
`ReLU(0 - (min(l_a,l_b)-u_c))`, where S is the clean top-2 pair. Its zero value
at equality is **not** a certificate of route stability. The reported
provable-single-route diagnostic is exactly one pair in the complete candidate
cover (a sufficient numerical-policy condition); all-tie cases remain unstable.

## S2a training freeze (before any trained outcome)

Four arms: D-act `[3072,602,301,10]`, D-tot `[3072,1758,879,10]`, M and M-reg
with the old L1E8 architecture (router `[128]`, experts `[256,128]`, ten classes).
CIFAR-10 raw ToTensor, no augmentation or normalization, the old seeded
45,000/5,000 train/validation split (seed 0), five training seeds 0–4.
Training does not load test data. Test evaluation starts after configuration
and checkpoint selection are complete.

Each arm gets two seed-0 configurations: D-act/D-tot/M-reg lr `.001`/`.0005`;
M lr `.001`, lambda_m `.1`/`1`. M-reg has lambda_m 0. AdamW, weight decay
`.0001`, batch 256, 100 epochs, float32 CPU, one thread per worker. No LR
scheduler; same natural/radius/kappa schedule in all arms: five natural epochs,
40 epochs of linear radius increase from 0 to 2/255 and kappa decrease from
1 to .5, then 55 epochs at those final values. Schedule is evaluated per batch.
This is a fixed common adaptation, not a claim to reproduce a paper's full
CNN/BN recipe. Natural warmup omits the robust and routing penalties; MoE's
original `.1 * switch balance` term remains. Afterwards the objective is
`kappa*CE + (1-kappa)*IBP_CE + lambda_m*route_penalty + .1*balance`.
Dense arms omit the two routing terms. IBP_CE is logsumexp of zero and negated
true-minus-other lower margins. No clean-route-only bound is substituted.

Checkpoint selection: highest validation IBP certified count at **final**
epsilon 2/255, then highest validation clean count, then earliest epoch.
Configuration selection uses the same two validation counts, then first listed
configuration. The clean tie-break is frozen here, not selected after outcomes.
The selected seed-0 training run/checkpoint is reused: eight tuning runs plus
four further seeds per arm = **24 runs**, not 28. Every run lasts 100 epochs;
failed attempts and resume costs stay recorded. No test-based early stopping.

## S2a evaluation and statistical gate

Primary: all 10,000 test inputs, IBP at 2/255, clean errors unaccepted. Secondary:
first 1,000 CROWN-IBP; first 200 complete verification; clean accuracy, provable
single-route fraction, mean property width and full time. These public-prefix
inputs have historical exposure and are development, not a new holdout.
Twenty selected checkpoints yield at most **4,000 complete calls**, each 300s,
at most eight single-thread CPU workers, each 8GiB RLIMIT_AS. Incomplete
evaluation and training costs are separate. The user specified a 40h complete
verification limit; wall-clock versus summed interpretation is pending their
answer before the evaluation queue is frozen. This does not block training.
The same engine revision and common branching heuristic settings must be bound
for both families before complete verification; intrinsic graph operators
still differ. The S1 dense-only kfsb setting is not automatically inherited.

Bootstrap: 10,000 replicates, fixed RNG seed 20261031; resample five seed IDs
with replacement, and independently resample test input IDs with replacement,
using the **same** seed/input draws across all arms. The crossed resampling
retains common-input and common-seed pairing. Report mean over the 5×10,000
grid, percentile 95% intervals and per-seed values. Do not treat 50,000 as
independent images. Clean-conditional figures are descriptive secondary only.

All required: M−D-act >=2 pp and its 95% lower bound >0; M−D-tot point >=0;
M's mean provable-single-route fraction at 2/255 >=90%. Failure means stop,
write results for TMLR/workshop, no new direction/configuration/sample expansion.
M-reg is a prespecified mechanism ablation. Passing is not a guarantee of ICML
acceptance or source-complete proof.

## Conditional S2b and manuscript

Only after all S2a gates pass: two ReLU conv stem layers (32/64 channels,
stride2, no BN), output E8 top2 head (router128, experts256/128), active/total
matched dense heads, five seeds; train radii 2/255 and 8/255, kappa_end 0/.5,
E4/8/16, top1/top2 and lambda ablations. Freeze that allocation before outcomes;
complete calls <=3,000. GPU only when idle after resource checks, otherwise CPU.
The proposed study is not silently substituted with a smaller easier object.

After S2a, restructure the manuscript around semantic reductions, route-envelope
IBP and regularization; S1 motivates the question and old negatives stay in the
appendix. W2: separate unconditional and conditional populations. W3: measured
width/router diagnostics, not assumed cause. W5: common engine/settings and
complete verification secondary. W7: scientific tables/figures, not execution
logs in the main text. Human review and clean empirical reproduction remain
PI/external tasks. ICML2027 deadline was not confirmed; no guessed deadline.

## Resources and delivery

Existing act-py312 only, no installation; CPU S2a. Per-process 8GiB address-space
limit, RSS telemetry is not group memory; math/identity/exit/cleanup checks stay.
S1's 512MiB amendment applies to S1 only. S2's compact evidence has the inherited
128MiB stage cap; training checkpoints/data/environment storage is separately
listed, not disguised as evidence or deleted. Model training and evaluation
use an immutable committed execution copy. Only this branch is changed.
Tool errors are preserved then repaired/resumed within the approved roster;
no gate or failed scientific result is rewritten. Each completed stage has
tests/audit, documentation, commit/push and synchronization report.

## Implementation and pretraining check

[Machine-readable training freeze](architecture_s2a_training_freeze_20261031.json)
fixes the 24-run catalogue and selected-seed reuse. Sixteen ordinary tests pass
in 3.326s including import/setup (0.801s test body), with full untrained dense
and E8 MLP shapes, under the 8GiB address-space limit. Peak process ru_maxrss
was 849,452KiB; this is not a simultaneous process-group memory measurement.
[Stage record](architecture_s2a_records_20261031.json) binds logs, review and
freeze. No CIFAR example, trained checkpoint, real training or verifier was
queried by these controls. They validate integration, not trained usefulness.
The runner preserves immutable epoch/attempt records and resumable optimizer,
shuffle RNG, best validation state and schedule position; seed0 selection
explicitly ignores test fields. Configuration choice is still pending training.
