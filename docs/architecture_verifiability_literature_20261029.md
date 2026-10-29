# Architecture verifiability: nearest work and S1 novelty boundary

Read on 2026-10-29, before new dense training or S1 verification results.
This is a scoped primary-source review, not a claim of exhaustive absence.
No equivalent **combined** study was identified: deterministic weighted top-2,
total/active capacity controls, full-request certification cost, and routing-aware
certified training. Several individual claims already have close precedents.

## Five closest works

| Work | Object, training and experiment | Guarantee / comparison | Consequence for this project |
| --- | --- | --- | --- |
| [On the Adversarial Robustness of Mixture of Experts, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3effb91593c4fb42b1da1528328eff49-Abstract-Conference.html) | Sparse vision MoE; JFT-300M pretraining and ImageNet experiments; dense ViT and MoE compute/size comparisons; PGD/AutoPGD, routing and expert-count ablations. | Conditional Lipschitz theory plus empirical adversarial accuracy. Not a table of deterministic complete-verification acceptance/time. | Most direct precedent for the capacity/compute question. Do not claim first evidence that sparse MoE can improve robustness at comparable compute. S1 must add certified coverage and full verification cost, not recycle this framing as novelty. |
| [MetaMoE, SAIV 2026](https://link.springer.com/chapter/10.1007/978-3-032-32357-6_8) | Heterogeneous, disjoint class-space domain experts, hard top-1; eight configurations and three perturbation budgets, robust versus ordinary experts. | Router stability plus selected-expert guarantee gives compositional verification; empirical and formal outcomes distinguished. | “MoE is a verifiable architecture” is already a research topic. Our candidate distinction is shared-output weighted top-2, changing routes and capacity-matched certified training. Publisher abstract checked here; full experimental detail also requires the project's archived author materials. |
| [Dual Randomized Smoothing: Beyond Global Noise Variance, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cd9664c7094d90e512ce27f2fd58198b-Abstract-Conference.html) | CIFAR-10/ImageNet; smoothed variance selector routes to expert smoothing scales; iterative selector/classifier training; global-variance and input-dependent alternatives. | Statistical L2 certification of the smoothed function, with selector local constancy. [Author code](https://github.com/eth-sri/Dual-Randomized-Smoothing) includes training/certification/reproduction. | Already couples routing, certification and the accuracy/robustness tradeoff. Its probabilities, norm and function differ from deterministic Linf top-2. Discuss side by side, not a fabricated shared SAFE table. |
| [RT-ER / JTDMoE, ICML 2025](https://proceedings.mlr.press/v267/zhang25cj.html) | [Paper §6](https://arxiv.org/html/2502.06832v2): CIFAR-10 ResNet18 and TinyImageNet ViT-small, four-expert top-1; expert-focused adversarial training and standard/robust dual models; PGD/AutoAttack comparisons. | Theoretical robustness bounds; reported experimental robust accuracies are attack evaluations. | “Train the experts because they dominate failures” is not new alone. Certified interval losses, routing-margin ablation and certified cost are the proposed additional evidence; no reuse of attack percentages as certification. |
| [CertMoE, arXiv 2610.29395, 21 Oct 2026](https://arxiv.org/html/2610.29395v1) | SST-2/AGNews, three pretrained MoE LLMs; masked text, certified expert pools and induced-subnetwork certification; RanMASK/SelfDenoise, pool/mask ablations and runtime. | Author-proposed probabilistic routing containment and output guarantees for smoothed text functions under token perturbations; no matching dense certified-training experiment identified. | Direct recent overlap with route-aware reduction. Do not claim the first certification allowing routing changes. Its sentence-aggregated smoothed routing, subnetwork and original-function correspondence require separate scrutiny; author claims are not independently proved here. |

The last work's general assertion that dynamic routing invalidates ordinary
randomized smoothing is not adopted: a deterministic routed network is still a
function. Any invalid certificate must be traced to the actual smoothed object,
sampling or subnetwork substitution, not discontinuity alone. Likewise this
review does not validate its aggregated-routing containment as token-wise coverage.

## Dense training references and experimental implications

[IBP](https://arxiv.org/abs/1810.12715) and
[CROWN-IBP](https://arxiv.org/abs/1906.06316) provide the named dense controls.
CROWN-IBP uses forward interval intermediate bounds with backward linear bounds;
its training loss/schedule is distinct from evaluating an ordinarily trained
network with that bound procedure. [Author code/configurations](https://github.com/huanzhang12/CROWN-IBP)
cover MNIST and CIFAR settings. S1 trains ordinary dense models, then evaluates
both architectures; it must not be labelled a certified-training result.

[SABR](https://github.com/eth-sri/SABR), ICLR 2023, trains on selected small
adversarial boxes and uses a separate certifier for full-domain evaluation.
Its training approximation is not permission to ignore unstable MoE routes at
evaluation. [ACERT](https://github.com/boschresearch/adaptive_robust_training)
is another accuracy/robustness Pareto precedent using sample-adaptive training
radii. [CTBench](https://github.com/eth-sri/CTBench) is a useful author-maintained
baseline implementation inventory; not installed or executed by this review.

S2 must isolate ordinary versus certified training, the routing regularizer, and
capacity matching. The clean-route training surrogate is explicitly approximate;
certification still covers all tie-legal histories. A 90% single-route target
must use proved stability, not PGD's failure to find a flip. Router parameters,
prefix work, failed bounds and repeated histories all enter the cost accounting.
Active parameter count is a conditional graph statistic: the old training code
evaluates all experts before gather, so it is not measured training FLOPs.

No literature novelty stop is triggered by the checked papers. This means S1 is
worth conducting, not that C1–C3 or a new training method have been established.
The broad title “MoE is easier to certify” must remain a question until S1/S2.

## Official venue check

[ICML's official future-meetings page](https://icml.cc/Conferences/FutureMeetings)
lists 2027 in South America. The official homepage still presents ICML 2026;
no ICML 2027 abstract/full-paper deadline or page limit was verified in this
check. January dates from third-party trackers are estimates, and the similarly
named International Conference on Minority Languages is unrelated.
Eight pages is a working manuscript target, not a verified 2027 requirement.
The user's mid-January 2027 completeness checkpoint and possible NeurIPS 2027
pivot are internal decisions, not confirmed conference deadlines. PI handles
submission; no publication or author contact occurred.
