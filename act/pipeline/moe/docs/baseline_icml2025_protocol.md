# ICML 2025 RT-ER Baseline Protocol

## Stage and scope

This document freezes Phase B0 for the official repository accompanying
*Optimizing Robustness and Accuracy in Mixture of Experts: A Dual-Model
Approach* (ICML 2025). B0 is a provenance, semantics, dependency, and protocol
audit only. It does not install packages, launch a smoke run, train a model, or
modify the external clone.

The first target is the single hard-top-1 RT-ER MoE invoked by

```text
python cifar10_RT_ER.py --net res18_moe
```

JTDMoE, Robust-MoE-CNN, V-MoE, F1, and new ACT model training remain out of
scope until this protocol's explicit gates are met.

## Official source identity

| Field | Frozen value |
|---|---|
| repository | `https://github.com/TIML-Group/Robust-MoE-Dual-Model` |
| branch | `main` |
| commit | `30ef94d77b5451595b82e739aa8938e1f4c4521f` |
| remote HEAD at audit | same as frozen commit |
| commit date | `2025-08-17T10:57:35-05:00` |
| license | Apache-2.0 |
| local read-only clone | `/data1/Kane/MOE/baselines/Robust-MoE-Dual-Model` |
| clone status | clean |
| released checkpoint | none found in repository, releases, or README |
| training/attack/model source | public in the pinned repository |
| trained checkpoint | not published |
| verifier implementation | not published; none found in tracked Python/Markdown/text files |

Per-file SHA-256 values are recorded in
`act/pipeline/moe/configs/baseline_icml2025_provenance.json`. The external clone
must remain clean. All run products will be rooted under
`/data1/Kane/MOE/baseline_runs/icml2025_rt_er`; ACT-side wrappers and patch files
will remain under the feature branch.

## Author model semantics

The CIFAR-10 model contains four independent ResNet18 experts and one router

```text
Flatten -> Linear(3072,4) -> argmax.
```

The router receives the same FFCV-normalized tensor as the experts. The code
uses CIFAR channel means `[125.307,122.961,113.8575]` and standard deviations
`[51.5865,50.847,51.255]` in the uint8 scale. The concrete hard route uses
PyTorch `argmax`, which chooses the first maximum. ACT verification will use its
more conservative `ANY_LEGAL_TOPK` semantics at exact ties. Concrete conversion
tests must therefore record tie proximity and distinguish a deterministic
concrete choice from all verification-legal choices.

The model executes only the selected expert and emits that expert's ten logits.
It is output-level hard top-1, not selected-softmax top-2 and not an intermediate
token-level MoE. F0 is inapplicable. This cleanly tests route conditioning
without weighted-gate sufficiency.

## Paper-config target

The ICML paper reports CIFAR-10 RT-ER with four ResNet18 experts, 130 epochs,
`L_inf` training epsilon 8/255, 10-step PGD training, and 50-step PGD evaluation.
For the top-1 model it reports:

| Metric | Paper value |
|---|---:|
| standard accuracy | 77.81% |
| PGD-50 whole-model robust accuracy | 69.09% |
| PGD-50 expert-targeted robust accuracy | 75.71% |
| PGD-50 router-targeted robust accuracy | 72.28% |
| AutoAttack whole-model robust accuracy | 54.36% |

The AutoAttack table lists standard accuracy 75.92%, not 77.81%. This difference
must be treated as a paper-level checkpoint/evaluation ambiguity until it is
explained; it is not silently averaged or replaced.

The author script fixes beta 6 by default, batch size 512, Adam learning rate
`1e-4`, cyclic base learning rate `5e-5` with `step_size_up=500`, 10-step PGD
training, 50-step PGD testing, epsilon 8/255, and attack step size 2/255. Training
augmentation is horizontal flip, two-pixel translation, and cutout size eight.

## Code/paper discrepancy ledger

These items are frozen before execution:

| ID | Observation | Protocol consequence |
|---|---|---|
| D1 | paper says 130 epochs; script default is 200 | the primary run passes `--n_epochs 130` and is named **official-code, paper-config reproduction** |
| D2 | the script never sets Python, NumPy, PyTorch, CUDA, or FFCV seeds | any injected seed is a disclosed reproducibility patch; the paper does not define a canonical seed |
| D3 | `usewandb = ~args.nowandb` is truthy for both Boolean values | an ACT-side compatibility patch is required to disable W&B; `--nowandb` alone is ineffective |
| D4 | the script tests/saves every ten epochs and overwrites one file; `best_acc` is never used for selection | the primary checkpoint is the last tested paper-config epoch, not a best-test checkpoint |
| D5 | the test FFCV loader uses random order | sample order is nondeterministic unless the disclosed seed/loader patch controls it; aggregate full-test metrics are still defined |
| D6 | the script unconditionally calls `net.cuda()` | B1 requires CUDA; a CPU fallback must not be claimed from the official script |
| D7 | repository requirements pin torch 2.4.0/torchvision 0.19.0 | `act-py312` currently has torch 2.9.1+cu128/torchvision 0.24.1+cu128; this is a compatibility deviation |
| D8 | the repository contains no certified-bound implementation | any theorem implementation is an **author-paper formula reimplementation**, not an official verifier |
| D9 | the paper's analytic bound assumes Lipschitz router weights, while the released model executes hard argmax | the formula is reported only when its applicability is formally established; route-changing hard-dispatch regions are not silently assigned a continuous-gate certificate |
| D10 | released experts return raw logits while Theorem 5.4 assumes an expert-output bound `M_Ri <= 1` | the formula reimplementation must define probability-versus-logit semantics and derive matching bounds |
| D11 | the paper reports no numerical Theorem 5.4/5.5 instance and specifies no Lipschitz-constant computation | there is no author certificate target to reproduce; sound, empirical, and unspecified constants receive different labels |

No undocumented repair is allowed. Each patch is classified as:

- `compatibility`: permits the pinned algorithm to run without changing its
  mathematical training objective;
- `reproducibility`: fixes otherwise unspecified stochastic state;
- `scientific`: changes data, model, objective, optimizer, scheduler, attack, or
  selection semantics.

Scientific patches cannot be labeled an official-code reproduction.

## Environment gate

Dependency installation was subsequently authorized in an isolated environment
under `/data1/Kane/MOE`; `act-py312` remains unchanged. The exact author pin is
installed at `/data1/Kane/MOE/envs/rt-er-repro`:

| Package | Author pin | Isolated author-pin environment |
|---|---:|---:|
| torch | 2.4.0 | 2.4.0+cu121 |
| torchvision | 0.19.0 | 0.19.0+cu121 |
| FFCV | 1.0.2 | 1.0.2 |
| timm | 1.0.15 | 1.0.15 |
| einops | 0.8.1 | 0.8.1 |
| wandb | not pinned | missing |

All imports pass. However, the installed GPU is Blackwell `sm_120`, whereas the
author-pinned PyTorch binary contains kernels only through `sm_90`; the first
CUDA tensor kernel fails with `no kernel image is available for execution on
the device`. Since the author script unconditionally calls `net.cuda()`, B1 is
now **blocked by an exact-pin hardware compatibility failure**, not by missing
dependencies. No dataset conversion, smoke training, or full training was
started. Replacing FFCV with a Torchvision loader remains a scientific data-
pipeline modification. A newer PyTorch run would be a separately labeled
Blackwell-compatible official-code reproduction, never a silent substitution.

## Dataset identity and write containment

The existing official Torchvision CIFAR-10 archive is
`/data1/Kane/MOE/ACT/data/torchvision/CIFAR10/raw/cifar-10-python.tar.gz`, SHA-256
`6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`.
All extracted-batch hashes are in the machine-readable provenance file.

The author script writes FFCV `.beton` files and checkpoints relative to its
working directory. Future execution must use a dedicated directory such as

```text
/data1/Kane/MOE/baseline_runs/icml2025_rt_er/seed0_paper130/
```

with the author repository supplied on `PYTHONPATH`. It must not run with the
external clone as its working directory. This keeps author source clean and
contains datasets, FFCV cache, checkpoints, W&B state, logs, and results under
the authorized root.

## B1 official-code reproduction

### Smoke gate

After dependency authorization and a committed compatibility wrapper, the
smoke must exercise:

1. official CIFAR decoding and FFCV conversion in the isolated run directory;
2. one clean forward and backward pass;
3. one adversarial training batch using the author RT-ER loss;
4. one bounded smoke epoch or a preregistered fixed number of batches;
5. checkpoint save/load and exact restored logits on a frozen batch;
6. router logits, selected routes, per-expert route counts, and device/dtype;
7. clean and adversarial evaluation on a fixed small batch;
8. unchanged external-clone status.

All four experts' usage is reported. Lack of usage is an important reproduction
result, not automatically a reason to modify the training loss. It blocks using
the checkpoint as a four-expert scalability claim unless corrected by an
author-supported configuration.

### Full seed-0 run

The primary run is seed 0, 130 epochs, beta 6, and otherwise the paper/code
settings above. Because the paper did not publish seeds or a checkpoint, the
result is compared against the reported metrics as a reproduction interval, not
bitwise equivalence. A provisional two-percentage-point target may be reported,
but it is not promoted to a hard scientific exclusion until the code/paper
ambiguities D1--D5 are resolved.

Three seeds `0, 1, 2` are required for scientific route-geometry and accuracy
results. Seed 0 passes the smoke, conformance, and artifact audit before seeds
1--2 are queued; this ordering is a failure-containment rule, not permission to
publish a single-seed result.

### Training-time route telemetry

The author script overwrites one checkpoint every ten epochs. The wrapper must
first preserve immutable epoch-qualified copies and then run the exact affine
batch oracle on all 10,000 ordered CIFAR-10 test images. Median/IQR route radius,
the fraction with a route boundary inside `8/255`, boundary-competitor counts,
and route-load entropy are reported from epochs 10 through 130 for every seed.
The frozen schema, checksum rules, label, and failure semantics are in
`act/pipeline/moe/docs/icml2025_route_telemetry.md` and
`act/pipeline/moe/configs/icml2025_route_telemetry.json`.

The Blackwell seed-0 execution is implemented by three ACT-side files without
editing the official clone: a deterministic launcher with a disclosed local
W&B no-op shim, a real-CIFAR objective/checkpoint smoke, and a process-group
supervisor. The supervisor waits for both the epoch checkpoint and its flushed
training/validation metrics, pauses the author process group, copies and hashes
the checkpoint, runs telemetry synchronously, and resumes only after telemetry
succeeds. The seed-0-only config is
`act/pipeline/moe/configs/icml2025_route_telemetry_blackwell_seed0.json` and is
labeled `official-code, Blackwell-compatible deps + FFCV`. Missing telemetry is
a run failure; it is never filled after training or silently omitted.

## B2 checkpoint-to-Route-A/CROWN conformance

Conversion starts only after B1 lands and passes its independent endpoint
audit. The executable B2 stage freezes the first 1,000 ordered test inputs and
their independently audited PGD-50 endpoints. It compares:

- normalized inputs and pixel-space perturbation conversion;
- router logits and deterministic concrete route;
- all-legal tie routes when margins are within the frozen tolerance;
- every expert's logits and the selected full-model logits;
- top-1 prediction;
- convolution, residual addition, BatchNorm running statistics/eval semantics,
  and normalization after concrete auto_LiRPA graph conversion;
- clean and adversarial inputs.

Required outcomes are 100% prediction agreement, 100% route agreement outside
explicit tie cases, and a preregistered maximum logit error. A route disagreement
blocks verification.

The frozen config, executable reference/worker/auditor, and exact scope are in
`configs/icml2025_b2_seed0_r3.json`, `icml2025_b2_conformance.py`,
`icml2025_b2_crown.py`, `audit_icml2025_b2_conformance.py`, and
`docs/icml2025_b2_conformance.md`. B2 checks only the conversion boundary that
B3 actually consumes; it does not claim a generic whole-program ACT conversion.
The B1 literal-fp16/autocast program and B3 real-float32 program are distinct
certificate identities. Their measured label drift is reported explicitly, and
B3 claims are scoped only to the latter; the frozen 20-sample B3 cohort must
still agree across both identities before execution.

## B3 verification comparison

The first frozen comparison uses 20 deterministic clean-correct samples from the
same converted checkpoint:

1. explicit route-invariance baseline;
2. exact route analysis plus route-conditioned expert verification;
3. true monolithic router/dispatch/all-expert formulation where executable;
4. author-paper analytic formula reimplementation with an applicability status;
5. a scalable expert-backend adapter, separately scoped from route handling.

Because the official router is affine, minimum hard-route-change radius should be
computed by exact box-constrained linear feasibility (or an equivalent analytic
distance when clipping is inactive), not attack-only bisection. Verification
ties remain inclusive.

ACT's affine oracle is implemented in
`act.back_end.moe.route_boundary.affine_top1_route_boundary`; its batch API
provides sub-second quantized-CIFAR telemetry on the current GPU. It folds input
normalization explicitly and handles pixel clipping by exact piecewise-linear
support inversion. It cannot produce an RT-ER distribution until a trained
checkpoint supplies the router parameters.

Route conditioning is frozen as a backend-independent interface. For official-
scale ResNet18 experts, the intended scalable comparison is CROWN over a sound
guarded-cell box hull, with HZ retained as the exactness reference. Direct
general linear input constraints are not supported by the audited α,β-CROWN
VNNLIB/expression front ends at commit `e5c7e17...`. An augmented-router-output
Clip-and-Verify adapter may preserve the guard but remains unvalidated until an
authorized isolated environment passes a conformance smoke. The complete
contract and provenance are recorded in
`act/pipeline/moe/docs/expert_backend_interface.md` and
`act/pipeline/moe/configs/alpha_beta_crown_provenance.json`.

The original ResNet18 is never silently reduced. If all 20 original-scale exact
expert runs time out, the result is retained as `Official-scale reproduction`.
A separately named `Verification-scale derivative` may then preserve the hard
top-1 linear router and training semantics while shrinking experts. Only the
derivative is used for exact monolithic ground truth at scale.

## Analytic-certificate rule

The official repository contains training and attack code but no implementation
of Theorem 5.4's certified radius. Any implementation must:

- cite the paper equation and state every Lipschitz and output-bound assumption;
- define how expert and router constants are obtained;
- distinguish continuous router weights from the released hard-argmax dispatch;
- return `NOT_APPLICABLE`, not zero or `UNKNOWN`, when an assumption is false;
- pass toy-model regression tests with independently computed bounds;
- never be described as the authors' official verifier.

For a hard top-1 model, route-changing regions are the crucial applicability
test. Treating raw router logits as mixture weights would change the released
model and is forbidden.

The full theorem/code assumption matrix and permitted result labels are frozen in
`act/pipeline/moe/docs/icml2025_certificate_applicability.md`.

The paper-only audit answers two previously open questions: there is no
numerical certificate experiment to reproduce, and no method is given for
computing `L_Ri` or `r_Ri`. The pre-registered five-leaf decision tree therefore
includes `NOT_FORMALLY_INSTANTIATED` and `VACUOUS_AT_REGISTERED_RADII` before
testing route applicability. Attack-derived margins or sampled gradients never
become formal bounds by naming them Lipschitz estimates.

## Stage gates

B0 is complete when this document and its provenance JSON validate and are
pushed. It unlocks only a dependency decision, not B1 execution.

B1 requires explicit dependency authorization, a clean external clone, a
committed wrapper/patch ledger, contained output paths, and a passing smoke. The
exact author pin currently fails the CUDA compatibility probe on `sm_120`, so
this gate remains closed pending an explicitly labeled compatibility variant.
B2 requires an audited seed-0 checkpoint. B3 requires B2 conformance.

No public baseline, F1, confirmatory extension, or new training was launched by
B0.
