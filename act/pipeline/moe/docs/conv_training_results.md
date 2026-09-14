# Convolutional family R1: compatibility and frozen training completed

Execution source: `6d2e6d299` (archived into the run before launch).
Raw root: `data/moe/results/conv_training_seed17_20260915_r1`.
Compact independently landed/reviewed record:
`../results/conv_training_review_20260915_r1.json`.
Pretraining controls: `../results/conv_pretraining_review_20260915_r1.json`.

## Frozen subject and result

The registered four-expert convolutional output-layer selected-softmax top-2
model has155,052 parameters. Seed17,100 epochs, AdamW1e-3, weight decay5e-4,
cosine decay, cross-entropy plus0.01 Switch balance, crop/flip augmentation and
the45k/5k split were unchanged. No outcome-dependent architecture, seed, radius,
training extension or checkpoint substitution was made.

| Item | Result |
|---|---:|
| Completed / expected epochs | 100 / 100 |
| Retained immutable checkpoints | 100 |
| Selected epoch (earliest validation maximum) | 89 |
| Validation correct / total | 3,404 / 5,000 (68.08%) |
| Test correct / total | 6,706 / 10,000 (67.06%) |
| Test effective experts (selection-share entropy) | 3.3962 |
| Independent training/selection audit | PASS, 0 issues |

Selected checkpoint: `checkpoints/epoch_089.pt`, SHA-256
`f5781a792f844a68de941f1a6b314d0e627ad5dd30e262088e6c1864d6bc5289`.
The model is versioned `act-output-conv-moe-v1`; raw checkpoint files stay local,
not in Git. Test was not used for selection. The final100th epoch remains
available but is not substituted for the registered validation-selected model.

Test expert-selection counts are6,691/8,639/2,934/1,736, normalized over20,000
top-2 selections:33.455%/43.195%/14.670%/8.680%. These are selection shares, not
per-image class accuracy or perturbation route-flip rates. All four experts are
selected on some test inputs; this does not prove routing changes inside any
particular perturbation box. No new robustness query was run on the checkpoint.

## Compatibility, supervision and evidence

The initial full-shape ACT control retained SparseHZ for router and all four
experts. CROWN initially failed because the entrypoint omitted a float64 default
for internally created convolution identities. R1 remains archived. R2 changes
that initialization only, retaining identical checkpoint and represented-input
hashes. Both backends then pass the frozen conformance tests. No dependency,
solver tolerance, SAFE policy, model or radius was changed. The older
non-dyadic singleton sparse-rounding limitation remains a known limitation.

CUDA smoke used separate disposable weights, verified nonzero router updates,
bitwise checkpoint inference replay and one restored optimizer update. Production
started afresh at seed17. Full training ran from the committed source snapshot,
with two loader workers/CPU threads and nice10. Observed worker GPU allocation
peaked at84.42MiB; nvidia-smi process usage was approximately820MiB including
context/cache. Other users' GPU processes were not interrupted. The recorded
training loop plus selected-test phase took229.84s, excluding earlier source
preparation, compatibility, smoke and separate landing audit. This is not a
paired speed comparison or a promised runtime on other hardware.

The separate audit process checked all100 checkpoint hashes and metrics, rebuilt
the complete disjoint split, checked45000/5000 denominators and the cosine LR
schedule, reconstructed earliest-maximum selection, and independently reloaded
the selected model for full validation/test evaluation. Both metric records
matched exactly. `CONV_LANDED_summary.json` and supervisor status are
`LANDED_AUDITED`. This is concrete training-artifact replay, not an independent
proof of network bounds or deployed floating-point robustness.

The final compact archive checks landing/source hashes and all epoch records.
The related factory, training/supervisor, staged/reuse and rational-evidence
regression suite passes66 tests. All failed controls and all training checkpoints
remain; no production training retry was needed. No family training job remains
running at this completion.

## Next scope, not an inferred success

This is a distinct trained architecture with higher observed clean accuracy than
the earlier approximately48% MLP family. It is not yet evidence of cross-
architecture verification gains, high-accuracy deep-model strict certificates,
or superiority to an external backend. Those conclusions require a separately
frozen three-arm experiment on this selected checkpoint: ACT adaptive,
reasonable monolithic and ACT-fronted variable-weight static CROWN. Explicit
four-expert handling, input/selection identities, per-request budget and evidence
levels must be checked before that experiment; do not silently reuse eight-
expert assumptions or open the old sealed AdvMoE holdout.
