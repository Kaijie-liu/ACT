# B1 landed: official-code RT-ER seed 0

The frozen 130-epoch official-code, Blackwell-compatible dependency
reproduction completed and passed the unattended identity and endpoint audit.

- Ordered full-test standard accuracy: `34.2200%`
- Ordered full-test PGD-50 accuracy: `32.7000%`
- Standard-accuracy branch: `SEED0_RUN_DOES_NOT_REPRODUCE_REPORTED_SA_WITHIN_PREREGISTERED_TOLERANCE_SEED1_REQUIRED_FOR_PIPELINE_LEVEL_WORDING`
- PGD-50 branch: `PGD50_RA_OUTSIDE_PREREGISTERED_TOLERANCE`
- Pipeline-claim status: `SEED1_REQUIRED_BEFORE_PIPELINE_LEVEL_FAILURE_WORDING`
- Full-model replayed attack endpoints: `10000`
- Endpoint audit issues: `0`
- Epoch-130 checkpoint SHA-256: `6954f7a4d0768c86853f1353e8f17c972dc41369d40c5fb7c5065da2a8b3dbd1`
- Endpoint summary SHA-256: `a6daa93ee5f19e0b6dc090743ae5b121a2d3afd0a4ef7d976db51eba9e636c4c`

The original thresholds remain unchanged. Matching or missing them is a
single-seed reproduction outcome under the disclosed compatibility environment.
If seed 0 misses the SA interval, seed 1 is required before any pipeline-level
failure wording. This result does not establish author-checkpoint identity,
theorem applicability, or a general claim about the paper's method.

## Post-landing diagnostics

The complete 130-epoch trajectory confirms that this is not a last-checkpoint
dip. Clean test accuracy peaks at `37.40%` at epoch 30 and remains in
`32.96%--37.40%` from epochs 20 through 130. The endpoint ratio
`RA/SA = 32.70/34.22 = 0.95558` is an explicit diagnostic of a run that learned
little, not a causal explanation. The released augmentation path is present in
the unchanged executed source; no tensor-level transform trace was retained.

The paper/source audit classifies optimizer, weight decay, mixed precision,
and exact augmentation as paper underspecification rather than contradictions.
The paper describes cyclic LR as starting at `1e-4`, while the released
`CyclicLR` initializes at `5e-5` and uses `1e-4` as its maximum; this remains a
text/code semantic ambiguity. Independent Tier-0/Tier-1 audit: `0` issues.

Seed 1 is frozen as the required follow-up and uses the same endpoint and
thresholds. It may support pipeline-level wording only after its independent
landing audit completes.
