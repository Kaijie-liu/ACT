# Full-budget shared-residual comparison frozen — before execution

Implementation `cc55c2ed8ba3425f6e4b0a508dadc16ae5efee52`; 121 controls pass.
The independent no-site review binds 565 source/protocol files, raw asset
hashes, the exact materialized center and all 252 output duties. No real model
was decoded by the freeze review. Output directory was absent at review time.

- Config: `configs/backend_controls/residual_proof_compare_r1.json`.
- SHA256: `e06061d1eacd0164dca1a0473c7945a01832614c35dc2de8caf8f2b9cf2113c8`.
- Model: existing seed0 checkpoint; manifest rank3, CIFAR4099, label4.
- Domain: exact 2/255, clip [0,1], unchanged original binary-rational margin.
- Calls: `pairwise`, then `shared`, once each. No retry or resumed directory.
- Whole pipeline: 300 s, two CPU threads, sampled 8 GiB, 2 s publication reserve.
- Same 56 ordered router-margin obligations; candidate representation is the
  only intervention. Both arms can discharge route-impossible duties and both
  must generate/check their own retained expert/output matrices and bounds.
- Same native output proposer and half-remaining proposal allocation. Nothing
  is borrowed from prior proofs or the other arm.

Review: `residual_proof_freeze_review_20260925_r1.json`, PASS, zero issues.
Commit/push this freeze before executing the finite roster requested by the
user. After execution, independent saved-only audit/archive must keep both
terminals, the original denominator and all costs. This freeze does not itself
claim any complete real certificate or validate any old missing source proof.

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -m residual_proof.run execute --acknowledge-new-frozen-comparison
/data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/archive_residual_proof_comparison.py
```

Do not rerun the first command after a failure. The archive supports read-only
reproduction via `--check`; its time is not budgeted verification time. Keep
inputs98/4088/4096/4098, historical source-gap claims and external tables sealed.
