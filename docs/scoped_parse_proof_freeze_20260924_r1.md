# Full-proof parsing comparison frozen — NOT EXECUTED

Implementation: `7c3a8915891498285e294fb50ca67b9867601f4b`.
Started from clean `feat/moe-route-verification`; original proof/parse code is
unchanged. The user requested integration, controls and freeze, not execution.

- [Controls](scoped_parse_proof_controls_20260924_r1.json): **85 PASS**,15 new
  full-pipeline/batch controls and70 original regressions;53.446 seconds control
  process wall time. Synthetic model/LPs only, not real-model latency evidence.
- [Protocol](scoped_parse_proof_protocol_20260924_r1.md): exact scope, charges,
  partial evidence, stopping rules and trusted boundaries.
- [Execution config](../configs/backend_controls/scoped_parse_proof_compare_r1.json):
  SHA-256 `0bfe49e812d3eefebce3d48ceb313571e33613baa96a49210e6ff6ccb044c908`.
- [Independent freeze review](scoped_parse_proof_freeze_review_20260924_r1.json):
  PASS/zero issues;515 source files checked, asset SHA-256 bindings checked,
  both planned requests and all252 obligations each verified. Review reproduced
  with `python -S --check`; no checkpoint/input decoding or inference.

## Fixed comparison

| Item | Frozen choice |
| --- | --- |
| Input | Old manifest seed0/rank1, CIFAR10 index4096, label8 |
| Selection | Next manifest rank after sealed4088; no outcome/route predicate |
| Model | Same frozen seed0 weighted top-2 checkpoint |
| Input domain | Exact stored float64 center, rational2/255, clipped[0,1] |
| Obligations | All28 unordered pairs ×9 classification margins per arm |
| Arms/order | Adapter cache disabled, then enabled; once each |
| Per-request budget | 300s including2s publication reserve; two thread limit, sampled8GiB |
| Proposals | Original half-remaining-time stage, one equal-share LP attempt per row |
| Acceptance | Original exact source/guard/output checks and complete positive bound aggregation |

Both arms generate their own new source and matrices. No old matrix, positive
bound, rounded input endpoint, route exclusion or witness is reused. This is
one observed engineering input, not two independent samples or new holdout
coverage. The disabled adapter is not the uninstrumented production builder.
The previous three-mode synthetic study remains the original-overhead reference.

The full synthetic tests demonstrated identical source/matrix/bound results
across modes; complete nonpositive evidence remains NOT_CLOSED. Tests reject
wrong bindings, partial proposals, late positive files, missing construction
receipts and matrix changes. Hard cutoff, proposer error, memory refusal and
receipt overrun are costed. Batch tests preserve unstarted entries with missing
cost explicitly, and reject missing/duplicated terminal records.

These controls do **not** establish that either real arm will reach its LPs,
close its252 bounds, improve cost or add a certificate. All real results remain
uncomputed. No route-changing or native floating-point guarantee is added.

## Explicit execution gate

New output root is absent:
`data/moe/results/scoped_parse_proof_source4096_compare_20260924_r1`.

Check-only commands, safe before execution:

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -S -m scoped_parse_proof.controls --check
/data1/Kane/miniconda3/envs/act-py312/bin/python -S -m scoped_parse_proof.review_freeze --check
```

Only after a later explicit user execution authorization, on the clean registered
branch and after resource admission:

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -m scoped_parse_proof.run execute --acknowledge-new-frozen-comparison
```

That command has **not run**. No queue or background watcher is installed.
It runs the two independent, fully charged requests serially, preserves all
terminals, and does not resume, retry or expand. Once complete, administrative
saved-only audit is separate from request acceptance/cost:

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -S -m scoped_parse_proof.batch_audit data/moe/results/scoped_parse_proof_source4096_compare_20260924_r1
```

Archive the actual outcome even if both time out or have nonpositive evidence.
An audit cannot manufacture absent evidence, erase expired time or turn a
construction/check-only result into a positive output proof. Seal this finite
comparison before proposing any further work.
