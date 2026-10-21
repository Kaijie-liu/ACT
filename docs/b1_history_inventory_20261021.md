# B1 historical identity inventory — bounded, not yet a selection freeze

This read-only investigation protects the proposed new-input comparison from
reusing previously diagnosed endpoints. It ran no model, clean-forward, solver
or attack and selected no new inputs. [Source identities and counts](b1_history_inventory_20261021.json)
bind the records below. This is **not** a final exclusion manifest or permission
to launch selection. Existing full-test telemetry must continue to be disclosed.

**Follow-up:** the Dual RS identity and named synthetic-family gaps described
in the first scan are now resolved below. The old scan's file sets were not
saved, so their counts cannot be retroactively promoted into an exhaustive
machine-checked exclusion manifest. No new input is selected by this document.

## Concrete omissions from the old selector

The902-entry exclusion list in `docs/reuse_supervised_v1_freeze.json` is historical,
not current. The following identities are absent from it:

| Source family | Previously observed CIFAR-10 test inputs | Meaning |
| --- | --- | --- |
| That freeze's own new samples | 220,222,230,232 | New at that old freeze, not new now |
| Router boundary figure | 3444 | Explicitly selected maximum route-boundary example |
| RT-ER conversion differences | 659,2316,4604,9957; route difference9604 | Particular anomalies selected from full-test telemetry |
| RoME frozen comparison | 6044,9399 | Higher-index endpoints already used; registered queue also contains1917,2890 |

RoME's configuration binds the same CIFAR-10 `test_batch` SHA256
`f53d8d457504f7cff4ea9e021afcf0e0ad8e24a91f3fc42091b8adef61157831`.
Its original control also used0. Separately, the RT-ER cross-conversion cohort
declares `ordered_test_indices="0--999"` for clean and PGD endpoints. This is a
schema-specific selected cohort, not a generic matrix `indices` field and not
merely a full-test aggregate. **An index above4389 is not evidence of novelty.**

Other schemas requiring explicit handling include `line_sample_indices`,
`sample_index`, `request_index`, `ordered_dataset_indices`, `probe_dataset_indices`,
and nested `requests[].{dataset,index}`. Generic `index/indices` also denotes LP
columns, CSR, layers, properties and slices; blindly unioning it is incorrect.

## Coverage actually checked

The initial bounded metadata scan covered55,336 JSON/JSONL filenames in the
five ACT results/config/docs and MOE-results roots;49,405 small files were read,
3,577 oversized files and2,354 source copies were skipped. Its six parse failures
were one historical mixed stdout file and five journal-fault fixtures; they were
not silently admitted as valid input records.

A follow-up read20,680,524 bytes, below its20MiB cap, including5,352 small
`baseline_runs` files,10 `baseline_data` manifests and selected larger ledgers:

- Four-arm `rows.jsonl`:1,200 rows/100 inputs, no bad tail; identities match frozen
  jobs and all1,200 independent terminals, including125 outer timeouts.
- AdvMoE Lagrangian `rows.jsonl`:100 rows/20 inputs0–19, no bad tail; matches its
  ordered selection. These identities were already observed despite failure.
- MetaMoE's selected CIFAR-10 inputs are1,2,4,5,7; MNIST inputs1,3,7,9,10 remain
  a separate dataset. Later paired/follow-up summaries reuse that selection.
- Dual RS certification uses CIFAR-10 inputs0,1. Training indices are not test
  endpoints, and clean-forward scan rows are distinguished from verification.

## Limits recorded at the first scan

This scan does not yet close all historical identity sources.125 large baseline
files mostly have smaller queue summaries, but not every link was verified;
76,040 other-family files (predominantly HZ/H1/H2 controls) were not individually
proved synthetic. Dual RS's first256 eligible-test selector identities were not
reconstructed from their hash/label references. A truncated Robust Experts
prepared record belongs to CIFAR-100, not CIFAR-10.

Do not convert the findings into a final count by appending only these12 omissions.
The next selector needs dataset/source-aware provenance and explicit treatment
of remaining schemas/coverage. It must exclude previously selected anomalies
without calling all10,000 full-test telemetry images new or, conversely, claiming
that every telemetry row was a verification endpoint. Fresh selection forwards
must be frozen and charged; no free128×3 preparation or previously consumed
control is authorized by this inventory. This closes specific selection-design
mistakes, not a new algorithm result or a completed B1 experiment.

## Follow-up: Dual RS test identity reconstruction

The standard-library-only [saved-source audit](../scripts/audit_b1_dual_rs_history.py)
read460,457 bytes and passed. Its [deterministic receipt](b1_dual_rs_history_20261021.json)
binds the original table, two configurations, two archives and mapping/digest
source files. Reproduce with the existing environment:

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/audit_b1_dual_rs_history.py
```

The exact frozen table is Fortran-order NPY `<f8`, shape `(10000,5)`; column-2
has8,719 nonzero rows. Its first256 identities reproduce the archived prefix
digest `691bbafe…` and occupy0–301 with46 omissions. The receipt lists all256.
They are explicit save/restore test controls and must be excluded. They are
already contained in the separate RT-ER0–999 cohort, but both provenance records
remain; overlap is not a reason to delete one source.

The full8,719 identities reproduce `60c9fbbe…`. The frozen training configuration
and saved archive select epoch90 regardless of test accuracy or certification.
Code review of `dual_rs_epoch_pipeline.py` confirms direct CIFAR-10 test Subset
indices and nonshuffled test evaluation, not a relabeled eligible-array index.
The test path computes selector sigma-label metrics, not clean predictions for
the three ACT MLPs. The scheduler receives no test metric, and the landing
script selects the final registered epoch, not the best test value.

Consequently disclose prior fixed-recipe test evaluation; do not call all8,719
rows new/unseen,8,719 ACT verification endpoints, or evidence of their clean
correctness on ACT models. The identity audit does not independently reprove
training or author certification. Formal certification remains the separate
two-input0/1 queue.

## Follow-up: named control families and large metadata

Read-only provenance review classified33 H1/H2 directories (84,078 files) and77
HZ directories (25,806 files). The [path/size roster](b1_history_family_roster_20261021.json)
binds the110 exact directory names and observed inventories. Classification is
by saved protocols/configuration/generator, not a general `hz_*` prefix rule.
The review also checked143 archived HZ configuration hashes against their
implementation bindings. This is source classification, not an audit of every
numeric matrix or bound.

| Reviewed family | Source basis |
| --- | --- |
| H1 capture/intake | Fixed-seed untrained Linear/ReLU fixtures, zero centers; no checkpoint or dataset |
| H1 sparse/H2 source, factored and rowwise controls | Handwritten `weighted_sign`, `tied_partial_reuse`, `unsafe_tied`, `unresolved_sign` sources or LPs derived from them |
| Full-size H2 capacity | `h2_capacity_recipe_20261001.json`, seed20261001, center1/2, explicit `no_checkpoint_or_dataset=true`;252 synthetic obligations, not CIFAR endpoints |
| HZ batch/device/physical | Four supplied HZ cases, including guarded positive/nonpositive and equality/private-binary controls |
| HZ propagation/binary64/sparse/store | Fixed coefficient, rounding, sparse-row or store fixtures;3072-dimensional generated sources are not dataset images |
| HZ source/endpoint families | Hash-bound H2 handwritten sources or tiny expert/gate fixtures |
| HZ native preparation/control | Explicit `fixed_control` sources with centers of shapes1×1×3,1×1×2 or1×1×1 |

Full-size capacity recipe SHA256:
`2f558fb692b1735d59a9814ad1744b7c500127fca80df47213e4f1e56512f3c0`.
Its saved preparation/execution bind the same source
`dda46f8cdf54a57eec207c70c7250f9bdc9433431bac9e2b254c219f650aaf54`.
`sealed_inputs` lists are exclusions, not executions. Statements about declared
real-number sources do not mean real dataset samples.

A **new, explicitly defined** metadata inventory finds127 JSON/JSONL files
strictly larger than65,536 bytes outside those110 named directories. The roster
records each path and size. It is not asserted to be the old125-file set:

| Large-file group | Files | Identity treatment |
| --- | ---: | --- |
| MetaMoE queues, traces, progress, diagnostics | 117 | Frozen CIFAR-10 identities0/1/2/4/5/7; MNIST0/1/3/7/9/10 separately |
| Dual RS full-training audits | 2 | Same fixed epoch90/8,719-test evaluation, not new certification |
| Robust Experts training/landing | 3 | CIFAR-100, not CIFAR-10 |
| Author plot font caches | 3 | Cached author-result replay, no new input call |
| Storage cleanup ledgers | 2 | File identities, not dataset indices |

Each MetaMoE large-record family is covered by its frozen small queue or the
explicit MNIST0 diagnostic. The saved LAS follow-up summary matches config SHA
`af662082da17e953960ba1127d0f0fcf8783f0dd5810ce9b0bd8f43ab6aa031c`.
No full LP matrices/checkpoints were loaded for this classification. Large
contents were not all rehashed or numerically reaudited, and the new110-directory
total109,884 must not be subtracted from the old filtered76,040 count.

## Concrete next step and remaining boundary

The two formerly open provenance questions now have specific answers. The next
selection implementation must combine the frozen902-entry base, its own later
selected inputs, schema-qualified cohorts/anomalies and these follow-up records,
preserving dataset identity and source hashes. A final prospective candidate
manifest and charged clean-forward queue are still required. Do not substitute
an index cutoff or a claim of global completeness for that manifest.

Separately, the [same-source diagnostic design](b1_same_source_diagnostic_design_20261021.md)
identifies the actual B1 data gap and a bounded54-slot proposal: original source
and raw solver-candidate capture in both staged and monolithic paths. It is not
yet implemented/frozen. No native model, selection, solve, environment repair
or GenBaB retry was performed by this historical audit. B2's failed saved audit
remains failed, and the real-stage ledger remains0/0/0.

**Later implementation update:** the optional source/raw-candidate interface
has since passed [complete synthetic request controls](b1_request_capture_results_20261021.md).
This supersedes “not yet implemented” above, but not the outstanding real
candidate/history manifest and diagnostic query freeze. No real selection has
been executed as a consequence of that interface work.
