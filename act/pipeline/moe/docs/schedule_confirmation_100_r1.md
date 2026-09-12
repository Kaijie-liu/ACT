# Separate hundred-input scheduling confirmation

Authorized by PI on 2026-09-12 after the completed thirty-input R2 result.
This is a new experiment, NOT a seventy-input append, pooled 130-image study,
or retry of the smaller result. The thirty-input result was separately
re-audited and archived at `cd32685ad`. No method changes are based on it.

## Frozen scope before any new verification endpoint

- Exactly 100 new, ordered jointly clean-correct CIFAR-10 test inputs, indices
  4088--4389, same three bal010 checkpoints, epsilon 2/255.
- Select the first eligible indices at/after 4000, excluding all earlier
  recorded HZ endpoint/selection indices, including the prior thirty. The
  immutable inventory binds 1,847 source files and 529 excluded indices.
  It includes retired selections and matched/legacy timeout ledgers even when
  there is no complete package. No route complexity, safety bound or endpoint
  decides selection. Coverage of unrecorded human inspections is not claimed.
- CPU/float64 eval batch-one clean forwards, with float64 initialized BEFORE
  ToTensor. Raw dataset, model state/checkpoint, center/lower/upper hashes and
  scanned clean predictions are saved. Prior telemetry means these are new
  verification endpoints, not completely never-seen images.
- Independent clean-only regeneration passes with zero issues, prior-thirty
  overlap zero. Manifest `configs/schedule_confirmation_100_selection_r1.json`;
  SHA256 `b4167d8ab501df06570bfb49ad646263962c79736c2252f1398dc0499d3a7ee6`.
  Audit: `results/schedule_confirmation_100_selection_review_20260912_r1.json`.
- No new checkpoints, training, smaller radius, altered numeric gates or
  candidate-superset fallback. No external tool, LP proof pack or relation-only
  ablation in this run. Their separate work remains pending.

## Three unchanged arms and full cost accounting

1. Adaptive route-complexity scheduling with scoped interval reuse: single
   legal pair direct weighted solve; multiple pairs allocate at most 25% of
   remaining time to Tier 1, then residual weighted obligations.
2. Primary: matched monolithic with the SAME independently generated, charged
   common guarded interval facts and scope-controlled property reuse.
3. Secondary: historical strongest monolithic config, no new prelude/reuse;
   legacy property limit 300 seconds within the same outer cap.

Method JSON hashes are unchanged from the thirty-input R2. The same worker
entry, graph, mathematics and HZ/HiGHS numerical policy are used. Only cohort,
experimental size, separate roots and size-aware audit/summary change.
All tie-legal selected-softmax top-2 paths are obligations; variable weights
are not frozen. Expert-only violations do not become UNSAFE without full-model
replay. Snapshot/structural audit is not independent reproof of all SAFE.

All 900 requests (100 inputs x three models x three arms) receive a hard
300-second subprocess cap, including startup/loading, route analysis, fact
generation, snapshot publication and solving. Audit/replay after each worker
is outside timed cost for all arms. One worker, one BLAS/OMP thread, no GPU.
Order rotates by input; each model/arm has 33/33/34 position counts, not exact
thirds (100 is not divisible by three). All failures/timeouts stay in each
100-row denominator, and load/versions are recorded on the shared server.

Both scheduled arms durably publish common facts before arm-specific solves;
even outer kills retain hash-bound observations. Missing facts remain unknown,
never equal. Legacy has no snapshot overhead. No cross-arm answers are shared.

## Fresh smoke and failure policy

Run all nine old-input smoke requests at index3000 first (three models/arms).
No new selected input is used for smoke. Re-audit smoke, require complete
packages for every arm and common-fact agreement, with source/config identity
matching full. Do not borrow old R2 smoke. Then run all 900 in fixed order and
save final structural audit with full-model UNSAFE replay. No resume, no
overwrite, no replacement, no outcome-based early stop. Errors/source/worktree
drift stop fail-closed and preserve evidence; this is not performance stopping.

All runs share the existing exclusive experiment lock. Do not edit this
checkout during smoke/full execution. Full completion triggers audit, not Git
writes; independently re-audit and archive compact results/docs/commit/push
after the run. Failed R1/R2 and the completed thirty remain untouched.

## Analysis fixed before outcomes

Primary: per-model gained/lost SAFE versus matched, denominator100. Report
SAFE, replayed UNSAFE, UNKNOWN and TIMEOUT independently; solved is SAFE plus
replayed UNSAFE, not certified accuracy. Secondary: the same comparisons versus
legacy. Preserve losses even when aggregate net gains are positive.

Costs include every capped request: paired means/medians and state counts,
not solved-only or uncensored speedups. Single-/multiple-route strata are
explanatory, never a replacement primary metric or selection criterion.

For both fixed contrasts and SAFE/solved, bootstrap 100 input blocks retaining
all three models (10,000 replicates, seed20260912, descriptive95% percentile
intervals). Never treat 300 model-input pairs or 900 method calls as independent
images. Intervals are unadjusted, not family-wise tests. Preserve per-model
effects and do not interpret a zero-width interval as population equivalence.
No retrospective effect threshold, baseline replacement, sample extension or
parameter tuning. Complete all900 regardless of partial effects. A small or
null SAFE gain remains a valid negative/mixed endpoint, not a reason to add
more inputs. Do not pool this run with the preceding thirty to chase significance.

This study addresses same-family moderate-accuracy confirmation only. It
cannot settle high-accuracy real-scale strict certificates, cross-architecture
validity, independent external-tool competition or a universal advantage.

## Launch and expected duration

Prelaunch validation: 49 focused unit tests pass across scheduling, snapshots,
scoped proofs, staged verification and both confirmation sizes. New checks
cover all900 jobs, 33/33/34 rotations, rank99 in the clustered summary,
30/100 profile confusion, prior-thirty overlap, immutable models/budgets,
matched/legacy package-less terminal exclusions and no-overwrite preparation.
A further separate process reproduces the 100-input clean selection AND both
old thirty-input smoke/full saved audits exactly after the runner extension.
No production verifier or three method JSON changed.

After tests/commit/push and a resource check, in act-py312:

```
python -m act.pipeline.moe.schedule_confirmation --config act/pipeline/moe/configs/schedule_confirmation_100_r1.json --pipeline
```

Target tmux `moe-schedule-confirm-100-r1`; stdout log
`data/moe/results/schedule_confirmation_100_pipeline_20260912_r1.log`.
Roots: `schedule_confirmation_100_smoke_20260912_r1` and
`schedule_confirmation_100_full_20260912_r1` under `data/moe/results`.
Registration is not completion: inspect live runtime/rows/final audit.
Prior observed costs suggest approximately 36--40 hours plus audit allowance;
hard worker ceiling is 75 hours full plus 45 minutes smoke, plus overhead.
There is no guaranteed effect size or significance from this sample size.
