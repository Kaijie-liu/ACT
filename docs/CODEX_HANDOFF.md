# MoE project handoff

Updated 2026-09-11. Latest schedule experiment execution HEAD: `d7ac0b0a9`.
Earlier reuse experiment execution HEAD: `751e386d8`.
Latest four-arm paired-result reference HEAD: `631f211bc`.
The earlier common-task result remains at `70869e84bb6f7023c03354a322e3e70abf393a38`.
Check live Git/process state on arrival;
do not infer running jobs from historical conversation progress reports.

## Objective and implementation

Prove complete MoE output robustness even when perturbations change routes,
using the relationships among input, router, legal guards, and expert outputs.
Demonstrate the contribution of retained relationships beyond path enumeration
or a stronger downstream backend.

The production API is `verify_staged_linf` in
`act/pipeline/moe/staged_verifier.py`. V1 supports eval-mode CPU/float64,
output-level selected-softmax top-2. Exact route analysis precedes guarded
expert-wise Tier 1 and conditional property-directed F0 McCormick fallback.
The normalized top-k proof is broader than this public implementation contract.
Read `act/pipeline/moe/docs/staged_verifier.md` and
`act/pipeline/moe/audit_staged_evidence.py` for the evidence/acceptance contract.

## Current audited facts

- Original boundary confirmatory: 36 route-changing SAFE, with 5 from Tier 1
  and 31 from F0. Original overall solved 56/100 failed its 60% gate. Closure
  is separate; do not backfill that failure.
- Earlier seed-1/2 boundary replication: 13/40 and 6/40 route-changing SAFE;
  the full registered conjunction passed 0/2. These use model-specific radii.
- New common task: the same 100 jointly clean-correct images and `2/255` for
  all three bal010 runs. SAFE/UNSAFE/UNKNOWN/TIMEOUT are 30/24/39/7,
  26/21/38/15, and 44/22/25/9. Route-changing SAFE: 8/8/7. Complete outcomes:
  54%/47%/66%. Full bundles pass 2/3; seed 1 misses the frozen 50% coverage
  threshold. The all-model conjunction is false. Audit: zero issues, 297
  complete packages, 67 replayed UNSAFE. Three hard deadlines explain the
  missing full packages; denominators remain 100 each.
- Common-task F0 SAFE additions: 12/7/15; complete resolutions 27/62, 19/50,
  31/53. These are within-run stage contributions, not a full-budget Tier-1-only
  ablation. Candidate reduction, width, and guard gates pass for each model.
- The models have about 48% clean accuracy and share one architecture/recipe.
  Common-task coverage is not certified accuracy, cross-architecture evidence,
  or a new matched baseline competition.
- Accepted AdvMoE compatibility checkpoint: 85.67% clean accuracy. CROWN and
  Lagrangian experiments supply numerical filters, not strict certificates.
  The frozen strict PyRAT pilot has two requests/four static-path queries, all
  TIMEOUT, strict SAFE 0/2. High-accuracy real-scale strict new certificates
  remain unachieved. Probe agreement does not prove full-domain export identity.
- Lagrangian development: local bound improvements but identical 2/100 complete
  positive filters and higher cost. Four-cell graph/config check did not show
  that router removal or best-bound retention explains the huge sparse-alpha
  negative bounds. No unique global cause was established.

## Authoritative reading order

LATEST COMPLETION: route-complexity smoke (6) and full pairing (60) finished.
Separate re-audit reproduces both saved summaries: full PASS / 0 issues,
49 packages, 13 UNSAFE replays, 11 outer deadlines retained. Adaptive versus
matched monolithic SAFE/solved counts (denominator ten per arm/model): 7/8 vs
6/7, 3/7 vs 2/5, 4/7 vs 3/4. Adaptive gains one multi-legal-route SAFE per
model, with no SAFE/solved losses. All 22 comparable common-fact pairs match;
8 are unavailable, not equal. Mean savings are tail-sensitive; median paired
differences are under 0.26 s. This is observed-cohort engineering, not a new
holdout or general superiority result. Read
`act/pipeline/moe/results/route_complexity_paired_review_20260911_r1.json` and
the completed-result section of `act/pipeline/moe/docs/route_complexity_schedule.md`.
No experiment remains running or queued by this completion. The following
launch/pending notes are historical. Defaults, numerical gates, old R1s and
sealed searches remain unchanged; further experiments require separate scope.

Next execution frozen (2026-09-11): `configs/route_complexity_paired_r1.json`
and `route_complexity_paired.py` under `act/pipeline/moe` run a six-request
smoke then a 60-request observed-cohort adaptive/matched-monolithic pairing.
Same ranks 0--9, three models, 2/255, 300-second external caps; no new holdout.
Both arms independently compute the common facts; the audit compares actual
intervals, not just fact counts. A complete matching smoke gates the full run;
errors fail-stop and retain artifacts, no resume/overwrite path. 47 focused
tests pass. See `docs/route_complexity_schedule.md` under the pipeline for
registered acceptance and caveats. Planned session `moe-route-complexity-r1`,
log `data/moe/results/route_complexity_pipeline_20260911_r1.log`. Verify live
state: registration does not itself mean execution. Do not edit the checkout
while the pipeline runs. Final audit is automatic, final commit/push is not.
No candidate-superset fallback, external tool or request-level LP stage is
included, and the historical runs remain untouched.

Newest implementation (2026-09-11): opt-in route-complexity scheduling and
matched monolithic scoped reuse are implemented. Read
`act/pipeline/moe/docs/route_complexity_schedule.md`. Configs
`route_complexity_reuse_v1.json` and `monolithic_matched_reuse_v1.json` share
the common guarded interval prelude and total budget; single pair goes direct
weighted, multi-pair adaptive allocates 25% of remaining time to Tier 1 and
then residual F0. Monolithic can discharge the same per-pair/property facts.
41 focused tests pass. Only analytic controls ran; no trained-model scheduling
comparison is started or queued. Native calls need an external watchdog;
the internal budget is cooperative. Defaults and old R1s remain untouched.
Do not conflate the new no-support common fact prelude with the historical
support-tightened Tier-1 source when comparing costs. Candidate-superset
fallback, external-tool integration and request-level LP proof packs remain
separate future stages, not part of this implementation or its validation.

Newest completion: `act/pipeline/moe/docs/proof_reuse_engineering.md` and
`act/pipeline/moe/results/proof_reuse_paired_review_20260911_r1.json` supersede
the launch/pending notes below. All 60 reuse off/on requests finished; separate
re-audit PASS, 0 issues, 60 packages and 12 UNSAFE replays. SAFE off/on counts
are 4/5, 1/1, 3/3 (ten inputs each); sole gain seed0/rank0, no losses. Mean
seconds 60.33/41.77, 106.23/92.05, 89.18/69.30; median paired savings under
one second. This is observed-cohort engineering, not holdout, general speedup,
high-accuracy evidence or independent full-network proof. Actual stored-HZ
export checking has separately passed (see linked result). No experiment is
currently running or automatically queued by this completed stage. Reuse stays
opt-in; mathematical/numerical gates and all historical results are unchanged.
Next work needs its own scope: no implicit enlargement of this sample or
sealed searches, and no inference that independently checked LP bounds certify
upstream network-to-HZ propagation or MILP search trees.

Latest addition first: `act/pipeline/moe/docs/paired_followup.md` and
`act/pipeline/moe/results/paired_followup_full_review_20260911.json` report
the completed four-arm comparison, including monolithic's coverage advantage.

1. `act/pipeline/moe/docs/staged_multimodel_performance_bundle.md`
2. `act/pipeline/moe/results/staged_verifier_multimodel_bundle_20260906_r1.json`
3. `act/pipeline/moe/docs/advmoe_strict_pyrat.md` and its linked R3 result
4. `act/pipeline/moe/docs/lagrangian_top1_guard.md` and linked diagnostics
5. `act/pipeline/moe/docs/monolithic_f0_baseline.md`
6. `paper/evidence_table.md`, `paper/sections/08_evaluation.md`
7. `act/pipeline/moe/EXPERIMENTS.md` for historical chronology

Tracked JSON manifests point to hash-bound raw artifacts under
`data/moe/results`; those and checkpoints are generally not in Git. A fresh
clone alone is insufficient to rerun experiments. Read the exact manifest
before choosing a checkpoint, cohort, or config. Resolve disagreements by
checking raw artifacts and the recorded auditor scope, not by trusting prose.

## Latest review: next work in priority order

1. Build the common-task, equal-total-budget paired follow-up comparing full
   staged, route-invariance plus the same weighted F0/backend, monolithic F0,
   and Tier-1-only with the whole budget available. A stable top-2 set still
   has variable gate weights. Include construction, candidate analysis and
   support in each method's total budget. Charge reused preprocessing explicitly;
   do not give one method free oracle results from the census. Interleave methods
   by input with frozen order balancing. Rerun the full method for paired timing;
   historical times cannot substitute. Report SAFE and UNSAFE separately,
   gained/lost solution sets, timeouts, and per-model results. Follow-up use of
   observed data is explicit; algorithm tuning makes it development data.
2. After establishing the comparison, evaluate scoped Tier-1 proof reuse in
   F0. Only reuse a proven property on a superset of the target pair domain,
   binding model, property, guard containment, frame, and numerical policy.
   Both expert obligations must justify skipping a mixture property. Never
   reuse variable identifiers as if independently propagated frames were shared.
3. First inspect recorded primal/dual progress for possible sign-based early
   completion. Preserve the current optimal-status acceptance gate until a
   separately checked bound contract is implemented and validated. A positive
   solver-reported dual estimate alone does not authorize a numerical-policy
   change. Start independent bound checking with a small LP obligation;
   do not describe it as a complete MILP proof checker.
4. A future external-validity target should be a preregistered different,
   moderate-scale configuration bridging bal010 and AdvMoE. Architecture,
   training and checkpoint selection must precede verification outcomes.
   This remains a proposed next model, not an already selected experiment.

The fair baseline runner/config are now implemented in
`act/pipeline/moe/paired_followup.py` and `configs/paired_followup_r1.json`.
See `act/pipeline/moe/docs/paired_followup.md` for exact semantics. Next execute
and independently audit the mandatory 12-job smoke (one-hour worker-budget
ceiling), then the 1,200-job full observed-cohort follow-up (100-hour ceiling,
plus auditing). All four methods have the same external 300-second cap,
including startup, data/model loading and route analysis. They run sequentially
with rotated ordering and one-thread solver settings. This is not a holdout or
a retrospective timing correction. No new training is authorized by this step.
Executable-source/config identity must match the smoke; documentation-only
commits are allowed. Preserve failed attempts; never overwrite a result root.

Execution started 2026-09-07 21:13 Sydney after implementation `154fce45b`
was tested (20 tests) and pushed. tmux: `moe-paired-followup-r1`; log:
`data/moe/results/paired_followup_pipeline_r1.log`. This is an active
smoke → independent audit → full follow-up → independent audit chain, with
fail-stop shell gates. Check the live log/JSON rather than inferring completion
from this handoff. Do not launch a second copy or change executable sources
while it runs. After completion, check the terminal audit, preserve all rows,
write compact tracked results and update paper/EXPERIMENTS, then commit/push.
This chain does not automatically publish or push experimental outcomes.

2026-09-08 live review: smoke completed 12/12, structural audit PASS with
11 complete packages, zero issues, and zero UNSAFE to replay. Independent
re-audit is tracked at
`act/pipeline/moe/results/paired_followup_smoke_review_20260908.json`.
Full run was live at 636/1200 rows (through rank 52), not complete or finally
audited. The active full command has no `--resume`; its parent shell records
the successful smoke/audit gate. Known defect: a first invocation with
`--resume` and a nonexistent output directory can bypass the smoke gate.
Do not use that path. Fix with regression tests after this frozen run, not by
changing executable sources midway. Smoke PASS is not evidence of superiority.
The four arms are internal ACT algorithm comparisons, not four independent
external tools. External baseline claims supplied in conversation still need
primary-source verification and pinned artifact/semantic registration before
adoption; no external tool installation or execution has been started.

## Frozen decisions and boundaries

Latest completion (2026-09-11): the four-arm full run has finished. Re-audit
PASS, 0 issues, 1,075 complete packages, 171 UNSAFE replays, all 1,200 rows.
Tracked analysis: `act/pipeline/moe/results/paired_followup_full_review_20260911.json`;
full tables: `act/pipeline/moe/docs/paired_followup.md`. Staged SAFE/solved
30/54, 26/47, 44/66 beats invariance and Tier-1-only counts on each model, but
monolithic gives 46/65, 36/54, 52/70. Staged is less expensive under the
registered schedules and has distinct route-changing certificates; it is NOT
the overall coverage winner. Both positive and negative findings are now in
the evaluation text. No MoE comparison process remains running. Fix the known
resume-entry defect next, with regression tests, without rewriting R1 data.
The recovery repair is now implemented: resume requires an existing directory
and runtime identity; full resume must pass smoke audit and code/config checks.
Three new negative-path tests and the complete 26-test focused suite pass.
R1's runtime/source hash and all outputs remain untouched. Next pursue scoped
proof reuse / independently checked bound evidence, keeping optimal-status
acceptance unchanged. No new experiment is currently running or queued by
this completion stage; do not assume automatic follow-on work.

Do not chase seed 1's 50% threshold, add seeds until a pass, overwrite any R1,
reopen Lagrangian/CROWN tuning or its locked holdout, silently expand PyRAT
budgets, or reopen init/census work. F1 remains untriggered. No third census
dataset or ViT-224 end-to-end certification. Contact is managed by PI;
publication accounts, licenses and release permissions are not supplied here.

All HZ SAFE wording is scoped to the recorded HZ/HiGHS acceptance policy.
Evidence-package integrity is not independent re-proving of SAFE. Shared
architecture replications are not stable superiority over competing methods.
The 50% gate is a registered acceptance criterion, not a baseline score.

## Local validation and handoff routine

2026-09-11 next-stage implementation: optional config
`act/pipeline/moe/configs/staged_verifier_proof_reuse_v1.json` enables scoped
Tier-1 **guarded interval property** reuse in F0. It does not reuse partial
MILP proofs (not yet exported) or relax the old acceptance policy. Independent
checking currently covers supplied finite-box LP bounds using rational
arithmetic, including interval margin arithmetic; it does not validate HZ
propagation/LP lowering or MILP trees. Read
`act/pipeline/moe/docs/scoped_f0_proof_reuse.md`. Next run the retained analytic
controls, then freeze a separate observed-cohort engineering comparison before
making any real-model efficiency claim. No old R1 rerun/closure is authorized.

Retained controls completed at `cceadd326`, result
`act/pipeline/moe/results/scoped_proof_reuse_controls_20260911_r1.json`:
reference/reuse SAFE, F0 rows 2/1, both audits 0 issues, exact supplied LP bound
3/4 independently rechecked. Full focused suite: 34 passing tests. Opt-in only;
no real-model comparison or HZ-to-LP proof-export check has been executed yet.

Next stage now registered in `configs/proof_reuse_paired_r1.json` (under
`act/pipeline/moe`): observed common ranks 0--9, three trained models, reuse
off/on, equal 300-second caps, 60 jobs / five-hour maximum worker budget.
Run `hz_lp_real_control` first for seed0/rank0, audit via `audit_hz_lp_real`,
then start `proof_reuse_paired`. Both write new roots. See
`docs/proof_reuse_engineering.md` under the MoE pipeline. Real HZ export checking
is limited to the given stored HZ -> continuous binary relaxation -> rational
LP bound chain; upstream network propagation remains trusted. Do not call this
a high-accuracy strict full-model certificate. Runner has no resume/replacement
path and stops on code drift or execution errors. Check live state before
launching; no implicit permission to overwrite a failed root.

Real HZ export completed at `f5ea8457d` and passed independent audit, zero
issues: `act/pipeline/moe/results/hz_lp_real_export_20260911_r1.json`.
Given-HZ support bound ~9.485071885777709, 3,075 factors / one relaxed binary,
seed0/index3000/pair{4,5}/score4-score0. No full output or network propagation
proof is claimed. Next launch target is tmux `moe-proof-reuse-paired-r1`, log
`data/moe/results/proof_reuse_paired_20260911_r1.log`, running the frozen
60-request comparison with an automatic final audit. Check live state: do not
infer completion from the launch target. It stops on a dirty checkout or source
drift between requests, so do not edit during execution. Commit/push compact
paired results only after audit; there is no automatic result push.

Use `conda run --no-capture-output -n act-py312 python -m unittest` with the
relevant test modules. Last targeted run passed 11 tests in
`act.pipeline.moe.test_freeze_staged_multimodel_bundle` and
`act.pipeline.moe.test_staged_verifier`. `pytest` is not installed in this env;
the evidence-audit tests are inside `test_staged_verifier`, not a standalone
`test_audit_staged_evidence` module.

Keep this file current after each completed stage. Record evidence links,
remaining questions and the exact next action; preserve previous scientific
endpoints. Commit and push completed stages. Only one session should write to
this checkout at a time.
