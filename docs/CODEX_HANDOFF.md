# MoE project handoff

Current next-stage implementation (2026-09-15): user authorized durable F0
substage timing, then ONE old smoke request at the unchanged300s cap. Read
`docs/conv_f0_timing_r1.md` and `scripts/conv_f0_timing_protocol_r1.json`.
Diagnostic-only wrappers under scripts preserve all frozen ACT/old wrapper
hashes. Fixed input0/matched monolithic; no full90, retry, support ablation,
budget or policy changes. Test/commit/push first, then execute
`python -m scripts.run_conv_f0_timing` in act-py312 and independently audit.
Raw output `data/moe/results/conv_f0_timing_20260915_r1`; check whether it exists
before launch. An existing directory is not permission to retry. Old smoke
remains FAIL. This entry precedes execution; later review must report measured
substage times and censored spans rather than assuming all F0 time is solving.

Latest completion (2026-09-15): conv three-arm smoke R1 completed6/6 under
execution `c192bca4d3c3161abe3ae816471894be24e3d908`. Read
`act/pipeline/moe/docs/conv_three_arm_r1.md` and
`act/pipeline/moe/results/conv_three_arm_smoke_review_20260915_r1.json`.
Separate re-audit exactly matches automatic audit: PASS,0 issues,1 complete
HZ package,2 complete CROWN records,1/1 full-model UNSAFE replay,2/2 common-fact
pairs equal. Adaptive:1 UNSAFE+1 outer TIMEOUT; monolithic:2 outer TIMEOUT;
CROWN:2 UNKNOWN. No SAFE or numerical positive. Three outer timeouts and all
snapshots preserved; total observed requests1021.96s. Thirty focused tests pass.
SMOKE GATE FAIL: monolithic has no complete non-error record. Supervisor is
STOPPED_REVIEW_REQUIRED, no worker remains, no full90 started/queued. Do not
run the full cohort or extend budgets/replace inputs to bypass this gate.
The outer scheduler and terminal auditor are implemented; this is a frozen
budget conformance failure, not missing orchestration. Stop positions are
monolithic F0 (two) and adaptive Tier2 F0 (one), not established root causes.
Any next investigation must be separately scoped and preserve this failed R1.
Post-archive test repeat exposed an exists/read /proc reaping race in the
original test (not the executor). Frozen wrapper hashes remain unchanged.
Use `scripts.test_conv_three_arm_lifecycle` for subsequent controls:31 focused
tests pass, plus two11-test repeats. Failure and test-only repair are retained
in `results/conv_smoke_posttest_review_20260915_r1.json` under the pipeline.
The following prelaunch entries are historical and do not authorize rerunning.

Latest implementation (2026-09-15): conv three-arm outer orchestration and
terminal audit are now integrated under `scripts/`, preserving the frozen ACT
source inventory. Read the execution-wrapper section of
`act/pipeline/moe/docs/conv_three_arm_r1.md`. Thirty focused tests pass, and the
separate clean-only freeze audit remains PASS. User explicitly authorizes the
six smoke requests ONLY; run `python -m scripts.run_conv_three_arm` with
act-py312 after commit/push. Output `data/moe/results/conv_three_arm_smoke_20260915_r1`.
Single owned process group, shared lock, resource gate, 300s complete request,
fail-stop ERROR and retained timeout/partial snapshots; independent terminal
audit and conformance gate. No positive-count requirement. No full/pipeline
option exists: 90 full requests require a separate instruction after smoke.
At this implementation commit no smoke has yet run. Inspect live artifacts
before launching; never overwrite or resume an existing root. The older freeze
status below records historical preparation, not current missing wrappers.

Latest preparation (2026-09-15): convolutional three-arm R1 protocol and selection
are FROZEN, NOT EXECUTED. Read `act/pipeline/moe/docs/conv_three_arm_r1.md`,
`configs/conv_three_arm_protocol_r1.json`, `configs/conv_three_arm_selection_r1.json`
and `results/conv_three_arm_freeze_review_20260915_r1.json` under the MoE pipeline.
Same selected conv epoch89, E4/C10;30 new clean-only selected inputs, epsilon2/255,
adaptive versus matched-reuse monolithic versus ACT-fronted plain CROWN.
832 recorded previous indices from5,416 artifacts excluded; smoke indices0,1
are disjoint. Exact materialized tensors and source/method/model hashes bind all
arms. Separate-process clean reconstruction PASS,0 issues;24 focused tests pass.
90 full calls +6 smoke calls,300s each, input-blocked rotated order, one worker;
positive evidence levels are not interchangeable. No route or bound query ran.
The E4 worker adapter is implemented, but outer three-arm orchestration and final
auditor are NOT_YET_INTEGRATED; do not launch the old three-model batch scripts.
Next implement/test those wrappers without changing the frozen decisions, freeze
their execution identities, then obtain an execution instruction and run audited
smoke before full. No outcome-driven budget/recipe/cohort change or old holdout.

Latest completion (2026-09-15): convolutional family full-shape compatibility,
supervised training and independent landing audit are COMPLETE. Execution
`6d2e6d299`; read `act/pipeline/moe/docs/conv_training_results.md` and
`act/pipeline/moe/results/conv_training_review_20260915_r1.json` first.
Frozen seed17 ran100/100 epochs; selected epoch89 by earliest validation maximum:
validation68.08%, test67.06% on the full5000/10000 respectively. All100 immutable
checkpoint hashes/metadata and exact validation/test metric replay pass the
separate audit. Selected checkpoint SHA256
`f5781a792f844a68de941f1a6b314d0e627ad5dd30e262088e6c1864d6bc5289`;
local path `data/moe/results/conv_training_seed17_20260915_r1/checkpoints/epoch_089.pt`.
Supervisor/landing status LANDED_AUDITED; no training worker remains. Training
used an immutable Git source export, not later edits. No retry, reselection,
dependency change or production numerical-policy relaxation.66 related tests pass.
The following preparation now freezes three-arm trained-family verification
(E4, not E8); see the newer top entry for scope and remaining execution work.
Training is not a new certificate or a cross-architecture verifier win. Do not
retrain for a better accuracy or reopen sealed AdvMoE/backend studies.

Earlier prelaunch completion (2026-09-15): full-size convolutional family gate
and supervised training implementation. Read
`act/pipeline/moe/docs/conv_family_r1.md` and
`act/pipeline/moe/results/conv_pretraining_review_20260915_r1.json`.
Full-shape R1 passed ACT sparse retention but external CROWN failed on a default
dtype omission; preserve it. R2 sets the declared float64 default, same model,
box and backend, and passes ACT + plain CROWN conformance. All five ACT components
retain SparseHZ; all six static pair expressions pass; external pair{0,1}
returns nine finite ordered bounds. No positive-bound acceptance requirement.
CUDA two-batch smoke passes finite gradients, actual router updates, exact
checkpoint replay and one optimizer continuation. This is not production state.
`conv_training_supervisor` and `conv_training` implement source-snapshot training,
immutable epochs, full validation selection and independent final metric replay.
Nine new supervision/training unit controls and four factory tests pass.
Production100-epoch training was authorized for the next launch and is now
completed above. Check live supervisor before launching another job.
Three-arm trained-model verification remains separately scoped and unlaunched.

Latest completion (2026-09-15): Advice/bb.md fixed ACT-only rational transfer,
execution `fbc48d6a9`. Read `act/pipeline/moe/docs/request_lp_act_only_results.md`.
All three fresh generations complete; 115 exports independently rechecked.
Positive obligations5/9,26/27,15/18, but complete positive requests0/3 (UNKNOWN),
not115 independent network proofs. No old proof facts reused; complete worker
generation/check costs and substantial evidence sizes archived. Do not tighten
gate ranges or add retries to force these selected cases positive.

Manuscript now has abstract, introduction and discussion; method03 follows the
actual schedule and evaluation08 centers on four primary experiments. Historical
tables remain in `paper/appendices/historical_evaluation.md`, not erased.
`scripts/run_moe_proof_demo.py` is a source-defined no-download example that
runs the standard verifier and fresh rational proof plus a Python -S checker.
It passed from outside the checkout; see `paper/artifact_quickstart.md`. This is
not yet a tested clean-container distribution of the empirical model artifacts.

Second-family scope is now fixed in `docs/conv_family_r1.md` and its training
config under the MoE pipeline. Versioned conv factory/checkpoint loader is
implemented, with small conv/pool controls. At that earlier stage training
supervision, full-shape external conformance and three-arm evaluation remained
next; see the newer status above. A non-dyadic singleton sparse control
exposed inconsistent independently rounded bounds and failed closed; no
numerical gate was relaxed. Dyadic point and nondegenerate sparse-box controls
pass. Do not interpret small compatibility controls as certified model quality.

No experiment remained running at that earlier completed-stage handoff. Existing
high-accuracy strict-certification and cross-architecture outcome goals remain
unachieved; there is no acceptance guarantee.

Historical prelaunch (2026-09-14): method section03 now describes
the actual scoped-fact/route-complexity/residual-obligation algorithm, with the
legacy-only F0 trigger removed from the current version's description. The new
generic rational-request generator and fixed three ACT-only protocol are in
`act/pipeline/moe/docs/request_lp_act_only_r1.md`. Tests and freeze precede new
queries. Do not substitute cases or interpret LP UNKNOWN as an unsafe model.
The next separate workstreams are a materially different moderate-scale model
family and a reviewer-runnable model/request artifact. Neither is complete just
because the rational checker can run without Torch. No submission/acceptance
guarantee is made; existing high-accuracy/external-comparison limits remain.

Latest completion: external complete-cost comparison, execution `0de4fe1c7`.
Read `act/pipeline/moe/docs/external_pair_comparison_results.md` and linked
compact re-audit. Smoke6/full60 completed; separate re-audit exactly matches
saved audits, zero issues. Full:25 ACT packages +30 external records,5 retained
outer timeouts,9 full-model UNSAFE replays. ACT11 HZ-policy SAFE versus13 CROWN
numerical positives,8 shared,3 ACT-only(two multi-pair),5 CROWN-only. All five
CROWN-only rows are ACT solver-limit UNKNOWN. Mean complete costs138.11/4.23s;
the external path is much cheaper and has more positives overall. Report
complementarity, NOT ACT dominance or interchangeable proof levels.
Final regression:42 focused tests PASS; compact archive independently
reconstructs exactly. No dependency, solver or production-policy change.
No experiment remains running or queued at this archival completion.
The two requested deliverables are complete: external complete-request table
and rational pre-F0 construction checking. Do not retune25%, expand this cohort,
make an outcome-selected portfolio or reopen sealed searches. Next experimental
scope/model must be frozen separately; high-accuracy strict certificates and
independent full-dynamic-model external comparison remain unachieved.

Latest completion: direct rational construction R3, execution `f30f8ccf4`.
Read `act/pipeline/moe/docs/request_lp_rational_results.md` and linked review.
All 18 required output obligations independently check (15 reused, 3 residual).
The trusted base no longer includes floating F0 construction; network-to-HZ,
guard lowering and route exclusion remain trusted. Same R2 ranges, no policy
change. 28 focused tests and read-only review PASS, zero issues.
Historical registration of the now-completed external comparison:
read `act/pipeline/moe/docs/external_pair_comparison_r1.md`. Same ten observed
inputs; six old-input smoke requests gate full execution. It materializes one
shared raw input file, charges loads and the complete cross-env request, and
kills the owned process group on timeout. No numerical-policy changes.

Updated 2026-09-14. Latest 100-input confirmation execution HEAD: `bc0791976`.
Latest 30-input confirmation execution HEAD: `6ad58bc9d`.
Earlier development schedule execution HEAD: `d7ac0b0a9`.
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

CURRENT FOLLOW-ON (2026-09-14, PI approved bounded continuation): request LP
order-only R2 COMPLETED; read `act/pipeline/moe/docs/request_lp_order_results.md`
and `results/request_lp_order_review_20260914_r2.json` under the MoE pipeline.
Same old request/18 obligations, at most2 router LPs+3 residual LPs,600s outer
cap. Exact score-order facts imply dyadic gate envelopes without evaluating
sigmoid. Reuse hash-checked R1 proofs; do not resample or replace the R1 UNKNOWN.
Execution `db0b4b281`,33.13s,exactly5 new LPs. All41 stored exports/duals
rechecked;18/18 positive obligations (15 reused+3 residual),minimum~.18304675.
Status CHECKED_REQUEST_CONDITIONAL_ON_TRUSTED_LOWERING,not full-network proof.
R1 UNKNOWN remains separate. No further refinement/search is queued.
The other, separate stage is COMPLETED; read
`act/pipeline/moe/docs/external_static_pair_results.md`: old index3000/seed0,
two whole-box static pairs{2,4},{4,5},nine properties each,CPU/float64 plain
CROWN,120s/pair,no retry/tuning. `StaticSelectedSoftmaxPair` preserves actual
variable weights; finite conformance and model/input hashes are checked.
NOT a complete dynamic-MoE benchmark. Execution `7e911dfab`:both pairs9/9
positive numerical margins,minimum LBs3.05086109/3.09949436,all finite probes
match. Re-audit PASS0. No formal CROWN SAFE or performance claim. No dependency
install or sealed search. Both bounded follow-on stages have completed; no
experiment is running or automatically queued. Keep R1 failures immutable.

EXTERNAL FRONTEND R1 completed separately (2026-09-14):
`act/pipeline/moe/docs/external_compatibility_r1.md`,
`act/pipeline/moe/external_compatibility.py`. Three bounded CPU probes on
pinned alpha-beta-CROWN/auto_LiRPA source, existing Python3.11 environment.
Dynamic top2, static variable-weight pair, and relational input parser; no
full BaB/holdout/performance comparison. Read
`act/pipeline/moe/docs/external_compatibility_results.md`: dynamic TopK/OneHot
rejected; static variable-weight pair returns numerical bounds; full API box
accepted but relational input halfspace rejected. Provenance audit PASS0;
execution `84352b890`. No install. No request-LP or compatibility job remains
running at R1 completion. Full-model external competition remains open; the
later order-only R2 above supersedes R1's absence of conditional positive
request evidence. These controls do not authorize expanded searches.

REQUEST LP R1 completed (2026-09-14): separate frozen index3000/seed0
control in `act/pipeline/moe/docs/request_lp_r1.md`, `request_lp_control.py`
and `check_request_lp.py`. Sparse supplied-HZ LP dual checks and exhaustive
output aggregation are implemented. Read `docs/request_lp_results.md` under
the MoE pipeline and `results/request_lp_review_20260914_r1.json`: all36
supplied LPs rechecked,18 output obligations,15 positive through scoped reuse,
three negative residual F0 lower bounds. Overall UNKNOWN, no missing obligation,
not a complete positive certificate. Execution58.63s at `5c98e5399`.
No production SAFE gate,
budget or sigmoid range is changed. External compatibility remains separate.

LATEST COMPLETION (relation R1,2026-09-14): all6 smoke +60 full calls finished;
separate re-audit exactly matches saved audits. Read
`act/pipeline/moe/docs/relation_ablation_results.md` and
`act/pipeline/moe/results/relation_ablation_review_20260914_r1.json`.
48 packages,16 UNSAFE replays,12 retained outer TIMEOUTs,30/30 common facts
equal. Shared vs independent SAFE4/4,3/2,4/2 (ten/model): +3 SAFE,+5 solved,
no losses. Two gains are multi-pair versus relaxation UNKNOWN; one is single-
pair versus solver limit.27 recorded gate pairs agree,3 one-sided,not universal
observability. This is observed-input mechanism evidence, not new confirmation.
This relation stage is complete. The separate LP and external R1 completions
above supersede its former next-step notes. No dependency installation, sample
expansion or sealed-search reopening is authorized.

HISTORICAL REGISTRATION (2026-09-14, now completed): relationship-only R1 is
implemented and preregistered in `act/pipeline/moe/docs/relation_ablation_r1.md`.
The independent arm duplicates all guarded expert factors/constraints into a
Cartesian product; it retains scoped reuse, gates, Tier1 and25% scheduling.
Read `relation_ablation.py` and its fixed config before execution. Six old-input
smoke calls gate60 calls on the FIRST10 already-observed thirty-input R2 images,
not the new100. Single CPU timing worker; no resume/overwrite or effect-size
smoke gate. The pipeline writes final structural audits but does NOT commit or
push results automatically. Check process/runtime/log files for actual launch
state; implementation alone is not an experiment result. External compatibility
and complete-request LP checking remain separate, not delivered by this stage.

LATEST COMPLETION (2026-09-14): hundred-input confirmation is complete and
independently re-audited, NOT pending. Read
`act/pipeline/moe/docs/schedule_confirmation_100_results.md` and
`act/pipeline/moe/results/schedule_confirmation_100_review_20260914_r1.json`.
All900 full requests, PASS/0 issues, 739 packages, 198 concrete UNSAFE replays,
161 outer TIMEOUT terminals retained. All300 common-fact pairs agree; all600
scheduled snapshots survive, including67 after kills. Separate review exactly
reconstructs both saved smoke/full summaries and the frozen execution source.
Adaptive SAFE/solved 59/89,57/81,63/86; matched50/76,47/65,59/78;
legacy46/68,45/63,50/68 (100/model/arm). Primary +23 SAFE/+37 solved, no
losses. ALL23 SAFE gains have multiple exact legal pairs: 2 Tier1,21 F0;
all21 F0 gains record scoped reuse. This is source accounting, not reuse-off
causal ablation. Legacy +40/-2 SAFE, net38, not set dominance. Both losses
are single-pair UNKNOWN_MONOLITHIC_SOLVER_LIMIT, indices4150/seed0 and4142/seed1.
Primary clustered SAFE difference +7.67pp [4.67,11.00], descriptive/unadjusted.
These results confirm scoped new-endpoint internal net benefit, not merely
development potential; no high-accuracy, cross-architecture, independent
external-tool or complete floating-point proof claim. Old2/3 failure and
thirty-input result remain separate. 54 focused tests pass; review --check
reconstructs the archived supplement. No training or optimization was rerun.

ARCHIVAL WORKSTREAM CONTRACT (R1 stages above now completed): stop same-family
sample expansion and keep25% frozen. Read separate
workstream contracts in `act/pipeline/moe/docs/post_confirmation_workstreams.md`:
relation-only sound outer-envelope ablation, one external semantic-compatibility
path, and request-level LP obligation checking. None was implemented/launched by
the archival stage; the later R1 entry above supersedes its implementation state.
None may be folded into the completed900 or replace ACT/HybridZ. No jobs remained
running at archival completion. The following
registration/launch entries are HISTORICAL; they do not override this completion.

LATEST NEXT EXECUTION (2026-09-12): PI explicitly approved SEPARATE 100 new
inputs. Read `act/pipeline/moe/docs/schedule_confirmation_100_r1.md` first.
Selection 4088--4389, 529 excluded indices / 1,847 hash-bound artifacts,
clean-only independent reconstruction PASS, zero overlap with prior thirty.
`schedule_confirmation --config act/pipeline/moe/configs/schedule_confirmation_100_r1.json --pipeline`
runs nine fresh OLD index3000 smoke requests, re-audits/gates, then900 full.
Same frozen three methods/25%/models/2/255/300sec, no math or solver changes.
Primary matched, secondary legacy, per-model SAFE/solved gains/losses; 100-input
clustered descriptive intervals (not300 independent pairs). No pooling or
retuning. Shared lock, fail-stop, snapshots survive kills, no resume/replacement.
Target tmux `moe-schedule-confirm-100-r1`, log
`data/moe/results/schedule_confirmation_100_pipeline_20260912_r1.log`.
Inspect live state; code registration is not launch or completion. Source
checkout must stay clean/frozen while running. Final audit automatic; outcome
commit/push remains a separate reviewed step. External tools, LP request proofs
and candidate-superset fallback are NOT included.
Validation: 49 focused tests pass; separate process reconstructs100 selection
and reproduces both old R2 smoke/full summaries exactly after the extension.
Thirty-input results already committed/pushed at `cd32685ad`. The100 protocol
and new selection are a separate preparation commit before any new endpoints.

LATEST COMPLETION: 30-input R2 finished 270/270; separate-process re-audit
exactly reproduces both smoke/full summaries. Full PASS, 0 issues, 190 packages,
87 UNSAFE replays, 80 retained TIMEOUTs; all 90 common-fact pairs equal.
Read `act/pipeline/moe/results/schedule_confirmation_review_20260912_r2.json`
and completed section of `act/pipeline/moe/docs/schedule_confirmation_r1.md`.
Adaptive SAFE/solved 11/22, 11/25, 11/25 (30/model); matched 11/18, 9/18,
10/17; legacy 9/16, 8/19, 7/14. Primary SAFE gains 0/2/1, no losses,
all three gains multi-legal-route. Clustered SAFE interval [0,0.0889] includes
zero. Legacy net +9 SAFE includes one loss. No general superiority claim.
PI authorizes a SEPARATE 100-new-input, 900-request experiment with unchanged
models/strategy/radius/budgets, excluding this 30 and all prior recorded HZ
endpoints. Freeze/audit selection, tests, commit/push, fresh old-input smoke,
then full. No pooling, retuning or performance-based stopping. External tools,
LP request proofs and candidate-superset fallback stay separate. No previous
job remains active at this completion; older launch notes below are historical.

R2 REPAIR BEFORE CONFIRMATION: R1 smoke stopped at its first OLD input 3000
with a represented-input identity mismatch. The selector scaled ToTensor in
float32 then cast; CLI initializes float64 before ToTensor. No new confirmation
endpoint ran. Preserve R1 failure and original selection/audit. The selector
now follows CLI initialization. R2 clean-only re-audit passes, with ALL thirty
indices unchanged, same exclusion union, and smoke request identity matching
the retained actual worker package. Read the R2 section of
`act/pipeline/moe/docs/schedule_confirmation_r1.md` and
`results/schedule_confirmation_smoke_dtype_repair_20260912_r1.json` under the
pipeline. Default config/selection now use `_r2.json`; all solver method configs,
budgets, numerical gates and analysis stay fixed. Repeat all nine old-input
smoke requests before new endpoints, at new `_20260912_r2` roots. Planned tmux
`moe-schedule-confirm-r2`, log `schedule_confirmation_pipeline_20260912_r2.log`.
Inspect live state; no code edits during execution. R1 launch notes below are
historical. Do not count the failed row or its positive package as smoke PASS.

LATEST REGISTERED CONFIRMATION (2026-09-12): PI chose 30 new inputs. Read
`act/pipeline/moe/docs/schedule_confirmation_r1.md`. Selection is frozen and
separately reconstructed: indices 4006--4086, excludes 442 earlier indices from
1,643 hash-bound artifacts; same three models, CPU/float64 batch-one clean
semantics, epsilon 2/255. No new verification endpoint used for selection.
`schedule_confirmation.py --pipeline` (act-py312) runs nine **old-input** smoke
requests, re-audits/gates them, then 270 new requests across adaptive, matched
monolithic (primary), and old strong monolithic (secondary). Each has a hard
300-second cap. Snapshot audit works after outer kills, bound to frozen request
identity; legacy has no common-prelude overhead. No resume or replacements.
No strategy, 25% fraction, numerical gates, default or prior result changed.
Planned tmux `moe-schedule-confirm-r1`, log
`data/moe/results/schedule_confirmation_pipeline_20260912_r1.log`; inspect live
state, registration alone is not launch/completion. Do not edit this checkout
during execution. After final audit, independently review, archive compact
results and update/commit/push. No automatic Git writes in the runner.

LATEST PREPARATION (2026-09-12): read
`act/pipeline/moe/docs/route_complexity_confirmation_preparation.md` first.
Common-fact snapshots persist before arm-specific solves, with atomic no-clobber
publication and hash/scope/interval checks. A real SIGKILL test retains the
snapshot without a final package. Opt-in only; no historical run changed.
The old strong configuration is preserved as `configs/monolithic_legacy_reference_v1.json`
under the pipeline. Read-only `analyze_schedule_complementarity.py` produces
`results/schedule_complementarity_phase_review_20260912_r1.json`: 50/53
monolithic-only SAFE are single-pair, 17/19 staged-only are multi-pair. Twelve
`UNKNOWN_SOLVER_LIMIT` labels mean Tier-1 `violation_region_undecided`, not
candidate failure; original data remain intact.
No new endpoint is running. Next: settle 30 versus 100 new-input scope, audit
the exclusion inventory, freeze selection/statistics, implement a separate
three-arm runner/auditor including timeout snapshots and old strong reference,
test/commit/push, then observed-input smoke before confirmation. Do not call
this preparation a frozen or launched confirmation experiment.

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

## Historical review and execution notes (superseded, not the current queue)

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
