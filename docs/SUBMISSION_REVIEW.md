# Final editorial pass, 2026-11-02

Zero model/solver calls; scientific data and conclusions unchanged. The finite
closeout remains complete; no research goal or experimental queue reopened.
[r4 PDF](/data1/Kane/MOE/results/architecture_closeout_manuscript_20261102_r4/architecture_main.pdf)
and [build.json](/data1/Kane/MOE/results/architecture_closeout_manuscript_20261102_r4/build.json).
Repeated qualifications are consolidated in section6; section4.4 points to the
same-input appendix. Cost is a remark; propositions1–3/proofs and the entire
route-envelope section are byte-identical. CTBench now cites Mao, Balauca and
Vechev (2024), [arXiv:2406.04848](https://arxiv.org/abs/2406.04848), checked on
2026-11-02. Disclosure records PI-set/approved decisions with AI assistance,
PI-directed implementation/analysis/drafting/review, non-independent AI review,
and human responsibility; final wording awaits PI confirmation.
Seventeen pages,4main tables/3figures; table rebuilding, source-preservation,
build and layout checks pass. Original r3 and private archive remain intact.
Submission and authorship remain PI tasks. Stop after this editorial report.

---

# Final closeout review status, 2026-11-01

The supplied [Claude review](/data1/Kane/MOE/survey/drafts/act_final_review_20261101.md)
is an internal AI review, not independent: Claude participated in directions
and gates. Codex revised code/analysis/manuscript. No human review is signed.
The finite closeout is authorized by [the current contract](architecture_closeout_20261101.md).

| Item | Revision and scope |
|---|---|
| M1 | Requested negative-result title; abstract gives question, scoped negative answer, diagnostic interpretation, and reusable method. |
| M2 | Exact clean/conditional arithmetic and two new figures. Corrected “entire deficit” to two components; learned-scale widths and exposure remain hypotheses, not identified causes. |
| M3 | Route-envelope IBP standalone section, tie-inclusive cover and enclosure proof, explicit E=1/top-1 dense degeneration; MetaMoE/Dual RS semantic distinctions; no priority claim. |
| M4 | One main-text section gives108/180vs0/180,2/52,+2pp[0,4]; full historical tables moved to appendix. Supplemental timing and178TIMEOUT+ERROR+UNSUPPORTED retained. |
| M5 | Training scope E8/top2/MLP/2/255; historical E16 and8/255 evaluation do not extend it. No CNN/SABR/TAPS/new experiment. |
| M6 | Removed storage, worker and audit-time narration from paper; retained experimental settings and one appendix registration timeline. |
| M7 | Four main tables generated from summaries; three figures; numerical-policy terminology and denominators retained. |
| Reproduction | Fixed two seed0 models/first50/IBP+complete; fresh environments on samehost. Complete: 100/100 discrete IBP flags agree; 97/100 complete terminals agree, three UNKNOWN/TIMEOUT exchanges, no positive or replayed-negative changes. [Result](../paper/results/s2a_reproduction.json). |
| Artifact | Locks, selected weight hashes, input list, engine snapshots/full9-edit8-file diff, smoke prefix and table rebuild. No publication performed. |

**Still external/unestablished:** independent human technical review; reproduction
on a separate host/OS; causal capacity/exposure explanation; source-complete
floating execution proof; priority or acceptance judgment. The finite closeout
need not pretend to complete these, and does not revive a scientific gate.

TMLR's official author guide requires its template; the draft uses preprint
mode to avoid a false under-review header. The AI contribution is disclosed
on page1. The official editorial policy retains human authors' responsibility;
PI must determine authorship and submission suitability. Sources checked
2026-11-01: https://jmlr.org/tmlr/author-guide.html and
https://jmlr.org/tmlr/editorial-policies.html (LLM-use/authorship sections).
The FAQ also expects human-sourced ideas/claims/results; this project must
not misrepresent AI research-direction assistance as mere grammar editing.

The older detailed review questions below remain available for a future human
reader; their blank sign-off is not replaced by the supplied AI review.

---

# Independent technical review: reductions and empirical limits

Updated 2026-11-01. This is a PI-managed review template, not a completed human
review or permission to contact anyone. Start with the current
[combined manuscript](../paper/architecture_main.tex), including its empirical
and source-boundary appendices. The [earlier verifier draft](../paper/review_main.tex)
is retained as a historical companion, not a separate submission. Consult the longer
[method](../paper/sections/03_path_conditioned_method.md),
[evaluation](../paper/sections/08_evaluation.md) and
[source contract](../paper/sections/05_soundness_engineering.md).
The current closeout uses the official TMLR preprint template; older builds retain their historical layout.

S1 has finished and its registered decision is **STOP_S1_RETURN_TO_REDUCTION_PAPER**.
The subsequent PI-authorized S2a has completed 24 training runs and its primary
evaluation on all 10,000 test inputs per model. Its decision is
**STOP_S2A_WRITE_RESULTS**: M has 30.124% IBP numerical certification versus
33.554% D-act and 31.972% D-tot, and a 73.920% single-route fraction. The fixed
secondary queues have finished all 20,000 CROWN-IBP and 4,000 complete requests.
The [final aggregate audit](../paper/results/s2a_final.json) reproduces the saved
statistics and retains the original cleanup error. M has 310/1000 complete
positives versus D-act 369/1000 and D-tot 336/1000, with higher all-request
median cost. This record audit does not independently reprove numerical bounds.
S2b is not launched.
Review the reductions and empirical limits actually supported, not the claim
that MoE must be easier to certify. The [S1 archive](architecture_s1_result_20261031.json) and
[final explanation](architecture_s1_20261029.md#final-s1-result-registered-stop-2026-10-31)
retain every comparison, seed, failure and cost. The architecture screen's
failure does not prove universal architectural inferiority. The
[S2a primary export](../paper/results/s2a_primary.json) and
[stage record](architecture_s2a_records_20261031.json) bind the separate
certification-training result; neither stop is relabeled as a passed screen.

## Read the argument before the receipts

The question is whether the problem, reductions and measured limitations support
a useful research contribution. Counts of controls or documents do not answer
it. No positive novelty verdict or review score is supplied.

| Claim / question | Evidence | Essential qualification |
|---|---|---|
| Complete-output composition | Short paper Proposition 1; detailed method | Every tie-legal history and property, same input/source, valid exclusions and bounds. An incomplete tree cannot be positive. |
| Output-atom guard reduction | Proposition 2; guarded frontend and semantic controls | Pointwise `max(p,-g)>0` equivalence, including ties; not equivalent finite-budget relaxation precision or a claim that generic tools can never exploit guards. |
| Independent-gate embedding and last endpoint | Proposition 3; M-A contract | Valid prefix ranges; last affine suffix and domain independent of last gate. Inner gates remain nonlinear. Numerical sigmoid intervals are not automatically exact enclosures. |
| Active capacity versus complete proof cost | Current paper cost lemma and both S1 capacity conventions | Routers, all retained histories, repeated construction, loading and evidence costs remain. Per-execution active parameters do not determine full certification cost. |
| Capacity-matched S1 primary | [Final archive](architecture_s1_result_20261031.json), complete 2/255 total/active comparisons | L1 114/600 versus 115/600 and 116/600; gain/loss 55/56 and 52/54. Both rate intervals include zero; both frozen point gates fail. Three fixed paired seeds share 200 inputs. |
| Incomplete-bound tradeoff | Same archive, incomplete 2/255 | 23/3000 versus 2/3000 each dense; +0.7pp/[0.400,1.067]pp, but 4.64%/6.27% higher median cost. It is a secondary endpoint, not certification training or a rescue of S1. |
| Larger-radius timing and guarantee | Same archive, radius 8; shared-clean-correct strata | All positives zero. MoE has more than half clean-wrong returns while dense has less than half, shifting the median from short replay to long unresolved proof. About 25x raw median ratio is not useful certification speedup. |
| S1 design limitations and accounting | [Frozen protocol and final report](architecture_s1_20261029.md), source-bound manifest | Ordinary MLP training; MoE-only balance penalty; approximate parameter matching; historically exposed all-input prefixes. All 31,200 calls settled, eight old PGD errors retained; 224.651 summed hours versus 28.094 parallel hours. |
| Route-envelope IBP and training | Current paper route-envelope section; [implementation](../act/pipeline/moe/route_envelope_ibp.py); [training freeze](architecture_s2a_training_freeze_20261031.json) | Every interval-compatible tie-legal pair is retained; property constants enter once and gate orientation is fixed. Candidate coverage does not prove route reachability. Training and primary evaluation share the bound function; floating-point results remain numerical-policy evidence. |
| S2a primary and stopped extension | [Primary export](../paper/results/s2a_primary.json), [stage record](architecture_s2a_records_20261031.json) | M minus D-act is -3.430pp, 95% interval [-4.108,-2.762]; M minus D-tot is -1.848pp, [-2.632,-1.130]. Five paired seeds and 10,000 common inputs are resampled, with selected configurations held fixed. No S2b or new configuration after failure. |
| Routing regularization and width interpretation | Same export; primary figure and layer appendix | M minus M-reg gives +29.370pp single-route fraction but only +0.200pp IBP accuracy, interval [-0.430,0.79005]. Learned-scale widths and different clean-correct populations are descriptive, not causal proof. Secondary findings cannot rescue the primary gate. |
| S2a completed CROWN-IBP portion | [Compact mode report](../paper/results/s2a_incomplete.json), stage record | All 20,000 requests settled, including one retained cleanup error. M 30.60% versus D-act 35.70%, D-tot 32.84%, M-reg 27.82%; M minus M-reg +2.78pp/[0.92,4.74]pp. M has higher median cost than dense. Method and cohort differ from the primary endpoint; the difference between these findings does not isolate either factor. The final audit reproduces this completed mode; the same-input post hoc analysis separates cohort and pipeline effects without changing any gate. |
| S2a complete secondary comparison | [Final export](../paper/results/s2a_final.json), stage record | All 4,000 requests settled with no errors: M 310/1000, D-act 369, D-tot 336, M-reg 301. M minus D-act -5.9pp/[-10.4,-1.2]; M minus D-tot -2.6pp/[-7.7,2.5]; M minus M-reg +0.9pp/[-3.3,5.0]. Same engine revision and branching configuration; all costs retained, including 144 timeouts. Different returned-unknown/timeout counts do not identify the cause. |
| Model validity and selection | [Training/model table](../paper/sections/08_evaluation.md#frozen-models-training-and-conditional-input-selection), original and residual training records | Clean accuracy27.65%–52.18%; low-accuracy/concentrated-routing seeds retained. Residual architecture and balance aggregation change together. Verification selects twenty common clean-correct inputs across all eighteen models, not a whole-test certified-accuracy sample. |
| Strong engine versus old ACT v1 | [Stage-1 records](multilayer_stage1_records_20261022.json), `final_result_20261025` | 108/180 pruned positives versus zero; ACT has 178 TIMEOUT, one ERROR and one UNSUPPORTED. Supplemental bottleneck gate was registered after 189 calls. No unique ReLU-cause claim. |
| Primary cascade gain and stop | [B4 records](moe_aware_b4_records_20261027.json), `main_and_ablation_review` | 122 versus 116, nine gains/three losses, +2pp/[0,4]pp; five-point threshold fails. Twenty input blocks preserve cross-model dependence. |
| Mechanism and cost | Same records, `main_gain_loss_trace_review` | All nine gains use V2; 48/52 endpoints close initially. Shared-positive overhead exceeds gain savings. Partial logs cannot prove a counterfactual budget choice. |
| Execution variability | Same records, ablation/repeated coupled rows | Four terminal changes among 120, including one positive to timeout; not four SAFE changes or a 3pp SAFE-noise estimate. |
| Revised guard contribution | [Result](guard_dual_refinement_result_20261029.json), [interpretation](guard_dual_refinement_20261029.md) | Full baseline retention and paid equal-extra-alpha control. One of eleven histories closes, not five; M-B/M-C stop. Local closure is not a full request. |
| Larger radius | B4 records, `radius_final_review` | 240/240 calls; 8 versus 6 numerical positives, gain/loss 2/0, +1.667pp/[0,4.167]pp. No gain among thirteen observed-multiple objects; 31 public counterexamples are not arm successes. Exact legal counts remain unknown. |
| Single-layer boundary and final accounting | B4 records, `final_review` | 57/60 positives in both arms, zero gain/loss; cascade median 18.688 versus 8.782 seconds. All 1,080 calls audited, 56.319 charged hours; no remaining queue. Zero observed difference is not population equivalence. |
| Historical internal gain | [Source audit](main_table_source_applicability_20260921.md), [input addendum](main_table_input_composition_20260921.md) | All 23 old gains lack full input-containment provenance. Preserve numeric results without calling them source-complete certificates. |
| Earlier independent tool challenges | [MetaMoE follow-up](metamoe_las_followup_result_20260923_r1.md), detailed evaluation | Author-compatible 9 versus ACT 4; earlier plain CROWN 13 versus ACT 11. Distinct semantics/cohorts/grades, not the current full GenBaB comparison. |
| Independent proof frontier | [Stopped source/output study](proof_closure_decision_20260925_r1.md), source-proof appendix | Old positive supplied-HZ objects differ from newer checked source matrices. No proof splicing; no full native-floating execution claim. |

## Human review report (leave unanswered until performed)

Reviewer/date, expertise and relationship to implementation: **pending**.
Access (paper/code/private raw artifacts): **pending**.
Actually executed commands, environments and exit codes: **pending**.

1. Restate the narrowest defensible contribution. Compare it with existing
   program/verification reductions, MetaMoE, GenBaB and output-constraint work
   including INVPROP. Which claimed part is new, useful, or merely standard?
2. Attempt to falsify the three propositions using a legal tie, nonzero
   property constant, an empty history, a missing history, a mismatched source,
   an invalid gate range, or a nonlinear suffix. Identify failed hypotheses.
3. Trace one primary gain and one loss through source identity, complete
   obligations and cost. Report unavailable raw evidence; do not substitute
   hash checking for numerical soundness.
4. Check the revised guard optimizer's evaluated alpha restoration, retained
   per-property prefix maxima and paid extra-alpha comparator. Does the
   one-history gain survive the stated numerical contract? The analytic ReLU
   guard dual gap is a useful counterexample to an unjustified positive oracle.
5. Check pairing, input-level uncertainty, failure preservation, the practical
   STOP, the post-189 supplemental rule and shared-host timing limitations.
   Do not infer unique ReLU hardness from timeouts or frozen-gate failures.
6. Separate factual defects, claim limits and optional future work. Assess
   venue fit independently; an honest negative review is a valid outcome.
7. For S1, verify the two capacity conventions, seed/input pairing and the
   distinction between complete primary and incomplete secondary endpoints.
   Check why the positive incomplete-bound difference and zero-positive 8/255
   timing do not change the frozen stop. Is the current architecture/loss/model
   scope sufficient for the stated limited claim? Do not infer certification-
   training performance from the ordinary-training S1 comparison.
8. For S2a, check tie-inclusive candidate retention, property-first endpoint
   bounds and shared training/evaluation code. Verify validation-only selection
   of 20 models from 24 runs, the crossed paired seed/input bootstrap, and all
   three failed gate components. Do the routing and width diagnostics support
   only the stated descriptive conclusions? Keep final secondary findings
   separate from the primary decision, and retain the original execution
   error and all charged costs when reviewing the completed queue.

AI review must identify itself. No packaging script may fill human sign-off,
independence or a novelty verdict. This template sends no invitation.

## MoE literature and the archived execution boundary

The historical verifier draft's bibliography cites all six requested MoE works directly. The
2026-10-29 check concerns primary-paper metadata and the stated research task;
it neither reruns their experiments nor refreshes every author repository.
[The six-work ledger](author_baselines_status_20260924.md) and
[object-level contracts](baseline_semantic_admission_20261013.md) retain their
dates, frozen versions, recipe differences and incomplete comparisons.

| Work and primary reference | What was actually executed here; remaining comparison boundary |
|---|---|
| [Dual RS, revision 4](https://arxiv.org/abs/2512.01782v4) | [Two-input statistical pilot](dual_rs_certification_archive_20260922_r1.json) after selector training; no same-smoothed-function ACT comparison or full certified-accuracy curve. The older paper/implementation versions bound to that pilot are unchanged. |
| [MetaMoE](https://doi.org/10.1007/978-3-032-32357-6_8) | [Ten-input author-compatible comparison](metamoe_las_followup_result_20260923_r1.md), with disclosed repairs: author 9 versus ACT 4 numerical/policy positives; not a complete author-table reproduction or strict-certificate contest. |
| [RoME](https://arxiv.org/abs/2607.06109v1) | [Four-input, three-norm batch](rome_multinorm_archive_20260922_r1.json): 5 completed and 7 timeouts, not a robustness rate. Continuous all-expert token mixtures are not the discrete top-2 object. |
| [J-TLAT, official ICLR paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f206871468dc89cb20fefbc75f2de861-Abstract-Conference.html) | No complete author execution in the archive. The September README-only observation is dated; this check does not infer current artifact absence. |
| [Feature Noise, revision 2](https://arxiv.org/abs/2601.14792v2) | Theoretical/experimental positioning; no complete author reproduction or same-object ACT comparison. A noisy-feature risk question is not the current pixel-box verification endpoint. |
| [Robust Experts, official CVF entry](https://openaccess.thecvf.com/content/ICCV2025W/STREAM/html/Pavlitska_Robust_Experts_The_Effect_of_Adversarial_Training_on_CNNs_with_ICCVW_2025_paper.html) | [Two training/evaluation variants](robust_experts_landing_20260922_r1/README.md), with recipe differences preserved; no complete ACT comparison on the trained intermediate top-2 object. |

The live arXiv pages identify Dual RS revision 4 and Feature Noise revision 2;
the historical detailed survey read revisions 3 and 1. The short paper uses the
newer metadata only for the broad task distinction and does not move old
numerical claims to new versions. Direct CVF retrieval returned HTTP 403; the
official indexed entry supplied the Robust Experts title, authors, workshop
and pages. No author contact or external deployment occurred. These citations
make the comparison scope inspectable; they do not complete G6 or establish
priority for the reductions.

## Current bounded table reproduction

The historical-table renderer reads committed JSON summaries plus the B4 freeze,
checks model-to-checkpoint identity and accounting, and reconstructs seven tables. It needs only the Python standard library:

```sh
python -I -S -B scripts/render_current_review_tables.py --check
```

It does not read checkpoint/data/server raw paths, execute a model, solve an
obligation or independently reprove a lower bound. Run without `--check` to
print the LaTeX. Relevant summary-view hashes are embedded in
[the table file](../paper/results/current_review_tables.tex); source identity
is distinct from mathematical evidence. The older draft's PDF build and targeted
checks are recorded in [the build note](current_review_paper_20261029.md).
The combined manuscript includes these tables unchanged. Its S1 seed table is
transcribed from all twelve models in the final S1 archive; the build checks
both source groups. The [S1 record](architecture_s1_records_20261029.json) binds
the combined source, included tables/appendix, PDF and checks. No new model
queries are used to merge the papers.

Compile from `paper/` with an already installed LaTeX distribution and a new
empty output directory:

```sh
pdflatex -no-shell-escape -halt-on-error -interaction=nonstopmode -output-directory /NEW/BUILD architecture_main.tex
pdflatex -no-shell-escape -halt-on-error -interaction=nonstopmode -output-directory /NEW/BUILD architecture_main.tex
```

The historical September review kits, manifests, table renderer and integrity
checker stay unchanged. Their inventories bind their own historical contents,
not this new paper. A separately trusted kit-manifest hash is required for
historical kit verification. `python -I -S` gives dependency isolation, not a
fresh OS or clean empirical model execution. No installer is run here.

## Work still requiring external review or a publication decision

| Task | Current boundary |
|---|---|
| Independent human technical review | Pending; PI selects and coordinates reviewer |
| Clean-environment empirical execution | Pending; fixed request/model/input/version/budget needed, not a table rebuild |
| Actual source-complete positive proof | Open scientific gap, not a packaging task |
| Distribution of weights/data/third-party code | Rights and anonymity decisions pending; private raw paths not automatically public |
| Submission, authorship and AI disclosure | Human authors must confirm claims and material AI assistance; no submission or contact authorized here |

A clean empirical run must retain errors and all cost; it cannot repair a past
online failure. Sealed cases, new installations and dissemination continue to
follow their existing authorization boundaries.

## Official dates and references checked 2026-10-29

- ISSTA 2027 lists mandatory abstract **2027-01-08** and paper **2027-01-11**.
  Use the official track rules for final formatting/timezone requirements;
  this note does not infer them from a date alone.
  [Official dates](https://conf.researchr.org/dates/issta-2027).
- OOPSLA 2027 round 1 (2026-10-14) has passed; round 2 is **2027-04-07, AoE**.
  [Official track](https://2027.splashcon.org/track/splashoopsla2027).

These dates are planning inputs, not a promise that the current evidence meets
either venue's novelty or evaluation expectations. GenBaB's
[author paper](https://arxiv.org/abs/2405.21063) identifies TACAS 2025;
[INVPROP](https://arxiv.org/abs/2302.01404) provides relevant prior work on
output-constrained bound propagation. HZ, DNNV, CROWN and MetaMoE metadata were
also checked against their primary pages linked in the short bibliography.
This targeted check is not an exhaustive priority search. Earlier reference
checks remain in [the historical note](submission_references_20260921.md).
