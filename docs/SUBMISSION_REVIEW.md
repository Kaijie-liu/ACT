# Independent technical review: reductions and empirical limits

Updated 2026-10-29. This is a PI-managed review template, not a completed human
review or permission to contact anyone. Start with the current
[short paper](../paper/review_main.tex), then consult the longer
[method](../paper/sections/03_path_conditioned_method.md),
[evaluation](../paper/sections/08_evaluation.md) and
[source contract](../paper/sections/05_soundness_engineering.md).
The generic article layout is not an official submission template.

## Read the argument before the receipts

The question is whether the problem, reductions and measured limitations support
a useful research contribution. Counts of controls or documents do not answer
it. No positive novelty verdict or review score is supplied.

| Claim / question | Evidence | Essential qualification |
|---|---|---|
| Complete-output composition | Short paper Proposition 1; detailed method | Every tie-legal history and property, same input/source, valid exclusions and bounds. An incomplete tree cannot be positive. |
| Output-atom guard reduction | Proposition 2; guarded frontend and semantic controls | Pointwise `max(p,-g)>0` equivalence, including ties; not equivalent finite-budget relaxation precision or a claim that generic tools can never exploit guards. |
| Independent-gate embedding and last endpoint | Proposition 3; M-A contract | Valid prefix ranges; last affine suffix and domain independent of last gate. Inner gates remain nonlinear. Numerical sigmoid intervals are not automatically exact enclosures. |
| Strong engine versus old ACT v1 | [Stage-1 records](multilayer_stage1_records_20261022.json), `final_result_20261025` | 108/180 pruned positives versus zero; ACT has 178 TIMEOUT, one ERROR and one UNSUPPORTED. Supplemental bottleneck gate was registered after 189 calls. No unique ReLU-cause claim. |
| Primary cascade gain and stop | [B4 records](moe_aware_b4_records_20261027.json), `main_and_ablation_review` | 122 versus 116, nine gains/three losses, +2pp/[0,4]pp; five-point threshold fails. Twenty input blocks preserve cross-model dependence. |
| Mechanism and cost | Same records, `main_gain_loss_trace_review` | All nine gains use V2; 48/52 endpoints close initially. Shared-positive overhead exceeds gain savings. Partial logs cannot prove a counterfactual budget choice. |
| Execution variability | Same records, ablation/repeated coupled rows | Four terminal changes among 120, including one positive to timeout; not four SAFE changes or a 3pp SAFE-noise estimate. |
| Revised guard contribution | [Result](guard_dual_refinement_result_20261029.json), [interpretation](guard_dual_refinement_20261029.md) | Full baseline retention and paid equal-extra-alpha control. One of eleven histories closes, not five; M-B/M-C stop. Local closure is not a full request. |
| Larger radius and single-layer scope | B4 frozen queues; handoff for state | Still pending. They cannot replace primary denominators or stopping gates. Public PGD is not free arm credit; sampled histories are not exact legal counts. |
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

AI review must identify itself. No packaging script may fill human sign-off,
independence or a novelty verdict. This template sends no invitation.

## Current bounded table reproduction

The current renderer reads three committed JSON summaries, performs accounting
checks, and reconstructs four tables. It needs only the Python standard library:

```sh
python -I -S -B scripts/render_current_review_tables.py --check
```

It does not read checkpoint/data/server raw paths, execute a model, solve an
obligation or independently reprove a lower bound. Run without `--check` to
print the LaTeX. Relevant summary-view hashes are embedded in
[the table file](../paper/results/current_review_tables.tex); source identity
is distinct from mathematical evidence. The current PDF build and targeted
checks are recorded in [the build note](current_review_paper_20261029.md).

Compile from `paper/` with an already installed LaTeX distribution and a new
empty output directory:

```sh
pdflatex -no-shell-escape -halt-on-error -interaction=nonstopmode -output-directory /NEW/BUILD review_main.tex
pdflatex -no-shell-escape -halt-on-error -interaction=nonstopmode -output-directory /NEW/BUILD review_main.tex
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
