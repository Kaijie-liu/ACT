# Manuscript reading order

The current entry is [the technical-review paper](review_main.tex), revised
2026-10-29: **Verifying Route-Variable Mixture-of-Experts Programs: Obligation
Reductions and Empirical Limits**. Its three propositions cover complete
history obligations, output-atom guards, and last-gate endpoints. Five tables
are generated from committed completed-result summaries. This is a generic
review article, not a venue-formatted submission or a completed human review.

The scientific disposition is unchanged by writing:

- The primary M-A comparison gives 122/300 versus 116/300 numerical positives,
  nine gains and three losses, net 2 percentage points, input-block interval
  [0,4]. The registered five-point gate fails.
- The revised M-B comparison preserves the alpha baseline, adds an equal-extra-
  alpha control, and closes one of eleven histories. Its five-closure gate
  fails. M-B and the conditional last-margin M-C test close; no new B4/B5.
- The original first-margin M-C test has zero extra closures. It does not test
  every possible last-margin or dual-conditioned implementation.
- The larger-radius queue is complete: 8/120 versus 6/120 positives, two gains,
  no losses, interval [0,4.167] percentage points. No gain appears among thirteen
  objects with multiple observed histories. The frozen single-layer queue
  continues; neither secondary cohort changes the primary decision.
- Historical 23-gain input-containment gaps, source-complete real proofs,
  external superiority and actual GPU benefit remain unresolved. A local
  history positive is not a complete MoE certificate.

Read [the handoff](../docs/CODEX_HANDOFF.md) for current execution state and
[the review guide](../docs/SUBMISSION_REVIEW.md) for claim/evidence questions.
The manuscript now centers on reductions and empirical limitations. It does
not claim that all remaining hardness has been isolated to expert ReLUs.

## Detailed companion

1. [Abstract](abstract.md)
2. [Introduction](sections/01_introduction.md)
3. [Related work and author-artifact boundaries](sections/02_related_work.md)
4. [Method and exact assumptions](sections/03_path_conditioned_method.md)
5. [Soundness engineering](sections/05_soundness_engineering.md)
6. [Evaluation](sections/08_evaluation.md)
7. [Discussion and threats](sections/09_discussion_threats.md)

The longer sections retain additional historical results and derivations.
Current statements are governed by later audited evidence, not old roadmap
entries. Negative and stopped experiments stay available:

- [Regime analysis](sections/04_design_space_regimes.md)
- [Artifact identity](sections/06_certified_artifact_identity.md)
- [RT-ER case study](sections/07_rt_er_artifact_case.md)
- [Historical evaluation](appendices/historical_evaluation.md)
- [Source-proof development and stopped input98 follow-up](appendices/source_proof_history.md)
- [Transfer and evidence chronology](appendices/transfer_evidence_history.md)

## Reconstruct current tables and build the paper

From the repository root, using the existing ACT environment:

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B scripts/render_current_review_tables.py --check
```

This verifies [current_review_tables.tex](results/current_review_tables.tex)
against the committed main/ablation, multilayer, revised-guard and radius summaries.
It does not run models, retrieve raw objects, replay attacks or prove CROWN
arithmetic. Relevant summary-view hashes appear in the generated table file.
Run the renderer without `--check` to print the table source.

With an existing LaTeX installation and a new empty build directory, from
`paper/`:

```sh
pdflatex -no-shell-escape -halt-on-error -interaction=nonstopmode -output-directory /NEW/BUILD review_main.tex
pdflatex -no-shell-escape -halt-on-error -interaction=nonstopmode -output-directory /NEW/BUILD review_main.tex
```

No dependencies are installed by these commands. Keep outputs outside the
checkout and immutable historical kits. The October source supersedes the old
September short manuscript for reading; Git and the old kits preserve that
manuscript and its tables. `render_submission_tables.py`, historical
`submission_tables.tex`, old kit manifests and their checker are unchanged;
they do **not** certify the contents of this new paper.

## Proof and review boundaries

The [source audit](../docs/main_table_source_applicability_20260921.md) and
[composed-input addendum](../docs/main_table_input_composition_20260921.md)
find requested and reconstructed input sets non-nested on all 100 historical
confirmation inputs. This affects all 23 old gains. Those frozen-policy
outcomes are preserved; they are not source-complete real-box certificates.
An audit defect does not itself prove the neural model unsafe.

The old portable input98 positive is conditional on its stored HZ. The later
source-checked construction has no complete positive output on its new matrices.
Combining their successful pieces would be invalid; the sealed case stays
closed. [Historical proof workflow](artifact_quickstart.md) and
[historical main tables](results/main_tables.md) remain separate artifacts.

Independent human technical review, a clean-environment empirical reproduction,
and a distributable/anonymous artifact are unfinished. The PI coordinates
review and publication decisions. AI self-checks, source hashes, table
reconstruction and a successful PDF build do not replace those activities or
establish novelty, full numerical soundness, or acceptance readiness.
