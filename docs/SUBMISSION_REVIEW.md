# Independent pre-submission review: current manuscript and bounded reproduction

This is a review invitation/template for PI-managed use, not a completed review
or authorization to contact anyone. The short manuscript is
[review_main.tex](../paper/review_main.tex); the longer
[method](../paper/sections/03_path_conditioned_method.md),
[source contract](../paper/sections/05_soundness_engineering.md) and
[evaluation](../paper/sections/08_evaluation.md) remain supporting material.
No venue has been selected by this packaging work. The PDF uses a generic
article layout, not an official ISSTA/ASE/FSE/ICSE submission template.

## Read the argument before the receipts

The question is whether the relational obligation method is a sufficiently
important and supported contribution, NOT whether enough tests passed.
Start with the short paper, write your own account of its contribution, then
challenge the evidence using the following map. A disagreeing or negative
review is a valid outcome; no proposed acceptance score is supplied.

| Claim | Evidence location | Required qualification / challenge |
|---|---|---|
| Mathematical complete-output rule | Short paper Proposition 1; method's complete-request composition theorem | Are coverage, tie inclusion, same-source enclosures and valid row bounds sufficient? They are hypotheses, not inferred from historical solver success. |
| Scoped reuse | Method and `scoped_f0_proofs.py` | The same property must hold on both membership domains; pair inclusion, factor identities and constant terms must match. Supplied interval validity remains an assumption. |
| New-input internal gain | Committed confirmation review; generated tables | 23 gained / 0 lost policy SAFE, 100 input clusters and three fixed models; not independently proved real-box coverage or 300 independent images. |
| Relation precision | Committed relationship ablation | Two completed-relaxation failures support a precision mechanism; a third solver-limit case has a cost confound. Not casewise attribution of all 23 gains. |
| External value | Committed external review | ACT 11 policy positives vs CROWN 13 numerical filters, 8 shared / 3 ACT-only / 5 CROWN-only; means 138.11 / 4.23 s. Explain complementarity without hiding the cost disadvantage or equating grades. |
| MetaMoE author-path competition | Repaired full20 archive and result (2026-09-23) | ACT 4 policy positives vs author 9 numerical filters, 4 shared / 0 ACT-only / 5 author-only. All ACT positives route-stable. Repaired request times out; this is compatibility, not a new certificate. Common-positive means 10.55 / 7.17 s. Adapter/repair disclosed; do not pool with weighted top-2 experiments or call it unchanged author execution. |
| Applicability of historical main table | Input source ledger and composed-input addendum | Requested/formula sets are not nested on all 100 inputs. No network counterexample follows, but no source-complete real-box certificate follows either. |
| Independent proof | Old portable-proof review; new full-source/property-range reviews | Old same-HZ conditional 9/9 positive is not a positive result on newer source-checked matrices. No cross-source splicing. |
| Generality | Conv V2 and general evidence reviews | Conv 30-input HZ positives: zero. New 20-input evidence positives: zero. High-accuracy/cross-family strict gains remain open. |

## Minimum human review report (leave unanswered until actually reviewed)

Reviewer and date: **pending**. Relationship to implementation: **pending**.
Access level (short paper / repository / private raw objects): **pending**.
Commands actually executed, exit codes, environment: **pending**.

1. Restate the target, three contributions, and the narrowest defensible claim.
2. Try to falsify Proposition 1 with a legal tie, nonzero property constant,
   partial fact, mismatched source, or missing route. Identify the exact failed
   hypothesis or implementation issue; do not treat a control as a full proof.
3. Trace one claimed gain from the statement through archived accounting and
   numerical contract. Report inaccessible raw material explicitly.
4. State whether the input-containment limitation makes the main empirical
   contribution too weak for the intended claim; disagreement with the authors'
   current conditional interpretation is welcome.
5. Assess the external path's practical challenge and whether related work
   distinguishes within-request facts from templates/conflicts and shared HZ
   relations from generic path enumeration.
6. Separate must-fix defects, honestly reportable limitations, and optional
   future research. Give a manuscript recommendation, not an A-tier guarantee.

An AI reviewer must label itself as such. A same-author reimplementation is not
third-party review. A human report, sign-off or novelty verdict must never be
filled in automatically by the packaging scripts.

## The local review kit

The kit contains the short paper, its audited table renderer, selected committed
reviews, source-contract records, and this review form. It intentionally does
not contain weights, raw input tensors, the real portable proof, a solver, or
an empirical re-execution. The checker verifies file identity, reconstructs
archived accounting and reproduces the manuscript tables. It neither executes
the neural model nor independently proves its safety bounds.
Links to longer sections, implementation files and ancillary histories refer
to the full repository and are not all included in this bounded kit. Its
manifest is the exact inventory, not a claim that every linked object has
been distributed. The short paper and three generated tables are self-contained.
The new MetaMoE table is reconstructed from all 20 archived rows, not a
partial-run summary. See [current claim disposition](competition_guarantee_disposition_20260923.md).

From an existing suitable Python installation, after receiving a complete kit
and its manifest hash **through a separately trusted channel**:

```sh
python -I -S /path/to/kit/scripts/check_submission_review_kit.py --manifest-sha256 HASH
```

The checker rejects missing/extra files, symlinks, unsafe paths, invalid grades,
changed bytes and stale table rendering. Expected hashes are not authority
when supplied only by the untrusted package itself. This is integrity and
accounting checking, not an adversarial execution sandbox or a mathematical
certificate checker. `-I -S` establishes Python dependency isolation, not a
fresh OS/container installation. The manifest also binds the checker itself.

The source can be compiled with an already installed LaTeX distribution:

```sh
cd /path/to/kit/paper
pdflatex -no-shell-escape -halt-on-error -interaction=nonstopmode -output-directory /NEW/BUILD review_main.tex
pdflatex -no-shell-escape -halt-on-error -interaction=nonstopmode -output-directory /NEW/BUILD review_main.tex
```

Use a separate build directory; adding LaTeX output to the immutable kit makes
its exact-file inventory check fail. No installer or network download is run
by this workflow. The supplied Python checker itself needs no LaTeX.

## Still required for the empirical artifact

Do not merge these incomplete tasks into the table-only success above:

| Operation | Required input | Current state / owner |
|---|---|---|
| Independent human method review | Paper, code, contracts, reviewer access | Not completed; PI chooses and coordinates reviewer |
| Actual trained-model request in a clean environment | Exact checkpoint, input tensor, configuration, execution version and dependency environment | Not performed here; separately scoped rerun, no performance claim from a smoke |
| Check a real stored conditional proof | Entire old 13-file bundle plus independently retained statement/bundle hashes | Local relocation already passed; transfer/rights decision pending |
| Reproduce saved source audit | Saved tensor packages and pinned decoding environment | Local audit complete; raw access is not included in this kit |
| Public/anonymous distribution | Rights review for data, weights, third-party files; private-path/identity redaction without altering originals | No publication or anonymization claim; PI decision required |
| Same-source complete positive proof | Checked source plus every positive output obligation on those exact matrices | Research goal open; not a packaging task |

For an eventual clean empirical run, prerecord the exact request and hashes,
environment, CPU/thread settings, immutable output directory, total budget,
expected outcome's evidence grade and failure handling. Execute once without
outcome-driven retries, retain all terminals, and distinguish execution
agreement from a proof of historical outcomes. Installing dependencies,
redistributing model/data objects, or reopening a sealed experiment requires
the relevant explicit authorization. This document launches none of them.

## AI assistance disclosure and reference check

The short paper includes a factual disclosure draft: AI assisted code, tests,
orchestration, analyses and writing, not just grammar. No final human sign-off
or exhaustive tool/version list is invented. Human authors must confirm the
inventory and all claims before submission. Current source checks for the
condensed bibliography are recorded in
[the reference note](submission_references_20260921.md); they are targeted
metadata/claim checks, not an exhaustive literature review or priority proof.
