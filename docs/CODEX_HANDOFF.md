# MoE project handoff — current entry point

H1 synthetic-control update: 2026-09-30. Real-model scientific status remains
the audited 2026-09-25 decision. No new real experiment has been launched.

## Start here

- [Project navigation / 工程总入口](PROJECT_INDEX.md): code, protocols, results,
  manuscript, author baselines, data and environments.
- [Algorithm improvement goal / 算法改进目标](ALGORITHM_RESEARCH_GOAL.md): the
  user's new research objective, baseline contracts and bounded next mechanism.
- [Complete directory catalogue](organization/DIRECTORY_CATALOG.md): all tracked
  top-level directory families and the local workspace's retention categories.
- [Working agreements](../AGENTS.md): use only `feat/moe-route-verification`;
  stop on unexplained dirty worktrees; no concurrent writers.
- Existing ACT Python: `/data1/Kane/miniconda3/envs/act-py312/bin/python`.
  Do not install or upgrade dependencies as part of housekeeping.

## Current decision: H1 synthetic proofs work; hard-budget integration is next

On 2026-09-28 the user explicitly requested an algorithm-improvement goal aimed
at external competitiveness and a software-engineering A-venue submission.
The [goal contract](ALGORITHM_RESEARCH_GOAL.md) separates same-source proof,
algorithmic benefit, external comparison and independent review. Acceptance at
a venue is not a promised outcome. None of the six related works is to receive
a fabricated or semantically mismatched ACT win/loss entry.

[H1 synthetic implementation and controls](h1_sparse_source_controls_20260930.md)
now have complete source/output positive proofs on four synthetic controls;
the negative control remains UNKNOWN. All five full/dependency comparisons
agree on checked bounds. A relation-dependent control closes six obligations
where interval facts alone do not suffice. This is not a real-model gain or an
old binary-HZ differential: both arms use the new direct-variable ReLU LP.
The dense control still needs every input/hidden node in the dependency union.

Next bounded task: integrate this optional synthetic path with a complete
300-second hard-budget supervisor and relocatable checker, accounting for
startup/source loading, construction, proposals, serialization and checking.
Currently only cooperative deadlines and moved JSON checking are controlled;
there is no self-contained relocated checker or competitive timing result.
No real-request freeze is authorized by this stage. All seals below remain.

## Audited stop decision retained

Read [proof-closure decision](proof_closure_decision_20260925_r1.md),
[derived ledger](proof_closure_20260925_r1.json) and
[validation](proof_closure_validation_20260925_r1.json) before making claims.

Seven archived real engineering calls on four old inputs ended in six TIMEOUTs
and one RESOURCE_LIMIT. They generated **zero output LP calls, zero output
bounds and zero complete source-to-output positive certificates**.
Only 4099/shared accepted 225/252 route-excluded duties online; 27 remain,
with no published retained construction. Offline exclusions are not online
receipts. Retained pairs are potential routes, not route-change witnesses.

The [four-call readonly upstream study](readonly_upstream_result_20260925_r1.md)
checked synthetic constructions, not output certificates. It showed one whole-
cost win and one loss. Keep the optional optimization default-OFF; stop further
cache/timing studies as the automatic next task.

**Inputs 98/4088/4096/4098/4099 remain sealed.** No new sample, larger budget,
changed numerical gate, revived holdout or real-request freeze is authorized by
this handoff. New proof research first needs a concrete complete-obligation
hypothesis, controls and a separately scoped decision. Do not restart serializer,
parser or exact-elimination experiments merely because their code remains here.

## Results and guarantee boundaries

- [Competition/guarantee disposition](competition_guarantee_disposition_20260923.md)
  and [main-table source audit](main_table_source_applicability_20260921.md):
  internal net gains are frozen numerical-policy outcomes. All 23 primary gains
  have audited input-containment gaps; they are not source-complete strict
  certificates for the requested domain. This does not establish model unsafety.
- [Six author-baseline status report](author_baselines_status_20260924.md):
  distinguishes deployment, controls, training and matched comparisons. Its
  dated experimental counts remain useful; its old "next gate" is superseded
  by the 2026-09-25 proof-closure decision above.
- MetaMoE repaired author comparison: ACT 4/10 policy acceptances versus author
  9/10 numerical filters, no ACT-only positive. The static weighted CROWN path
  also has an adverse aggregate coverage/cost comparison. Do not claim stable
  superiority over external tools.
- A positive proof for a stored HZ and successful checking of a new source
  cannot be spliced into a complete proof: new matrices require new output
  evidence. Full floating-point execution guarantees remain distinct.

## Current useful work, not an automatic experiment queue

1. Use the [manuscript reading order](../paper/README.md) and
   [short review manuscript](../paper/review_main.tex); retain adverse results
   and scoped guarantees in the main narrative.
2. Prepare independent review using [SUBMISSION_REVIEW](SUBMISSION_REVIEW.md).
   Human technical review, PI-managed access/licensing and clean empirical
   reproduction remain OPEN. Do not contact reviewers/authors automatically.
3. The relocated accounting kit and compiled generic PDF are useful review
   infrastructure, not a clean empirical installation, independent network
   reproof, anonymous release or submission-readiness guarantee.
4. The latest [PI advice](../../Advice/ee.md) provides direction; later audited
   source-gap findings and stop decisions supersede older positive/next-step
   prose. Lack of a concrete new research hypothesis is not a reason to rerun
   sealed inputs.

## Safe, read-only accounting checks

Run from the ACT repository with the existing environment:

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S scripts/check_project_layout.py --workspace
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S scripts/summarize_proof_closure.py --check
```

These check navigation, recorded source identities and archived accounting.
They do not train, launch a solver, certify a network or prove manuscript claims.

## History preserved, not an active task list

The former 4,632-line handoff is preserved **byte-for-byte** in
[CODEX_HANDOFF_HISTORY_20260928.md](CODEX_HANDOFF_HISTORY_20260928.md),
SHA-256 `9681affe28c4b400c444a77df74255bd37c58ffc3801a10e397ba4bffe9a005a`.
It remains in the same directory so its original relative references keep
their context. Historical "NEXT", "NOT RUN" and pending entries do not override
the current decision above. Consult individual frozen protocols for provenance,
not permission to rerun.
