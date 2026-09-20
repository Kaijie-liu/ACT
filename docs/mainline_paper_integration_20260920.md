# Return to complete MoE proofs and manuscript integration

Starting branch `feat/moe-route-verification`, clean and synchronized at
`fb092e232b4125b62beed3c2ab1987dd1fe8c09a`. This stage follows Advice/dd.md.
No production algorithm, frozen protocol, checker acceptance rule, cohort,
checkpoint or result was changed. No real LP proposal, training, new empirical
verification request or custom elimination search was launched.

## Deliverables

1. Section03 now states and proves the complete route/property composition
   proposition, including tie coverage, scoped fact reuse, residual enclosures
   and explicit upstream assumptions. It organizes existing mathematics; it
   does not claim a new generic convexity theorem or machine-checked semantics.
2. Section05 separates checked construction/bounds/aggregation from trusted
   network-to-HZ, guard and route-exclusion steps. Section08 foregrounds the
   complete first-family results, external complementarity and convolutional
   negative results. Abstract, introduction, discussion and reading guide agree.
3. Previous checking and transfer chronology is preserved verbatim in separate
   appendices. The four-LP SoPlex limit has a bounded appendix, not a new main
   contribution or an instruction to resume arithmetic research.
4. `scripts/rebuild_moe_main_tables.py` rebuilds21 rows from four committed
   reviews using only the standard library. It checks denominators, grades,
   paired deltas, row inventories, cohort matching and finite costs, and prints
   source hashes. `--check` detects a stale committed rendering. It never reads
   raw paths embedded in the reviews. Confirmation uses archived aggregates;
   external and Conv V2 counts are recomputed from committed rows. This is
   result reconstruction, **not** independent reproof of empirical SAFE.
5. `paper/artifact_quickstart.md` now separates the fresh analytic request,
   portable real proof and model-free table workflow. Real bundles/weights are
   still not publicly distributed by this checkout; no artifact was published.

## Exercised complete-proof workflows

Raw local work is retained under
`/data1/Kane/MOE/mainline_integration_20260920_33imzg` (not committed).

**Real stored proof:** copy the entire original input98 relocated bundle into
the new `proof/` directory. Recursive comparison reports no file differences.
Run its verifier with `python -I -S`, the independent bundle hash
`8f35a4ba23b51bdcc829535a47880e5f6158a6fbaaf119b4f7744e0c2278606b`
and statement hash
`7c31f551137b33257e178c40eeea55bf4b94e3438ae00ddb3ed16e2808e56b00`.
Result:9/9 necessary obligations positive (eight rational residual, one reuse),
minimum `199593373867685/1125899906842624`, same as the original review.
Checker elapsed32.975684s; no model/solver imports, site packages or outside
bundle reads. This new check timing is not a generation cost or paired speedup.
Status remains conditional on upstream lowering; floating F0 construction is
not trusted. It is a stored single-pair proof, not a new route-changing result.

**Fresh source-defined demo:** the unchanged `run_moe_proof_demo.py`, bounded
externally at120s, creates the analytic all-tie three-expert model and executes
the standard verifier plus rational generator. SAFE and structural audit PASS
with0issues; independent request check covers3/3 obligations (one reused, two
residual), minimum `14411518807585587/36028797018963968`. Generation including
staged work1.859395s; independent process check0.027240s. These are analytic
control costs, not trained-model effectiveness evidence. Summary SHA256:
`c2f6be163b825dd2c1be48bdca40f977abd0b8afc99f72f279d338ac632e372c`.
The solver-license warning is nonfatal; no Gurobi license or dependency change
is needed by this exercised workflow.

## Tests

- Three main-table tests pass, including nine mutation subcases: missing and
  duplicate rows, mismatched evidence grade, denominator, paired delta,
  nonfinite cost, failed source review, terminal disagreement, changed cohort.
- Five unchanged portable-proof regression tests pass (pure extraction,
  ambiguous JSON, array references and corruption).
- `python -I -S scripts/rebuild_moe_main_tables.py --check` passes.
- Fresh analytic complete proof and relocated real proof pass as above.
- `git diff --check` passes. Historical source/result files are unchanged.

## Research disposition

The paper can now present complete-output capability, scoped empirical gains
and the exact trusted boundary without waiting for a feasible LP upper witness.
The first-family23 added HZ-policy SAFE are not23 independently checked rational
network proofs. Input98 is not a cross-family route-changing gain. No claim of
high-accuracy real-scale strict certification or universal external superiority
is added. The four-LP admission-limited study stays closed.

Further research must name a full request's blocking obligations and explain
why a proposed representation/evidence change could discharge them. A new
empirical protocol is not authorized by a paper edit. The remaining artifact
gap is empirical model/input/bundle distribution and clean-environment testing;
the remaining scientific gap is transferable complete positive certificates
on the convolutional/high-accuracy setting. Neither requires another automatic
arithmetic-to-supervisor loop.
