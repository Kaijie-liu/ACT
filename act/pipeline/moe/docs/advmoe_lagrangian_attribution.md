# AdvMoE Lagrangian null-endpoint attribution

## Question and fixed scope

Development r1 improved 241 of 1,800 property bounds but added no complete
sample-radius filter.  This diagnostic asks which of three explanations is
responsible: truncation of the multiplier grid, the CROWN relaxation, or the
fixed-multiplier sufficient reduction itself.  It reuses only the accepted
20-input development cohort.  It is not a holdout, coverage estimate, or
formal-certificate experiment, and it never rewrites the parent artifacts.

The first stage deterministically selects the five unresolved rows without a
full-model prediction-flip witness whose compiled endpoint is closest to zero:
`sample6:eps0.5`, `sample16:eps0.5`, `sample11:eps0.5`,
`sample3:eps0.5`, and `sample10:eps0.5`.  This rule and its exact outcome are
hash-bound in the configuration.

## Stage A: multiplier-grid truncation

The parent normalized coefficients `{0, 0.25, 0.5, 1, 2, 4}` remain
immutable.  Only parent-negative branches receive the preregistered extension
`{8, 16, 32}`, using the same frozen router-margin scale and the same plain
CROWN call configuration.  Parent and extension calls are then aggregated
row by row.  A nonnegative multiplier remains fixed over the entire input box.

This is an attribution run, not a cost-matched method comparison.  Added wall
time is reported, but the expanded grid is not compared to the parent under
the old 60-second method cutoff.

Interpretation is frozen as follows:

- any new complete endpoint establishes finite-grid truncation for at least
  that selected row;
- property-level improvement without a complete endpoint shows that the old
  grid truncated some bounds but does not explain the null endpoint;
- no improvement makes finite-grid truncation an implausible primary cause on
  these closest residuals.

The latter two outcomes trigger a separately frozen fixed-multiplier-family
diagnostic.  Stage A alone must not assign the remaining gap to CROWN or to the
sufficient reduction.

## Evidence and safety semantics

An independent auditor reconstructs the subset from the immutable parent
rows, re-aggregates every property bound, checks all multiplier identities and
rejects any regression.  Negative relaxed bounds are always `UNKNOWN`, never
`UNSAFE`.  Positive CROWN filters are not called formal `SAFE`, because this
backend is not outward rounded.  No new attack or witness is generated in
Stage A; parent rows with a replayed prediction flip are excluded.

Raw results are written to a new directory under
`data/moe/results/advmoe_lagrangian_attribution_grid_r1` and never overwrite
development r1.

## Stage A result (2026-09-06)

The run completed in 163.55 seconds and the independent audit passed with zero
issues.  Extending the normalized coefficients to `{8,16,32}` strictly
improved 18 property bounds but produced zero complete endpoint gains.  All 18
improvements occur on one branch of `sample3:eps0.5` and one branch of
`sample10:eps0.5`; neither is sufficient to close its companion blocking
branch.  Across all five selected rows, the final row-blocking branch still
selects `mu=0` for its worst property.

The registered classification is therefore
`FINITE_GRID_CONTRIBUTES_WITHOUT_ENDPOINT_GAIN`.  The old grid did truncate
some nonblocking bounds, but finite-grid truncation is not an endpoint-level
explanation on the closest residuals.  Stage B is required to distinguish
backend relaxation from a fixed-multiplier certificate-family limitation.

## Stage B preregistration

Stage B freezes one necessary property per selected row: the most negative
property that still selects `mu=0` after Stage A.  For each obligation it runs
20-iteration alpha-CROWN on the graph-matched `mu=0` compiler, performs a
50-step, three-restart minimum-phi search at all nine combined multipliers, and
replays every concrete point through the literal router, static path, and full
dynamic model.

The concrete pairs `(s_t,m_t)` also define the exactly solvable finite problem

`sup_{mu>=0} min_t (s_t - mu m_t)`.

Because the finite points are a subset of the input box, this value is an
upper bound on the exact fixed-multiplier certificate-family value.  A
strictly negative upper bound therefore shows that no scalar fixed multiplier
can certify that recorded obligation, independently of CROWN and grid density.
This is an executed-float diagnostic, not an outward-rounded real-arithmetic
proof, and it does not establish that the guarded property is safe.  A
route-consistent negative property point becomes `UNSAFE` only after the full
dynamic model independently reproduces the changed prediction.

The first Stage-B audit attempt is retained as a failed audit.  It replayed
CUDA-generated convolution outputs on CPU and found safety differences up to
`1.58e-3` and router-margin differences up to `9.91e-4`.  A direct check on
the registered CUDA device reproduced every stored value bit exactly.  The
auditor repair therefore changes only the replay device to the one frozen in
the targeted-search configuration; it does not change the runner output,
points, classification rules, or tolerances.

### Stage B r1 result

The repaired independent replay passes with zero issues.  All five
20-iteration alpha-CROWN calls nevertheless end in CUDA out-of-memory errors:
the process reaches approximately 70.7 GiB while an unrelated process holds
approximately 23.6 GiB.  This configuration omitted the scalable sparse-alpha
controls that had already been validated for the same router family.

The other diagnostic layers are informative but not decisive.  Across 55
replayed points, targeted searches find no route-consistent static-path
violation and no full-model prediction witness.  Each finite-point dual upper
bound is positive and selects `mu=0`, ranging from 0.5284 to 6.5269.  These
points therefore do not rule out a positive exact fixed-multiplier
certificate.  All five obligations remain
`UNRESOLVED_AFTER_REGISTERED_DIAGNOSTIC`.

R1 is retained unchanged.  A resource-only r2 repair may replace the dense
alpha configuration with the repository's previously validated scalable
sparse-alpha settings.  Obligations, multiplier points, targeted search,
classification precedence, and evidence semantics must remain unchanged.
