# Separately frozen reuse engineering comparison and actual HZ export

This stage follows the analytic controls. No old result is overwritten.
`configs/proof_reuse_paired_r1.json` freezes both tasks before execution.
Use act-py312; no installation, training, GPU workload, new holdout or F1.

## Real trained checkpoints: paired reuse off/on

Use ranks 0--9 of the already-observed jointly clean-correct 100-input manifest
on its three accepted bal010 checkpoints, at the same 2/255. Prefix selection
does not filter on solver status, runtime, reusable obligations or prior SAFE.
These are actual trained models, not the constant analytic controls, but their
roughly 48% accuracy and shared architecture do not answer high-accuracy or
cross-architecture generalization. The cohort is observed engineering data,
not untouched holdout. Both arms rerun; no historical timings are reused.

60 requests use the same 300-second external cap, including startup/loading,
propagation, fact construction and all solving. The only configuration
difference is the boolean scoped_proof_reuse. Model order rotates by rank,
and arm order by rank plus model index, balanced 5/5 in each position/model.
Single-thread BLAS/OpenMP, sequential jobs, shared-server load recorded.
Post-run audit/replay is outside both measured executions. Maximum worker
budget is five hours; elapsed runtime can also include audit overhead.

All scheduled inputs remain. No early stop for desirable results, no outcome-
based replacements or timeout enlargement. Failure stops with artifacts
preserved; this small runner has no resume path. Code/config changes require
a new run identity, never mixing code in one output root.

Report per-model SAFE/UNSAFE/UNKNOWN/TIMEOUT; gained/lost SAFE and solved;
mean/median paired observed time differences; F0 completed solve-row and reused
row counts with missing counters on outer timeouts. Known counters are only
observed partial totals, not an imputation of zero for censored requests.
Different terminal paths mean solve-count reductions cannot alone be read as
speedup. There is no post-hoc superiority gate. Every complete package is
structurally audited and every UNSAFE fully replayed; the final auditor also
reconstructs reused properties inside UNKNOWN packages and checks paired literal
model/domain/property identity. SAFE/UNSAFE conflicts fail the run audit.

## Actual guarded-router HZ to LP

Freeze seed0/rank0 from the same manifest. Propagate its router in ACT, retain
the clean unordered top-2 guard, and query the minimum-index selected member's
score minus the minimum-index outsider's score. This single query is selected
without inspecting whether its lower bound is positive. A zero/negative checked
bound is an admissible result, not a reason to choose another query.

Export the actual SparseHZono center, Gc/Gb, equality and inequality matrices,
RHS vectors and frame. The LP keeps every factor in [-1,1], relaxing binary
{-1,+1} factors to continuous [-1,1]. Thus the HZ is contained in the LP's
represented output set. This is NOT an exact MILP encoding. The objective and
constant are exact rational combinations of stored binary float coefficients;
no floating dot-product result is silently treated as exact.

An independent stdlib checker does not call the exporter, SciPy or HZ code:
it reconstructs objective and all constraint coefficients from serialized CSR,
checks source identity, retained rows/RHS/signs and factor bounds, then invokes
the exact rational LP certificate checker. The SciPy proposal has 60 seconds,
and any approximate proposal objective is corrected by exact residual terms.
The raw source, export, query, checkpoint and represented request are hash-bound.

This closes the **given stored HZ -> LP relaxation -> checked lower bound**
chain. It does not prove network-to-HZ propagation, every numerical operation
used to construct the guard, the soundness of the original input box, deployed
floating-point semantics, or a MILP search tree. There is no new full-model
SAFE claim from this margin query. Tests mutate row omission, coefficient
sign, output constant, factor box, objective and source to require rejection.

Run the real export first, retain/check it, then launch the paired runner on
committed sources. Commands: `python -m act.pipeline.moe.hz_lp_real_control
--config act/pipeline/moe/configs/proof_reuse_paired_r1.json`, then
`python -m act.pipeline.moe.proof_reuse_paired`. Final audit is automatic;
an independent rerun uses `--audit RESULT_ROOT`. Publish compact results only
after actual completion. No result is asserted by this registration.

## Actual HZ export result

The registered query completed at `f5ea8457d`: seed0, dataset index 3000,
clean pair {4,5}, objective router score 4 minus score 0. There are 3,075
factors and one explicitly relaxed binary. The independent process reports
PASS / zero issues and checks the rational lower bound

    12910410846830751887830790100398256853793 /
    1361129467683753853853498429727072845824

(approximately 9.485071885777709). The exact value, export/checkpoint/request
hashes and independent audit hash are in
`results/hz_lp_real_export_20260911_r1.json`. This is a checked guarded-router
support lower bound relative to the stored HZ, not a full output certificate
or an independent proof of network propagation. Paired reuse results are
separate and not implied by this positive margin.
