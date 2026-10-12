# Checked sparse row enclosure

The [frozen design](hz_sparse_rows_design_20261002.md) is implemented as a new,
opt-in given-HZ primitive. All nine control groups, the producer-free saved-evidence
audit and 101 existing regressions pass. Each complete check parses the owned
source and target once, preserves shared/private factor identities and accumulates
the full exact sparse residual. This removes the old repeated whole-source parsing
from this new primitive, not from the existing network conversion path.

The result is **checked inclusion of a supplied exact HZ in a stored binary64
HZ**, not a new bound-tightening theorem, real-model certificate, speedup or GPU
result. Production defaults and all old source, block, query and portable limits
remain unchanged. All six research goal gates remain OPEN.

## Implementation and mathematical boundary

[Producer](../scoped_source/hz_sparse_rows.py) and
[independent checker](../scoped_source/check_hz_sparse_rows.py) use a new
`HZ_SPARSE_ROW_OUTER_ENCLOSURE_V1` schema. Original factors keep their IDs, owners
and common frame. Output compensation and equality slack have distinct new
continuous factors that may appear only in their designated row. Inequalities
weaken their right-hand side outward. Binary factors remain binary in the
inclusion argument; they are not silently merged with expert or shared factors.

For every output or equality, the checker recomputes the exact constant error
plus the L1 coefficient residual on the union of original source and target
columns. Target-only original columns are included. A separate complete pass
checks the placement of added compensation factors across all six matrices.
The producer rounds only after summing each complete residual; chunk sizes 1
and 256 therefore create identical targets, maps and compensation. Sparse key
sorting and all coefficient visits are counted, not described as free work.

The checker uses only the Python standard library, does not call the producer
and rejects partial, mutated, expired or resource-exceeding objects. It checks
the caller's identities again before return. Owned snapshots prevent ordinary
caller mutation from changing the object being checked; this is not a streaming
or constant-memory format. The 4096-factor, sixteen-row-per-kind and arithmetic
limits are specific to this new primitive. They do not admit a whole model.

## Fixed controls and independent audit

The [runner](../scripts/run_hz_sparse_rows.py) binds the prior clean HEAD
`910c0ab12a295bc8e61f2f622da117248cc95cfc`, the design hash and implementation
snapshots. [Final compact audit](hz_sparse_rows_20261012_r3.json) records:

| Check | Result |
|---|---|
| Eight unchanged legacy references | Exact old/new coefficients, maps, ownership and residuals agree |
| One fixed 3072-factor mixed reference, two chunk sizes | Identical target and proof; independent analytic reconstruction agrees |
| Actual SparseHZono instantiation | All ten executions round-trip stored coefficients exactly |
| Supplementary exact assignment extensions | 15 checked; these do not replace the universal residual argument |
| Concrete corruption queries | 36 independently reconstructed and rejected, including target-only columns in all six matrices |
| Arithmetic and resource refusals | Eight checked, including both original overflow cases and intermediate bit growth |
| Deadline and mutation faults | Eight reached their specified phase and exact terminal reason |
| Archive completeness and checker independence | Missing/substituted records rejected; no producer or numeric libraries imported by the separate audit |

For the fixed wide reference, each check visits 12,288 original residual union
entries and 12,291 target layout entries. Source and target each parse once;
the target adds only three factors. The compensation equals the frozen analytic
expectation `D=7*2^-96`, with output radii `2^-54+D` and `D`, equality radius `D`,
and inequality RHS `1`. Original binary columns shift from flat offset 1536 to
1539 without changing their identities. The actual compensation and target hashes
are in the compact audit.

Independent review found no inclusion defect. Archive hardening remains visible:
R1 passed before independent reconstruction of the complete wide target and
prescribed resource-control candidate; R2 added these bindings and precise cost
types; R3 binds exact fault messages and rejects substitutions with unrelated
errors. All ten normal references, proofs and actual snapshots are unchanged
across all three attempts. These are audit repairs, not parameter search or
additional mathematical fixtures. R1 and R2 remain preserved.

## Cost and execution limits

Each normal control uses one cooperative 300-second deadline covering creation,
old differential where applicable, production, actual instantiation,
serialization, checking and supplementary points. Auxiliary controls use 30
seconds; the suite has a finite 1800-second envelope and at most three attempts.
There is no hard-watchdog or production-budget claim in this stage.

| R3 observed component | Seconds |
|---|---:|
| Test-module import | 0.0122 |
| Complete control suite | 10.5580 |
| Archive publication | 0.1096 |
| Total including implementation snapshot setup | 10.6833 |
| Separate saved-evidence audit | 2.8886 |

These are one execution's accounting intervals, not a speed comparison. In
particular, the first legacy instantiation includes lazy ACT imports. No native
optimizer, model/data load, new output query or CUDA call ran. The missing Gurobi
license warning occurred during imports, not a failed or substituted solver run.

The 101 passing regressions cover legacy binary64 enclosure, checked source
integration, expert templates, block support, capacity, navigation and the
sealed proof ledger. Read-only AI code review is not independent human review.

## Evidence and next gate

Detailed records remain in
`/data1/Kane/MOE/baseline_runs/hz_sparse_rows_20261012_r1`, `r2` and `r3`
(the latter two abbreviate the same full directory-name prefix). They contain
implementation snapshots, concrete objects, all refusals and ordered costs.
R3 summary SHA-256 is
`492ee1346543f88ca519c97ddc303a516bee05dad5f0ba9f5141bb1370b9b97b`.
The final independent audit can be replayed with the recorded implementation:

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -S -m scripts.run_hz_sparse_rows audit /data1/Kane/MOE/baseline_runs/hz_sparse_rows_20261012_r3
```

The three local archives occupy 44,544,000 allocated bytes. The stage-end
workspace reading is 225,902,252,032 bytes; that whole-workspace value is not
attributed entirely to this stage. No scientific files or caches were deleted.

Stop this primitive's width and timing study. The next scoped decision is how
to integrate it into the exact affine/ReLU/guard source path while retaining
independent reference checks, common/private identities and fresh output evidence
for every new matrix. Full source dimensions, complete property inventories,
convolution/pooling and package capacity remain open. Do not add another wrapper,
raise old caps, reopen sealed inputs or retry physical GPU admission under this
protocol. GPU and real-request benefit require their own compatible algorithm
path and frozen admission/measurement stages.
