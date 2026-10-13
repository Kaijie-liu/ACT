# Lossless wire, bounded intake and archive controls

Implemented after the user-approved [frozen protocol](hz_wire_interface_design_20261014.md),
commit `84a79e8f912441d4770e3e0c208c73f385c5cbf9`. This is an **opt-in interface
stage**, not permission to increase capacity or start a registered model/GPU run.
The [machine-readable record](hz_wire_interface_controls_20261014.json) contains
identities, exact outcomes, per-phase costs, archive totals and failed controls.

## Implemented boundary

- `wire_forest`: little-endian finite binary64 fields, bound first-affine source
  row references, explicit compensation tails, immutable lazy node access.
  Logical V3 identity is recomputed without building the full coefficient JSON
  or Fraction bank. Physical and logical byte limits both remain enforced.
- `check_source_dag.initialize`: common independent mathematical checks remain
  on both paths. The wire intake is a closed type, not a duck-typed substitute.
  Input inclusion, affine compensation, all ReLU rows, guard signs, factors and
  all tie-legal pair roles are still checked. The terminal rechecks every source
  chunk/tensor and the manifest/inventory under the same deadline.
- `wire_proof`: complete proposed and explicit zero candidates are decoded one
  tile at a time. Session `retain_raw=False` cannot emit the old eager proof DOM.
  The original logical proof hash and cap are checked incrementally; missing
  candidates remain missing, and all output obligations are aggregated.
- `wire_archive` / `wire_package`: no-overwrite single-owner publication,
  terminal reserve, actual-file and failed-prefix reconciliation, bounded child
  logs, owned process-group cleanup, durable terminal reception and package
  inventory checks before and after proof replay. Both relocation copies count.

The binary row format has derived V3 shape limits before materialization:
inline local rows at most four terms (input, compensation, ReLU, guard), source
rows at most input width plus two compensation terms, references at most 4,096.
The existing mathematical checker still determines whether each such row is
correct. These are not relaxed mathematical or resource gates.

## Controls and review

21 new control groups pass. They cover exact same-candidate differentials
against both old V3 and an expanded scalar oracle; ties/multiple pairs/classes;
source/row/factor/guard/property tampering with recomputed transport identities;
ReLU omissions; mutation during reads and after the last matrix access;
nonfinite/unsupported exact values; malformed layout; large descriptors before
row materialization; incomplete/duplicate evidence; Work/deadline exhaustion;
two-copy and small-record quotas; partial prefixes including zero-byte files;
symlink/FIFO rejection; exception/log/cutoff cleanup; durable terminal tampering.

Portable replay passes under `python -I -S -B`, with an audit hook denying reads
of the original fixture/archive and the repository. It uses only bundled
stdlib checker code and the relocated source/forest/contract/tiles. No producer,
optimizer, checkpoint, dataset, numerical library or solver is loaded by that
check. The process has a bounded deadline; it does not turn an incomplete online
result into a success. The four archived rechecks also share each control's
single Work object and 300-second deadline.

48 existing controls pass: 26 source-DAG/store unit regressions and 22 tensor-row
regressions. These are finite legacy tests, not new models or GPU execution.
Two read-only code reviews found and helped close source resealing, preallocation,
post-cleanup timeout, actual archive inventory and durable reception gaps.
This is not a substitute for independent human review or a clean-environment
scientific reproduction.

Initial failures remain disclosed: the first portable bundle missed the
`upstream_source.checker` stdlib dependency; the first source-reseal implementation
missed the tensor hash's dtype/shape prefix. Both were fixed, not bypassed.

## Four unchanged finite fixtures

No optimizer is used: proposals are fixed finite signed duals and explicit zero.
The old checker and independent scalar oracle give the same exact results.

| Fixture | Old logical forest bytes | Wire bytes | Required endpoints | Full result |
| --- | ---: | ---: | ---: | --- |
| mixed | 9,998 | 14,025 | 6 | checked positive captured real source |
| deep | 14,734 | 21,380 | 6 | UNKNOWN nonpositive |
| half | 5,841 | 9,644 | 5 | checked positive captured real source |
| guarded | 3,945 | 6,520 | 24 | UNKNOWN nonpositive |

These are existing synthetic controls, **not new real-model certificates**.
Fixed-width/header overhead makes these tiny forest files larger, not smaller.
Consequently this stage does not establish a compression ratio, a speedup or
the earlier 110.6 MB full-shape design envelope.

R2 total control wall times were approximately 0.160, 0.208, 0.152 and 0.135
seconds. They include fixture preparation, old construction, encoding, source
publication/checking, fixed proposals, tile writing/checking, aggregation and
source reseal, code bundling, both copies, relocated rechecking, temporary-input
cleanup, durable terminal reception and archive reconciliation. Individual
phase times and Work are recorded. These tiny-control observations are not a
performance experiment or a prediction of registered-model completion.

Owned archive logical bytes, including both snapshots and terminal, were
376,528; 411,133; 365,289; and 361,190. Generated given-source fixture files are
also disclosed in peak control disk counts and removed after checking. The
128 MiB limit is logical file bytes, not allocated filesystem blocks/RSS.

R1 and R2 artifacts are retained under:

- `/data1/Kane/MOE/runs/hz_wire_interface_controls_20261014_r1`
- `/data1/Kane/MOE/runs/hz_wire_interface_controls_20261014_r2`

R1 completed but its verbose stdout exceeded the tool display cap. R2 changes
only report verbosity; it repeats the same four fixed controls, not a search.
No raw source/weights or runtime package binaries are committed to Git.

A separate read-only stdlib artifact review confirms all four terminal/summary
identities, actual archive bytes equal to two snapshots plus terminal, independent
file inodes, and all 15 bundled checker-file identities matching current code.
R1/R2 bundle and member identities are identical. This review checks records and
identities; it does not claim a second independent proof of every bound.

The outer archive continues to contain logs, terminal and both copies. **The
portable checking target is its frozen `relocated/` snapshot**, not the growing
outer directory. The archive ledger and a portable mathematical package are
different objects. Code in a package is trusted checker software; a hash alone
does not prove the checker implementation correct.

## Reproduction and remaining limits

From the ACT repository using the existing environment:

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -m unittest scripts.test_wire_evidence -v
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -m unittest scripts.test_source_dag scripts.test_hz_request_store_units scripts.test_tensor_rows -v
```

`scripts.run_wire_archive_controls --output NEW_DIRECTORY` is the explicit
four-fixture archive driver. It never starts a native candidate optimizer.
`wire_package.check_package(root, expected=..., work=...)` is the reusable
bounded portable entry. The expected bundle identity is external, not trusted
from the package's own claimed status. Its command-line entry is for labelled
offline replay; an online caller must pass its existing Work/deadline.

The encoder still consumes an existing V3 JSON document. Live capture,
construction peak memory, full native COO workspaces and GPU lifetimes are not
bounded by this interface. Terminal hash resealing is not an OS atomic snapshot
or a defense against malicious concurrent writers in the same account/process.
The verified statement is about the captured real-arithmetic source, not live
checkpoint correspondence or native floating-point execution.

All old gates remain: 8M Work, 4,096/8,192 stored/intermediate bits, 128 MiB
forest/archive, 16 MiB proof, 4 MiB tile and strict `1e-7`, including the old
logical-byte limits even when physical compression is smaller. The registered
eight-expert model remains **unadmitted**. No G1–G6 success criterion is closed
by control counts. A future capacity assessment must account for lazy-source
repeated reads/resealing, logical limits, construction/native memory and both
retained copies; real/GPU execution is not authorized by this stage.
