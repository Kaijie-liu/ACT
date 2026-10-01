# Remaining capacity limits after HybridZ block support

The [metadata assessment](hz_block_capacity_20261002_r1.json) finds that direct
block support does **not** yet admit the registered real models. It removes joint
matrix assembly, but leaves source dimensions, wide-row checking, duty limits,
unsupported convolution operators and portable storage unresolved. No checkpoint,
dataset, model, HZ propagation, native optimizer or GPU was loaded or executed.

The next mechanism is a separately versioned sparse row enclosure and checker:
parse the source once, accumulate exact row residuals over all nonzero columns,
and retain one checked compensation factor per affected output/equality. This
addresses a specific source-construction obstacle before another supervisor or
GPU wrapper. It is not permission to raise the old caps or run a real request.

## What block factoring changed

The prior [finite block result](hz_blocks_20261002.md) establishes same-domain and
same-candidate equality, not real timing or memory. The registered bal010 recipe
has eight experts and ten classes. Static extrapolation of the current template
loop gives eight expert propagations and 52 network-layer records, versus the old
per-pair path's 56 and 340. These are recipe counts, not executions on checkpoints.

All 28 unordered pairs and nine classification properties per pair remain:
252 duties and between 252 and 504 endpoint targets, depending on gate degeneracy.
The current support request still retains 28 common snapshots, 28 guarded-entry
snapshots and 56 expert-template snapshots. Each expert appears seven times.
The block path also retains global dense objective, residual and argmin arrays;
local duals and their best copies; local sparse matrices and transposes. Avoiding
joint construction alone does not establish lower peak memory.

For the **old all-unstable recipe scenario**, the row accounting is:

| Row collection | Rows |
|---|---:|
| Common router constraints | 384 |
| Guarded entry | 396 |
| Each expert template including its common prefix | 1536 |
| Four stored snapshots | 3852 |
| Three executed blocks | 2700 |

The difference of 1152 is repeated common-prefix storage, not removal of required
constraints. These counts exclude layer traces and row-proof copies. They are not
measured nonzero counts, memory or time.

The old **6684-factor scenario is not a bound for the new path**. It includes
source affine-error factors and unstable ReLU factors, but not the additional
output/equality compensation introduced by row enclosure. Those new continuous
factors survive into later propagation. This audit does not estimate their
trained-model count. The row lift itself adds columns, not constraint rows.

## Remaining admission gates

| Interface | Existing restriction | Consequence |
|---|---|---|
| Source property generation | ≤4 experts, ≤5 classes | bal010 has 8 and 10 and is rejected first |
| Global state and block support | ≤128 factors/outputs, ≤256 constraint rows | 3072 input factors and outputs already exceed the limits; zero radius still creates an input factor |
| Local row enclosure | ≤8 original nonzero factors | Wide rows are unsupported; their actual occurrence was not measured |
| Complete property roster | ≤4 properties | Ten-class classification needs nine |
| One support batch | ≤8 queries | Even a degenerate gate needs nine; a nondegenerate gate needs eighteen |
| Source operators | Flatten, Linear, ReLU | The convolutional family needs unsupported Conv2d and AvgPool2d |
| Old portable envelope | 4 MiB/member, 32 MiB total | Registered parameter base64 alone is at least 74,254,592 bytes |
| Complete execution and portable checker | Old source schema | They do not automatically check or supervise the new block schema |

The eight-factor row gate is **not an exponential corner-enumeration requirement**.
The exact checker adds absolute coefficient differences. For original factors
with absolute value at most one, this bound is valid for arbitrary row width.
No trained weights were inspected, so input dimension alone does not establish
that a particular trained row has more than eight nonzero columns. Global input
and property refusals are definite; real row widths and running capacity are not.

## A concrete repeated checking cost

In [the row checker](../scoped_source/check_hz_row_enclosure.py), `check()` first
parses the reference, then calls `projected()` for each of its R rows. Each call
again parses and canonicalizes the entire reference, including its matrices.
The static audit binds this call pattern by AST and fails if the pattern changes.
On the complete successful path there are R+1 whole-reference parses, apart from
target parses, hashes and local checks. For N stored coefficients this repeats approximately R×N coefficient
visits, with additional sorting and exact arithmetic. This is a structural work
count, not proof that parsing dominates measured end-to-end time.

The next [sparse row design](hz_sparse_rows_design_20261002.md) changes that
algorithmic organization: all coefficients and ownership still get checked,
but a row no longer triggers reparsing all unrelated rows. Exact residuals are
combined before one outward rounding and one compensation factor. It must not
split shared input coordinates or give each chunk an independent copy of them.

Per-request immutable template storage and duty streaming remain separate
follow-ups. Streaming does not fix a failed source inclusion proof; GPU proposals
cannot compensate for a source that the checker cannot construct or admit.

## Checks and decision

Eight standard-library controls pass, including read-inventory restrictions,
complete-duty arithmetic, stored versus executed row accounting, and mutations
of the repeated-parse pattern and query contracts. A separate regeneration
matches the saved report. The first test run had five errors because the new
inventory listed a nonexistent, unused `scoped_source/hz_portable.py`; removing
that erroneous path fixed the diagnostic. No algorithm or frozen result changed.

The report rechecks the previous intake report against its original source
bindings, then binds current block and row code. Model hashes remain registered
metadata, not freshly verified checkpoint hashes. Read-only agent reviews are
not independent human technical review.

```sh
PYTHONDONTWRITEBYTECODE=1 /data1/Kane/miniconda3/envs/act-py312/bin/python -B -S -m unittest scripts.test_hz_block_capacity
PYTHONDONTWRITEBYTECODE=1 /data1/Kane/miniconda3/envs/act-py312/bin/python -B -S -m scripts.audit_hz_block_capacity --check docs/hz_block_capacity_20261002_r1.json
```

Continue with the separate given-reference sparse row mechanism; do not launch
real propagation, another full-size model trial, physical GPU, or sealed input.
Its success alone will not remove global source/property/package gates or close
G1–G6. No new wrapper, larger budget or repeated tiny timing run is the next task.
