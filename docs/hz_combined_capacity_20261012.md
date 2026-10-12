# Combined sparse-source/block capacity decision

The current `CHECKED_SPARSE_HZ_BLOCK_SOURCE_V1` chain does **not** admit the
registered real-model recipes. This read-only assessment closes the capacity
question left by [source integration](hz_sparse_source_20261012.md); it does not
run another integration control or raise an admission limit. G1–G6 stay OPEN.

The next mechanism is a **request-owned checked source/block store with complete
row and endpoint tiling**. It must change how the source and support code consume
data, not just add a wrapper or replace inline objects with references that are
expanded again at every pair. GPU block candidates remain a required subsequent
device factor, not a substitute for this pre-proposal capacity work.

## Evidence and scope

The [versioned report](hz_combined_capacity_20261012_r1.json) assesses the
implementation at HEAD `0f37840a4fd9e24d586513a68189456f1e8b2d1c` and binds
46 files, including the two newly added audit/test scripts, by their own hashes. Its new
[derivation](../scripts/audit_hz_combined_capacity.py) reads only that inventory,
checks the previous metadata report unchanged, and inspects current call sites
and literal contracts without importing ACT. Eleven static controls pass.
Two read-only AI reviews check the producer count and device/interface boundary;
they are not independent human review or formal verification of the derivation.

No checkpoint, image, HZ producer, numerical library, optimizer or CUDA context
is loaded. Actual trained nnz, object sizes, peak memory and full-request time
remain unknown. This is not a new input selection, source/output certificate,
performance experiment or permission to rerun a sealed request.

## A corrected structural bound, including enclosure compensation

The registered MLP recipe has router `[3072,128,8]` and expert
`[3072,256,128,10]`. Seed 1/2 use this registered training recipe; seed 0 is
bound here through its recorded state counts, not a newly loaded topology.

For the current producer, conditional on successful finite conversion:

- The input box is already outward binary64; sparse enclosure adds no factor.
- A Linear output adds at most one exact affine-error factor and one subsequent
  binary64 enclosure factor: at most two continuous factors per output.
- An unstable ReLU adds two continuous and one binary factor, then at most two
  continuous compensation factors for its new output and equality. Stable rows
  and inherited rows are already binary64 and need no new compensation in this
  producer. Each unstable node adds one equality and two inequalities.
- Flatten, common-input restoration and guard addition add no factors. Guard
  inequalities are rounded outward in their RHS, not given slack variables.

With `A` Linear outputs and at most `U` unstable ReLUs, including the common
router once, `C <= 3072 + 2A + 4U`, `B <= U`.

| Object | A | U | Continuous bound | Binary bound | Total factor bound |
|---|---:|---:|---:|---:|---:|
| Common input/router | 136 | 128 | 3,856 | 128 | 3,984 |
| One expert template, including common | 530 | 512 | 6,180 | 512 | 6,692 |
| One pair, common counted once | 924 | 896 | 8,504 | 896 | 9,400 |
| Whole request, common once and all eight templates | 3,288 | 3,200 | 22,448 | 3,200 | 25,648 |

The pair constraint bound is 896 equalities plus 1,804 inequalities, including
12 pair-guard rows: 2,700 total rows. Compensation adds columns, not rows.
The last line is the distinct factor inventory across all templates, **not** a
simultaneously solved pair. Its unique domain rows plus all 28 guard suffixes
have a 9,936-row bound. Neither count includes repeated trace/evidence storage;
in particular, a 9,400-factor pair workspace limit would not bound the request.
The earlier 6,684 scenario omitted the new enclosure additions and is **not**
a bound for this producer. Historical records are retained unchanged.

These are conservative structural bounds, not measured sizes or guarantees that
all allocations fit. In particular, a bound above 4,096 does not prove an actual
trained state exceeds 4,096. The independent enclosure checker permits sound
overcompensation; therefore this producer-specific bound is not a universal
bound for every certificate that the checker could accept.

## Independent blockers remain in the same call chain

| Boundary | Current contract / finding |
|---|---|
| Declaration properties | At most 4 experts and 5 classes, versus 8 and 10 |
| Source, ownership, live snapshot and old inherited parsers | 128 factors/outputs and 256 constraints |
| Sparse enclosure | 4,096 reference factors; **16 entries in each entire cumulative output/equality/inequality vector**; 262,144 total nnz across six CSR matrices |
| Direct block support | Old source parser plus 128 global factors, 256 global rows |
| Endpoint request | At most 4 properties and 8 queries per batch; real classification needs 9 properties and 9–18 queries per pair |
| Source operators | Flatten/Linear/ReLU only; Conv2d and AvgPool2d are still unsupported |
| Portable / hard-supervised entry | Older schema/code inventory; 4 MiB/member, 32 MiB total; source parameters alone have a 74,254,592-byte base64 lower bound |

The sparse `chunk_size` controls visits/checkpoints within coefficient loops. It
does not tile an oversized complete state through the sixteen-row gate. Both
the producer and independent source checker still invoke old `parse` before
and after enclosure. Changing only the new primitive or candidate kernel
cannot admit the real pipeline.

Expert propagation occurs once per expert, but the pair loop recreates live
common/entry/expert objects and retains all pairs before support preparation.
For eight experts, support packages have **112 snapshot occurrences**: 28 common,
28 entry and 56 expert occurrences; each expert occurs seven times. This counts
stored occurrences, not unique underlying allocations or peak RAM. Local copied
row dictionaries, global dense objective/residual columns, dual/best-dual arrays,
and whole-object JSON hashing remain. There is no measured memory/speed claim.

All 28 unordered pairs and nine properties remain: **252 duties, 252–504 distinct
endpoint objectives** depending on gate degeneracy. Conv has six pairs and 54
duties, but is separately blocked by operators as well as source capacity.

## Next implementation contract: change consumption, preserve mathematics

This is a design decision, **not a real-model or GPU execution freeze**. Before
capacity controls execute, their finite objects, attempt envelope, complete
byte/nnz/work limits and expected decisions must be independently frozen.
The 9,400 count alone cannot determine byte/nnz or time limits.

1. One request-anchored manifest binds the declaration, resource contract, code,
   immutable source members, factor ownership and full obligation inventory.
   Store common once, each expert output/private suffix once, and each pair's
   guard suffix plus references. Pair guards never become common template facts.
2. Source states are checked views over complete row ranges and shared/private
   factor namespaces. Producer and checker each validate their own input;
   producer-side validation is not an acceptance authority. Exact affine/ReLU/
   guard equations, outward compensation, sparse-union residuals, factor maps
   and actual SparseHZono coefficients must all remain checked.
3. Change the consumer closure together: declaration tensor reader; source
   pack/unpack/identity; exact source production/checking; sparse enclosure;
   inherited owners/common/guard/live/snapshot/property functions; block source
   parsing and projection; endpoint inventory and aggregation. The existing
   `check_block_support.structure -> check_hz_endpoints._source` edge must not
   leave a hidden 128-factor gate. Preserve old APIs/contracts under old schemas.
4. Prepare one pair and at most eight objectives at a time. Do not build every
   live pair or serialize dense objective copies for every endpoint. A tile is a
   partition of the predeclared full inventory: `8+1` for nine degenerate targets,
   `8+8+2` for eighteen distinct targets. Equal endpoint weights may share a query
   only after equality is independently checked. No missing, duplicated or
   transplanted leaf may produce a complete positive conclusion.
5. Keep the same 128-update float64 candidate algorithm, shared-residual-before-
   box-support rule, exact original-coefficient reception, zero fallback and
   acceptance threshold. No new tightening, pair-specific propagation or changed
   mathematical duties is bundled into the storage factor. Tile costs share one
   request deadline; no fresh 300-second allowance per tile.

The six required control categories are: complete fixed recipe-shaped synthetic
inventory; cross-old-limit full rows/factors with nonzero compensation; expanded
small-object domain/target/same-candidate exact differential; complete/partial
tile reception; reference/alias/guard-scope attacks; and finite-resource plus
relocation checks. Their execution is not authorized by this report alone.
Old controls remain regressions, not new coverage claims. If given-source
subcontracts alone pass, report only those—not complete real-source admission.
The `8+1` and `8+8+2` static arithmetic checks are not execution tests of a tiled
endpoint implementation; that implementation does not yet exist.

This reduces repetition by representing the same obligations once, not by
discarding constraints or checking fewer properties. It is necessary capacity
engineering for the MoE block algorithm, not sufficient on its own to close G1.

## GPU seam and unchanged scientific boundary

`block_support._candidate_columns` is the numerical device seam: retain three
local sparse blocks, sum their contributions on shared residual coordinates,
then take one absolute-value box bound. A GPU must return untrusted finite
multipliers to `check_block_support.evaluate` for exact checking on the original
coefficients. The flat device kernel is not interchangeable with this block
representation. Source dispatch, request dispatch and candidate metadata currently
admit CPU only; a kernel-local device flag would leave the chain disconnected.

GPU removes none of the pre-proposal blockers above. After representation is
fixed, implement the same block algorithm as a separate device factor, obtain
new physical admission, and test complete coverage before full-cost acceleration.
Do not reuse flat CPU/stub controls as physical block-device evidence or retry
the previously refused device attempt automatically. No GPU benefit, real full
certificate or external victory is established by this capacity decision.

## Recheck

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -S -m unittest scripts.test_hz_combined_capacity scripts.test_hz_block_capacity scripts.test_hz_real_intake
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -S -m scripts.audit_hz_combined_capacity --check docs/hz_combined_capacity_20261012_r1.json
```

The combined command also including `scripts.test_project_layout`,
`scripts.test_proof_closure` and `scripts.test_proof_closure_costs` passed **48
tests** (11 new static controls plus 37 regressions). A separate-process report
re-derivation matched. Navigation checked 337 local links, all 619 frozen source
identities and 4,632 archived handoff lines unchanged. These checks do not
independently prove a neural network or execute the proposed tiling.

No historical result, checkpoint, threshold, model or sealed input is changed.
No cache or scientific artifact is deleted in this stage; there is no new raw
experiment archive. The separate workspace allocated-size reading was
225,909,108,736 bytes, not an attribution of total growth to this stage.
