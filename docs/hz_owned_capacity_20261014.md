# Implemented owned workspace: full-shape capacity decision

**The optional interfaces passed their finite controls, but the implemented
dense workspace cannot admit the registered shape at width 8 under the existing
128 MiB arena.** A structural lower bound, not a worst-case estimate of weights,
already exceeds that ceiling during preparation of endpoint 6. Earlier source
checks also still refuse this shape. Stop retrying this unchanged full-shape
path; no quota change, real-model run or GPU run follows from this assessment.

Starting branch was `feat/moe-route-verification`, clean and synchronized at
`a0c955cb1fcc3ad6494297c4e79c93ad128d96c9`. This stage reads pinned source text
and the [frozen topology ledger](hz_full_resource_ledger_20261014.json), without
constructing a source, loading weights, allocating model-sized arrays or
calling a solver. The [machine report](hz_owned_capacity_20261014.json) binds
those inputs and the analyzer. The prior
[implementation and controls](hz_bounded_runtime_controls_20261014.md) remain
unchanged; this report assesses their actual layouts, not the earlier proposed
compact format.

## Owned-buffer lifetimes

The registered seed 1/2 shape has 3,072 input coordinates, eight experts, ten
shared classes and 28 conservative pair obligations. This roster does not prove
that all pairs are reachable. Its 252 properties require 252–504 endpoints.

Let `N` be nodes, `F` factors, `H` all ordered view memberships, `M` pair dual
rows and `k` endpoints in one tile. The current workspace always uses general
working rational slots: `s = 1 + 16 + 2*1024 = 2065` bytes per coordinate,
including its presence byte. The implemented dyadic helper is not selected by
`Workspace.mapping`; substituting its smaller size would describe different
code. The root index owns `R = 16N + 4H` bytes. For native candidate encoding,
the frame size is `T = 128 + k(2152 + 16M)` bytes.

Reading `Session.prepare`, `pullback` and `_consume` gives:

| Phase | Simultaneously owned payload |
| --- | --- |
| Prepare peak | `R + s((k+1)N + kF)` |
| Prepared tile | `R + sk(N+F)` |
| Candidate immutable sealing copy | `R + sk(N+F) + 2T` |
| Nonempty candidate consumption peak | `R + sk(3N+2F) + T` |
| After tile clear and frame release | `R` |

The objective node maps and factor vectors remain live while the original dual
node maps, their pullback copies and dual factor vectors are allocated. Counting
only one residual vector would miss these overlaps. The formula for consumption
assumes at least one present candidate; the preparation refusal does not depend
on candidate contents. This is an owned-payload calculation, **not process RSS**.

Every input has a factor; every affine output and guard appends a node. Hence
`N >= 3072+3288+336 = 6696`, `F >= 3072`, `H >= 126632` and `M >= 12`.
The conservative upper counts are `N <= 19496`, `F <= 32048`, `H <= 136232`
and `M <= 2700`. The upper counts need not be attained together.

| Structural counts substituted | Width | Prepare peak, bytes | Consume peak, bytes |
| --- | ---: | ---: | ---: |
| Required lower counts | 1 | 34,611,824 | 54,785,216 |
| Required lower counts | 8 | 175,808,264 | 433,985,184 |
| Conservative upper counts | 1 | 147,554,464 | 254,038,304 |
| Conservative upper counts | 8 | 892,622,984 | 2,026,307,488 |

Each pair has at least nine endpoints, so width 8 starts with eight. At the
sixth endpoint's factor-vector allocation, even the lower counts require
**135,466,424 bytes**, exceeding **134,217,728** by 1,248,696 bytes. If earlier
source/profile checks allowed execution to reach this point, `Owner` would
refuse before allocating that vector. This establishes a necessary width-8
refusal. Events after that point in the ledger are counterfactual capacity
requirements, not memory actually allocated.

Width 1's lower counts fit; its upper counts do not. This does **not** prove
that a trained width-1 request would overflow, nor that it would be admitted.
Actual counts remain null. No runtime or 300-second completion prediction is
inferred from these bytes.

## Candidate frames: actual format, not the old design budget

The implementation stores full proposed **and explicit-zero** dual vectors.
Its prefix is 128 bytes, endpoint record 72 bytes and each general stored
rational 1,040 bytes. Both advisory lower values occupy general rational slots,
even in the native profile. With scalar width `κ = 8` or `1040`, an endpoint
uses `S(M,κ) = 2152 + 2Mκ` bytes. The complete stream uses
`128*tiles + sum_pair(endpoints_pair * S(M_pair,κ))`.

At the registered maxima of 504 endpoints and 2,700 rows per pair:

| Profile | One-endpoint frame | Eight-endpoint frame | Whole stream, width 1 | Whole stream, width 8 |
| --- | ---: | ---: | ---: | ---: |
| Native binary64 | 45,480 | 362,944 | 22,921,920 | 22,868,160 |
| General rational | 5,618,280 | 44,945,344 | 2,831,613,120 | 2,831,559,360 |

All values are bytes. Width 8 uses `8+8+2` endpoints per pair: 84 tiles. Under
the unchanged 4 MiB tile and 16 MiB minus 8,192 bytes cumulative limits, the
native tile upper fits but its complete stream upper does not; the general
profile's one-endpoint upper already exceeds the tile cap. These are exact
format sizes at specified dimensions, **not actual model evidence sizes or
proof that every actual stream overflows**. Fewer endpoints or rows can reduce
them. Both profiles retain every logical role and dimension check.

The earlier approximately 96.8 MB whole-store design used compact zero-role
descriptors. It cannot be applied to this implementation. The table here also
excludes source, forest, receipts, archive indices and duplicate archive copies.

## Earlier refusals and unclosed costs

`RootTable` is built only after the old source checker succeeds. It does not
replace that checker or release the forest. Pair aliases alone require 86,016
memberships; the mandatory total is 126,632, both above the old 65,536 limit.
The new index's 856,864-byte upper does not remove this earlier refusal. The
existing source builder/checker first-layer visit floor is still 13,369,344,
above the unchanged 8M Work cap.

**Complete cost remains NOT CLOSED.** Source capture/checking, planning objects,
Fraction/PyLong temporaries, Python metadata and allocator lifetime, native
coexistence, external byte aliases, full archive copies, request supervision
and both methods' cumulative costs are not bounded by this arena ledger.
Default ACT execution is unchanged. G1–G6 remain open; there is no new numerical
certificate, algorithm-effect result, GPU result or full-source proof.

## Checks and next decision

Eight new standard-library static tests check source/format bindings, independent
lifetime formulas, cleanup arithmetic, lower-versus-upper interpretation,
candidate dimensions and report recomputation. Thirteen existing handoff and
navigation checks also pass. Two read-only AI code reviews independently
derived the workspace and frame formulas; they are not human review or model
reproduction. Prior synthetic interface tests are not rerun or counted as new
scientific evidence in this stage.

Recompute the report, without importing ACT or numerical libraries:

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -I -S scripts/assess_owned_capacity.py --check-report docs/hz_owned_capacity_20261014.json
```

The next representation decision has concrete targets: each classification
objective initially has at most four nonzero roots, yet uses an `N`-slot bank;
several dense banks then coexist. A separately scoped design could retain sparse
objective roots and avoid overlapping copies, while preserving complete factor
residual checks and ownership controls. Existing shared/private factor scope
may be reused, but is not new merely by renaming it. Compact explicit-zero
encoding addresses candidate bytes only; direct source construction/checking
needs its own contract. Neither removes the other barriers automatically.

Do not start another full-shape capacity retry, silently switch scalar profiles,
change caps or count passing small controls as admission. No new representation
mechanism is implemented by this static stage. Any follow-up must bind its
representation and lifetime before testing; quota adjustments and real/GPU
execution still require separate approval. All mathematical checks, acceptance
gates and the 300-second budget remain unchanged.
