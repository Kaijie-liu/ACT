# ImageHZ-lite VGG/Tiny Prototype — Design Plan

Date authored: 2026-06-04
Authorizing decision: roadmap §6b VGG mini-atlas gate = PROCEED
Authorizing root: `audit_results/vgg_mini_atlas_canonical_plus_missing_20260604T023543Z/`

This document is the design plan **before** any solver code begins. It
encodes the advisor's invariants, scope, and hard stop gates so future
work cannot drift. If a step in the plan must change, edit this file
first, then code.

---

## 1. Why this exists

VGG mini-atlas final gate (§6b) showed:

- 15 / 15 iids with snapshots have severe root-correlation loss
  (`root_ng ∈ {1, 5, 10, 20, 100}` vs `n_input = 150528`).
- 15 / 18 iids fire Girard cap reductions at MAXPOOL2D (L11, L18, L25,
  L32) or wide-ReLU (L17, L29, L35) layers.
- The loss is concentrated exactly where ImageHZ's locality
  representation could plausibly preserve more root provenance.

CIFAR atlas v3 (§5/§3.3) showed the opposite: `root_ng = n_input`
across 185 / 185, so CIFAR is **NOT** a target. ImageHZ for CIFAR
stays STOP.

---

## 2. Invariants (must hold throughout the prototype)

All of these come from the project design principles (§2 of roadmap)
plus the advisor's 2026-06-04 directive:

| ID | Invariant |
|---|---|
| I1 | **Forward-only.** No CROWN-style backward propagation. |
| I2 | **No gradients.** No autograd, PGD, FGSM, CW, DeepFool, or any gradient-derived candidate. |
| I3 | **No Gurobi, no MILP.** Continuous LP via SciPy / HiGHS / highspy is acceptable; integer / MILP solvers are not. |
| I4 | **No fallback verifier.** UNKNOWN stays UNKNOWN unless ACT/HZ itself proves or finds a strict replay witness. |
| I5 | **No branch-and-bound, no input splitting.** |
| I6 | **No random / no center / no corner sampling.** FAL candidates must come from structured HZ/LP programs and pass raw ONNX strict replay. |
| I7 | **CIFAR untouched.** ImageHZ never runs on `cifar100_2024`. The existing production endcap sidecar keeps owning CIFAR. |
| I8 | **No benchmark-specific patches.** Anything dataset-name-gated is forbidden. The prototype must be a general representation, structurally gated. |
| I9 | **Receipts carry provenance.** Every FAL receipt records `canonical_root + instances_csv_sha256 + onnx_sha256 + vnnlib_sha256`. |
| I10 | **Fail-closed.** Any shape, type, or contract violation raises and the verifier returns UNKNOWN; never silent fallback. |

A breach of any of these is a project rule violation and must be
fixed before the work continues.

---

## 3. Operator scope (Phase 0)

The prototype targets the smallest set of operators that exercises the
Girard cap sites identified by §6b:

```text
Conv2D
ReLU
MaxPool2D
Flatten
```

Everything else (BatchNorm, Concat, Add, residual blocks, dynamic
slicing) is **explicitly out of scope** for Phase 0. If a benchmark
uses them, the prototype fail-closes and the production HZ path keeps
ownership.

Rationale: VGG16's conv body is exactly Conv2D + ReLU + MaxPool2D up
to the FLATTEN at L33, then Dense+ReLU+Dense in the tail. Tiny variants
that match this skeleton are eligible; ResNet variants are not unless
ADD support is added later under a separate gate.

---

## 4. Representation design (Phase 0)

### 4.1 Per-tile generator block

```text
ImageHZ_lite:
  c     : (C, H, W) center tensor                          (float64)
  blocks: list of TileGenerator
  root_provenance:
    blocks → original input root-factor ids (1..n_input)
```

Each `TileGenerator` is a (tile_C, tile_H, tile_W) block over a fixed
spatial slice of the (C, H, W) feature map. It carries:

```text
TileGenerator:
  origin_chw       : (c0, h0, w0)         # top-left in feature map
  shape            : (tc, th, tw)
  values           : (tc, th, tw) tensor  # generator coefficient
  aux_kind         : 'root' | 'relu_aux'  # provenance label
  root_factor_ids  : list[int]            # for 'root' kind only
```

The center plus the sum over `blocks` reconstructs the exact set of
possible activations. The advantage over dense HZ is that
`MaxPool2D(kxk)` over a feature map only acts on at most `k×k` tiles
locally; a tile-aligned reduction can preserve more block boundaries
instead of collapsing all generators into a single shared `(n, 1)` box
column the way the current Girard cap does.

### 4.2 Operator semantics

| Op | Action on ImageHZ_lite |
|---|---|
| Conv2D | Convolve `c` and each block's `values` independently. Block shape grows by `(k-1)` per spatial dim; block origin shifts by stride; tiles never cross output channels (Conv preserves channel decomposition for our scope since groups=1). |
| ReLU (per-position) | Compute `lb, ub` per position from block bounds. Stable-active passes through. Stable-inactive zeros. Unstable applies DeepZ triangle: scale all block contributions at that position by `λ_i`; introduce one new `relu_aux` tile generator with value `μ_i` localized to that single position. |
| MaxPool2D (kxk, stride s) | Compute `lb, ub` per pooling window. For each output position, identify the input position with the largest `ub` as the soft argmax. The output center is `max(ub) - max(ub)-c_input_argmax` (exact when stable). For unstable windows, the prototype uses a tile-preserving over-approximation: keep each candidate input position's block, scale it by 1 (no λ_max), and add a single aux generator equal to `max(ub) - max(lb)` localized to the output position. **TBD in Phase 0 design review.** |
| Flatten | Concatenate the (C, H, W) center into a flat vector and flatten each block into a sparse (n_flat,) column. The output is a `SparseGcZ`-style object that the existing tail LP can consume directly without further code changes on the dense side. |

### 4.3 Reduction policy

The whole point of ImageHZ_lite is **NOT to fire a global Girard cap
inside the conv body**. The intended behaviour is:

- Inside the conv body: keep all tiles independent. No reduction.
- Memory pressure is bounded by `tile_count × tile_size` rather than
  `n × ng`. With locality, the effective `ng` per tile is small.
- A reduction only fires at FLATTEN time, if at all, and operates on
  the flattened SparseGcZ.

If even tile-local memory pressure forces a cap inside the conv body,
the prototype fail-closes (UNKNOWN) rather than silently collapsing.

---

## 5. Phase 0 — Representation-only metrics (no verdict change)

Per advisor 2026-06-04: "第一阶段不要追 verdict，先追 representation
metric." This phase is a representation prototype that the verifier does
**not** invoke. It runs on a fixed set of sentinel iids and reports
representation statistics.

### 5.1 Sentinel iids

```text
no-lost baseline: iid 0 (FAL — must remain FAL or representation
                          parity must hold; never regress to ERROR)
correlation-loss targets: iids 1, 2, 3, 6, 9, 12, 13, 14
```

These iids span the per-iid Girard-fire variation observed in the
§6b reparse (some have only L11+L18+L25+L32+L35, some also L17+L29).

### 5.2 Hard representation gate (Phase 0 → Phase 1)

ALL of these must hold or Phase 0 closes negative and ImageHZ work stops:

1. `root_ng_at_flatten` improves by at least **10×** over the §6b
   baseline on the 8 correlation-loss sentinels. Baseline values are
   `{1, 5, 10, 20, 100}`; the prototype must hit at least `{10, 50,
   100, 200, 1000}` respectively.
2. At L32 MAXPOOL the post-op tile count carries **measurable
   root-factor provenance** (concretely: each output tile's
   `root_factor_ids` is non-empty for at least 50% of output
   positions on every sentinel).
3. At L35 RELU the same provenance survives the triangle aux gen
   (the new `relu_aux` tiles must not double the global aux generator
   count beyond a measured budget — TBD in design review).
4. No OOM. Per-iid wall budget is at most **2× the §6b baseline wall**
   on the same iid.
5. No crashes / no silent fallback. Every fail-closed event is logged
   with the operator and iid.

If any of these fail, the prototype closes negative and the project
moves to §10 stabilization + paper.

### 5.3 What Phase 0 does NOT do

- Phase 0 does NOT call the LP solver, the witness sidecar, or
  `verify_once_hz`. It is a representation-only experiment.
- Phase 0 does NOT touch CIFAR. Not the dispatcher, not the cli.py,
  not the production HZ path.
- Phase 0 does NOT emit FAL receipts; no receipt format changes here.
- Phase 0 does NOT add operator support beyond §3.

---

## 6. Phase 1 — V/A gate (only if Phase 0 representation gate passes)

Only enter Phase 1 if every condition in §5.2 holds.

### 6.1 V/A gate

```text
Run prototype + production LP tail on 20 VGG / Tiny sentinels.
Pass iff
    >= 1 new V/A (FALSIFIED or CERTIFIED) over the §6b baseline
    OR
    median LP/box margin improves by >= 30%
Fail iff
    0 V/A AND median margin movement < 10%
```

A pass moves to Phase 2 productionization (separate plan).
A fail closes the line definitively; project moves to §10 + paper.

### 6.2 Strict replay still required

Any FAL receipt the prototype produces in Phase 1 must:

- come from a structured HZ / LP program (no random, no PGD, no
  corner sampling per I6),
- pass raw ONNX strict replay (`input_box_holds`, `vnnlib_query_holds`,
  `spec_zero_tol_holds`),
- carry the provenance bundle (I9),
- and have its witness candidate fail-close if any tile / shape
  contract is violated (I10).

---

## 7. Phase 2 — Productionization (only if Phase 1 V/A gate passes)

Out of scope for this plan. Will be a separate task spec.
Sketch of expected work:
- routing integration in `hz_routing.py` or a new profile module,
- `ACT_HZ_IMAGEHZ_PROFILE=1` env gate,
- memory budget knobs,
- fail-closed export contract,
- full canonical regression sweep,
- documentation + receipt schema audit.

---

## 8. Hard "don't do" list

| ID | Don't |
|---|---|
| D1 | Don't extend ImageHZ to CIFAR. The production endcap sidecar owns CIFAR; atlas v3 closed that direction. |
| D2 | Don't add a "VGG tail LP" before Phase 0 representation gate passes. Tail LP without preserved correlation is phantom-heavy. |
| D3 | Don't add ResNet residual ADD until L25/L32 MAXPOOL2D pass Phase 0. ADD adds factor-aliasing complexity that must be designed separately. |
| D4 | Don't add operator support beyond §3 in Phase 0. Each new operator must be authorized in a subsequent revision of this doc. |
| D5 | Don't reuse the LOCAL `/data1/Kane/ACT/data/vnnlib` pool. `canonical_provenance.py` already fail-closes; keep it that way. |
| D6 | Don't quietly modify the existing endcap_witness sidecar to call ImageHZ. The witness path is locked until Phase 1 V/A gate fires PASS. |
| D7 | Don't optimize the cctsdb_yolo Slice frontend gap as part of this prototype. It is a deferred frontend cleanup, not a representation question. |

---

## 9. Open design questions (must resolve before Phase 0 coding starts)

1. **MaxPool2D semantics in §4.2 — exact bound construction.** The
   sketch in §4.2 picks the argmax of `ub` for the center and treats
   unstable windows with a single aux generator. Need to confirm this
   over-approximation matches the existing forward HZ MaxPool semantics
   (lower or equal in tightness; never tighter and never unsound).
2. **Tile shape policy.** Should tiles be fixed at the kernel-aligned
   block size (e.g. `2x2` or `3x3`) or dynamically per-layer? A fixed
   policy is simpler and easier to fail-close; dynamic gives better
   memory bounds at the cost of more bookkeeping.
3. **`relu_aux` generator budget.** When ReLU triangle introduces new
   per-position aux tiles, the aux count can grow up to `dim`. Need a
   policy: cap, fold by tile, or stream.
4. **Flatten-time export contract.** The flatten step must produce a
   SparseGcZ that the existing tail LP consumes without code changes
   on the dense side. Need to confirm the column-ordering and the
   `xi_id` numbering scheme match.

These four items must be resolved in a design review BEFORE any code
is written. The review's outcome will be appended to this file as
§9-resolved.

---

## 9-resolved. Design lock (2026-06-04 advisor review)

The four open items in §9 are decided as below. Phase 0 code MUST honor
these decisions; any change requires a new resolution entry, not a
silent edit.

### 9R-1. Tile data structure → `TileBlock`, not single generator

A tile is NOT a single (C,H,W) values tensor with an opaque
`root_factor_ids` list. A tile carries N independent generators that
share the same spatial footprint:

```text
TileBlock:
  origin_chw      : (c0, h0, w0)              # top-left in feature map
  shape           : (tc, th, tw)              # spatial footprint
  G_tile          : (n_gen_tile, tc, th, tw)  # n_gen_tile independent
                                              # generators stacked along
                                              # the leading axis (float64)
  factor_ids      : list[int]                 # len == n_gen_tile
                                              # one root-factor id per
                                              # generator column
  aux_meta        : dict
                    {
                      'kind'         : 'root' | 'relu_aux',
                      'spawn_layer'  : int,
                      'spawn_op'     : str,
                      'parent_block' : int | None,
                    }
```

Two contracts:

- `G_tile.ndim == 4` always; `G_tile.shape[0] == len(factor_ids)`.
- The concretization at `(c, h, w)` (with `c0 ≤ c < c0+tc`, etc.) is
  `Σ_k G_tile[k, c-c0, h-h0, w-w0] · ξ_{factor_ids[k]}`.

Multiple input pixels mapping to the same `factor_ids` list (e.g. after
a tile-aligned Girard merge) is allowed; multiple TileBlocks may carry
disjoint `factor_ids` lists for the same spatial footprint without
aliasing.

### 9R-2. MaxPool2D — sound-first semantics; no "tighter but unproven" hull

Per pooling window `W = {i_1, …, i_p}` with per-position pre-pool
bounds `(lb_i, ub_i)`:

```text
Stable case: exists m in W such that lb_m >= max_{i in W \ {m}} ub_i.
  Output value at this window position = x_m (the input value at m).
  The output TileBlock copies that input position's generators verbatim
  (with the same factor_ids). NO ReLU-style λ scaling, NO aux generator.
  This is exact: the max is provably the input at m for every
  realization of ξ in the box.

Unstable case: no stable winner.
  Pick the deterministic candidate m = argmax_{i in W} ub_i (ties
  broken by lowest i for reproducibility). The true max set is
      MAX_SET(W) = { max_{i in W} x_i  |  x_i in [lb_i, ub_i] }
                 ⊆ [ max_i lb_i,  max_i ub_i ].
  Encoding (this is a CONSERVATIVE over-approximation, NOT a tight
  interval; see "Soundness note" below):
    - Keep the m-position TileBlock so the output retains m's root
      provenance (and contributes a per-position radius equal to the
      sum over m's generators of |G_tile[k, m]|, which is at most ub_m
      − lb_m all by itself).
    - Add ONE new aux TileBlock with:
        kind         = 'relu_aux' (re-using the aux machinery)
        shape        = (1, 1, 1)
        G_tile       = (1, 1, 1, 1) with value D_m / 2,
                        D_m = max_{i in W} ub_i − lb_m  ≥ 0
        factor_ids   = [<fresh aux factor id>]
        aux_meta     = {'kind': 'relu_aux',
                        'spawn_layer': L,
                        'spawn_op': 'maxpool_unstable',
                        'parent_block': <m's parent block id>}
    - The output center at this position is set so that the encoded
      box contains lb_m and max ub_i: center = lb_m + D_m / 2.

  Soundness note (READ BEFORE EDITING THIS FILE):

  The encoded box is NOT exactly [lb_m, max ub_i]. Because the
  encoded position carries BOTH the m-position TileBlock contributions
  and the new aux generator, its per-position radius is the SUM of
  those two contributions, so the encoded interval is
        [ center − rad_total,  center + rad_total ]
  with rad_total = (D_m / 2) + sum_k |G_tile_m[k]|. This is at least
  as wide as [lb_m, max ub_i] and may be strictly wider when m's
  generators carry nontrivial mass. That extra width is the price of
  preserving m's root provenance and is what makes the encoding sound
  without a multi-candidate hull proof.

  We do not claim the encoded interval equals [lb_m, max ub_i]; we
  only claim
        encoded interval  ⊇  MAX_SET(W).
  This is what unit test `test_maxpool_unstable_containment`
  empirically checks, and what any future audit must verify.
```

This is a sound over-approximation: the output set under any ξ
contains every realizable maxpool output. We do not attempt a
multi-candidate convex hull in Phase 0 because we cannot prove it
sound on paper before coding.

Forbidden in Phase 0: any MaxPool implementation that "tightens" the
unstable case using more than one candidate path without a written,
peer-reviewed soundness proof. If proposed in a future revision, that
proof must be appended here BEFORE the code is written.

### 9R-3. ReLU aux generator budget — fail-closed, no silent fold

Per unstable ReLU position, the existing DeepZ triangle introduces
exactly one new aux generator. Phase 0 keeps this 1:1 mapping.

Budget rule:

```text
budget_relu_aux per ImageHZ_lite instance =
    max_relu_aux_per_image      # static cap, configurable per run

If, at any single ReLU layer, the number of newly-introduced aux
generators would exceed the remaining budget:
  raise ImageHZ_lite.BudgetExceeded  -> the prototype fail-closes
  (Phase 0 logs the event, the iid is recorded as "REPRESENTATION
  BUDGET EXCEEDED", and the verifier never sees an inconsistent state).
```

Forbidden: any "tile-fold" or "aux-merge" optimization in Phase 0 that
collapses two or more aux generators into a single shared column.
Merging would silently lose factor independence and produce a strictly
looser set without a paper-trail.

### 9R-4. Phase 0 Flatten — metrics + metadata only, no verifier link

The Flatten operator at Phase 0 produces a `Phase0FlattenSnapshot`:

```text
Phase0FlattenSnapshot:
  c_flat         : (n_flat,)               # flattened center
  blocks_meta    : list per TileBlock
                   {
                     'origin_chw'   : (c0, h0, w0),
                     'shape'        : (tc, th, tw),
                     'n_gen_tile'   : int,
                     'factor_ids'   : list[int],
                     'aux_kind'     : 'root' | 'relu_aux',
                     'spawn_layer'  : int,
                   }
  root_ng_at_flatten             : int
  total_aux_count                : int
  per_layer_girard_fires_observed: list[dict]
  peak_memory_bytes              : int
  wall_s                         : float
```

Phase 0 does NOT:
- emit a SparseGcZ or HZono,
- call `verify_once_hz`,
- call the witness sidecar,
- call any LP solver,
- write a FAL receipt.

The snapshot is consumed only by `run_vgg_phase0.py` to compute the
representation gate metrics. Phase 1 will define the actual export
contract to the existing tail LP; that contract is out of scope here.

### 9R-5. Structural gating — never benchmark-name

The prototype activation condition is structural, not name-based:

```text
ImageHZ_lite is invoked iff
    the network's prefix (everything before FLATTEN) contains only:
        Conv2D, ReLU, MaxPool2D
    AND
    a prior trace (from the standard production HZ path with
    ACT_HZ_LAYER_PROGRESS=1) shows at least one Girard cap fire at a
    MaxPool2D or ReLU operator on this iid.
```

Forbidden: `if benchmark_name == "vggnet16_2022"` anywhere in the code.
The gate is composed of:

- Operator scan over the model graph (a function on `pair.layers`).
- A trace-derived signal recorded as part of the iid's pre-run
  diagnostic (the §6b reparse output qualifies; for future iids it
  must be computed on demand).

Side effect: the gate auto-includes Tiny/VGG-like nets, auto-excludes
CIFAR (no MaxPool in resnet_medium / resnet_large), auto-excludes
ResNet (ADD not in the operator allow-list), auto-excludes LSNC and
soundnessbench (their conv prefixes also fail the operator check or
the trace signal). No per-dataset patch.

### 9R-6. Forbidden during Phase 0 (advisor 2026-06-04, restated)

- **No touch on CIFAR.** Atlas v3 closed CIFAR-ImageHZ.
- **No production routing integration.** `cli.py`, `hz_routing.py`,
  `verify_once_hz`, and the witness sidecar are untouched in Phase 0.
- **No V/A claims.** Phase 0 is representation-only.
- **No "tighter but unproven" hull on MaxPool unstable.** See 9R-2.
- **No silent aux fold.** See 9R-3.

### 9R-7. Phase 0 hard gate (revised, supersedes §5.2)

ALL of these must hold or Phase 0 closes negative and ImageHZ work stops:

1. On the 8 loss-target sentinels (1, 2, 3, 6, 9, 12, 13, 14):
   `root_ng_at_flatten` is at least **10×** the §6b baseline value on
   that iid. Baselines per §6b: iids 1→1, 2→1, 3→5, 6→10, 9→20, 12→100,
   13→100, 14→100. So the targets are at least 10, 10, 50, 100, 200,
   1000, 1000, 1000 respectively.
2. After L32 MaxPool2D, on every sentinel: at least **50% of output
   positions carry numeric root provenance** — that is, at least one
   TileBlock with `aux_kind == 'root'` and `len(factor_ids) > 0`
   overlaps the position. Metadata-only preservation does NOT count.
3. After L35 ReLU, `total_aux_count` does NOT exceed the configured
   `max_relu_aux_per_image` budget on any sentinel.
4. Per-iid wall time at most **2× the §6b baseline wall** on the same
   iid (for iid 0, baseline 99s → cap 198s; for iid 14, baseline 308s
   → cap 616s; etc.).
5. Zero OOM. Zero silent fallback. Every fail-closed event is logged
   with `(iid, layer, op, reason)`.

If any of (1) – (5) fails, Phase 0 closes negative. Project moves to
roadmap §10 (stabilize + paper).

### 9R-8. Code organization (Phase 0 only)

```text
research/imagehz_lite/
  __init__.py
  domain.py              # TileBlock + ImageHZ_lite container + invariants
  ops.py                 # apply_conv2d, apply_relu_triangle,
                         # apply_maxpool2d (sound-first), apply_flatten
  budget.py              # BudgetExceeded exception + counters
  run_vgg_phase0.py      # sentinel driver, no verifier link
  metrics.py             # representation metrics + gate eval
  tests/
    __init__.py
    test_imagehz_lite_ops.py   # toy unit tests, see §9R-9
```

No file outside `research/imagehz_lite/` is modified in Phase 0.

### 9R-9. Unit-test required matrix (Phase 0 entry condition)

Before Phase 0 runs on any VGG sentinel, these tests must pass:

| Test | What it covers |
|---|---|
| `test_conv2d_equiv_dense_hz` | A toy Conv2D on a single TileBlock matches the result of the existing dense HZono Conv2D on the same input. |
| `test_relu_triangle_soundness_small_random` | On 1000 random small boxes, the ImageHZ-lite ReLU output set (sampled at ξ extremes) is contained in the dense HZono ReLU output set. |
| `test_maxpool_stable_exact` | When all input positions have lb_i >= max ub_others, the output is exactly the chosen position's TileBlock with no aux. |
| `test_maxpool_unstable_containment` | When unstable, the output set (sampled at ξ extremes) is a superset of the true {max over input realizations}. |
| `test_flatten_column_order` | Flattening preserves a deterministic, reproducible column ordering. |
| `test_budget_fail_closed` | When ReLU aux count exceeds budget, `BudgetExceeded` is raised; no partial output is written. |
| `test_structural_gate_matches_only_conv_relu_maxpool_flatten` | The structural gate function returns True on a synthetic VGG-shape graph and False on a synthetic ResNet (with ADD), CIFAR (no MaxPool), or LSNC graph. |

These tests are part of `research/imagehz_lite/tests/`. They run with
pytest and are independent of any production module.

---

## 10. Audit trail

- Authorizing gate evidence:
  - `audit_results/vgg_l29_forensic_canonical_20260603T121532Z/` (forensic)
  - `audit_results/vgg_mini_atlas_canonical_20260603T142250Z/` (original mini-atlas)
  - `audit_results/vgg_mini_atlas_missing_rerun_iid2_20260604T020257Z/` (iid 2 rerun)
  - `audit_results/vgg_mini_atlas_missing_rerun_15to17_20260604T020448Z/` (15-17 reruns)
  - `audit_results/vgg_mini_atlas_canonical_plus_missing_20260604T023543Z/` (merged final)
- Related memos:
  - `vgg_l29_forensic_canonical_20260603`
  - `vgg_mini_atlas_20260604`
  - `atlas_v3_decision_20260603`
  - `canonical_root_recovery_20260603`
- This plan supersedes the original §7 "Task C: ImageHZ Prototype"
  paragraph in the roadmap. §7 should now point at this file as the
  canonical scope.
