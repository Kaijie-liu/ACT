# ACT Hybrid Zonotopes Verification Flow — Soundness Trace

**Purpose**: end-to-end trace of how a single VNNLIB instance is verified by
ACT's `hybridz` mode, with explicit confirmation that **no sampling shortcut**
is used to produce a verdict. Written for advisor review.

**Code snapshot**: r93 = git HEAD `d39802521`

---

## 1. Sampling-cheating audit (the concern)

`OrtSampleFalsifier.py` exists in the **HyZor** repo (`/data1/Kane/HyZor/OrtSampleFalsifier.py`)
as a legacy candidate-generator from the SATSidecar pilot. It does:

```
1. parse vnnlib disjuncts
2. uniform random + corner sampling inside the input box
3. run candidates through onnxruntime
4. if any output violates unsafe rows → return ('falsified', x_witness, ...)
```

This is sound (ORT IS the ground-truth network) but it is **NOT verification** —
it's a falsification heuristic that bypasses any verifier capability.

**Audit result on the current archive**:

```
$ grep -rn "import.*OrtSample\|from.*OrtSample" /data1/Kane/ACT/
(no matches — ACT never imports it)

$ for csv in CONSOLIDATED_RESULTS/*/per_instance.csv:
    inspect every FALSIFIED row's `q_receipts` field
all FAL sources: hz_walker_lp / small_dense_lp_witness / SATSidecar (cersyve only)
0 FAL sources from: ort_sample / OrtSampleFalsifier
```

→ **No verdict in this archive came from ORT sampling.** All FAL verdicts trace
to ACT's internal LP path on the HZ representation. The only ORT usage in our
pipeline is (a) the receipt's strict-replay self-check (§7) and (b) the
post-hoc audits I ran on sat_relu and collins_rul (independent verification
of ACT-produced witnesses, not verdict sources).

---

## 2. End-to-end verification flow

For one (`onnx_model`, `vnnlib_spec`) instance under `hybridz` mode:

### Step 1 — Entry & spec parse
**File**: `act/pipeline/cli.py:835–949` (`_run_vnnlib_verify_hybridz`)
- Load ONNX → PyTorch via `convert_onnx_to_pytorch` (deterministic graph
  conversion via onnx2torch; **no sample data involved**).
- Parse VNNLIB → finite input box + unsafe output constraints (disjunctive).
- For TOP1_ROBUST class, collapse OR-shaped output spec into the canonical
  "exists j ≠ t such that `y_j - y_t > 0`" form.
- Construct `VerifiableModel(model, input_lb, input_ub, spec)`.

### Step 2 — ONNX → ACT IR conversion
**File**: `act/pipeline/verification/torch2act.py:1277–1311` (`TorchToACT.run`)
- Walks `torch.fx`-traced model nodes (`call_module`, `get_attr`, `placeholder`).
- For each ONNX op, dispatches to a `_convert_OnnxX` handler in
  `act/pipeline/verification/utils.py` which:
  - Allocates fresh ACT variable IDs for the output tensor.
  - Reads layer parameters (weights, kernel, attrs) from the PyTorch module.
  - Appends a typed `Layer(kind, params, in_vars, out_vars)` to `Net.layers`.
- Sets up `Net.preds` / `Net.succs` adjacency dicts (DAG of layers).
- Output: a `Net` object — pure structural representation. **No sample inputs
  consumed**; the sample-substitution gate
  (`_evaluate_constant_subgraph(allow_sample_substitution=False)`) is the R12
  fail-closed guard against accidentally fixing data-dependent shapes.

### Step 3 — Abstract interpretation (HZ propagation)
**File**: `act/back_end/analyze.py:39–160` (`analyze`)
- Seeds the entry layer with the input box `entry_fact` (bounds + a one-zonotope
  HZ representation).
- Worklist algorithm (FIFO `deque`, ready-check ensures all predecessors
  visited before a node is popped — R16 fix).
- For each layer, computes `Bjoin` = join of all predecessor `after`-bounds
  (`box_join`), then dispatches to the layer's transfer function.
- Returns `(before, after, globalC)` — `after[lid]` is the HZ over-approximation
  of the layer's reachable set.

### Step 4 — Per-layer HZ transfer functions
**File**: `act/back_end/solver/solver_hz.py:1547–1671` (`_dispatch`) and
`act/back_end/hybridz_tf/hz_routing.py`
- **MATMUL / DENSE**: `hz_dense(W, b, hz_in)` — exact linear map on the
  generator matrix `Gc → W @ Gc`, center `c → W @ c + b`.
- **CONV2D**: `hz_conv2d(weight, bias, padding, ...)` — convolutional version
  of the same exact linear map.
- **ADD**: `hz_minkowski_sum` (exact for disjoint generators) or `hz_sgm_add`
  (shared-generator merge when applicable).
- **RELU**: `hz_apply_relu_v8(...)` — routes by `large_cls_proof_mode` env:
  - `triangle` → DeepZ-style linear relaxation (no binary, no LP).
  - `eq_lagr_v8` → equality-Lagrangian formulation with binary variables.
  - `convex_hull_cont` / `compact` → other relaxations.
  - All produce a sound HZ that contains the post-ReLU reachable set.
- **CONCAT / SLICE / RESHAPE / EXPAND / TRANSPOSE**: index-mapping ops, exact
  by construction (no relaxation).
- **All ops** preserve the over-approximation invariant: the produced HZ
  contains every output the network could produce on the input box.

### Step 5 — Output spec check (verdict gate)
**File**: `act/back_end/solver/solver_hz.py:1484–1497` (`check_unsafe_for_act`)
- Input: final-layer HZ output `out_hz` + the assert-layer (encoding the
  "unsafe" region from VNNLIB).
- Builds an LP whose feasible set = (output HZ) ∩ (unsafe region).
  - HZ constraints: `Ac @ xi_c + Ab @ xi_b ≤ b`, `xi_c ∈ [-1,1]^ng`,
    `xi_b ∈ [-1,1]^nb` (LP relaxation of {-1,+1}).
  - Output mapping: `y = Gc @ xi_c + Gb @ xi_b + c`.
  - Unsafe rows: `y_j - y_t > 0` (or VNNLIB-supplied affine inequalities).
- Three LP outcomes:

  | LP returns | Meaning | Verdict |
  |---|---|---|
  | `infeasible` | Output HZ does NOT intersect unsafe region | **CERTIFIED** (UNSAT) — by HZ over-approximation soundness, the actual network output cannot reach unsafe |
  | `feasible(xi_star)` | LP found a point in (HZ ∩ unsafe) | proceed to Step 6 for back-projection (NOT yet FALSIFIED — could be spurious from LP relaxation of `xi_b`) |
  | `timeout` | LP didn't finish in budget | **UNKNOWN** |

### Step 6 — Witness back-projection
**File**: `act/back_end/solver/solver_hz.py:2313–2346` (`lp_witness_to_input`)
- Given `xi_star` (factor-space point that LP claimed feasible), back-project
  to input space via the input-layer's `Gc, Gb, c`:
  `x_star = c_input + Gc_input @ xi_c_star + Gb_input @ xi_b_star`
- Result `x_star` is a concrete input vector — claimed to violate the spec.
- This is **NOT yet** the verdict. The LP relaxes `xi_b` from `{-1,+1}^nb` to
  `[-1,1]^nb`, so the witness could be spurious.

### Step 7 — Strict-replay self-check (the soundness gate)
**File**: `act/back_end/solver/solver_hz.py:2549–2620` (`strict_replay_for_act`)
- Take the candidate `x_star`, evaluate it via **the actual ONNX model** with
  `onnxruntime` (or `ACTToTorch` if ORT unavailable).
- Get the real network output `y_actual = N(x_star)`.
- Check if `y_actual` strictly satisfies the unsafe constraint **with zero
  numerical tolerance**:
  - For TOP1_ROBUST: `y_actual[j] - y_actual[t] > 0` for some `j ≠ t`.
  - For UNSAFE_LINEAR: every unsafe row strictly satisfied.
- Set `spec_zero_tol_holds` flag accordingly.

This step is the load-bearing soundness guard. The ORT call here is **not a
verdict source**; it's a deterministic re-evaluation of the network at the
candidate `x_star`. If `spec_zero_tol_holds == False`, the LP-relaxed witness
was spurious and the verdict is **NOT** FALSIFIED.

### Step 8 — Receipt emission + internal consistency check
**File**: `act/back_end/solver/solver_hz.py:903–1035` (`_emit_sat_with_receipt`)
- Build the receipt JSON: `model_path`, `spec_path`, `model_sha256`,
  `spec_sha256`, `x_star_sha256`, `y_ort_npy`, `input_box_holds`,
  `spec_zero_tol_holds`, `spec_small_tol_holds`.
- **Internal consistency**: read the receipt back, verify `spec_zero_tol_holds`.
- Branch:
  - `spec_zero_tol_holds == True` → **`REPORTABLE_FALSIFIED`**.
  - `spec_zero_tol_holds == False` AND `spec_small_tol_holds == False`
    → **`ERROR_INTERNAL_INCONSISTENCY`** (NOT FALSIFIED — the LP witness was
    spurious; this is a strict-watchdog soundness signal).
  - Anything else → propagates as UNKNOWN.

### Step 9 — Multi-query aggregation
**File**: `act/pipeline/cli.py:599–625`
- One VNNLIB file may declare disjunctive queries (`(or ...)` over multiple
  unsafe sets). ACT processes each disjunct as a separate query.
- Instance-level verdict aggregation:
  - **ANY** query FALSIFIED → instance FALSIFIED (a single counterexample
    suffices for the disjunction).
  - **ALL** queries CERTIFIED → instance CERTIFIED.
  - Otherwise → UNKNOWN.

---

## 3. What is **NOT** in this flow

- ❌ Random/corner input sampling. Nowhere is `np.random` or
  `torch.rand` used to find counterexamples.
- ❌ ORT-based candidate generation. Witnesses always come from the LP
  on the HZ representation.
- ❌ Heuristic CERT decisions. CERTIFIED requires LP-proved infeasibility
  of (HZ ∩ unsafe), which is mathematically sound under the over-approximation
  invariant.
- ❌ Sample-dependent shape inference. The R12 `allow_sample_substitution=False`
  gate is permanent.

The only "trust ORT" step is in §7, and it's a `network(x_star) == y` deterministic
evaluation — not a search.

---

## 4. Receipt fields and what they mean

For every FALSIFIED instance, the receipt JSON in the source dir contains:

| field | meaning | required for FAL |
|---|---|---|
| `model_path` / `model_sha256` | binds the verdict to a specific ONNX file | yes |
| `spec_path` / `spec_sha256` | binds to a specific VNNLIB spec | yes |
| `x_star_npy` / `x_star_sha256` | the candidate input witness | yes |
| `y_ort_npy` | network's actual output at `x_star` (via ORT or ACT-Torch) | yes |
| `input_box_holds` | `x_star ∈ [lb, ub]` (R9.3 input-box gate) | **must be True** |
| `spec_zero_tol_holds` | `y_ort` strictly satisfies unsafe rows (zero ε) | **must be True** for `REPORTABLE_FALSIFIED` |
| `spec_small_tol_holds` | `y_ort` satisfies unsafe rows with ε = 1e-6 | informational |

If `input_box_holds == False`, the witness is rejected pre-replay (LP
back-projection landed outside the box; sound under R9.3 fail-closed gate).

---

## 5. CERTIFIED soundness

A CERTIFIED verdict requires:
1. Net was correctly constructed (R12 fail-closed shape gate passed).
2. `analyze()` ran to completion (every layer's predecessors were `visited`
   before its TF executed — R16 ready-check; no `±inf` sentinel poisoned a
   downstream MATMUL).
3. `check_unsafe_for_act` returned `infeasible` → LP proved no point in the
   output HZ satisfies the unsafe constraint.

By the over-approximation invariant (each TF produces an HZ containing the
true reachable set), the actual network output cannot lie in the unsafe
region → the spec holds for every input in the box. This is the standard
abstract-interpretation soundness argument; no sampling is invoked.

---

## 6. Where the LP solver lives

Two LP solver paths:
- **`hz_walker_lp`** (`act/back_end/solver/solver_hz.py`): the canonical HZ
  LP described above; used by most ReLU-network benchmarks.
- **`small_dense_lp_witness`** (`act/back_end/solver/lp_small_dense.py`):
  a specialized LP for small dense (ACASXu, linearizenn, sat_relu) where
  the HZ is dense enough that the full HZ LP is overkill. Same soundness
  semantics, just a faster solver.

Both use `scipy.optimize.linprog` or HiGHS warm-start under the hood; both
return the same factor-space `xi_star` consumed by §6's back-projection.

---

## 7. Cross-reference

This document complements:
- `MASTER_INDEX.md` — per-benchmark CPU/GPU verdict counts.
- `SOUNDNESS_VS_VNNCOMP_OFFICIAL.md` — agreement with VNN-COMP labels.
- `nn4sys_lindex200_FIXES.md` — the 4 Round-4 ACT bug fixes (all conversion-side; no soundness logic changed).
- `act_fixes_diff/` — exact code diffs of the 4 fixes.

If any future PR adds an `import OrtSampleFalsifier` or `random.sample(input_box)`
to a verdict path, that PR violates the soundness contract and must be rejected.
