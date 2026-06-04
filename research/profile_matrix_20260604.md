# ACT / HyZor Profile Matrix — Honest Disclosure (2026-06-04 night)

**Status**: written to forestall the reviewer question "is this a single
verifier or per-dataset hand-tuning?". The honest answer, grounded in the
actual code in `act/pipeline/cli.py`, is **a single forward-HZ trunk + a
profile portfolio** where some profile gates are structural and some are
still benchmark-name-scoped historical decisions.

This document is paper-ready and should be cited from
`research/paper_skeleton_20260604.md` Section 2 ("Principles") and
Section 3 ("Positive Results"). The matrix MUST be presented in the
paper; omitting it would let the reviewer assume something the code
does not support.

---

## 1. What's shared across every benchmark (the invariant trunk)

Every iid that produces a verdict — V, A, or UNKNOWN — runs the same
forward HZ verifier with the same fixed contract. No profile can break
any of these:

| Contract | Enforcement |
|---|---|
| Forward HZ propagation; no CROWN backward | code path `verify_once_hz` is the single entry into the verifier |
| ReLU `eq_lagr_v8` / DeepZ-triangle forward relaxation | shared kernel under `act/back_end/solver` |
| Continuous LP only (HiGHS); no MILP, no integer reasoning | HiGHS-only solver in `solver_hz` |
| No fallback verifier counted as an ACT/HyZor result | UNKNOWN stays UNKNOWN unless ACT/HyZor's own forward-HZ / LP path decides it |
| No BaB, no input splitting | only one verifier call per (iid, query); no recursive subdivision |
| No PGD / FGSM / CW / AutoAttack / random / corner falsification | all FAL candidates come from structured HZ + LP programs |
| FAL receipts must pass strict ORT zero-tolerance replay | `input_box_holds + vnnlib_query_holds + spec_zero_tol_holds` |
| Provenance bundle on every receipt | `canonical_root + instances_csv_sha256 + onnx_sha256 + vnnlib_sha256` |
| Canonical-root only; LOCAL-pool fail-closed | `research/canonical_provenance.py` |

These are project-level guarantees that no profile turns off.

---

## 2. The dispatcher — six profiles, four name-gated, two structural

`act/pipeline/cli.py` Lines 1090–1201 declare six profile flags. Each
either fires (`True`) or doesn't (`False`) per iid. Multiple may fire
simultaneously; the verifier then opens the corresponding env knobs
for the duration of that one iid (tracked in `_iid_env_restore`).

> **Table 1 — Profile dispatcher, ground truth from cli.py.**

| Profile flag | Gate type | Trigger | Env knobs opened | Effect on V/A | Sound? |
|---|---|---|---|---|---|
| `_small_dense_witness_profile` | **BENCHMARK-NAME** | `p['category'] in {linearizenn_2024, tllverifybench_2023, acasxu_2023, safenlp_2024, sat_relu, dist_shift_2023, malbeware, metaroom_2023}` | small-dense LP witness sidecar | enables FAL receipts under ORT strict replay | yes — receipt audited via standard contract |
| `_small_dense_dag_profile` | **BENCHMARK-NAME** | `p['category'] == "cersyve"` | small-dense DAG path | enables FAL receipts under ORT strict replay | yes |
| `_nn4sys_lindex_profile` | **BENCHMARK-NAME + SPEC-PATH** | `p['category'] == "nn4sys" AND "lindex" in vnnlib path` | `ACT_HZ_SMALL_DENSE_DIRECT_QUERY`, `ACT_HZ_SPECAWARE_BOUND_CACHE`, `ACT_HZ_STABLE_AFFINE_FASTPATH` | enables fastpath CERT for one-dim box specs that interval propagation already closes | yes — three knobs are fail-closed forward-LP checks; no BaB/sampling |
| `_cifar_endcap_profile` | **BENCHMARK-NAME + ENV OPT-OUT** | `p['category'] == "cifar100_2024" AND ACT_HZ_CIFAR_ENDCAP_PROFILE != 0` | `ACT_HZ_FACTOR_ID_SGM`, `ACT_HZ_ENDCAP_SNAPSHOT_DIR`, `ACT_HZ_ENDCAP_SNAPSHOT_KIND`, `ACT_HZ_CIFAR_ENDCAP_WITNESS` | enables L38 FLATTEN snapshot + factor-aware ADD + endcap LP witness sidecar | yes — receipt audited via standard contract |
| `_generic_mlp_endcap_profile` | **STRUCTURAL** | `supports_generic_mlp_endcap()` — tail shape ∈ {Dense, Dense+Dense, Dense+ReLU+Dense} after FLATTEN, top-1 robust spec, CIFAR-narrow excluded | (re-uses CIFAR endcap machinery on snapshot for other top-1 robust nets) | enables endcap LP witness on non-CIFAR top-1 robust MLPs | yes |
| `_residual_sparse_conv_profile` | **STRUCTURAL** | `_n_conv >= 6 AND _n_add >= 2 AND not has_dense_tail AND _out_dim >= 1024 AND not _cifar_endcap_profile AND not _generic_mlp_endcap_profile` | `ACT_HZ_RESIDUAL_SPARSE_PROFILE` | enables exact factor-aware ADD + sparse pre-conv materialisation for detector-style residual conv nets | yes |

### 2.1 What the structural gates check (concretely)

- **`_generic_mlp_endcap_profile`** (`act/back_end/profiles/generic_mlp_endcap_gate.py`):
  - the layer sequence after the first FLATTEN is one of:
    - `[Dense]`
    - `[Dense, Dense]`
    - `[Dense, ReLU, Dense]`
  - the model has a `labeled_tensor` in the vnnlib pair (top-1 robust)
  - CIFAR endcap is NOT active for this iid
  - covered by 14/14 cases in `tests/test_generic_mlp_endcap_gate.py`
- **`_residual_sparse_conv_profile`**: ≥6 Conv layers, ≥2 Add layers, no
  Dense tail, output dim ≥ 1024, and the two named-gate profiles are off.
  This is detector-style residual nets (e.g. some YOLO variants).

### 2.2 What benchmark-name gates check (concretely)

`p['category']` is the VNN-COMP-2025 directory name (e.g.
`"cifar100_2024"`). The gate is a string membership test, not a graph
property. **This is the part of the dispatcher that is honestly named-
gated** and is what we must disclose.

---

## 3. The 5 benchmarks in the canonical sweep (§9), traced to their profiles

> **Table 2 — Profile assignments for the 253-V/A canonical sweep.**

| Benchmark | Profile that fired | Gate type | V | A | UNK | ERR | Counts toward 253? |
|---|---|---|---|---|---|---|---|
| cifar100_2024 | `_cifar_endcap_profile` | **NAME** (env opt-out) | 0 | 15 | 185 | 0 | yes |
| tinyimagenet_2024 | `_generic_mlp_endcap_profile` | **STRUCTURAL** | 1 | 34 | 165 | 0 | yes |
| cctsdb_yolo_2023 | (no profile; parser raised) | n/a | 0 | 0 | 0 | 39 | no — parser gap |
| nn4sys | `_nn4sys_lindex_profile` | **NAME + SPEC-PATH** | 101 | 0 | 93 | 0 | yes |
| malbeware | `_small_dense_witness_profile` | **NAME** | 89 | 13 | 51 | 0 | yes |
| **Totals** | — | — | **191** | **62** | **494** | **39** | **253 V/A counted** |

So in the §9 sweep, **3 of the 4 V/A-producing profile branches are
benchmark-name-gated**:

- `_cifar_endcap_profile` (cifar100_2024 → 15 A)
- `_nn4sys_lindex_profile` (nn4sys → 101 V)
- `_small_dense_witness_profile` (malbeware → 89 V + 13 A = 102)

One is structural:

- `_generic_mlp_endcap_profile` (tinyimagenet_2024 → 1 V + 34 A = 35)

cctsdb_yolo_2023 is parser-blocked and contributes 0 V/A.

### 3.1 Soundness of named-gated profiles

Although three of the V/A-producing profiles are benchmark-name-gated,
each selected profile is a fixed forward-HZ / LP procedure and each
receipt it emits still satisfies the project's standard soundness
contract:

1. Forward HZ → forward LP, no backward step.
2. FAL candidates come from structured LP programs only.
3. FAL receipts pass strict ORT zero-tolerance replay
   (62 / 62 receipts validated in the 2026-06-04 evening audit, see
   `paper_skeleton_20260604.md` Section 3.2 / Table 3).
4. CERT outputs come from a sound forward LP UB on the rival margin.
5. Provenance bundle attached.

The name-gating affects **WHICH** profile is invoked. It does not create
ad hoc per-benchmark exceptions inside a profile, and it does not relax
the receipt requirements.

---

## 4. The honest paper statement

The paper SHOULD NOT claim:

> "ACT/HyZor runs identically on every benchmark with no per-dataset
> configuration."

The paper SHOULD say:

> ACT/HyZor is a forward-only HZ verification portfolio. All profiles
> obey the same principle set (P1-P5, Section 2 of the paper skeleton)
> and the same receipt contract. Profile selection is currently mixed:
> two of the six dispatched profiles are structurally gated
> (`_generic_mlp_endcap_profile`, `_residual_sparse_conv_profile`),
> while four use the benchmark-directory name as the gate
> (`_cifar_endcap_profile`, `_nn4sys_lindex_profile`,
> `_small_dense_witness_profile`, `_small_dense_dag_profile`). The 253
> V/A contributions in the canonical sweep come from three
> name-gated profiles (CIFAR endcap +15A, nn4sys lindex +101V,
> malbeware small-dense +89V+13A) and one structural profile
> (Tiny generic MLP endcap +1V+34A). Each profile's receipts are
> independently soundness-checked via strict ORT zero-tolerance
> replay. The name-gating affects which fixed profile fires; it does
> not permit benchmark-specific exceptions to that profile's soundness
> contract.

This is the truthful framing. It does not hide the name-gating, and it
does not overclaim what the structural gates already cover.

---

## 5. Migration list — which benchmark-name gates can become structural

For the next paper revision, three of the four name-gated profiles are
candidates for migration to structural gates without changing what
they verify:

| Profile | Migration candidate? | Sketch |
|---|---|---|
| `_cifar_endcap_profile` | **YES** | The mechanism is "FLATTEN snapshot + factor-aware ADD + endcap LP witness". The structural conditions are: ResNet conv body (≥ N residual Adds), Dense tail of length 1-3 ending in `out_dim = n_classes`, top-1 robust spec. Replacing `p['category'] == "cifar100_2024"` with these conditions reproduces the behavior on CIFAR and naturally extends it to other CIFAR-shape benchmarks. |
| `_small_dense_witness_profile` | **PARTIALLY** | Some of the 8 named benchmarks share the structural pattern "small input box (≤ 1024 dims), small MLP body (≤ 8 Dense layers), top-1 / OR-disjunct spec". For acasxu, safenlp, sat_relu, dist_shift this would generalize cleanly. For metaroom and malbeware the gate would need additional conditions (e.g. ADD presence). For linearizenn / tllverifybench, structural gates already overlap with `_generic_mlp_endcap_profile`; some of the small-dense work is already covered there. |
| `_nn4sys_lindex_profile` | **NO (not without expansion)** | The "lindex" string in the vnnlib path encodes the spec being a one-dim box with thousands of independent UNSAFE_LINEAR rows. This is a spec-shape property, not a graph property. A structural gate would require parsing the vnnlib up front and detecting that shape — viable but more code than the others. |
| `_small_dense_dag_profile` | **YES** | The cersyve gate is just `p['category'] == "cersyve"`. The DAG-profile mechanism is structural (sparse DAG verifier path); the gate condition can be replaced with "small input + small layer count + DAG shape". |

**This migration is paper-revision work, not improvement-phase work.**
It does not change any V/A number. The migration is a soundness /
generality claim, not a precision lever.

---

## 6. What this matrix does NOT touch

- It does NOT modify the 253 V/A headline.
- It does NOT re-open ImageHZ for CIFAR or VGG (closed in §6b / §6c /
  §7).
- It does NOT change any FAL receipt contract.
- It does NOT relax any principle in Section 2 of the paper skeleton.

Its purpose is to be the **honest disclosure document** that the paper
cites when answering "what runs on which benchmark, and is it really
one method?".

---

## 7. Audit trail

- Code source for the dispatcher: `act/pipeline/cli.py` lines 1090–1201.
- Structural gate test coverage: `tests/test_generic_mlp_endcap_gate.py`
  (14/14 cases).
- Per-benchmark V/A sources: `audit_results/clean_canonical_combined_summary_20260604.json`.
- Soundness audit (62/62 FAL provenance): `paper_skeleton_20260604.md`
  Section 3.2 / Table 3.
- Memory entries cited by named profiles:
  `project_safenlp_b14_frozen`, `project_sat_relu_reopened_20260601`,
  `project_audit_final_consolidated_20260601`,
  `project_canonical_main_20260417`.
