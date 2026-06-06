# Phase E Day 1 — Conv Chunked Propagation (S2 infrastructure)

**Date**: 2026-06-05 late-night
**Status**: Day 1 deliverable COMPLETE — `conv_chunked.py` + 12 unit tests
**Headline impact**: NONE (1472 unchanged; this is infrastructure, no V/A)

Per advisor 2026-06-05 path-A plan: build chunked-Conv memory infrastructure
so cifar deep variants and tinyimagenet can be measured. NO score-chasing
in this Day; only memory correctness + numerical parity.

---

## 1. Day 1 deliverables

### `research/sc_hz/conv_chunked.py`

NEW module with three public APIs:

- `apply_conv2d_chunked(state, W, b, input_shape, stride, padding, groups, chunk_size)`
   — drop-in replacement for `apply_conv2d` that processes generator columns
   in chunks of `chunk_size` (default 256). Bit-equal output to dense version.
- `estimate_chunk_memory(input_shape, output_shape, chunk_size)` — analytic
   transient-memory estimate per Conv call.
- `adaptive_chunk_size(input_shape, output_shape, budget_gb, min_chunk, max_chunk)`
   — pick chunk_size against a GB budget.

Key implementation detail:

```python
for start in range(0, K, chunk_size):
    end = min(start + chunk_size, K)
    chunk_input = state.G_kept[:, start:end].T.reshape(cs, Ci, Hi, Wi)
    chunk_t = torch.from_numpy(chunk_input).to(torch.float64)
    chunk_out = F.conv2d(chunk_t, W_t, None, stride=stride,
                          padding=padding, groups=groups)
    new_G[:, start:end] = chunk_out.detach().numpy().reshape(cs, n_out).T
    del chunk_t, chunk_out  # immediate free
```

Memory profile per chunk: `chunk_size × (C_in × H_in × W_in + C_out × H_out × W_out) × 8` bytes.

### `research/sc_hz/tests/test_conv_chunked_parity.py`

12 unit tests, all PASS:

| Test | Check |
|---|---|
| `test_chunked_equals_dense_basic` | Bit-equal c & G_kept output |
| `test_chunked_equals_dense_no_bias` | Same without bias |
| `test_chunk_sizes_match` (1, 2, 4, 5, 8, 16, 32, K, K+5) | chunk_size has zero effect on result |
| `test_stride_2` | stride=2 matches dense |
| `test_padding_0` | padding=0 matches dense |
| `test_grouped_conv` (groups=4) | grouped conv matches dense |
| `test_tail_matches` | tail_radius via `|W| @ tail` identical |
| `test_metadata_preserved` | `input_coord_origin` lineage carried unchanged |
| `test_estimate_chunk_memory_basic` | byte counts correct |
| `test_adaptive_chunk_size_budget` | within [min, max] |
| `test_adaptive_chunk_size_tight_budget` | floors to min_chunk=16 |
| `test_adaptive_chunk_size_loose_budget` | caps at max_chunk=1024 |

Suite total: **40/40 tests pass** (28 prior + 12 new).

---

## 2. Memory profile on real shapes

Chunked Conv transient memory at chunk_size=1024 on representative
cifar100/tinyimagenet conv layers:

| Layer | Input shape | Output shape | Transient |
|---|---|---|---|
| cifar L0 first Conv | (3, 32, 32) | (64, 32, 32) | 0.52 GB |
| cifar L8 stride-2 | (64, 32, 32) | (128, 16, 16) | 0.75 GB |
| cifar L16 stride-2 | (128, 16, 16) | (256, 8, 8) | 0.38 GB |
| cifar L24 stride-2 | (256, 8, 8) | (512, 4, 4) | 0.19 GB |
| tiny L0 | (3, 56, 56) | (64, 56, 56) | 1.60 GB |
| tiny L8 stride-2 | (64, 56, 56) | (128, 28, 28) | 2.30 GB |
| tiny L16 stride-2 | (128, 28, 28) | (256, 14, 14) | 1.15 GB |
| tiny L24 stride-2 | (256, 14, 14) | (512, 7, 7) | 0.57 GB |

vs the Day-of pilot peak of ~70-80 GB at the dense (full K materialize)
path. The chunked path's transient is bounded by chunk_size, not by ng.

**The remaining memory term** is the STATE STORAGE between layers:
`(n_out × K)` for each in-flight state in the value-DAG dict. This is
NOT reduced by chunked Conv. To address it, Day 2-3 will need to add
strategic PRUNE between residual branches (already supported via the
existing `prune()` with `incoming_tail_radius` fix from 2026-06-04).

---

## 3. What chunked Conv does NOT yet do (Day 2-3 work)

- Integration into `onnx_walker_resnet.py` (the `Conv` branch still
   calls `apply_conv2d`)
- Adaptive chunk_size selection based on free-RAM (not just shape)
- Per-residual-branch state eviction (release intermediate states once
   their downstream Add is consumed)
- 40-sentinel memory gate run (the actual S2 acceptance criterion)

These are Day 2-3 deliverables per the Phase E roadmap.

---

## 4. Day 1 gates checklist

| Gate | Status |
|---|---|
| Numerical parity vs dense apply_conv2d | 12/12 tests confirm bit-equal |
| chunk_size has no effect on output | tested for 9 different chunk sizes |
| Stride / padding / groups handled correctly | individual tests pass |
| Tail radius propagation soundness | tested vs dense |
| Metadata (origin) lineage preserved | tested |
| Memory estimator returns sensible values | tested |
| Suite did not regress (G1, G6 unchanged) | 40/40 tests pass |
| G10 enforced in production driver | applied at S2 driver level (Day 2) |

---

## 5. Phase E Day-10 kill switch — still binding

Per advisor: "如果 S2 两周内仍然 0 NEW + LP excess 降不动, 就关闭 SC-HZ 提升阶段".

Day 1 contribution to that gate: NONE (memory infrastructure only).
Day 2-3 will produce the first memory-gate data on the 40 sentinel set.
Day 4-7 will measure LP excess on cifar PHANTOMs under chunked + (maybe)
tighter final-tail relaxation. Day 7 milestone: ≥30% LP excess reduction
on 8 cifar PHANTOMs.

---

## 6. Files

| File | Status |
|---|---|
| `research/sc_hz/conv_chunked.py` | NEW (3 public APIs, ~180 lines) |
| `research/sc_hz/tests/test_conv_chunked_parity.py` | NEW (12 tests, all pass) |
| Unit test suite total | 40/40 pass |
| `act/` | UNCHANGED (`git diff --stat -- act/` empty) |
| 1472 freeze | UNCHANGED |

---

## 7. Honest framing

This is an infrastructure-only day. The chunked Conv module does not
itself produce headline lift, but it removes a hard memory ceiling that
blocked tinyimagenet + cifar deep-variant pilots. The S2 path forward
now has:

- Day 1 ✅ Chunked Conv with parity tests
- Day 2-3 (next): integrate + 40-sentinel memory gate
- Day 4-7: cifar PHANTOM LP excess reduction via final-tail relaxation
- Day 10 gate: ≥30% excess drop or close SC-HZ phase

The 1472 V/A headline stands regardless of S2 outcome.
