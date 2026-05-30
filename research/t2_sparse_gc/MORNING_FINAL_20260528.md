# 早晨工作 — B3 compact mode v2 LP-redundancy 发现 (2026-05-28)

**起点**: 用户问 "Sparse PEE 会让效果更好吗?" 我重新审视后发现自己昨晚的 "Sparse PEE" 描述不够准确, 改做 diagnostic-first 路径。

## Phase 0: Rank diagnostic 发现真正机会

跑了 `b3_ac_rank_small.py` (3 层 toy MLP), 测量 B3 累积 Ac 的 numerical rank:

| Layer | row_count | rank | rank/row |
|---|---|---|---|
| L1 | 1,536 | 320 | **0.21** |
| L2 | 4,608 | 832 | **0.18** |

**rank == ng (full column rank), but row_count = 5×ng**. 意味着大量行 linearly dependent. 启发了 LP-redundancy 检查.

## Phase 1: LP-redundancy proof (`verify_redundancy_multi.py`)

用 scipy linprog 测试 B3 加的 6 rows per neuron, 对每行问: "把这一行从 LP 移除, max LHS 还会突破 RHS 吗?" 如果不会, 这行 LP-redundant.

结果 across 5 random α/β trials + 2-neuron + 2-layer chain: **rows blk2, blk4, blk6 (the "−xi_j ≤ 1" 负号形式) 总是 LP-redundant**, 因为它们 implied by 变量 box constraints (xi_c, xi_b ∈ [-1, 1]; z ∈ [-1, 1] LP-relax).

**直觉**: LP solver 自己 enforce variable boxes. blk2 (-xi1 ≤ 1) 等价于 xi1 ≥ -1 — 但 xi1 已被代换为 (xi_old, xi_b, xi2, z) 的线性组合, 而每个原始变量的 box 已经 enforce xi1 不超出 [-1, 1] 的可能范围. 所以 blk2 行不切割任何 LP feasible region.

## Phase 2: Fix `compact_rows` mode

之前我的 `compact_rows=True` drop blk3, blk4 (错误选择 — drop 了 1 needed + 1 redundant).

修正后: drop blk2, blk4, blk6 (正确 — drop 全部 3 个 LP-redundant). **3 rows per neuron, 50% reduction, sound, ZERO precision loss**.

## Phase 3: Soundness gates

- **6/6 unit tests PASS** (含修正后的 compact test)
- **regression pack 8/8 PASS** with `ACT_HZ_SPARSE_EQ_LAGR=1 ACT_HZ_SPARSE_EQ_LAGR_COMPACT=1`
- **Bounds-equivalence proven (`verify_bounds_equivalence.py`)**: 8/8 seeds, **max LP-bound difference = 0.0000e+00**. 不是 "compact 是 over-approximation 包含 full" — 是 "compact LP bounds 等于 full LP bounds".

## Phase 4: 实测收益 (tinyimagenet 5-iid)

| iid | RSS (v2 3-row compact) | Verdict |
|---|---|---|
| 0 | 9,523 MiB | UNK_TIMEOUT |
| 40 | 10,012 | UNK_TIMEOUT |
| 80 | 9,582 | UNK_TIMEOUT |
| 120 | 9,109 | UNK_TIMEOUT |
| 160 | 9,172 | UNK_TIMEOUT |
| **mean** | **9,480** | 5/5 no OOM |

vs 昨晚 sweep 数据:

| Mode | Mean RSS | RSS vs baseline | Precision change |
|---|---|---|---|
| t2b_only baseline | 16,725 | — | — |
| B3 v1 full (6 rows) | 14,399 | -14% | — |
| B3 v1 compact OLD (4 rows wrong) | 15,021 | -10% | lossy! |
| **B3 v2 compact NEW (3 rows correct)** | **9,480** | **-43%** | **ZERO loss** ⭐ |

**关键提升: 老 compact (-10%) → 新 compact (-43%)**. 即比 v1 full (-14%) 还多 33% reduction.

## Phase 5: cifar100 resnet_large 仍 OOM (但已尽力)

iid 100 with k_max=4096 + new compact: OOM at 25,384 MiB (38s).

iid 100 with k_max=100 (强限制) + compact: UNKNOWN @ 291s, 23,577 MiB (under cap, no OOM, no precision lift vs baseline).

cifar100 resnet_large 的 OOM 不是仅 Ac 主导 — 还有 LP solver intermediate + cumulative Gc 等. 即使 Ac 减半也不够. 需要真正的 sparse PEE 跨层压缩或不同方向.

## 设计原则合规

5/5 全部满足:
- No CROWN ✓
- No backward ✓
- No Gurobi (scipy linprog/HiGHS only) ✓
- No fallback (representation flavor switching internal) ✓
- No B&B backtracking ✓

## 文件改动 (今早)

| 文件 | 改动 |
|---|---|
| `act/back_end/hybridz_tf/algorithms/sparse_eq_lagr.py` | 修 `compact_rows` 逻辑: blocks=[blk1, blk3, blk5] (drop blk2/4/6) |
| `act/back_end/hybridz_tf/sparse_gc_t2.py` | 更新 `act_sparse_eq_lagr_compact_rows` docstring |
| `tests/test_sparse_eq_lagr.py` | 更新 compact test 期望 50% reduction |
| `research/t2_sparse_gc/b3_ac_rank_scan.py` | NEW diagnostic |
| `research/t2_sparse_gc/b3_ac_rank_small.py` | NEW small-scale rank scan |
| `research/t2_sparse_gc/verify_redundancy.py` | NEW LP-redundancy test |
| `research/t2_sparse_gc/verify_redundancy_multi.py` | NEW multi-config redundancy verifier |
| `research/t2_sparse_gc/verify_bounds_equivalence.py` | NEW bounds-equivalence prover |

## 推荐生产配置 (更新)

```bash
# Tinyimagenet 及中型 CNN benchmarks
export ACT_HZ_DENSE_TO_SPARSE=1 ACT_HZ_PRECONV_SPARSE=1 \
       ACT_HZ_SPARSE_EQ_LAGR=1 ACT_HZ_SPARSE_EQ_LAGR_COMPACT=1 \
       ACT_HZ_SPARSE_EQ_LAGR_K_MAX=2000
# 43% RSS reduction vs baseline, zero precision loss, sound, 5/5 design 原则

# cifar100 resnet_large 及深 CNN: 仍只用 T2b
export ACT_HZ_DENSE_TO_SPARSE=1 ACT_HZ_PRECONV_SPARSE=1
unset ACT_HZ_SPARSE_EQ_LAGR
```

## Memory

- `project_b3_sparse_eq_lagr_20260528.md` 更新 (含 LP-redundancy 发现)
- `MEMORY.md` 索引行更新 ("MAJOR WIN on tinyimagenet")

## 下一步可能方向 (按 ROI)

1. **应用 LP-redundancy 分析到 dense eq_lagr_v8 + PEE**: dense path 也可能有同样冗余, fix 可能让 dense 也 save RSS — 但需要不破坏现有 PEE / large_cls_proof_mode 兼容性
2. **Sparse PEE 跨层 redundancy detection**: 之前估计 1-2 周, 现在重新评估 — 因为单层已有 50% 减少, 跨层增量收益可能 < 1.5×. ROI 可能不值得
3. **跑 tinyimagenet 全 200 iids with v2 compact + longer wall**: 可能首次有 verdict 变化, 验证 RSS save 是否转化为更多 LP iterations / CERT
4. **Metaroom benchmark with B3 v2**: 100 iids GPU 已 CERT 但 CPU OOM, B3 v2 可能解锁部分

要不要我继续 (3) tinyimagenet 全 200 sweep, 看是否 longer wall + lower RSS 能拿到首个 CERT?
