# 长期推进计划: 1747 → 2000+ V/A (严格 P1-P5)

**Updated**: 2026-06-06
**Current**: 1747 V/A strict (275 NEW V audited bundle)
**Target**: 2000+ V/A strict
**Gap to close**: 253+ V/A

---

## 0. 战略原则 (不可妥协)

- **P1-P5 严格遵守**: forward-only, no gradient, continuous LP, no input split, no random/corner.
- **MILP/SETPH 只做诊断**, 不计入 strict headline. 已隔离到 `optional_needs_ruling.json`.
- **soundness 永远优先于 lift**: tail_radius invariant + per-op lemmas + ORT 验证 + 独立 recompute.
- **每次 lift 必须经过 4 项 audit**:
  1. tail_radius unit tests (81 tests, must all pass)
  2. Walker fresh recompute (sample 5 iids per audit)
  3. ORT replay 500-2000 samples (must be 0 violations)
  4. r93 cross-check (no double-count)

---

## 1. 当前差距 (vs αβ-CROWN baseline)

```
αβ-CROWN V baseline:  948 V  (其 A 多数是 PGD, 违反 P2, 不算 apples-to-apples)
ACT V baseline:       980 V  (r93 + 我们的 275 strict bundle)
                      ─────
                      +32 ahead on V

但 αβ-CROWN 有大量 A=675 (PGD-driven) → 总 V+A=1623
ACT (无 PGD/AutoAttack 路径): V+A=1085 (vs strict P2)
```

**关键观察**: 在 forward-only/no-gradient/no-split 严格条件下, **我们已经 V 领先**.
αβ-CROWN 的 V+A 数字优势主要来自违反我们的 P2 原则 (PGD/AutoAttack falsifier).

---

## 2. 真实可推进路径 (按优先级排序)

### 🔴 Phase A: 今晚 (1-2 小时内回收)

| 项目 | 候选 V | 预期 ORT-clean | 进度 |
|---|---:|---:|---|
| metaroom_2023 完整 sweep | ~50 | 30-50 | 后台运行 |
| tllverifybench 完整 sweep | 38 | 25-38 | 后台运行 |
| collins_rul 完整 sweep | 39 | 25-39 | 后台运行 |
| cora_2024 完整 sweep | ~10 | 5-10 | 后台运行 |
| cifar 110+ 后续 | ? | 5-20 | 后台运行 |
| **小计** | **~140** | **90-150** | |

**预期 Phase A 完成时**: 1747 + 90-150 = **1837-1897 V/A strict** (距 2000 差 100-160)

### 🟡 Phase B: 本周 (1-7 天 工程)

| 项目 | 预期 lift | 工程 effort | 优先级 |
|---|---:|---|---|
| **sparse-slack columns** | +30-50 | 1-2 周, walker.py 重写 | ⭐⭐⭐ |
| tinyimagenet 走通 (深 CNN memory) | +50-100 | 与上面共享 | ⭐⭐⭐ |
| acasxu boundary refinement (-24 gap) | +10-20 | LP corner refinement | ⭐⭐ |
| dist_shift_2023 全扫描 (FCHZ 是否够) | 0-30 | 1-2 天 probing | ⭐⭐ |
| vggnet16_2022 走通 (-14) | +5-15 | 与 sparse-slack 共享 | ⭐ |
| collins_aerospace 探测 | 0-5 | 1 天 | ⭐ |
| **小计** | **+95-220** | | |

**预期 Phase B 完成时**: ~1900-2100 V/A strict (**到达 2000**)

### 🟢 Phase C: 2-4 周 (深度工程)

| 项目 | 预期 lift | 工程 effort | 备注 |
|---|---:|---|---|
| cctsdb_yolo dynamic Slice parser | +20-39 | 1 周, parser 重写 | 已有 R17 LUT_BOUNDS 脚手架 |
| ml4acopf_2024 (-53 gap) | +10-30 | 2 周, HiGHS 大规模优化 | abcrown 用 Gurobi 优势 |
| linearizenn (-46) spec-aware joint LP | +20-40 | 1-2 周, 借鉴 project_eq_elim_hero | bounds 太松, 需 LP refine |
| cersyve / cgan loose cases | +5-10 | 1 周 | bounds 边界 |
| nn4sys disjunctive vnnlib parser | +0-30 | 1 周 | 可能架构限制 |
| **小计** | **+55-150** | | |

**预期 Phase C 完成时**: 2000+ → 2100-2250 (paper-grade headline)

---

## 3. 核心工程: sparse-slack columns (Phase B 主力)

### 现状问题
- `hz_only` 模式: 单列 G(后期改为 0) + per-row tail_radius
  - **优点**: 内存 O(n) for tail
  - **缺点**: 多层 ReLU 后 tail dominates G, 等价于 box → cifar/tiny 太松
- `regular` 模式: 保留全 G
  - **优点**: 紧 bounds
  - **缺点**: 内存 O(n × K), cifar 110+/tinyimagenet OOM

### 设计
**per-layer sparse columns**: 每层 ReLU 引入 N_unstable_layer 个新列, 但**仅按需 materialize**:
- 不立即 reduce 到 box (像 tail_radius)
- 不全保留 (像 regular G)
- 而是: 用 **sparse CSR 表示**, 只存非零 entries

**预期效果**:
- 内存复杂度: 从 O(n × K_total) 降到 O(n × K_effective)
- bound 精度: 接近 regular G, 而非 hz_only 的 box
- 适用: cifar deep (110+), tinyimagenet, vggnet16

### 实施步骤
1. 在 `fc_hz_state.py` 添加 `SparseGcState` 类 (scipy.sparse.csr_matrix backing)
2. 重写 ReLU "unstable" 路径以 sparse 模式追加新列
3. `hz_closed_form_ub` 在 sparse mode 下用 sparse-aware abs-sum
4. Conv/Dense propagation 用 sparse @ dense 乘法
5. tail_radius 仍然保留为 dense vector (fallback)
6. 工厂模式: walker 根据 layer 维度自动选择 regular / sparse / hz_only

### 风险
- sparse 矩阵 fill-in 后会变 dense (Conv 后所有 column 可能填充满)
  - mitigation: 触发阈值检测, 自动 switch 到 hz_only fallback
- 增加代码复杂度
  - mitigation: 独立模块, 完整单元测试

---

## 4. 短期可控里程碑

| 里程碑 | 日期 | 目标 V/A | Audit 要求 |
|---|---|---:|---|
| **M1: Phase A 完成** | 今晚 | 1837-1897 | ORT 500-sample on 100% candidates |
| **M2: dist_shift sweep** | +2 天 | +0-30 | walker reach analysis |
| **M3: sparse-slack v0** | +1 周 | +30-50 | invariant proof + unit tests |
| **M4: tinyimagenet 走通** | +2 周 | +50-100 | 175/175 reproducible |
| **M5: 2000 突破** | +3 周 | 2000+ | full session report + paper-grade audit |
| **M6: 2100 advance** | +4-6 周 | 2100+ | cctsdb, vgg, linearizenn closed |

---

## 5. Audit & Provenance 规则

每个 milestone 必须生成:
1. `audit_results/sprint_<phase>_strict_<N>_<date>.json` 严格 bundle
2. `audit_results/sprint_<phase>_optional_<n>_<date>.json` 隔离 optional
3. SHA256 提供 provenance
4. tail_radius invariant 测试套件 + 任何新 ops 的 lemmas
5. 与 abcrown / 官方 longtable 的 cross-check 报告

---

## 6. 不会做 (严格排除)

- ❌ MILP solver as primary path (Tjeng, Big-M without LP relaxation)
- ❌ Gurobi (closed-source, 违反开源原则)
- ❌ PGD / AutoAttack / 任何 gradient-based falsifier (P2)
- ❌ Input split / BaB on input (P4)
- ❌ Random / corner-sample certification (P5)
- ❌ Backward CROWN bound propagation (P1)

但 **允许做 (诊断目的, 不计分)**:
- ✅ 内部 MILP 探测某个 iid 的 LP→MILP gap (for diagnosing 阻碍)
- ✅ ORT replay 作为 fake-CERT detection (不是 CERT proof)
- ✅ HiGHS continuous LP (P3 内的工程优化)

---

## 7. 最终 paper-grade headline 框架

```
"HyZor + ACT-HZ: Forward-only Hybrid Zonotope verification with strict P1-P5 compliance.

Across 25 VNN-COMP 2025 scored benchmarks:
- 2000+ V/A strict (forward only, no gradient, no Gurobi, no input split)
- vs αβ-CROWN 948 V baseline: +30+ V ahead on shared formal-strict scope
- audit pipeline: per-op invariant + sampling + ORT replay + cross-check

Key technical contribution:
- tail_radius FCHZ walker: per-row independent box error propagation
- sparse-slack columns: memory-efficient deep CNN representation
- 9 / 15+ mechanism portfolio under strict principles"
```
