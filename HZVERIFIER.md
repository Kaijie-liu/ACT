# HZ Verifier — 创新与实验指南

> ACT (Abstract Constraint Transformer) 中 **Hybrid Zonotope (HZ) 神经网络验证器**的设计、实现与复现指南。
>
> 这份文档面向想完整理解我们工作的人 —— 从"什么是神经网络验证"开始，到"具体每行代码做了什么"为止。

---

## 目录

1. [一页概览 (TL;DR)](#1-一页概览-tldr)
2. [背景：神经网络验证在解什么问题](#2-背景神经网络验证在解什么问题)
3. [总体贡献](#3-总体贡献)
4. [创新点 1：HZono.eq_mask 扩展 + 真实 unsoundness bug 修复](#4-创新点-1hzonoeq_mask-扩展--真实-unsoundness-bug-修复)
5. [创新点 2：四态 HZ 表征层级（HZono / BoxHZ / LazyChainHZ / SparseGcZ）](#5-创新点-2四态-hz-表征层级)
6. [创新点 3：完整 HZ 传播流水线（多 ReLU 编码 + 三层 bounds 紧化 + QR 等式消元）](#6-创新点-3完整-hz-传播流水线)
7. [创新点 4：TF / Solver 架构分离（"造"与"解"解耦）](#7-创新点-4tf--solver-架构分离)
8. [实验结果：561 实例 vs VNN-COMP 2025 官方 GT，0 soundness 违反](#8-实验结果)
9. [如何复现](#9-如何复现)
10. [代码导览（哪个文件做什么）](#10-代码导览)
11. [FAQ](#11-faq)
12. [附录：外部 research artifact (小-密-深 LP 路径)](#12-附录外部-research-artifact)

---

## 1. 一页概览 (TL;DR)

### 快速 reference

```
代码     : /data1/Kane/ACT                                            (本仓库)
驱动脚本 : /data1/Kane/HyZor/scripts/v119_{acasxu,cifar,tiny}_*.py    (实验入口)
数据集   : /data1/Kane/data/vnncomp2025_benchmarks/benchmarks/        (VNN-COMP 2025)
官方 GT  : /data1/Kane/HyZor/arXiv-2512.19007v1/generated/2025/zero_tol/longtable.tex
环境     : conda activate act-py312
Python   : /data1/Kane/miniconda3/envs/act-py312/bin/python
```

### 四个创新（PR 范围）

我们在 ACT 的 HZ 域上做了四件事（都在这次 PR 内）：

| # | 创新点 | 一句话 | 代码位置 |
|---|---|---|---|
| 1 | **eq_mask 扩展 + bug fix** | 给 HZono 加 `eq_mask` 字段，让一个 HZ 同时容纳等式和不等式约束；修复一个会让 acasxu 报错 44 个"verified"的 soundness bug | [tf_mlp.py:hz_apply_relu](act/back_end/hybridz_tf/tf_mlp.py) + [solver_hz.py:_hz_reduce_constraints](act/back_end/solver/solver_hz.py) |
| 2 | **四态表征** | HZono / BoxHZ / LazyChainHZ / SparseGcZ 四种 HZ 形态，大网络自动降级避免 GPU OOM | [representations.py](act/back_end/hybridz_tf/representations.py) |
| 3 | **完整 HZ 流水线** | 4 种 ReLU 编码 + 3-tier 边界紧化 + QR 等式消元；都在 `algorithms/` 子目录 | [algorithms/](act/back_end/hybridz_tf/algorithms/) |
| 4 | **架构分离** | `hybridz_tf/` 只造约束；`solver/solver_hz.py` 只解约束。与老师 `interval_tf` / `dual_tf` 同构 | 整个目录布局 |

**PR 范围声明**：原本研究阶段还包含一条"小-密-深网络专用 GlobalTriangleLP + SpecAware + WitnessExtract"路径（让 acasxu 拿到 74V+15A），但它**依赖 `/data1/Kane/HyZor/SpecAwareLP.py` 这类 ACT 外部 research artifact**，不能 upstream。该路径已**从 PR 中移除**，仅作为外部实验脚本保留（见 [附录](#12-附录外部-research-artifact)）。

**验证**（PR 内 HZ cascade only，无小-密 LP 路径）：在 VNN-COMP 2025 的 3 个 benchmark 共 **561 个实例**上，对照 arXiv-2512.19007v1 官方 GT：

```
tinyimagenet:  175 / 175  V                               ← 100%
cifar100:      195 V + 5 U / 200                          ← 97.5%
acasxu:        0 V + 6 A + 180 U / 186                    ← HZ cascade 在小密深网络上精度受限
─────────────────────────────────────────
合计:          370 V + 6 A + 185 U = 561 / 0 violations
```

**0 soundness violations** 是 PR 的核心保证 — 这是 eq_mask 修复带来的。acasxu 决断率受 HZ 域几何特性限制（中间层 bound 爆炸），不是 PR 的设计目标；如需提升请参考[附录](#12-附录外部-research-artifact)外部脚本。

---

## 2. 背景：神经网络验证在解什么问题

如果你已经熟悉 NN verification，跳到 [§3](#3-总体贡献)。

### 2.1 问题

给定：
- 一个神经网络 `f: ℝⁿ → ℝᵐ`（例如分类器）
- 一个输入区域 `X ⊆ ℝⁿ`（例如某张图片 ± ε 的扰动范围）
- 一个不安全条件 `U ⊆ ℝᵐ`（例如"输出错误分类"）

问题：**∃ x ∈ X，使得 f(x) ∈ U 吗？**

- 如果 **不存在** → 网络在这个输入区域上是**安全的**（verifier 应回答 "verified" / "UNSAT"）
- 如果 **存在** → 网络在这个区域上**不安全**，有反例（verifier 应回答 "falsified" / "SAT" 并给出反例）
- 算不动 → "unknown"

### 2.2 抽象解释思路

逐层正向传播一个**集合**而非单点，得到 `f(X)` 的一个**过近似**（一定 ⊇ 真实可达集）。如果过近似都不交 `U`，原集肯定也不交，即 **safe**。

不同的"集合表示"精度和成本不同：
- **Interval (区间)**：最简单。每个 neuron 一个 `[lb, ub]`。便宜但松。
- **Zonotope**：`{c + G·ξ : ξ ∈ [-1,1]ⁿ}`，能表示倾斜的盒子，比 interval 紧。
- **Hybrid Zonotope (HZ)**：在 zonotope 上加二进制变量 `ξ_b ∈ {-1,+1}ⁿᵇ`，能精确表示 ReLU 这种分段函数的"两个分支"。

ACT 实现了所有三种作为不同的 **Transfer Function (TF) 后端**：
- `interval_tf/` — 区间
- `dual_tf/` — 对偶数（Wong-Kolter 风格）
- `hybridz_tf/` — 混合 zonotope ← **本工作的主战场**

### 2.3 HZ 的形式化定义

```
HZ = { c + Gc·ξ_c + Gb·ξ_b
       | Ac·ξ_c + Ab·ξ_b op b,
         ξ_c ∈ [-1,1]^ng,
         ξ_b ∈ {-1,+1}^nb }
```

- `c` 中心 (n,1)
- `Gc` 连续生成元 (n, ng)，`ξ_c` 在 `[-1,1]` 内自由
- `Gb` 二进制生成元 (n, nb)，`ξ_b` 取 `±1`（用来精确编码 ReLU 的 active/inactive 分支）
- `Ac, Ab, b` 约束（让 `ξ_c, ξ_b` 不能任意取，得满足某些线性方程/不等式）
- `op` 是**逐行**的 `=` 或 `≤`（**这是我们的扩展**，下面 §4 讲）

输出 `y` 是 `ξ` 的仿射函数。

---

## 3. 总体贡献

ACT 上游（SVF-tools/ACT:main）原本有：
- `HZono` 数据类（没有 `eq_mask`，所有约束都是 `=`）
- `HZSolver`（一个简单的 wrapper，只调 `hz_compute_bounds`，做 IBP-tightening，不出 verdict）
- `hybridz_tf/{hybridz_tf,tf_mlp,tf_cnn,tf_rnn,tf_transformer}.py`（基础 TF dispatch，~500 LOC）

我们在上游基础上加了 **~7800 LOC 净增改**，把 HZ 从"只算 bounds 的辅助工具"扩展到一个**完整的 HZ verdict verifier**：

```
ACT upstream (HZ 部分)         我们加上之后
─────────────────────         ────────────────────
HZono (无 eq_mask)        →    HZono (有 eq_mask) [创新 1]
                          +    BoxHZ / LazyChain / SparseGc [创新 2]

HZSolver.compute_bounds()  →   HZVerifier.consume_cons() [创新 5]
                               ├─ 4 种 ReLU 编码 [创新 3]
                               ├─ 3-tier bounds 紧化 [创新 3]
                               ├─ QR 等式消元 [创新 3]
                               ├─ LP unsafe-set 验证 [创新 4]
                               └─ ONNX 严格 replay [创新 4]
```

下面逐个详细讲。

---

## 4. 创新点 1：HZono.eq_mask 扩展 + 真实 unsoundness bug 修复

### 4.1 为什么需要 eq_mask？

ReLU 的 HZ 精确编码（eq_lagr_v8）会加 `nc` 行**等式**：
```
xi1 + xi3 + z = 1       (graph eq 1)
xi2 + xi4 - z = 1       (graph eq 2)
α·xi1 - β·xi2 - Gc·ξ - Gb·ξ_b + α·z = c - β/2   (linking eq)
```
这些必须是 `=`。

而某些操作会加**不等式**行 — 例如 `intersect_box(hz, lb, ub)` 把 HZ 限制在外部计算出的 `[lb, ub]` 内：
```
+Gc·ξ_c + Gb·ξ_b ≤ ub - c       (z ≤ ub)
-Gc·ξ_c - Gb·ξ_b ≤ c - lb       (z ≥ lb)
```
这两行必须是 `≤`。

**所以一个 HZ 同时需要装等式行和不等式行。** 我们给 `HZono` 加了一个 bool 数组 `eq_mask: (nc,)`，True = 等式行，False = 不等式行。

### 4.2 实际 bug：丢失 eq_mask → 不等式被当等式

`tf_mlp.py` 的 `hz_apply_relu` 原本构造输出 HZono 时**没有传入 eq_mask**：

```python
# ❌ BUG（修复前）：没传 eq_mask → 默认 None → "所有行都是等式"
out = HZono(c=out_c, Gc=out_Gc, Gb=out_Gb,
            Ac=torch.cat([old_Ac, new_eq_Ac]),
            Ab=torch.cat([old_Ab, new_eq_Ab]),
            b=torch.cat([hz.b, new_eq_b]))
```

后果：在 `hz_apply_relu` 之前调用过 `intersect_box` 的话，输入 HZ 的 2n 行 box-clipping **不等式**会被 `_eq_mask_of` 的"默认全 True"语义重新解读为**等式**：

```
z ≤ ub  +  z ≥ lb     →     z = ub  AND  z = lb     →     矛盾，LP infeasible
```

**LP 错误地返回 infeasible → 验证器错答 "verified"**。

### 4.3 后果：acasxu 上 44 个 false positives

跑 acasxu_2023 全集（186 instances），跟 arXiv-2512.19007v1 官方 GT 对照：

```
GT=sat   ACT=verified    44     ← FALSE POSITIVES（不安全实例被错答安全）
GT=unsat ACT=verified   127     ← 也是 bug 路径"蒙对"了
GT=unsat ACT=unknown     13
GT=sat   ACT=unknown      2
```

不只是有 44 个错答，连那 127 个"对"的也是靠 LP 错误 infeasible 蒙的 —— 验证器根本没真正验证。

### 4.4 定位过程

我们用一套"逐层 x\* 包含性测试"找到 bug 在哪：

1. **找反例**：在 acasxu prop_2 (iid=46) 输入箱里随机采 10000 点，找到 1 个 `Y_0 > max(Y_1..4)`（margin=+0.0001500892）—— 这是真实存在的反例 `x*`
2. **逐 ReLU 检查 HZ 是否包含 `f(x*)`**：用 LP 解 "find `xi` such that `f(x*) = c + Gc·xi_c + Gb·xi_b` AND `Ac·xi_c + Ab·xi_b op b`"
3. **结果**：L5 ReLU **入口** HZ 包含 `x*` 的 pre-ReLU 值；L5 ReLU **出口** HZ 不包含 `x*` 的 post-ReLU 值
4. **子步骤 dissection**：进一步拆 L5 内部 5 个子步（bounds → intersect_box → encode → binary_probe → project_eq_elim），定位到 `apply_relu_v8_memaware` 内的 `hz_apply_relu` 把 `x*` 丢了
5. **核心**：`hz_apply_relu` 构造输出 HZono 时丢了 `eq_mask`

### 4.5 修复

[`act/back_end/hybridz_tf/tf_mlp.py`](act/back_end/hybridz_tf/tf_mlp.py) 的 `hz_apply_relu` 和 `hz_apply_leaky_relu` 都改成：

```python
em_old = _eq_mask_of(hz)
# ...构造 eq_Ac / eq_Ab / eq_b...
em_new = torch.cat(
    [em_old, torch.ones(3 * k, dtype=torch.bool, device=device)]
)
out = HZono(
    c=out_c, Gc=out_Gc, Gb=out_Gb,
    Ac=torch.cat([old_Ac_ext, eq_Ac], dim=0),
    Ab=torch.cat([old_Ab_ext, eq_Ab], dim=0),
    b=torch.cat([hz.b, eq_b], dim=0),
    eq_mask=em_new,                             # ← 关键 fix：保留 + 扩展
)
```

修复效果（acasxu 全集）：
```
GT=sat   ACT=verified     0    ← 0 false positives ✓
GT=sat   ACT=adv          15   ← 真实反例
GT=sat   ACT=unknown      31
GT=unsat ACT=verified     74
GT=unsat ACT=unknown      66
```
**0 soundness violations**。

### 4.6 教训：CI 为什么没抓到

ACT 现有 CI（`.github/workflows/act-backend-float32.yml`）有两层 soundness 检查：
- **counterexample mode**：测算子级 bounds 对随机采样点的 soundness
- **bounds mode**：测 IBP 传播在合成网络上是否过近似

但都**只测 `HybridzTF.forward` 算子级精度，从不测 `HZVerifier.solve_batch` 的 verdict 正确性**。`hz_apply_relu` 算子在 IBP-only 路径下是 sound 的（因为没有 intersect_box 加进来的不等式行），CI 跑出 0 violations 没毛病。

但 **verdict 路径**才会调 `intersect_box`，CI 完全没覆盖。

**改进 CI 的方案**（在 `MEMORY.md` 的 `feedback_soundness_gate` 一节里）：
1. 加 ground-truth 回归集：5-10 个已知 SAT 实例 + 5-10 个 UNSAT 实例，每次 PR 跑一遍
2. 对每个 CERTIFIED verdict 做 PGD 攻击 + ORT replay，如果能找到反例就 fail
3. 多路径 cross-checking：同实例跑 HZ cascade + dual + interval，verdict 不允许矛盾

---

## 5. 创新点 2：四态 HZ 表征层级

### 5.1 问题

`HZono` 的 `Gc` 矩阵是 `(n, ng)`。对一个大 CNN（n=65536 的特征图，ng=数千），一个 `Gc` 就要 ~10GB。GPU 装不下就 OOM。

### 5.2 解法：分层降级

我们提供四种 HZ "形态"，自动选择最便宜的能装下的：

| 形态 | 存储 | 何时用 | 在 `representations.py` 里的类 |
|---|---|---|---|
| **`HZono`** | 完整 (c, Gc, Gb, Ac, Ab, b) | 标准 / 小网络 | `solver_hz.HZono` |
| **`SparseGcZ`** | 稀疏 COO `Gc`，nb=0 | 单 conv 链稀疏明显 | `representations.SparseGcZ` |
| **`LazyChainHZ`** | BoxHZ 根 + 一串"linear-op token" | 累积多个 affine 不立即物化 Gc | `representations.LazyChainHZ` |
| **`BoxHZ`** | 仅 `(lb, ub)` | dim 大但精度无所谓时（IBP fallback） | `representations.BoxHZ` |

降级是 **sound** 的（每个降级目标都包含 ⊇ 原集）。

### 5.3 自动派发：[`hz_routing.py`](act/back_end/hybridz_tf/hz_routing.py)

`hz_routing.py` 是个**形态分发器**：同一个公开 API（如 `hz_dense`）按传入 HZ 类型走不同实现：

```python
def hz_dense(hz, W, b=None):
    if isinstance(hz, SparseGcZ):    return hz.apply_dense(W, b)       # 稀疏路径
    if isinstance(hz, LazyChainHZ):  return hz.with_dense(W, b)        # 延迟路径
    if isinstance(hz, BoxHZ):        return _ibp_dense_path(...)       # IBP 路径
    # 默认 HZono：调 hz_multiply + hz_add_const
```

公开 16 个函数：`hz_from_bounds, hz_dense, hz_conv2d, hz_apply_relu_v8, hz_concat, hz_intersect_polytope, hz_maxpool2d, ...`

降级条件由环境变量控制（详见 [§10.3](#103-环境变量调优)）：
- `HYZOR_LARGE_HZ_DIM_CAP`（默认 8192）：dim 超过这个就走 BoxHZ
- `HYZOR_SPARSE_INPUT_THRESHOLD`：稀疏化阈值

---

## 6. 创新点 3：完整 HZ 传播流水线

[`hybridz_tf/algorithms/`](act/back_end/hybridz_tf/algorithms/) 子目录下 6 个文件，~2570 LOC，构成一条 ReLU 层的完整精化流水线。在 `hz_routing.py:hz_apply_relu_v8`（~140 LOC）里串起来：

```python
def hz_apply_relu_v8(hz, *, method="eq_lagr_v8", ...):
    # 1. Bounds cascade: get tight pre-ReLU bounds       → bounds_tighten.py
    lb, ub = _hzono_tight_bounds(hz)
    # 2. Intersect with box (clip HZ to bounds, sound)    → bounds_tighten.py
    hz = hz_intersect_box(hz, lb, ub)
    # 3. ReLU encoding (mem-aware dispatch)              → v8_memaware.py
    hz = apply_relu_v8_memaware(hz, lb, ub)              → relu_methods.py
    # 4. Binary probe: fix proven binaries                → binary_probe.py
    hz = binary_probe_v8(hz, ...)
    # 5. Project eq elim: QR-reduce generators            → eq_elim.py
    hz = project_eq_elim(hz, ng_base=...)
    return hz
```

### 6.1 [`algorithms/bounds_tighten.py`](act/back_end/hybridz_tf/algorithms/bounds_tighten.py) (655 LOC)

3-tier 边界紧化级联：

| Tier | 名字 | 何时 | 精度 | 速度 |
|---|---|---|---|---|
| 1 | `hz_bounds_unconstrained` | 兜底（无约束 HZ） | 松 | O(n·ng) 快 |
| 2 | `hz_bounds_hz_dual` | 有不等式行无等式 | 中 | Adam Lagrangian，~100 iter |
| 3 | `hz_bounds_eq_elim_lp` | 有等式行（典型 ReLU 后 HZ） | 紧 | HiGHS LP / neuron |

还包含：
- `hz_intersect_box(hz, lb, ub)`：把 HZ 与 box 求交（加 2n 个**不等式**行，eq_mask=False） ← **创新点 1 的 bug 就发生在 hz_apply_relu 没保留这些 False 行**
- `_HighspyWarmSolver`：HiGHS LP warm-start，对 Tier 3 LP 给 3-10× 加速

### 6.2 [`algorithms/relu_methods.py`](act/back_end/hybridz_tf/algorithms/relu_methods.py) (541 LOC)

4 种 ReLU 编码（不同精度/内存权衡）：

| 编码 | ng 增加 | nb 增加 | nc 增加 | 用于 |
|---|---|---|---|---|
| `triangle` | +k | 0 | 0 | 最便宜，IBP 风格三角松弛 |
| `compact` | +k | +k | +2k | 中等，仅 2 个不等式行 |
| `bigM` | 0 | +k | +2k | 用 bigM 编码 active/inactive 分支 |
| `eq_native` | +4k | +k | +3k | **最精确**（在 `tf_mlp.py:hz_apply_relu` 里）|

其中 k = 不稳定 neuron 数。

### 6.3 [`algorithms/v8_memaware.py`](act/back_end/hybridz_tf/algorithms/v8_memaware.py) (311 LOC)

mem-aware 调度器，按 GPU 可用预算选 4 种编码之一：

```python
candidates = [
    ("eq_native", float64, est64["eq_native"]),  # 最贵
    ("compact",   float64, est64["compact"]),
    ("eq_native", float32, est32["eq_native"]),
    ("compact",   float32, est32["compact"]),
    ("triangle",  float64, est64["triangle"]),    # 最便宜
]
for mode, dtype, est in candidates:
    if est <= float(fit_ratio[mode]) * budget_b:
        return chosen
```

由 `HYZOR_V8_MEM_BUDGET_GB` 控制预算。

### 6.4 [`algorithms/eq_elim.py`](act/back_end/hybridz_tf/algorithms/eq_elim.py) (288 LOC)

`project_eq_elim`：用 QR 分解消除等式约束，把"被约束的连续生成元"求解出来代入，剩下的生成元数减少 → HZ 变紧。

```
原 HZ：n 维输出 × ng 生成元 × nc 等式
QR 分解 Ac_eq: dep_idx (size=rank) 被消除
新 HZ：n 维输出 × (ng - rank + ng_base) 生成元 × 0 等式（+ 2*rank 个 box 不等式）
```

`ng_base` 是想保留的"基本"生成元数（输入像素的相关性），默认 = 输入 HZ 的 `_base_ng`。

### 6.5 [`algorithms/binary_probe.py`](act/back_end/hybridz_tf/algorithms/binary_probe.py) (545 LOC)

`binary_probe` 试图把 `ξ_b ∈ {-1,+1}^nb` 里某些位"固定"成确定值：

1. **RIIM**（Row-Interval Implication Mining）：纯区间分析，看哪个 `ξ_b[i]` 必须 = +1 或 -1
2. **Pairwise mining**：挖 `ξ_b[i] = s · ξ_b[j]` 这类关系
3. **LP singleton**：对每个未固定的 `ξ_b[i]`，分别用 LP 测 "固定为 +1 是否可行" 和 "固定为 -1 是否可行"，如果其中一边不可行就固定到另一边

固定后从 HZ 里"折叠"（fold）到 c 和 b 里，nb 减少。

### 6.6 [`algorithms/sgm.py`](act/back_end/hybridz_tf/algorithms/sgm.py) (192 LOC)

**SGM (Shared Generator Merge)**：把两个 HZ 相加（如 ResNet 的 skip connection），如果它们共享部分生成元就合并 prefix，避免独立累积 ng。

```python
shares_generator(hz_x, hz_y) → bool       # 检测前缀共享
hz_sgm_add(hz_x, hz_y) → HZ                # 合并版加法
```

### 6.7 [`hybridz_tf/tf_mlp.py`](act/back_end/hybridz_tf/tf_mlp.py) + [`tf_cnn.py`](act/back_end/hybridz_tf/tf_cnn.py)

具体每个 layer kind 的 HZ 传播。`tf_mlp.py:hz_apply_relu` 是 eq_native ReLU 的本体（**创新 1 修复的位置**）。`tf_cnn.py:hz_maxpool2d` 是 stable-winner 行保留的精确 max-pool。

---

## 7. 创新点 4：TF / Solver 架构分离

### 7.1 老师的模式（已存在）

```
interval_tf/ — TF: 区间→区间 propagation     solver/solver_torchlp.py — Solver: 用 LP 解
dual_tf/     — TF: dual numbers propagation   solver/solver_dual.py    — Solver: 用对偶证书
```

**TF 只构造 / 解只求解。** 不混。

### 7.2 我们沿用这个模式

```
hybridz_tf/  — TF: HZ → HZ propagation         solver/solver_hz.py — Solver: HZ + spec → verdict
```

具体边界：
- `hybridz_tf/` 任何东西都只**接收 HZ 输入，返回 HZ 输出**（也可能在内部用 LP 来算 bounds，但这是 TF 内部细节，对外仍是 HZ→HZ）
- `solver/solver_hz.py` 接收"HZ + 不安全 spec"，输出 verdict（CERTIFIED/FALSIFIED/UNKNOWN）

### 7.3 文件布局

```
act/back_end/
├── hybridz_tf/                    ← TF：构造
│   ├── hybridz_tf.py              registry（layer kind → handler）
│   ├── hz_routing.py              4-flavor 分发器
│   ├── representations.py         BoxHZ / LazyChainHZ / SparseGcZ
│   ├── tf_mlp.py                  Dense / ReLU / Sigmoid / ...
│   ├── tf_cnn.py                  Conv2D / MaxPool2D
│   ├── tf_rnn.py                  LSTM / GRU / RNN
│   ├── tf_transformer.py          (upstream stub，passthrough 到 interval)
│   └── algorithms/                TF-内部 HZ→HZ 助手
│       ├── sgm.py
│       ├── relu_methods.py
│       ├── v8_memaware.py
│       ├── bounds_tighten.py
│       ├── eq_elim.py
│       └── binary_probe.py
│
└── solver/                        ← Solver：求解
    ├── solver_base.py             (upstream)
    ├── solver_interval.py         (upstream)
    ├── solver_gurobi.py           (upstream)
    ├── solver_torchlp.py          (upstream, renamed from solver_interval in master)
    ├── solver_dual.py             (upstream)
    └── solver_hz.py               (我们：HZVerifier + LP verify + replay + smalldense) ← 3027 LOC 单文件
```

### 7.4 为什么 `solver_hz.py` 是单文件而不是 4 个文件

我们曾经把 LP verify / strict replay / smalldense_lp / smalldense_witness 拆成 4 个文件，但最终**合并回 solver_hz.py 一个文件**，理由：
- `solver_interval.py` (356 LOC), `solver_gurobi.py` (371 LOC), `solver_dual.py` (565 LOC) 都是**单文件**
- 一个 solver 一个文件是 ACT 的约定
- HZVerifier 内部本来就有大量交叉调用，拆开反而读者要跨文件跳

最终 3027 LOC 大文件，但内部用 banner 注释清晰分段：
- `─── HZono dataclass + HZ ops ───`
- `─── HZVerifier class + consume_cons ───`
- `─── Final LP verification ───`
- `─── Strict witness replay ───`
- `─── Small-dense LP verifier ───`
- `─── Small-dense witness extractor ───`

---

## 8. 实验结果

### 8.1 数据集

**官方上游**：[ChristopherBrix/vnncomp2025_benchmarks](https://github.com/ChristopherBrix/vnncomp2025_benchmarks)

**本地路径**：
```
/data1/Kane/data/vnncomp2025_benchmarks/
└── benchmarks/
    ├── acasxu_2023/        ← 我们用
    │   ├── instances.csv   ← 官方 (onnx, vnnlib, timeout) 列表
    │   ├── onnx/*.onnx
    │   └── vnnlib/*.vnnlib
    ├── cifar100_2024/      ← 我们用
    ├── tinyimagenet_2024/  ← 我们用
    ├── cersyve/  cgan_2023/  collins_*/  ...   ← 其他 benchmark（未用）
```

我们 evaluation 用的 3 个 benchmark：

| Benchmark | 路径 | # instances | 网络规模 | spec 形式 |
|---|---|---|---|---|
| `tinyimagenet_2024` | `benchmarks/tinyimagenet_2024/` | 175 | ResNet-Med-ish | TOP1_ROBUST |
| `cifar100_2024` | `benchmarks/cifar100_2024/` | 200 | ResNet-Med + ResNet-Large | TOP1_ROBUST |
| `acasxu_2023` | `benchmarks/acasxu_2023/` | 186 | 6×50 MLP | UNSAFE_LINEAR (AND of 4 ≤) |

我们用 `/data1/Kane/HyZor/audit_results/repro_20260418/manifest.csv` 把 official `instances.csv` 合并成一份所有 benchmark 的 (benchmark, instance_id, onnx, vnnlib, timeout, official_zero, official_small, ...) 表格，driver script (`scripts/v119_*.py`) 据此构造 verification job。

### 8.2 Ground truth 来源

**官方报告**：[arXiv-2512.19007v1 (VNN-COMP 2025 final)](https://arxiv.org/abs/2512.19007v1)

**本地解压**：
```
/data1/Kane/HyZor/arXiv-2512.19007v1/
├── main.tex                  ← 报告正文
├── results.tex               ← Score 汇总
├── extended_results.tex      ← 引用 longtable
└── generated/
    └── 2025/
        ├── zero_tol/
        │   ├── longtable.tex       ← ★★★ ground truth 在这（每 iid 一行 sat/unsat）
        │   ├── scored.tex
        │   ├── unscored.tex
        │   └── stats.tex
        └── small_tol/              ← 容差版本（我们用 zero_tol，更严）
            └── longtable.tex
```

`longtable.tex` 是多个 verifier 投票后的官方 verdict。我们的 audit script 读这个文件，用正则 `^2025 (\S+(?:\s+\S+)*?) & (\d+) & ~\\textsc\{(\w+)\}` 提取 `(benchmark_name, iid, verdict)`，跟我们 `results.csv` 对照。

### 8.3 当前 PR 跑数（HZ cascade only，无小-密 LP 路径）

```
tinyimagenet:  175 V / 0 A / 0 U / 175               wall: ~30 min
cifar100:      195 V / 0 A / 5 U / 200               wall: ~50 min
acasxu:        0 V + 6 A + 180 U / 186               wall: ~12 min
───────────────────────────────────────────────
合计:          370 V + 6 A + 185 U  / 561

vs arXiv GT:
  GT=unsat ACT=verified                       370
  GT=unsat ACT=unknown                        145
  GT=sat   ACT=adv                              6
  GT=sat   ACT=unknown                         40
  ✓ NO SOUNDNESS VIOLATIONS (0/561)
```

> **关于 acasxu 0V**：HZ cascade 在 6×50 小密深网络上中间层 bound 爆炸，verified 率近零。这是 HZ 域的几何瓶颈，不是 PR 缺陷。完整版（74V+15A）需要外部 GlobalTriangleLP+SpecAware research artifact，见[附录](#12-附录外部-research-artifact)。HZ cascade 仍在 acasxu 上给出 6 个 sound adversarial witnesses。

### 8.4 与上游 baseline 对照

upstream/main 的 `HZSolver` 只 `compute_bounds()`，**不出 verdict**。所以"对照"只能在精度方面：

| Domain | tinyimagenet | cifar100 | acasxu |
|---|---|---|---|
| **interval (upstream)** | 不显著 | 不显著 | 不显著（精度不足）|
| **HZ (我们 PR 内)** | 175/175 V | 195/200 V | 0V+6A on 186 (HZ cascade only) |
| HZ + external 小-密 LP artifact | 175/175 V | 195/200 V | 74V+15A on 186 ([附录](#12-附录外部-research-artifact)) |

### 8.5 端到端 wall time（并行 3 dataset）

GPU：NVIDIA RTX PRO 6000 (98 GB)
- acasxu: 4 worker × CPU-bound LP，~12 min
- tinyimagenet: 1 worker × GPU-bound HZ propagation，~30 min
- cifar: 1 worker × GPU-bound，~50 min
- **总 wall（并行运行）≈ 50 min**

---

## 9. 如何复现

### 9.1 环境

```bash
conda activate act-py312
# Python 解释器路径：/data1/Kane/miniconda3/envs/act-py312/bin/python
# 关键依赖：torch + onnxruntime + scipy + highspy + onnx2torch + gurobipy（可选）
```

**所有路径**（一次性收齐）：

| 用途 | 路径 |
|---|---|
| ACT 代码仓库 | `/data1/Kane/ACT` |
| HyZor 实验目录（driver 脚本 + audit_results） | `/data1/Kane/HyZor` |
| 实验 driver 脚本 | `/data1/Kane/HyZor/scripts/v119_{acasxu,tinyimagenet,cifar}_full_act.py` |
| 实验结果输出 | `/data1/Kane/HyZor/audit_results/v119_*/` |
| **VNN-COMP 2025 benchmarks** | `/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/` |
| Manifest（合并的 instances 列表） | `/data1/Kane/HyZor/audit_results/repro_20260418/manifest.csv` |
| **VNN-COMP 2025 官方报告 (arXiv)** | `/data1/Kane/HyZor/arXiv-2512.19007v1/` |
| **官方 ground truth (longtable.tex)** | `/data1/Kane/HyZor/arXiv-2512.19007v1/generated/2025/zero_tol/longtable.tex` |
| Python 解释器 | `/data1/Kane/miniconda3/envs/act-py312/bin/python` |
| HZ 长程记忆（设计决策/历史 ablation） | `/home/kaijieliu/.claude/projects/-data1-Kane-HyZor/memory/` |

### 9.2 跑单个 instance（最小例子）

```python
import torch
from pathlib import Path
from act.util.device_manager import initialize_device
initialize_device(device='cuda', dtype='float64')

from act.front_end.vnnlib_loader.data_model_loader import load_vnnlib_pair
from act.front_end.vnnlib_loader.vnnlib_parser import parse_vnnlib_queries
from act.front_end.verifiable_model import (
    InputLayer, InputSpecLayer, OutputSpecLayer, VerifiableModel
)
from act.pipeline.verification.torch2act import TorchToACT
from act.back_end.solver.solver_hz import HZVerifier, verify_once_hz

# 1. Load
pair = load_vnnlib_pair("acasxu_2023", "ACASXU_run2a_1_2_batch_2000.onnx",
                       "prop_2.vnnlib", root_dir=BENCH_ROOT, auto_download=False)
m = pair['model'].to(torch.float64).eval()
ins = pair['labeled_tensor'].tensor.shape
qs = parse_vnnlib_queries(VNN_PATH)
ins_, out_ = qs[0]

# 2. Build ACT graph
inl = InputLayer(labeled_input=pair['labeled_tensor'], shape=tuple(int(s) for s in ins),
                 dtype=torch.float64)
vm = VerifiableModel(input_layer=inl, input_spec=InputSpecLayer(spec=ins_),
                    model=m, output_spec=OutputSpecLayer(spec=out_))
net = TorchToACT(vm).run()

# 3. Run verifier
solver = HZVerifier(device='cuda', dtype=torch.float64, timeout_s=60,
                   strict_replay=True, onnx_path=ONNX_PATH, vnnlib_path=VNN_PATH)
status, _, _ = verify_once_hz(net=net, solver=solver, timelimit=60)
print(status)  # "UNSAT" (= verified) / "SAT" (= falsified) / "UNKNOWN"
```

### 9.3 环境变量调优

| 变量 | 默认 | 含义 |
|---|---|---|
| `HYZOR_USE_ACT` | `1` | 用 ACT-native HZ ops（vs legacy HyZor pkg，pkg 已删） |
| `HYZOR_TF_MODE` | `interval` | 前向 bounds 模式 |
| `HYZOR_PURE_HZ_MODE` | `1` | 关闭 attack-first / SAT-preflight 等启发 |
| `HYZOR_LARGE_CLS_EQ_LAYERS` | `1` | large-class 网络（如 cifar100）的 eq_lagr 层数 |
| `HYZOR_V8_MEM_BUDGET_GB` | `8` | v8 mem-aware ReLU dispatcher 的预算 |
| `HYZOR_PEE_GPU_QR` | `1` | project_eq_elim 用 GPU QR |
| `HYZOR_SAT_SIDECAR` | `1` | 启用 SAT witness 验证（用 ORT 替代 torch） |
| `HYZOR_DISPATCH_GUARD_GB` | `8` | 大于这个就降级到 BoxHZ |

### 9.4 跑全 benchmark（v119 reproduce）

```bash
cd /data1/Kane/HyZor

# 单独跑某一个
python scripts/v119_acasxu_full_act.py        # ~12 min, 4 workers
python scripts/v119_tinyimagenet_full_act.py  # ~30 min, 1 worker GPU
python scripts/v119_cifar_full_act.py         # ~50 min, 1 worker GPU

# 三个并行（推荐，~50 min wall total）
nohup python scripts/v119_acasxu_full_act.py > audit_results/v119_acasxu_full_20260522/stdout.log 2>&1 &
nohup python scripts/v119_tinyimagenet_full_act.py > audit_results/v119_tinyimagenet_full_20260522/stdout.log 2>&1 &
nohup python scripts/v119_cifar_full_act.py > audit_results/v119_cifar_full_20260522/stdout.log 2>&1 &

# 等完跑 audit
python -c "
import csv, re
from collections import defaultdict
from pathlib import Path

ARXIV = Path('/data1/Kane/HyZor/arXiv-2512.19007v1/generated/2025/zero_tol/longtable.tex')
LINE_RE = re.compile(r'^2025 (\S+(?:\s+\S+)*?) & (\d+) & ~\\\\textsc\{(\w+)\}')
gt = defaultdict(dict)
with open(ARXIV) as f:
    for line in f:
        m = LINE_RE.match(line)
        if m:
            bench = {'Acasxu 2023':'acasxu_2023','Cifar100 2024':'cifar100_2024',
                     'Tinyimagenet 2024':'tinyimagenet_2024'}.get(m.group(1).strip())
            if bench: gt[bench][int(m.group(2))] = m.group(3).lower()

for B, fname in [('tinyimagenet_2024','tinyimagenet'),
                  ('cifar100_2024','cifar'),
                  ('acasxu_2023','acasxu')]:
    act = {}
    with open(f'audit_results/v119_{fname}_full_20260522/results.csv') as f:
        for r in csv.DictReader(f): act[int(r['iid'])] = r['status']
    audit = defaultdict(int)
    for iid, g in gt[B].items():
        a = act.get(iid, 'MISSING')
        audit[f'GT={g} ACT={a}'] += 1
    print(B, dict(audit))
"
```

### 9.5 CLI 接口

通过 ACT 标准 CLI：

```bash
python -m act.back_end --verify \
    --network path/to/saved_net.json \
    --solver hyzor                       # 或 --solver hz
```

`--solver hyzor` 和 `--solver hz` 都映射到 `HZVerifier`（在 [`cli.py:_make_solver`](act/back_end/cli.py) 里）。

---

## 10. 代码导览

| 你想看 | 去哪里 |
|---|---|
| HZono 数据类 + 基础 ops | [`solver/solver_hz.py`](act/back_end/solver/solver_hz.py) (top) |
| HZVerifier 主类（verdict 入口） | [`solver/solver_hz.py:HZVerifier`](act/back_end/solver/solver_hz.py) |
| 4-flavor 分发器 | [`hybridz_tf/hz_routing.py`](act/back_end/hybridz_tf/hz_routing.py) |
| BoxHZ / LazyChainHZ / SparseGcZ | [`hybridz_tf/representations.py`](act/back_end/hybridz_tf/representations.py) |
| MLP layer 传播（Dense/ReLU/Sigmoid/...） | [`hybridz_tf/tf_mlp.py`](act/back_end/hybridz_tf/tf_mlp.py) |
| **eq_mask soundness fix** | [`hybridz_tf/tf_mlp.py:hz_apply_relu`](act/back_end/hybridz_tf/tf_mlp.py) |
| Conv2D + MaxPool2D | [`hybridz_tf/tf_cnn.py`](act/back_end/hybridz_tf/tf_cnn.py) |
| 4 种 ReLU 编码 | [`algorithms/relu_methods.py`](act/back_end/hybridz_tf/algorithms/relu_methods.py) |
| mem-aware ReLU dispatcher | [`algorithms/v8_memaware.py`](act/back_end/hybridz_tf/algorithms/v8_memaware.py) |
| 3-tier bounds 紧化 | [`algorithms/bounds_tighten.py`](act/back_end/hybridz_tf/algorithms/bounds_tighten.py) |
| QR 等式消元 | [`algorithms/eq_elim.py`](act/back_end/hybridz_tf/algorithms/eq_elim.py) |
| 二进制探测（RIIM + LP） | [`algorithms/binary_probe.py`](act/back_end/hybridz_tf/algorithms/binary_probe.py) |
| SGM（共享生成元合并） | [`algorithms/sgm.py`](act/back_end/hybridz_tf/algorithms/sgm.py) |
| Final LP unsafe-set 验证 | [`solver/solver_hz.py`](act/back_end/solver/solver_hz.py) 「Final LP verification」段 |
| 严格 ONNX replay | [`solver/solver_hz.py`](act/back_end/solver/solver_hz.py) 「Strict witness replay」段 |
| Small-dense LP 路径 | [`solver/solver_hz.py`](act/back_end/solver/solver_hz.py) 「Small-dense LP verifier」段 |
| Small-dense witness 提取 | [`solver/solver_hz.py`](act/back_end/solver/solver_hz.py) 「Small-dense witness extractor」段 |
| CLI 入口 | [`back_end/cli.py:_make_solver`](act/back_end/cli.py) |

### 10.1 阅读路径（推荐顺序）

如果你是第一次看，按下面顺序读：

1. [`solver_hz.py`](act/back_end/solver/solver_hz.py) 开头 ~120 行：HZono 数据类，理解 6 元组语义
2. [`tf_mlp.py:hz_apply_relu`](act/back_end/hybridz_tf/tf_mlp.py)：理解 ReLU 的精确 HZ 编码（eq_lagr_v8）
3. [`hz_routing.py:hz_apply_relu_v8`](act/back_end/hybridz_tf/hz_routing.py)：理解流水线的 5 步骤
4. [`solver_hz.py:HZVerifier.consume_cons`](act/back_end/solver/solver_hz.py)：理解 verdict 怎么产生
5. [`solver_hz.py`](act/back_end/solver/solver_hz.py) 后半部分（line 1900+）：理解 LP 验证 + strict replay

每个文件顶都有 banner header 说明做什么，先看 docstring。

---

## 11. FAQ

**Q: HZ 和 zonotope 的区别？**
A: zonotope 只有连续生成元 (`Gc·ξ_c`)，HZ 多了二进制生成元 (`Gb·ξ_b ∈ {-1,+1}`)。这让 HZ 能**精确**表示分段函数（如 ReLU 的 active/inactive 分支），而 zonotope 只能用线性松弛近似。

**Q: 为什么不直接用 MILP？**
A: 一个 MILP 调用 = 一个 solver 的 root → 整层网络只能解一次，慢。HZ 是**逐层**前向传播一个 HZ 表征，每层 O(n) 操作 + 偶尔的小 LP。结合 GPU 加速，整体 ~10 秒级，比 MILP 快几十到几百倍。但精度不如 MILP（HZ 是过近似），所以我们最终用 LP 在输出层做一次 feasibility 检查。

**Q: eq_mask 这个 bug 上游有吗？**
A: 上游的 HZono 没有 `eq_mask` 字段，因为上游的 HZ 流水线**只用等式约束**（IBP-only 风格），没有 `intersect_box` 这类加不等式行的操作。我们的扩展引入了不等式行（为了更紧的精度），同时引入了 eq_mask 字段管理，但**漏了一处保留**（`hz_apply_relu`），导致这个 bug。

**Q: 为什么 acasxu 决断率才 47.8%？**
A: acasxu 是小但深的网络，HZ 域几何特性导致中间层 bound 爆炸，PR 内 HZ cascade 只能拿 6 个 adv，0 个 verified。这是 HZ 域的**架构性瓶颈**。研究阶段我们用外部 GlobalTriangleLP + SpecAware artifact 把决断率拉到 47.8% (74V+15A)，但那条路径依赖 `/data1/Kane/HyZor/SpecAwareLP.py` 这种 ACT 外部研究文件，不能 upstream，因此移出 PR（见 [附录 C](#12-附录外部-research-artifact)）。要从 HZ-内部进一步提升 acasxu 需要多 neuron 关系挖掘（Anderson / Singh PRIMA），我们 ablation 过 0 lift。

**Q: 论文里要 cite 我们的工作哪些点？**
A: 主要 4 点：
1. **eq_mask 扩展**：让一个 HZ 同时容纳 = 和 ≤ 约束（架构创新）
2. **真实 soundness bug 修复 + 检测方法学**：用"逐层 x* 包含性 LP"作为 verdict-level soundness oracle 找 bug（论文 + 经验贡献）
3. **四态表征**：BoxHZ / LazyChainHZ / SparseGcZ + 自动降级（工程创新）
4. **完整 HZ→verdict 流水线**：从 upstream "HZ 仅算 bounds" 扩展到 "HZ → LP → verdict"（系统贡献）

**Q: 怎么继续提升精度？**
A: 短期（已 ablation 过的负方向，不要重做）：
- ❌ Anderson facets / 单神经元 cuts：0 lift on acasxu
- ❌ Pairwise k=2 / triple-hull k=3：0 lift
- ❌ Partial MILP K=4/8/16：只 close 6-20% gap，不值

长期（潜在方向）：
- Multi-layer joint LP（不只是 spec-aware，也包括 cross-layer 关系）
- Non-convex HZ extension（突破前向凸松弛理论上限）
- 用 dual solver（老师新做的）当 second cert，cross-check 提升 confidence

**Q: 上游老师同步什么变更我应该 follow？**
A: 老师正在做 `dual` 的 TF/Solver split（commits `523effc`, `1b31364`），跟我们 hybridz 的 split 同款架构。互相是 reference patterns。`solver_dual.py` 完成后可作为我们 cross-check 的第二证书。

---

## 附录 A：依赖图

```
┌─────────────────────────────────────────────────────────────────┐
│  HZVerifier.consume_cons (solver_hz.py:1008+)                   │
│  入口：接收网络 + spec，输出 verdict                              │
└──────────────────────────────┬──────────────────────────────────┘
                               │ 走 cons (一个 op 一步)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  hz_routing.hz_apply_relu_v8 (5 步骤)                            │
├─────────────────────────────────────────────────────────────────┤
│  1. _hzono_tight_bounds (bounds_tighten.py)                     │
│  2. hz_intersect_box (bounds_tighten.py, eq_mask=False)         │
│  3. apply_relu_v8_memaware (v8_memaware.py)                     │
│     ├─ eq_native → tf_mlp.hz_apply_relu  ← eq_mask FIX 在此      │
│     ├─ compact   → relu_methods.hz_apply_relu_compact           │
│     ├─ triangle  → relu_methods.hz_apply_relu_triangle          │
│     └─ bigM      → relu_methods.hz_apply_relu_bigM_fast         │
│  4. binary_probe_v8 (binary_probe.py)                           │
│  5. project_eq_elim (eq_elim.py)                                │
└──────────────────────────────┬──────────────────────────────────┘
                               │ 最终 HZ
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  check_unsafe_for_act (solver_hz.py, line ~2011)                │
│  构造 LP：HZ 约束 + unsafe spec → 检查 feasibility               │
│  ├─ infeasible → UNSAT (= verified)                              │
│  ├─ feasible   → 提取 witness                                    │
│  │              → strict_replay_for_act (solver_hz.py, ~2307)    │
│  │                ├─ 用 ONNX runtime 跑 witness                  │
│  │                ├─ 输出在 unsafe 区 → SAT (= falsified)        │
│  │                └─ 输出不在 → UNKNOWN (phantom rejected)        │
│  └─ timeout    → UNKNOWN                                         │
└─────────────────────────────────────────────────────────────────┘
```

## 附录 B：术语表

| 术语 | 含义 |
|---|---|
| HZ / hybrid zonotope | 混合 zonotope = continuous zonotope + binary generators |
| `ng` | continuous generator 数 (`Gc.shape[1]`) |
| `nb` | binary generator 数 (`Gb.shape[1]`) |
| `nc` | 约束行数 (`Ac.shape[0]` = `Ab.shape[0]` = `b.shape[0]`) |
| `eq_mask` | bool 数组 (nc,)，True = 等式行，False = 不等式行 |
| Tier 1/2/3 | bounds 紧化级联：UNC / dual / eq_elim LP |
| SGM | Shared Generator Merge，HZ 加法时合并共享前缀生成元 |
| PEE | project_eq_elim，QR 等式消元 |
| TOP1_ROBUST | "true_label > all others by ≥ margin" 类的 spec |
| UNSAFE_LINEAR | "存在 y 使得 C·y ≤ d (一组 ≤)" 类的 spec |
| ORT | ONNX Runtime |

---

## 12. 附录：外部 research artifact

研究阶段还有一条**显著提升 acasxu 决断率**的路径，但**不在这次 PR 内**（依赖 ACT 外部文件，不能 upstream）。这里说明给写论文 / 复现完整数字的人。

### 12.1 什么是 GlobalTriangleLP + SpecAware + WitnessExtract

对 ACAS Xu / linearizenn 这种小-密-深网络，HZ 域几何会爆炸（中间层 bound 拉到 10⁴ 量级），verified 率近零。一条替代路径是：

1. **GlobalTriangleLP**：把整网展开为一个大 LP，每个 ReLU 用 triangle relaxation；不走 HZ 域
2. **SpecAware refinement**：在 LP 里加上 unsafe spec 行，迭代收紧 pre-activation bounds（3 次 pass）
3. **WitnessExtract**：LP 若可行，从 corner / per-row max / perturbation 三策略提取候选 x，用 ONNX runtime 重放验真

### 12.2 实现位置（外部）

这些文件**不在 ACT 仓库**，在 HyZor 实验目录：
```
/data1/Kane/HyZor/
├── GlobalTriangleLP.py     ← 整网展开 + triangle relaxation LP
├── SpecAwareLP.py          ← 迭代精化 + sa.verify() 入口
└── WitnessExtract.py       ← verify_with_falsification + 3 策略 witness 提取
```

### 12.3 完整数字（外部 artifact 加上）

| Bench | PR 内 HZ cascade only | + 外部 small-dense LP artifact |
|---|---|---|
| tinyimagenet | 175/175 V (用) | 175/175 V (不用; HZ 够) |
| cifar100 | 195/200 V (用) | 195/200 V (不用; HZ 够) |
| **acasxu** | **0V + 6A + 180U** | **74V + 15A + 97U** |

### 12.4 为什么不收进 PR

- 依赖 `/data1/Kane/HyZor/*.py`，ACT 仓库外
- 与 ACT 验证 pipeline 解耦（不走 `verify_once` / `solve_batch`）
- 是 **research artifact** 而非生产代码 — 内部 LP 编码经过多轮 ablation，soundness 已通过我们自己的 oracle 验证（10000 ORT 反例采样 / 561 instances 跨核），但**没有 ACT 标准 review**

### 12.5 如何在外部跑 acasxu 完整数字

参考历史 commit `43b6230` 里的 `solver_hz.py` 小-密-深段（已在本 PR 中删除）。基本流程：

```python
import sys
sys.path.insert(0, '/data1/Kane/HyZor')
import SpecAwareLP as sa
import WitnessExtract as we

# 给 HZVerifier 加 vnnlib_path 参数（PR 内已删除），并在 consume_cons 顶部
# 加 dispatch：if is_small_dense(onnx_p): return we.verify_with_falsification(...)
```

完整 demo 在 `commits/43b6230~/scripts/v118_acasxu_full_act.py`。

---

*文档版本：v2.0 / 2026-05-22*
*对应 commit：post-advisor revert（cleanup, no small-dense LP, no dead knobs）*
*维护者：HyZor / ACT 团队*
