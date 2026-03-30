# Diff Review: `hybridz_tf/` vs `upstream/main`

This document explains every change, file by file, function by function.
本文档逐文件、逐函数说明每个改动的必要性。

---

## 1. `hybridz_tf/__init__.py`

| Change / 改动 | Type / 类型 | Necessity / 必要性 |
|------|------|--------|
| `from .hybridz_tf import HybridzTF` → `import HybridzTF, HZono` | New feature / 新功能 | `HZono` is the core HZ data structure, required by `tf_mlp.py` via lazy import. `HZono` 是 HZ 核心数据结构，`tf_mlp.py` 需要通过 lazy import 获取它 |
| `__all__`: comment → `'HZono'` | New feature / 新功能 | Keep `__all__` consistent with actual exports. 保持 `__all__` 与实际导出一致 |

---

## 2. `hybridz_tf/hybridz_tf.py`

### 2.1 New imports / 新增导入

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
from act.back_end.hybridz_tf.tf_mlp import (
    _hz_multiply, _hz_add_const, _hz_compute_bounds, ...
)
```

**Necessity / 必要性**: `@dataclass` for `HZono`. `_hz_*` functions are called in `_hz_transform()`. `HZono` 使用 `@dataclass`，`_hz_*` 函数在 `_hz_transform()` 中被调用。

### 2.2 New `HZono` dataclass / 新增 `HZono` 数据类

```python
@dataclass
class HZono:
    c, Gc, Gb, Ac, Ab, b  # 6 tensors
```

**Necessity / 必要性**: Data container for Definition 3.1 of the paper. All HZ operations are based on this structure. Placed here (not in `core.py`) because only `hybridz_tf/` uses it internally. 论文 Definition 3.1 的数据容器，所有 HZ 运算都基于此结构。放在此文件而非 `core.py`，因为只在 `hybridz_tf/` 内部使用。

### 2.3 New `__init__` / 新增构造函数

```python
def __init__(self):
    self._hz_cache: Dict[int, HZono] = {}
    self._cache_net_id: Optional[int] = None
    self._tanh_K: int = 2
    self._sigmoid_K: int = 2
```

**Necessity / 必要性**:
- `_hz_cache`: Cache for propagating HZ state between layers (key=layer_id, value=HZono). HZ 状态在层间传播的缓存。
- `_cache_net_id`: Detects network switch to clear stale cache. 检测网络切换时清空缓存。
- `_tanh_K/_sigmoid_K`: Number of piecewise tangent segments, user-configurable. 分段切线包络的段数，用户可配置。

### 2.4 `_LAYER_REGISTRY` — No change / 无改动

**Identical to `main` branch.** All transfer functions keep their original signature `(L, bounds) -> Fact`, no `tf` parameter passed. 与 `main` 分支**完全一致**。所有 transfer function 保持原始签名，不传 `tf`。

### 2.5 New `_HZ_MAX_INPUT_DIM = 1024` / 新增维度阈值

**Necessity / 必要性**: Skip HZ tracking when input dimension exceeds 1024 (`Gc = diag(rad)` is an n×n matrix, causes OOM for large networks). CIFAR networks (3072-dim) fall back to interval arithmetic. 输入维度超过 1024 时跳过 HZ 追踪，避免 OOM。

### 2.6 New `_hz_from_bounds()` / 新增初始化方法

**Necessity / 必要性**: Creates initial HZ from INPUT layer bounds: `c=(lb+ub)/2, Gc=diag((ub-lb)/2)`. This is the starting point for HZ propagation. 从 INPUT 层的 `Bounds` 创建初始 HZ，是 HZ 传播的起点。

### 2.7 New `_hz_transform()` / 新增 HZ 调度方法

**Necessity / 必要性**: Core HZ dispatch method. Called by `apply()`, selects the corresponding HZ operation based on layer type. 核心 HZ 调度方法，在 `apply()` 中调用，根据层类型选择对应的 HZ 运算：

| Layer / 层类型 | HZ Operation / HZ 运算 | Description / 说明 |
|--------|---------|------|
| DENSE | `_hz_multiply + _hz_add_const` | Linear map, exact / 线性映射，精确 |
| BIAS | `_hz_add_const` | Translation, exact / 平移，精确 |
| SCALE | `_hz_multiply(diag)` | Scaling, exact / 缩放，精确 |
| RELU | `_hz_apply_relu + _hz_reduce` | Equality encoding (Eq.5-6), graph-exact / 等式编码，graph-exact |
| LRELU | `_hz_apply_leaky_relu + _hz_reduce` | Same structure as ReLU, inactive branch y=αx / 同 ReLU 结构 |
| TANH | `_hz_apply_tanh(K)` | Piecewise tangent envelope (Theorem 3.8) / 分段切线包络 |
| SIGMOID | `_hz_apply_sigmoid(K)` | Same as TANH / 同 TANH |
| ABS | `_hz_compute_bounds + fresh` | Nonlinear, falls back to interval / 非线性，退回区间 |
| CONV2D | `_hz_conv2d` | Batch conv2d on c/Gc/Gb / 批量卷积 |
| MAXPOOL2D | `return_indices` greedy selection | Preserves generator correlations / 保留生成器相关性 |
| FLATTEN/RESHAPE | Pass-through / 直接传递 | HZ is already in flat format / HZ 已是 flat 格式 |
| ADD | `_hz_minkowski_sum` | Minkowski sum / Minkowski 和 |
| MUL | interval corners + fresh | Bilinear, not exact / 双线性不可精确 |
| Others (LSTM etc.) | `return None` | No HZ transform, `apply()` creates fresh HZ / 无 HZ 变换 |

### 2.8 `apply()` changes / `apply()` 改动

Original / 原版：
```python
transfer_fn = self._LAYER_REGISTRY[k]
return transfer_fn(L, input_bounds, self)
```

New additions / 新增：
1. **Cache management / 缓存管理**: Network switch detection, INPUT layer seeding, predecessor propagation / 网络切换检测、INPUT 层种子、前驱传播
2. **HZ processing / HZ 处理**: Call `_hz_transform()` to compute HZ bounds / 调用 `_hz_transform()` 计算 HZ bounds
3. **Call transfer function / 调用传递函数**: Signature unchanged / 签名不变
4. **Post-processing / 事后处理**: Create fresh HZ for layers without transform; use tighter HZ bounds when available / 无 HZ 变换的层创建 fresh HZ；有 HZ 时用更紧 bounds

**Necessity / 必要性**: All HZ logic is centralized in `apply()`, transfer functions are unaware of HZ. This follows the advisor's design requirement: "make the tf processing before entering the method". 所有 HZ 逻辑集中在 `apply()` 中，transfer function 不感知 HZ 的存在。

---

## 3. `tf_mlp.py`

### 3.1 New imports and lazy import / 新增导入和延迟导入

```python
from __future__ import annotations
_HZono = None
def _get_HZono(): ...
import gurobipy / scipy (optional)
```

**Necessity / 必要性**: Avoids circular import between `hybridz_tf.py ↔ tf_mlp.py`. Gurobi/SciPy are used for HZ bounds computation. 避免循环导入。Gurobi/SciPy 用于 HZ bounds 计算。

### 3.2 New `_hz_multiply(hz, R)` / 新增线性映射

**Necessity / 必要性**: Linear map `c'=R@c, Gc'=R@Gc, Gb'=R@Gb`, used for DENSE layers. Paper Definition 3.2 (R·Z_h). 线性映射，用于 DENSE 层。论文 Definition 3.2。

### 3.3 New `_hz_add_const(hz, v)` / 新增平移

**Necessity / 必要性**: Translation `c'=c+v`, used for BIAS layers and DENSE bias. 平移操作，用于 BIAS 层和 DENSE 偏置。

### 3.4 New `_hz_is_unconstrained(hz)` / `_hz_bounds_unconstrained(hz)` / 新增无约束快速路径

**Necessity / 必要性**: Fast path for unconstrained HZ (empty Ac): computes bounds directly via `c ± |Gc|·1 ± |Gb|·1`, avoids calling Gurobi. 无约束 HZ 的快速路径，避免调用 Gurobi。

### 3.5 New `_hz_compute_bounds(hz)` / 新增 bounds 计算入口

**Necessity / 必要性**: Entry point for HZ bounds computation. Tries fast path (unconstrained), otherwise calls Gurobi MILP or SciPy LP. HZ bounds 计算入口，先尝试快速路径，否则调用 Gurobi 或 SciPy。

### 3.6 New `_hz_compute_bounds_gurobi(hz)` / 新增 Gurobi MILP 求解

**Necessity / 必要性**: Exact HZ bounds via Gurobi MILP. All constraints are equalities (`==`), `ξ_b` mapped to {-1,+1} via `GRB.BINARY` + `2ζ-1`. 用 Gurobi MILP 精确求解 HZ bounds。约束全部为等式，`ξ_b` 用 `2ζ-1` 映射到 {-1,+1}。

### 3.7 New `_hz_compute_bounds_scipy(hz)` / 新增 SciPy LP 回退

**Necessity / 必要性**: LP relaxation fallback when Gurobi is unavailable. `ξ_b` relaxed to continuous [-1,1], bounds are over-approximation but still sound. 无 Gurobi 时的 LP 松弛回退。bounds 是 over-approximation 但仍然 sound。

### 3.8 New `_hz_apply_relu(hz)` / 新增 ReLU 等式编码

**Necessity / 必要性**: ReLU with equality-based encoding, based on paper Section 3.1 + `applyReLU_eq_native_exact`: 基于论文 Section 3.1 + `applyReLU_eq_native_exact` 的 ReLU 等式编码：
- Per unstable neuron / 每 unstable neuron: 4 Gc + 1 Gb + 3 equality constraints / 等式约束
- Eq 1 / 等式 1: `ξ₁ + ξ₃ + z = 1` (graph)
- Eq 2 / 等式 2: `ξ₂ + ξ₄ - z = 1` (graph)
- Eq 3 / 等式 3: `α/2·ξ₁ - β/2·ξ₂ + α/2·z - Gc[i]·ξ_old - Gb[i]·ζ_old = c_i - β/2` (linking)
- Graph-exact, backed by paper / Graph-exact，有论文支撑

### 3.9 New `_hz_apply_leaky_relu(hz, alpha)` / 新增 LeakyReLU 等式编码

**Necessity / 必要性**: LeakyReLU with equality encoding. Same structure as ReLU, but inactive branch yields y=αx instead of y=0. LeakyReLU 等式编码，结构同 ReLU，但 inactive 分支 y=αx：
- Per unstable neuron / 每 unstable neuron: 6 Gc (2 real + 4 slack) + 1 Gb + 5 equality constraints / 等式约束
- Box equalities + linking equality for branch selection / Box 等式 + linking 等式实现分支选择

### 3.10 New `_hz_apply_piecewise(hz, func, dfunc, K)` / 新增分段切线近似

**Necessity / 必要性**: Piecewise tangent parallelogram approximation for Sigmoid/Tanh (paper Theorem 3.8 + Definition 3.4). Sigmoid/Tanh 的分段切线平行四边形近似（论文 Theorem 3.8 + Definition 3.4）：
- K tangent segments, default K=2 / K 段切线包络，默认 K=2
- Union operation via equalities + slack variables / 用 Union 操作的等式 + slack 变量实现段选择
- Per wide neuron / 每 wide neuron: 6K Gc (2K real + 4K slack) + K Gb + (4K+2) equalities / 等式

### 3.11 New `_hz_apply_sigmoid(hz, K)` / `_hz_apply_tanh(hz, K)` / 新增便捷包装

**Necessity / 必要性**: Convenience wrappers that call `_hz_apply_piecewise`. 调用 `_hz_apply_piecewise` 的便捷包装。

### 3.12 New `_hz_minkowski_sum(hz1, hz2)` / 新增 Minkowski 和

**Necessity / 必要性**: HZ operation for ADD layers. Paper Definition 3.3 (Minkowski sum). ADD 层的 HZ 运算，论文 Definition 3.3。

### 3.13 New `_hz_from_bounds_fresh(bounds, dtype, device)` / 新增 fresh HZ 创建

**Necessity / 必要性**: Creates unconstrained fresh HZ from Bounds, used for layers that don't support exact HZ operations (ABS, MUL, LSTM, etc.). 从 Bounds 创建无约束 fresh HZ，用于不支持精确 HZ 运算的层。

### 3.14 New `_hz_reduce(hz, max_order=10.0)` / 新增复杂度约简

**Necessity / 必要性**: Prevents HZ complexity explosion. PhD thesis Chapter 6. 防止 HZ 复杂度爆炸（PhD 论文 Chapter 6）：
- Step 1: Binary relaxation (small Gb columns → Gc columns) / 二值松弛
- Step 2: Girard continuous reduction (keep large columns, box small ones) / Girard 连续约简

### 3.15 Original transfer functions — signatures unchanged / 原有传递函数 — 签名未改动

All `hybridz_tf_dense`, `hybridz_tf_relu`, etc. keep signature `(L, Bin) -> Fact`. No `tf` parameter added. Internal logic (interval bounds + ConSet generation) unchanged. 所有函数签名保持 `(L, Bin) -> Fact`，未添加 `tf` 参数，内部逻辑未变。

---

## 4. `tf_cnn.py` — Detailed Line-by-Line Review / 逐行详细审查

### 4.1 Import changes / 导入改动

```python
# Added / 新增
from __future__ import annotations                          # deferred type evaluation
from typing import List, TYPE_CHECKING                      # replaced Tuple with TYPE_CHECKING
if TYPE_CHECKING:
    from act.back_end.hybridz_tf.hybridz_tf import HZono    # type hint only, no runtime import
from act.back_end.hybridz_tf.tf_mlp import (
    _hz_compute_bounds, _hz_from_bounds_fresh, _get_HZono,  # HZ helper functions
)

# Removed / 删除
from typing import List, Tuple                              # Tuple was never used
```

**Necessity / 必要性**: `TYPE_CHECKING` guard avoids circular import at runtime (hybridz_tf.py → tf_cnn.py → hybridz_tf.py). `Tuple` was imported but never used in the original code. `TYPE_CHECKING` 避免运行时循环导入。`Tuple` 在原代码中从未使用。

### 4.2 New `_parse_input_shape(input_shape)` / 新增形状解析

```python
def _parse_input_shape(input_shape):
    """Extract (C, H, W) from input_shape, which may be (N,C,H,W) or (C,H,W)."""
    if len(input_shape) == 4:
        _, C, H, W = input_shape
    elif len(input_shape) == 3:
        C, H, W = input_shape
    else:
        raise ValueError(f"Unexpected input_shape={input_shape}")
    return C, H, W
```

**Type / 类型**: Refactor (extract method) / 重构（提取方法）

**Necessity / 必要性**: The original `hybridz_tf_conv2d` had this exact if/elif/else block inline (lines 40-47 of main). Now `_hz_conv2d` also needs the same parsing. Extracting it eliminates duplication. 原 `hybridz_tf_conv2d` 中有完全相同的 if/elif/else 块。`_hz_conv2d` 也需要相同的解析，提取消除重复。

### 4.3 New `_reshape_bounds_4d(lb, ub, in_shape)` / 新增 bounds reshape

```python
def _reshape_bounds_4d(lb, ub, in_shape):
    """Reshape flat/3D bounds to (1, C, H, W) for pooling/conv operations."""
    if lb.dim() == 1 and in_shape:
        return lb.view(1, *in_shape), ub.view(1, *in_shape)
    elif lb.dim() == 3:
        return lb.unsqueeze(0), ub.unsqueeze(0)
    return lb, ub
```

**Type / 类型**: Refactor (extract method) / 重构（提取方法）

**Necessity / 必要性**: The original `hybridz_tf_maxpool2d` (lines 105-110 of main) and `hybridz_tf_avgpool2d` (lines 141-146 of main) had identical 6-line reshape blocks. Extracted into a shared helper. 原 `hybridz_tf_maxpool2d` 和 `hybridz_tf_avgpool2d` 各有完全相同的 6 行 reshape 块，提取为共享辅助函数。

### 4.4 New `_conv2d_generators(G, weight, C, H, W, ...)` / 新增生成器卷积

```python
def _conv2d_generators(G, weight, C, H, W, stride, padding, dilation, groups, n_out, dtype, device):
    """Apply conv2d to a generator matrix (Gc or Gb)."""
    if G.shape[1] == 0:
        return torch.zeros((n_out, 0), dtype=dtype, device=device)
    ncols = G.shape[1]
    imgs = G.t().contiguous().view(ncols, C, H, W)      # each column → image
    out = F.conv2d(imgs, weight, bias=None, ...)          # batch conv2d
    return out.permute(1, 2, 3, 0).contiguous().reshape(-1, ncols)  # back to columns
```

**Type / 类型**: New feature / 新功能

**Necessity / 必要性**: Applies `F.conv2d` in batch to all columns of a generator matrix. Used by `_hz_conv2d` for both Gc and Gb, avoiding code duplication. Without this, the same transpose-view-conv-permute-reshape pattern would be written twice (once for Gc, once for Gb). 将 `F.conv2d` 批量应用到生成器矩阵的所有列。`_hz_conv2d` 对 Gc 和 Gb 都调用此函数，避免重复。

### 4.5 New `_hz_conv2d(hz, weight, bias, ...)` / 新增 Conv2d HZ 运算

```python
def _hz_conv2d(hz, weight, bias, stride, padding, dilation, groups, input_shape):
    """Apply conv2d to a hybrid zonotope: convolve center and each generator column."""
    C, H, W = _parse_input_shape(input_shape)
    # conv2d on center (with bias)
    new_c = F.conv2d(hz.c.view(C,H,W).unsqueeze(0), weight, bias=bias, ...).reshape(-1, 1)
    # conv2d on Gc and Gb (without bias, via _conv2d_generators)
    new_Gc = _conv2d_generators(hz.Gc, weight, C, H, W, ...)
    new_Gb = _conv2d_generators(hz.Gb, weight, C, H, W, ...)
    # Constraints unchanged (conv2d is a linear operation)
    return HZono(c=new_c, Gc=new_Gc, Gb=new_Gb, Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone())
```

**Type / 类型**: New feature / 新功能

**Necessity / 必要性**: Core HZ operation for Conv2d layers. Conv2d is a linear operation, so it maps exactly to matrix multiplication on the HZ components: `c'=conv(c), Gc'=conv(Gc), Gb'=conv(Gb)`, with constraints unchanged. Called by `_hz_transform("CONV2D")` in `hybridz_tf.py`. Conv2d 的 HZ 核心运算。卷积是线性操作，精确映射到 HZ 的各分量上。由 `_hz_transform("CONV2D")` 调用。

### 4.6 `hybridz_tf_conv2d`: inline shape parsing → `_parse_input_shape()` / 内联解析 → 提取函数

```python
# Original (main) — 8 lines inline / 原版，8 行内联
if len(input_shape) == 4:
    _, C, H, W = input_shape
elif len(input_shape) == 3:
    C, H, W = input_shape
else:
    raise ValueError(...)

# New — 1 line / 新版，1 行
C, H, W = _parse_input_shape(input_shape)
```

**Type / 类型**: Refactor / 重构

**Necessity / 必要性**: Replaced with the extracted helper function. Logic is identical. 替换为提取的辅助函数，逻辑完全相同。

### 4.7 `hybridz_tf_conv2d`: `cons.add_op()` line wrap / 长行换行

```python
# Original — single long line / 原版，单行过长
cons.add_op(..., input_shape=..., output_shape=...,)

# New — wrapped / 新版，换行
cons.add_op(..., input_shape=...,
output_shape=...,)
```

**Type / 类型**: Formatting only / 纯格式

**Necessity / 必要性**: Line length reduction. No functional change. 减少行长，无功能变化。

### 4.8 `hybridz_tf_maxpool2d`: inline reshape → `_reshape_bounds_4d()` / 内联 reshape → 提取函数

```python
# Original (main) — 6 lines / 原版，6 行
if len(Bin.lb.shape) == 1 and in_shape:
    Bin_lb = Bin.lb.view(1, *in_shape)
    Bin_ub = Bin.ub.view(1, *in_shape)
else:
    Bin_lb = Bin.lb.unsqueeze(0) if len(Bin.lb.shape) == 3 else Bin.lb
    Bin_ub = Bin.ub.unsqueeze(0) if len(Bin.ub.shape) == 3 else Bin.ub

# New — 1 line / 新版，1 行
Bin_lb, Bin_ub = _reshape_bounds_4d(Bin.lb, Bin.ub, in_shape)
```

**Type / 类型**: Refactor / 重构

**Necessity / 必要性**: Same reshape logic existed in both `maxpool2d` and `avgpool2d`. Extracted to shared helper. `maxpool2d` 和 `avgpool2d` 中有相同的 reshape 逻辑，提取为共享辅助函数。

### 4.9 `hybridz_tf_avgpool2d`: same reshape extraction / 同样的 reshape 提取

Identical change as 4.8. Replaced 6-line inline block with `_reshape_bounds_4d()`. 与 4.8 相同，6 行内联块替换为 `_reshape_bounds_4d()`。

### 4.10 `hybridz_tf_avgpool2d`: removed comment / 删除注释

```python
# Original / 原版
    # Average pooling is linear - exact bounds

# New — removed / 新版 — 删除
```

**Type / 类型**: Cleanup / 清理

**Necessity / 必要性**: The comment was obvious from context. Not strictly necessary to remove, but harmless. 注释内容从上下文即可看出，删除无害。

### 4.11 `hybridz_tf_flatten`: `cons.add_op()` line wrap / 长行换行

```python
# Original — single long line / 原版，单行过长
cons.add_op(f"flatten:{L.id}", ..., input_shape=..., output_shape=...)

# New — wrapped / 新版，换行
cons.add_op(f"flatten:{L.id}", ..., start_dim=..., end_dim=...,
input_shape=..., output_shape=...)
```

**Type / 类型**: Formatting only / 纯格式

**Necessity / 必要性**: Line length reduction. No functional change. 减少行长，无功能变化。

### 4.12 `hybridz_tf_reshape`: merge reshape + flatten / 合并 reshape 和 flatten

```python
# Original — 4 lines / 原版，4 行
lb = Bin.lb.reshape(target_shape) if target_shape else Bin.lb
ub = Bin.ub.reshape(target_shape) if target_shape else Bin.ub
lb = lb.flatten()
ub = ub.flatten()

# New — 2 lines / 新版，2 行
lb = Bin.lb.reshape(target_shape).flatten() if target_shape else Bin.lb.flatten()
ub = Bin.ub.reshape(target_shape).flatten() if target_shape else Bin.ub.flatten()
```

**Type / 类型**: Simplification / 简化

**Necessity / 必要性**: Chained `.reshape().flatten()` in one line instead of two separate statements. Functionally identical. 链式调用代替两条独立语句，功能完全相同。

### 4.13 `hybridz_tf_reshape`: `cons.add_op()` line wrap / 长行换行

Same as 4.7 and 4.11 — long line wrapped for readability. 同 4.7 和 4.11，长行换行提高可读性。

### 4.14 All original transfer function signatures — unchanged / 所有原有传递函数签名 — 未改动

`hybridz_tf_conv2d(L, Bin)`, `hybridz_tf_maxpool2d(L, Bin)`, `hybridz_tf_avgpool2d(L, Bin)`, `hybridz_tf_flatten(L, Bin)`, `hybridz_tf_reshape(L, Bin)` — all keep original signature `(L: Layer, Bin: Bounds) -> Fact`. No `tf` parameter added. Internal interval arithmetic logic unchanged.

所有函数保持原始签名 `(L, Bin) -> Fact`，未添加 `tf` 参数，内部区间算术逻辑未变。

### Summary for `tf_cnn.py` / 总结

| Change # | Type / 类型 | Lines changed / 改动行数 | Functional impact / 功能影响 |
|----------|-------------|--------------------------|------------------------------|
| 4.1 | Import / 导入 | +8 -1 | No / 无 |
| 4.2 | Refactor / 重构 | +8 (new function) | No — extracted from existing code / 从现有代码提取 |
| 4.3 | Refactor / 重构 | +7 (new function) | No — extracted from existing code / 从现有代码提取 |
| 4.4 | New feature / 新功能 | +11 (new function) | Yes — enables HZ conv2d / 启用 HZ 卷积 |
| 4.5 | New feature / 新功能 | +18 (new function) | Yes — core HZ conv operation / HZ 卷积核心 |
| 4.6 | Refactor / 重构 | +1 -8 | No — uses extracted helper / 使用提取的辅助函数 |
| 4.7 | Formatting / 格式 | +2 -1 | No / 无 |
| 4.8 | Refactor / 重构 | +1 -6 | No — uses extracted helper / 使用提取的辅助函数 |
| 4.9 | Refactor / 重构 | +1 -6 | No — same as 4.8 / 同 4.8 |
| 4.10 | Cleanup / 清理 | -1 | No / 无 |
| 4.11 | Formatting / 格式 | +2 -1 | No / 无 |
| 4.12 | Simplification / 简化 | +2 -4 | No — functionally identical / 功能相同 |
| 4.13 | Formatting / 格式 | +2 -1 | No / 无 |
| 4.14 | — | 0 | No — signatures unchanged / 签名未变 |

---

## 5. `tf_rnn.py` — Bug Fix / 缺陷修复

### 5.1 `cons.add_lstm()` → `cons.add_op()`

| Original / 原版 | New / 新版 |
|------|------|
| `cons.add_lstm(L.id, L.in_vars, L.out_vars, input_size, hidden_size)` | `cons.add_op(f"lstm:{L.id}", list(L.out_vars + L.in_vars), ...)` |

**Necessity / 必要性**: **Bug fix.** `ConSet` class has no `add_lstm()` method; the original code raises `AttributeError` at runtime. Changed to the generic `add_op()` method. **缺陷修复。** `ConSet` 类中不存在 `add_lstm()` 方法，原代码运行即报 `AttributeError`。改为通用 `add_op()` 方法。

Same fix applied to / 同理修复: `add_gru()`, `add_rnn()`, `add_embedding()`.

### 5.2 Formatting cleanup / 格式清理

Trailing whitespace removed. **Formatting only**, no functional impact. 部分 trailing whitespace 被移除。**仅格式变化**，无功能影响。

---

## 6. `tf_transformer.py` — Bug Fix / 缺陷修复

### 6.1 `cons.add_layernorm()` → `cons.add_op()`

| Original / 原版 | New / 新版 |
|------|------|
| `cons.add_layernorm(...)` | `cons.add_op(f"layernorm:{L.id}", ...)` |

**Necessity / 必要性**: **Bug fix.** Same issue as tf_rnn.py — these methods don't exist in `ConSet`. **缺陷修复。** 同 tf_rnn.py，`ConSet` 中不存在这些方法。

Same fix applied to / 同理修复: `add_gelu()`, `add_simplex()`, `add_posenc()`, `add_attention_scores()`.

### 6.2 List initialization simplification in `gelu_approx` / 列表初始化简化

```python
# Original / 原版
y_candidates = []
y_candidates.append(gelu_approx(x_min))
y_candidates.append(gelu_approx(x_max))

# New / 新版
y_candidates = [gelu_approx(x_min), gelu_approx(x_max)]
```

**Necessity / 必要性**: Code style simplification, no functional change. 代码风格简化，无功能变化。

### 6.3 Removed unused variables / 删除未使用变量

```python
# Original / 原版
gelu_values = torch.tensor([-0.0, -0.159, 0.0, 0.841, 3.0], ...)  # never used / 从未使用
input_range = Bin.ub - Bin.lb  # never used in softmax / softmax 中从未使用
```

**Necessity / 必要性**: Dead code removal. 消除无用代码。

---

## 7. `layer_schema.py`

### 7.1 Added `"lstm", "gru", "rnn", "embedding"` to `SUPPORTED_EXPORT_OPS` / 新增到白名单

**Necessity / 必要性**: `cons.add_op(f"lstm:{L.id}", ...)` requires `"lstm"` in the whitelist, otherwise export is rejected. This is a companion change to the tf_rnn.py bug fix. `cons.add_op(f"lstm:{L.id}", ...)` 需要 `"lstm"` 在白名单中。这是 tf_rnn.py 缺陷修复的配套改动。

---

## Summary / 改动分类总结

| Type / 类型 | Count / 数量 | Description / 说明 |
|------|------|------|
| **Bug fix / 缺陷修复** | 9 | tf_rnn.py (4) + tf_transformer.py (5): non-existent method calls / 不存在的方法调用 |
| **New feature / 新功能** | ~20 functions | HZono + _hz_* operations + bounds computation + apply() refactor / HZ 运算 + bounds 计算 + apply() 改造 |
| **Companion fix / 配套修复** | 1 | layer_schema.py whitelist / 白名单 |
| **Code cleanup / 代码清理** | 3 | Dead code removal + list simplification / 无用代码删除 + 列表简化 |
| **Formatting only / 纯格式** | ~20 | Trailing whitespace cleanup / trailing whitespace 清理 |
