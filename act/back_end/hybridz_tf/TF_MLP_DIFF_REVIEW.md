# TF_MLP_DIFF_REVIEW.md -- Comprehensive Diff Review for `tf_mlp.py`

**File**: `act/back_end/hybridz_tf/tf_mlp.py`
**Branch**: `hyzor-gurobi` vs `main`
**Growth**: 243 lines (main) -> 1130 lines (current), +887 lines net (+365%)
**Review Date**: 2026-03-27

---

## Table of Contents

1. [High-Level Architecture Changes / 高层架构变更](#1-high-level-architecture-changes)
2. [Import Section (Lines 16-48) / 导入部分](#2-import-section-lines-16-48)
3. [Basic Linear Operations (Lines 51-66) / 基本线性操作](#3-basic-linear-operations-lines-51-66)
4. [Bounds Computation Dispatch (Lines 69-109) / 边界计算调度](#4-bounds-computation-dispatch-lines-69-109)
5. [Solver Backends (Lines 112-204) / 求解器后端](#5-solver-backends-lines-112-204)
6. [ReLU Activation Encoding (Lines 209-314) / ReLU激活编码](#6-relu-activation-encoding-lines-209-314)
7. [LeakyReLU Activation Encoding (Lines 317-454) / LeakyReLU激活编码](#7-leakyrelu-activation-encoding-lines-317-454)
8. [Piecewise Sigmoid/Tanh Encoding (Lines 457-760) / 分段Sigmoid/Tanh编码](#8-piecewise-sigmoidtanh-encoding-lines-457-760)
9. [Minkowski Sum (Lines 765-795) / 闵可夫斯基和](#9-minkowski-sum-lines-765-795)
10. [Fresh HZ from Bounds (Lines 798-810) / 从边界创建新HZ](#10-fresh-hz-from-bounds-lines-798-810)
11. [Complexity Reduction (Lines 815-913) / 复杂度约简](#11-complexity-reduction-lines-815-913)
12. [Modified Transfer Functions (Lines 916-1131) / 修改后的传递函数](#12-modified-transfer-functions-lines-916-1131)
13. [Summary Table / 总结表](#13-summary-table)

---

## 1. High-Level Architecture Changes

### English

The file was originally a flat collection of transfer functions that computed interval bounds and generated `ConSet` constraints. Each transfer function took `(Layer, Bounds)` and returned `Fact`.

In the current branch, the file is split into two logical halves:

1. **Lines 16-913**: A complete hybrid zonotope (HZ) computational library operating directly on `HZono` objects. This includes linear algebra primitives, solver-backed bounds computation, exact nonlinear activation encodings (ReLU, LeakyReLU, piecewise sigmoid/tanh), set operations (Minkowski sum), and complexity reduction. All constraints use **equalities only** (`Ac * xi_c + Ab * xi_b = b`), following the paper's Definition 3.1.

2. **Lines 916-1131**: The original transfer functions (largely unchanged), which continue to operate on interval `Bounds` and produce `Fact` objects. These serve as the fallback / bounds-only path; the actual HZ processing is delegated to `hybridz_tf.py`'s `apply()` method which calls the `_hz_*` functions.

### Chinese / 中文

原始文件是一组平坦的传递函数集合，计算区间边界并生成`ConSet`约束。每个传递函数接收`(Layer, Bounds)`并返回`Fact`。

在当前分支中，文件被分为两个逻辑部分：

1. **第16-913行**：完整的混合多面体(HZ)计算库，直接操作`HZono`对象。包括线性代数基元、基于求解器的边界计算、精确非线性激活编码(ReLU、LeakyReLU、分段sigmoid/tanh)、集合运算(闵可夫斯基和)和复杂度约简。所有约束仅使用**等式**(`Ac * xi_c + Ab * xi_b = b`)，遵循论文定义3.1。

2. **第916-1131行**：原始传递函数(基本未变)，继续在区间`Bounds`上操作并产生`Fact`对象。这些作为后备/仅边界路径；实际HZ处理委托给`hybridz_tf.py`的`apply()`方法调用`_hz_*`函数。

---

## 2. Import Section (Lines 16-48)

**Change Type / 变更类型**: New feature + Refactor

### Before (main)

```python
import torch
from typing import Optional
from act.back_end.core import Bounds, Fact, Layer, ConSet
```

### After (current)

```python
from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from act.back_end.core import Bounds, Fact, Layer, ConSet

if TYPE_CHECKING:
    from act.back_end.hybridz_tf.hybridz_tf import HZono

# Lazy import to avoid circular dependency (hybridz_tf.py imports tf_mlp.py)
_HZono = None

def _get_HZono():
    global _HZono
    if _HZono is None:
        from act.back_end.hybridz_tf.hybridz_tf import HZono as _cls
        _HZono = _cls
    return _HZono

# Optional solver imports
try:
    import gurobipy as gp
    from gurobipy import GRB
    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False

try:
    import numpy as np
    from scipy.optimize import linprog
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
```

### Detailed Changes / 详细变更

| # | Change | Type | Description (EN) | Description (CN) |
|---|--------|------|-------------------|-------------------|
| 2.1 | `from __future__ import annotations` added (line 16) | New feature | Enables PEP 604 `X | Y` union syntax and deferred evaluation of type annotations. Necessary because `HZono` is referenced in function signatures before it is importable at module load time. | 启用PEP 604 `X | Y`联合语法和延迟类型注解求值。因为`HZono`在模块加载时无法导入，但需要在函数签名中引用。 |
| 2.2 | `Optional` replaced by `TYPE_CHECKING` (line 19) | Refactor | `Optional` was imported but never used in main. `TYPE_CHECKING` is now needed for the conditional HZono import. | `Optional`在main中导入但从未使用。现在需要`TYPE_CHECKING`进行条件式HZono导入。 |
| 2.3 | Conditional `HZono` import (lines 22-23) | New feature | `HZono` is imported only during static type checking (mypy/pyright), avoiding circular import at runtime since `hybridz_tf.py` already imports `tf_mlp.py`. | `HZono`仅在静态类型检查(mypy/pyright)时导入，避免运行时循环导入，因为`hybridz_tf.py`已经导入了`tf_mlp.py`。 |
| 2.4 | Lazy `_get_HZono()` pattern (lines 26-33) | New feature | Singleton lazy-loader: first call triggers actual import, subsequent calls return cached class. This cleanly solves the circular dependency `hybridz_tf.py <-> tf_mlp.py`. Uses `global` for module-level caching. | 单例懒加载器：首次调用触发实际导入，后续调用返回缓存类。干净地解决了`hybridz_tf.py <-> tf_mlp.py`的循环依赖。使用`global`进行模块级缓存。 |
| 2.5 | Optional Gurobi import (lines 36-41) | New feature | Guarded import of `gurobipy`. Sets `_HAS_GUROBI` flag. Gurobi is a commercial MILP solver (free academic license) used for exact HZ bounds computation. | 受保护的`gurobipy`导入。设置`_HAS_GUROBI`标志。Gurobi是商业MILP求解器(免费学术许可证)，用于精确HZ边界计算。 |
| 2.6 | Optional SciPy import (lines 43-48) | New feature | Guarded import of `numpy` and `scipy.optimize.linprog`. Sets `_HAS_SCIPY` flag. SciPy's HiGHS LP solver serves as fallback when Gurobi is unavailable. | 受保护的`numpy`和`scipy.optimize.linprog`导入。设置`_HAS_SCIPY`标志。SciPy的HiGHS LP求解器在Gurobi不可用时作为后备。 |

### Design Decision / 设计决策

The lazy import pattern is essential because `hybridz_tf.py` (which defines `HZono`) imports this file's transfer functions. A direct import would create a circular dependency crash. The `_get_HZono()` function defers the import to first use, by which point both modules are fully loaded.

懒加载导入模式是必要的，因为定义`HZono`的`hybridz_tf.py`会导入本文件的传递函数。直接导入会导致循环依赖崩溃。`_get_HZono()`函数将导入延迟到首次使用时，此时两个模块都已完全加载。

**`_get_HZono()` call sites / 调用点** (11 call sites throughout tf_mlp.py, avoids repeating the lazy import logic / 在 tf_mlp.py 中有 11 个调用点，避免重复懒加载逻辑):
- `_hz_multiply` (line 58), `_hz_add_const` (line 65) — construct output HZono
- `_hz_apply_relu` (lines 251, 313) — construct stable/unstable output
- `_hz_apply_leaky_relu` (lines 365, 453) — same pattern
- `_hz_apply_piecewise` (lines 504, 747) — narrow/wide neuron output
- `_hz_minkowski_sum` (line 795), `_hz_from_bounds_fresh` (line 810), `_hz_reduce` (lines 849, 910)

---

## 3. Basic Linear Operations (Lines 51-66)

**Change Type / 变更类型**: New feature

### `_hz_multiply(hz, R)` -- Lines 55-58

**English**: Applies a linear map `R` to a hybrid zonotope. Transforms center `c -> R@c`, continuous generators `Gc -> R@Gc`, binary generators `Gb -> R@Gb`, while leaving all equality constraints `(Ac, Ab, b)` unchanged. This is mathematically exact because the constraint system operates on the generator coefficients (xi_c, xi_b), which are unaffected by output-space linear maps.

**Chinese / 中文**: 对混合多面体施加线性映射`R`。变换中心`c -> R@c`、连续生成器`Gc -> R@Gc`、二值生成器`Gb -> R@Gb`，同时保持所有等式约束`(Ac, Ab, b)`不变。这在数学上是精确的，因为约束系统作用于生成器系数(xi_c, xi_b)，不受输出空间线性映射影响。

**Mathematical basis**: If `Z = {c + Gc*xi_c + Gb*xi_b | Ac*xi_c + Ab*xi_b = b, xi_c in [-1,1]^ng, xi_b in {-1,1}^nb}`, then `R*Z = {R*c + R*Gc*xi_c + R*Gb*xi_b | same constraints}`.

**Code notes**:
- `R` is cast to match `hz.c.dtype` and `hz.c.device` for safety.
- All constraint matrices are `.clone()`d to prevent aliasing bugs.

**Called by / 调用者** (avoids duplicating linear map logic in each caller / 避免在每个调用者中重复线性映射逻辑):
- `hybridz_tf.py:132` — `_hz_transform("DENSE")`: `hz = _hz_multiply(hz_in, L.params["weight"])`
- `hybridz_tf.py:143` — `_hz_transform("SCALE")`: `return _hz_multiply(hz_in, torch.diag(a))`

### `_hz_add_const(hz, v)` -- Lines 60-65

**English**: Translates the center of a hybrid zonotope by a constant vector `v`. Only `c` changes; generators and constraints are unaffected. Handles both 1D and 2D inputs via `.view(-1, 1)` reshape.

**Chinese / 中文**: 将混合多面体的中心平移常数向量`v`。只有`c`改变；生成器和约束不受影响。通过`.view(-1, 1)`变形处理1D和2D输入。

**Called by / 调用者** (avoids duplicating translation logic / 避免重复平移逻辑):
- `hybridz_tf.py:136` — `_hz_transform("DENSE")`: bias addition after `_hz_multiply`
- `hybridz_tf.py:140` — `_hz_transform("BIAS")`: `return _hz_add_const(hz_in, c)`

---

## 4. Bounds Computation Dispatch (Lines 69-109)

**Change Type / 变更类型**: New feature

### `_hz_is_unconstrained(hz)` -- Lines 71-78

**English**: Checks whether a hybrid zonotope has no active equality constraints by testing if `Ac`, `Ab`, and `b` are all near-zero (tolerance `1e-12`). An unconstrained HZ is equivalent to a classical zonotope, enabling a fast closed-form bounds computation.

**Chinese / 中文**: 通过测试`Ac`、`Ab`和`b`是否全部接近零(容差`1e-12`)来检查混合多面体是否没有活跃的等式约束。无约束HZ等价于经典仿射集，可以使用快速闭式边界计算。

**Called by / 调用者**: `_hz_compute_bounds` (line 95) — guards the fast path to avoid unnecessary Gurobi/SciPy calls. 由 `_hz_compute_bounds`（第95行）调用，保护快速路径以避免不必要的 Gurobi/SciPy 调用。

### `_hz_bounds_unconstrained(hz)` -- Lines 81-90

**English**: Fast closed-form bounds for unconstrained HZ (classical zonotope). The key formula is:

```
lb = c - |Gc|_rowsum - |Gb|_rowsum
ub = c + |Gc|_rowsum + |Gb|_rowsum
```

This exploits the fact that `xi_c in [-1,1]` and `xi_b in {-1,1}` both have magnitude at most 1, so the worst-case deviation from center in each dimension is the sum of absolute generator values.

**Chinese / 中文**: 无约束HZ(经典仿射集)的快速闭式边界。关键公式：

```
lb = c - |Gc|行和 - |Gb|行和
ub = c + |Gc|行和 + |Gb|行和
```

利用`xi_c in [-1,1]`和`xi_b in {-1,1}`幅值最多为1的事实，因此每个维度偏离中心的最坏情况是生成器绝对值的行和。

**Called by / 调用者**:
- `_hz_compute_bounds` (line 96) — fast path when no constraints exist / 无约束时的快速路径
- `_hz_compute_bounds` (line 109) — ultimate fallback when both Gurobi and SciPy fail / Gurobi 和 SciPy 都失败时的最终回退

**Code notes**:
- Empty generator matrices (`hz.Gc.numel() == 0`) are handled with zero fallback.
- Result is flattened to 1D for `Bounds` compatibility.

### `_hz_compute_bounds(hz)` -- Lines 93-109

**English**: Dispatcher function implementing a cascading fallback strategy for bounds computation:

1. **Unconstrained fast path**: If no active constraints, use closed-form `_hz_bounds_unconstrained`.
2. **Gurobi MILP** (exact): If Gurobi is available, solve the constrained optimization exactly via `_hz_compute_bounds_gurobi`.
3. **SciPy LP relaxation** (sound over-approximation): If SciPy is available, relax binary generators to continuous and solve LP via `_hz_compute_bounds_scipy`.
4. **Unconstrained fallback**: If all solvers fail, fall back to the unconstrained over-approximation (ignoring constraints entirely).

Each solver call is wrapped in `try/except Exception: pass` to ensure graceful degradation.

**Chinese / 中文**: 调度函数，实现边界计算的级联后备策略：

1. **无约束快速路径**：如果没有活跃约束，使用闭式`_hz_bounds_unconstrained`。
2. **Gurobi MILP**（精确）：如果Gurobi可用，通过`_hz_compute_bounds_gurobi`精确求解约束优化。
3. **SciPy LP松弛**（可靠过近似）：如果SciPy可用，将二值生成器松弛为连续并通过`_hz_compute_bounds_scipy`求解LP。
4. **无约束后备**：如果所有求解器失败，回退到无约束过近似(完全忽略约束)。

每个求解器调用都包装在`try/except Exception: pass`中以确保优雅降级。

**Called by / 调用者** (centralizes solver selection to avoid repeating the dispatch logic / 集中求解器选择以避免重复调度逻辑):
- `hybridz_tf.py:230` — `apply()`: compute HZ bounds after `_hz_transform` / `apply()` 中 `_hz_transform` 后计算 HZ bounds
- `hybridz_tf.py:153,164,186` — `_hz_transform(ABS/MAXPOOL2D/MUL)`: compute bounds for interval-based ops / 计算基于区间的运算的 bounds
- `tf_mlp.py:229,338,483` — `_hz_apply_relu/_hz_apply_leaky_relu/_hz_apply_piecewise`: compute input bounds for neuron classification / 计算输入 bounds 用于神经元分类
- `tf_cnn.py:28` — imported for `_hz_conv2d` internal use / 由 `_hz_conv2d` 内部使用导入

**Design note**: The priority order (Gurobi > SciPy > unconstrained) reflects a precision-vs-availability tradeoff. Gurobi gives exact bounds but requires a commercial license; SciPy LP is always available on pip but only gives a relaxation.

**设计说明**：优先级顺序(Gurobi > SciPy > 无约束)反映了精度与可用性的权衡。Gurobi给出精确边界但需要商业许可证；SciPy LP总是可通过pip获得但只给出松弛。

---

## 5. Solver Backends (Lines 112-204)

**Change Type / 变更类型**: New feature

### `_hz_compute_bounds_gurobi(hz)` -- Lines 112-165

**English**: Computes exact bounds of a constrained hybrid zonotope by formulating and solving a Mixed-Integer Linear Program (MILP) in Gurobi.

**Mathematical formulation**: For each dimension `i`, solve:

```
max/min  c_i + Gc[i,:] @ xi_c + Gb[i,:] @ xi_b
s.t.     Ac @ xi_c + Ab @ xi_b = b
         xi_c in [-1, 1]^ng
         xi_b in {-1, 1}^nb
```

The binary constraint `xi_b in {-1,1}` is implemented via the substitution `xi_b = 2*zeta - 1` where `zeta in {0,1}^nb` is a standard binary variable. This is a standard MILP reformulation.

**Code structure**:
1. Convert all HZ tensors to numpy float64 (lines 114-119).
2. Create Gurobi model with suppressed output (`OutputFlag=0`) (lines 125-126).
3. Add continuous variables `xi_c in [-1,1]` and binary variables `zeta in {0,1}` (lines 128-130).
4. Add equality constraints `Ac @ xi_c + Ab @ (2*zeta-1) = b` (lines 132-138).
5. Loop over dimensions, solving MAX then MIN for each (lines 143-160).
6. Convert results back to torch tensors (lines 162-165).

**Chinese / 中文**: 通过在Gurobi中构建和求解混合整数线性规划(MILP)来计算约束混合多面体的精确边界。

**数学表达式**：对每个维度`i`，求解：

```
max/min  c_i + Gc[i,:] @ xi_c + Gb[i,:] @ xi_b
s.t.     Ac @ xi_c + Ab @ xi_b = b
         xi_c in [-1, 1]^ng
         xi_b in {-1, 1}^nb
```

二值约束`xi_b in {-1,1}`通过替换`xi_b = 2*zeta - 1`实现，其中`zeta in {0,1}^nb`是标准二值变量。这是标准MILP重新表述。

**Performance note / 性能说明**: The MILP is solved `2*n` times (once max, once min per dimension). For large networks this is expensive (NP-hard per solve). In practice, the equality constraints added by ReLU/LeakyReLU make the MILP tractable because the branching structure is sparse.

对于大型网络，MILP求解`2*n`次(每个维度一次max、一次min)代价昂贵(每次求解都是NP-hard的)。实际上，ReLU/LeakyReLU添加的等式约束使MILP可处理，因为分支结构是稀疏的。

### `_hz_compute_bounds_scipy(hz)` -- Lines 168-204

**English**: Computes bounds via LP relaxation using SciPy's HiGHS solver. This is a **sound over-approximation**: it relaxes binary generators `xi_b in {-1,1}` to `xi_b in [-1,1]`, converting the MILP into a standard LP. The LP relaxation always produces bounds that are wider than or equal to the exact bounds, so soundness is preserved.

**Mathematical formulation**: Same as Gurobi but with all variables continuous in `[-1,1]`:

```
max/min  c_i + [Gc[i,:], Gb[i,:]] @ [xi_c; xi_b]
s.t.     [Ac, Ab] @ [xi_c; xi_b] = b
         [xi_c; xi_b] in [-1, 1]^(ng+nb)
```

**Code structure**:
1. Convert tensors to numpy (lines 174-179).
2. Concatenate `Ac` and `Ab` horizontally for equality constraint matrix (line 181).
3. Set all variable bounds to `(-1, 1)` -- note this relaxes the binary constraint (line 183).
4. For each dimension, solve `linprog` for min (direct) and max (negate objective) (lines 188-199).
5. Convert back to torch (lines 201-204).

**Chinese / 中文**: 使用SciPy的HiGHS求解器通过LP松弛计算边界。这是一个**可靠的过近似**：它将二值生成器`xi_b in {-1,1}`松弛为`xi_b in [-1,1]`，将MILP转换为标准LP。LP松弛总是产生不窄于精确边界的边界，因此保持了可靠性。

**Key difference from Gurobi / 与Gurobi的关键区别**:
- LP relaxation (polynomial time) vs MILP (NP-hard).
- Bounds are potentially wider (looser) but always sound.
- LP松弛(多项式时间) vs MILP(NP-hard)。
- 边界可能更宽(更松)但始终可靠。

---

## 6. ReLU Activation Encoding (Lines 209-314)

**Change Type / 变更类型**: New feature

### `_hz_apply_relu(hz)` -- Lines 209-314

**English**: Implements exact ReLU encoding for hybrid zonotopes using equality constraints. This follows the `applyReLU_eq_native_exact` style from the paper (Definition 3.1, Section 3.2).

**Design philosophy**: Rather than using triangle relaxation (which introduces over-approximation), this encodes ReLU exactly using binary generators that select between the active branch (`y=x`) and inactive branch (`y=0`).

**Per-neuron cost (unstable neurons only)**:
- `+4` continuous generators: `xi_1, xi_2, xi_3, xi_4`
- `+1` binary generator: `z in {-1, +1}`
- `+3` equality constraints

**Mathematical encoding** for unstable neuron `i` with bounds `[alpha, beta]` where `alpha < 0 < beta`:

```
Equality 1:  xi_1 + xi_3 + z = 1
Equality 2:  xi_2 + xi_4 - z = 1
Equality 3:  (alpha/2)*xi_1 - (beta/2)*xi_2 + (alpha/2)*z - Gc[i]*xi_old - Gb[i]*xi_b_old = c_i - beta/2
```

**How it works**:
- When `z = +1` (inactive branch, x <= 0):
  - Eq 1: `xi_1 + xi_3 = 0` and Eq 2: `xi_2 + xi_4 = 2` -> `xi_2 = 1` forced (since `xi_2 in [-1,1]` and `xi_4 in [-1,1]`)
  - Output: `y_i = beta/2 + (-beta/2)*1 = 0` (correct: ReLU(x) = 0 for x <= 0)
- When `z = -1` (active branch, x >= 0):
  - Eq 1: `xi_1 + xi_3 = 2` -> `xi_1 = 1` forced
  - Linking equality ensures `y = x` via the original generators

**Output construction**:
- Active neurons (`lb >= 0`): Copy original center and generators (identity map).
- Inactive neurons (`ub <= 0`): Zero center and generators (zero map).
- Unstable neurons: Center = `beta/2`, single generator `xi_2` with coefficient `-beta/2`.

**Chinese / 中文**: 使用等式约束为混合多面体实现精确ReLU编码。遵循论文中`applyReLU_eq_native_exact`风格(定义3.1，第3.2节)。

**设计哲学**：不使用三角松弛(会引入过近似)，而是使用二值生成器精确编码ReLU，在活跃分支(`y=x`)和非活跃分支(`y=0`)之间选择。

**每个神经元的代价(仅不稳定神经元)**：
- `+4`个连续生成器：`xi_1, xi_2, xi_3, xi_4`
- `+1`个二值生成器：`z in {-1, +1}`
- `+3`个等式约束

**工作原理**：
- 当`z = +1`(非活跃分支，x <= 0)：
  - 等式1: `xi_1 + xi_3 = 0`，等式2: `xi_2 + xi_4 = 2` -> 强制`xi_2 = 1`
  - 输出: `y_i = beta/2 + (-beta/2)*1 = 0`（正确：x <= 0时ReLU(x) = 0）
- 当`z = -1`(活跃分支，x >= 0)：
  - 等式1: `xi_1 + xi_3 = 2` -> 强制`xi_1 = 1`
  - 链接等式通过原始生成器确保`y = x`

**Called by / 调用者**:
- `hybridz_tf.py:145` — `_hz_transform("RELU")`: `return _hz_reduce(_hz_apply_relu(hz_in))`

**Code structure overview**:
1. Compute bounds via `_hz_compute_bounds` (line 229).
2. Classify neurons: active/inactive/unstable (lines 233-236).
3. Initialize output matrices `out_Gc`, `out_Gb`, `out_c` (lines 239-241).
4. Handle stable neurons (lines 244-252).
5. Set up unstable neurons: compute column indices, output center `beta/2`, generator `-beta/2*xi_2` (lines 255-263).
6. Build 3k equality constraint rows (lines 271-303).
7. Extend old constraints to new column dimensions and concatenate (lines 305-314).

---

## 7. LeakyReLU Activation Encoding (Lines 317-454)

**Change Type / 变更类型**: New feature

### `_hz_apply_leaky_relu(hz, alpha_arg)` -- Lines 317-454

**English**: Implements exact LeakyReLU encoding using the paper's Definition 3.4 union approach with box equalities and slack generators. More complex than ReLU because the inactive branch is `y = alpha*x` (not zero), requiring the encoding to preserve the input's linear structure in both branches.

**Per-neuron cost (unstable neurons only)**:
- `+6` continuous generators: `g1, g2` (real) + `s1+, s1-, s2+, s2-` (slack)
- `+1` binary generator: `z in {-1, +1}`
- `+5` equality constraints

**Mathematical encoding** for unstable neuron `i` with bounds `[l, u]` where `l < 0 < u`:

```
Box eq 0:  g1 + s1+ + 0.5*z = 0.5     (g1 active when z=+1)
Box eq 1: -g1 + s1- + 0.5*z = 0.5
Box eq 2:  g2 + s2+ - 0.5*z = 0.5     (g2 active when z=-1)
Box eq 3: -g2 + s2- - 0.5*z = 0.5
Linking:   Gc[i]*xi + Gb[i]*zeta - (l/2)*g1 + (u/2)*g2 - ((l-u)/4)*z = (u+l)/4 - c_i
```

**How box equalities work (key insight)**:

The box equality pattern `g + s + 0.5*z = 0.5` with both `g, s in [-1,1]` and `z in {-1,1}` implements a conditional activation:
- When `z = -1`: `g + s = 1`, so `g + s = 1` with `g, s in [-1,1]`. Since `g <= 1` and `s <= 1`, and `g + s = 1`, both are forced to be non-negative. Combined with `(-g) + s' = 1`, this forces `g = 0`.
- When `z = +1`: `g + s = 0`, which allows `g in [-1, 1]` freely (with `s = -g`).

This is the "box equality" technique from the paper (Def 3.4) that implements conditional generator activation without inequalities.

**Branch semantics**:
- `z = -1` (active, x >= 0): `g1 = 0`, `g2` free -> `y = x in [0, u]`
- `z = +1` (inactive, x < 0): `g2 = 0`, `g1` free -> `y = alpha*x in [alpha*l, 0]`

**Output for unstable neurons**:
```python
c_y = (u + alpha*l) / 4
out_Gc[i, col_g1] = alpha*l / 2    # inactive branch contribution
out_Gc[i, col_g2] = -u / 2         # active branch contribution
out_Gb[i, col_z]  = (alpha*l - u) / 4  # binary selector
```

**Chinese / 中文**: 使用论文定义3.4的联合方法和盒等式与松弛生成器实现精确LeakyReLU编码。比ReLU更复杂，因为非活跃分支是`y = alpha*x`(不是零)，需要编码在两个分支中保持输入的线性结构。

**盒等式如何工作(关键洞察)**：

盒等式模式`g + s + 0.5*z = 0.5`，其中`g, s in [-1,1]`和`z in {-1,1}`实现条件激活：
- 当`z = -1`时：`g + s = 1`，强制`g = 0`（因为结合`-g + s' = 1`）。
- 当`z = +1`时：`g + s = 0`，允许`g`自由取值。

这是论文中的"盒等式"技术(定义3.4)，无需不等式即可实现条件生成器激活。

**Called by / 调用者**:
- `hybridz_tf.py:147` — `_hz_transform("LRELU")`: `return _hz_reduce(_hz_apply_leaky_relu(hz_in, alpha))`

**Code structure overview**:
1. Compute bounds, classify neurons (lines 338-345).
2. Handle stable neurons: active -> identity, inactive -> scale by alpha (lines 347-366).
3. Unstable: set up 6 column blocks for generators + 1 binary column (lines 369-381).
4. Compute output center and generator coefficients (lines 383-394).
5. Build 5k constraint rows (lines 396-443).
6. Extend and concatenate constraints (lines 445-454).

---

## 8. Piecewise Sigmoid/Tanh Encoding (Lines 457-760)

**Change Type / 变更类型**: New feature

### `_hz_apply_piecewise(hz, func, dfunc, K)` -- Lines 457-748

**English**: The most complex new function. Implements a piecewise-linear approximation of monotone activations (sigmoid, tanh) using the **tangent parallelogram** method (paper Theorem 3.8). Each neuron's input range `[lb, ub]` is divided into `K` pieces, and each piece is enclosed by a parallelogram formed by tangent lines at the piece endpoints.

**Design philosophy**: Classical zonotope approaches use a single secant line + error rectangle, which is too loose for sigmoid/tanh on wide intervals. The tangent parallelogram method:
1. Divides the interval into `K` pieces for finer granularity.
2. Uses tangent lines at piece endpoints instead of secant, giving a tighter enclosure.
3. Uses box equalities to select exactly one active piece via binary generators.

**Per wide neuron cost**:
- `+6K` continuous generators: per piece `k`, `g1_k, g2_k` (real) + `s1+, s1-, s2+, s2-` (slack)
- `+K` binary generators: `z_k` for piece selection
- `+(4K+2)` equality constraints: `4K` box equalities + 1 linking + 1 exactly-one

**Mathematical basis -- Tangent parallelogram**:

For piece `[a, b]` of a monotone function `f`:
1. Compute tangent slopes: `la = f'(a)`, `lb_slope = f'(b)`.
2. Find intersection point of tangent lines: `p1 = (f(b) - f(a) + lb_slope*a - la*b) / (lb_slope - la)`.
3. Form parallelogram with vertices at `(a, f(a))`, `(p1, tangent_b(p1))`, `(b, f(b))`, `(p2, tangent_a(p2))`.
4. Generator `g1` runs along one tangent, `g2` along the other.

**Fallback for nearly-linear pieces** (lines 549-562):
When `|la - lb_slope| < 1e-10`, the tangent parallelogram degenerates. A secant + sampled-error-rectangle fallback is used:
- `g1` along the secant direction.
- `g2` orthogonal with magnitude = max residual sampled at 50 points.

**Soundness verification** (lines 570-595):
After computing generators, 50 sample points are checked for containment. If any point falls outside the parallelogram (xi coefficients exceed 1), generators are scaled up by `max_xi * 1.01` (1% safety buffer). This is a runtime soundness guard.

**Constraint structure** (`(4K+2)` rows per wide neuron):
1. **Box equalities** (4K rows): For each piece k, 4 equalities enforce conditional activation of `g1_k` and `g2_k` via binary `z_k`:
   ```
   g1 + s1+ - 0.5*z_k = 0.5
   -g1 + s1- - 0.5*z_k = 0.5
   g2 + s2+ - 0.5*z_k = 0.5
   -g2 + s2- - 0.5*z_k = 0.5
   ```
   Note: sign is `-0.5*z_k` (active when `z_k = -1`).

2. **Linking equality** (1 row): Ensures input `x_i` equals the piecewise-encoded value:
   ```
   Gc[i]*xi + Gb[i]*zeta - sum_k(g1x*g1 + g2x*g2) + sum_k(cx_k/2)*z_k = sum_k(cx_k/2) - c_i
   ```

3. **Exactly-one constraint** (1 row): Ensures exactly one piece is active:
   ```
   sum_k z_k = K - 2
   ```
   Since `z_k in {-1, +1}`, exactly one `z_k = -1` (active) means `sum = K - 2`.

**Chinese / 中文**: 最复杂的新函数。使用**切线平行四边形**方法(论文定理3.8)实现单调激活函数(sigmoid、tanh)的分段线性近似。每个神经元的输入范围`[lb, ub]`被分成`K`段，每段由端点处切线形成的平行四边形包围。

**设计哲学**：经典仿射集方法使用单一割线+误差矩形，对于宽区间上的sigmoid/tanh过于宽松。切线平行四边形方法：
1. 将区间分为`K`段以获得更细粒度。
2. 使用端点处的切线而非割线，给出更紧的包围。
3. 使用盒等式通过二值生成器精确选择一个活跃段。

**可靠性验证**（第570-595行）：
计算生成器后，检查50个采样点是否包含在内。如果任何点超出平行四边形(xi系数超过1)，生成器按`max_xi * 1.01`放大(1%安全缓冲)。这是运行时可靠性保护。

**约束结构**(`(4K+2)`行/宽神经元)：
1. **盒等式**(4K行)：通过二值`z_k`条件激活`g1_k`和`g2_k`。
2. **链接等式**(1行)：确保输入`x_i`等于分段编码值。
3. **恰好一个约束**(1行)：确保恰好一段活跃(`sum_k z_k = K - 2`)。

### `_hz_apply_sigmoid(hz, K)` -- Lines 751-754

**English**: Thin wrapper that calls `_hz_apply_piecewise` with `func=torch.sigmoid` and `dfunc=sigmoid*(1-sigmoid)`. Avoids duplicating the piecewise encoding logic for different activation functions.

**Chinese / 中文**: 薄包装器，以`func=torch.sigmoid`和`dfunc=sigmoid*(1-sigmoid)`调用`_hz_apply_piecewise`。避免为不同激活函数重复分段编码逻辑。

**Called by / 调用者**: `hybridz_tf.py:151` — `_hz_transform("SIGMOID")`

### `_hz_apply_tanh(hz, K)` -- Lines 757-760

**English**: Thin wrapper that calls `_hz_apply_piecewise` with `func=torch.tanh` and `dfunc=1-tanh^2`. Same design as `_hz_apply_sigmoid` — shares the common `_hz_apply_piecewise` implementation.

**Chinese / 中文**: 薄包装器，以`func=torch.tanh`和`dfunc=1-tanh^2`调用`_hz_apply_piecewise`。与 `_hz_apply_sigmoid` 相同设计——共享通用的 `_hz_apply_piecewise` 实现。

**Called by / 调用者**: `hybridz_tf.py:149` — `_hz_transform("TANH")`

---

## 9. Minkowski Sum (Lines 765-795)

**Change Type / 变更类型**: New feature

### `_hz_minkowski_sum(hz1, hz2)` -- Lines 765-795

**English**: Computes the Minkowski sum of two hybrid zonotopes. The result represents the set `{x + y | x in hz1, y in hz2}`. Used for ADD layer processing in HZ mode.

**Mathematical construction**:
- Center: `c1 + c2`
- Generators: horizontal concatenation `[Gc1, Gc2]` and `[Gb1, Gb2]` (generators from both sets are independent)
- Constraints: block-diagonal structure to maintain independence:
  ```
  Ac = [Ac1  0  ]    Ab = [Ab1  0  ]    b = [b1]
       [ 0  Ac2 ]         [ 0  Ab2 ]        [b2]
  ```

**Chinese / 中文**: 计算两个混合多面体的闵可夫斯基和。结果表示集合`{x + y | x in hz1, y in hz2}`。用于HZ模式下的ADD层处理。

**数学构造**：
- 中心：`c1 + c2`
- 生成器：水平拼接`[Gc1, Gc2]`和`[Gb1, Gb2]`（两个集合的生成器独立）
- 约束：块对角结构以保持独立性

**Called by / 调用者**: `hybridz_tf.py:181` — `_hz_transform("ADD")`: `return _hz_minkowski_sum(hz_in, hz2) if hz2 is not None else None`

**Code notes**:
- Device/dtype casting via `.to()` on all `hz2` tensors ensures compatibility.
- Block-diagonal constraint assembly uses `torch.cat` with zero-padding blocks.

---

## 10. Fresh HZ from Bounds (Lines 798-810)

**Change Type / 变更类型**: New feature

### `_hz_from_bounds_fresh(bounds, dtype, device)` -- Lines 798-810

**English**: Creates a fresh, unconstrained hybrid zonotope from an interval `Bounds` object. The resulting HZ is an axis-aligned box (independent intervals per dimension). Used when operations lose inter-dimensional correlation (e.g., after non-HZ operations).

**Construction**:
- Center: `c = (lb + ub) / 2`
- Continuous generators: diagonal matrix `Gc = diag((ub - lb) / 2)` (one generator per dimension)
- Binary generators: empty (`Gb = zeros(n, 0)`)
- Constraints: empty (`Ac = zeros(0, n)`, `Ab = zeros(0, 0)`, `b = zeros(0, 1)`)

**Called by / 调用者** (avoids duplicating the "bounds → unconstrained HZ" conversion / 避免重复 "bounds → 无约束 HZ" 转换):
- `hybridz_tf.py:155` — `_hz_transform("ABS")`: create fresh HZ from abs output bounds
- `hybridz_tf.py:188` — `_hz_transform("MUL")`: create fresh HZ from McCormick corners
- `hybridz_tf.py:238` — `apply()`: create fresh HZ for layers without HZ transform (LSTM, etc.)

**Chinese / 中文**: 从区间`Bounds`对象创建新的、无约束的混合多面体。结果HZ是轴对齐盒(每个维度独立区间)。当操作失去维度间相关性时使用(例如，非HZ操作之后)。

**构造**：
- 中心：`c = (lb + ub) / 2`
- 连续生成器：对角矩阵`Gc = diag((ub - lb) / 2)`（每维一个生成器）
- 二值生成器：空
- 约束：空

---

## 11. Complexity Reduction (Lines 815-913)

**Change Type / 变更类型**: New feature

### `_hz_reduce(hz, max_order)` -- Lines 815-913

**English**: Reduces hybrid zonotope complexity by applying sound over-approximation techniques. This is critical for preventing exponential growth of generators and constraints through deep networks. Based on PhD thesis Chapter 6 (Propositions 6.2.3 and 6.2.4).

**Two-step reduction process**:

**Step 1: Binary generator relaxation** (lines 838-858, Prop 6.2.4)
- Target: at most `2*n` binary generators.
- Method: Move smallest L1-norm binary generator columns from `Gb` to `Gc`, relaxing `xi_b in {-1,1}` to `xi_b in [-1,1]`.
- Corresponding constraint columns move from `Ab` to `Ac`.
- This is sound because `{-1,1} subset [-1,1]`.

**Step 2: Continuous generator reduction** (lines 862-911, Prop 6.2.3 / Girard's method)
- Target: at most `max_order * n` continuous generators.
- Method: Girard's heuristic -- keep the `max_ng - n` largest-norm columns, replace the rest with an axis-aligned bounding box.
  - Dropped columns' contribution is over-approximated by `box_rad = sum(|dropped_cols|, axis=1)`.
  - New box generators `Gc_box = diag(box_rad)` are added.
- Constraint handling: Any constraint row referencing a dropped generator column is removed entirely (conservative but sound).

**Chinese / 中文**: 通过应用可靠的过近似技术来降低混合多面体的复杂度。这对于防止生成器和约束在深层网络中的指数增长至关重要。基于博士论文第6章(命题6.2.3和6.2.4)。

**两步约简过程**：

**步骤1：二值生成器松弛**（第838-858行，命题6.2.4）
- 目标：最多`2*n`个二值生成器。
- 方法：将最小L1范数的二值生成器列从`Gb`移到`Gc`，将`xi_b in {-1,1}`松弛为`xi_b in [-1,1]`。
- 对应约束列从`Ab`移到`Ac`。

**步骤2：连续生成器约简**（第862-911行，命题6.2.3 / Girard方法）
- 目标：最多`max_order * n`个连续生成器。
- 方法：Girard启发式——保留`max_ng - n`个最大范数列，用轴对齐包围盒替换其余部分。
- 约束处理：引用已丢弃生成器列的任何约束行被完全删除(保守但可靠)。

**Parameters / 参数**:
- `max_order` (default 10.0): Maximum ratio `ng/n`. Higher values preserve more precision at the cost of memory/computation.
- `max_nb = max(2*n, 1)`: Binary generator budget.

**Called by / 调用者** (centralizes complexity control to avoid repeating reduction logic after each activation / 集中复杂度控制，避免在每个激活后重复约简逻辑):
- `hybridz_tf.py:145` — `_hz_transform("RELU")`: `return _hz_reduce(_hz_apply_relu(hz_in))`
- `hybridz_tf.py:147` — `_hz_transform("LRELU")`: `return _hz_reduce(_hz_apply_leaky_relu(hz_in, alpha))`

**Edge case**: Empty zonotope (`n == 0`) is returned as-is (line 829-830).

---

## 12. Modified Transfer Functions (Lines 916-1131)

**Change Type / 变更类型**: Refactor + Cleanup + Formatting

The transfer functions from main are preserved with minor modifications. These functions continue to serve as the interval-arithmetic path. Below are all individual changes.

### 12.1 Section Header (Lines 916-918)

**Change type**: Formatting

**Before**: No section header; functions started immediately after imports.

**After**:
```python
# ============================================================================
# Transfer functions -- (L, Bounds) -> Fact
# ============================================================================
```

**EN**: Added visual section separator to distinguish HZ helper functions from transfer functions.
**CN**: 添加视觉分隔符以区分HZ辅助函数和传递函数。

---

### 12.2 `hybridz_tf_dense` -- Lines 920-954

**Change type**: No change (identical to main)

**EN**: Dense layer transfer function is completely unchanged.
**CN**: 密集层传递函数完全未变。

---

### 12.3 `hybridz_tf_bias` -- Lines 957-970

**Change type**: No change (identical to main)

---

### 12.4 `hybridz_tf_scale` -- Lines 973-989

**Change type**: No change (identical to main)

---

### 12.5 `hybridz_tf_relu` -- Lines 992-1015

**Change type**: Refactor + Cleanup

**Changes**:

1. **Removed upfront phase determination** (moved after bounds):

   Before (main, lines 97-101):
   ```python
   # Determine ReLU phases
   idx_on = torch.where(Bin.lb >= 0)[0]  # Always active
   idx_off = torch.where(Bin.ub <= 0)[0]  # Always inactive
   idx_amb = torch.where((Bin.lb < 0) & (Bin.ub > 0))[0]  # Ambiguous

   # Compute output bounds
   lb = torch.clamp(Bin.lb, min=0)
   ub = torch.clamp(Bin.ub, min=0)
   ```

   After (current, lines 995-1008):
   ```python
   lb = torch.clamp(Bin.lb, min=0)
   ub = torch.clamp(Bin.ub, min=0)
   Bout = Bounds(lb=lb, ub=ub)

   # Constraint generation (always from interval Bin)
   cons = ConSet()

   # For ambiguous neurons, use HybridZ slope computation
   slope = torch.zeros_like(Bin.lb)
   shift = torch.zeros_like(Bin.lb)

   idx_amb = torch.where((Bin.lb < 0) & (Bin.ub > 0))[0]
   idx_on = torch.where(Bin.lb >= 0)[0]
   idx_off = torch.where(Bin.ub <= 0)[0]
   ```

   **EN**: Phase determination (`idx_on`, `idx_off`, `idx_amb`) moved to after bounds computation and constraint initialization. The order of the three `torch.where` calls is also changed (`idx_amb` now comes first). This is a code organization change with no functional impact.
   **CN**: 相位确定(`idx_on`、`idx_off`、`idx_amb`)移到边界计算和约束初始化之后。三个`torch.where`调用的顺序也改变了(`idx_amb`现在排在第一位)。这是代码组织变更，无功能影响。

2. **Comment updated** (line 999):

   Before: `# HybridZ-specific ReLU constraint generation`
   After: `# Constraint generation (always from interval Bin)`

   **EN**: Clarifies that this transfer function always works from interval bounds, not from HZ objects.
   **CN**: 澄清此传递函数始终从区间边界工作，而非HZ对象。

3. **Removed dead code** -- unused `s` and `t` variables in else branch:

   Before (main, lines 116-118):
   ```python
   else:
       s = torch.empty(0, dtype=Bin.lb.dtype, device=Bin.lb.device)
       t = torch.empty(0, dtype=Bin.lb.dtype, device=Bin.lb.device)
   ```

   After: `else` branch removed entirely.

   **EN**: Removed dead code. Variables `s` and `t` were created but never used.
   **CN**: 删除死代码。变量`s`和`t`被创建但从未使用。

---

### 12.6 `hybridz_tf_lrelu` -- Lines 1018-1047

**Change type**: Refactor + Formatting

**Changes**:

1. **Moved phase determination after bounds computation**:

   Before (main, lines 131-136):
   ```python
   alpha = float(L.params.get("negative_slope", 0.01))

   # Determine phases
   idx_on = torch.where(Bin.lb >= 0)[0]
   idx_off = torch.where(Bin.ub <= 0)[0]
   idx_amb = torch.where((Bin.lb < 0) & (Bin.ub > 0))[0]

   # Output bounds
   ```

   After (current, lines 1021-1031):
   ```python
   alpha = float(L.params.get("negative_slope", 0.01))

   # Output bounds
   lb = torch.where(Bin.lb >= 0, Bin.lb, alpha * Bin.lb)
   ub = torch.where(Bin.ub <= 0, alpha * Bin.ub, Bin.ub)
   Bout = Bounds(lb=lb, ub=ub)

   # Constraint generation
   idx_on = torch.where(Bin.lb >= 0)[0]
   idx_off = torch.where(Bin.ub <= 0)[0]
   idx_amb = torch.where((Bin.lb < 0) & (Bin.ub > 0))[0]
   ```

   **EN**: Reordered to compute bounds before phase determination. Added `# Constraint generation` comment. Consistent with `hybridz_tf_relu` refactoring.
   **CN**: 重新排序，在相位确定之前计算边界。添加了`# Constraint generation`注释。与`hybridz_tf_relu`重构一致。

2. **`cons.add_op` formatting**:

   Before (main, lines 155-156):
   ```python
   cons.add_op(f"lrelu:{L.id}", list(L.out_vars + L.in_vars), alpha=alpha, idx_on=idx_on, idx_off=idx_off, idx_amb=idx_amb,
        slope=slope[idx_amb], shift=shift[idx_amb])
   ```

   After (current, line 1045):
   ```python
   cons.add_op(f"lrelu:{L.id}", list(L.out_vars + L.in_vars), alpha=alpha, idx_on=idx_on, idx_off=idx_off, idx_amb=idx_amb, slope=slope[idx_amb], shift=shift[idx_amb])
   ```

   **EN**: Formatting change only -- line continuation removed, now a single long line.
   **CN**: 仅格式变更——删除续行，现为单行。

---

### 12.7 `hybridz_tf_tanh` -- Lines 1050-1060

**Change type**: Refactor + Formatting

**Changes**:

1. **Docstring updated**:

   Before: (no docstring)
   After: `"""Tanh with piecewise linear HZ encoding. Returns Fact."""`

   **EN**: Added docstring mentioning piecewise linear HZ encoding, even though this transfer function itself only does interval arithmetic. The docstring references the parallel HZ path.
   **CN**: 添加了提及分段线性HZ编码的文档字符串，即使此传递函数本身只做区间算术。文档字符串引用了并行HZ路径。

2. **Inlined intermediate variables**:

   Before (main, lines 164-170):
   ```python
   lb = torch.tanh(Bin.lb)
   ub = torch.tanh(Bin.ub)

   lb2 = torch.minimum(lb, ub)
   ub2 = torch.maximum(lb, ub)

   Bout = Bounds(lb=lb2, ub=ub2)
   ```

   After (current, lines 1053-1055):
   ```python
   lb = torch.tanh(Bin.lb)
   ub = torch.tanh(Bin.ub)
   Bout = Bounds(lb=torch.minimum(lb, ub), ub=torch.maximum(lb, ub))
   ```

   **EN**: Eliminated intermediate variables `lb2`/`ub2` by inlining `torch.minimum`/`torch.maximum` into the `Bounds` constructor. Blank lines between statements also removed.
   **CN**: 通过将`torch.minimum`/`torch.maximum`内联到`Bounds`构造函数中消除了中间变量`lb2`/`ub2`。语句之间的空行也被删除。

---

### 12.8 `hybridz_tf_sigmoid` -- Lines 1062-1071

**Change type**: Refactor + Formatting

**Changes**: Same pattern as `hybridz_tf_tanh`:

1. **Docstring added**: `"""Sigmoid with piecewise linear HZ encoding. Returns Fact."""`

2. **Inlined intermediate variables**:

   Before (main, lines 178-183):
   ```python
   lb = torch.sigmoid(Bin.lb)
   ub = torch.sigmoid(Bin.ub)
   lb2 = torch.minimum(lb, ub)
   ub2 = torch.maximum(lb, ub)

   Bout = Bounds(lb=lb2, ub=ub2)

   cons = ConSet()
   ```

   After (current, lines 1065-1069):
   ```python
   lb = torch.sigmoid(Bin.lb)
   ub = torch.sigmoid(Bin.ub)
   Bout = Bounds(lb=torch.minimum(lb, ub), ub=torch.maximum(lb, ub))

   cons = ConSet()
   ```

   **EN**: Same simplification as tanh -- eliminated `lb2`/`ub2` intermediates, removed extra blank lines.
   **CN**: 与tanh相同的简化——消除了`lb2`/`ub2`中间变量，删除了多余空行。

---

### 12.9 `hybridz_tf_abs` -- Lines 1074-1089

**Change type**: Cleanup + Formatting

**Changes**:

1. **Removed inline comments** on `torch.where` calls:

   Before (main, lines 193-196):
   ```python
   idx_pos = torch.where(Bin.lb >= 0)[0]  # Always positive
   idx_neg = torch.where(Bin.ub <= 0)[0]  # Always negative
   idx_amb = torch.where((Bin.lb < 0) & (Bin.ub > 0))[0]  # Crosses zero

   # Output bounds
   ```

   After (current, lines 1077-1080):
   ```python
   idx_pos = torch.where(Bin.lb >= 0)[0]
   idx_neg = torch.where(Bin.ub <= 0)[0]
   idx_amb = torch.where((Bin.lb < 0) & (Bin.ub > 0))[0]
   ```

   **EN**: Removed descriptive comments (`# Always positive`, `# Always negative`, `# Crosses zero`) and the `# Determine phases` / `# Output bounds` section headers. Trailing whitespace changes in the `torch.where` multi-line expression (comma vs no trailing space).
   **CN**: 删除了描述性注释(`# Always positive`等)和`# Determine phases` / `# Output bounds`节标题。`torch.where`多行表达式中的尾部空白变更。

---

### 12.10 `hybridz_tf_add` -- Lines 1092-1103

**Change type**: No change (identical to main)

---

### 12.11 `hybridz_tf_mul` -- Lines 1106-1131

**Change type**: Formatting + Bug fix

**Changes**:

1. **`cons.add_op` uses `Bin1.lb`/`Bin1.ub` instead of `lx`/`ux`**:

   Before (main, line 239):
   ```python
   cons.add_op(f"mcc:{L.id}", list(L.out_vars + L.in_vars), lx=lx, ux=ux, ly=ly, uy=uy)
   ```

   After (current, lines 1128-1129):
   ```python
   cons.add_op(f"mcc:{L.id}", list(L.out_vars + L.in_vars),
               lx=Bin1.lb, ux=Bin1.ub, ly=Bin2.lb, uy=Bin2.ub)
   ```

   **EN**: References `Bin1.lb`/`Bin1.ub`/`Bin2.lb`/`Bin2.ub` directly instead of the aliased local variables `lx`/`ux`/`ly`/`uy`. Functionally equivalent since `lx = Bin1.lb` etc., but makes the data flow clearer and avoids potential aliasing issues if the local variables were mutated (they are not in this case). Also reformatted to two lines.
   **CN**: 直接引用`Bin1.lb`/`Bin1.ub`/`Bin2.lb`/`Bin2.ub`而非别名局部变量`lx`/`ux`/`ly`/`uy`。功能等价，但使数据流更清晰。也重新格式化为两行。

2. **Added newline at end of file**:

   Before: File ended without newline (`\ No newline at end of file`).
   After: File ends with proper newline.

   **EN**: Fixed missing trailing newline (POSIX compliance).
   **CN**: 修复了缺失的尾部换行(POSIX合规)。

---

## 13. Summary Table

| # | Function/Section | Lines (current) | Type | New Gens/Constraints per neuron | Description (EN) | Description (CN) |
|---|------------------|-----------------|------|----------------------------------|-------------------|-------------------|
| 1 | Import section | 16-48 | New feature + Refactor | -- | Lazy HZono import, optional Gurobi/SciPy | 懒加载HZono导入，可选Gurobi/SciPy |
| 2 | `_hz_multiply` | 55-58 | New feature | -- | Linear map on HZ (dense layer) | HZ上的线性映射(密集层) |
| 3 | `_hz_add_const` | 60-65 | New feature | -- | Constant translation on HZ (bias) | HZ上的常数平移(偏置) |
| 4 | `_hz_is_unconstrained` | 71-78 | New feature | -- | Check if HZ has no active constraints | 检查HZ是否无活跃约束 |
| 5 | `_hz_bounds_unconstrained` | 81-90 | New feature | -- | Fast closed-form bounds (zonotope) | 快速闭式边界(仿射集) |
| 6 | `_hz_compute_bounds` | 93-109 | New feature | -- | Cascading bounds dispatch | 级联边界调度 |
| 7 | `_hz_compute_bounds_gurobi` | 112-165 | New feature | -- | Exact MILP bounds via Gurobi | 通过Gurobi的精确MILP边界 |
| 8 | `_hz_compute_bounds_scipy` | 168-204 | New feature | -- | LP relaxation bounds via SciPy | 通过SciPy的LP松弛边界 |
| 9 | `_hz_apply_relu` | 209-314 | New feature | +4 Gc, +1 Gb, +3 eq | Exact ReLU encoding (equalities only) | 精确ReLU编码(仅等式) |
| 10 | `_hz_apply_leaky_relu` | 317-454 | New feature | +6 Gc, +1 Gb, +5 eq | Exact LeakyReLU with box equalities | 带盒等式的精确LeakyReLU |
| 11 | `_hz_apply_piecewise` | 457-748 | New feature | +6K Gc, +K Gb, +(4K+2) eq | Tangent parallelogram for sigmoid/tanh | sigmoid/tanh的切线平行四边形 |
| 12 | `_hz_apply_sigmoid` | 751-754 | New feature | (delegates) | Piecewise sigmoid wrapper | 分段sigmoid包装器 |
| 13 | `_hz_apply_tanh` | 757-760 | New feature | (delegates) | Piecewise tanh wrapper | 分段tanh包装器 |
| 14 | `_hz_minkowski_sum` | 765-795 | New feature | -- | Minkowski sum for ADD layers | ADD层的闵可夫斯基和 |
| 15 | `_hz_from_bounds_fresh` | 798-810 | New feature | -- | Create unconstrained HZ from Bounds | 从Bounds创建无约束HZ |
| 16 | `_hz_reduce` | 815-913 | New feature | -- | Complexity reduction (Girard + binary relax) | 复杂度约简(Girard + 二值松弛) |
| 17 | `hybridz_tf_dense` | 920-954 | No change | -- | Dense layer (interval) | 密集层(区间) |
| 18 | `hybridz_tf_bias` | 957-970 | No change | -- | Bias addition (interval) | 偏置加法(区间) |
| 19 | `hybridz_tf_scale` | 973-989 | No change | -- | Scaling (interval) | 缩放(区间) |
| 20 | `hybridz_tf_relu` | 992-1015 | Refactor + Cleanup | -- | Reordered phases, removed dead code | 重排相位，删除死代码 |
| 21 | `hybridz_tf_lrelu` | 1018-1047 | Refactor + Formatting | -- | Reordered phases, formatting | 重排相位，格式调整 |
| 22 | `hybridz_tf_tanh` | 1050-1060 | Refactor + Formatting | -- | Added docstring, inlined vars | 添加文档字符串，内联变量 |
| 23 | `hybridz_tf_sigmoid` | 1062-1071 | Refactor + Formatting | -- | Added docstring, inlined vars | 添加文档字符串，内联变量 |
| 24 | `hybridz_tf_abs` | 1074-1089 | Cleanup | -- | Removed inline comments | 删除行内注释 |
| 25 | `hybridz_tf_add` | 1092-1103 | No change | -- | Addition (interval) | 加法(区间) |
| 26 | `hybridz_tf_mul` | 1106-1131 | Formatting | -- | Direct Bin refs, trailing newline | 直接Bin引用，尾部换行 |

### Statistics / 统计

| Metric | Value |
|--------|-------|
| Lines added (new HZ library) | +900 |
| Lines modified (transfer functions) | ~30 |
| Lines removed | ~13 |
| New functions | 16 |
| Modified functions | 6 |
| Unchanged functions | 4 |
| New feature changes | 16 |
| Refactor changes | 5 |
| Cleanup changes | 2 |
| Formatting changes | 4 |
| Bug fixes | 1 (trailing newline) |

### Key Design Decisions Summary / 关键设计决策总结

1. **All HZ constraints are equalities** (`Ac*xi_c + Ab*xi_b = b`), following paper Definition 3.1. No inequality constraints are ever generated. This is a fundamental design choice that simplifies the constraint system at the cost of more generators.
   所有HZ约束都是等式，遵循论文定义3.1。从不生成不等式约束。

2. **ReLU uses 4 Gc + 1 Gb + 3 equalities** per unstable neuron. The exact encoding avoids over-approximation at the cost of increased dimensionality.
   ReLU每个不稳定神经元使用4个Gc + 1个Gb + 3个等式。精确编码以增加维度为代价避免过近似。

3. **LeakyReLU uses box equalities with slack generators** (6 Gc + 1 Gb + 5 eq). The box equality technique from Definition 3.4 implements conditional generator activation purely through equalities.
   LeakyReLU使用带松弛生成器的盒等式(6 Gc + 1 Gb + 5 eq)。定义3.4的盒等式技术纯粹通过等式实现条件生成器激活。

4. **Piecewise sigmoid/tanh uses tangent parallelogram** (Theorem 3.8) with configurable K pieces. Tighter than secant + error-rectangle.
   分段sigmoid/tanh使用切线平行四边形(定理3.8)，可配置K段。比割线+误差矩形更紧。

5. **Gurobi MILP for exact bounds, SciPy LP as fallback**. Graceful degradation via try/except cascade.
   Gurobi MILP用于精确边界，SciPy LP作为后备。通过try/except级联优雅降级。

6. **Transfer functions keep original signatures**. HZ processing is delegated to `hybridz_tf.py`'s `apply()`. The transfer functions in this file serve as the interval-only fallback path.
   传递函数保持原始签名。HZ处理委托给`hybridz_tf.py`的`apply()`。本文件中的传递函数作为仅区间后备路径。
