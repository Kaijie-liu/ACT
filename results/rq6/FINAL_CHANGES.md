# RQ6 最终修改（3处小改动）

## 修改对比

### ❌ → ✅ 修改1: 采样主体
```diff
- sampled by complementary unsoundness checks per network
+ sampled by CBR per network
```
**原因**: CBR才是做采样的，不是"complementary unsoundness checks"

---

### ❌ → ✅ 修改2: BBL描述
```diff
- BBL is executed once per network on a single concrete input
+ BBL is executed once per network to audit intermediate layer bounds
+ via hook-based activation capture
```
**原因**:
- BBL不使用"concrete input"（那是CBR用的）
- BBL使用abstract bounds + hooks捕获激活值
- 原描述技术上不准确

---

### ❌ → ✅ 修改3: 语法
```diff
- complementary unsoundness checks remains low
+ complementary unsoundness checks remain low
```
**原因**: "checks"是复数，动词用remain

---

## ✅ 修改后的完整段落

```latex
\subsection{RQ6: Overhead}

We measure the validation overhead across CBR sampling budgets (5, 10, 20, 50),
i.e., the number of concrete inputs sampled by CBR per network, and model sizes
(Small/Medium/Large). We report mean validation time (ms) for CBR, BBL, and
complementary unsoundness checks; BBL is executed once per network to audit
intermediate layer bounds via hook-based activation capture.

Table~\ref{tab:rq6-overhead} shows that CBR scales near-linearly with the
sampling budget: increasing the budget from 5 to 50 (10×) increases CBR time
by 5.6--10.0× (Small:0.50→2.81ms; Medium: 0.29→2.84ms; Large:0.36→3.59ms).
BBL is budget-independent and constant per model size (Small: 0.09ms;
Medium/Large: 0.13ms). Consequently, complementary unsoundness checks remain
low in absolute time: 1.28--1.57ms at budget=20 and 3.72ms at the largest
setting (Large, budget=50), with CBR dominating the combined cost at higher
budgets (96--97% at budget=50).

\input{Main/Tables/tab:rq6-overhead}

Overhead does not monotonically follow parameter count. For example, at
budget=20, Medium is faster than Small for CBR (1.16 vs. 1.34ms), suggesting
that CBR cost is dominated by architecture-dependent execution (e.g., input
shape and operator mix) rather than parameters alone; BBL mainly depends on
the number of audited layer boundaries.
```

---

## 📊 数据验证清单

- [x] CBR scaling: 5.6-10.0× (正确)
- [x] Small: 0.50→2.81ms (正确)
- [x] Medium: 0.29→2.84ms (正确)
- [x] Large: 0.36→3.59ms (正确)
- [x] BBL Small: 0.09ms (正确)
- [x] BBL Medium/Large: 0.13ms (正确)
- [x] Budget=20 combined: 1.28-1.57ms (正确)
- [x] Largest config: 3.72ms (正确)
- [x] CBR占比 at budget=50: 96-97% (正确)
- [x] Medium vs Small at budget=20: 1.16 vs 1.34ms (正确)

---

## ✅ 优点保留

你的简化版本有很多优点，都保留了：
- ✅ 简洁（3段，vs 我的版本4段）
- ✅ 直接说明测试了4个budgets
- ✅ 强调BBL budget-independent
- ✅ 证明CBR线性scaling
- ✅ 讨论非单调behavior
- ✅ 所有数字正确

---

## 🎯 最终文件

使用: `rq6_section_final.tex`

这个版本：
- 修正了3个小问题
- 保留了你简洁的风格
- 所有数据accurate
- 完全回应导师质疑 ✓
