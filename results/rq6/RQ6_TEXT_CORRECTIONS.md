# RQ6 文本修正对照表

## 🔴 原文的问题

### 问题1: 使用旧实验数据
**原文**: "overhead ranges from 65\% (Small) to 25\% (XLarge)"
**问题**:
- 新表格报告**绝对时间**(ms)，不是overhead ratio (%)
- 新实验没有XLarge模型
- 这些数字来自完全不同的实验

---

### 问题2: CBR时间完全错误
**原文**: "CBR time is roughly constant at $\sim$4.0\,ms across sizes"
**实际数据** (budget=20):
- Small: 1.34 ms
- Medium: 1.16 ms
- Large: 1.44 ms

**问题**:
- ❌ 不是4.0ms（差了3倍）
- ❌ 不是"constant"（CBR随budget变化）

---

### 问题3: BBL时间范围错误
**原文**: "BBL time grows with size (0.02\,ms $\rightarrow$ 0.66\,ms)"
**实际数据**:
- Small: 0.09 ms
- Medium: 0.13 ms
- Large: 0.13 ms

**问题**:
- ❌ 范围错误（实际是0.09-0.13ms，不是0.02-0.66ms）
- ❌ 增长描述错误（Medium→Large没有增长）

---

### 问题4: 参数量错误
**原文**: "Medium ($\sim$65K params)"
**实际数据**: Medium = ~34K params

**问题**: ❌ 参数量差了一倍

---

### 问题5: **最关键** - 没有讨论多个budgets
**原文**: 只提到"default configuration (20 samples)"
**导师质疑**: "You claim to test budgets (5,10,20,50) but only report budget=20"

**问题**:
- ❌ 没有明确说明测试了4个budgets
- ❌ 没有讨论CBR随budget的scaling行为
- ❌ 没有说明BBL在所有budgets下都是恒定的

这正是导师质疑的核心问题！

---

### 问题6: 只有最后一段是对的
**原文**: "Medium has more parameters than Small, yet CBR time is comparable or lower (e.g., at budget=20: 1.16\,ms vs.\ 1.34\,ms)"

**实际数据**: ✅ 这段是正确的
- Medium: 34K params, CBR=1.16ms
- Small: 2K params, CBR=1.34ms

---

## ✅ 修正后的版本

### 改进1: 明确说明测试了所有budgets
```latex
We measure the overhead ... across \emph{CBR sampling budgets} (5, 10, 20, 50)
...
Table~\ref{tab:rq6-overhead} reports results across all tested configurations.
```
→ **直接回应导师质疑**

---

### 改进2: 讨论CBR线性scaling
```latex
CBR time scales approximately linearly with sampling budget:
for the Small model, overhead increases from 0.50\,ms (budget=5)
to 2.81\,ms (budget=50), a 5.6× increase for a 10× budget increase.
Medium and Large models exhibit near-perfect linear scaling,
with 9.8× and 10.0× increases respectively.
```
→ 用实际数据证明线性关系

---

### 改进3: 强调BBL独立于budget
```latex
BBL time is independent of sampling budget and grows modestly
with model size (0.09\,ms for Small, 0.13\,ms for Medium/Large),
remaining constant across all budgets for each model size.
```
→ 这是表格设计要展示的关键点

---

### 改进4: 报告所有budgets的数据
```latex
At the default configuration (budget=20), combined overhead ranges
from 1.28\,ms (Medium) to 1.57\,ms (Large).
...
At higher budgets (e.g., budget=50), CBR dominates even more strongly
(96--97\% of total cost), while at lower budgets (e.g., budget=5),
BBL becomes relatively more significant (15--31\% of cost).
```
→ 讨论了不同budgets下的行为

---

### 改进5: 更新所有数值
| 指标 | 原文（错误） | 修正后 |
|------|-------------|--------|
| Model sizes | Small/Medium/Large/XLarge | Small/Medium/Large |
| Params | ~65K (Medium) | ~34K (Medium) |
| CBR time | ~4.0ms | 1.16-1.44ms (budget=20) |
| BBL range | 0.02-0.66ms | 0.09-0.13ms |
| Overhead metric | Ratio (%) | Absolute time (ms) |

---

### 改进6: Finding明确总结
```latex
\finding{Our unsoundness detection framework exhibits predictable and low
overhead: CBR cost scales linearly with sampling budget (0.29--3.59\,ms across
all configurations), while BBL adds a small constant cost (0.09--0.13\,ms)
independent of budget. At the default budget=20, total overhead is 1.28--1.57\,ms
across all model sizes.}
```

---

## 📊 关键数据对照

### CBR Overhead (ms)
| Model | Budget=5 | Budget=10 | Budget=20 | Budget=50 |
|-------|----------|-----------|-----------|-----------|
| Small | 0.50 | 0.85 | **1.34** | 2.81 |
| Medium | 0.29 | 0.58 | **1.16** | 2.84 |
| Large | 0.36 | 0.73 | **1.44** | 3.59 |

### BBL Overhead (ms) - Constant across budgets
| Model | All Budgets (5/10/20/50) |
|-------|--------------------------|
| Small | 0.09 |
| Medium | 0.13 |
| Large | 0.13 |

### Combined Overhead (ms)
| Model | Budget=5 | Budget=10 | Budget=20 | Budget=50 |
|-------|----------|-----------|-----------|-----------|
| Small | 0.59 | 0.93 | **1.42** | 2.90 |
| Medium | 0.42 | 0.70 | **1.28** | 2.97 |
| Large | 0.49 | 0.86 | **1.57** | 3.72 |

---

## 🎯 修正清单

- [x] 删除XLarge引用
- [x] 修正Medium参数量（65K → 34K）
- [x] 更新CBR时间（4.0ms → 1.16-1.44ms）
- [x] 更新BBL范围（0.02-0.66ms → 0.09-0.13ms）
- [x] 明确说明测试了budgets (5,10,20,50)
- [x] 讨论CBR线性scaling (5.6-10×)
- [x] 强调BBL独立于budget
- [x] 报告不同budgets下的overhead
- [x] 改用绝对时间(ms)而非ratio(%)
- [x] 保留正确的Medium vs Small观察

---

## 📝 使用建议

1. **直接替换**: 用 `rq6_section_corrected.tex` 替换论文中的RQ6 subsection

2. **验证数字**: 确保所有数字与 `table_rq6_full.tex` 一致

3. **检查引用**: 确认 `\ref{tab:rq6-overhead}` 指向新表格

4. **对导师的回复**:
   > We have updated RQ6 to report results across all four sampling budgets
   > (5, 10, 20, 50). The revised Table X and text demonstrate that:
   > (1) CBR overhead scales linearly with budget (5.6-10× for 10× budget increase)
   > (2) BBL overhead remains constant (0.09-0.13 ms) across all budgets
   > (3) Combined overhead ranges from 0.42 ms to 3.72 ms across all configurations
   >
   > The updated section now fully reflects our experimental design and results.

---

**总结**: 原文描述的是**完全不同的实验**，必须全部重写以匹配新数据！
