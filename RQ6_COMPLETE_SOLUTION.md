# RQ6 完整解决方案 - 针对导师问题3B

## ✅ 问题已解决

**导师原始问题**：
> "你说测试了budgets (5, 10, 20, 50)，但Table只报告了budget=20的结果"

**解决方案**：
1. ✅ 已运行完整实验（所有4个budgets）
2. ✅ 已生成包含所有结果的新表格
3. ✅ 已更新论文文本描述所有configurations
4. ✅ 已生成可视化图表

---

## 📂 生成的文件

所有文件位于：`/Users/z5524562/Desktop/Ai2ware/ACT/results/rq6/`

### 1. 实验结果
- `results.json` - 完整实验数据（所有budgets × 所有model sizes）

### 2. LaTeX材料
- `table_rq6_full.tex` - **新表格**（包含所有4个budgets的结果）
- `rq6_updated_text.tex` - **更新后的RQ6章节文本**

### 3. 图表
- `fig_rq6_overhead.pdf` - 开销随budget线性增长的可视化图
- `fig_rq6_overhead.png` - 预览版本
- `fig_rq6_overhead.csv` - 数据文件

### 4. 说明文档
- `RQ6_UPDATE_INSTRUCTIONS.md` - 详细修改指南

---

## 📊 实验结果摘要

### CBR开销（随budget线性增长）：

| Model Size | Budget=5 | Budget=10 | Budget=20 | Budget=50 |
|-----------|----------|-----------|-----------|-----------|
| **Small**   | 0.50 ms  | 0.85 ms   | **1.34 ms**  | 2.81 ms   |
| **Medium**  | 0.29 ms  | 0.58 ms   | **1.16 ms**  | 2.84 ms   |
| **Large**   | 0.36 ms  | 0.73 ms   | **1.44 ms**  | 3.59 ms   |

### BBL开销（恒定，不随budget变化）：
- Small: 0.09 ms
- Medium: 0.13 ms
- Large: 0.13 ms

### 组合开销（CBR + BBL）：
- **最小**: 0.42 ms (Medium, budget=5)
- **最大**: 3.72 ms (Large, budget=50)
- **默认配置** (budget=20): 1.28--1.57 ms

### 关键发现：
1. ✅ CBR开销与budget**线性相关**（5→50增长约5-10倍）
2. ✅ BBL开销与budget**无关**（恒定值）
3. ✅ 即使最大budget(50)，总开销也< 4ms（非常高效）

---

## 🔧 论文修改步骤

### 步骤1：复制新表格

```bash
cp results/rq6/table_rq6_full.tex Main/Tables/tab:rq6-overhead-full.tex
```

### 步骤2：替换论文中的RQ6章节

**位置**：找到 `\subsection{RQ6: Overhead}`

**替换整个小节内容为**：`results/rq6/rq6_updated_text.tex` 中的内容

**关键变化**：
- ✅ 现在明确说明"tested across budgets (5, 10, 20, 50)"
- ✅ 讨论所有budgets的结果，不只是20
- ✅ 说明CBR线性scaling，BBL恒定
- ✅ 报告绝对时间（ms），不是overhead ratio

### 步骤3：更新表格引用

**原来**：`\ref{tab:rq6-overhead}`
**改为**：`\ref{tab:rq6-overhead-full}`

### 步骤4：（可选）添加图表

如果想要更直观，可以在RQ6章节添加Figure：

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=0.95\textwidth]{Main/Figs/fig_rq6_overhead.pdf}
  \caption{Validation overhead scaling with CBR sampling budget.
  (a) CBR overhead grows linearly with budget.
  (b) Total overhead (CBR + BBL) across configurations.}
  \label{fig:rq6-overhead}
\end{figure*}
```

然后复制图片：
```bash
cp results/rq6/fig_rq6_overhead.pdf Main/Figs/
```

### 步骤5：更新Evaluation metrics定义

**找到"Evaluation metrics"段落，修改第(4)项：**

**原来**：
```latex
(4) \emph{Overhead ratio:} $T_{\text{Detection}} / T_{\text{verification}}$,
where $T_{\text{Detection}}$ includes CBR and BBL checks...
```

**改为**：
```latex
(4) \emph{Validation overhead:} absolute time (in milliseconds) for CBR and BBL checks,
measured across sampling budgets (5, 10, 20, 50) and model sizes.
```

---

## ✅ 修改后验证清单

请确认以下所有项都正确：

- [ ] 新表格显示**所有4个budgets** (5, 10, 20, 50)
- [ ] RQ6文本讨论**所有budgets的结果**
- [ ] 删除了"only tested budget=20"的说法
- [ ] Evaluation metrics中overhead定义已更新
- [ ] 表格caption清晰准确
- [ ] 所有数字与实验结果一致
- [ ] （可选）添加了可视化图表

---

## 📈 新表格预览

```latex
\begin{table*}[t]
\caption{RQ6: Validation overhead across sampling budgets...}
\begin{tabular}{lrrrrr}
\toprule
Model Size & \multicolumn{4}{c}{CBR Sampling Budget} & BBL \\
\cmidrule{2-5}
& 5 & 10 & 20 & 50 & (constant) \\
\midrule
Small   & 0.50 & 0.85 & 1.34 & 2.81 & 0.09 \\
Medium  & 0.29 & 0.58 & 1.16 & 2.84 & 0.13 \\
Large   & 0.36 & 0.73 & 1.44 & 3.59 & 0.13 \\
\midrule
\multicolumn{6}{l}{\textit{Combined overhead (CBR + BBL):}} \\
Small   & 0.59 & 0.93 & 1.42 & 2.90 & --- \\
Medium  & 0.42 & 0.70 & 1.28 & 2.97 & --- \\
Large   & 0.49 & 0.86 & 1.57 & 3.72 & --- \\
\bottomrule
\end{tabular}
\end{table*}
```

---

## 🎯 对导师的回复

**原始质疑**：
> "You claim to measure across budgets (5,10,20,50) but only report budget=20"

**现在可以回答**：
> ✅ 我们已经补充了完整的实验，测试了所有4个sampling budgets (5, 10, 20, 50)。
>
> 更新后的Table X展示了所有配置的结果，证明：
> 1. CBR开销与budget线性相关（0.29-3.59 ms范围）
> 2. BBL开销恒定（0.09-0.13 ms，不随budget变化）
> 3. 在默认配置(budget=20)下，总开销为1.3-1.6 ms
>
> 实验数据完整，声明与结果完全一致。

---

## 📝 新RQ6章节文本预览

```latex
\subsection{RQ6: Overhead}

We measure the overhead of our unsoundness detection framework across
\emph{CBR sampling budgets} (5, 10, 20, 50)—i.e., the number of concrete
inputs sampled by CBR per network—and model sizes (Small, Medium, Large).
CBR overhead is expected to scale linearly with budget, while BBL overhead
depends on model size but remains constant per budget.

Table~\ref{tab:rq6-overhead-full} reports overhead across all configurations.
CBR time scales approximately linearly with sampling budget: at Small model
size, overhead increases from 0.50 ms (budget=5) to 2.81 ms (budget=50),
confirming that sampling cost dominates CBR.
BBL time is independent of sampling budget and grows modestly with model
size (0.09 ms for Small, 0.13 ms for Medium/Large).

At the default configuration (budget=20), combined overhead (CBR+BBL) ranges
from 1.28 ms (Medium) to 1.57 ms (Large). The cost breakdown shows CBR
accounts for 85--95% of total overhead, with BBL contributing a small
constant cost.

\finding{Our unsoundness detection framework exhibits predictable overhead
scaling: CBR cost grows linearly with sampling budget (0.5--3.6 ms for
budgets 5--50), while BBL adds a small constant cost (0.09--0.13 ms)
independent of budget.}
```

---

## ✨ 总结

**问题状态**: ✅ **完全解决**

**已完成**:
1. ✅ 运行完整实验（4 budgets × 3 sizes = 12 configurations）
2. ✅ 生成包含所有结果的新表格
3. ✅ 编写更新后的章节文本
4. ✅ 创建可视化图表
5. ✅ 准备详细的修改指南

**需要你做的**:
1. 复制新表格到 `Main/Tables/`
2. 用新文本替换RQ6章节
3. 更新Evaluation metrics定义
4. （可选）添加可视化图表
5. 编译LaTeX验证

**预计修改时间**: 10-15分钟

---

**生成时间**: 2026-02-13
**状态**: Ready for integration
**验证**: All numbers verified against experiment results
