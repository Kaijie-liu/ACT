# RQ6 Table Versions - Usage Guide

## 📊 Available Versions

### 1. `table_rq6_updated.tex` (推荐 - RECOMMENDED)
**类型**: Two-column table (`\begin{table*}`)
**优点**:
- ✅ 完整展示所有数据（CBR + BBL分开，再加Combined）
- ✅ 包含参数量信息（Params列）
- ✅ 有详细注释说明线性scaling关系
- ✅ 最全面，最适合回应导师质疑

**缺点**:
- ❌ 需要两列宽度（`table*`环境）
- ❌ 行数较多（6行数据 + 3行combined）

**使用场景**: 如果期刊/会议允许two-column table，**强烈推荐**使用这个版本

---

### 2. `table_rq6_compact.tex` (紧凑版)
**类型**: Single-column table (`\begin{table}`)
**优点**:
- ✅ 更紧凑，只需单列空间
- ✅ 直接显示Combined overhead（最关心的指标）
- ✅ 仍包含参数量信息
- ✅ 注释说明了BBL恒定特性

**缺点**:
- ❌ 没有显示CBR/BBL的详细分解
- ❌ 读者需要从注释中理解BBL是恒定的

**使用场景**: 空间有限，或者reviewer/editor要求减少table数量时使用

---

### 3. `table_rq6_full.tex` (原始版本)
**类型**: Two-column table
**特点**:
- 最早生成的版本
- **没有参数量列**
- 其他内容与`table_rq6_updated.tex`基本相同

**使用场景**: 如果不需要显示参数量，可以用这个

---

## 🔄 与旧表格的对比

### 旧表格 (`tab:rq6-overhead`)
```latex
| Size   | Params | CBR (ms) | BBL (ms) | Overhead |
|--------|--------|----------|----------|----------|
| Small  | ~4K    | 4.0      | 0.02     | 65%      |
| Medium | ~64K   | 4.0      | 0.09     | 58%      |
| Large  | ~260K  | 3.9      | 0.24     | 45%      |
| XLarge | ~999K  | 4.0      | 0.66     | 25%      |
```

**问题**:
1. ❌ 只显示了一个budget配置
2. ❌ 没有说明测试了多个budgets
3. ❌ 使用"Overhead ratio"而不是绝对时间
4. ❌ 与实际实验数据不符（新实验没有XLarge，数值也完全不同）

### 新表格 (`table_rq6_updated.tex`)
```latex
| Size   | Params | Budget5 | Budget10 | Budget20 | Budget50 | BBL |
|--------|--------|---------|----------|----------|----------|-----|
| Small  | ~2K    | 0.50    | 0.85     | 1.34     | 2.81     | 0.09|
| Medium | ~34K   | 0.29    | 0.58     | 1.16     | 2.84     | 0.13|
| Large  | ~297K  | 0.36    | 0.73     | 1.44     | 3.59     | 0.13|
```

**改进**:
1. ✅ 显示所有4个budgets (5, 10, 20, 50)
2. ✅ 使用绝对时间（milliseconds）
3. ✅ 匹配实际实验数据
4. ✅ 清晰展示CBR线性scaling，BBL恒定

---

## 📝 参数计算说明

新表格中的参数量是这样计算的：

### Small (~2K)
- Architecture: [16 → 32 → 32 → 4]
- Params: (16×32+32) + (32×32+32) + (32×4+4) = **1,732**

### Medium (~34K)
- Architecture: [64 → 128 → 128 → 64 → 10]
- Params: (64×128+128) + (128×128+128) + (128×64+64) + (64×10+10) = **33,738**

### Large (~297K)
- Architecture: [256 → 512 → 256 → 128 → 10]
- Params: (256×512+512) + (512×256+256) + (256×128+128) + (128×10+10) = **297,098**

*(计算公式: weight + bias)*

---

## ✅ 推荐使用流程

1. **首选**: 使用 `table_rq6_updated.tex`
   ```bash
   cp results/rq6/table_rq6_updated.tex Main/Tables/tab:rq6-overhead-full.tex
   ```

2. **如果空间不够**: 使用 `table_rq6_compact.tex`
   ```bash
   cp results/rq6/table_rq6_compact.tex Main/Tables/tab:rq6-overhead-full.tex
   ```

3. **更新文本引用**: 在RQ6章节中使用 `results/rq6/rq6_updated_text.tex` 的内容

4. **验证**: 确保所有数字与 `results/rq6/results.json` 一致

---

## 🎯 对导师的回应

使用新表格后，你可以这样回复导师：

> **Advisor concern**: "You claim to test budgets (5,10,20,50) but only report budget=20"
>
> **Our response**: We have updated Table X to report results across all four sampling budgets. The new table demonstrates:
> 1. CBR overhead scales linearly with budget (0.29-3.59 ms across configurations)
> 2. BBL overhead remains constant (0.09-0.13 ms) regardless of budget
> 3. At the default configuration (budget=20), total overhead is 1.28-1.57 ms
>
> The updated table and text now fully reflect the experimental configurations tested.

---

**生成时间**: 2026-02-13
**数据来源**: `results/rq6/results.json` (master_seed=42, experiment_seed=6042)
