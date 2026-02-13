# 导师反馈 - 剩余问题检查

## ✅ 已解决的问题

### 1. RQ6 Overhead Inconsistency ✅ FIXED
**原始问题**: "You say budgets (5,10,20,50) but only report 20"

**当前状态**: ✅ **完全修复**
- Text明确说测试了4个budgets ✓
- 报告了所有budgets的数据 (0.50→2.81ms等) ✓
- 表格应该是更新后的 `table_rq6_full.tex` (14列版本) ✓

**验证**:
```latex
Table~\ref{tab:rq6-overhead} shows that CBR scales near-linearly with
the sampling budget: increasing the budget from 5 to 50 (10×) increases
CBR time by 5.6--10.0× (Small:0.50→2.81ms; Medium: 0.29→2.84ms;
Large:0.36→3.59ms).
```

---

### 2. Statistical Claims ✅ FIXED
**原始问题**: "Claims CI and Wilcoxon tests but shows nothing"

**当前状态**: ✅ **已移除声明**
- Evaluation metrics不再提"95% CI" ✓
- 不再提"Wilcoxon tests" ✓
- 只报告简单指标（率、准确度） ✓

---

### 3. CDR Typo ✅ NOT FOUND
**原始问题**: "Table 3: CDR Fail (should be CBR)"

**当前状态**: ✅ **不存在**
- `grep -r "CDR" Main/Tables/` → 未找到 ✓
- 全项目搜索只在notebook中找到，不在论文里 ✓
- 可能已经修复过了

---

## 🟡 需要修复的问题

### 1. Grammar Error in RQ6 (Minor)

**位置**: `\subsection{RQ6: Overhead}` 第二段

**错误**:
```latex
❌ Consequently, complementary unsoundness checks remains low in absolute time
```

**修正**:
```latex
✅ Consequently, complementary unsoundness checks remain low in absolute time
```

**原因**: "checks"是复数，动词应该用 `remain` 不是 `remains`

---

## ✅ 已正确处理的内容

### Case 2 措辞 ✅ GOOD
**之前**: "We inject..." (误导性，像实验)
**现在**: "During development... we inadvertently implemented..." (正确，真实bug)

**状态**: ✅ **正确描述为开发bug**

---

## 📝 修改建议

### 必须修改（1处）

1. **RQ6 grammar fix**:
   ```diff
   - complementary unsoundness checks remains low
   + complementary unsoundness checks remain low
   ```

### 建议检查（预防性）

1. **确认Table引用正确**:
   - `\input{Main/Tables/tab:rq6-overhead}` 应该指向更新后的14列表格
   - 表格label应该是 `\label{tab:rq6-overhead-full}` 或匹配的label

2. **确认Case 2位置**:
   - 如果Case 2在论文的"Case Studies"小节
   - 确保开头有说明："Case 1 shows systematic mutation, Case 2 shows real development bug"

---

## 🎯 修改优先级

| 问题 | 严重性 | 状态 | 操作 |
|------|--------|------|------|
| RQ6 budgets报告 | High | ✅ Fixed | 已解决 |
| Statistical claims | High | ✅ Fixed | 已解决 |
| CDR typo | Medium | ✅ Not found | 无需操作 |
| RQ6 grammar | Low | 🟡 To fix | 改1个词 |
| Case 2 措辞 | Medium | ✅ Fixed | 已解决 |

---

## ✅ 总结

**好消息**: 导师的主要技术问题（RQ6 budgets不一致、统计声明）都已修复！

**需要做**: 只需修复1处语法错误（remains → remain）

**建议**: 最后检查一遍表格引用是否指向正确的文件
