# Case 2 修订指南 - 真实Bug案例

## 🎯 核心问题

**原文问题**: 用"We inject"让人以为是故意注入的实验，但实际是**开发时真实遇到的bug**

---

## 📊 措辞对比

| 原文（误导性） | 修订版（真实描述） | 效果 |
|--------------|------------------|------|
| "We inject a parameter mismatch" | "We inadvertently implemented an off-by-one error" | ✅ 明确是无意的bug |
| "modeling verifier--model semantic drift" | "causing verifier--model semantic drift" | ✅ 不是"模拟"，是真实发生的 |
| （没说背景） | "During development of the Conv2D abstract transformer" | ✅ 说明这是开发阶段 |
| （结尾弱） | "demonstrates BBL's practical value for catching implementation errors during verifier development" | ✅ 强调实用价值 |

---

## ✅ 两个修订版本

### 版本1: `case2_rewrite.tex` (保守修改)
- 保持原结构，只改关键措辞
- "inadvertently implemented" 替换 "inject"
- 添加"During development"说明背景
- 结尾强调"detected a real implementation bug"

**优点**: 改动最小，保持原文风格
**推荐**: 如果想minimize changes

---

### 版本2: `case2_alternative.tex` (重写，更清晰)
- 标题改为"Detecting a development bug"
- 开头就说明"During implementation, we encountered a bug"
- 明确debugging流程: 发现→定位→修复→验证
- 强调"real-world debugging scenario"

**优点**: 故事性更强，学术价值更明确
**推荐**: 如果想强调BBL的实际应用价值

---

## 🔑 关键改动

### 改动1: 背景说明
```diff
- We verify a LeNet-style CNN...
+ During development of the Conv2D abstract transformer, we inadvertently
+ implemented an off-by-one error...
```
**作用**: 让读者知道这不是实验，是真实bug

---

### 改动2: 动词选择
```diff
- We inject a parameter mismatch
+ we inadvertently implemented an off-by-one error
```
**作用**: "inadvertently"（无意中）vs "inject"（故意注入）

---

### 改动3: 因果关系
```diff
- modeling verifier--model semantic drift
+ causing verifier--model semantic drift
```
**作用**: 不是"模拟"drift，是真的造成了drift

---

### 改动4: 结尾升华
```diff
- confirming that the unsoundness stems from incorrect Conv2D operator handling
+ confirming that BBL successfully detected a real implementation bug during
+ development
```
**作用**: 强调这是真实bug，BBL有实用价值

---

## 📝 学术价值定位

### 原文定位（模糊）
> 看起来像controlled experiment的一部分

### 修订后定位（清晰）
> **Real-world debugging case study** - 展示BBL的实际应用价值

---

## 🎨 论文中的位置

**建议**: 在Case Studies section明确分类

```latex
\subsection{Case Studies}

We present two case studies demonstrating our framework's effectiveness:
\textbf{Case 1} illustrates systematic mutation detection (M3\_SWAP),
while \textbf{Case 2} demonstrates practical value through a real development
bug that BBL caught during implementation of our Conv2D abstract transformer.
```

这样读者就知道Case 2是真实debugging，不是controlled experiment。

---

## ⚠️ 审稿人可能的反应

### 原文（"We inject"）
**审稿人可能想**:
- "Where's the code for this stride injection?"
- "Why isn't this in your mutation taxonomy?"
- "Can you reproduce this experiment?"

### 修订后（"real bug"）
**审稿人会理解**:
- "Oh, this is a development story, not a systematic mutation"
- "BBL helped them debug their own code - nice practical validation"
- "This shows the tool is useful beyond controlled experiments"

---

## ✅ 推荐方案

### 如果导师/审稿人比较严格 → 用版本2 (alternative)
- 标题就说"Detecting a development bug"
- 明确是"real-world debugging scenario"
- 无法被误解为controlled experiment

### 如果想保持简洁 → 用版本1 (rewrite)
- 只改关键措辞
- 保持原结构
- 最小化改动

---

## 🎯 与导师沟通

**如果导师质疑Case 2**，可以这样回答:

> **Advisor**: "Where's the code for the stride+1 mutation?"
>
> **Your answer**: "Case 2 describes a real bug we encountered during
> development of the Conv2D abstract transformer, not a systematic mutation
> from our M1-M6 taxonomy. We've revised the text to clarify this is a
> real-world debugging scenario that demonstrates BBL's practical value
> for catching implementation errors. The stride bug was fixed in commit
> [XXX] after BBL detected it."

---

## 📊 完整性检查

如果保留Case 2作为真实bug案例，确保：

- [ ] 文字明确说明是"development bug"或"inadvertently"
- [ ] 不要用"inject"、"introduce"等主动词
- [ ] 说明BBL的实用价值（不只是controlled实验）
- [ ] 如果可能，提供git commit证明修复过程
- [ ] 与Case 1（systematic mutation）明确区分

---

**总结**: Case 2是**真实且有价值**的案例，只需改措辞澄清它是debugging story，不是controlled experiment！
