# Correct Framing: HyZor 是 Verifier, ORT 是 Audit Guard

**Date**: 2026-06-07
**Reason**: Advisor 2026-06-07 修正了之前不严谨表述

---

## ❌ 之前不严谨的说法 (撤回)

- "walker-CERT ∩ ORT-clean = official CERT" — **错误**: 暗示 ORT 是 CERT 条件
- "Pure walker without ORT filter" 作为选项 — **错误**: 会让实现 bug 进结果
- "ORT 找到反例 / 找不到反例" 当作判断证据 — **错误**: 采样不构成证明

## ✅ 正确的角色定义

### CERT (Verified Safe)

```
来源:     HyZor / FCHZ walker 数学 verdict
        - forward HZ closed-form bound, 或
        - F1 LP bound (HiGHS, deterministic)
        - sparse-slack compression (sound)
        - sigmoid analytical chord (sound)

不来源: ORT 采样 (NEVER)
        random 采样 (NEVER, P5 violation)
        gradient / PGD (NEVER, P2 violation)
        BaB / Gurobi / MILP (NEVER for strict)

数学:    bound < t (deterministic check)
        for all (d, t) in unsafe spec
```

### FAL (Verified Unsafe)

```
来源:    HZ/LP 结构化 witness decode (deterministic)
        + strict ORT replay (zero-tolerance)

不来源: random search (NEVER)
        PGD attack (NEVER)
```

### ORT (ONNX Runtime) 真实角色

```
ORT 不是 verifier.
ORT 不产生 CERT.
ORT 不产生 FAL.
ORT 不搜索反例.

ORT 是 audit guard:
  - 对 FAL witness: strict replay, 验证真实 NN 输出违反 spec
  - 对 CERT 候选: 可选 audit, 检测 walker/实现 bug
  
ORT 行为:
  - 找到反例 → "walker 算错了" 信号 → 撤回该 CERT (回查 bug)
  - 没找到反例 → audit 通过 (NOT 增强证明)
```

## 比喻 (更正)

```
之前错的:   学生 (walker) 交答案 + 老师 (ORT) 检查 → 老师 OK 才算分
正确的:    学生 (walker) 写数学证明 (deterministic, sound)
          检查员 (ORT) 单独抽查作业, 找逻辑漏洞 
          如果检查员发现矛盾 → 学生原始证明有 bug, 撤回
          如果检查员没发现矛盾 → 证明本身仍是数学证明, ORT 没增强它
```

## 内部 vs 对外

```
内部记录 (full diagnostic, 不对外):
  raw walker candidate count       (走通 + bound<0 的)
  audit-rejected count             (ORT 找到反例 → 实现 bug)
  accepted strict count            (剩下没 bug 嫌疑的)

对外/论文 (paper-grade):
  mathematical HyZor proof count
  + strict replay/audit-clean provenance
```

## 当前数字 (更正)

```
现 official:
  2013  STRICT FLOOR     可对外, defensible
  2107  STRICT CANDIDATE 待 1472 keys audit

已撤回:
  2144  双计错误 (proxy bug)
  2384  per-row r93 overlap bug
  2597  tail_radius soundness bug + dups

不再使用 "2144 audit-validated" 表述.
不再说 "ORT-clean = part of CERT".
不再问 "要不要不过 ORT filter".
```

## 今晚收益 (advisor's 5-point)

```
1. Sparse-slack + tail-radius sound compression
   支撑 CIFAR / TinyImagenet 这类大 CNN 跑通
   计数: 在 2107 candidate 中, 不在 2144

2. Walker op coverage 扩展
   MatMul 1D/2D, Sub broadcast, Reshape -1, Cast, Squeeze,
   Unsqueeze, Slice, Concat, Gather, Pad, Transpose
   让之前 parser-blocked 模型进入 HZ 路径

3. Sigmoid analytical bound (replace sampling)
   dist_shift 相关收益
   forward deterministic, 不是采样

4. F1 LP / closed-form HZ certificate
   small dense / safenlp / acasxu / tll 一部分收益

5. Audit 修复 (anti-inflation, 不是涨分机制)
   per-iid dedup (修了 per-row filter bug)
   r93 overlap detection (92+ records 撤回)
   tail_radius bug 修正 (Add/Mul preserve)
   prevent false claims, 不是新增 V
```

## 下一步 (按 advisor)

1. ✅ 修正所有 doc 里 2144 提法 (本文件 + STRICT_CANONICAL_AUDIT_MEMO)
2. ✅ 修正 ORT 表述 (本文件)
3. ⏳ Build BASELINE_1472_KEYS.json (重建 1472 baseline)
4. ⏳ Re-filter session 648 vs 1472 keys → confirm 94 fresh additions
5. ⏳ 若 confirm → 升 2107 → FINAL; 若不 confirm → 报真实数字

## 一句话总结 (paper-quote ready)

> HyZor / FCHZ walker 通过 forward HZ closed-form bound 和 F1 LP bound
> 产生数学 CERT verdicts (deterministic). ORT 仅作为 post-hoc audit guard
> 检测实现 bug, 不参与证明. 当前 strict P1-P5 数字: 2013 floor 可对外引用,
> 2107 candidate 待 1472 baseline overlap audit 升级.
