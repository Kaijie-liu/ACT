# 同 pair 生命周期的净 Work 合同

**两条保留映射的路径，其新增生命周期计数之差已可闭合；完整模型仍不准入。**
对同一来源、私有 Plan、候选和缺失模式，在两臂都完整消费的条件下，global 路径比
scoped 路径多付每 pair 四次封存。这个条件等式补齐
[实施前分析](hz_pair_scope_lifetime_20261018.md)中未定的 `J_global-J_scoped`，
不改变 [R1 控制失败](hz_pair_scope_lifetime_results_20261018.md)。

本次起点 `d181f6b37a0ff2ab3ead32d16de74df2460a9909`，分支正确、工作树干净。
只读源码与保存记录；[静态重算](../scripts/assess_pair_lifetime_net_cost.py)仅解析文件、
整数公式和既有计数，不导入验证器或生成来源／候选。

## 共同生命周期费用

设 R 为完成的 pair 数，M 为 tile 数，C 为经 pair workspace 执行的 pullback 次数。
C 包含每个端点一次 objective 回传，以及每个至少一个候选 present 的检查块一次
dual 回传。即使块内只有一个候选 present，dual 的列数仍为该块的完整宽度。
Plan 构建时每 pair 的 gate 回传没有 workspace，不进入这个 C，也不享受新遍历收益。

| 调用位置 | 计数 |
| --- | ---: |
| PairLease 建立时检查四个封存块 | 4R |
| 每 tile 的 select、consume、finish 各一次 plan 与 lease 检查 | 18M |
| 每次 workspace pullback 的 lease 检查 | 5C |
| global 路径额外封存四个映射 | 4R |

scoped 的四次封存已包含在 canonical 准备成本 H 中，不能再计一次。因此：

```text
J_common = 4R + 18M + 5C
J_global = J_common + 4R
J_scoped = J_common
J_global - J_scoped = 4R
```

旧 Scope 构造 K 在相同 pair 生命周期下抵消；scoped 的 selector 每调用 1 Work 已计入
gamma，不计入 J。释放与统计操作没有新增的 `Work.visit` 差异，但仍有墙钟成本。
这个等式不是完整 Python 运行时间、驻留内存或任意失败前缀的保证。

在每 pair 九条分类性质下，全部候选 present 的两种端点计划为：

| 端点数 | tile 布局 | objective／dual 回传 | J_scoped | J_global |
| --- | --- | --- | ---: | ---: |
| 9 | 8、1 | 9＋5 | 110 | 114 |
| 18 | 8、8、2 | 18＋9 | 193 | 197 |

缺失证据会减少两侧共同的 dual 调用数，不能沿用全部 present 的 C；但在相同缺失模式、
同样完整结束的条件下，差值仍是 4R。中途异常、截止或发布失败没有这个完成保证。

## 注册拓扑上的条件差量

沿用已经核对的 Shared／tuple-node 计数：全局 N/F、pair 内 S/T，以及

```text
H = 2(N+F) + S+T + 53
G_b = (b+1)(N-S) - 3S
gamma_b = G_b * 1[G_b>1] - 1
```

一对 retained 路径完整完成的精确计数差为：

```text
Delta_pair = endpoints*gamma_1 + sum_present_blocks(gamma_block_width) - H + 4
```

这不是两个无关总上界相减，而是同来源／同候选的耦合差量。共享源检查、角色与符号、
残差、范数和完整聚合均未省略。Wire descriptor 的计费不同，不套用本式。

将新闭合的 `+4` 加入原先 135 个计数角点的凸上界，28 pair 总共增加 112：

| 完成假设与角点 | 单检查器净 Work 节省的乐观上界 |
| --- | ---: |
| 全部 present，所有计数角点的最大值 | +2,839,088 |
| 允许 completed-missing，所有角点的最大值 | +2,839,088 |
| router 与全部专家全不稳定角点，present | +1,602,608 |
| router 与全部专家全稳定正角点，present | −747,012 |
| 同一全稳定正角点，允许 completed-missing | −746,872 |

这里仍使用 `max(0,G_b)-1` 的凸上界、必需因子下界，并允许各 pair 自选更有利的
9／18 端点模式。可选补偿没有被假定不存在，而是仅在上界计算中忽略其额外准备成本。
最大角点不是观测到的训练权重或已证明可达的状态；**正上界不是正收益保证**。
两遍 production／搬迁检查若都完成可以分别计账，不能据此将总请求自动准入。

已有合成记录对新 J 的算术核对一致：guarded 三个单 tile pair 对应 J_scoped=246、
J_global=258；六输出来源 28 个双 tile pair 对应 3,220、3,332。其算术一致性不替代
未执行完的 factor 污染测试，更不代表完整注册拓扑已经运行。

## 下一步边界

本项结束旧 J 差量未闭合的问题，不再追加相同生命周期包装或小型性能对照。
R1 的最小待决修复仍只是两个 factor 污染子用例，使用已有多因子来源；未经另行批准
不修改测试、不执行补跑，也不将两轮拼成一次全部通过。

完整两遍来源合法性访问仍至少 13,922,736>8M，membership 限制及完整自有内存账本的
剩余缺口未被这个差量等式消除。它不能拿来产生新额度、外推 300 秒完成能力或启动
模型／GPU。该执行改进不改变抽象精度或接受下界，不能单独充当 G1 的算法创新验收。
G1–G6 保持 OPEN；下一次真实算法验证需要解决这些明确准入问题，而不是继续累加
组件测试数量。历史 23 增益的来源缺口和外部领先未确立的状态保持不变。
