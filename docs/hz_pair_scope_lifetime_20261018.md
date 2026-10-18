# 同一 pair 的检查映射跨 tile 生命周期静态评估

**下一候选是保留已检查对象的有效期，不是跳过检查。** 同一 pair 的所有 tile 使用相同
forest 和完整 node/factor 映射；若映射完整检查后不可变，可以研究只构造、复核和封存
一次。当前代码没有实现这个生命周期。它有尚未被计数否决的机会，但不是新证书、
算法创新完成或完整模型准入。

起点 `59da49b657f4e817bf2f12c5fa7159aa68762e69`，工作树干净。
[上一轮否决](hz_support_amortization_20261018.md)继续有效：逐 tile 先准备的 R2 不应推广。
本轮只检查源码和整数公式；[标准库重算](../scripts/assess_pair_scope_lifetime.py)及
[绑定记录](hz_pair_scope_lifetime_20261018.json)不生成来源、候选或验证运行。

## 当前代码为什么不能直接复用

`Session.prepare` 从私有冻结 Plan 取得下一 tile，调用 `PairWorkspace.select`，后者要求
没有现存 Scope。`Session._consume` 每 tile 调用 `clear_tile`，释放局部累加器、目标
描述符和 Scope。`Driver` 另外销毁候选 token／frame，并保持单调 ordinal。

因此，延迟一次 `close` 不是安全实现。至少要分离 tile 临时对象释放与 pair 映射释放，
而不是保留整个旧 tile；同一 Scope 只能在当前 owner、forest、Work、session／Plan、
pair 和连续 tile 范围内使用。production 与搬迁 replay 必须各自构建、独立检查。

## 遍历与准备部分的计数机会

沿用上一轮 Shared／tuple-node 公式，N/F 为全局节点／因子，S/T 为 pair 内节点／因子。
原 Scope 构造成本与新增 canonical 准备成本分别为：

```text
K = 2(N+F) + 2(S+T) + 4
H = 2(N+F) + S+T + 53
G_b = (b+1)(N-S) - 3S
gamma_b = G_b * 1[G_b>1] - 1
h_b = max(0,G_b) - 1 >= gamma_b
```

为避免把生命周期优化冒充遍历算法收益，global 与 scoped 对照应获准使用同样的
pair 生命周期，使 K 抵消。每 pair 只支付一次 H 后，完整 present 的部分差量上界为
`max(10h_1+4h_2-H, 18h_1+9h_2-H)`；允许 completed-missing 时为
`max(9h_1+4max(0,h_2)+max(0,h_1)-H, 18h_1+9max(0,h_2)-H)`。

沿用已核对的 135 个凸上界计数角点，忽略补偿只为得到有利的上界：

| 条件 | 遍历及准备部分的单检查器最大上界 |
| --- | ---: |
| 全部 present | +2,838,976 Work |
| 允许 completed-missing | +2,838,976 Work |
| router 和全部专家均不稳定的角点 | +1,602,496 Work |
| router 和全部专家均稳定正的角点，present | −747,124 Work |

最大角点为 router 全稳定正、八专家全不稳定，N=18,984、F_min=12,288、S=7,080、
T_min=5,376、H=75,053；取 18 端点时每 pair 为 101,392 Work。这个角点不是已证明
可达的模型状态；正上界不证明任何真实权重获益，也不保证其他状态非负。

**表中不是完整实现的净成本上界。** 跨 tile 绑定、许可检查、释放和 global 侧可能需要
的额外封存尚未实施。设这些新增成本分别为 J_global、J_scoped，则完整差量为：

```text
Delta = 遍历及准备差量 + J_global - J_scoped
```

不能把两个非负成本之差默认成零或非负；两臂需各自完整计费。如果直接与旧逐 tile
global 比，还包含 `(tile数-1)*K` 的生命周期收益。该项不能归给遍历创新；新增补偿因子
还会增加被省掉的 K，所以不能忽略补偿后声称它是完整收益上界。原生强参考应同时保留。
这些公式只讨论完整消费；中途截止没有将来收益保证，Wire 也不适用这些精确差量。

## 保留检查的对象与状态合同

首次创建 pair Scope 时，完整 node/factor roster、canonical rank、ancestry、owner／
buffer 大小与四映射封存必须通过。准备失败立即终止，不退回未检查遍历。后续使用
保留所有 `local/global_id` 逆映射、源／dual／zero、符号、残差、范数和逐性质聚合检查。

跨 tile 保留的依据必须是同一个不可变已检查对象，而非 pair 名字或 hash 缓存命中。
`Block.put` 会拒绝 sealed buffer，但 Block 的属性本身仍可赋值。新 lease 至少绑定
四个 block 对象、owner、大小与原 sealed bytes 对象，检查 closed／released 状态及
可信 session／Plan 的连续 ordinal。旧候选 token、root descriptor 和累加器不得继承。
这仍是现有进程内 API 信任边界，不是防恶意 Python 任意改私有内存的沙箱。

global 参考若也封存映射，必须明确其最小需求和额外费用，不能人为强加仅 scoped
需要的检查来拖慢对手。所有数学接受门相同，不表示不同算法必须执行无关的同一串操作。
已有 R2 的首次封存前 roster 污染应仍被完整检查拒绝；跨 tile 替换、旧 token／错误 owner、部分
准备失败及截止是新增故障面，尚未测试，不能提前宣称完全等价。

## 内存与终态要求

只保留当前一个 pair。四映射原始 payload 为 `4(N+F+S+T)` 字节，封存还需临时复制
重叠；不是 CPython 全堆或 RSS 上界。保存 sealed bytes 的身份引用不应再复制字节，
但引用仍持有对象：释放必须清空 lease 中的引用，不能只把 arena 的 live_bytes 减到零。

可信 Plan 必须识别 pair 的最后一块，在该块结束时立即释放 Scope；不能等下一 pair
或 `finish`，否则残留映射会与终态来源核验、序列化额外重叠。切 pair 前释放旧对象，
不允许两个 Scope 共存。异常／硬截止保留已发生成本、禁止重试与退款。只有这些状态
和实际对象寿命被验证，才有资格声称峰值不增加；当前没有这项结论。

## 有限决定

这是一项值得单独审查的支持检查生命周期摊销候选，替代“跳过 canonical 检查”的方向，
不改变 eager R2 的停止决定。它本身是工程性摊销，不足以关闭 G1，也不能解决两次
完整源参数检查至少 13,922,736>8M 和 membership 拒绝。

下一实施／控制必须另行授权并冻结：默认关闭、当前 pair 单一 lease、强 global 参考，
完整身份／污染／截止／部分证据和计费控制。先给实际 J 和存活对象账本，再决定是否
有理由执行；不补跑旧 R2，不自动扩大控制、提额、启动模型或 GPU。本轮静态数值和
生命周期审查由另一名 AI 分别复核，不替代人类审阅。G1–G6 均保持 OPEN。

当前原生 production／portable Driver 只启用 `pair_scoped=True`，尚未启用 R2
`scoped_traversal`；更不能将这个未实现的跨 tile 生命周期写成已接通原生路径。
