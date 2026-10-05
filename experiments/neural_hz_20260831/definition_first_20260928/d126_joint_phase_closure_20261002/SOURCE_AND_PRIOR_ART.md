# 联合消费者的真实图范围与已有方法

本轮补齐两个 CIFAR 保存原图中的剩余仿射支路范围，并核对多神经元、共享符号与消元先例。拓扑提供了一个有用更正：父幅值不必永远作为最终独立残差输出保留，但它在此前的所有宽层消费者仍需统一处理。没有数值秩、低维核空间或真实收益结论。

## 保存原图中的完整潜在前沿

从原 Relu_2 的幅值 Q 出发，沿 Conv、已认证推理 BN、Add、Flatten、Gemm 传播它的仿射系数；到每个 ReLU 时把该消费者记入前沿，并将其输出作为新的保留坐标。以下列表描述 tensor 拓扑上的潜在非零依赖，不证明每个标量系数非零。

| 模型 | 保存原图的潜在 ReLU 消费者 | 最后的原始 Q 仿射路径 |
| --- | --- | --- |
| CIFAR100 large | 5、11、17、25、31、39、45、53、59 | Add56 到 Flatten57 到 Gemm58 到 Relu59 |
| CIFAR100 medium | 5、13、19、25、31、39、45、51、57 | Add54 到 Flatten55 到 Gemm56 到 Relu57 |

large 的首个 identity shortcut 是 Q 的 tensor 127 进入 Add8；medium 的 tensor 121 经 Conv8 和 BN9 进入 Add10。两个 Add 的输出都还进入更晚的 residual Add，不能在首个块就宣布所有 raw Q 消费已关闭。

两个模型最后均为 head Gemm、head ReLU、最终 Gemm。保存图中没有绕过所列全部前沿 ReLU、直接把 raw Q 送往最终输出的仿射旁路。因此把完整前沿包含在候选语义后，最终直接 raw Q 端口为空。这不表示只保留窄 head 就足够：此前所有宽层 preactivation 对 Q 的联合映射仍须覆盖，相关秩是整个前沿堆叠映射的秩。本轮没有读权重数值或计算这个秩。

Tiny 的既有记录证明 shortcut 存在，但本轮没有重新认证其完整原图前沿。不使用旧转换图的 head 宽度或旧 BN 结果填补这个缺口。D120 五个位置的 1600 个局部读出不是上述完整空间人口。

## 可复核的本地来源

- [large 原图](../../results/d025_interval_capacity_20260930_v1/complete_0.json)，字段 packet.graph.nodes；模型 SHA256 为 5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16。文件 SHA256 为 fbab84537df071153a7d9362161b2c9aa85b80c8b4c605ff1e1149170e0965e0。
- [medium 保存记录](../../results/d015_source_shielding_20260928_v2/partial_source_evidence.json)的第 2 项，字段 packet.graph.nodes；模型 SHA256 为 aba117ad0ad4abdd630c220beca70cd58825e72e7bada5dffdda10bb725cece4。文件 SHA256 为 bbf0d17e48439edc11b352ad4d38ea8fc3ad0e9e0bf99108802b51fd7faac456。
- [D125 的消费者合同](../d125_signed_phase_component_20261002/NEXT_REAL_STRUCTURE.md)保留原来的证据和限制，不修改其冻结结论。本次新图结论只写在这里。

这些是已保存数据的只读拓扑核对，没有新模型解码、forward、source worker 或候选资格。

## 不能重新包装为创新的先例

[PRIMA 的第 6 节](https://files.sri.inf.ethz.ch/website/papers/mueller2021precise.pdf)已经把多神经元约束和部分精确 MILP 编码结合。因此“联合行加原整数相位”不是足够的区分；其逐步求解与界细化流程也不因此变成本项目获准的运行路径。

[Tjandraatmadja 等的 Theorem 1](https://arxiv.org/pdf/2006.14076)描述了仿射输入盒上单个 ReLU 图的完整凸包，使用多维输入信息，而非仅一个 preactivation 区间。候选不能只赢标量 triangle 就宣称新的同源关系优势。

[Sharp Hybrid Zonotopes](https://arxiv.org/html/2503.17483v2)研究保留混合因子时的 RLT 提升。原相位与连续源乘积、联合有效关系与松弛强度必须分别比较；多项式写法本身不自动更强。

[正矩阵 small-gain 论文](https://arxiv.org/pdf/math/0506434)的 Corollary 7 证明明确使用非负 Neumann 逆。有限 H_K 是基础单调代入的应用，不主张发明这种代数。

[Symbolic Bucket Elimination](https://ssanner.github.io/papers/cpaior18_sbe.pdf)已经用分段符号关系消元，并在第 4.3 节研究 ReLU 网络的参数化输出最大化。把保留边界源的分段消息改名为 Neural-HZ，并不能获得新颖性；分段大小与实际消元费用仍必须支付。它依赖 case/XADD 运算，本轮只作比较，不导入相位搜索或另一个终端求解路径。

[Noori 等的完整 ReLU 二次约束](https://arxiv.org/pdf/2407.06888)在 Theorem 1 给出由 copositivity 刻画的完整 QC 家族，第 5 节讨论数值松弛和计算费用。共同二次残差、增量关系本身也不是空白；保原 bits 后怎样省去幅值、怎样前向查询仍需独立算法，不能只是换一种约束容器。本轮不引入 SDP 或新求解路径。

[Constrained Polynomial Zonotopes 的 Proposition 5](https://arxiv.org/pdf/2005.08849)将线性映射写为改变中心和输出生成矩阵，保留原因子指数及约束。因此“非凸多项式表示对投影闭合”不自动意味着内部 latent 被删除。本项目也不因引用它而允许退化或整体替换成 CZ、Zonotope 或其他域。

小分离边界可能有用，但不能只由边界维数预告精确消息成本。SBE 本身讨论连续 case 表示的复杂度；[Telgarsky 的三角函数复合](https://proceedings.mlr.press/v49/telgarsky16.pdf)还给出固定宽度、一维源而线性片段随深度指数增长的经典例子。这个先例仅用来否定无条件的紧凑性承诺，不把此类构造变成实验目标，不围绕特殊网络另写规则。

这些先例不证明本项目所有候选均无价值，也不建立“必须超越所有等价 HZ 编码”的不可能晋级门。需要证明的是一条明确的新抽象或组合算法，在公平信息、相位、成本及真实结构下有用，而不是术语区别。

## 比较范围

至少保留旧 HZ 加同一有效关系作为强参考；在小控制上还应对照适用的 input-aware 和多神经元包络。不能要求非凸集合在线性 support 上严格超过它自身的完整凸包，因为两者 support 相同。可争取的价值是紧凑的相位条件关系、经过后续非线性后的组合精度和实际可计算成本。

2026-10-02 Australia/Sydney；redu-hz；HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。配置为纸面研究、原始文献和已保存图只读审计；没有新数值执行、生产修改或成绩更新。
