# 跨层共同关系的适用边界

本页防止两个研究捷径被误当成强 Neural-HZ：固定二次型不能不看相位语义地穿过 ReLU，但也不能因此排除全部交叉能量；动态 Attention 的 QK 分解是真实重复结构，但仅有低秩和 normalization 在当前两个布局上没有额外普适概率约束。以下是纸面证明和只读来源核对，不是新验证结果。

## 相位兼容的非对角能量可以保留

若要求固定 G≽0 对任意坐标 mask P、任意向量 z 都满足 PᵀGP≼G，则独立坐标 mask 会迫使 G 对角化。取仅第 i 坐标保留的 P，G−PGP 为 PSD 且其 ii 项为零，因此该行列必须为零，得到 G_ij=0。这只适用于忽略 sign guard、要求任意 mask 对任意向量成立的规则。

真正 ReLU 的 mask 与输入符号绑定。若 G≽0 且非对角项非正，写 z=p−n，其中 p=ReLU(z)、n=ReLU(−z)、p_i n_i=0，则

```text
zᵀGz−pᵀGp=nᵀGn−2pᵀGn≥0.
```

因此该类非对角 G 确实收缩。图 Laplacian 加非负对角是便于构造的子类，但并非所有上述 G 都具有这种不缩放的分解；不能遗漏对角占优条件。对于固定非负边权，逐边 |ReLU(z_i)−ReLU(z_j)|≤|z_i−z_j| 也直接证明图能量收缩。

它可以跨 bank：若旧活 q≥0，新 g=q+f，则对联合向量 (q,g) 同时施加 ReLU，得到 (q,ReLU(g))，并保留连接两组对应坐标的图能量。具体地，

```text
Δ=ReLU(q+f)−q,    ||Δ||₂≤||f||₂,
B ReLU(q+f)+Tq=(B+T)q+BΔ.
```

共同 q 的取消在读出时先发生，不需要独立预算之和。[NeurIPS 2021 原文](https://proceedings.neurips.cc/paper/2021/file/b6417f112bd27848533e54885b66c288-Paper.pdf) Lemma 6 已明确讨论 ReLU/Leaky-ReLU 的 Dirichlet energy 收缩；这不是本轮首次发现的数学原理。本地 [D020](../d020_phase_difference_20260930/D020_PHASE_DIFFERENCE.md) 也已给出带原相位的差分关系。

单把每个新输出换成 q+Δ，仍有同样数量的连续幅值。[D151](../d151_exact_comparator_audit_20261004/THEORY.md) 的同坐标四行精确参照必须计入比较；范数关系并不自动删掉原始非线性成本。投影到更低维消费者则重新面对共同源、全部 guards、活 skip、原谓词和终端成员检查，不可以只报前端变小。

一个普通两层纸面控制展示该关系能消费什么，而不作为新候选收益。对 x,y∈[-1,1]，取

```text
q₁=ReLU(x+y/4+1/4), q₂=ReLU(−x/4+y−1/4),
f=(q₁−q₂+x+1/4, −q₁+q₂−y−1/4)/16,
z=ReLU(q+f).
```

父域可靠盒为 0≤q₁≤3/2、0≤q₂≤1，故每个 |f_i|≤11/64，||f||²≤121/2048。于是混权及活源 skip 读出 J=(z₁−q₁)−(z₂−q₂)+(x+y)/64 满足 J≤sqrt(2·121/2048)+1/32=3/8，下一 ReLU(J−7/16) 恒零。四门均 crossing，且两层有非零偏置。它优于丢掉 q/z 相关后分开支付预算的查询，但 D020 同信息差分界已经能给相同证书；不能将其计为胜过强旧 HZ 的能力。

## 当前 CIFAR 图不是未经混权的直接残差 ReLU

只读 [D151 来源记录](../d151_exact_comparator_audit_20261004/RESEARCH_RECORD.md)：large 的 Add8 后经过 Conv9/BN10 才到 ReLU11，Add8 的输出还活到 Add14；medium 的 Add10 后经过 Conv11/BN12 才到 ReLU13，且输出还活到 Add16。因此不能不经绑定就把这个后继说成 ReLU(q+small f)。如果实际预激活为 W(q+f)+b，与 q 比较的增量含 (W−I)q+Wf+b，必须全部支付。

这些现有 packet 尚缺完整跨 Add 参数，Tiny 也没有本轮新绑定。只读图或局部窗口不等于完成源资格；不能将上述界称为 CIFAR/Tiny 实测收益。

## 动态 Attention 的真实完整范围

正式未解结构 manifest 记 ViT 共 200 例、90 已解、110 未解。这是样本人口，不是证明某种结构造成 UNKNOWN 的因果证据。已存的两份 [PGD 图](../../results/d107_vit_graph_inventory_20261002_v1/pgd_2_3_16_graph.json) 与 [IBP 图](../../results/d107_vit_graph_inventory_20261002_v1/ibp_3_3_8_graph.json) 具有共同源 Q/K/V，动态 QK，Softmax/PV，混权 output projection，原 token residual 和下一 MLP/ReLU；首块节点 64 的结果还活到节点 73 的第二个 residual。

[D127 来源边界](../d127_native_attention_component_20261002/REAL_VIT_SCOPE.md) 已证明只有首 CLS query 是固定的，patch query 和后层 CLS 不能套用固定 query 公式。[D130](../d130_import_isolation_20261002/RESULTS.md) 只完成 38/192 个首 CLS 方向，完整来源超时，原生/模型/GPU 资格均未通过。动态 patch 结构仍值得研究，但不能通过改名称重跑失败来源或宣布已有全模型覆盖。

## 自由 QK 的低秩在这两个布局上不增加约束

考虑 κ≠0、N 个 token、key 维数 d≥N−1。给定任意严格正行随机矩阵 P，取

```text
k_N=0,    k_j=e_j  (1≤j<N),
q_rj=log(P_rj/P_rN)/κ,
其余 d−(N−1) 个 query 坐标为零。
```

于是 κq_rᵀk_j=log(P_rj/P_rN)，参考列 score 为零，Softmax 恰为 P。故单凭“score 有 QK 分解且每行归一化”，在 d≥N−1 时不能推出额外普适的跨 query 概率限制。

已核布局每 head 的 d=16，N 为 5 或 17，均满足该维数关系。这里的构造只针对自由 Q/K；实际同源仿射映射、输入盒、谓词和权重会施加额外限制，绝不能因此说真实 Attention 相关性无用。κ 的实际绑定尚未在本轮认证；若 κ=0，Softmax 已为均匀值，是另一明确情况，不套本构造。

[D108](../d108_centered_attention_relations_20261002/DEFINITION.md) 已允许完整动态二次 score 进入中心化关系；[D109](../d109_cross_query_research_20261002/RESEARCH.md) 已允许同状态动态 K 进入跨 query 循环。weighted-key 不是一般 PV，不能直接替换。本轮未得到优于强同源 Taylor/多项式参考的新动态不等式。

下一候选至少要共同保留同一 θ 的 Q/K/V、二次 QK、每 query 的同一个 p/分母、完整 PV 与两条活残差，并给出可付的统一前向查询。只保存精确生成元表属于 [D124](../d124_source_phase_fiber_20261002/NATIVE_ATTENTION.md) 的既有框架。朴素显式 p×源 每 query 为 N 乘源维数的产品量，二次 score 与 p 再相乘还有更高阶成本；不能把这些藏在“共同关系”四字里。
