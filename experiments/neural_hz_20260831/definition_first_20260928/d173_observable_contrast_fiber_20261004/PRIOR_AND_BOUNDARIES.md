# 联合关系研究的先例和适用边界

本页记录本轮为避免重复研究而核对的先例，以及未作为通用 Neural-HZ 候选实施的两条思路。主推导见 [共同均值与差异纤维](THEORY.md)。论文关系仅用于定义推导和新颖性比较，不接入其他验证器或其求解算法。

## 已有项目结果

| 已有记录 | 本轮不能重新宣称的贡献 | 仍未解决的问题 |
| --- | --- | --- |
| [D007 同源 circuit](../d007_conservation_bundles_20260928/D007_DOMAIN_AND_RESULTS.md) | 从 affine dependencies 推出幅值守恒；所有 circuit hull 仍可能缺共同像 | 完整源与多项关系的统一见证和成本 |
| [D020 相位差分](../d020_phase_difference_20260930/D020_PHASE_DIFFERENCE.md) | 两原 bits 的四行理想差分关系，严格强于 phase-free sector | 局部投影遗忘共同 baseline，不能当完整源理想性 |
| [D091 共同见证传播](../d091_joint_realization_relay_20261001/THEORY.md) | 已能穿过 mixed affine 和下一 ReLU | 观察量与历史接口增长、真实网络总费用 |
| [D126 联合像边界](../d126_joint_phase_closure_20261002/JOINT_IMAGE_BOUNDARY.md) | 全 affine identities 加所有 pairwise 和差界不联合完备 | 更高阶且保相位的共同关系 |
| [D155 共同纤维投影](../d155_common_fiber_projection_20261004/THEORY.md) | 共同消元、同一源见证与精确投影原则 | 少一个幅值可付出更多行和 nnz |
| [D156 双分支纤维](../d156_two_branch_consumer_fiber_20261004/THEORY.md)与[D158查询](../d158_joint_forward_support_20261004/THEORY.md) | 共同 packet、相位矩阵透视、代数前向支持已有 | 点态与全局预算差异、保真 lowering 与真实完整代价 |

这些边界没有证明所有 Neural-HZ 不可能；它们要求后续明确改变了哪一项表示、联合精度或完整费用，而不是换名称。

## 直接相关的外部研究

[Fazlyab、Morari、Pappas 的 quadratic-constraint 框架](https://arxiv.org/html/1903.01287v3#S3.SS3.SSS3)第三节 C3、Lemma 2 用非负加权图 Laplacian 汇集 repeated nonlinearity 的两两 slope 关系。本轮的等权 completegraph QC 是它的直接特例；采用这种关系本身不是新颖性证据。本文不采用其 SDP、S-procedure 优化或额外验证路径。

[PRIMA](https://arxiv.org/abs/2103.03638)研究多神经元联合凸外包及可扩展凸包近似。它说明“比逐门凸包更强”不是新的研究结论。本轮保留原二元语义、共同源和输入重构；不把整个域替换成 PRIMA 或其凸域，也不使用其算法作为 helper。

[Tjandraatmadja 等的 tightened single-neuron relaxation](https://research.google/pubs/the-convex-relaxation-barrier-revisited-tightened-single-neuron-relaxations-for-neural-network-verification/)保留 affine preactivation 的多维输入信息，比单变量 triangle 更强。本轮正控因此明确比较 retained-source 单门 hull，而不是只比较松区间；分离来自共同混合不能独立实现。

[Goubault 等的 tropical polyhedra](https://arxiv.org/pdf/2108.00893)第三节说明另一种取舍：ReLU 适合 tropical 表示，经典混权 affine map 仍可能需要近似。它启发的是必须同时证明线性层与非线性层的闭包，不能只展示 ReLU 一步变简单。没有因此改成 tropical 域、执行 subdivision 或把完整图包装为新域。

[Froese 等关于 zonotope 问题的公开问题](https://proceedings.mlr.press/v291/froese25b.html)讨论 ReLU 与几何优化之间的复杂度联系。本轮只把它作为“换表示不自动使精确查询容易”的背景，不把 zonotope 当新域，不用一般困难性否定普通结构突破。

## 直接消费 pair Laplacian 的限制

设 L=Σ a_ij(e_i−e_j)(e_i−e_j)ᵀ、a_ij≥0。已有 QC 为 rᵀLr≤gᵀLr。若希望不作有损投影，仅把这一多项式通过 y=Ur 精确改写，且要求其在整个线性纤维 r+ker U 上取值不变，那么必须 ker U⊆ker L；沿 k∈ker U 的二次项要求 kᵀLk=0，PSD 性给 Lk=0。反之满足该条件时，可由 U 的商空间写成相应输出二次式。

对非负边权，条件还要求每条正权边 e_i−e_j∈row(U)。例如 U=[[1,−1,1/2],[1/4,1,−1]] 的 kernel 由 (4,9,10) 张成；三个坐标不同，所以没有非零 pair-Laplacian 能直接这样因式分解。

这不是一般有损投影的不可能性，也不允许忽略已保留 source 对可行 r 的限制。本轮 mean–contrast 推导正是通过真正的存在量投影绕开“同一个表达式直接因式分解”的条件；其代价是近似、均值量、range/范数和查询费用。

## 相反列差分替换的窄正向规则

若完整消费者确为 Σ_j u_j[ReLU(f_j)−ReLU(h_j)]，每对可保一个差分幅值 a_j、两个原 bits 和全部原 guards，用 D020 四行：

```text
L_j η_j≤a_j≤U_j β_j,
−U_j(1−η_j)≤a_j−(f_j−h_j)≤−L_j(1−β_j).
```

每对2幅值降至1、8行仍为8行，但一般多2nnz(f_j−h_j)。旧双门 LP 的精确差分投影需要4guards加8端点差行，即12行；故这里是明确的8行有损替换，不是免费精确压缩。

普通控制 f=x+s/4+1/8、h=x−s/4−1/8，源[-1,1]²，得到 a+x/20≤4/5，下一ReLU(a+x/20−13/16)恒零；旧八行LP在x=0,s=1、β=3/5、η=0、r=33/40、p=0允许下一值1/80。但源零点、βη=10时新域允许a=1/4，真实仅1/8。

这条规则要求全部消费者具有认证的相反列结构。一般训练 mixed U 不能假定满足，近似列的残差也不能丢掉。其核心 D020 因子早已存在，本轮不以它替代完整研究目标或启动通用组件。

## 参数化不变性不是新的定义收益

正对角 D 给同一真实块的重参数化 (U D⁻¹,DV,Dc)。固定标量 λ 以及 ||E||·||V|| 的分开界不必对此不变，D172 的负结果仅针对其冻结证书，不是实际网络缺陷的下界。

逐列 λ_j=max(0,−u_jᵀv_j)/||v_j||² 在非零普通列下有 λ'_j=λ_j/D_jj²，可形成协变的加权 sector 部分；但随后若仍用分开的矩阵范数，最终证书又可能依赖参数化。逐列 Σ||u_j+λ_jv_j||·||v_j|| 保持不变但可能很松。这是经典 sector 证书的修正边界，不是新抽象域；本轮没有据结果重调 D172 的参数或重跑它。
