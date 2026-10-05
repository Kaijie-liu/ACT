# Attention 的共享源乘积与联合约束候选

本轮从 D105 指出的融合 Attention 余项缺口继续研究。纸面结论是：共享概率乘积和 simplex 守恒仍可能遗漏 Softmax 与其源之间的联合信息；一个已知的单调性事实可以在同一乘积表示上变成线性约束，并严格排除一个三 token 双通道控制中的伪状态。它不是新的 Softmax 定理，也尚未建立 PLDI 级新域、真实模型收益或 GPU 能力。

上一轮归档属于 progress：补齐 D105 和可核验恢复入口。本轮新的证据是 [严格控制](CONTROL.md)及 [真实接入和完整费用边界](COST_AND_BINDING.md)。全部是纸面推导和静态阅读，没有 freeze、RUN、候选执行或模型加载。

## 域元素与具体化

工作定义为 D=(H,N,M,E,decoder)。H 保存同一个 latent/frame 中的原连续因子 xi、全部原二元因子 b、EQ/LE 和输入重构。N 保存经来源绑定的归一化关系；M 保存被多个输出共同引用的概率加权源乘积；E 保存由这些关系推得的有限联合约束。此记号是研究候选，不是把已有具名乘积首次发明出来。

对一组 n 个普通有限 logits，用 z_j 统称本组实际读取的原连续 xi 和 signed 原二元 b。这里的统一记号绝不把 b 当成连续变量。认证的仿射源图和输出为：

```text
s_i = c_i + Σ_j A_ij*z_j
V_ik = d_ik + Σ_j B_ikj*z_j
p_i = exp(s_i)/Σ_r exp(s_r)
T_ij = p_i*z_j
Y_k = Σ_i d_ik*p_i + Σ_(i,j) B_ikj*T_ij.
```

只有 score 或 value 实际出现的 (i,j) 产品需要具名；同一产品在全部通道、残差消费者中使用同一身份。这里假设 s 已包含真实 scale；正 inverse temperature 也可先并入 s。当前纸面范围不包括未经认证的 mask、dropout 或轴变换。

具体化要求存在同一 z、全部原 b、p、T 和其他内部连续量，同时满足 H、N、M、E，再用原 decoder 解释输入及当前输出。语义包含次序为同一可见变量上的集合包含，不声称可计算最佳抽象、完整格或普遍高效包含检查。N/M/E 为空时精确嵌入原 HZ。保留原 bits 和非线性共同图使其没有整体退化为 Z/CZ 或区间。

精确实数层面，给定已认证的 s、V 和原源，p、T、Y 都有规范共同见证。仿射/Conv 及同 frame Add/Concat 只改变读出、复用同一关系库；新 ReLU 仍使用原门的二元相位与完整 guard，零点两标签保留。对 smooth 函数只有已列明的 Softmax 图，不能顺带声称覆盖 GELU/LN。

这里最重要的语义区别是：保留原 HZ 谓词和 bits，不等于保留一个旧近似 HZ 的全部伪赋值。当前生产融合结果 exact=False，新增有效关系可以排除其伪误差状态。健全性要求每个真实网络输入及原合法相位仍有同一个扩展；不能声称对旧近似集合的精确无损重参数化。端到端浮点语义、真实节点绑定和见证协议仍须独立认证。

## 普通终端的有证线性外包

查询只使用已证线性行及原 LP/MILP，不调用非线性、SOC、SDP 或新的 dual 求解。保留原概率的全部可用身份、界、simplex 和 ratio 谓词，不能用一份概率界替代真实来源绑定。对产品使用可靠界生成 McCormick 行；原 signed binary 留作整数时，binary×p 的四行编码精确，不增加替代 bit。

当一个源因子 z_j 的全部 token 产品保留时，加入 Σ_i T_ij=z_j。若仅保留 token 集 S_j，则遗漏产品之和可消去，得到：

```text
−(1−Σ_(i∈S_j) p_i) ≤ z_j−Σ_(i∈S_j) T_ij
z_j−Σ_(i∈S_j) T_ij ≤ 1−Σ_(i∈S_j) p_i.
```

因为 |z_j|≤1、p_i≥0，真实遗漏和 z_j*Σ_(i∉S_j)p_i 满足这两行。它们是已知 simplex 产品守恒的线性推论，不是新域定理；也不构成含任意 HZ 谓词及 Softmax 绑定的完整联合凸包。

## 可编译为一行的源联合关系

令 u=(1/n,...,1/n)。Softmax 是 log-sum-exp 的梯度，故单调性给出：

```text
E = (p−u)ᵀs = Σ_i p_i*s_i − (Σ_i s_i)/n ≥ 0.
```

自含证明：f(s)=log Σ exp(s_i)，其 Hessian 为 diag(p)−p*pᵀ。对任意 v，vᵀ Hessian v 是概率 p 下的方差，非负。沿 0 到 s 积分，sᵀ(grad f(s)−grad f(0))≥0，且 grad f(0)=u。无需相位或输入划分，也无需求解器状态。

在上述同一乘积坐标中，E 是线性读出：

```text
E = Σ_i c_i*(p_i−1/n)
    + Σ_(i,j) A_ij*T_ij
    − Σ_j [(Σ_i A_ij)/n]*z_j.
```

因此可统一为每个认证的 Softmax 组加入一条线性下界，不按模型身份、margin、公开标签或 LP 状态挑选。原二元 z_j 原样出现。均匀参考不要求源域包含零点；定理在整个实数域有效。有限精度实现仍须认证系数求和、1/n、舍入和源界；本轮没有赋予实现资格。

[Gao 与 Pavel，Proposition 3](https://arxiv.org/pdf/1704.00805)已有 Softmax 单调性。[Nair，Section 4.1 Corollary 1](https://arxiv.org/pdf/2510.23012)还给出 E≥2*||p−u||²（对标准 Softmax）。本候选仅使用其线性弱化 E≥0，不添加平方变量或新求解器。上述公式本身均不算本项目创新。

## 与已有抽象域和本项目的区别边界

原有限线性 HZ 无法精确表示一般 Softmax 输入输出曲线，N 扩充了数学表达，但仅记录函数图仍可能只是计算图包装。共享二次产品在多项式域、精确乘法和本项目 [D035](../d035_cross_phase_source_20260930/THEORY.md)已有先例；perspective/共同源提升见 [D018](../d018_order_relations_20260930/COMMON_SOURCE_COMPARISON.md)，单源乘积凸包限制见 [D049](../d049_mixed_source_envelopes_20260930/ALTERNATIVES.md)。

本轮证明的区分仅是：同时保留产品守恒和强边际仍不足，认证的 Softmax 源联合行提供额外信息。尚未证明它超越完整 IQC、精确多项式产品图或完整联合余项图。外部强比较及其访问范围见 [先行研究记录](PRIOR_ART.md)。需要进一步证明真实重复结构上的持续传播、可控费用和净能力，才有可能构成所要求的定义贡献；不为已有公式换名后直接晋级。
