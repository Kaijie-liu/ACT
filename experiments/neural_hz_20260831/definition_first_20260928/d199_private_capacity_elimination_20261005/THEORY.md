# 私有阈值覆盖下的联合神经纤维消元

本轮得到一条限定的结构定理：当独立私有输入在整个共享源盒上都能覆盖每个 ReLU 的阈值时，完整联合神经图的查询凸包不必保留逐相位的私有幅值与守卫容量。共同的源与相位关系，加每门四条容量行就足够。若相位关联具有给定的 running intersection 分解，全局相位表也可由局部表取代。

这是供 Neural-HZ 本体研究使用的精确消元定理，不是已完成的新域或已认证实现。原生整数关系与原精确 HZ 神经图对应；改进在于明确哪些联合见证信息可以无损删除，而非宣称精确 HZ 表达不了这个集合。真正的深层组合、实际总成本和新颖性仍需证明。

## 原生关系和保留对象

保留原 HZ 的连续源、全部 signed binary 因子、EQ/LE、共享 latent/frame 身份、全部消费者和原输入 decoder。以下 beta=(sigma+1)/2 只是原 bit 的可逆记号。原生 beta 始终属于 {0,1}，零激活的两个合法标签都保留。空的新关系集合精确嵌入原 HZ，输出仍为同一赋值的仿射读出；包含偏序按共同接口的具体化集合包含定义，不声称廉价最佳抽象或完整格。

局部定理考虑完整乘积接口

```text
z in [0,1]^d,
p_i in [L_i,U_i], independent across i and from z,
h_i(z)=a_i^T z+b_i,
g_i=p_i+h_i(z), q_i=ReLU(g_i), i=1..k.
```

一般有界共享盒可先做准确仿射归一化，固定坐标折入偏置并保留其身份。不把其他父谓词、私有像素或旧 bank 静默独立化。令 G 是同时保存 z,p,q 和全部 beta 的这个真实局部图。

关键可观察结构条件为

```text
L_i+h_i(z) <= 0 <= U_i+h_i(z)  for every z in the shared box.   (COVER)
```

它允许所有门 crossing、非平行和非零偏置，不要求门近重复。对实际独立盒输入，令 c_i 为完整预激活的盒中心值，r_private_i 和 r_shared_i 为对应私有、共享系数的绝对值加权半径，则 COVER 等价于

```text
r_private_i >= r_shared_i + abs(c_i).
```

该式是数学结构判据，不依赖实例身份、历史标签或 terminal margin。实际系数、BN、输入界必须可靠绑定和认证；形状或 crossing 标签本身不能证明 COVER。

## 稀疏共同源关系

先允许查询坐标 beta in [0,1]，只用于描述 conv(G)，不改变原生整数语义。令 I_j={i:a_ij is nonzero}；未参与任何门的源保留原身份和界，不需提升。取一份共同相位质量 lambda_s，s in {0,1}^k，满足

```text
lambda_s>=0, sum_s lambda_s=1,
beta_i=sum_(s_i=1)lambda_s.
lambda^I_t=sum_(s restricted to I=t)lambda_s.

0<=mu_j,t<=lambda^(I_j)_t,
sum_t mu_j,t=z_j.
v_ij=sum_(t_i=1)mu_j,t.
B_i=b_i*beta_i+sum_j a_ij*v_ij.
```

v 和 B 可以内联，但内联和物化的不同费用都要计。[已有自由源定理](../d195_source_incidence_core_20261005/INCIDENCE_CORE.md)给出这份接口的真实共同混合：对每个 lambda_s>0，取

```text
z_j(s)=mu_j,s|I_j / lambda^(I_j)_(s|I_j).
```

分母至少为 lambda_s，故不为零；所有坐标在盒内，混合同时恢复源均值、局部源矩和原相位边缘。所有消费者用这同一批源点，不能每条行另选一份。

## 四容量联合理想性定理

在 COVER 和完整乘积接口前提下，上述共同源关系加以下四行，其在 z,p,q,beta 上的投影恰为 conv(G)：

```text
q_i >= 0,
q_i >= p_i+h_i(z),
q_i <= U_i*beta_i+B_i,
q_i <= p_i-L_i*(1-beta_i)+B_i.                    (CAP)
```

必要性：每个真实整数点取 one-hot lambda 和相应 mu。active 时 q=p+h，不超过 U+h；inactive 时 q=0，不超过 p-L。四行成立，线性性保证所有真实凸组合成立。

充分性：先由共同源核心选出一份真实自由混合，不要求它已经满足神经守卫。写 hbar_i=h_i(z)，并定义

```text
n_i=q_i-p_i-hbar_i,
Qhi_i=U_i*beta_i+B_i,
Nhi_i=-L_i*(1-beta_i)-hbar_i+B_i.
```

COVER 保证每个源点的 U_i+h_i 与 -L_i-h_i 非负。CAP 等价于 0<=q_i<=Qhi_i 和 0<=n_i<=Nhi_i。若 Qhi_i>0，取 alpha_i=q_i/Qhi_i；否则 q_i=0，取 alpha_i=0。对 Nhi_i 同样定义 eta_i=n_i/Nhi_i 或零。两系数均在 [0,1] 内。

在同一个相位为 s、源为 z(s) 的原子上，给每门分配

```text
s_i=1: q_i(s)=alpha_i*(U_i+h_i(z(s))),
       p_i(s)=q_i(s)-h_i(z(s)).
s_i=0: n_i(s)=eta_i*(-L_i-h_i(z(s))),
       p_i(s)=-n_i(s)-h_i(z(s)), q_i(s)=0.
```

COVER 给 L_i<=-h_i<=U_i。因此 active 的 p_i(s) 位于 [-h_i,U_i]，inactive 的 p_i(s) 位于 [L_i,-h_i]；私有界、原符号守卫和 ReLU 值都正确。原子上的所有门可同时分配，因为各 private 变量独立。汇总恢复 q_i、n_i，并由 q_i-n_i-hbar_i 恢复 p_i。零容量无需除法，弱守卫保留零点双标签。故产生一个真正的完整局部神经混合，证明充分性。

这不是说 canonical 源点单独满足旧标签的守卫。新的关键正是：在覆盖条件下，可以合法重新分配私有输入以满足全部守卫，同时保住它们各自的物理均值。没有把这一混合存在性当作原始网络 ADV 见证。

## 有界关联宽度的表消元

设给定一棵相位 bag 树，满足 running intersection，每个 I_j 包含于至少一个 bag C。只存各 bag 的完整非负相位表 lambda_C；一个 root 表归一化，每条树边的完整 separator 边缘逐格一致，每个原 bit 的边缘绑定一次。由任一包含 I_j 的 bag 导出其相位边缘，再放置上面的 mu_j,t。

这些局部相位表具有一份共同全局 lambda。可沿树逐次以 separator 条件分布胶合；零质量 separator 下的子格也全为零，不需除以零。这是经典离散 junction-tree 定理，而不是新的概率推断方法。[Wainwright 与 Jordan，2.5.2 节 Proposition 1](https://people.eecs.berkeley.edu/~jordan/sail/readings/wainwright-jordan-fnt.pdf)给出了该条件及最大 bag 带来的指数费用。

存在的全局 lambda 交给前述稀疏源构造和私有容量分配，即得到完整局部神经 hull。推导不要求运行时物化全局 2^k 表；局部 2^|C| 表仍明确收费。此处没有逐相位求解、输入分区、消息优化、额外验证器或 helper。若未来实现静态扩展，仍须是一次统一的关系编译与原终端路径，不据证明启动分支子任务。

仅有 source/phase 二分 forest 时，edge McCormick 已足以得到共同自由源核心，COVER 再给联合神经 hull；该特例的完整同源单门 hull 交也已足够，不能在那个特例寻找新的单门精度分离。有环关联则可确实超过单门乃至全部两门接口，见 [物理控制](CONTROL_AND_SCOPE.md)。

## 原生非凸性与跨层算子范围

beta 为整数时，每个局部相位表被迫 one-hot；mu 只在对应格为 z。于是 B_i=beta_i*h_i(z)，CAP 恢复原 ReLU 图：beta=1 给 q=g>=0，beta=0 给 q=0>=g。此原生等价性甚至不需要 COVER，只需合法 private 界；COVER 用于分数查询层的理想性。连续源、原 bits、guards、原 EQ/LE、所有消费者和 decoder 均不删。

Affine/Conv 和共同身份的 Add/Concat 对 z,p,q 作同一线性读出，精确保留此关系。追加下一 ReLU 时，原整数图变换保持精确和健全；局部 hull 上的一般下一 ReLU 查询却不自动是两层神经图 hull。只有新接口再次满足完整乘积、共享盒和 COVER，才能重用这里的理想性定理。禁止把上一层仅有的均值假装成新的独立输入。

若 p 只是多个原像素的标量仿射读出，将其绑定回像素均值并不能使本定理成为全像素理想 hull。一般父 HZ 与这些局部有效关系相交仍健全，原生图仍精确，但分数理想性不可继承。完整原始图与历史谓词只读且保留，不以新 bank 替换它们的强关系。

因此本轮尚未提出具备可付深层闭包的最终 Neural-HZ。它比单纯变量换名更具体，因为证明了一个联合存在量化的精确消元条件；但表达本身仍是 HZ 可承载的混合整数线性扩展，不能以此宣称新集合表达力或 PLDI 新颖性。
