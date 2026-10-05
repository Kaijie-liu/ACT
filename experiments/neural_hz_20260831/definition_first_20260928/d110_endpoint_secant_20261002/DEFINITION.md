# 共享端点割线的 Neural-HZ 关系模块

本轮将 D109 中只能直接连接加权 key 的关系，推进为对任意共同 value 均成立的端点割线合同。所有输出通道共用一组正系数，给出精确实数语义、普通终端的线性外包及明确的误用反例。它尚未通过强参考的严格正控制，不是已完成的新抽象域或 smooth 突破。

前一轮 D109 补齐了研究存档并校验关键清单，属于 progress。本轮仍是纸面推导、原始文献和生产源码只读研究；无候选、freeze、RUN 或数值执行。

## 域载体与端点身份

研究元素记为 D=(H,R,E,decoder)。H 保留原连续因子 xi、全部 signed 原 bits、EQ/LE、共享 latent/frame。R 是有限的同源端点关系块；E 是从这些块健全推得、安装到当前输出的线性关系。空 R/E 精确嵌入原 HZ；一般添加的是对实际非线性图的有证强化，不声称无损保留旧近似 HZ 的所有伪误差赋值。

每个块绑定同一个原赋值 theta 下两个真实 query 的 s(theta)、t(theta)、p=softmax(s)、q=softmax(t)，以及同一 value bank V_i(theta)。所有 logits 必须有限且支持集一致；温度已并入 logits。不同 head、layer、batch、mask 或 value 身份不能仅凭相同形状合并。V 和 K 均可随 theta 变化；不要求 V=K，也不要求它们在整个输入域内常量。

块的新增语义量由端点唯一确定，不能由每个消费者自行选择：

```text
d_i = s_i − t_i
c = logsumexp(s) − logsumexp(t)
lambda_i = L(p_i,q_i)
L(a,b) = (a−b)/(log(a)−log(b)), a != b
L(a,a) = a
S = sum_i lambda_i
```

有限真实 logits 保证 p_i,q_i,lambda_i>0。相等端点采用连续定义，不添加求解路径或相位 split。这个关系模块的图语义本身仍不足以构成创新，须由下面可生成、可消费的合同及强对照检验其价值。

## 精确的共同割线与 value 传递

由 log(p_i)−log(q_i)=d_i−c，逐项有 p_i−q_i=lambda_i(d_i−c)。质量守恒给出 c=(lambdaᵀd)/S，因此：

```text
delta_p = p−q = M_lambda d
M_lambda = diag(lambda) − lambda lambdaᵀ/S
Delta_Y = Y_p−Y_q = sum_i delta_p_i V_i
        = (1/S) sum_(i<j) lambda_i lambda_j (d_i−d_j)(V_i−V_j).
```

证明最后一行只需展开双和、交换 i,j 并使用 c 的表达式。每个输出通道使用完全相同的 lambda、c 和 delta_p；不为不同通道另取输入见证。原有共同 V 的加性分量在差值中精确抵消。V 与 logit 差没有匹配时，Delta_Y 没有自动的单调符号，这与 D109 的 −tanh 反例一致。

该块可以在同 frame Affine／Conv／Add／Concat 中复用源身份并线性变换读出。若 W 为下游固定混权，W Delta_Y=sum_i delta_p_i(WV_i)；这不是重新估计一份独立 lambda。进入原 ReLU 时仍保留原二元相位及全部 guard，将已获的线性行传给后继；不据此保证更紧 ReLU 界。两个零点标签原样保留。最终输入仍由原 decoder 重构，新增 ADV 仍须具体网络验证。

这些是实数层面的共同扩展证明，尚无真实 native 出生／浮点舍入／GPU 认证。只写这些字段而不能得到更有用的抽象变换，仍然只是函数图包装。

## 系数的可用性质

对数均值有积分表示 L(a,b)=积分_0^1 a^u b^(1−u) du。由此正性和对两个参数的单调性成立，且 L(a,b)≤(a+b)/2。因此 S≤1。令 w=lambda/S，M_lambda=S(diag(w)−wwᵀ)，为半正定且消去常向量。对单位向量 z，概率方差不超过其坐标跨度平方的 1/4，而跨度平方≤2‖z‖²，故谱范数≤S/2。记 P=I−11ᵀ/N，有：

```text
0 <= M_lambda <= (S/2) P
delta_pᵀ d >= (2/S) ||delta_p||².
```

后一行由 M_lambda²≤(S/2)M_lambda 得到。它是精确语义的条件关系，不是声称下面的有限 McCormick 外包自动蕴含它，更不为此添加新的二次或 SDP 求解器。

## 不能将当前割线当作通用 Jacobian

取 t=(0,0,0)、s=(log4,0,0)，于是 q=(1/3,1/3,1/3)，p=(2/3,1/6,1/6)。在 u=(0,1,−1) 方向，M_lambda 的特征值为 lambda_2=1/(6 log2)。但沿真实线段的平均 Jacobian 在该方向的特征值是：

```text
积分_0^1 1/(4^a+2) da = 1/4.
```

两者不等。二者只保证对当前 d=(log4,0,0) 作用相同。因此 lambda 可以共享给同一对端点的任意 V 读出，不能把 M_lambda 转用为新的任意扰动方向的精确导数摘要。这个普通三 token 反例经独立纸面复核，没有数值实验。

## 当前成立与尚未成立

成立的是任意共同 V 的精确端点传递、全部通道共享的幅值语义、一个可核算的线性外包和禁止跨方向滥用的反例。尚无通过完整指定 Taylor／ratio／已知增量关系参考的严格正控制；没有新域新颖性、实际吞吐、全物理资格或新增 CERT/ADV。

先行研究与项目对照见 [文献与源码](PRIOR_ART_AND_SOURCE.md)，终端合同和费用见 [线性化及成本](LOWERING_AND_COST.md)。不把本公式已有的熵／均值机制改名宣称为 PLDI 贡献。
