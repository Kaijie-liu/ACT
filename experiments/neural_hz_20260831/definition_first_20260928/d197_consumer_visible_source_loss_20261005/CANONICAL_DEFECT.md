# 稀疏源关联的守卫缺陷与本体边界

本文分析 D195 自由源关联接口的分数查询层。得到的共同误差合同不是新的原生非凸域，也不提供额外求解器或具体反例。其作用是准确定位混权消费者所需的信息。

## 定义与恒等式

源 X 为坐标独立盒，g_i(x)=a_i^T x+b_i。每个源列 j 的 incidence I_j 包含所有 a_ij 非零的门。共同完整相位质量 lambda 与局部矩 mu 满足 D195 的盒界、边缘和共享身份条件。对 lambda_s>0 定义

```text
x_j(s) = mu_(j,s restricted to I_j) / lambda^(I_j)_(s restricted to I_j).
qbar_i = sum_s lambda_s s_i g_i(x(s)).
d_i(s) = [-(2s_i-1)g_i(x(s))]_+,
e_i = sum_s lambda_s d_i(s) >= 0.
```

局部 incidence 包含全部实际依赖，保证所声明的 gated mean 确实等于 qbar；canonical x(s) 逐点属于 X，并保持源均值 xbar。零质量单元不除法。直接分 s_i=0 和 1，得到

```text
ReLU(g_i(x(s))) = s_i g_i(x(s)) + d_i(s),
q_true_mix = qbar + e.
```

这是同一批盒内源点在实际网络中的物理输出均值，实际标签可能不同于原 s。不能把它当作标签保持的状态修复或 ADV 见证。全部读出 Y=Cq+Bx 满足 Y_true_mix=Ybar+Ce，shared skip 精确保留。

## 查询方向的准确范围

对固定 c 与源 skip 系数 v，令 S_true 为 sup_x in X [c^T ReLU(g(x))+v^T x]。上式给出

```text
c^T qbar + v^T xbar <= S_true + sum_i max(0,-c_i) e_i.
```

只需支付负系数是上述比较方向的事实，不是说负权必然导致损失，更不是免费计算 S_true 的算法。

当 c>=0 时，自由关联核心在该读出上的 support 已精确：canonical 真实混合支配其值，而每个实际网络点都有 one-hot 扩展。当再保留普通逐门下界 qbar>=0、qbar>=g(xbar) 时，c<=0 的 support 也精确：qbar>=ReLU(g(xbar))，平均源 xbar 属于 X，故其读出不超过这个实际单点的读出。任何包含全部真实点的更小接口也继承相应结论。尚未闭合的是同一读出的混合正负系数。

这些证明要求 canonical 源逐点属于实际父域。对仅在均值上满足一般父谓词的状态，不能从自由盒结论推到完整网络。

## 不枚举完整单元的点态上界

按相同 incidence I_B 对源列分块。对门 i，只计 i 属于 I_B 的块。设 p_b=sum_(s_i=b)lambda_s，且 p_b>0 时

```text
m_(B|b) = sum_(t_i=b) mu_(B,t) / p_b,
gbar_(i|b) = b_i + sum_B a_(iB)^T m_(B|b),
v_(iB,t) = a_(iB)^T (mu_(B,t) - lambda^(I_B)_t m_(B|t_i)).
```

正部次可加性与边缘化给出

```text
e_i <= sum_b p_b [-(2b-1)gbar_(i|b)]_+
       + sum_(B,t) [-(2t_i-1)v_(iB,t)]_+.
```

证明：每个 g_i(x(s)) 分成对应条件均值与各块偏差。对负符号后的和取正部不超过逐项正部之和；对块项按局部 t 汇总，lambda^(I_B)_t 消去 canonical 分母，正好得到 v 项。零质量块的 mu 由盒界强制为零，整项按零定义。

保留 qbar_i>=0、qbar_i>=g_i(xbar) 时，p_1 gbar_(i|1)=qbar_i>=0，p_0 gbar_(i|0)=g_i(xbar)-qbar_i<=0，因此第一项为零。private 块 I_B={i} 的 v 恒零；剩余只支付共享 incidence 的条件失配。

预有 lambda、mu 后，主要系数乘积数为 sum_B 2^abs(I_B) nnz(A_:B)，标量块读出表至多 sum_B abs(I_B)2^abs(I_B)。四门 patch 的声明支持计法给出 240 个原始 mu 槽、16 个 lambda、104 个块读出、480 个系数乘积；这是依据 D195 结构的纸面计数，不是实际网络系数审核或测时。lambda 边缘化、完整 2^k 表、原源、父谓词、全部消费者和 decoder 均仍收费。点态评估便宜不代表可以便宜地优化它的最坏情况。

若可靠源界 l_i<0<u_i 且保留上述下界，旧 secant 直接给出

```text
e_i <= u_i (g_i(xbar)-l_i)/(u_i-l_i) - qbar_i
     <= -l_i u_i/(u_i-l_i).
```

稳定门的缺陷为零。点态 incidence 证书可与该界取 min，但目前不经新全局优化得到的统一常数仍是旧 triangle gap，尚无廉价新最坏情况改进。

## 合法混合也会有正缺陷

取 X=[-1,1]^2，g_1=x+y+1/10，g_2=x-1/10。两个等权真实点为

```text
A=(4/5,-7/10),   phase 11,   q=(1/5,7/10),
B=(-4/5,4/5),    phase 10,   q=(1/10,0).
```

源点均严格内部、guards 均严格。其 xbar=0、ybar=1/20，qbar=(3/20,7/20)。x 关联两个门，y 只关联门 1；canonical 因而保留 x=±4/5，却把两单元的 y 都换成 1/20。canonical g_1 分别为 19/20 与 -13/20，故 e_1=13/40>0，e_2=0。

原接口显然来自真实网络混合。正缺陷只说明这个 canonical 见证不合 guards，并不说明不存在别的共同真见证。要求 canonical e=0 会删除合法凸组合，不能冒充真实图凸包的必要 cut。它不会由此删除真实整数网络点；将其另作非凸限制是另一未解决问题。

## 混权损失与原生非凸性的区别

旧三门 circuit 已给出普通 sharp 控制：g=(x+y,x,y)，X=[-1,1]^2，lambda_100=lambda_000=1/2，canonical 源分别为 (1/2,1/2) 与 (-1/2,-1/2)。取由这些源和标签导出的局部矩，得到 qbar=(1/2,0,0)、源均值零。它满足普通逐门均值 guards 与 triangle，但 q_1-q_2-q_3=1/2；真实网络由 ReLU 次可加性必有该读出<=0。e=(0,1/4,1/4)，负系数费用恰好为 1/2。此例仅检查合同方向与 sharpness，旧 circuit 已能排除，不报新收益。

若 lambda 的边缘绑定原生 beta 属于 {0,1}^k，lambda 必集中单一相位。canonical 源合成一个实际 x，保留原 guards 后 e=0。以上分数假点和正缺陷因此属于查询接口层，不是原生域新增的非凸表达能力。若将 lambda 与原 bits 脱钩，必须另证身份与具体化，不能隐去这一变化。下一非线性联合闭包也未得到证明。
