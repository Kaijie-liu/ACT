# 共同源条件下的联合神经幅值容量

本轮给出一个完整局部神经图的凸包参照：共享源参与混合时，如何让所有门的符号和幅值同时来自一个共同见证。它比自由源与相位乘积核心多包含了联合神经守卫，并可精确描述固定条件下的混权输出纤维。但它仍保留完整局部相位表，属于已知 perspective 与区间投影原理的特化，不是已完成的新 Neural-HZ。

## 语义对象与作用域

考虑 k 个原 ReLU，原输入接口为

```text
z in Z=[ell,u],   p_i in [L_i,U_i],
g_i=p_i+a_i^T z+b_i,   q_i=ReLU(g_i).
```

Z 是有界凸盒；各 p_i 在本接口内独立，且与 z 构成乘积。beta_i 为原相位的 0/1 记号：active 时 g_i>=0、q_i=g_i；inactive 时 g_i<=0、q_i=0。所有零点的双标签保留。

目标集合 G 保存 z,p,q 和全部原 beta。下面表述的是 conv(G) 的参照接口；原生状态中的 beta 始终为整数，不把 Neural-HZ 替换成凸域。原连续因子、历史 EQ/LE、共享身份、所有消费者和输入 decoder 在整体域中都必须保留。

p_i 可以是私有像素块的标量仿射像，但“独立”只针对上述投影。完整私有像素均值、外部谓词和其他非线性消费者不在本定理中；只把 p_i 重新连接到原像素均值，不能升级为完整原输入图的理想凸包。[D194 的原像反例](../d194_sparse_crossing_source_fiber_20261005/CONDITIONAL_FIBER.md)仍然有效。

## 一份共同相位分布和条件源

令 s 遍历 {0,1}^k，N=2^k。引入

```text
lambda_s>=0,  sum_s lambda_s=1,
beta_i=sum_(s_i=1) lambda_s,
ell*lambda_s <= z_s <= u*lambda_s,  sum_s z_s=z.
```

所有门共同使用同一个 z_s。记

```text
H_i,s=a_i^T z_s+b_i*lambda_s,
G_i=p_i+sum_s H_i,s=p_i+a_i^T z+b_i,
n_i=q_i-G_i.
```

n_i 是实际负幅值 ReLU(-g_i) 的均值读出，可以内联，不要求新变量。

在每个 active 单元，正幅值的可分配区间是

```text
[max(0,L_i*lambda_s+H_i,s), U_i*lambda_s+H_i,s].
```

在每个 inactive 单元，负幅值的可分配区间是

```text
[max(0,-U_i*lambda_s-H_i,s), -L_i*lambda_s-H_i,s].
```

必须逐单元要求相应上端非负。这就是同一条件源上的联合守卫；不能只要求各上端的总和非负。L_i<=U_i 保证上端非负时单元区间非空。

分别把 active 的下、上端相加得到 Qlo_i、Qhi_i，把 inactive 的下、上端相加得到 Nlo_i、Nhi_i。容量条件为

```text
Qlo_i <= q_i <= Qhi_i,
Nlo_i <= q_i-G_i <= Nhi_i.                         (C)
```

## 精确局部联合凸包定理

存在上述共同 lambda、z_s 并满足全部单元守卫及 (C)，当且仅当 (z,p,q,beta) 属于 conv(G)。这不是逐门独立凸包的交。

必要性：把任意真实图的有限凸组合按完整原标签分组，得到 lambda_s、z_s。每组的 private 源和正负幅值满足各自带质量区间；汇总即得 (C)。所有行都作用于同一组源见证。

充分性：固定任意满足条件的 lambda、z_s、p、q。有限区间的 Minkowski 和仍是端点和区间，因此可在 active 单元内分配 q_i,s，使其总和为 q_i；在 inactive 单元内分配 n_i,s，使其总和为 q_i-G_i。然后定义

```text
active:   p_i,s=q_i,s-H_i,s,
inactive: p_i,s=-n_i,s-H_i,s.
```

区间条件保证 L_i*lambda_s<=p_i,s<=U_i*lambda_s、对应原守卫正确，且 sum_s p_i,s=p_i。各 private 块独立，所以不同 i 的分配可以同时组合。对每个 lambda_s>0，用同一个 z_s/lambda_s 和全部 p_i,s/lambda_s 构造真实完整局部神经点；按 lambda_s 混合就恢复全部物理坐标及相位。零质量单元的 z_s、H 和幅值全为零，不参与除法。弱守卫保留所有合法零标签。

若 beta 为原生整数，边缘约束迫使 lambda 在原标签上 one-hot；条件源回到实际 z，(C) 即原 ReLU 图。因此在原父域上加回该扩展不会删除合法原点。这个精确扩展本身仍属于 HZ 可表达的已知析取提升，不构成新的集合表达力。

## 可共同消费的条件输出纤维

固定 (lambda,z_s,p) 后，定义

```text
lq_i=max(Qlo_i, G_i+Nlo_i),
uq_i=min(Qhi_i, G_i+Nhi_i).
```

若每个区间非空，整个 q 向量的条件可行集恰为 product_i[lq_i,uq_i]。充分性证明同时允许各 q_i 独立选择，仍有一个共同局部神经混合见证。

因此，对于实际混权读出和活旁路

```text
y=c^T q+r^T p+t^T z+d0,
```

其条件支持精确为

```text
r^T p+t^T z+d0 + sum_i max(c_i*lq_i,c_i*uq_i).
```

给定 H 后，容量和单方向条件支持只需 O(kN) 算术。这不是对 lambda、z_s、p 的全局优化，也不是一个已经实现的 GPU 验证器。多输出必须共用同一个 q，通过 Y=Cq+skip 表达；不能把各方向的最大值拼成可行输出点。

## 与旧结果及下一激活的关系

[D194](../d194_sparse_crossing_source_fiber_20261005/CONDITIONAL_FIBER.md)证明固定实际 z 时的私有幅值条件凸包。若记其非凸联合对象为 R，则本轮描述 conv(G)=conv(R)，并不是在集合上严格收紧 R；推进的是共同混合与联合守卫的完整参照描述。它可用于审计具体的廉价线性化，但不能把“完整凸包”一词当作新域成绩。

[D175](../d175_joint_phase_physical_projection_20261004/THEORY.md)已经有超过全部两门接口的物理输出分离，并有含源旁路的后继 ReLU 恒零控制。完整局部图凸包当然保留对应有效物理行；这只是继承旧证据，不是本轮新控制或新收益。

对一个标量后继 h，若已能精确求父图上 sup h，则 sup ReLU(h)=max(0,sup h)，线性支持在父图及其凸包上相同。但本轮没有免费提供该全局支持查询。更重要的是，ReLU(E h) 一般不等于 E ReLU(h)：不能把条件均值的 ReLU 当成下一原相位的精确条件幅值。完整子图、多个后继的相关性及有损递归闭包仍未解决。

多个门的共同条件源和容量是下一定义的有用语义参照；将完整相位表直接物化不是本轮选择的实现方向。原 bits 不删、不 split、不以 helper 求解全表。既有精确观察闭包下界仅限制其声明的线性商类别，不是否决所有 Neural-HZ 创新。

## 已知数学来源

[Anderson 等 §2.2](https://arxiv.org/html/1811.08359v2)用带相位质量的输入副本给出单 ReLU 理想 extended formulation，并说明辅助输入增长的成本；本轮使用同类条件源思路及区间总量投影，不采用动态分离或分支算法。[Sharp Hybrid Zonotopes §IV](https://arxiv.org/html/2503.17483v2)给出产品提升仍可表达为 HZ 的结果。两者均不能被改名为本项目新颖性；这里的具体多门容量推导作为自包含、范围明确的项目支撑定理保存。
