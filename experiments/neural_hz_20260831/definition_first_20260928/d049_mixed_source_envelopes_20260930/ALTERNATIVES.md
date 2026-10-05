# 两个共同源方向的限制证明

本页保留本轮另外两项纸面探索。它们改变了候选选择，但不执行新程序、不重跑旧版本，也不否定精确 HZ 中已经保留的源信息。固定源展开、ReLU 三角关系和逐坐标 McCormick 均已有先例，不因换成相位符号就成为新域。

## 逐源坐标求界仍可被各门凸包吸收

设同一盒中 x_j in [l_j,u_j]，g_i=a_i*x+c_i，q_i=ReLU(g_i)，原 active bit 为 alpha_i。考虑带符号实际读出

```text
E=s*x+sum_i w_i q_i,
q_i=alpha_i*(a_i*x+c_i) in original integer semantics.
```

一个看似保留共同源的新规则，会把 E 拆成每个源坐标的 `s_j*x_j+sum_i w_i*a_ij*alpha_i*x_j`，逐坐标取一个对 binary alpha 有效的仿射上包络，再相加。这类行已被各门完整 source labelled hull 的交蕴含。

证明只引入辅助 t_ij=alpha_i*x_j 来描述各门完整析取提升；这些变量不被候选运行时创建。每个门的完整 hull 可扩展到标准 McCormick 行及相应门守卫。固定 j 后，共同 x_j 与所有 t_ij、alpha_i 的 star McCormick 多面体本身就是对应二元乘积图的凸包。

构造证明：令 theta=(x_j-l_j)/(u_j-l_j)。在 x_j=u_j 端点的概率为 theta，在 l_j 端点的概率为 1-theta；若分母非零，定义

```text
Pr(alpha_i=1 | x_j=u_j)=(t_ij-l_j*alpha_i)/((u_j-l_j)*theta),
Pr(alpha_i=1 | x_j=l_j)=(u_j*alpha_i-t_ij)/((u_j-l_j)*(1-theta)).
```

McCormick 行保证这些数在 [0,1]。在各端点条件下让不同 bits 独立，就同时得到所有所需均值。端点概率为零时忽略该条件分布；l_j=u_j 时结论直接成立。这是存在性证明，不执行概率采样或相位枚举。

因此任何对单个 star 的整数图有效的仿射行，在其 McCormick 系统中已经成立；逐 j 相加不会再强化全部独立 source hull。即便权重混合正负、保留了 source ID，这个结论仍成立。它不覆盖跨多个源坐标共同使用同一个相位分布的证书，也不表示 D049 的成对原门差式在完整各门 hull 中必然冗余。

完整 source labelled hull 是这里明确的数学比较对象，不等于生产 big M 已有全部这些行，也不要求下一候选胜过无法计算的全网 hull。该结果只否决“逐坐标绝对值包络就是新的联合源能力”的主张。

## 一个聚合伴随量不能支持一般后继混权

对 lambda_j>=0，令 e=g-sum lambda_j h_j。ReLU 次可加和正齐次给出

```text
ReLU(g)-sum lambda_j ReLU(h_j) <= ReLU(e).
```

若已认证 e<=U，使用 g 的原 bit alpha，还能得到 `ReLU(g)-sum lambda_j ReLU(h_j)<=U*alpha`，允许 U<0：alpha=0 时左侧非正，alpha=1 时左侧不超过 e。这是已有 D029 条件残差行的实例，不是新逻辑。原门零点的两个相位也满足相应非严格证明。

其闭包缺口更重要。只保存 `E=q-lambda^T p` 与 `S=lambda^T p`，只能恢复 q 和该单个 p 投影。若 p 有开集自由度，后继 `a*q-mu^T p` 能由 E、S 线性重构，当且仅当 mu 属于 span{lambda}：写 A*E+B*S=A*q+(B-A)lambda^T p，逐系数比较即得。原 bits 全部保留也不会自动补回缺少的实值方向。

普通严格内部反例取 x,y,z in [-1,1]：

```text
p1=ReLU(x+y/4), p2=ReLU(x/4+y),
q=ReLU(z+x/5-y/10+1/10),
P=(4/5,1/5,9/25), Q=(1/5,4/5,27/50).
```

两个点分别有 `(p1,p2,q)=(17/20,2/5,3/5)` 与 `(2/5,17/20,3/5)`；所有父 bits 都是 1。于是 S=p1+p2=5/4、E=q-S=-13/20 完全相同。但下一真实门

```text
ReLU(1/5+7*q/10-4*p1/5+3*p2/10)
```

分别为 3/50、111/200，其原 bit 也都为 1。该后继门在完整输入盒上并非稳定门：输入 (1,-1,-1) 给预激活 -2/5。

结论仅是这个有限摘要不能无损转移一般混权；原 HZ 的 x/y/z 和全部节点仍区分这两点。新候选须保留足够的真实 companion 方向或重新支付共同源关系生成，不能把两个聚合标量宣称为全网无损闭包。D049 也只主张健全闭包，不主张这种无损性。

两项证明均由独立代理和根代理纸面复核，没有候选导入、模型执行、求解器、GPU 或资格测试。日期 2026-09-30；redu-hz；commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。正式1870/2413与独立61/400不变；本页不是新的数值预注册或执行入口。
