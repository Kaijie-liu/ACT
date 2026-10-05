# 盒源纤维的精确投影与强旧对照

本轮回答上一轮留下的问题：在两源 pooling 控制中，共享源纤维确实比四行聚合保留更多信息，但这部分信息可以完全解释为已有 source-subset 关系的扩展表示。它不是已证的新抽象域能力。这个消元结论不适用于任意耦合源多面体或任意共享多输出读出。

## 完整存在性投影

一组门共享 z 属于 [0,1]^k，且

```text
g_i=b_i+sum_j a_ij*z_j, a_ij>=0,
theta_j=sum_i a_ij*beta_i, M_j=sum_i a_ij,
zeta=sum_i b_i*beta_i,
R=zeta+sum_j W_(j,j).
```

每一 W_j 向量使用[原共享源定义](../d184_shared_source_factor_20261005/DEFINITION.md)的盒 perspective 和补环境。全部原 bits、guards、父关系、消费者、原源与 decoder 保留。这里的关键局限是每个 W_j 仅有对角坐标被本组 R 使用，没有其他读出或谓词消费非对角坐标。

记 w_j=W_(j,j)。每个坐标恰落在区间

```text
max(0,M_j*z_j-M_j+theta_j) <= w_j <= min(theta_j,M_j*z_j).
```

因此完整存在 W 的投影恰为

```text
zeta+sum_j max(0,M_j*z_j-M_j+theta_j)
    <= R <=
zeta+sum_j min(theta_j,M_j*z_j).
```

必要性由区间求和。充分性来自区间和仍为区间，其内任何 R-zeta 都可由各个 w_j 同时取得；未消费的非对角坐标统一取 W_(j,l)=theta_j*z_l，即满足其各自盒行。这只是证明中的延拓，不是引入执行时搜索。

等价线性投影为每个源坐标子集 S 的两族行：

```text
R >= zeta+sum_(j in S)(M_j*z_j-M_j+theta_j),
R <= zeta+sum_(j in S)theta_j+sum_(j not in S)M_j*z_j.
```

上下各至多2^k行，未声称每一行都是 facet。这是有限数学描述，不建议运行时枚举全部子集。

## 为什么这是已知类型的源关系

设所有门 crossing，b_i<0<U_i=b_i+sum_j a_ij。下界的空子集由 R>=0 蕴含，全子集由 R>=sum_i g_i 蕴含。两端上界就是已有四行聚合的两个上界。

令 kappa_i=b_i+sum_(j in S)a_ij。任一中间上界可写成逐门有效关系的和：

```text
q_i <= sum_(j not in S)a_ij*z_j+kappa_i*beta_i.
```

beta_i=0 用右侧非负，beta_i=1 用 z_j<=1；这属于[已有静态源证书](../d052_static_source_certificates_20260930/THEORY.md)的一般形式。若 kappa_i 允许为负，证明仍成立，不能将其擅自截断为零。

每个中间下界还被一个预先按系数确定的更强旧关系支配：

```text
R >= sum_(i:kappa_i>=0) g_i.
```

证明逐门进行。对应投影下界的门项是
sum_(j in S)a_ij*(z_j-1)+kappa_i*beta_i。
kappa_i<0 时该项<=0；kappa_i>=0 时该项<=g_i，因为差额为
sum_(j not in S)a_ij*z_j+kappa_i*(1-beta_i)>=0。
所选门集只由固定系数决定，不是输入或相位分裂，也不读 LP 状态。

对原两源开放族 b_i<-a_(i,y)，每组至多需要四个旧聚合行、两个中间 upper 和一个中间 lower。非对角 W 没有任何 visible 精度贡献；去掉这些量只是等价表示简化，不能记成定义能力突破。

## 一个完整但不足晋级的严格分离

源 x,y 属于 [0,1]²，plus 组为

```text
g1=2x+y-11/10,
g2=x/2+y-11/10;
```

minus 组为

```text
g3=3x/4+y-23/20,
g4=3x/2+y-21/20.
```

完整消费者仍是 Q=R_plus+R_minus、B=R_plus-R_minus 和 live x。四门非平行、有偏置，均 crossing。在 x=1/5,y=9/10，实际 g=(1/5,-1/10,-1/10,3/20)，原整数 bits 为1001，无零点。取 R_plus=49/100、R_minus=3/20，即 Q=16/25、B=17/50，满足[上一轮便宜对照](../d185_definition_strength_audit_20261005/RESEARCH_RECORD.md)的四聚合行和 R_plus<=(5/2)x。

共享源域却有

```text
R_plus <= (5/2)x-(beta1+beta2)/10,
R_plus <= (19/10)(beta1+beta2).
```

第一行加第二行的1/19倍，得到 R_plus<=19x/8，对整个该抽象父域成立。于是后继

```text
h=ReLU((Q+B)/2-19x/8-1/100)
```

恒零；便宜对照中的上述合法有损成员给 h=1/200。这不是 ADV，也不是实际网络运行结果。

然而同一开放结构有更便宜的统一旧关系。令 delta_i=-(a_(i,y)+b_i)>0，则 g_i<=a_(i,x)x-delta_i，U_i=a_(i,x)-delta_i>0，故 q_i<=U_i*x：inactive 时用 x>=0；active 时用 x<=1。plus 组因此直接给 R_plus<=23x/10，比19x/8更强，只需要一条同样大小的源关系。

所以这是真实的域精度分离，但不是新的项目能力分离。不能据此重启大规模实现，也不能把较弱对照的失败当成 Neural-HZ 创新成功。

## 推导的作用域

本消元针对盒源、正系数分组以及独占的对角消费者。非盒源的坐标耦合、多个权重之间的联合可行关系、同一 W 被多个输出同时消费，都不在独立区间求和的等价范围内。下一定义需要研究这些共同关系，而不是重复添加本节已被解释的标量 cuts。

全部结论为精确实数纸面证明与独立复核，没有候选执行、机器证明、真实性能或成绩资格。原档案保持只读。
