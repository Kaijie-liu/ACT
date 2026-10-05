# 混合相位容量的定量守卫误差

本研究把已有 mixed 关系的硬适用条件推广成一条始终相同的有限公式：对两个不同相位事件分别计算 clip 超出线性段的最大误差，将其作为已有相位系数支付。旧守卫成立时误差严格为零，逐行恢复旧规则；不成立时仍有健全关系，而不是直接删除守卫或调用另一算法。普通 residual 构造上的强物理分离见 CONTROL.md。

这是一项已独立纸面复核的项目规则推广，不是完整新域、新集合表达力、机器证明或已执行候选。尤其不能把一般 Lipschitz、乘积提升或 RLT 本身称为新发明。

## 原非凸语义及完整来源

原状态 H 保留连续因子、全部原 signed binary 因子、EQ/LE、shared latent/frame、完整消费者与输入 decoder。alpha_i 是原位的 0/1 视图，不是被替换成连续的原生相位。仅在分析查询松弛时允许其取分数值。

沿用已认证的两父归一化 x_i in [-1,1]、q_i=ReLU(x_i)，以及

~~~text
p_i = alpha_i-q_i >= 0,
n_i = 1-alpha_i-q_i+x_i >= 0,
Y1 = ReLU(z+d), Y2 = ReLU(z-d),
z = a1*x1+a2*x2+b1*q1+b2*q2+r,
d = tau*(q1-q2)+e, tau>0.
~~~

r、e 是同一个 H 中的完整读出；偏置、skip、其他父输出及误差均不可省略。可靠界 Lr<=r<=Ur、Le<=e<=Ue 须覆盖整个 H。两个父的来源系数仍按同一固定提取规则获得；tau 仍由原 child 差分系数确定，不按输出 margin、求解状态或失败结果重新选择。

六个已有连续观察及完整 MC 保留：c12=alpha1*x2、c21=alpha2*x1、d12=alpha1*q2、d21=alpha2*q1、v1=alpha1*r、v2=alpha2*r。定义

~~~text
K = tau*(alpha1-alpha2)
  + a1*(q1-c21)+a2*(c12-q2)
  + b1*(q1-d21)+b2*(d12-q2)+v1-v2.
~~~

原位整数时 K=(alpha1-alpha2)*(z+tau)。三条已有 X 容量也完整保留：设 T=q1-q2+c12-c21、U=q1+q2-c12-c21、Dabs=2*q1-x1+2*q2-x2，则 T<=p2+n2、-T<=p1+n1、U<=Dabs。

## 按结构计算的四个误差

分别保留两个不同相位事件的源界，而不只保存它们的并集：

~~~text
L_10 = Lr+min(0,a1+b1)-max(0,a2),
U_10 = Ur+max(0,a1+b1)-min(0,a2),
L_01 = Lr-max(0,a1)+min(0,a2+b2),
U_01 = Ur-min(0,a1)+max(0,a2+b2).

eta_s_minus = max(-L_s-tau,0),
eta_s_plus  = max(U_s-tau,0),   s in {10,01}.
~~~

这些只是四个系数和完整 r 界的静态代数公式，不建立相位子问题或输入分区。在事件 10，q1=x1、q2=0、x1>=0、x2<=0，给出前两式；事件 01 对称。使用弱符号端点，因此保留原零点的全部合法标签。

统一登记如下两行，替代新候选自身的两条 hard-guard defect：

~~~text
 Y1-Y2-K <= 2*tau*p2+2*max(Ue,0)
           +eta_10_minus*alpha1+eta_01_plus*alpha2,
-Y1+Y2+K <= 2*tau*p1-2*min(Le,0)
           +eta_10_plus*alpha1+eta_01_minus*alpha2.       (QG)
~~~

无论 eta 是否为零，都执行同一计算和安装合同；不是 hard guard 失败后选择 soft 路径。未证来源、无可靠界、秩/数值/资源不合格仍 fail closed。旧冻结版本不被改写。

## 整父域健全性

令 Phi(d,z)=ReLU(z+d)-ReLU(z-d)、h=alpha1-alpha2、S_tau(z)=clip(z+tau,0,2*tau)。原整数位给 Phi(tau*h,z)=h*S_tau(z)。精确恒等式为

~~~text
S_tau(z)-z-tau = ReLU(-z-tau)-ReLU(z-tau).
~~~

在事件 10，h=1，因此 h*(S_tau-z-tau) 属于 [-eta_10_plus,eta_10_minus]；在事件 01，h=-1，故它属于 [-eta_01_minus,eta_01_plus]；相位相同则该差为零。

由于所有 eta 非负，用事件指示 1_10<=alpha1、1_01<=alpha2 支付上下界仍健全，得到 QG 中四项相位支付。这里没有新增交集位，也没有将同相位的两个单项分别错误线性化。

Phi 对 d 单调且为 2-Lipschitz。由 d-tau*h=tau*(p2-p1)+e，得

~~~text
-2*tau*p1+2*min(Le,0)
 <= Phi(d,z)-Phi(tau*h,z)
 <= 2*tau*p2+2*max(Ue,0).
~~~

与前述精确参考误差相加即证 QG。证明不使用求解器状态，也不要求图节点的 preactivation 恰好避开零。

任意原整数 H 点取六个真实产品便给唯一辅助扩展，所有 MC、源容量及 QG 均成立；反向投影保留全部原 H 谓词，故整数投影仍恰为 H。decoder 不变。具体化按共同完整接口上的集合包含比较；空关系嵌入原 HZ，不声称存在廉价最佳抽象或完整格。

## 旧规则保留和算子范围

原 mixed guard 即两个事件的 [L_s,U_s] 均含于 [-tau,tau]。此时四个 eta 全零，QG 的精确有理行与旧 mixed defect 相同，六产品、源容量及消费者也未变。因此这是旧已合格数学关系的保守扩展，而不是以更弱的条件 clip 替换它。

这个逐行性质不等于新实现已经通过旧测试或保住全部 benchmark：改变安装人口会影响时间、存储和求解成本；全部晋级门仍须执行。

Affine/Conv、同 frame 的 Add/Concat 线性消费同一组关系。下一 ReLU 继续保留原非凸图，并须在它的完整父状态上重新验证合同；没有任意深度的精确查询闭包或固定宽度结论。QG 的 eta 是可靠条件区间给出的常数支付，尚不是未解决的多消费者联合残余向量域。

## 完整成本与先例

相对同一 mixed 关系，新增变量、原位、LE/EQ 均为零。原两条 defect 已经含 alpha1、alpha2，修改其系数不扩大符号支持上界。独立 x/q/Y/r 坐标的通用上界仍为六连续观察、29 LE、96 nnz；另显式写产品界时为41 LE、108 nnz。若旧系数恰好抵消为零，新系数可能恢复非零，所以不声称每个退化实例的实测 nnz 都逐项不增。

四个 eta 的可靠计算、数值位长、来源/界认证及行舍入都收费。r 若内联 s 项，原关系支持上界仍为92+4s，其他多项读出另加真实展开费用。原 H、Gram、decoder、全部候选对、证据、host/device 共存、普通终端降低和查询均不可省略。D249 构造样例的104 native nnz不自动转成此新合同的实测结果。

对于此前因 guard 失败而未安装的关系，现在安装六产品有实际成本。因此“没有额外列”只比较同一关系合同，不是整个网络零成本，也不能声称已通过四并发不回退门。

[Sharp HZ 第 IV 节](https://arxiv.org/html/2503.17483v2#S4)已研究 RLT 保留 HZ 原整数集合而改变松弛，故本项目不将这一事实或产品 lift 重记为创新。本轮有限新增是有证条件超额支付、旧关系的逐行恢复及 guard 外物理分离。没有证明超过完整跨层 RLT、完整四门原源 hull、所有调参后的 hard-mixed 规则或新颖性已达到论文标准。

