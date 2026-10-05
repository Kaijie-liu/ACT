# 共同残余块加强跨层相位容量

本研究保留同一 HZ 中均值残余 r 和差分残余 e 的共同来源，改变此前把它们分别求界的容量规则。得到三项纸面结果：一个零新辅助变量的健全跨层物理行；一个按共同源系数同号及异号重叠精确计算付款改进的定理；一个严格超出此前完整六产品 MC 加 X 加 QG 系统的普通开区间控制。

这次改变了信息强度，不只是把旧行消元。不过，它仍是 Neural-HZ 域演算的候选组件，不是已完成的新抽象域、全网能力或 GPU 结果。没有执行新候选或数值实验。

## 原非凸语义与共同残余块

沿用 D262 的完整 H：保留全部连续因子、原 signed binary、EQ/LE、shared latent/frame、全部消费者及原输入 decoder。x_i in [-1,1]、q_i=ReLU(x_i) 使用原父相位 alpha_i。两个实际 child 为

```text
Y1=ReLU(z+d), Y2=ReLU(z-d),
z=a1*x1+a2*x2+b1*q1+b2*q2+r,
d=tau*(q1-q2)+e, tau>0,
Lr<=r<=Ur, Le<=e<=Ue,
mu=(Lr+Ur)/2, R=(Ur-Lr)/2, rho=r-mu.
```

新增研究对象不是两个边际区间，而是共同残余块：同一原源键集合上的二行仿射读出 (rho,e)，其完整常数、系数、源界、原相位依赖及方向证书共同保存。它不创建一份独立 r/e 赋值，不替换原 H，也不删除任何已有变量。源可以是有原谓词约束的内部物理前沿；盒仅用于给支持上界。

空关系仍嵌入原 HZ。全部新增行在原整数 H 上有效，故具体化及 decoder 不变；松弛可以更紧。普通 HZ 加上相同行是同信息参照。因此这个块接口本身并不证明新的可表示集合类或最佳抽象。

## 联合付款与固定生成规则

对同一 H 上四个仿射读出分别取得可靠上界 U：

```text
Jraw=max(U(rho),U(-rho),U(rho+2*e),U(-rho+2*e)),
Jsep=R+2*max(Ue,0),
J=min(Jsep,Jraw).                                      (J)
```

必须先合并共同源系数再求 U，不能把 U(rho+2e) 偷换成 U(rho)+2U(e) 后宣称保存了相关性。允许有效的分组上界，但不把它称为整个 H 的精确支持。两种上界都在同一整数 H 上有效，固定 min 保证不会因某个分组证书较松而弱于旧独立付款。这是一条统一代数规则，不依赖实例身份或求解状态。

逐点有

```text
|rho|+2*max(e,0)=max(rho,-rho,rho+2*e,-rho+2*e)<=J.
```

最后一步同时使用 Jraw 和 Jsep 的有效性。取最大值不枚举输入或原相位，不调用支持优化器；可由共同源盒的固定仿射求和生成。不能引入只满足 v>=四个平面的自由变量，再把 v 当作它们的有证上界。

## 加强的物理容量定理

保留 D262 的所有 a/b/tau、r 界与守卫误差，不重新选择它们：

```text
A=(a1+a2)/2, Jphase=(a1-a2)/2,
B=(b1+b2)/2, Hc=(b1-b2)/2, c=tau+mu,
Dabs=2*q1-x1+2*q2-x2,
eta10minus=max(-Lr-min(0,a1+b1)+max(0,a2)-tau,0),
eta01plus=max(Ur-min(0,a1)+max(0,a2+b2)-tau,0),
C1=c/2+A_minus+Jphase_minus,
C2=-c/2+A_minus,
C3=-c/2+A_plus+Jphase_minus+2*tau,
C4=c/2+A_plus,
M=max(C1,C2,C3,C4), s0=M-Jphase_plus,
Qjoint=abs(B)+Hc_plus+eta10minus+eta01plus+J.
```

其中 t_plus=max(t,0)、t_minus=max(-t,0)。仍有 M>=tau>0，s0 可为负。直接生成

```text
s0*Dabs+Y1-Y2-c*(x1-x2)/2 <= 2*M+Qjoint.              (JP)
```

证明必须回到整数图，而不能把旧 QG 的常数付款直接减小。记 h=alpha1-alpha2、p_i=alpha_i-q_i、n_i=1-alpha_i-q_i+x_i。Phi(d,z)=ReLU(z+d)-ReLU(z-d) 对 d 单调且2-Lipschitz，且

```text
d-tau*h=tau*(p2-p1)+e.
```

因此它的差额逐点不超过2*tau*p2+2*max(e,0)。参考值 Phi(tau*h,z) 与 h*(z+tau) 的上侧守卫误差仍不超过 eta10minus*alpha1+eta01plus*alpha2。

沿 D262 对 T/U/V/W 的容量组合，但暂不把 h*r 和 e 分开求界，得到

```text
Y1-Y2-c*(x1-x2)/2-Jphase_plus*Dabs
 <= abs(B)+Hc_plus+eta10minus+eta01plus
    +h*rho+2*max(e,0)+C1*p1+C2*n1+C3*p2+C4*n2.
```

原整数 h 属于 {-1,0,1}，所以 h*rho<=|rho|。代入 (J)，再加恒等式 M*Dabs=2*M-M*(p1+n1+p2+n2)，即得到 (JP)，余项为四个非正的 -(M-Ci) 乘 deficit。原零点两个合法相位都包含在证明中。

整个证明不需要在候选中创建产品变量，原 H 全保留。JP 的六个物理系数和 D262 相同，只降低有证右端。故在相同已认证参数下，它蕴含旧 D262 单行；不意味着它替代完整旧六产品系统的所有信息。CONTROL.md 证明它也不被那个旧系统普遍蕴含。

## 同一源盒上的精确付款改进

先考虑按同一源身份合并并吸收源半径后的确切仿射式

```text
rho=Avec*xi, e=e0+Bvec*xi, xi in [-1,1]^m.
Ra=sum_j |Avec_j|, E=sum_j |Bvec_j|,
Lsame=sum_{Avec_j*Bvec_j>0} min(|Avec_j|,2*|Bvec_j|),
Lopp =sum_{Avec_j*Bvec_j<0} min(|Avec_j|,2*|Bvec_j|),
mcredit=min(Lsame,Lopp).
```

这里 mu 是 r 的该源盒中心，不能把任意旧中心悄悄当作它。共同源盒上四方向的精确最大付款为

```text
Jbox=Ra+2*max(0,e0+E-mcredit).
```

证明用两个逐系数恒等式：

```text
||Avec+2*Bvec||1=Ra+2*E-2*Lopp,
||-Avec+2*Bvec||1=Ra+2*E-2*Lsame.
```

四支持分别是 Ra、Ra、2e0加上述两个范数，取最大即得。因此相对于同一盒的独立付款 Ra+2*max(e0+E,0)，精确改进为

```text
2*min(max(e0+E,0),mcredit).                           (G)
```

两类同号和异号的非零共同源重叠都存在、且 e 的盒上界为正时，该付款严格减少。这个结构条件可以通过固定系数扫描观察，不读取 LP 或 property。付款减少本身不保证整个旧松弛严格收紧；CONTROL.md 另给完整旧可行点及新行证书。

若 rho=delta+Avec*xi，则不能套居中简式。正确四方向最大值是

```text
max(Ra+abs(delta),
    delta+2*e0+Ra+2*E-2*Lopp,
   -delta+2*e0+Ra+2*E-2*Lsame).
```

Ra 是该源盒系数半径，不是任意先前区间的 R。原中心、常数、系数不确定性、同一 BN 误差都必须被保留。实际有舍入或分组外包时回到可靠的 (J)，不能从未经认证的数值相消扣掉信用。

## 仿射组合与成本范围

共同残余块的 Affine/Conv/Add/Concat 组合沿原源身份线性变换二行系数，偏置同样组合；重用 skip 不能复制成独立源。新 ReLU 仍保留原非凸图和原相位，再对新结构重新认证残余块。此规则没有任意深度固定宽度或有限模板精确闭包保证。

共同支持已合并时，(G) 所需四个和可一次 O(m) 扫描形成，适合固定批量归约；无需构造四套 m 维向量。原支持合并、父归一化、r/e 提取、共享原 H、所有实例位置、数值外包、设备拷贝、证据及终端消费均另计。零新辅助列、1 LE、最多6物理系数只是最终行的局部成本，不是完整 native 成本。

四方向最大恒等式及 L1 相消并非新数学原理，项目 D029/D062 已有固定共同源多平面支持。有限新增是把联合残余信息接入跨层相位容量、给出结构性付款差公式，并严格超过指定旧六产品系统。[Sharp HZ 第IV节](https://arxiv.org/html/2503.17483v2#S4) 已有一般 RLT 强化；本页没有证明超过完整共同源 RLT 或全部既有抽象变换，也没有证明 PLDI 级新颖性已经成立。
