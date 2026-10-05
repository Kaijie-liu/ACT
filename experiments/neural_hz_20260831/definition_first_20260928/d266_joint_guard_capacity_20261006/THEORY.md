# 联合守卫误差与残余的跨层容量定理

本页在 D263 的同一原非凸 H 上给出严格更强的容量规则。关键不是再添加条件乘积，而是避免对同一个残余同时收取相位守卫最坏值与幅值最坏值。定理不创建新因子、不做输入或相位搜索；它仍只是域演算的候选组件。

## 共同原图与记号

令 xi∈[-1,1]、qi=ReLU(xi)，alpha_i 是原相位的 0/1 记法，不新建或替换 signed binary。设

```text
k=a1*x1+a2*x2+b1*q1+b2*q2,
z=k+r, d=tau*(q1-q2)+e, tau>0,
Y1=ReLU(z+d), Y2=ReLU(z-d),
mu=(Lr+Ur)/2, rho=r-mu, c=tau+mu,
h=alpha1-alpha2, S(z)=clip(z+tau,0,2*tau),
J >= sup_H (|rho|+2*max(e,0)), B_e=2*max(Ue,0).
```

全部读出指向同一个 H；r/e 不变为独立残余变量。J 可固定取两个同源可靠上界的 min。保留 D263 的所有系数和原 r 范围。

## 守卫与残余的点态联合界

定义 E=h*rho+h*(S(z)-z-tau)。父相位为 10 时，k>=Lk10=min(0,a1+b1)-max(0,a2)，并有

```text
E <= max(rho,A10), A10=-Lk10-c.
```

证明：z<=-tau 时 E=-k-c<=A10；-tau<=z<=tau 时 E=rho；z>=tau 时 E=tau-k-mu<=rho。父相位为 01 时，k<=Uk01=-min(0,a1)+max(0,a2+b2)，相同恒等式给

```text
E <= max(-rho,A01), A01=Uk01+mu-tau.
```

同相时 E=0。这里只是证明对原有限相位的恒等分情况，运行算法不分裂 H。零点的两种合法原标签均适用。因此统一有

```text
E+2*max(e,0) <= G=max(J,A10+B_e,A01+B_e).
```

若仍用旧相位独立守卫界，也有 E+2*max(e,0)<=J+eta10minus+eta01plus，其中

```text
eta10minus=max(-Lr-Lk10-tau,0),
eta01plus=max(Ur+Uk01-tau,0).
Gstar=min(J+eta10minus+eta01plus,G).
```

故 Gstar 永不弱于旧共同残余付款。不能用 min 包装一个未经证实的上界；这里两个分支均已在同一原整数图上证明。

## 零辅助列物理行

沿 D263 的 deficit 恒等式，记

```text
A=(a1+a2)/2, Jphase=(a1-a2)/2,
Bq=(b1+b2)/2, Hc=(b1-b2)/2,
C1=c/2+max(-A,0)+max(-Jphase,0),
C2=-c/2+max(-A,0),
C3=-c/2+max(A,0)+max(-Jphase,0)+2*tau,
C4=c/2+max(A,0), M=max(C1,C2,C3,C4),
s0=M-max(Jphase,0), Dabs=2*q1-x1+2*q2-x2.
```

在旧证明使用独立 eta 之前，保留 E 而不是分别界定 h*rho 和 clip tail。代入上一节统一界，得到

```text
s0*Dabs+Y1-Y2-c*(x1-x2)/2
    <= 2*M+abs(Bq)+max(Hc,0)+Gstar.                 (JG)
```

具体地，p_i=alpha_i-qi、n_i=1-alpha_i-qi+xi 都非负，且 Dabs=2-sum(p_i+n_i)。D263 的代数左端上界保持，只将原 eta10minus+eta01plus+h*rho+2*e_plus 替换为 E+2*e_plus；其余为 abs(Bq)+Hc_plus+sum(C_i*deficit_i)。用 C_i<=M 即得 JG。所有旧因子、等式、来源、消费者、输入 decoder 完全不变。

不宣称 JG 胜过完整共同来源 RLT 或所有已有方法。它严格强于指定有限旧系统的证据见 CONTROL.md。新颖性和新的域定义资格必须另外判断。

## 宽残差族的旧冗余证书不再适用

对 D265 的普通族 a>=0、0<=b<=1、kappa>=3/2，r=kappa*(u-v)、e=kappa*(u+v)/2，原共同来源给 R=4*kappa/3、J=2*kappa、B_e=2*kappa。新付款恰为

```text
Gstar=2*kappa+max(a+b-1,0),
new_rhs=2*a+3+b+2*kappa+max(a+b-1,0).
```

旧两个 unit 三角及已有 Y1 上界的支持是 Sold=4*a+b+5+2*kappa，因此 new_rhs-Sold=max(a+b-1,0)-2*a-2<0。旧的那张冗余证书不能再否决新行，但仅凭此不构成对完整旧系统的严格分离证明。CONTROL.md 另给一个满足全部指定旧六产品、QG、旧 JP、普通单门图松弛和更紧同源子界的宽残差显示点，以及穿过后继 ReLU 的开参数族；不把 not_excluded 本身记为能力。

## 组合范围

Affine、Conv、Add、Concat 保留原共享来源键和既有物理谓词；所有消费者共用同一行而非各复制一个自由误差。对固定非负权重组合这些行可得到多个消费者的联合有效上界，但这本身是已有线性谓词组合，不是新的精确闭包定理。负权重必须有对应反向有效行。

后继 ReLU 仍使用其原图、原二元相位及健全前向上界。CONTROL.md 的固定 F 读出提供一个跨过下一门的严格消费正例；不从此推出任意深度精确闭包。新相位 gamma 下的 gamma*e 不能由旧相位的有限一阶观察自动恢复，已有 D029、D035、D233、D251 反例仍有效。

标量 Gstar 只加常量工作，物理行不新增非零位置；然而原源提取、所有谓词展开和完整终端成本仍需支付。D265 的真实来源预算失败、native 消费缺口和 GPU 未实现状态均未由本定理解除。
