# 真实拓扑错位与不采用的替代路线

本轮先复核D243与三个固定真实前缀的匹配，再据此转向共同源mixed容量。以下静态阴性不是实际模型guard运行，也不是所有Neural-HZ路线不可行。

## 真实来源的语法系数确实为零

D241 source_0的Relu5预激活port129只被节点5输入0消费；source_1和source_2的port123同样只有这一消费者。它们不是图输出。large下一目标子预激活为135，medium/Tiny为131；完整仿射路径经父ReLU输出130/124及较早ReLU的skip进入。

D243的_expand在ReLU输出边界停止，_alias只追零偏置、单项系数1的affine，不追Conv。即使目标BN是identity，alias最多退至Conv3输出128/122，而该输出仅被BN消费。其他稳定ReLU别名不会创造所选crossing父的列。因此忠实逐节点声明这三份拓扑时，原固定X提取给a=b=0。

一般的“f仅被ReLU消费”本身不够：若f=identity(t)，而t另有skip，literal alias仍可命中f的代表列。真实三图的上述Conv边界排除了这个例外。不能省掉alias限定，也不能将该结论说成父f在数学上没有共享源关联。

于是D243当前r=z=(g1+g2)/2，guard要求整个共同均值的无条件盒界包含于[-tau,tau]。它失去了原live-input控制的相位异号缩界优势，但尚不能断言三个真实模型所有pair都失败；D241没有完整中间bounds。

对一个固定child pair，令Mz=max(abs(Lz),abs(Uz))，对所有eligible父定义delta_i=s_i*(coef(g1,Q_i)-coef(g2,Q_i))。任何有序父对的tau=(delta_i-delta_j)/4，因此若Mz>(max(delta)-min(delta))/4，该child pair下全部父对都不能通过当前X判据。反向不成立，也不是当前已有实际通过率；求这些量仍需完整来源与界，不能免费假定。

## a 和 b 为零时的局部投影冗余

此时c12=alpha1*x2、c21=alpha2*x1不进入defect读出。若它们没有其他已注册消费者，仅被其8条MC和3条X容量约束，则对旧H的LP物理投影没有加强。

证明：任意单门分数点(x,q,alpha)满足完整原四行时，可分解为质量alpha的active点x+=q/alpha及质量1-alpha的inactive点x-=(x-q)/(1-alpha)；零质量项略去。取两个单门分布的独立乘积，便能补出c12/c21，满足MC与全部三容量。其均值仍是原x/q/alpha。

不要求这些分布原子满足旧H的其余共同源谓词：这里仅构造辅助行的存在见证，旧H仍作用在未变的物理点。若c另有源观察一致性或消费者，上述自由延拓不再适用。整数投影保持与这里的LP冗余是两个不同结论，不能混称。

## 不能原样启动的构造路径

D243逐有效Conv连接的bounds计算至少30 work：_conv_terms的12、_bound_terms的2、两次mul各3、两次add各5；至少6累计逻辑entries。medium一层128到128、8乘8、3乘3、pad1已有7,929,856个有效连接，仅这层237,895,680 work就超过单公共操作200M。large/tiny相应路径更大。这是当前源码计费下界，不是一次实际OOM、物理峰值或普遍算法下界。

直接GPU逐tap双向FMA也不自动过门。三个完整前缀共有199,412,352个有效tap，两FMA即398,824,704个标量操作，超过256M共同账；不能按一次kernel launch收费。D241总账246,412,868是其特定审计实现的成本，不是未来实现固定税，但也不能免费复用其完整解析工作或每模型清零。

纸面固定8乘8 tile范围证书可能减少算术：全部输出仍保留，按padding mask聚合正负kernel及共享源envelope，估计通道级双端点dot为6,072,320次FMA。它不包括聚合、扫描、BN、完整H、终端等费用，不是完整资格。共享范围也可损失guard：真实父尺度(1,1)、z=.75(f1+f2)、d=Q1-Q2时X判据通过；将第一尺度可靠放宽到3，得到a=2.25,b=.75,tau=2，判据失败。因此本轮不把tile构造当能力进展，不创建此候选。

原ImplicitConv2DOp仅有普通浮点CPU matvec；生产hybridz_tf.py在ASSERT仍遇lazy affine前驱时记录lazy_affine_reached_terminal并丢弃该HZ。它不是可直接继承的完整可靠隐式/GPU终端。旧D017/D043 CUDA初始化失败也没有证明根因，更不能把未测设备占用算成零。本轮未再次运行环境诊断。

## 删除硬守卫并不健全

记D=Y1-Y2。e=0且支付为0、原相位不同时，原两defect行成立当且仅当abs(z)<=tau。例alpha=(1,0)、q=(1,0)、tau=1、z=2给D=2、K=3、p1=0，下行要求1<=0。不能直接放行guard失败实例。

一种有证替代是取完整[L,U]上的非负仿射secants ell(z)>=(-z-tau)_+、u(z)>=(z-tau)_+，加入

```text
D-K <= 2*tau*p2+alpha1*ell(z)+alpha2*u(z)+2*max(Ue,0),
-D+K <= 2*tau*p1+alpha1*u(z)+alpha2*ell(z)-2*min(Le,0).
```

完整同源alpha_i*z若已由旧产品表达，可以线性展开这些行，不需新位、split或solver。但是解除拒绝不等于取得精度收益。

本轮证明一个限定阴性：当L<-tau<tau<U、alpha_i*z只有独立共享-z MC且没有源容量/其他观察一致性时，该整个双侧因子不强于D239的自由clip因子，因而不能超过凸化父域上的联合两child真图hull。令Fz=z+tau+ell(z)、Gz=z+tau-u(z)，最小secant有Fz(L)=0、Fz(U)=U+tau、Gz(L)=L+tau、Gz(U)=2*tau。消元上侧为

```text
D <= 2*tau*p2+2*max(Ue,0)
     +min((U+tau)*alpha1,Fz(z))
     -max((L+tau)*alpha2,Gz(z)-2*tau*(1-alpha2)).
```

若s_minus/s_plus为clip标量凸包上下界，端点比较给Fz>=s_plus、Gz<=s_minus。展开min减max的四项，各自不小于D239自由clip上侧四项的对应值；交换下标证明下侧。任意MC可行产品给出的允许区间宽度非负，完整线性投影是连续区间，因此这不是只证明两个必要端点。更高的仿射tail外包只会更弱。

自由版本可投影为0新列、8LE、至多38nnz；D239已有同样0列8LE且更强的36nnz合同，不为它另建runner。此阴性不适用于受共同源容量耦合的alpha*z，不能推广为所有tail规则无用。

## 本轮取舍

不原样重跑已知超账的Fraction Conv；不把GPU批处理当减少标量人口；不为自由tail补偿加一个helper。新的主候选见THEORY.md：在共同源上同时保留x和q的相位条件容量，并用完整物理消费者和强参照检验它。Gram提取与旧精度的简单运输仅是支撑；mixed合同的严格分离才是本轮继续实施的理由。
