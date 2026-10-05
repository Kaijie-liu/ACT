# 联合负缺口关系的混权传递与精度边界

本轮补齐D082从归一化门组到实际混权接收端的一个可构造前向规则：固定容量补齐生成相位行，再用确定的非负证书消费它，保留负权重门的共同源贡献。一个普通残差控制保留了收紧；另一个仅将正权改为1、1、2的控制却退回逐门独立上界。因此，不能把D082的小例子推广为宽卷积必然增强。

直接文献核查也确认：舍入是经典MIR，包含原相位的跨层cuts及GPU处理已有神经网络先例。本交付是有明确局限的静态生成与传递算法，不是完成了新域定义。本文只含纸面推导、字面档案核查和原始文献阅读，没有候选代码或数值执行。

## 非凸语义与实际读出

沿用原HZ的同一个连续赋值、全部原binary bits、EQ/LE、guards、共享frame和decoder。所有新增观察都由原整数图推出；全部原零点相位保留。可以把元素写成原H及带证据观察O，但这一observational接口是已有框架，不作为新颖性。

同一可靠源盒D上的实际接收读出为

```text
F = v(z) + sum_j lambda_j*q_j - sum_l delta_l*p_l,
q_j=ReLU(h_j(z)), p_l=ReLU(g_l(z)), lambda_j,delta_l>0.
```

先按原物理身份合并重复项，再按固定系数正负分组。v、h、g必须绑定同一原frontier，不复制残差分支的源。该规则可用于有可靠源形式的Affine、Conv及Add读出，但不声称这些前提已在真实模型上全部取得。

下面先证明确切系数版本。容量满足 `0<=q_j<=U_j*b_j` 和 `0<=q_j-h_j<=A_j*(1-b_j)`，其中b_j是原active bit的认证0/1视图，A_j、U_j为可靠非负常数。主要情形U_j>0；U_j=0的输出确为0，可不进入本读出算术，但原门、谓词及bits不删除。没有正项时直接用 `F<=v-ReLU(sum delta_l*g_l)`；不是求解状态触发的替代路径。

## 不求公分母的加权整数关系

对正项统一取

```text
tau = max_j lambda_j*(A_j+U_j) > 0,
a_j = 1-lambda_j*U_j/tau,
Hplus = sum_j lambda_j*h_j,
N = sum_j lambda_j*(q_j-h_j)/tau,
C = sum_j a_j, m = sum_j b_j,
T = N + sum_j a_j*b_j.
```

由于 `a_j>=lambda_j*A_j/tau>=0`，逐门负缺口容量推出T<=C；由正输出容量又得 `T<=m-Hplus/tau`。在同一源上先合并Hplus，再取可靠 `c<=Hplus/tau`。令 `k=floor(C+c), rho=C+c-k`，D082的整数论证给

```text
sum_j lambda_j*q_j <= Hplus + R + sum_j kappa_j*b_j,       (1)
R = tau*(C-rho*(k+1)),
kappa_j = tau*rho-tau+lambda_j*U_j.
```

这不是将rational权重乘巨大公分母来构造整数计数；m仍是原bits之和。代价是补齐负容量，可能变弱。tau按同一数学式确定，不扫描多个缩放后按效果挑选；不依赖模型身份、LP点、margin或公开标签。

对每个整数原状态，容量与共同源下界同时成立，所以(1)有效；在激活零点q=h=0时，不论b取何合法值仍成立。加入(1)不改变原整数具体化或输入重构，只可能加强查询外包。原HZ安装同一行得到相同逻辑强度。

## 不用对偶求解的混权消费证书

直接把(1)的bits独立取极值可能丢掉输出容量。固定以下代数消费规则：kappa_j<0时用 `b_j>=q_j/U_j`，kappa_j>=0时用b_j<=1。于是

```text
sum_j nu_j*q_j <= Hplus+K,                               (2)
nu_j = lambda_j+max(-kappa_j,0)/U_j,
K = R+sum_j max(kappa_j,0).
```

保留原(1)，(2)只是额外有效观察；不是删除或连续化bits。由于nu_j>=lambda_j>0，定义

```text
eta = min_j lambda_j/nu_j,
r_j = lambda_j-eta*nu_j >= 0.
```

取已有可靠点态仿射上式 `S_j(z)>=q_j`，完整支付它的计算与证据。乘(2)以eta，并对剩余的非负r_j使用S_j，即得

```text
sum_j lambda_j*q_j <= eta*Hplus+eta*K+sum_j r_j*S_j.
```

负权重侧不独立取每个p的界，而保留共同源聚合 `Hminus=sum_l delta_l*g_l`。由正齐次和次可加性，`sum delta_l*p_l>=ReLU(Hminus)`。因此

```text
B = v+eta*Hplus+eta*K+sum_j r_j*S_j,
F <= B-ReLU(Hminus),
U_F = sup_{z in D} [B(z)-ReLU(Hminus(z))].                 (3)
```

这个支持不是免费oracle：使用D068/D069已有的全盒排序填充构造，或在可手算控制中给出完整解析支持。它是该盒函数的精确支持，通常只是原网络F的上界。没有解相位子域、枚举输入或调用LP；源盒的最大点也不是具体ADV。

固定的eta和r给出了D078所要求的一种可执行非负证书，不需要从LP对偶中找乘子。它只消费指定结构行，不等于对全部已有关系的最优组合。对-F执行相同规则可得下界；不把上下方向拆成多个候选路径取正式成绩。

## 后继变换的范围

对下一原门 `t=ReLU(F+bias)`，仍保留原child bit及完整门谓词。由(3)得到 `t<=max(0,U_F+bias)`；对下界同理。此步健全但可能损失联合关系，不能称为任意深度精确闭包。

若保留的是共同相位仿射上式 `F<=a0+sum a_i*b_i`，也可以先认证该上式的整体范围，再对其ReLU使用secant取得仍含原bits的仿射上式；这同样是已有的健全近似，而非新定理。只共享bits不能保证全部消费者有共同连续见证，D078反例仍有效。

Affine、Conv、Add、Concat按同一赋值形成真实读出与相应证书。仅保存每个消费者的标量界，不能据此声称保留了共同幅值。原HZ及全部联合观察必须继续共享同一frame；未证可消去的源与谓词不丢弃。

## 普通混权残差正控

沿用D082的x,y in [-1,1]及三个原门：

```text
h1=4x/5+y/5+1/10,
h2=-3x/5+2y/5+1/10,
h3=-x/5-4y/5+1/10,
qj=ReLU(hj),
Z=q1+q2+q3+11y/60.
```

lambda_j=1时，tau=2，R=11/10，全部kappa=-1/10；因此nu_j=12/11、eta=11/12、r_j=0。对v=11y/60，(3)恢复常数B=77/60，无需重新优化系数。

增加普通原门 `p=ReLU(g), g=2x/5+y/5+1/10`，保留它的原bit，接混权残差

```text
F=Z+g/2-p/2.
```

g跨零、含两输入和非零偏置，负权重为-1/2；不是极小误差特例。统一公式给B=77/60+g/2、Hminus=g/2，故

```text
F <= 77/60+g/2-ReLU(g/2) <= 77/60.
```

在(1,1)处g=7/10>0、Z=77/60，取等。旧三门比较点x=y=0、q=(13/25,37/100,2/5)、b=(1/2,1/2,1/2)可接第四门真实值g=p=1/10、bit=1，仍有F=129/100。这个点通过的是D082列明的三门比较系统再接第四门的精确图；没有证明额外添加全部涉及第四门的pair hull后仍通过。

再接原child `t=ReLU(F-1)`：新界17/60，小于性质阈值57/200；旧点29/100高于该阈值。真实源(0,0)使child inactive，(19/20,19/20)使其active。正控表明统一传递可以消费原关系，不是新benchmark CERT或ADV。

## 普通权重变化的负控

仍用同三门和可靠单门界[-9/10,11/10]，改实际读出为 `F=q1+q2+2q3`，没有负项，v=0。不改原source或bits，也不人为制造数值边界。

```text
lambda=(1,1,2), tau=4,
Hplus=2/5-x/5-y,
c=-1/5, C=19/10, k=1, rho=7/10,
R=2, kappa=(-1/10,-1/10,1).
```

D082旧比较点的weighted输出为169/100，(1)右侧为14/5，不再将其排除。这里只说明这个点的分离消失，不证明(1)在所有点上完全冗余。

进一步，固定物理消费恰给

```text
K=3, nu=(12/11,12/11,2), eta=11/12, r=(0,0,1/6),
S3=11/20-11x/100-11y/25,
B=77/24-121x/600-99y/100,
sup_D B=22/5.
```

22/5恰是逐门独立上界 `11/10+11/10+2*(11/10)`。同一个健全规则在普通权重下没有得到额外物理界收益。因此，“能写出通用公式”不等于“对真实宽混权层统一有效”，也不证明该规则支配D061、D062、D069或完整native已有信息。

## 区间参数与完整成本

选定的容量、tau、c、floor和rho都必须使用可靠且位长受控的算术。对实际区间h_j，Hplus在共同source上先合并区间系数再认证c；(1)左侧和h_j仍为实际原物理量，不能替换为中点网络。相位不从参考模型重新产生。

若接收权重为区间，可用固定参考权重选择证书，并以 `sum |w-w_ref|*M_q` 等可靠误差包住实际读出。若B、Hminus为区间仿射形式，则选择确切参考形式并认证误差E_B、E_H，利用ReLU的1-Lipschitz性质将全盒支持扩大E_B+E_H。误差、参数来源、浮点语义和原端口绑定尚未实现认证，不能把这一数学说明当loader资格。

设原fan-in为p，输入源系数出现总数S，共同规范化源支持为d。容量max、C、floor、kappa、nu、eta及r需要O(p)标量工作；共同Hplus/Hminus、secants和B需实际O(S+d)归并工作。已有全盒hinge支持再付O(d log d)比较及O(d)算术；排序、字长、中间乘积、缓存、全部原source与证据均另计。没有给每个接收端免费共享不同排序的假设。

可保留一条phase行(1)，原q/h/b坐标下至多3p_positive个nnz；展开共同source按实际支持计。物理上/下界、child行、原读出等式、latent展开、slacks、RHS、decoder和所有旧谓词另计。没有新bit，不等于没有新增行或求解成本。多层观察及支持增长没有常数空间保证。

GPU可能承载固定归约、稀疏行及批量支持，但可靠排序、floor、512位超限、设备与主机共存及终端查询未测量。本文没有解除原16GiB AS或其他资源门，没有GPU执行和性能声明。

## 直接先行工作改变了新颖性判断

[Marchand–Wolsey原CORE作者稿](https://webdoc.sub.gwdg.de/ebook/serien/e/CORE/dp9839.pdf)，1998，§2 Proposition1：令s=C-T>=0、K=C+c=k+rho，原约束给-m<=-K+s。对rho>0，该命题直接得到s>=rho*(k+1-m)，即D082舍入式；rho=0退回容量行。本文已读取这个命题及§5，后者按fractional LP点选择聚合/缩放，与我们固定前向规则不同，不移植该separation流程。

[GCP-CROWN](https://arxiv.org/pdf/2208.05740)，NeurIPS2022，§1、§3.1式15–16及§5，已支持涉及pre-activation、post-activation、原phase变量的通用cuts，明确包括MIR，并结合GPU bound propagation。故“相位加MIR”或“GPU处理这类行”不是我们的新颖性。其对偶式传播与整体验证器的BaB不在本项目授权范围。未发现D082完全同式的NN推导不能作为原创证明。

研究决定是保留本静态生成器及明确的前向消费证书作为支撑，暂不为它启动一套新组件测试或命名新域。只有证明新的关系表示/组合机制及普通真实结构上的完整价值，才继续将它纳入定义级候选；同事实HZ必须是比较对象。论文级目标不以已知cut的重新组织冒充完成。

## 记录与执行状态

上一Goal轮是progress：来源组件的3817项完成，D082证明与边界已存档。本轮重新校验D082全部6项SHA通过；分支redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。既有九个tracked修改未动。

本轮配置paper_only、primary_source_review、independent_hand_proof_review、read_only_literal_archive_inspection。根代理手工推导并复核公式与分数，独立审查核对健全性；没有AST/导入/编译/候选数值程序、模型解析、LP、GPU、测试或回放。本文不是执行预注册，不消耗或重跑任何既有版本。

新资料只写本隔离目录；旧档、冻结源、历史模型、生产和远端不改。正式1870/2413（1063 CERT +807 validated ADV）、独立CIFAR100 25和TinyImageNet36即61/400不变，formal_gain=0。定义创新、native/GPU、真实网络增益与保旧回放仍未完成，Goal保持active。

write-page技能用于将文献已知性、项目推导、正负控制与执行状态分开记录；只本地保存及读回，不发布外部Page。SHA256SUMS保护本研究记录和依赖，不授予实现资格。
