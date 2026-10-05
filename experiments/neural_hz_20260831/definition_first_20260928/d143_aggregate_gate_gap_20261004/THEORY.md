# 整组相位缺口表述及其能力边界

本轮从域关系本身检验一种候选：将逐门 ReLU 值等式替换为同一源与全部原相位之间的一条正权零缺口关系。精确式成立，且可按源坐标收缩双线性表达；但通常的线性终端外包既有增益也有损失，并被已有完整输入相关单门凸包覆盖。它不进入实现，不作为新的 Neural-HZ 突破。

这份 D143 记录是纸面证明与独立复核，不是数值实验、真实网络收益或形式化机器证明。上一 Goal 回合仅确认方向，分类为 no progress；本回合新增了明确的等价式、强参照包含证明、同结构双向后继控制和成本否决依据。正式 1870/2413 与独立外部 61/400 均没有新增。

## 原源与原相位上的精确定义

令 z 是当前共同状态的一个真实连续读出，全部源谓词、共享身份和原输入 decoder 保留。当前仿射或 Conv 块为 g=A z+b，输出为 q，原 signed bits 仅记作 beta=(1+bit)/2。beta 的身份和整数性不变，零预激活处两原标签均保留。设 a_i>0。

在 q>=0、q>=g 上，每个缺口都非负：

```text
d_i = q_i-beta_i*g_i
    = beta_i*(q_i-g_i)+(1-beta_i)*q_i >= 0.
```

因此以下关系等价于全部原 ReLU 图及原相位标签：

```text
beta in {0,1}^m, q>=0, q>=g,
sum_i a_i*q_i = sum_i a_i*beta_i*g_i.
```

证明：正权总和为零迫使每个 d_i=0。beta_i=0 给 q_i=0、g_i<=0；beta_i=1 给 q_i=g_i>=0。反向显然。精确式甚至不需要额外 sign guards，它们已被推出；实现若使用外包，则不能假装这些推论仍免费成立。a_i=0 会漏掉该门，负权会允许抵消，都不在定理内。

将右端改写为

```text
k_ij = a_i*A_ij,
theta = A^T*diag(a)*beta,
a^T*q = theta^T*z + (a*b)^T*beta.
```

这是 m 个 beta_i*g_i 与 d 个 theta_j*z_j 之间的代数收缩。theta_j 是多个原 bits 的加权和，不是一个 binary。所有 q、全部原 bits、共享消费者和源约束仍在；它没有减少输出自由度，也没有证明终端求解变容易。

固定 g 时，min a^T*q subject to q>=0,q>=g 的对偶是 max lambda^T*g subject to 0<=lambda<=a。取 lambda=a*beta 就得到上述零 gap。这只是数学归属辨认，不运行或授权任何对偶优化、LP 状态救援或相位枚举。

## 可送入原线性终端的有损版本

为避免把 theta 当作 binary，明确建立一个外包，而不声称等价编译。设已有可靠源界 ell_j<=z_j<=u_j，使用仅由自由 bit 盒得到的范围

```text
L_j=sum_i min(k_ij,0), U_j=sum_i max(k_ij,0).
```

保留 theta 的等式，新增 p_j，并以标准四行包络处理 p_j=theta_j*z_j：

```text
p_j >= L_j*z_j + ell_j*theta_j - L_j*ell_j,
p_j >= U_j*z_j + u_j*theta_j   - U_j*u_j,
p_j <= U_j*z_j + ell_j*theta_j - U_j*ell_j,
p_j <= L_j*z_j + u_j*theta_j   - L_j*u_j.
```

以 a^T*q=sum_j p_j+(a*b)^T*beta 替换精确总缺口等式。同时保留 q>=0、q>=g、原 sign guards 及 q_i<=gUpper_i*beta_i 的 off-mask。全部下游 Conv、mixed readout、skip 和谓词消费同一 q；没有独立噪声副本。所有真实原状态取 p_j=theta_j*z_j 都满足它，故这是健全外包。原 bit 保持整数时，包络通常仍不精确。

它仍是带二元因子的 HZ 型多面体并，不是把整个状态换成凸域。可见状态也不必变凸：下面三门 bank 的真实端点 z+=(1,0)、z-=(-1,0)，q1 分别为 5/4、0。取端点权重 1/4、3/4，得到 z=(-1/2,0)、q1=5/16；此处 g1=-1/4，原 guard 要求 beta1=0，off-mask 要求 q1=0，故该混合点不在外包内。

普通 Affine/Conv/Add/Concat 可继续消费这个共同 q；下一 ReLU 若在整个当前外包上作精确原门变换，则包含性可组合。这不证明固定宽度再抽象或廉价查询。外包返回的整数 SAT 也不自动是网络反例，必须按原协议从 decoder 重构输入并验证原具体网络。

## 已有完整输入相关单门凸包覆盖这个外包

本节比较终端 LP，因此暂将同一 beta 的查询视图置于 [0,1]，不是修改域的整数语义。令 H 是所有单门完整 source-labelled 图凸包的交，再交共同原源约束及新外包保留的相同幅值界、guards、off-mask。若预激活界直接来自同一源盒，这些额外行已由对应单门凸包推出；若界用了更强共同源信息，则必须显式同给两侧。每个门的标准 perspective 见证 v_ij 满足

```text
ell_j*beta_i <= v_ij <= u_j*beta_i,
z_j-u_j*(1-beta_i) <= v_ij <= z_j-ell_j*(1-beta_i),
q_i = sum_j A_ij*v_ij + b_i*beta_i,
q_i>=0, q_i>=g_i.
```

这些见证由该门 source-box 真实图的任一凸组合直接得到 v_ij=mean(beta_i*z_j)。反向按 active 与 inactive 两部分归一化可恢复该 source-box 单门图的凸组合，端点 beta=0/1 使用零质量约定；不能据此声称恢复了另带一般共同谓词 P 的单门图凸包。这就是已知输入相关单门强参照，不是本轮发明。[Anderson 等，第 2.2 节式 5及第 2.3 节](https://arxiv.org/pdf/1811.08359)

对任意 H 中的点，取

```text
theta_j=sum_i k_ij*beta_i,
p_j=sum_i k_ij*v_ij.
```

逐门读出等式给出新聚合等式。下面证明这组 p 也满足全部 aggregate McCormick 行。

对固定 j，正 k 用 literal t_i=beta_i、r_i=v_ij；负 k 用 t_i=1-beta_i、r_i=z_j-v_ij，权重 c_i=abs(k_ij)。所有 r_i 都满足 t_i*z_j 的四行包络。令

```text
T=sum_i c_i*t_i=theta_j-L_j,
C=sum_i c_i=U_j-L_j,
R=sum_i c_i*r_i=p_j-L_j*z_j.
```

按非负 c_i 加总四行得到

```text
R>=ell_j*T,
R>=C*z_j-u_j*C+u_j*T,
R<=u_j*T,
R<=C*z_j-ell_j*C+ell_j*T.
```

代回 R、T、C 恰好是新四行。所以 H 的每个点都有新外包见证；完整单门 source hull 的交支配这个新外包。共同原谓词可以同时交入两侧，证明不变；甚至与同一保留边界上的下游约束共同相交后，包含方向仍保持。

范围限定：这里 L、U 必须是上述自由 bit 盒范围。若另外使用跨门 guards 推导的更紧 theta 界，不能不证明便把此定理延伸到它；单门凸包交的分数点未必满足那种额外信息。定理也不声称强参照可免费生成或必然更快。它否定的是把这个收缩包络的精度当作超越已有单门输入几何的新能力。

## 同一普通结构上的双向分离

两方向使用同一个 bank，不更换模型身份或按结果选择规则：

```text
x,y in [-1,1], a=(1,1,1),
g1=x-y+1/4,
g2=x-2*y+1/4,
g3=2*x-y+1/4.
```

三法向不平行、偏置非零、各门都跨零。精确预激活范围为 [-7/4,9/4]、[-11/4,13/4]、[-11/4,13/4]。新式有

```text
theta_x=beta1+beta2+2*beta3 in [0,4],
theta_y=-beta1-2*beta2-beta3 in [-4,0].
```

### 对旧四行 LP 的严格增益

在 x=y=3/4、beta=(1/2,1/2,1/2)，旧四行允许

```text
q=(9/8,7/8,13/8), sum(q)=29/8.
```

此时 theta=(2,-2)。新包络给 p_x<=2、p_y<=-1，从而 sum(q)<=11/8，排除该点。更强的是，新外包全局具有物理方向界：由 p_x<=theta_x、p_y<=theta_y-4*y+4 得

```text
sum(q)+4*y <= 4+beta1/4-3*beta2/4+5*beta3/4 <= 11/2.
```

在真实源 x=y=1 上 q=(1/4,0,5/4)，该界取等号。旧点则给 53/8>11/2，说明不只是固定标签切片上的差异。

为展示 mixed readout 与 live skip，改取同一旧源和 bits 下仍合法的 q=(9/8,0,13/8)，定义

```text
Jplus=q1-q2+q3+4*y,
r=ReLU(Jplus-45/8).
```

新外包因 q2>=0 推出 Jplus<=11/2，故后继预激活<=-1/8；旧点给 Jplus=23/4，后继预激活=1/8。正向成立，但已经被上一节的已知强单门参照覆盖，不能算新域独有能力。

### 旧能力也会穿过后继结构而丢失

仍用同一 bank，取内部源与原整数 bits：

```text
x=-1/10, y=1/10, beta=(1,0,0),
g=(1/20,-1/20,-1/20),
q=(5/4,0,0), theta=(1,-1), p_x=p_y=1/2.
```

每个 product 包络在此都允许 p in [-1,1]。聚合等式为 5/4=1/2+1/2+1/4；epigraph、原 sign guards、off-mask 和正常幅值上界全部成立。真实 q1 却只有 1/20，旧整数图排除该点。

即使消去旧 beta，旧单门 LP 也有 q1<=9*(x-y+2)/16；此源的界为 81/80<5/4，所以它也是物理投影上的严格分离。

定义另一个普通 mixed readout、raw skip 和下一门：

```text
Jminus=q1-q2/5+q3/20-9*x/16+9*y/16-21/16,
s=ReLU(Jminus).
```

旧父层 LP 用上面的 q1 界、q2>=0、q3<=13/4 推出

```text
Jminus <= 18/16+13/80-21/16 = -1/40.
```

新外包点给 Jminus=1/20，因此可在精确下一门上取 s=1/20、原后继 phase=1。双方可靠的粗后继范围 [-4,3] 可由原幅值界及源盒直接给出，无须另求解。

严格结论是父层对性质 Jminus<=0 的证明能力回退；若采用正常前向稳定识别，旧界使 s 恒为零，新界不能。不能偷换为“子门同用 [-4,3] 的四行 fractional LP 时旧系统也直接强制 s=0”：宽界子门松弛并没有该推论。正向 r 的稳定性结论也采用同一限定。

这些点和界是纸面有理数控制，没有执行 benchmark，不是有效 ADV，更不是正式结果的已测回归。它们足以否决无条件保旧或单调变强的声明。

## 完整终端费用不能按一个点积计

精确式可以重新展开为 m 个原 beta_i*g_i，回到普通门建模；也可以逐源展开为 nnz(A) 个 beta_i*z_j。这里没有证明所有精确编码都必然这么大，但尚未建立较小的原终端精确编码。原 beta 整数不能使 d 个连续乘连续包络自动精确。

公平计费保留物理 z、g、q，令 n=nnz(A)，k=nnz(b)，全部门 crossing。共享 g=A z+b 的 m 条等式、n+m 个系数及所有源谓词两边同计。排除共同 scalar bounds 后：

| 终端部分 | 旧四行形式 | masked aggregate 外包 |
| --- | --- | --- |
| 原 bits | m | m |
| 额外连续坐标 | 0 | 2d 个 theta 与 p |
| 门及新接口行 | 3m | 3m+5d+1 |
| 对应 nnz | 7m | 7m+n+2d+k+Nmc |

新侧的三门行是 q>=g、q<=gUpper*beta、g>=gLower*(1-beta)，合计 6m nnz；g<=gUpper*beta 已由前两行推出，不重复收费。theta 定义有 d 行、n+d nnz；聚合行有 m+d+k nnz；四行包络有 4d 行、Nmc<=12d nnz，零系数按实际删除。因此表中式子是精确分解，通用上界为 7m+n+14d+k，不把上界当实测值。

不物理存 theta 可以少 d 个坐标和 d 行，但包络中的 theta 展开会重复 A 的系数。没有免费删除依赖；q 的全部 mixed consumers、skip、decoder、可靠界、证书、host/device 共存也仍需计费。

已存真实 Conv 边界的源维数/输出维数分别为 large 65536/65536、medium 14400/8192、Tiny 46656/25088。在这些当前目标上，d<m 的期待本身不成立；按单空间位置改组，3x3 的输入通道槽为 576，而输出通道是 64 或128，并且还要维护重叠窗口的同一源。这些是已有形状证据，不是本轮模型执行。[完整边界与成本记录](../d137_phase_frontier_admission_20261003/REAL_FRONTIER.md)

将多个层合成原始像素源也不能直接维持固定 A：下一层含前层相位乘积，必须另付展开或因子化及 guards 成本。点值 theta 的转置 Conv 与点积可在 GPU 上执行，但这不证明集合查询、跨层传播或正式验证 GPU 加速。

## 文献边界与研究决定

项目 D114 已含 q_i-beta_i*g_i 恒等式；D132 的共同能量已含正权非负缺口聚合，不过使用 q_i*(q_i-g_i)；D123 已将该类聚合互补归为经典数学。本次未在所查档案中找到完全相同的 source-side contraction，不据此作新颖性判断。

[Mangasarian 的作者稿](https://ftp.cs.wisc.edu/pub/dmi/tech-reports/11-01.pdf)第 2 节 Lemma 2.1、式 2.5、Proposition 2.2 用非负互补 gap 的零和恢复绝对值关系。这支持其已知数学归属，并非逐字给出当前 Neural-HZ 公式。该文的交替 LP 算法没有被采用。

[Dey、Han 与 Wang](https://arxiv.org/pdf/2410.14163)研究双线性等式聚合后的凸化及强度边界。其结果不直接证明这里的包含定理；这里已给独立的 literal 求和证明。其基于求解结果选权重、分支求界等算法也未采用。

决定：保存精确重参数化和比较定理，拒绝把这个 McCormick 替换升级为实现候选。保留全部旧四行再添加它，只会转回辅助约束路线；删除旧 active upper 则已经有普通跨层退化控制。没有省下完整端到端成本或新增真实能力的证据，不启动这种 helper，不修改原验证器。

下一主线仍是普通 Conv/ReLU/residual 中的共同源、相位、幅值和全部 live consumers 的联合域定义。已有零 gap 能量、简单正权聚合及其独立 product 包络不再作为新主线重复启动。真正的新候选必须给出超出这个已知信息转移的跨层规则或可付再抽象，并同时解释旧解保全和全部终端费用；不能仅以“更少双线性符号”入场。

2026-10-04 Australia/Sydney；分支 redu-hz；HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。全程只有只读检查、文献阅读、纸面推导和新隔离记录，无候选执行、GPU、模型重跑、后台任务、commit 或 push。
