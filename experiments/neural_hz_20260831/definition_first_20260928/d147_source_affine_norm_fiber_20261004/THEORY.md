# 保留源仿射部分的共同范数候选

本轮确认了一个定义缺陷，并推导出原生修正。只保留相位相关中心及一个各向同性误差球，即使半径精确、原相位保持整数，也会丢失源与幅值的联系。不能靠调小半径解决这一缺陷。修正是在域元素中保留源仿射部分，只将激活的非线性余项放入共享范数块。

下面给出明确具体化、原 HZ 嵌入、可重复的健全变换与原线性终端的外包合同。这仍是研究候选：修正恢复了本轮反例中旧 Gram 关系已有的能力，没有证明新颖性、全面保旧、完整成本优势或真实网络新增解。没有实现或执行候选。

上一 Goal 回合 D146 分类为 progress：其强比较否定了实施条件并形成冻结记录。本轮进一步改变下一研究动作：不再仅优化 phase-only 球半径，转向保源仿射及真正有用的相位耦合结构。

## 相位中心球为什么不够

D146 的形式 y=c+G beta+r、相位仿射半径 R(beta) 可以由 L1 换成 L2。令新实际算子为 g=Wy+d、q=ReLU(g)，固定参考 tau，保留的 skip 为 Sy，则同一个真实见证满足

```text
gbar = W(c+G tau)+d
delta = (diag(gamma)[Wr+WG(beta-tau)], Sr)
||delta||2 <= kappa R(beta) + sum_j d_j |beta_j-tau_j|
kappa >= ||[W;S]||2, d_j >= ||(WG)_:j||2.
```

半径仍对 beta 仿射，因为 beta、tau 为二元。保留旧状态、共同 skip 与原 guards/masks，放松新激活的精确 delta 映射，是健全外包。但次数不升高并不意味着保留了需要的关系。

取 d=m=64，H 为 Sylvester Hadamard 矩阵，A=H/8，因此 A^T A=AA^T=I。原源、原门和消费者为

```text
x in [-1,1]^64
b_i=1/8+i/1024, i=0,...,63
g=Ax+b, q=ReLU(g), original phases beta
s=x_1/10
w_i=1 for i<56, w_i=-1/4 for i>=56
J=w^T q+s.
```

该精确正交矩阵是解析控制，不是声称真实 CNN 满足正交结构，也不作为针对特殊模型的运行分支。各行非平行、偏置非零。均匀规则的候选中心为 diag(beta)b；由 ||x||2<=8，可取联合球

```text
||(q-diag(beta)b, s)||2 <= 81/10.
```

证书为 ||[A;(1/10)e_1^T]||2^2=101/100<(81/80)^2。

在严格内部源 x=0，取原整数 beta=1、q=1+b、s=0。此时 g=b>0，所有原 sign guards、q>=g、q>=0、q<=(8+b)beta 均成立，误差范数为 8。只要按候选合同删除了 active-value 上侧行，完整非线性球语义本身就允许这个假幅值，不是终端 LP 才出现的伪点。数值为

```text
sum_i b_i = 319/32
sum_(i<56) b_i = 2177/256
w^T b = 8333/1024
J = 63629/1024 > 50.
```

而旧 [D009 共同源 Gram 行](../d009_bounded_phase_energy_20260928/D009_DOMAIN_AND_RESULTS.md)已经给出

```text
2 sum(q)-sum(g) <= sum(b)+64
sum(g)=8*x_1+sum(b)
J <= sum(q)+s <= 32+319/32+(41/10)x_1 <= 7371/160 < 50.
```

于是下一原门 ReLU(J-50) 在旧有证关系下恒为零，新球却允许值 12429/1024。完整旧 HZ 当然也排除假幅值；没有发生具体 ADV。

这不只是 8.1 太松。同一全激活相位内，真实源 x=sign(A_i) 分别产生残差 8e_i，且其他门仍因 b_j>0 严格激活。因此同一中心下，任何仅依相位的健全球半径都至少为 8，仍接纳全一残差。也可用严格内部源 x=(7/8)sign(A_i)，得到 7e_i；对应全坐标 7/8 的假残差已使 J>50。结论只针对这种相位中心各向同性球，不否定一般关系域、各向异性表示或源相关约束。

## 超出两门关系仍不足以晋级

同一 bank 的分数点 x=0、beta_i=1/2、q_i=3/2+b_i/2、s=0，属于每一对原门的完整 source-labelled hull。对任意 i!=j，取

```text
u=(3/8)(sign(A_i)+sign(A_j)).
```

u 的分量为 0 或 +/-3/4，且 A_i u=A_j u=3。真实源 +u、-u 的各半混合恰好给出相同的源、skip、两门幅值与原 bits 的均值。不同 pair 使用各自见证，这是 pair-hull 交的准确比较含义。

假点的共同误差范数为 12，超过 8.1，新球排除它。用 ||(w,1)||2<=8 可得新 J<=93829/1280<80，而假 J=174221/2048>80。因此存在穿过后继 ReLU(J-80) 的解析分离。

但 D009 同样给出 J<=7371/160，明显更强。这个全 pair-hull 正控不能冒充超出现有全组能量关系的新增能力。所有这些计算是纸面有理数推导与独立审查，没有执行矩阵生成、数值搜索、模型或求解器。

## 新的候选元素与原 HZ 嵌入

为恢复源联系，考虑共享载体

```text
y = c + C xi + G beta + sum_k E_k e_k
xi in [-1,1]^p, beta in {0,1}^b
||e_k||2 <= R_k(beta)=a0_k+a_k^T beta, R_k(beta)>=0
P(xi,beta,e_1,...,e_K).
```

P 保留原 EQ/LE、源与输入 decoder、所有原相位 guards、必要输出 masks、已存在的物理读出和额外用户谓词。每个 e_k 有唯一身份；两个消费者使用同一个块，不能重新抽取噪声。具体化取满足全部条件的同一赋值并读出 y。语义偏序是具体化包含，未证明可计算最佳抽象或完整格。

原 HZ y=c_H+G_c xi+G_b sigma 通过 beta=(sigma+1)/2、c=c_H-G_b*1、C=G_c、G=2G_b、K=0 精确嵌入，P 及 decoder 同步作该双射。原 bits 不删除、不 pivot、不连续化。范数块与显式二元选择共存，不把整体替换成凸 ellipsotope、CZ 或 zonotope。一般曲面切片也不能由有限 HZ 线性行精确表达；这一表达差异本身不构成新颖性。

该候选也不只是旧完整门图附加 norm helper：下述新激活的 active-value 等式被替换，而不是与球同时保留。代价是有损性，不能预先声称保住所有旧 CERT。

## 保源仿射的 ReLU 变换

对 g=Wy+d，选固定结构参考 tau 属于 {0,1}^b，令 gbar=W(c+G tau)+d。参考不必可达，但不能当实际界。由 ReLU(t)=(t+|t|)/2，真实输出满足

```text
q = (1/2)g + (1/2)|gbar| + e_new
e_new = (1/2)(|g|-|gbar|)
||e_new||2 <= (1/2)||g-gbar||2.
```

假设已可靠认证

```text
K0 >= sup_(xi in [-1,1]^p) ||WC xi||2
Kk >= ||WE_k||2
d_j >= ||(WG)_:j||2.
```

则整个当前域的真实新算子像都有共同见证，满足

```text
R_new(beta)=(1/2)[K0 + sum_k Kk R_k(beta)
                          + sum_j d_j |beta_j-tau_j|].
```

半径依旧对原 beta 仿射；其常数为 (K0+sum Kk*a0_k+sum tau_j*d_j)/2，旧 bit 系数为 (sum Kk*a_kj+(1-2tau_j)d_j)/2。它在当前允许状态上非负，不要求每个系数非负。

更新输出系数为

```text
c' = ReLU(gbar)-(1/2)WG tau
C' = (1/2)WC, G'=(1/2)WG, E'_k=(1/2)WE_k
E'_new=I.
```

增加原新门 gamma，保留它对同一 g 的 guards、q>=0、q>=g 和 q<=U_positive*gamma。用新球取代 e_new 的精确绝对值映射，即是健全外包。它保留每个真实零点的两原标签，但不声称所有外包零纤维都精确。新增 ADV 必须从原 decoder 取输入并经具体原网络验证。

Affine/Conv 只左乘这些共同系数，精确消费当前域。Add/Concat 先按同一块身份合并或堆叠系数，保留共享输入和所有谓词。对同时保留的 Sy，新误差块在 skip 部分的系数为零；旧 e_k 在新输出和 skip 中仍是同一个坐标。这不需要把 skip 再复制进一个更大的新误差球。实际卷积布局、padding 和别名认证尚未实现。

该推导适用于整个当前外包状态，不仅原网络真轨迹，所以可递归组合。在 64 维控制中 gbar=b、K0=8，新半径为 4；假 q=1+b 需要 e_new=1，范数 8，立即被排除。同时

```text
sum(q) <= (1/2)sum(g)+(1/2)sum(b)+sqrt(64)*4
         = 4*x_1+sum(b)+32,
```

恢复 D009 的界，而不是超越它。证明依靠真实 Gram/正交证书；仅凭各行单位范数不能推出 K0=8。

## 平滑算子的同一合同和范围

若每个 phi_i 在包含实际 g_i 与参考 gbar_i 的整个区间上，有证斜率范围 [l_i,u_i]，令 Lambda=diag((l_i+u_i)/2)、L=diag((u_i-l_i)/2)。则

```text
phi(g)=Lambda*g + phi(gbar)-Lambda*gbar + e_new
||e_new||2 <= ||L(g-gbar)||2.
```

用 LWC、LWE_k、LWG 的相应范数证书构造 R_new，不再额外乘 1/2。均值定理或积分证明每个去掉中点斜率的函数为 L_i-Lipschitz；ReLU 的全局割线范围 [0,1] 给出上一节。可靠参考函数值若有误差，其共同范数上界必须再进入半径。这里没有认证任何 sigmoid/tanh/GELU 实现。

固定 1/2 会无谓放松稳定 ReLU。若整个实际与参考区间已有同一常斜率证书，取 l_i=u_i=0 或 1 可精确变换该行；若参考跨过零，不能冒用稳定证书。这是结构条件，不是按实例或 solver 状态择路。

对 softmax，令 Q=I-11^T/n。已知 Jacobian 为 diag(p)-pp^T，满足 0<=J<=Q/2。积分得到

```text
p(g)=p(gbar)+(1/4)Q(g-gbar)+e
1^T e=0, ||e||2 <= (1/4)||Q(g-gbar)||2.
```

1/4 是减去固定中点线性部分后的残差常数，不是 softmax 自身 Lipschitz 常数。这只是已有 [D108 曲率关系](../d108_centered_attention_relations_20261002/CONTROL.md)与 [D113 sector](../d113_ordered_probability_flow_20261002/CONTRACT.md)的完成平方/积分后果，不是新的 Softmax 定理。该例说明候选载体可容纳共同零和余项；不包含动态 QK、pV、LayerNorm 或完整 Transformer 资格。

## 原终端的有证外包与完整费用

对固定的实际消费者方向 v，先将同一块的全部引用合并，再取可靠 alpha_k>=||E_k^T v||2，可发布

```text
|v^T[y-c-C xi-G beta]| <= sum_k alpha_k R_k(beta).
```

这是两条线性 EQ/LE 可用的上、下侧行。普通终端 LP/MILP 仍只查询这种有证外包及保留的 P/guards/masks，不引入 SOCP、QP、SDP、搜索、split、backward 或对偶修复。外包不可行可以支持相应 CERT；可行点不自动满足球，更不自动是具体 ADV。

递归的数学输入必须始终是保留共同球的域，而不是先把球替换为有限线性平面后，又假装那个更大的多面体仍满足原球。有限方向行一般不能精确表示 L2 球。

成本仍未过门。每个非线性块增加最多 m 维共同误差，全部旧 E_k、原 bits、guards、source 与 decoder 留存。WC、WG、WE_k 可能填充，历史相位半径列也增长。K0 的精确盒上二次范数最大值不能作为免费 oracle；可用行绝对和平方的上界或有证 Gram 界，但前者可能松，后者需要付 Gram 工作与存储。Kk、d_j、参考值及平方根向外认证均计费。

对已有物理 residual，一个方向的两行通常至少支付源/输出方向系数及半径相位系数；完整源仿射公式还要支付 vC、vG 与各 E_k 的合并。新 e 若物理化，q=Lambda*g+constant+e 的定义行、原 g 的连接、bounds、masks/guards 另计。没有固定宽度、少存储或测得的 GPU 加速；稀疏乘法与归约适合 GPU 是实现可能性，不是性能结论。

还有一个实质性弱点：此固定斜率更新不给新 gamma 创建中心或半径系数。若初始 G=0 且各半径为常数，则这个事实持续成立。此时非凸相位作用来自保留的 guards/masks/epigraph，而不是球自动学到了新的相位耦合。仅保留这些标签不能证明已形成强大的 Neural-HZ。

## 先行研究与下一项区分

[Ellipsotopes Definition 2](https://arxiv.org/pdf/2108.01750)已有连续范数块及线性约束、仿射操作；其基本定义是凸的，没有本项目要求的原神经元二元身份。[Combastel 的 typed-symbol 集合](https://arxiv.org/pdf/2009.07387)已有连续、signed、Boolean 符号及全局共享身份。因此范数、混合符号、共享因子与仿射闭包分别都不是新原理。

[k-ReLU](https://proceedings.neurips.cc/paper/2019/file/0a9fdbb17feb6ccb7ec405cfb85222c4-Paper.pdf)已研究联合多门关系，但它的实验比较不是本记录完整 source-labelled pair-hull 命题的证明。项目 D009/D012 也已有能量、共同相位残差及前向界。本轮的可保留增量是源丢失反例、其半径不可修复证明，以及具体保源仿射再抽象合同；不能仅据此声称新域或论文级新颖性。

下一研究对象是普通非零中心、混合正负权重、共享 skip 及下一激活的组合：这个载体是否能以低于保留整图和附加关系的完整代价，保住有效源依赖并产生新的相位相关信息。必须与 D009 能量及已有低成本混权关系比较。若只是已知 ellipsotope 与旧相位 guards 的并置，仍未完成目标；不得靠恢复全部旧门图再加球 helper 冒充成功。

本轮无实现晋级，不再围绕正交控制扩张试验。后续候选执行仍须新预注册和原完整测试、真实同结构、shadow、逐家族、全量回放；数学记录不放宽任何预算或人口。

## Provenance 与成绩

2026-10-04 Australia/Sydney；分支 redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。依赖为链接的冻结文档、所读一手论文和本轮纸面推导，无可执行候选配置或新依赖。

没有候选 import/AST/compile、测试、模型、求解器、GPU、shadow、replay 或后台作业。生产与旧档未改。正式 1870/2413=1063 CERT+807 validated ADV；独立 CIFAR100 25、TinyImageNet 36，共61/400，均未更新，新收益均为零。Goal 保持 active、未完成。归档技能仅用于本地文档及证明/能力边界区分，未创建外部 Page。
