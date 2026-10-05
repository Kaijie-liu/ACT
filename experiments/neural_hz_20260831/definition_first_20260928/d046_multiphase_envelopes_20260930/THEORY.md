# 多原相位仿射包络的前向关系闭包

本页定义一种保留完整原 HZ 的关系观察载体：观察的上下界可以同时是多个原 phase bits 的稀疏仿射式，而不再先把其他 bits 全局化为常数。给出固定原结构顺序下的前向仿射与 ReLU 规则；不枚举 phase 组合，不求条件 LP，不产生新 bit 或 bit 乘积。

核心 clamp 来自已知超模集合函数的贪心仿射上界，伴随量下界使用单元素边际仿射下界。因此本页是一个有完整语义和构造的数学候选，不将这些经典机制宣称为新定理，也不等于已完成的 Neural-HZ 创新、真实网络收益或实现资格。

## 原非凸载体和共享见证

固定原网络的带类型 frame，包含原连续因子 xi、全部原二元因子 b、原物理端口及其身份。令

```text
H = (B_H,D_H,P_H,F_H,decoder),
xi in B_H, b in D_H, P_H(xi,b).
```

`B_H` 是原连续盒，`D_H` 是全部原二元因子的域，`P_H` 保留原 EQ/LE、门关系及所有其他谓词，`F_H` 是物理读出。原 decoder 与共享 latent 身份保留。本文写 `b_i in {0,1}`，其中 1 是认证的 active 方向；如果原编码为 -1/+1，使用已认证的仿射编码转换而不删除或重命名原 bit。

一个有限观察为

```text
O = (E,L,U,source_identity,certificate),
L(b) = l0 + sum_i l_i*b_i,
U(b) = u0 + sum_i u_i*b_i,
L(b) <= E(xi,b) <= U(b).
```

`E` 是既有物理端口的有符号线性读出，`L,U` 的每个非零系数绑定具体的原 bit ID。所有观察都解释在同一个原赋值 `(xi,b)` 上；值、标签或区间相同不是身份相同的证明。候选元素 `N=(H,O_set)` 的具体化为

```text
gamma(N) = { F_H(xi,b) : xi in B_H, b in D_H, P_H(xi,b),
             all observations in O_set hold on this SAME assignment }.
```

空观察嵌入原 HZ。只有在原整数语义下已认证的观察才能由构造器加入，所以加入这些观察不改变 `gamma(H)` 或原输入重构。非凸性仍由原二元域和谓词保持，不能把这一定义替换为连续的 bit 盒或纯凸域。

在共同 typed frame 上，用具体化包含定义语义预序，按具体化相等取商后得到偏序。不声称存在可计算最佳抽象、完整格或最佳闭包。同一个 H 的不同有效观察集可以有完全相同的整数具体化，却给连续查询不同的外包；这两种精度概念必须区分。原 HZ 加相同有效行具有相同逻辑强度。

## 原算子与观察算子的职责

完整算子由两部分组成。原 HZ 部分按原模型执行已认证的仿射、卷积、Add、Concat 与原非线性门构造，保留精确的原网络关系和全部原 bits；本页不替换或简化这些谓词。观察部分仅生成由同一原关系推出的有限有效包络。

若某个原算子尚无已认证的精确 backbone 构造，本页观察规则不填补这个缺项，也不授予该原算子的资格。身份缺失、前提不一致、非有限系数、数值或预算失败时，不安装该候选观察，保留原 HZ；不能把 UNKNOWN/ERROR 解释成收益或相位不可行。

预激活为零时原 inactive 与 active 两种合法 bit 选择都保留。下文对完整 binary cube 成立的界也覆盖这两个赋值，不用把值相同的门合并。不同原 bits 可以同时出现在一个仿射界中，但绝不被重命名为同一个锚。

## 有限种子和有符号仿射传递

原有常数界是支持集为空的仿射包络。D044 的单锚四端点观察可直接写成

```text
L(b)=L0+(L1-L0)*alpha,
U(b)=U0+(U1-U0)*alpha,
```

因此是本载体的子类。原门幅度行如 `0<=q<=Q*alpha` 也可以作为已认证种子。构造这些种子所需的真实源界、差界、相位方向及其证书仍须付费；本页不假设免费支持函数或源约束闭包。

设已存观察 `L_j(b)<=E_j<=U_j(b)`，实际固定系数 t_j 和常数 c 给出 `E=c+sum_j t_j E_j`。定义

```text
L_E = c + sum_{t_j>=0} t_j*L_j + sum_{t_j<0} t_j*U_j,
U_E = c + sum_{t_j>=0} t_j*U_j + sum_{t_j<0} t_j*L_j.
```

这在每个共同原赋值上逐项成立，故可把不同观察所引用的原 bit 支持作并集而不丢失其身份。它不是把不同单锚的两状态表逐列相加后假设那些状态同时发生；合并的是有明确原 bit 变量的仿射式。

在 intervalization 之前，先按固定读出身份合并相同原项；有证的线性恒等式可作为同一前沿的固定基。尤其对 `d=q-p` 和伴随 `p`，两个消费者

```text
h=Aq+a, w=Bp+b
```

可使用 `h-w=A*d+(A-B)*p+(a-b)`。更一般的两消费者对同时线性使用 q、p 时，仍先用 `q=d+p` 规范化其真实系数。混合权重不要求相等；仅保留差值而丢掉伴随量一般不足以执行这个步骤。

Conv 是具有原 stride、dilation、padding 和 channel 映射的稀疏仿射算子，按同一规则逐接收行处理。padding 只代表实际零输入，不创造源 bit；BN 等参数的读取和融合次序必须保持原模型语义。Add 在同一原 frame 上相加，不是对两个独立 latent 副本作 Minkowski addition。Concat 只作带原身份的坐标嵌入；后续混权访问对应原端口。

本页主定理采用有限、认证的固定系数。非点 BN 包络需另有可靠区间系数或显式误差变换，不能用中点替代；D045 草稿不因本页而自动获得执行资格。相同非点区间也不证明参数相同。

## 固定原身份顺序的 hinge 上界

对任一原 bits 仿射式

```text
V(b)=c+sum_i v_i*b_i,
```

先合并同一原 bit 的系数，再把负系数写成补 literal：若 `v_i>=0`，令 `a_i=v_i,z_i=b_i`；若 `v_i<0`，令 `a_i=-v_i,z_i=1-b_i`，并把常数加上 v_i。于是

```text
V(b)=c_bar+sum_i a_i*z_i, a_i>=0.
```

literal 是原 bit 的表达式，不是新增变量。zero coefficients 可从此观察的稀疏支持删除，但原 bit 仍留在 H。按已认证的原 phase ID 的唯一 canonical 顺序排列这些项；顺序不依赖 LP 点、待证性质、求解状态或事后收益，不尝试所有排列后挑一个。

令 `F(S)=max(0,c_bar+sum_{i in S} a_i)`。对于 `a>=0`，增量

```text
max(0,t+a)-max(0,t)
```

随 t 单调不减，因此 F 是超模集合函数。设 canonical prefix 为 `P_i={1,...,i}`，`P_0=empty`，定义

```text
m_i=F(P_i)-F(P_{i-1}),
G_plus[V](b)=F(empty)+sum_i m_i*z_i.
```

这是一个稀疏原 bits 仿射式。对任意集合 S，按 canonical 顺序把 S 的元素逐个加入；每个加入时的前驱集合都包含于对应 P_{i-1}。超模性给出

```text
F(S)-F(empty)
  = sum_{i in S} marginal_at_previous_elements_of_S(i)
 <= sum_{i in S} marginal_at_P_{i-1}(i).
```

故 `max(0,V(b))<=G_plus[V](b)` 对全部 binary b 成立，并在 canonical chain 顶点上取等。计算只需对 k 个非负系数作一次 prefix 和及 k 个标量 hinge 差；这些是封闭标量式，不执行 k 个网络 phase 子问题，也不枚举 2^k 个组合。

实现可以使用相等但不含两个 hinge 相减的闭式。设包含当前项的前缀为 `t_i=c_bar+sum_{j<=i}a_j`，则 `m_i=min(a_i,max(0,t_i))`；singleton 边际同理为 `n_i=min(a_i,max(0,c_bar+a_i))`。这是因为 ReLU 在长度 a_i 的区间上的增量等于该区间位于正半轴的长度。该写法不改变数学界，也不自动提供浮点舍入证书。

此上界事实上还覆盖 literal continuous cube `[0,1]^k`：任意点是 binary 顶点的凸组合，`max(0,c_bar+a*z)` 是凸函数，而 G_plus 仿射且在各顶点为上界。因此凸性与顶点上界推出全 cube 上界。这里没有声称原域的 bits 已被放松；这是这一个上界公式的额外数学性质。

## 伴随激活所需的 binary 下界

伴随值的下界不能误用上述 majorant。对相同的 F，定义单元素边际

```text
n_i=F({i})-F(empty),
G_minus[V](b)=F(empty)+sum_i n_i*z_i.
```

逐项加入 S 时，超模性给出每项实际边际不小于空集边际，因此

```text
G_minus[V](b) <= max(0,V(b)),  b binary.
```

这个下界与 G_plus 使用相同补 literal 和原 bit 身份，没有新增 phase。它一般较弱，但有限、统一、无需搜索，并足以形成下面的闭包。

必须严格限定：G_minus 的结论是 binary 语义，不是对整个 continuous cube 的逐点结论。例如 `V=-1/2+b1+b2` 时，`G_minus=(b1+b2)/2`；在 `b1=b2=1/4` 上下界右侧 hinge 为零而 G_minus 为 1/4。这个例子不影响由整数真实赋值导出的线性行对其凸包有效，也不许可把 fractional b 当原精确语义。

如果查询松弛中需要额外保留原 gate 的 `t>=0,t>=w`，这些原行照旧存在；不能因增加 G_minus 而删除它们。

## 成对 ReLU 和伴随量的统一闭包

设 `r=R(h),t=R(w)` 是两个真实原门，在同一原赋值上已有

```text
L_delta(b)<=h-w<=U_delta(b),
L_w(b)<=w<=U_w(b).
```

使用逐点 ReLU 关系

```text
-max(0,w-h) <= r-t <= max(0,h-w),
```

生成

```text
L_new(b) = -G_plus[-L_delta](b),
U_new(b) =  G_plus[ U_delta](b),

L_companion(b) = G_minus[L_w](b),
U_companion(b) = G_plus [U_w](b).
```

于是 `(r-t,L_new,U_new)` 与 `(t,L_companion,U_companion)` 仍是原 bits 仿射包络。前两式的 hinge 上界甚至对 continuous cube 成立；伴随下界只主张原 binary 赋值及其有效线性外包。四个 bound 可以引用不同的原 bit 支持及补 literal 方向，但全部仍解释在同一原 witness 上。

健全性由 `h-w<=U_delta`、`w-h<=-L_delta`、ReLU 单调性，以及 G_plus/G_minus 的相应定理直接推出。消费者自身的原 bits 及 zero 合法选择均仍由原门谓词解释，不被改成某个祖先锚，也不需创造 shadow gate。

输出包络可被下一次固定混权、Add 或 Conv 继续消费，再用相同规则通过下一成对 ReLU。对有限原网络按拓扑顺序归纳，即得到任意有限次这种已覆盖算子组合的健全观察；这不是无损关系闭包、全网完备性或始终不退化的精度定理。

当每个输入 bound 只依赖同一个单锚 alpha 时，k=1 的 prefix 边际与 singleton 边际完全相同，恰为 hinge 在两个 binary 端点间的仿射插值。故差值和伴随的四个状态端点完整退化为 D044 的

```text
delta_s -> [min(0,delta_s.lower),max(0,delta_s.upper)],
w_s -> [max(0,w_s.lower),max(0,w_s.upper)].
```

这个退化是整数端点及有效行的一致，不把伴随 lower chord 错称为 continuous hinge 的逐点下界。支持集为空时得到常数 clamp。

## 相对于先全局化的具体区别

先把所有 bits 取全局范围，再应用旧单锚规则，会丢掉多个原 phase 同时出现的仿射条件。一个直接示例是已认证的

```text
h-w >= 1/4-alpha/2-eta/2.
```

按 canonical 顺序 alpha 再 eta，F 的三个 prefix 值为 `0,1/4,3/4`，新规则给出

```text
t-r <= alpha/4+eta/2.
```

若先全局化 eta、只保留 alpha，旧两端点 clamp 给出 `t-r<=1/4+alpha/2`；反向单锚处理给出 `t-r<=1/4+eta/2`。新行可以严格强于这两个旧行的交，但这不意味着它强于每一种可能的精确 source-observer 生成器。另一排列产生不同有效行；本候选仅使用预定 canonical 顺序，不把选排列当作 LP 点驱动菜单。

严格物理输出正控与强比较资格另列在本目录 CONTROL.md：比较须明确包含哪些真实 prefix/单门 hull、全局 D020 及旧实际生成的单锚行。不能仅从上面 binary 仿射小例子推断真实网络收益，也不能假设任意新增 observer 都有免费的精确源支持。原 HZ 加新行与本载体的逻辑强度相同。

## 数值与完整成本

对一个支持数 k 的 bound，已 canonical 排序且身份已认证时，literal 规范化、prefix majorant 或 singleton minorant 都需要 O(k) 个有限标量运算和 O(k) 输出系数。若输入未排序，须计入实际 canonical 排序或已排序支持的合并成本；不能把 O(k log k) 排序、hash 身份检查或系数归并隐藏为零。实际算术包括 prefix 加法、hinge、边际差、补 literal 回写及每个系数的位长控制。

有符号仿射组合需触及各输入包络的实际稀疏支持；其工作不是单纯接收层权重 nnz。观察支持是原 bit 集合的并集，可能随网络深度扩大。固定同锚的 8m 个端点常数前沿不再适用：对 m 个差值加伴随量，每个分量至多 K 个原 bit 时，仅四个稀疏仿射 bounds 就有 O(mK) 个系数及身份引用。历史观察、原谓词与证据若保留还会继续累计，不声称总内存恒定。

若预注册设置支持、数值位长或工作上限，超额不得偷偷删原 bit、换实例或减测试人口；可 fail-closed 不安装这条候选观察并保留原 backbone。是否以固定有证方式退回更弱界属于另行明确的统一规则，不由本页默认授权。

在已有物理坐标上，差值 bound `L=l0+l*b,U=u0+u*b` 编译为

```text
r-t-u*b <= u0,
-r+t+l*b <= -l0.
```

伴随量同理为两行 `t-u_comp*b<=u_comp0` 与 `-t+l_comp*b<=-l_comp0`。四行的坐标 nnz 上限为

```text
6 + nnz(l)+nnz(u)+nnz(l_comp)+nnz(u_comp).
```

每个 bound 支持不超过 K 时为 `6+4K`；K=1 退回至多十个坐标 nnz。没有新 bit、bit product 或必要的新差值变量，不意味着没有终端成本。若物理端口本身是宽 latent 读出，必须计其实际展开或隐式算子，而不是把坐标 nnz 当最终 HZ 矩阵规模。

正向 A 算子读取实际原物理端口和所列原 bits；转置 A^T 将对应乘子累加回完全相同的原端口及 bit 列。稀疏访问、scatter/reduction、RHS、原 bit 编码转换、slack、dtype、传输、临时与保留内存、decoder 和终端求解均须纳入完整账单。CPU 数学操作或上述公式不授予 GPU 性能资格。

对 R 条实际生成的行，设 E 为其相位支持总数。已有 canonical 布局时，分段 prefix、clamp 与 literal 恢复的标量工作为 O(E+R)，理想并行扫描深度为 O(log k_max)；这是算法分析，不是 GPU 性能测量。若读出必须展开到 latent，总支持 L 也需计入，材料化及每次正向/转置查询成本随 E+L 增长。排序、系数汇总、设备 context、原 HZ 与新行的峰值共存、证据遍历及重构都另计；matrix-free 查询也不能免除每次系数生成。

GPU 认证还有独立缺项：必须为前缀和及 literal 回写提供可靠的向外界。clamp 闭式避免直接作两个近似 prefix 正部之差，却不会自动解决 scan 误差、transpose 累加误差或 BN 非点系数。对上包络可在非负 literal 坐标中使用每个边际的可靠上界；下包络则要求相反方向。恢复原 bit 系数和常数时也必须保持整个不等式的可靠方向，不能只把最终 scalar 向外移动一格就宣布合格。

本页有限系数运算可用精确有理数证明；若后续实现用浮点，系数本身的可靠界和每一步向外舍入需另证。特别是 prefix 相邻差不能未经误差分析就当精确系数。全部数学、真实同结构、shadow、逐家族及完整回放门和原资源上限保持不变。

## 已知机制与未完成的贡献

超模函数的 prefix 边际上界属于已知 greedy polyhedral 与 Lovasz-extension 框架；对 F 取负可回到标准次模 base-polyhedron 论述。可参 Francis Bach，Learning with Submodular Functions A Convex Optimization Perspective，2013 版本第 3 章的定义、greedy 算法与凸性联系：[作者论文](https://arxiv.org/pdf/1111.6453)。本页给出该 hinge 特例的完整短证明，不宣称发现新的 supermodular 定理或新 ReLU 凸化原语。

更直接的既有仿射上界表述见 Iancu、Sharma、Sviridenko，Supermodularity and Affine Policies in Dynamic Robust Optimization，Operations Research 2013，p.947 Lemma 1 的式 (10) 至 (11)。完整凹包络涉及所有相应排列，本候选仅固定一条结构顺序，不能继承完整包络的紧性。[作者全文](https://dan-a-iancu.github.io/publications/supermodularity-affine-policies-dynamic-robust/supermod_robust.pdf)

载体的共同赋值具体化属于既有观察式 reduced-product 思路；差值加伴随的仿射传播也有 ReluDiff 等先例。这里实际扩展的是让多个不同原 bits 保留在同一有限仿射关系里，并给出不枚举组合的前向 clamp 与伴随闭包。它不只是重排存储，但是否成为有价值的定义贡献，还取决于统一源生成、普通残差结构上的适用率、强公平比较和完整成本；精确 HZ 原本就能表达这些网络关系。

固定 greedy 排列不保证给出最紧包络；singleton lower 也可能很弱。源条件只通过现有观察进入计算，没有自动吸收全部原 guard，因此强制声称所有 phase 依赖已经捕获或所有普通网络都会改善是错误的。此处不要求超越 HZ 加相同 facts，也不以全网最佳抽象作为新的晋级门。

## 存档与资格

本页是 2026-09-30 在 `redu-hz` 上新增的隔离数学文稿。具体 commit 与文件身份由主代理统一封存；没有修改旧理论、生产默认、模型、日志或结果。没有导入或执行候选、数学测试、模型、求解器或设备调用，没有本页对应的实现、资源或真实网络资格。

保留原连续因子、全部原 bits、EQ/LE、共享 latent、decoder 和 fail-closed。禁止 attack/PGD、BaB、输入或 phase split、backward/dual rescue、实例或 LP 状态菜单。数学证明对任意集合讨论不是执行 phase 枚举；实际实现若执行子问题，则超出本页范围。

正式基线仍为 1,870/2,413（1,063 CERT + 807 validated ADV），全部旧解和 13 家族保留要求不变。独立 CIFAR100 25、TinyImageNet 36，共 61/400，不与正式基线相加。本页 formal_gain=0，不默认启用、不宣称创新完成。
