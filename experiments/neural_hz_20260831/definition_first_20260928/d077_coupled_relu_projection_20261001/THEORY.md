# 跨领域机制如何加强 ReLU 关系传播

跨领域文献应帮助改变关系语言及其可组合变换，而不只是扩大书单。本次在已有综述基础上，推导一个具体结论：对已经认证的四槽相位区间，可以用有限条输入输出耦合关系，得到该区间抽象的完整投影，而不引入新的槽内连续副本。普通混权残差例子表明，它严格强于只对端点做 ReLU 的旧表变换。

这属于已有析取凸包和支持函数机制的迁移，不是新的凸化原理，也不证明完成了 Neural-HZ 定义创新。原连续源、全部原二元相位、EQ/LE、共享身份及输入 decoder 均保留。没有实现、数值运行、真实模型资格或正式成绩收益。

## 文献改变了什么判断

已有[跨领域近邻研究](../../literature_nearest_neighbor_20261001/RESEARCH_NOTE.md)及[关系接口研究](../../literature_interface_decision_20261001/REVIEW.md)已经覆盖以下方向；本次不把重新阅读当作首次发现。

程序分析的 [observational reduced product](https://pcousot.github.io/publications/CousotCousotMauborgne-FoSSaCS11-LNCS6604-proofs.pdf)，FoSSaCS 2011，§4.1，提供通过共同观察交换信息的已有理论。项目推断是：定义必须说明原源、相位及幅值的共同语义和前向消费规则；仅给 HZ 附一份表不是足够的新颖性证据。Trace partitioning 可用于理解关联丢失，但不迁入分支分析。

控制理论的 [Fazlyab 等 QC 研究](https://arxiv.org/pdf/1903.01287)，§III-C、式15至17，通过重复非线性的斜率关系联系不同神经元。项目推断是借鉴关系内容，而不把验证器改成 SDP 或引入 dual rescue。对同一赋值的 ReLU 输出差 d 与输入差 e，d*(e-d)>=0 是已有有效关系，不是本项目新定理。

混合整数优化的 [Vielma Cayley embedding 论文](https://arxiv.org/pdf/1704.03954)，§3 Proposition 1、§4 Definition 3及 Proposition 4，研究析取凸包的投影和共同支持方向。这直接启发下述有限耦合变换。[Anderson 等的强神经 MIP 表述](https://optimization-online.org/wp-content/uploads/2018/11/6911.pdf)，§5.2 Proposition 13，已经给出保留原输入及一个二元相位的非扩展理想单门表述。因此，零新增连续变量或胜过 big-M 都不自动证明创新。此处不采用其松弛点驱动分离算法。

符号可达性中的 [Sparse Polynomial Zonotopes](https://arxiv.org/pdf/1901.01780v2)，Definition 1、Proposition 10，用共同因子身份区分相关加法和独立集合的 Minkowski 和。[神经网络 polynomial zonotope 研究](https://arxiv.org/pdf/2207.02715)，§3，则用多项式和可靠误差外包处理激活。项目推断是借共享赋值的语义，不能用该近似替换原二元 ReLU 图；共享符号本身已有直接近邻。

图模型的 [Bucket Elimination](https://ics.uci.edu/~dechter/publications/r76A.pdf)，§2.1、§2.3，要求 join 在共享变量的同一取值上成立，并明确连续不等式消元不能只凭图宽度得到离散情形的复杂度结论。项目推断是：残差两路的均值一致或相位分布一致不能替代共同连续见证；树形组织不免费提供小型精确摘要。

## 现有接口及研究前提

[现有四槽接口](../d066_wide_phase_interface_20261001/phase_interface.py)的 relu 只把每槽 [L,U] 变成 [max(L,0),max(U,0)]，compile_rows 对每个读出编译两条界。它保留原 child bit 的身份与证据，但表本身没有同时约束该门输入 h 和输出 r。

固定两个已有原 bits alpha、beta，使用已有连续 overlap delta。这里 bits 的 0/1 写法只是原相位的标签记号，不删除或连续化生产中的二元变量。分析 LP 松弛时定义四个质量：

```text
lambda00 = 1-alpha-beta+delta
lambda10 = alpha-delta
lambda01 = beta-delta
lambda11 = delta.
```

已有四条 McCormick 行保证 lambda>=0，sum lambda=1；原 bits 取整时 delta=alpha*beta，只有一个质量为1。

前提是同一原 readout h 在每槽 s 有有限、已认证的 L_s<=U_s，r=ReLU(h)，且所有身份与前提都指向同一原赋值。本结论不免费生成条件界，不使用相位限制 LP、不枚举求解子问题。若界无证书、不有限或次序反转，本定理不能使用，也不能据此删位或宣布 SAFE。旧 D066 允许反转端点表示空相位，因此不能把本结论未经检查直接接入其所有输入。

## 有限耦合投影

记 R(t)=max(t,0)，并从已认证端点确定有限斜率集：

```text
Theta = {0,1} union { U_s/(U_s-L_s) : L_s<0<U_s }
C_s(theta) = max(R(L_s)-theta*L_s, R(U_s)-theta*U_s).
```

除原 h 表的两条行外，加入：

```text
r-theta*h <= sum_s lambda_s*C_s(theta)     for theta in Theta
r         >= sum_s lambda_s*R(L_s)
r-h       >= sum_s lambda_s*R(-U_s).
```

Theta 只依赖已认证结构端点，不依赖待排除 LP 点、终端 margin、模型身份或求解状态。没有新增相位或槽内连续副本；重用原 delta。对四槽最多六条上行及两条下行。

**纸面命题。** 对每个固定 lambda，上述行及原 h 的加权上下界，恰好刻画

```text
sum_s lambda_s * conv{ (t,R(t)) : L_s<=t<=U_s }.
```

因此整体是带四槽标签的区间抽象图，在 (h,r,alpha,beta,delta) 上的完整凸包投影。它不是原网络带全部共同源的完整凸包，也不声称保留 child bit gamma 后的完整扩展 formulation ideal。本命题只在数学比较中投影 gamma；实际 HZ 不移除 gamma 或任何原 guard。

证明如下。每槽的图凸包是三角形、线段或点。其上侧边方向只有斜率0、1及跨零槽的 secant 斜率 U/(U-L)；下侧方向只有0、1。固定 lambda 后，二维多边形的 Minkowski 和只合并各项的边方向，不产生其他方向。上侧方向 r-theta*h 的支持值为各槽支持值之和，而各槽最大值位于一个端点，恰为 C_s(theta)。下侧两个最小支持值分别为 R(L_s) 和 R(-U_s)。左右支持值是 L_s、U_s，已由原 h 表提供。于是列出的有限方向覆盖全部支撑边；线段、点或零质量项由相同支持等式处理。

再利用带标签联合凸包在固定 lambda 纤维上等于上述加权 Minkowski 和，即得整体投影结论。这是经典 Cayley/析取投影几何的应用，不以该证明宣称新的通用凸包理论。原 bits 取整时，所有新增行对原真 ReLU 图均有效，故作为冗余谓词加强不会删掉任何真实输入或合法零点相位。

## 普通混权残差的严格对照

以下是解析控制，不是数据集实例或网络 ADV。令 x,y in [-1,1]，保留所有原相位：

```text
q=R(x), p=R(y), alpha=phase(q), beta=phase(p)
h=x+y+q/2-p/4-1/2
r=R(h), gamma=phase(r).
```

四槽按00、10、01、11排序，h 的精确区间为：

```text
[-5/2,-1/2], [-3/2,1], [-3/2,1/4], [-1/2,7/4].
```

取四个真实父图点的源坐标：

```text
A=(19/20,19/20), B=(19/20,-1/20)
C=(-1/20,19/20), D=(-1/20,-1/20).
```

各占1/4，父联合图凸包中的平均元组为：

```text
x=y=9/20, q=p=19/40
alpha=beta=1/2, delta=1/4, h=83/160.
```

比较点令 r=83/160、gamma=3/4。原 h 表允许 h in [-3/2,5/8]；旧端点 clamp 允许 r in [0,3/4]，均不拒绝。全局 h 区间 [-5/2,7/4] 的原四行也不拒绝。

它甚至通过“完整父联合带标签凸包 P，然后在凸化父集合 P 上建立完整 child 带标签图凸包”这个更强对照，不只是通过 scalar big-M。构造 P 中两点：

```text
T0=(48/61)*C+(13/61)*D
T1=A/3+B/3+(13/183)*C+(48/183)*D.
```

逐式计算 h(T0)=0、h(T1)=83/120。T0 的 child 取合法零点标签 gamma=0、r=0；T1 取 gamma=1、r=h。以1/4、3/4混合，恢复全部上述父均值、r=83/160及gamma=3/4。这里 child 的输入域是已经凸化的 P，不是原始父真图，不能称其为整个网络的完整图凸包。

另一方面，新下行强制

```text
r-h >= (1/2)*lambda00 = 1/8,
```

而比较点 r-h=0，严格违反。该行也写成 r-h>=(1-alpha-beta+delta)/2。

这并非仅拒绝不可实现的纯相位分布。四个原源点的真实 h 依次为131/80、7/8、13/80、-3/5，真实 child 相位为1、1、1、0；它们各占1/4，正好给出比较点相同的 (alpha,beta,delta,gamma)。所以完整相位边际可以真实实现，问题仍在相位内部幅值的共同解释。四个原源见证都在盒内部，所有原预激活均非零；只有局部凸化 child 的证明点 T0 位于零点并使用允许的标签，不能删掉这个零点语义。

这个例子证明旧端点变换有可修复的关系损失；不证明生产 baseline 当前缺此关系、不证明超越全部已知多神经元关系，也不证明新的抽象域。原 HZ 加完全相同的线性谓词会有同样的逻辑强度。

## 费用与仍未解决的问题

定义 T(c)=c00+(c10-c00)*alpha+(c01-c00)*beta+(c11-c10-c01+c00)*delta，可直接编译每条行。若 h、r 已有物理坐标，每行最多涉及 h、r、alpha、beta、delta 五个变量。完整八行的保守账是至多40个系数 nnz；原 h 表、delta 四行及全部原 HZ 另计。theta=0 上行与第一下行正是旧 clamp 两行，因此去重后至多增六行、30 nnz。新增连续辅助变量为0，是以 delta 已存在为前提。

四槽系数生成是固定有限工作，但除法和端点比较仍有位长、可靠舍入与证据费用。若 h、r 只有 latent 表达，展开其共同源系数后的实际 nnz 必须另计；若改为显式坐标则要计变量及等式。不能把这个接口账当作全网计费，也不能继承旧组件测试或 GPU 资格。GPU 可能承担批量系数生成和稀疏传播，是实现设想，不是已测加速或全 GPU 求解。

相比每槽引入 active 质量及正负幅值副本的经典 Balas lift，上述投影不需要这些副本；代价是定理不包含 gamma 的 ideal 性，更不恢复被区间表丢弃的全部连续源关系。显式 lift 仍是已有对照，不作为无相位析取的新方法启动。

目前最重要的定义问题依旧是：怎样让多个后继在相同原赋值上共同消费关系，并在普通混权、下一激活和残差合流中保留有用信息，而不退回整张未简化网络图或无成本投影假设。本次结果只是可复用的算子理论支撑。不能用它把目标重新缩成“优化现有算子”；需要实质的域元素、具体化、嵌入及组合定理后，才进入默认关闭的预注册实现。

## 范围和 provenance

2026-10-01，branch redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。配置 primary_literature、paper_derivation、read_only_history、no_candidate_import、no_numeric_run。根代理与并行只读审查核对证明及控制；没有执行模型、求解器、测试或 GPU，也没有新后台实验。

依赖是上述论文及校验清单列出的目标、既有综述、D066源码和D076推导，不新增运行时依赖。只新增本隔离文档及 SHA256SUMS，不修改旧文档、冻结源、失败诊断、生产代码、Goal 或远端；不 commit/push。已有九个 tracked 文件差异为3806 insertions、57 deletions，属于先前工作。

正式 baseline 1870/2413 = 1063 CERT + 807 validated ADV；独立 E0 为 CIFAR100 25、TinyImageNet 36，共61/400。两账不相加；formal_gain=0，未完成新候选的任何正式回放，不保证满分或 PLDI 新颖性。Goal 保持 active。write-page 用于分开记录论文事实、项目推断、纸面命题及未验证收益；只读回本地文本，不发布外部 Page。
