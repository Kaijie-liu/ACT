# 从跨领域文献推导 Neural HZ 的研究问题

需要跨出 HZ 文献圈，但本轮的目标不是扩大书单，而是让文献改变下一项定义研究。建议优先检验：把共享源谓词作为非凸激活关系的一部分，能否获得在普通卷积与残差结构上可组合、成本可控的前向规则。它目前是研究假设，不是已完成的新抽象域，也不是验证成绩提升。

精确 HZ 已能保留共同输入、全部相位和神经网络的分段仿射关系。真正需要改善的是这些关系在可负担的传播、松弛和终端查询中能被利用到什么程度。不能把“旧 LP 松弛允许虚假点”说成“精确 HZ 不会表达这个网络”。

## 本轮与既有综述的区别

此前的[首轮综述](../../literature_cross_domain_20260929/REVIEW.md)、[补充综述](../../literature_cross_domain_20260930_followup/REVIEW.md)及[共同源研究](../d019_projected_source_20260930/CROSS_DOMAIN_AND_GPU_REVIEW.md)已经涉及 ImageStar、DeepPoly、控制理论、混合符号域、强混合整数建模、静态分析和消元。这些阅读不重新记成新发现。

本轮新增的重点是三个具体联系：用抽象解释的相对完备性方法指导关系语言设计；用析取规划的共同谓词吸收解释局部松弛的组合缺口；用知识编译的表示与查询分离审视完整成本。两项独立文献审查还复核了关系分析、互补系统和带标签 formulation 的先例。所有文献事实均来自论文正文的相关章节，不以搜索摘要作数学依据。这不是穷尽式新颖性检索。

## 最值得借鉴的机制

### 从算子需要出发设计抽象域

Giacobazzi、Ranzato、Scozzari 的 *Making Abstract Interpretations Complete*，JACM 2000，§§1.2–1.4、4–5，研究在相应条件下通过域的最小扩展或限制获得相对给定语义算子的完备性。[作者全文](https://www.sci.unich.it/~scozzari/paper/JACM00.pdf)

项目推断：先固定一类普通神经块及希望保留的观测，找出组合时哪些信息缺失，再设计必要的关系元素；不再先选漂亮表示、后问它能否用于验证。论文的存在性结论不保证有限、便宜的 Neural-HZ 实现。这里的“完备”相对所选抽象与算子，不是全网满分，也不新增“必须求完整凸包”的项目门槛。本文只借域设计方法，不导入运行时 backward 分析。

### 先纳入共同谓词再松弛

Ruiz、Grossmann 的 *A hierarchy of relaxations for nonlinear convex generalized disjunctive programming*，§2.6，区分合并两个真正析取，与把共同凸谓词纳入一个析取；后一操作不增加析取项数，但不保证 formulation 的行数或连续副本没有增加。[作者全文](https://egon.cheme.cmu.edu/Papers/RuizGrossmannEJORFinal.pdf)

项目推断：源条件不应仅在局部投影完成后才接回来。应寻找能在激活关系内部处理的低支撑共同条件。该原则已有先例；真正需要研究的是神经结构中的闭式规则及其组合成本。通用析取展开、全相位枚举、论文的分支搜索流程均不导入。

### 同源观测必须交换信息

Cousot、Cousot、Mauborgne 的 *The Reduced Product of Abstract Domains and the Combination of Decision Procedures*，FoSSaCS 2011，§3.3、§4–4.1，讨论观测扩展和保持具体化不变的信息归约。[作者全文](https://www.di.ens.fr/~cousot/COUSOTpapers/publications.www/CousotCousotMauborgne-FoSSaCS11-LNCS6604.pdf)

项目推断：差值、原始值和相位必须是同一 latent 赋值的观测，不能让每条关系边自行选择源见证。给 HZ 附加一个关系缓存不自动成为新域；还需要说明哪些信息经什么规则传递，以及为什么在相同预算下更有用。

Cousot 等的 *Combination of Abstractions in the ASTRÉE Static Analyzer*，ASIAN 2006，§2.2、§6.3，区分算子不够精确、关系变量组选择不当和抽象域无法表达所需性质。[作者全文](https://cs.nyu.edu/~pcousot/publications.www/CousotEtAl-asian06.pdf)

项目推断：后续按重复结构定位信息变得不可用的位置，再选择关系作用范围。不能用实例标签、旧成绩或 terminal margin 选菜单；也不应默认全网 all-pairs。其精度诊断方法可借鉴，不代表具体 CNN 关系图已被证明有效。

### 激活差值与非凸互补关系是必要先例

Fazlyab 等的 *Safety Verification and Robustness Analysis of Neural Networks via Quadratic Constraints and Semidefinite Programming*，§III-C.3、式 15–17，研究重复激活的增量斜率关系。ReLU 差值满足 `d(d-delta)<=0`。[论文](https://arxiv.org/pdf/1903.01287)

Aydinoglu 等的 *Stability Analysis of Complementarity Systems with Neural Network Controllers*，§3.1–3.2，使用互补关系组织 ReLU 网络及其层间结构。[论文](https://arxiv.org/pdf/2011.07626)

Banerjee、Xu、Singh 的 *Input-Relational Verification of Deep Neural Networks*，DiffPoly §4.1–4.5，给出差值关系的激活和仿射传播；其差值跨仿射传播已经是既有方法。[论文](https://ggndpsngh.github.io/files/raven.pdf)

项目推断：保留共同非线性的关系很有价值，但“差值传播”“ReLU 互补形式”“结构化整网系统”均不能单独报新。我们保留全部原 bits 和零点合法选择，不整体换成 QC、LCP 或凸差值域；也不整包移植 DiffPoly 的 back-substitution。原始值锚点不能被差值替代。

### 离散标签和查询成本不能隐藏

Vielma 的 *Embedding Formulations and Complexity for Unions of Polyhedra*，§2–2.1，研究附有离散编码的多面体并集及其 formulation 大小。[作者全文](https://juan-pablo-vielma.github.io/publications/Embedding-Formulations-and-Complexity.pdf)

项目推断：要比较带全部原相位标签的关系，不只比较输出集合；我们不能通过重编码删除原 bits。带标签凸包、连续提升副本都已有先例，须与候选逐项比较。

Darwiche、Marquis 的 *A Knowledge Compilation Map*，JAIR 2002，§§2、4–5，将表示紧凑性、支持的查询与组合操作分开考察。其 decomposability 要求合取子项的变量集合互不相交。[论文全文](https://arxiv.org/pdf/1106.1819)

项目推断：两个神经块共享输入或残差变量时，不能直接套用这种独立性。不能把存储一个小 DAG 等同于终端易解；也不导入基于相位 conditioning 的编译流程。该文是完整成本的设计参照，不是把 Boolean 查询复杂度直接搬到连续 HZ 的证明。

Dechter 的 *Bucket Elimination: A Unifying Framework for Reasoning*，§2.3，明确指出 Fourier 消元还受不等式数量增长影响，不能仅由图的 induced width 给出相同复杂度界。[作者全文](https://ics.uci.edu/~csp/r48b.pdf)

项目推断：局部卷积或稀疏关系图并不自动保证廉价全局消元。此处只用于核算 fill、谓词和接口成本，不恢复旧“先删变量再说”的主线。

### 精确商化简作为比较对象而非唯一主线

Ressi 等的 *Neural Networks Reduction via Lumping*，2022 预印本，Definition 6、Theorems 1–2，给出保持网络函数的比例商化简；§4 还讨论一般正线性组合所需的同号条件。[论文](https://arxiv.org/pdf/2209.07475)

项目推断：可借鉴行为等价与组合证明，但函数等价不自动保留每一个内部原 bit。它也不能成为只寻找完全重复或比例神经元的理由。当前应优先面对普通混合权重和残差，而不是把改进寄托于少数精确恒等式。

## 文献怎样改变下一项数学问题

设 `G` 是带原相位标签的非凸激活图，`H` 是同一源坐标上的凸共同条件。已知的基本关系是：

```text
conv(G intersect H) subseteq conv(G) intersect H.
```

左侧先结合条件再松弛，右侧先松弛再接条件；二者可能严格不同。这个包含关系只用于分析查询松弛，候选的具体化仍保留二元相位，不变成凸域。对于含上游离散变量的真实源，不能擅自把它当作凸 H；需要保留完整源关系，再明确哪一部分有效条件被当前局部规则使用。

[已有共同源反例](../d019_projected_source_20260930/D019_PROJECTION_AND_FIBERS.md)正说明这种问题：`x in [0,1]^2`，`t=x1+x2`，`y=ReLU(t-1)`，残差为 `x2-y`。只对 `(t,beta,y)` 取精确凸包，再接回输入，允许 `x=(0.9,0.6), beta=y=0.75`，但所有真实点均有 `y<=x2`。这个反例已经存档，不是本轮新实验或新定理；添加已知有效不等式即可修复它，因此修复这个例子本身远远不够构成域创新。

[已有相位差分因子](../d020_phase_difference_20260930/D020_PHASE_DIFFERENCE.md)进一步保留了原 bits，但仍是投影上的局部 ideal 因子。它既不自动解决共同源一致性，也不证明多个因子接起来得到全局 ideal hull。这正是从“改一个算子”转向“设计可组合关系语义”的原因。

### 待检验的一个主假设

在普通 `Conv -> ReLU -> mixed Conv -> ReLU` 以及带 `Add/Concat` 的结构上，能否用按结构统一选取的小接口，保留源条件、原始值、差值与原相位的必要联系，并使这些关系跨块传播，而不在每一层重新展开完整源或相位组合？

接口的含义必须是对同一原始连续因子和二元因子赋值的观察，不是新的独立源副本。所有原 EQ/LE、零点的独立 bit 选择、共享身份、输入解码和 fail-closed 语义保留。具体关系语言和闭式变换尚未给出，因此不将这句话命名为已经完成的 Neural-HZ。

先在纸面上回答三个问题，然后再冻结默认关闭的实验版本：

1. 写出域元素、具体化及 Affine/Conv、ReLU、Add/Concat 的规则；区分整数语义的精确对应与查询松弛的改善。说明共同条件在何处参与，避免只重写原计算图。
2. 给出普通多算子块上的有效组合例子与适用条件。独立标量界、相位无关差值、强单节点 formulation 和 HZ 加相同有效事实都应进入比较；不要求全局完备或每块完整凸包。
3. 计算传播、谓词行数与 nnz、全部 bits、连续辅助量、源副本、位宽、临时内存、终端转换、查询与重构的完整账单，同时给出 GPU 正向与转置谓词算子的执行形式。公式可并行不等于已测得 GPU 加速。

这些是接下来要产出的研究材料，不是本轮已完成的证明。若候选与“普通 HZ 加完全相同的事实”最终生成同一套终端约束，逻辑精度当然相同；可成立的贡献只能来自可证明的组合规则、同预算下保留更多有效关系或更低完整成本，而不能声称集合表达力凭空更强。更多变量或约束本身不是否决理由，但必须证明值得付出该成本。

## 不改变的实验和权限边界

本轮没有导入任何论文求解算法，也没有运行模型、求解器、CPU/GPU 数值实验、shadow 或全量 replay。没有修改生产代码、默认配置或历史文件，没有新建分支、commit 或 push。没有重试已失败的 GPU 跟踪路径，也没有更改系统权限。

后续仍按数学测试、真实同结构样例、shadow、逐家族和完整 2413 回放晋级；外部 400 单独回放。禁止 attack/PGD、BaB、输入或相位 split、backward/dual rescue、实例菜单；原普通终端求解及具体见证验算边界不变。能力与纯速度门继续解耦。此次文献建议没有增加新的全局完备性门，也没有放宽任何既有验证门。

正式成绩仍为 **1870/2413 = 1063 CERT + 807 validated ADV**，保全 13 家族。外部 E0 仍为 **61/400 = CIFAR100 25 + TinyImageNet 36**，不与 1870 相加。本轮新增正式收益为 **0**。综述可以指导创新，但不是已达到 PLDI 级创新或能够保证满分的证据。

## 来源和保存记录

日期为 2026-09-30；分支 `redu-hz`；commit `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`，工作区原有修改保留。配置为一手文献相关章节阅读、本地存档只读比对和研究假设整理；无需新增执行依赖。当前 Goal 保持 active，未完成或暂停。

本轮读取并记录的冻结输入 SHA256：

```text
0fd93920413f63ad6b3b65849ea4d857c416c6bf4ab56263c63ff73416ebe80c  GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md
06d70c10a6730b93e55637d2554f9460ffcef9ab70b927f1d782fb1c68f3dccc  literature_cross_domain_20260929/REVIEW.md
f0dde6059d4fb7306cbbebf81df7199d4c91bdb43bee342619799ff0455f758a  literature_cross_domain_20260930_followup/REVIEW.md
7bf446c3600b349f09648a32f535b011198b479767e5945693dbedb7a52073f7  definition_first_20260928/d020_phase_difference_20260930/D020_PHASE_DIFFERENCE.md
```

`pages:write-page` 技能用于区分文献事实、项目推断和未验证假设；文件保存在新的隔离目录，并做本地文本核对。独立只读审查确认了 GDP 的适用范围、非凸语义与松弛的区分、新颖性声明和既有权限边界。未发布外部 Page。冻结综述不编辑，其后续修订或实验必须另存新版本。
