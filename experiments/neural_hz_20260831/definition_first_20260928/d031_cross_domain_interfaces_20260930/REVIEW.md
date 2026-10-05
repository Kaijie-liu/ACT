# 跨领域研究怎样改变 Neural HZ 的下一步

本次回应用户提出的跨领域研究建议。结论是需要问题驱动的文献综述，但不必再从零堆一份书单：既有研究已经读过抽象解释、析取规划、多神经元分析、非凸可达集和知识编译。当前应把这些机制转成可反证的域设计问题，重点研究普通卷积与残差块之间的共同源关系怎样组合，而不是继续只压缩矩阵或给计算图换名字。

精确 HZ 本来能够表达网络的分段仿射关系。待改进的是关系在可承受的前向传播、查询松弛和终端成本中能被利用到什么程度。本记录不是新颖性认证，也不是数值实验预注册或正式成绩提升。

## 已有基础与本次增加的判断

[已有综述](../d026_cross_domain_synthesis_20260930/REVIEW.md)和[构造研究](../d028_constructive_support_20260930/RESEARCH_NOTE.md)仍为只读依据，不把重读列为新的研究成果。[条件观察记录](../d029_conditional_observations_20260930/THEORY.md)已有共享源支持、误差界和四行投影；[边界记录](../d029_conditional_observations_20260930/LIMITS.md)已说明旧相位的一阶条件观察不能免费跨越新相位。

本次补充的是三个联系：图模型中的一致性要求不能简化为连续接口均值相同，下面的树形双门反例可排除这种误用；条件可分的纤维提供了精确组合的充分前提，但需要证明真实适用性和查询代价；析取与 RLT 文献提示把多个条件观察共同投影，检验逐条处理是否遗漏了联合信息。

## 文献中的机制及可迁移范围

### 抽象解释指导关系语言而非先选容器

Giacobazzi、Ranzato、Scozzari 的 *Making Abstract Interpretations Complete*，JACM 2000，研究相对于指定语义算子的域扩展与限制。Cousot 等的 *Combination of Abstractions in the ASTRÉE Static Analyzer*，ASIAN 2006，§2.2 区分变换不够精确、变量分组不合适和域无法表达所需事实；§4.1 以具体化和健全变换组织域，不要求所有实现都具备最佳抽象。[域设计原文](https://www.sci.unich.it/~scozzari/paper/JACM00.pdf)，[ASTRÉE 原文](https://cs.nyu.edu/~pcousot/publications.www/CousotEtAl-asian06.pdf)

项目推断：先确定普通 `Conv -> ReLU -> mixed Conv` 与残差汇合需要哪些共同源观测，再定义域元素和前向规则。Cousot、Cousot、Mauborgne 的 FoSSaCS 2011 论文 Theorem 4 还表明，迭代两两归约未必得到完整 reduced product；增加几个互不交换信息的缓存并不充分。[归约原文](https://www.di.ens.fr/~cousot/COUSOTpapers/publications.www/CousotCousotMauborgne-FoSSaCS11-LNCS6604-proofs.pdf)

这些是设计方法和先例，不保证有限实现或全网满分，也不增加必须获得全局理想凸包的要求。不导入运行时 backward 分析。

### 联合神经元和析取规划提供强对照

PRIMA，POPL 2022，§3 使用重叠小组保留神经元依赖，其 Split Bound Lift 算法会切分输入多面体。Anderson 等的强混合整数 formulation 在原输入坐标保留单门源关系；Ruiz 与 Grossmann 的析取松弛研究区分先吸收共同条件与先松弛再相交。[PRIMA 原文](https://ggndpsngh.github.io/files/PRIMA.pdf)，[强单门 formulation 原文](https://optimization-online.org/wp-content/uploads/2018/11/6911.pdf)，[析取松弛原文](https://egon.cheme.cmu.edu/Papers/RuizGrossmannEJORFinal.pdf)

项目推断：把研究单位从一个标量门扩展到共同源及真实消费者是合理的，但联合有效行本身已有充分先例。只能借鉴关系组织与证明，不移植 split、相位枚举、解点驱动 cuts 或搜索。比较要包括强局部门和已有联合关系；不能只胜过弱 big M 就称定义创新，也不能要求候选逻辑上胜过普通 HZ 加完全相同的行。

Vielma 的 *Small and Strong Formulations for Unions of Convex Sets from the Cayley Embedding*，§6 Definition 5、Theorem 8，进一步区分输出投影正确的 sharp 与带离散标签的 ideal。Bestuzheva、Gleixner、Achterberg 的 *Efficient Separation of RLT Cuts for Implicit and Explicit Bilinear Products*，§2–3，研究原约束隐含乘积的识别和利用。[Cayley 原文](https://juan-pablo-vielma.github.io/publications/Small-and-Strong-Formulations.pdf)，[RLT 原文](https://arxiv.org/pdf/2211.13545v1)

项目推断：原相位和零点选择必须在规格中保留，单条提升系统的精确投影不等于整个网络的 ideal formulation。可复用原门的乘积身份，不能采用论文的 LP 解点分离和分支节点流程。原门 `q=alpha*g` 的复用并不是新发明。

### 非凸可达集提示共享身份必须进入语义

Sparse Polynomial Zonotopes 的 Proposition 10 区分共同因子的 exact addition 与使用新身份的 Minkowski sum。它说明 Add 的依赖语义和存储去重是两回事。[SPZ 原文](https://arxiv.org/pdf/1901.01780)

项目推断：原连续源和相位身份必须穿过 Add、Concat 和卷积组合。可借鉴依赖管理，不把原 HZ 换成多项式域，不删原 bits，也不把区间误差或降阶噪声伪装成同一个原源。共享 ID 本身不构成新颖性。

### 图模型和消元说明接口中必须保留什么

Wainwright 与 Jordan 的 *Graphical Models Exponential Families and Variational Inference*，2008，§2.5.2 Definition 1、Proposition 1，以 running intersection 和相容的完整局部边缘分布建立 junction tree 的组合结论。这里不是仅要求各变量均值相等。§2.5.2 同时说明离散表格推理的代价依赖最大团大小。[原文](https://www.cs.princeton.edu/courses/archive/fall11/cos597C/reading/WainwrightJordan2008.pdf)

Dechter 的 *Bucket Elimination A Unifying Framework for Reasoning*，§2.3，明确提醒 Fourier 消元的复杂度还取决于不等式增长，不能直接沿用有限表格的 induced width 界。[原文](https://ics.uci.edu/~csp/r48b.pdf)

项目推断：可以借共同变量接口、关系连接和投影的数学框架，但不能声称局部卷积天然低树宽，也不能把概率消息传递或期望值当成健全网络证书。仅把原谓词放进若干袋中，是已有因子化，不是新域；必须证明接口保留了哪类共同关系，及传播、终端和重构总成本。

### 知识编译区分表示短与查询便宜

Darwiche 与 Marquis 的 *A Knowledge Compilation Map*，JAIR 2002，分别考察表示大小、查询和变换复杂度；decomposability 要求合取子项的变量集合不相交。[原文](https://arxiv.org/pdf/1106.1819)

项目推断：共享输入和残差通常破坏这种直接独立性。紧凑 DAG、共享子式或 ADD 不能自动提供便宜的连续支持查询。既有 D001 和 D027 已处理过相关语法路线，本次不重启一个改名的决策图候选，也不通过 phase conditioning 编译偷偷实施分裂。

Tropical 神经分析的取舍同样值得保留：ReLU 在所选代数中简单，普通混合权重仿射映射却需要抽象。这里只借从算子反推表示的思路，不整体换域，不采用 subdivision。[原文](https://arxiv.org/pdf/2108.00893)

## 树形局部凸包仍不能保证共同源一致

以下为本次纸面推导和独立复核的教学反例，不宣称新定理，也不作为需要特殊调参的攻击目标。令

```text
x in [-1,1], y = ReLU(x), z = ReLU(2x + 1/2),
原相位分别为 alpha、beta。
```

两个局部关系的变量组为 `{x,y,alpha}` 和 `{x,z,beta}`，共享接口只有 x，因此构成两节点树。分别取各门带原相位标签的完整凸包，再令接口坐标 x 相同，允许

```text
(x,y,alpha,z,beta) = (0, 1/2, 1/2, 1/2, 1)。
```

第一门来自 x=-1 和 x=1 的等权混合，第二门来自具体点 x=0；二者共享 x 的均值，却不共享同一个源分布。所有真实共同输入都满足 `z >= (5/2)y`：x<=0 时右侧为零，x>=0 时差为 `(1-x)/2 >= 0`。上述点违反该关系，所以不在共同图的凸包内。

这只否定“树形关系图加局部理想凸包便足够”的推断。不否定精确非凸关系的自然连接，不否定原 HZ 的表达力，也不把加入这条已有类型的联合不等式当作新域。所有原零点相位选择保留，示例没有求解器、相位搜索或数值实验。

## 可检验的下一项定义问题

优先问题仍是共同源关系怎样穿过普通混合权重和残差，而非全网图分解。域元素至少要明确：原连续与二元 latent、EQ/LE 谓词、共同 frame、所选关系观察、具体化和输入 decoder；凡新摘要没有覆盖的依赖，不能凭独立性假设丢掉。

来自关系消元的一条已知充分条件可以用作精确对照。设源能分成共同接口 s 和互不相交私有量 u_j，且全部相关原谓词确实分解为 `P0(s) and P1(s,u_1) and ... and Pm(s,u_m)`。对固定 s，若各可行纤维非空且读出有界，则

```text
sup [a(s) + sum_j z_j(s,u_j)]
  = a(s) + sum_j sup_{u_j : Pj(s,u_j)} z_j(s,u_j)。
```

左侧只对这个固定 s 下的所有私有量取上确界。证明是纤维笛卡尔积上的可分优化；上界逐项相加，下界用每项逼近上确界的可行点同时组合。全部原 bits 仍属于 s 或原私有组，谓词和 decoder 并未删除。

这不是免费支持 oracle：函数消息怎样表示、无 split 怎样构造、随后对 s 怎样求界，仍未解决。小接口不保证少分段。若两个 fan in 共享私有祖先，或原谓词跨私有组耦合，前提立即失效；不能仅凭张量分组或不同通道判为独立。

一个更直接衔接当前数学工作的优先假设是多个观察的共同投影。对同一原 bit alpha，设纸面提升量 `t_i=alpha*v_i`，i=1,2；原共同谓词已经认证 `lambda_1*v_1+lambda_2*v_2<=d`，其中 lambda 非负且由原谓词固定，不是对偶优化变量。真实点因此满足 `lambda_1*t_1+lambda_2*t_2<=d*alpha`。

若各 t_i 的局部线性系统有三个仿射下界 `ell_i,k` 和三个上界，保留各系统独立可行性的投影后，额外共同上界的精确投影为至多九条线性行：

```text
lambda_1*ell_1,k + lambda_2*ell_2,l <= d*alpha,
k,l in {1,2,3}。
```

这是已知线性消元的直接推论：选择每个 t_i 的最大下界便同时最小化非负加权和。充分性还要求 t_i 没有其他未纳入的消费者或耦合约束。这里的九个组合是线性下界组合，不是相位枚举；原二元变量一个也不消去。

尚未证明的是这组关系能否在普通混合残差上严格强于两套 D029 单观察投影及已有聚合观察，也未证明完整成本净收益。消去两个辅助量可能增加行数和 nnz，不能把“无辅助变量”直接记为简化。若只是既有聚合观察或 RLT 的重复表达，就归为支撑组件，不宣称新域。其候选价值在于检验有限关系接口怎样组合，不是多加九条 cut 的名称。

下一项研究先检验这个共同投影假设及新相位下的延续范围，D029 已有的一阶矩反例仍是必要反证检查。条件可分分解则只在真实结构满足前提时作为精确对照，记录接口增长与混合卷积破坏前提的位置；若通常不满足，就停止该候选，不转向寻找罕见比例门或特殊模型来凑成功。

有意义的交付应是一个有范围的组合定理和可构造变换，而不是“关系缓存更小”。随后才进行新的默认关闭实现、真实同结构集合、shadow、逐家族和全量回放。存储、传播、证据、终端转换、查询、输入重构以及 GPU 传输都在完整成本内；本次没有 GPU 性能证据。

## 当前执行状态与存档

2026-09-30，分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。本次仅做一手论文相关段落核对、旧档只读检查和纸面推导；未导入或运行候选、测试、模型、solver、GPU、shadow 或 replay。没有重跑旧失败版本。

相邻 D030 参考实现仍为未执行草稿，本轮未修改它，也没有把静态审查记为通过。后续首次冻结前需处理已有静态审查指出的 ROOT 导入路径、候选导入前身份核对，以及缓存生命周期和瞬时 numeric entries 的保守记账；总资源门和测试人口不变。若继续该组件，它也不能替代完整 Neural HZ 定义、真实完整残差块或正式 replay。

正式 baseline 保持 1870/2413，即 1063 CERT 和 807 validated ADV；全部 13 家族、每个旧解和 invalid ADV=0 的要求不变。外部 E0 仍为 CIFAR100 25、TinyImageNet 36，共 61/400，独立计账。本次正式增益为零。Goal 仍为 active，整体目标未完成。

仅新增当前隔离目录；生产与历史数据未改，原九个 tracked 修改的 diffstat 仍为 3806 insertions、57 deletions；未 commit 或 push。不引入 attack/PGD、BaB、输入或相位 split、backward/dual rescue、实例菜单；普通终端和独立具体见证边界不变。

本次使用 pages:write-page 将论文事实、项目推断、纸面证明与未证假设分开存档；保存为项目约定的本地文档，没有发布外部 Page。文献复核限于相关段落，不是穷尽式系统综述或新颖性认证。

本轮核对的项目输入 SHA256 如下，旧文件保持只读：

```text
0fd93920413f63ad6b3b65849ea4d857c416c6bf4ab56263c63ff73416ebe80c  GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md
a29e38379e02afebe9ad04b602d924a07e4d6e6dc0412060356764c99671f77f  d026_cross_domain_synthesis_20260930/REVIEW.md
a3987e6e2ea566c1f9697983cbf708c8c48516da6615ec6e70d6b2c7813d30ec  d028_constructive_support_20260930/RESEARCH_NOTE.md
65879b0d8d8a4eedd8e204f2d53a0fc8814e907677c0efb3bdfec7e4a2f01bda  d029_conditional_observations_20260930/THEORY.md
905f5891955bf774076459d760096fbb3a32e9e987c8b8aa610d292aa73c4910  d029_conditional_observations_20260930/LIMITS.md
```
