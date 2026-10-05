# 用跨领域研究重新选择 Neural-HZ 的突破点

结论是应当借鉴 HZ 以外的研究，但不应继续只增加书单。下一项优先问题是：在普通混权卷积和残差块中，如何用可组合的非凸关系保留共同源、原相位与多个输出的联系，并在完整成本下改善验证。它是待检验问题，不是已完成的新域，也没有产生新增正式解。

本文回应用户关于跨领域研究的建议。已有[首轮综述](../literature_cross_domain_20260929/REVIEW.md)、[补充综述](../literature_cross_domain_20260930_followup/REVIEW.md)和[定义导向综述](../definition_first_20260928/d021_literature_to_definition_20260930/REVIEW.md)已经涉及抽象解释、强混合整数建模、控制理论、关系验证和分段仿射代数。这些不是本轮首次发现。本轮将原论文与后续真实适用性负证据对照，明确哪些方向应该推进、哪些不能再包装成创新。

## 为什么要调整研究重心

[固定配对审查](../definition_first_20260928/d040_fixed_pair_audit_20260930/RESULTS.md)在一个已存档块的完整 1440 个固定源对中，没有得到非平凡的已认证共同关断组。它只否定那个固定配对、锚点及既有证书组合的新增收益依据，不否定所有层或整个方向。

[已有门同余规则](../definition_first_20260928/d041_existing_gate_congruence_20260930/THEORY.md)给出了纸面严格正控，但仍要求已有对应门和精确同源证书。真实网络适用性尚未获得认证。下一项研究不应继续把主要希望放在神经元恰好相等、成比例或共同关断上。

还应区分精确语义与计算过程：精确 HZ 本来可以保留分段仿射网络和共同输入。问题不是它在数学上完全不能表达相关性，而是这些关系在可负担的前向传播、查询松弛与终端处理中能被利用多少。只找到一个旧 LP 松弛中的虚假点，不等于发现 HZ 集合表达能力的缺陷。

## 文献给出的具体启发

### 程序分析中的关系交换

Cousot、Cousot、Mauborgne 的 *The Reduced Product of Abstract Domains and the Combination of Decision Procedures*，FoSSaCS 2011，通过观测交换和归约组合不同抽象组件。[作者全文](https://www.di.ens.fr/~cousot/COUSOTpapers/publications.www/CousotCousotMauborgne-FoSSaCS11-LNCS6604.pdf)

项目推断：源读出、相位、输出差值不能只是各自维护的缓存，必须说明它们如何在同一赋值下相互约束。值得研发的是适合神经算子的有限关系语言及前向规则，不是把现有 HZ 加一个关系表重新命名。本文不导入 SMT 搜索、回溯或运行时 backward 分析。该启发在旧综述已出现，本轮继续作为定义要求。

### 多神经元验证中的小组关系

Müller 等的 *PRIMA: General and Precise Neural Network Certification via Scalable Convex Hull Approximations*，POPL 2022，§§3、5，使用重叠的小神经元组建立联合约束，其 SBLM 明确切分输入多面体的激活区域。[作者全文](https://ggndpsngh.github.io/files/PRIMA.pdf)

项目推断：研究对象可以从一个门转为共享感受野的小组及其后续消费者，不必以精确相等为前提。但是只借鉴联合关系的组织问题，不能整包移植区域切分流程，也不能把非凸 HZ 整体替换成其凸近似。分组选择、源投影和联合谓词均须付费；论文中的局部算法复杂度不能直接当成我们的端到端成本。

### 运筹优化中的分组表示和强单门基线

Tsay 等的 *Partition-Based Formulations for Mixed-Integer Optimization of Trained ReLU Neural Networks*，NeurIPS 2021，§3.2，将一个门的加权输入和分组。这里 partition 是索引分组，不是搜索式输入或相位 split。其 Proposition 1 后的 Remark 说明，在所述区间界设定下，消去辅助量对应预选的 Anderson 凸包不等式；Propositions 2–3 给出单组与逐输入分组两个端点。[正式论文](https://proceedings.neurips.cc/paper/2021/file/17f98ddf040204eda0af36a108cbdea4-Paper.pdf)

Anderson 等的 *Strong Mixed-Integer Programming Formulations for Trained Neural Networks*，2020，§5.2、Proposition 13，给出盒源上单个仿射 ReLU 的带相位 ideal formulation。[作者稿](https://optimization-online.org/wp-content/uploads/2018/11/6911.pdf)

项目判断：逐门增加 partial sums 或几个条件不等式已经有充分先例，不能据此宣称新域。候选至少要与这些强单门表示比较，而不只比较 scalar big-M。值得进一步问的是多个消费者之间怎样保留有用的共同源关系；仅使用同名输入变量还不能保证其松弛采用同一联合解释。论文的 OBBT、依赖当前 LP 点的切割循环和搜索流程不移植。

### 析取几何中的标签和共同解释

Vielma 的 *Small and Strong Formulations for Unions of Convex Sets from the Cayley Embedding*，§6 Definition 5、Proposition 6，区分仅保证输出凸包的 sharp 与保留标签联合凸包的 ideal，并分析固定混合权重下的集合关系。[作者全文](https://juan-pablo-vielma.github.io/publications/Small-and-Strong-Formulations.pdf)

项目推断：模块摘要不能只保证输出范围，还需要保留输出与原相位、共同源之间的联系。这是可借鉴的语义区分，不是论文已经给出了 Neural-HZ 的跨块组合定理。原文针对凸集合的析取；不能未经证明便把仍含其他二元相位的 HZ 当作一个凸分支，也不使用新编码删除原 bits。

### 差分验证和块摘要

Paulsen 等的 *ReluDiff: Differential Verification of Deep Neural Networks*，ICSE 2020，§4，直接传播对应网络在共同输入下的差值。Zhong 等的 *Scalable and Modular Robustness Analysis of Deep Neural Networks*，2021 预印本，§3，以 BBPoly 的块摘要组织输入输出关系。[ReluDiff 作者全文](https://chaowang-vt.github.io/pubDOC/PaulsenWW20_ICSE_ReluDiff.pdf)，[BBPoly 原文](https://arxiv.org/html/2108.11651)

项目推断：关系对象可以是带有认证误差的差值，不必只存等式；作用范围也可以跨过 Add 后的 Conv/BN，不能只比较相加前的两个节点。但 ReluDiff 的两网对应结构不是任意残差的现成条件，BBPoly 的 back-substitution 和精度成本折中也不能直接移植。可借的是定义问题和比较对象，仍须研发符合当前限制的前向非凸规则。“多层摘要”本身已有先例。

### 分段仿射代数作为备选而非重启旧路

Balestriero、Baraniuk 的 *A Spline Theory of Deep Networks*，ICML 2018，§4.1、Theorem 1，将相应网络算子写成最大仿射样条算子的组合，同时指出多层组合不必仍是单个凸最大仿射算子。[正式论文](https://proceedings.mlr.press/v80/balestriero18b/balestriero18b.pdf)

Zhang、Naitzat、Lim 的 *Tropical Geometry of Deep Neural Networks*，ICML 2018，Proposition 5.6 给出实权重 ReLU 网络的 tropical rational signomial 表示。[正式论文](https://proceedings.mlr.press/v80/zhang18i/zhang18i.pdf)

项目推断：成对保留分段凸成分有助于理解混权与残差中的抵消，但精确重写不自动降低验证成本。旧 D011 和补充综述已研究 tropical/DC 路线；不能将它重记为新发现。只有发现普通结构上的新组合化简且原 bits、谓词、源身份及解码仍完整，才有理由重新实施。平坦展开或仅保留原计算 DAG 都不足以成为突破。

### 稀疏消元只做成本约束

Gärtner 等的 *Large Shadows from Sparse Inequalities*，§4，表明稀疏不等式系统仍可能有很大的二维投影。[原论文](https://arxiv.org/pdf/1308.2495)

项目判断：不能从卷积局部、约束稀疏或接口维数小直接承诺精确投影便宜。这只是避免错误的无条件复杂度声明，不据此研究极端实例，也不把旧变量消元线重新设为主目标。普通网络是否有可利用结构仍由真实同结构证据决定。

## 下一项研究如何落到定义上

优先研究普通的混权 Conv → ReLU → 多个仿射消费者，以及它们进入 Add 后的传播。按图结构固定作用范围，不要求精确相等，不按模型身份、历史成绩、margin 或求解状态选菜单。

待验证假设是：存在一种可计算的小范围关系语言，使同源的多个读出及其原相位条件可以一起前向传播；相较于独立处理每个门，在支付全部成本后保留更多有用关系。暂称“源耦合块关系”只是问题标签，不是已定义的新 Neural-HZ。

下一份候选说明应交付：

1. 明确的域元素、具体化和与原 HZ 的对应，不能只画一个模块图或附加缓存。
2. Affine/Conv、ReLU、Add/Concat 的关系变换，标明精确或健全近似；原连续因子、全部 bits、EQ/LE、共同赋值和输入解码保留。
3. 一个普通多算子结构上的严格收益及完整成本分析；比较包含强单门表示、已有分组方法，以及 HZ 加完全相同事实。相同终端约束当然具有相同逻辑强度，可能的创新应在关系生成、组合定理或同预算可用精度上。
4. 随后按原顺序做数学测试、预注册真实同结构检查、shadow、家族与全量回放。纸面正控不替代真实适用率，GPU 可并行公式不替代设备测量。

这不是要求任何固定小接口完整表达整个网络，也不新增“必须达到全局 ideal hull”的门槛。不允许靠 split、BaB、attack/PGD、backward/dual rescue 或动态状态菜单获得所谓域收益。文献方法中不符合边界的部分留作比较，不执行。

## 本轮结果和保管

日期 2026-09-30；分支 redu-hz；commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。配置为一手论文相关章节复核、旧档只读比较和研究决策整理。没有数值实验、网络或 solver 执行、GPU 初始化、shadow 或全量回放；没有 commit/push、生产改动或默认启用。

正式成绩仍为 1870/2413，即 1063 CERT 与 807 validated ADV；保全 13 家族的要求不变。独立 E0 仍为 CIFAR100 25、TinyImageNet 36，共 61/400，不与正式基线相加。本轮 formal_gain=0。综述不证明满分可达，也不证明已达到 PLDI 级创新。当前 Goal 保持 active。

只新增本隔离目录；旧综述和失败版本不改。原有九个 tracked 文件的 diffstat 保持 3806 insertions、57 deletions。文献检索并非穷尽式新颖性检索。按 pages:write-page 技能区分论文事实、项目推断和待验证假设，完成本地文本核对；不创建外部 Page。

本轮只读证据 SHA256：

```text
0fd93920413f63ad6b3b65849ea4d857c416c6bf4ab56263c63ff73416ebe80c  GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md
800c702cc3797b03f20e46d38a2e7f3862d500f6d23327b2c7bb33760fa7e3b9  definition_first_20260928/d021_literature_to_definition_20260930/REVIEW.md
e2c8056eede3c7807be75ec828e7ba489506bd19ed1e105dbee39b1c3f4d3971  definition_first_20260928/d040_fixed_pair_audit_20260930/RESULTS.md
3c2a502990af09e3e8f6cc11650ca383da7f0899a7449e6099f087f3bcac2a0e  definition_first_20260928/d041_existing_gate_congruence_20260930/THEORY.md
```
