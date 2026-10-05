# 跨领域文献对 Neural HZ 定义研究的启发与边界

需要继续跨出 HZ 文献圈。当前最值得研究的问题是：共同输入产生的带符号关系，怎样成为可以穿过混合权重、ReLU 和残差的域内关系，而不靠重新展开全部源变量或相位组合。这个问题尚未解决；本文是有边界的机制综述，不是新的抽象域、完整新颖性检索或验证成绩。

精确 HZ 已能承载网络的分段仿射关系。需要改善的是关系在可负担的传播、查询松弛和终端处理中能被利用的程度，而不是声称原 HZ 没有共同输入语义。凸关系可以作为辅助事实，不能替换原非凸具体化。

## 与已有阅读的关系

[既有跨领域综述](../d021_literature_to_definition_20260930/REVIEW.md)已包含抽象解释、析取规划、控制理论、关系验证、知识编译和消元。本次不把重读记成新发现，也不重启已否决的坐标换名或容量电路候选。

新增的实质判断有三项：本轮提出的共同分量联合行已有 k-ReLU 类先例，不能报定义创新；共同抵消量和耦合投影这条直觉有明确的否决理由；新近 CLAD 的约束求界机制提供了新的比较提醒，但其算法不符合本项目当前边界。纸面正控和否决证明分别保存在 [数学记录](CONTROL_AND_REJECTIONS.md)。

## 文献机制及项目推断

### 抽象解释中的域设计和信息归约

Giacobazzi、Ranzato、Scozzari 的 *Making Abstract Interpretations Complete*，JACM 2000，研究相对于指定语义算子的域扩展与限制。Cousot、Cousot、Mauborgne 的 FoSSaCS 2011 论文 §3.3、§4–4.1 用共同观测和归约组织多个抽象域；Theorem 4 特别指出，两两归约即使迭代也未必达到完整 reduced product 的精度。[域设计原文](https://www.sci.unich.it/~scozzari/paper/JACM00.pdf)，[带证明的归约原文](https://www.di.ens.fr/~cousot/COUSOTpapers/publications.www/CousotCousotMauborgne-FoSSaCS11-LNCS6604-proofs.pdf)

项目推断：先固定普通神经块和需要保留的观测，再找组合时缺少的关系。只把两个缓存并列并不充分；必须定义同一 latent 上的信息交换和前向传递。这里的相对完备性不是全网满分、全局 ideal hull 或有限实现的保证。只借域设计方法，不导入 backward 分析，也不增加全局完备性门。

### 多神经元分析和强混合整数建模

k-ReLU 的 *Beyond the Single Neuron Convex Barrier for Neural Network Certification*，NeurIPS 2019，§1 的例子已利用共同输入限制多个 ReLU 的总幅值。§3.3 式(6) 的通用构造涉及相位组合。PRIMA，POPL 2022，§3 用重叠小组保留更多依赖；§5 Algorithm 3 是 Split-Bound-Lift 构造。[k-ReLU 原文](https://papers.nips.cc/paper_files/paper/2019/file/0a9fdbb17feb6ccb7ec405cfb85222c4-Paper.pdf)，[PRIMA 原文](https://ggndpsngh.github.io/files/PRIMA.pdf)

项目推断：研究单位可以是普通神经元小组及其真实消费者，而非孤立的标量激活。但仅增加联合有效行已有充分先例。只借联合源关系和有界重叠的设计问题，不移植相位枚举、SBLM 的切分或论文整套求解路径。小组大小、共享输入、源证书和终端代价均须完整付账。

Anderson 等的 *Strong Mixed-Integer Programming Formulations for Trained Neural Networks*，本次核对的 44 页版本 §5.2 Proposition 13、式(29)，在原输入坐标上给出单个仿射 ReLU 的 ideal formulation。其非扩展不等式族可能指数大；Proposition 14 的高效 separation 不等于免费存下所有行。[本次核对版本](https://optimization-online.org/wp-content/uploads/2018/11/6911.pdf)

项目推断：非线性因子需要看到可认证的共同源条件，不只是一个输入区间。比较对象必须包括强单门源 formulation 和普通 HZ 加相同事实，不能只胜过弱 big-M。编号以此版本为准，不覆盖旧档引用的其他版本。解点驱动 separation、cut callback、BaB 不导入候选路径。

### 差分验证和共享依赖

ReluDiff，ICSE 2020，§3.3、§4，通过共同输入上的符号差值减少分别求界造成的依赖损失。NeuroDiff，ASE 2020，§4.3–4.4，进一步研究 ReLU 差值的符号界及中间符号。[ReluDiff 原文](https://arxiv.org/pdf/2001.03662)，[NeuroDiff 原文](https://arxiv.org/pdf/2009.09943)

项目推断：可借直接观察共同分量、差值和带符号残差的做法。不能将独立噪声冒充同一源，也不能只留下差值而丢掉绝对值锚点和全部原相位。差分坐标本身是已知方法，且只做可逆换元不会改善整个 LP。原论文的细化和搜索流程不自动获得采用权限。

### 非凸可达集中的因子身份

Sparse polynomial zonotopes 的 Proposition 10 区分保留共同因子的 exact addition 与丢掉依赖的 Minkowski sum；其降阶和部分非线性处理会使用外包近似。神经网络多项式可达分析则把激活近似与有界误差一起传播。[SPZ 原文](https://arxiv.org/pdf/1901.01780)，[神经网络可达分析原文](https://arxiv.org/pdf/2207.02715)

项目推断：共享因子身份应进入 Add/Concat 的数学语义，不仅是指针去重。多项式集合可能非凸，不应混同普通凸 zonotope；但它们不因此成为符合本项目全部原 bits 条件的替代品。共享 ID、dependent generator 和误差封装已有先例，不能单独作为创新声明。

用户此前提到的 ImageStar 也是必要对照：*Verification of Deep Convolutional Neural Networks Using ImageStars* 的 Definition 2 及 reachability 部分将图像 anchor/generator 与共享谓词结合；精确 ReLU 路线使用分裂，无分裂路线使用新增谓词变量和凸外包。[ImageStar 原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7363231/)

项目推断：可借鉴张量运算与谓词语义的分离，但仅做图像化基底或新增三角约束不满足本项目的定义创新目标。既不采用分裂，也不以单个凸 ImageStar 替换保留全部相位的 HZ。

### Tropical 几何中的算子取舍

Goubault 等的 *Static analysis of ReLU neural networks with tropical polyhedra*，2021，§2.2 中 ReLU 是 tropical 线性操作；§3 处理普通加权仿射映射时需要抽象，§6 讨论表示转换成本。[原文](https://arxiv.org/pdf/2108.00893)

项目推断：应从目标算子反推合适的代数，但必须同时核算 Affine/Conv 与 ReLU。让 ReLU 变简单而把困难转给混合权重层，不足以证明整体改进。不会整体替换 HZ，也不采用文中的输入 subdivision。旧容量电路的非凸性审查仍有效，见[既有审查](../d025_interval_capacity_20260930/DEFINITION_AUDIT.md)。

### 新近 CLAD 的比较提醒

Duong、Le、Nguyen 的 *CLAD: Constrained Abstract Domain for Neural Network Verification* 是 2026-09-28 的预印本。本次核对 §3–§4：它让附加凸输入约束参与符号仿射界的具体求值；§3.2 式(20) 从拉格朗日函数的切平面产生界，而非把未收敛目标值直接当证书。本文未完整审计其附录证明或复现实验。[原文](https://arxiv.org/pdf/2609.34628)

项目推断：必须检查“谓词被保存”是否真的意味着廉价求界过程利用了谓词。不过其反向代入、输入投影梯度与对偶乘子迭代不纳入本项目路线。§4 的比较中，基线得到包围区间，CLAD 另得到约束；因此其报告不能证明胜过拥有并利用相同约束的 HZ，也不是我们的 CIFAR100、TinyImageNet 或 13 家族结果。

## 对下一项定义研究的具体影响

优先保留一个可证伪问题：在同一连续与二元 latent frame 上，能否定义有限的带符号关系接口，使普通 `Conv -> ReLU -> mixed Conv -> ReLU` 和真实残差 Add 的共同源信息按统一前向规则继续传递？

需要交付域元素、具体化、原 HZ 嵌入、算子规则及有适用范围的组合定理。若最终只是普通 HZ 加相同有效行，逻辑精度相同，只能先归为支撑组件；实质贡献必须体现在新的可证明组合规律或同等完整预算下利用更多关系。不能把全部计算图装进新名字、局部严格例子或 GPU 并行语法当作完成创新。

研究比较采用同一源谓词、全部原相位和同一性质。包含独立强单门、既有差值和容量关系、联合关系先例，以及 HZ 加完全相同事实；不用弱化对手制造收益。无需全层 all-pairs 或全局 ideal hull，不以罕见比例门、极端数值和测试框架扩张替代普通结构。

一旦有实质候选，先做完整成本预注册：传播、谓词行与 nnz、所有 bits、连续辅助量、共享源与临时量、终端转换和查询、见证解码、证据成本、GPU 设备及传输成本。当前没有新的 GPU 性能证据，不重试冻结的失败版本，也不通过文献阅读放宽资源门。

## 保存记录和不变边界

日期 2026-09-30；分支 `redu-hz`；commit `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`。模式为一手论文相关章节阅读、旧档只读对照、独立纸面数学复核；无新增执行依赖。工作区原有 9 个 tracked 修改保持原状。本次只新增此隔离目录文档；未修改生产默认、历史模型或结果，未 commit/push。

权威输入 SHA256：

```text
0fd93920413f63ad6b3b65849ea4d857c416c6bf4ab56263c63ff73416ebe80c  GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md
800c702cc3797b03f20e46d38a2e7f3862d500f6d23327b2c7bb33760fa7e3b9  d021_literature_to_definition_20260930/REVIEW.md
5538f41275a93bcf7e1f268085128431a99f0a1d82ad2c28bf7dc19571903b60  d025_interval_capacity_20260930/DEFINITION_AUDIT.md
```

服务端 Goal 保持 active；其 GPU 要求及当前状态优先于旧权威文件末尾的服务未同步说明。禁止 attack/PGD、BaB、输入或相位 split、backward/dual rescue、实例菜单；普通终端求解和原始网络见证验证边界不变。

无候选导入、模型或求解器执行，无 CPU/GPU 数值实验、shadow、家族或完整 replay。正式基线仍为 1870/2413（1063 CERT + 807 validated ADV），保全 13 家族；独立 E0 仍为 CIFAR100 25、TinyImageNet 36，共 61/400。本轮正式增益为 0。以后仍需同候选完整回放才能更新成绩或默认启用。

使用 `pages:write-page` 技能区分论文事实、项目推断、纸面证明与未验证假设，并做本地文本复核。未创建外部 Page；旧综述和失败实验保持冻结。本文不是实验预注册或实现授权的扩大。
