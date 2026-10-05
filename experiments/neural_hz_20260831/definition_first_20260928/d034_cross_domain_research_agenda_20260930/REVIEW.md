# 跨领域文献如何指导 Neural HZ 的定义研究

需要跳出 HZ 文献，但研究单位应是具体瓶颈，而不是扩大书单。当前优先问题是：共同输入产生的非凸关系，如何经过混合权重、下一次 ReLU 和残差汇合继续发挥作用，同时保留全部原相位、共享源身份和可承担的终端成本。文献可以改变关系语言和组合规则的设计；单纯改存储、添加已有 cuts 或换一个求解器不等于解决这个问题。

精确 HZ 原本能够表达这些分段仿射关系。需要改进的是有限成本下的传播、关系利用和查询，而非声称旧 HZ 没有共同输入语义。本文是问题驱动的相关研究与候选筛选，不是穷尽式综述、创新认证或新的实验成绩。

## 已有工作与本次阅读的区别

[此前综述](../d031_cross_domain_interfaces_20260930/REVIEW.md)及其引用已覆盖抽象解释、析取规划、控制理论、多项式可达集和知识编译。本次没有把重读算成新发现。尤其 Tsay 的输入分组 formulation、增量斜率关系、ReLU 互补形式与 reduced product 都已出现在旧档。

本次补充或深化三点：复核 Taylor 模型中主函数依赖与区间余项的区别；核对规范化 SDD 的组合代价不能继承未规范化版本的界；补读 Bernstein 多项式求界及其隐式 GPU 表示，明确紧凑表示和廉价范围查询不是同一件事。这些判断用来选择下一项数学问题，不作为新增能力。

## 五类研究机制和迁移边界

| 研究领域 | 原文中的机制 | 对本项目的启发与限制 |
| --- | --- | --- |
| 程序分析与抽象解释 | 具名观察和保持具体化的信息归约 | 让原值、残差与相位成为同一源的观察，并真正参与后续变换。并排放几个缓存不够；不假定归约有免费、有限的最佳实现。 |
| 析取规划与强混合整数建模 | 把输入加权和分组，保留同一门标签；提升量投影为强不等式 | 研究共同条件进入非线性关系的位置。代数索引分组不是输入区域 split，但现成分组与消元不能重新命名为域创新。 |
| 控制理论与互补系统 | 重复激活的增量关系、非凸互补约束 | 不只记录每个神经元的独立区间；研究跨门关系。但不整体改为 SDP/QC/LCP，也不引入对偶优化救援。 |
| 可靠数值计算与多项式求界 | 共同变量上的函数表示、可认证余项、张量化传播 | 同时考虑依赖、误差和 GPU 运算。只借机制，不替换原 HZ、不平滑替换原网络、不把余项当成免费同源变量。 |
| 知识编译与数据库消元 | 变量作用域、等价复用、共同可行关系过滤 | 明确组合接口及完整查询成本。不同消费者共享 latent 时不能当作独立；不采用相位条件化或枚举。 |

以下分别给出一手依据及本项目推断。表中的迁移建议不是原论文已经证明的 Neural HZ 结论。

### 观察式抽象域组合

Cousot、Cousot、Mauborgne，FoSSaCS 2011，§3.3 Lemma 4 与 §4 Definition 1 说明如何扩展观察、并在不改变具体化的前提下归约信息。Theorem 4 的两两归约限制已在旧档说明。[带证明原文](https://www.di.ens.fr/~cousot/COUSOTpapers/publications.www/CousotCousotMauborgne-FoSSaCS11-LNCS6604-proofs.pdf)

项目推断：先确定普通神经块需要观察什么，再定义域元素及变换。这里不要求整个网络的最佳抽象或全局理想凸包；需要的是有明确适用范围的前向组合规律。原连续源身份、全部原 bits、EQ/LE 语义和 decoder 保留，观察不能自行选择另一个源赋值。

### 分组 formulation 是对照而非待发明的答案

Tsay 等，NeurIPS 2021，§3 式 13 至 17 按输入索引对加权和分组；§3.2 Proposition 1 将提升系统投影为预选的 Anderson 不等式。Propositions 2 和 3 描述从 big M 到单门盒域凸包的端点。[正式全文](https://papers.nips.cc/paper/2021/file/17f98ddf040204eda0af36a108cbdea4-Paper.pdf)

项目推断：这种分组不等于切分输入区域，不能因为术语 partition 就一概排除；但不能连同论文的 OBBT、搜索和回调一起引入。当前[共同投影研究](../d032_joint_observation_closure_20260930/THEORY.md)已把相关交叉行归约为已知单观察形式，且证明一类独立区间证书下的新增行冗余。下一项研究不应只是继续增加同一相位的分组条件量。

### 重复非线性的关系比独立标量界更丰富

Fazlyab、Morari、Pappas，IEEE TAC 2022 版本，§III C.3 式 15 至 17、Lemma 2，用重复激活的增量斜率建立跨神经元关系。对于 ReLU，记 delta 为两个输入之差、d 为两个输出之差，则 d(d-delta) 不大于零。[原文](https://www.georgejpappas.org/wp-content/uploads/2022/01/Safety_Verification_and_Robustness_Analysis_of_Neural_Networks_via_Quadratic_Constraints_and_Semidefinite_Programming.pdf)

Aydinoglu 等，HSCC 2021，§3.1 Lemmas 1 和 2 给出 ReLU 及多层网络的互补表示。单门 q=ReLU(g) 可写为 q 非负、q-g 非负、q(q-g)=0。[作者全文](https://dair.seas.upenn.edu/assets/pdf/Aydinoglu2021.pdf)

项目推断：关系语言应能利用同一种非线性在多个位置出现的事实，但这两个机制本身不是新颖性。上述关系也不能无代价地变成线性谓词：必须提供符合当前终端边界的健全构造、误差与成本。原相位及零点合法选择继续存在，不能以互补式省掉 bits；本次不实现 SDP 或 LCP 求解路径。

### 主函数依赖和余项依赖要分清

Makino、Berz，2003，§2 Definitions 1 和 2 将共同变量的多项式与区间余项分开传播。标准余项仍按区间算术合成，并未保存全部余项相关性。Theorem 1 有光滑性前提；Remark 1 区分函数逼近阶数与最终范围界精度。[作者全文](https://www.bmtdynamics.org/pub/papers/TMIJPAM03/TMIJPAM03.pdf)

项目推断：若考虑功能余项 e_i=q_i-(m_i*g_i+c_i)，它必须指向原 q_i、g_i 和相同源赋值，而不是新建互不相关的噪声。这样的观察目前只是一个候选语言；单纯引入 e_i 是换元，不会自己提高能力。跨零 ReLU 不能直接套用光滑 Taylor 定理，也不能把独立余项区间的相加宣称为联合求界。

### 多项式求界的 GPU 启发及其适用范围

BERN NN，2022 预印本，§5.3 至 5.4 用多项式上下界和张量运算传播。正文同时说明次数随深度增长、周期性线性化的取舍，以及高输入维度下的内存问题。[原文](https://arxiv.org/pdf/2211.14438)

BERN NN IBF，IEEE TCAD 2024，作者页介绍隐式 Bernstein 表示、张量运算和 ReLU 二次下界。本次期刊 PDF 两个镜像未能读取，未据搜索摘要认定具体定理；改读第一作者学位论文第 4 章相关内容。[论文作者页](https://arthurfeeney.github.io/papers/bern-nn/)，[作者学位论文](https://escholarship.org/content/qt41t4b6w9/qt41t4b6w9.pdf)

学位论文 §4.3 给出的隐式张量尺寸仍依赖变量数、项数和次数；§4.9.1 Algorithm 9 的一般求极值过程仍访问显式系数索引，只是不必完整物化系数张量。项目判断：不能由“紧凑、GPU 并行”推断所有范围查询成本已经线性化，更不能从低维案例外推 CIFAR 或 Tiny 的收益。

值得借鉴的是把数学表示、可计算范围证书和 GPU 算子共同设计，而不是先建立一个昂贵表示再搬到 GPU。若仅作辅助求界，仍需保留原 HZ 非凸语义和全部 bits，证书必须适用于原网络，且 CPU、设备、传输、误差和终端代价完整计入。没有实测 GPU 晋级或提速声明。

### 规范化表示和共同关系过滤

Van den Broeck、Darwiche，2014，Algorithm 1、Theorems 1 和 3，区分未归约 SDD 的多项式 Apply 与规范化 SDD 的组合；后者一般可能有指数增长。论文也报告规范化在其实际基准上有价值，因此最坏界不是一概放弃复用的理由。[原文](https://arxiv.org/pdf/1404.4089)

项目推断：可以借作用域及复用契约，不开一个基于相位展开的决策图实现。数值表达式相同、源前提相同与原 gate 身份相同是三件事，不能只凭共享输出计算合并原相位。

Abo Khamis、Ngo、Rudra 的 FAQ，本文核对 arXiv v7，§1.3 与 §5 的 InsideOut 使用其他因子的 indicator projections 限制中间结果；主要复杂度分析依赖相应因子表示与消去顺序。[原文](https://arxiv.org/pdf/1504.04044)

项目推断：先让共同可行关系参与局部计算，比各自计算后只对齐坐标更值得研究。但是有限条目表格的复杂度不能直接套到连续非凸 HZ，也不移植对输入区域或相位状态的回溯、条件化与枚举。普通稀疏系数或 Bernstein 索引遍历并非该禁令的对象，仍须完整计费。旧档的树形局部凸包反例仍有效：接口均值一致不等于共同源一致。

## 下一项研究应交付什么

优先假设是：在普通 Conv、ReLU、混合 Conv、ReLU 与残差 Add 组合上，能否为不同原相位之间的共同源关系定义有限观察接口，并给出能实际构造的前向传递规则。功能余项只是一个待比较的观察选择，不是确定答案；不要求任意网络都具有固定宽度或全局闭包。

下一项数学交付应包含三部分：

1. 域元素、具体化及原 HZ 嵌入，明确原值、相位、观察和误差属于同一赋值；分别定义 Conv、ReLU、Add 的变换，区分精确对应与健全外包。
2. 普通两跳混合残差结构上的可反证组合命题，比较强局部门、既有差分与容量、同相位观察闭包；不能仅证明单门更紧。不同方法得到完全相同有效行时，不声称逻辑精度不同，研究价值只能来自新的构造或成本定理。
3. 传播、关系数量与 nnz、全部因子、源与余项证据、终端查询、输入重构和 GPU 成本。先判断真实普通结构是否有用，不为罕见恒等式或极端数值另建路线。

若仅是可逆换元、已有 cuts 重排，或需要隐藏的相位枚举和昂贵支持 oracle，就将其归为支撑组件或记录负结论。纸面通过后才另行冻结默认关闭候选，按数学测试、真实同结构、shadow、逐家族、完整回放晋级。这里没有追加“必须全局 ideal”一类新门，也没有放松现有验证要求。

## 保存状态与未执行草稿

日期 2026-09-30；分支 redu-hz；commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。配置为一手文献相关章节阅读、既有存档只读比对和候选筛选；没有新增运行依赖或数值实验。

本轮交接后的只读核查中，相邻 D033 目录已有 joint_support.py、test_joint_support.py、run_reference.py 三个隔离草稿，来自前序已分配的组件准备工作。它们未导入、未运行、未冻结首跑；worker 与预注册尚未完成，结果目录不存在。本次综述保留这些草稿，不继续实现或启动它们，也没有把静态检查记为测试通过。旧 D030 和全部冻结记录保持原状。

本次只新增当前目录的综述与校验清单，未改生产代码、默认配置、历史模型或结果，未 commit/push。原有九个 tracked 修改仍为 3806 insertions 和 57 deletions。Goal 保持 active，不重设或虚报完成。

正式 baseline 仍为 1870/2413，即 1063 CERT 和 807 validated ADV；全部旧解及 13 家族必须保住。独立 E0 仍为 CIFAR100 25、TinyImageNet 36，共 61/400，不与 1870 相加。本次 formal gain=0，没有候选的保旧回放或 GPU 性能资格。

保留连续因子的域语义、原连续源身份、全部原二元因子、EQ/LE、共享 latent/frame、具体输入重构和 fail closed；连续辅助量仅可经已证等价投影及所需反向重构处理，不因此获得删除原 bits 的权限。禁止 attack/PGD、BaB、input/phase split、backward/dual rescue 及身份或求解状态菜单。普通终端与具体网络见证边界不变。

使用 pages:write-page 区分论文事实、迁移推断和未证假设，按项目约定保存本地文本；未发布外部 Page。只核对相关章节，未声称完整复现所有论文。

本次只读核对的项目输入 SHA256：

```text
0fd93920413f63ad6b3b65849ea4d857c416c6bf4ab56263c63ff73416ebe80c  GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md
e44209a7dab24aa583f78869811a20837b62e57716e831e5d9ef4ab852cb8174  d031_cross_domain_interfaces_20260930/REVIEW.md
dfd2297490a85b79870a2229b2541e484a073f3216f8ea8d0b9be99f90e67aa4  d032_joint_observation_closure_20260930/THEORY.md
4ef3aaec15d044549b3cc7ffdea464eee67a486e25553e960ff8d6327355b7fe  d033_joint_support_reference_20260930/joint_support.py
f134a42792e57d34949ee3ea7a924f9ee0cbda8aaad1d9f345783ade4d629781  d033_joint_support_reference_20260930/run_reference.py
afddf2a9a80072eb28a0b6933393171a8f084c4393688809f4573c0b5b6ce489  d033_joint_support_reference_20260930/test_joint_support.py
```
