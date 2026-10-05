# 跨领域文献对 Neural HZ 定义研究的新增判断

需要跨领域研究，但本轮不再重复建立一份大书单。研究问题是：普通混权卷积、ReLU 和残差中的共同源关系，能否用有限、可组合的非凸关系语言保存，并在原终端查询边界内产生实际收益。借鉴其他域的机制，不等于把 HZ 替换为那个域。

本轮核对既有文档并补读原论文，最值得继续检验的是多读出及其差值的共同赋值关系。尚未得到新域定义、结构定理或真实网络收益。以下明确区分论文事实、旧成果和待证迁移；不是系统性综述或新颖性认证。

## 已有研究不能重新记作发现

[已有跨领域综述](../d058_cross_domain_definition_review_20260930/RESEARCH.md)及[机制综合](../d060_cross_domain_synthesis_20261001/RESEARCH.md)已经讨论 reduced product、析取规划、互补系统、热带几何、知识编译及精确消元。[更早的研究议程](../d034_cross_domain_research_agenda_20260930/REVIEW.md)还涉及 Taylor 模型、Bernstein 求界与数据库查询。因此“开始跳出 HZ”不是今天才发生的进展。

[有界关系传播](../d042_bounded_relational_transfer_20260930/THEORY.md)已有差值加伴随值的混权传播式，以及原相位下的差值区间。不能把下面文献中的仿射差分公式再作为新算法。[共同负项研究](../d069_joint_negative_support_20261001/RECORD.md)改善了共同相位端点的生成方法，但尚未形成新的关系载体；支持算法和域定义必须分别报告。

本轮到达时，相邻共同负项组件目录只有 PREREG.md 和 THEORY.md。没有启动其实现、导入、测试或运行目录，也没有修改这两份文件。本轮按用户最新请求做文献研究，不把它记成组件完成。

## 差分验证提供共同误差身份的直接对照

Teuber、Kern、Janzen、Beckert，TACAS 2025，[Revisiting Differential Verification](https://arxiv.org/pdf/2410.20207)，§4 Definition 4、Lemmas 1–2：两个路径及其差值使用一个联合生成元赋值，差分表示复用各路径产生的近似生成元身份，而不只是保存三个独立范围。§5 的完整流程还包含输入切分，不能移植；其凸 zonotope 载体也不能整体替换原 HZ。

本项目的迁移假设是：对原网络中结构固定、可对齐的读出，显式保存多个读出、差值和激活误差之间的共同关系，研究它们能否经过下一混权块继续被利用。这里的误差若定义为原读出之差，就必须绑定同一个原输入与原相位，不能变成任意独立噪声。原 HZ 已有这种身份语义，候选的任务是使有限成本的传播和查询真正利用它，而不是声称首次发现依赖。

这不是把两网络验证直接套给相邻通道。普通残差两支不一定逐层对齐、权重接近，也不能给 identity shortcut 虚构一个 ReLU 来对齐。旧差分传播本身已知，新增研究应比较保留联合关系和只保留条件端点的差别；如果结果仍被现有规则覆盖，就记录负结论，不再为相同公式建立组件。

## 其他领域分别提供定义原则和必要对照

**抽象解释。** Cousot、Cousot、Mauborgne，[The Reduced Product of Abstract Domains and the Combination of Decision Procedures](https://pcousot.github.io/publications/CousotCousotMauborgne-FoSSaCS11-LNCS6604-proofs.pdf)，FoSSaCS 2011，§3–4，提供共同语义上的观察与信息交换。项目推断：载体必须定义哪些共享量能参与后续变换；把 HZ 和几个不交换关系的缓存放在一起不够。Reduced product 本身是先行工作，不能作为我们的新颖性。

**析取规划。** Tsay 等，[Partition-Based Formulations](https://arxiv.org/pdf/2102.04373)，NeurIPS 2021，§3及Propositions 1–3，用同一原激活标签下的分组连续贡献连接不同强度的表述。借鉴的是共同贡献接口；按索引静态分组并不等于输入或相位 split。但不能连同 OBBT、搜索或动态救援引入，也不能忽略消去辅助量后的行数增长。

**互补系统。** Aydinoglu 等，[Stability Analysis of Complementarity Systems with Neural Network Controllers](https://arxiv.org/pdf/2011.07626)，HSCC 2021，§3.1，把 ReLU 写成非负输出、非负缺口及二者互补。它启发对称处理输出与负缺口，但这套等价式已知；改写为 LCP 不是创新，也不能省掉原相位，尤其激活零点的两个合法原标签。

**热带几何。** Goubault 等，[Static Analysis of ReLU Neural Networks with Tropical Polyhedra](https://arxiv.org/pdf/2108.00893)，§1及§3–4，展示了选择原生数学语言可以使 ReLU 精确，而普通线性映射反而需要近似。项目推断：不能只让一个算子变简单，必须同时分析混权 Conv 和残差。这里借代数设计思想，不整体改成热带域，也不采用其输入细分。

**带类型符号的集合。** Combastel，[Functional Sets with Typed Symbols](https://arxiv.org/pdf/2009.07387)，§3.3–3.6，已经使用连续、signed、Boolean 符号、唯一身份和函数像语义。它是直接的新颖性对照：“连续因子加二元因子加函数表达式”以及惰性符号复用本身都不是新成果。候选须提供神经结构专属的关系变换或精确简化定理，并支付展开与查询成本。

**关系数据库。** [FAQ](https://arxiv.org/pdf/1504.04044) 区分保留的自由变量和被聚合变量；[Factorised Representations of Query Results](https://arxiv.org/pdf/1104.0867)，§7.3，研究输出关系的重复度与结构。项目推断：全部原 bits 应留在语义接口，不可借数据库聚合删掉；共享 DAG 不自动带来廉价连续查询。普通重叠卷积的作用域也未必嵌套。只借作用域和复用原则，不移植枚举、回溯或把连续约束当免费有限表。

## 近期文献的排除判断

检索同时发现 2026年9月28日提交的两篇预印本，仅核对作者 arXiv 摘要，未据此认证其理论或成绩。[CLAD](https://arxiv.org/abs/2609.34628) 描述以投影 primal-dual 优化处理复合输入约束；这不是当前允许的定义创新路径。[ZonoGPT](https://arxiv.org/abs/2609.34457) 描述结构化 zonotope、生成元削减和块级变换；可将块级设计列为后续对照，但不能把其凸域及削减机制直接用于我们的非凸语义。本轮不实现二者、不照抄报告的速度或覆盖率，也不由这次有限检索声称已覆盖全部最新文献。

## 综述应导向的下一项可反证问题

优先结构仍是普通 Conv → ReLU → 混权 Conv → ReLU 及残差，不追逐精确相等的特殊门或极端数值。候选问题为：在原非凸 HZ 和全部原相位上，保存一组被多个后继共同使用的条件连续关系，是否比逐读出的条件端点更有用，而且能够在既定预算内前向组合。

需要回答以下问题后才值得开启另一项实现，而不是读到一种表示便创建新组件：

1. 写出域元素及具体化。原连续源、原 bits、EQ/LE、共享 frame 和 decoder 保留；所有局部关系必须由同一个赋值同时解释。证明与原 HZ 的嵌入或精确投影，不能仅画关系图。
2. 给出至少跨一个后继混权加激活块的变换。单独增加差值变量、已知互补式或相同有效行不是定义贡献；与 HZ 加同样关系的逻辑强度相同就明确承认。
3. 比较完整 native 约束、旧差分与条件端点，以及适用的强分组或多神经元方法。若用一个被排除的点说明精度，它只能是旧查询松弛中的假点，不能删除原 HZ 的真实可行输入。未证明任意深度闭包不等于失败，也不新增“必须全局 ideal”的要求。
4. 分别计算关系生成、传播、辅助变量、谓词 nnz、证据、GPU传输与计算、终端降级和输入重构。更强关系可能更贵；是否值得由原能力门决定，不重新要求所有能力候选必须降低 nnz 或翻倍提速。

上述是研究建议，不是已经成立的定理或新的数值预注册。没有证明普通 CIFAR/Tiny 结构必然出现净收益，更没有承诺满分。若收益仅来自现有差分传播、已有 cuts 或无成本假设，应保留否定结果并换问题。

GPU需单独分账。[cuPDLP.jl](https://arxiv.org/pdf/2311.12180)，§3.3及§4.1，讨论 GPU 连续 LP 迭代，不提供整数 HZ 语义、完整 MILP 或严格 CERT 的自动保证。批量构造支持关系和终端求解是两项不同工作；本轮未改 GPU 资源限制、未初始化设备、未申请或推定新权限。

## 本轮范围与 provenance

2026-10-01，分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。配置为 read_only_archive、primary_literature、paper_only；无候选运行依赖、模型执行、数值试验、LP/MILP、GPU或全量回放。三路只读复核分别检查优化与互补、关系抽象与差分验证、数据库作用域与GPU边界；根代理核对项目原档和相关原文。

只新增本目录的研究文档与校验清单。生产代码、旧实验、历史模型及结果不改，未 commit/push，既有九个 tracked 修改保持 3806 insertions / 57 deletions。校验范围限于清单列出的文档，不声称重新核验全部历史数据。

正式基线仍为 1870/2413（1063 CERT + 807 validated ADV）；独立 E0 为 CIFAR100 25、TinyImageNet 36，共61/400，不相加。本轮 formal gain=0，也没有新候选已完成保旧回放的声明。Goal仍 active。

write-page 技能用于在文中区分来源事实、迁移假设、旧成果和未证结论；仅保存本地研究记录，不创建外部 Page。全文读回检查，不宣称外部页面渲染资格。
