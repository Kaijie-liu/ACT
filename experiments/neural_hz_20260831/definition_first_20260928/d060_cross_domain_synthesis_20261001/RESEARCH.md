# 跨领域机制如何进入 Neural HZ 的定义

需要跨领域研究，但不应再把阅读论文的数量当作进展。当前要回答的是：共同输入产生的非凸关系，怎样经过混权卷积、下一层激活和残差汇合继续发挥作用，并以可承担的成本进入原终端查询。目标仍是改进域元素、具体化和抽象变换，不是只压缩现有 HZ 的矩阵。

原 HZ 本身已经能够精确表达 ReLU 与共同源；不能以“首次保留非凸性或共享依赖”作为创新。研究价值应来自适合神经结构的有限关系语言、可证明的传播或简化，以及有限预算下新的正确验证结论。全局 ideal 凸包不是新增要求。

## 阅读范围和已有工作

[前一跨领域综述](../d034_cross_domain_research_agenda_20260930/REVIEW.md)已涉及抽象解释、控制、多项式表示、知识编译及数据库消元；[后续机制综述](../d058_cross_domain_definition_review_20260930/RESEARCH.md)补充了精确遗忘与神经关系组合。这些不是本轮的新发现。本轮复核其中相关原文，补读稀疏多项式优化和多面体贪心机制，并把两个候选真正算到终端编码成本。

这是有范围的机制研究，不是完整系统性综述或新颖性认证。以下“迁移判断”均是项目推断，不是原论文已经证明的 Neural HZ 结果。

| 来源方向 | 已核对的机制 | 迁移判断 |
| --- | --- | --- |
| 抽象解释 | 多个组件通过具名观察交换关系，且具体化指向同一环境 | 主线。研究哪些原值、相位和条件观察必须成为域的语义对象，而非互不作用的缓存。 |
| 析取优化 | 同一个激活标签下保留输入分组贡献；提升量与投影行存在代价交换 | 主要对照。静态索引分组不是输入区域 split，但复制已知 formulation 不能算新域。 |
| 稀疏多项式优化 | 按变量耦合或单项式支持组织局部关系，显式处理重叠 | 辅助启发。先确定缺少哪项共同关系，再研究有限观察，不能先建不断升阶的矩层级。 |
| 互补系统与热带表示 | 改变原生语言可令 ReLU 的关系更直接 | 语义对照。必须同时面对混权仿射层和原相位身份；不能换名或整体替换 HZ。 |
| 约束消元与知识编译 | 完整作用域、共同见证、子关系复用与投影 | 支撑方法。编译、规范化、填充及终端展开均需计费，表达式小不等于查询便宜。 |
| 组合优化 | 同一排序锥内的多个线性目标可以共享一个贪心最优点 | 本轮得到有限适用边界；当前显式编码更贵，尚不进入实现。 |

## 优先借鉴关系交换与条件贡献

Cousot、Cousot、Mauborgne 的 [The Reduced Product of Abstract Domains and the Combination of Decision Procedures](https://pcousot.github.io/publications/CousotCousotMauborgne-FoSSaCS11-LNCS6604-proofs.pdf)，FoSSaCS 2011，§4.1 Definition 4，说明具名观察配合归约可增强组件间的信息交换。项目应借鉴的是共享语义及交换契约，而不是声称组合两个现有组件本身就有新颖性。

Tsay 等的 [Partition-Based Formulations for Mixed-Integer Optimization of Trained ReLU Neural Networks](https://arxiv.org/pdf/2102.04373)，2021，§3及Proposition 1，给出分组贡献的提升表述及其投影。辅助量数量线性增长时，等价的无辅助表述可以需要指数数量的行。这是“变量更少不一定更好”的直接对照；不移植其优化式界收紧或搜索流程。

迁移方向是：保留原 HZ 的连续源、全部原 bits、EQ/LE、frame 和 decoder，把有限的共同源条件关系作为同一赋值上的语义对象。需要证明该关系怎样被一个具体后继消费，不能只给它一个新名字。对照包含原 native 约束与适用的已知分组及多神经元关系；得到完全相同行时，不宣称逻辑精度不同。

## 稀疏优化提醒我们保留什么而不是直接换求解器

Newton、Papachristodoulou 的 [Sparse Polynomial Optimisation for Neural Network Verification](https://arxiv.org/pdf/2202.02241)，§4.3至4.6，区分单项式稀疏性和变量耦合稀疏性。尤其约束变量共同出现也会形成耦合，不能只看目标函数或单个权重矩阵。其方法采用 SOS 和 SDP，不直接纳入本项目路径。

Lasserre 的 [Convergent SDP Relaxations in Polynomial Optimization with Sparsity](https://optimization-online.org/wp-content/uploads/2006/04/1367.pdf)，2006，Assumptions 3.1和3.2、Theorem 3.6，为特定有界性和 running intersection 条件下的稀疏层级给出渐近收敛。这个结论不是“有限阶局部矩加均值一致就精确”，也不是“卷积天然有廉价的小 separator”。

项目推断：若下一门确实需要 beta*q 这样的共同条件量，就研究它和原源之间的最小一致接口。不能用互相独立的局部见证代替一个共同输入，也不预设二阶足够或有限接口对任意深度闭合。[已有跨层反例](../d059_cross_layer_interface_20261001/THEORY.md)已说明局部凸包只共享坐标均值仍可能失配。这支持定向研究，而不授权引入 SDP、对偶救援或提高矩阶直到通过。

## 激活原生语言和符号编译作为对照

Aydinoglu 等的 [Stability Analysis of Complementarity Systems with Neural Network Controllers](https://dair.seas.upenn.edu/assets/pdf/Aydinoglu2021.pdf)，HSCC 2021，§3.1 Lemmas 1和2，把 ReLU 及多层网络写成互补系统。Goubault 等的 [Static Analysis of ReLU Neural Networks with Tropical Polyhedra](https://arxiv.org/pdf/2108.00893)，§3和4，则提供 max plus 语言下的神经抽象对照。迁移启发是原生关系语言的选择；限制是混权仿射近似、表示转换及原 bits 不能被略去。我们不照搬输入细分，也不整体改成 LCP 或热带域。

Darwiche 的 [SDD](https://www.ijcai.org/Proceedings/11/Papers/143.pdf)，2011，§2至5，给出固定作用域树上的布尔分解与共享；Van den Broeck、Darwiche 的 [On the Role of Canonicity in Bottom-up Knowledge Compilation](https://arxiv.org/pdf/1404.4089)，2014，Theorems 1和3，说明一般情况下要求约化规范形可带来指数膨胀。这一点此前已记录，本轮只是再次确认。

静态符号 DAG 含析取不等于运行时 split，但按相位赋值枚举、conditioning 或分支求解来构造它不在当前授权范围。两个叶谓词分别可满足，也不代表其共同连续源可满足；例如 x<=0 与 x>=1。仅有布尔 Apply 的复杂度界不能支付连续可行性、矩阵展开或 decoder 的成本。

Dechter 的 [Bucket Elimination](https://ics.uci.edu/~dechter/publications/r76A.pdf)，1999，§2.3，也明确连续 Fourier 消元不能仅靠 induced width 得到所需复杂度界。因此精确消元仍作为配套方法，不重新变成整个研究主线。

## 本轮从文献走到两个可检查结论

第一，盒切片上的多个上侧消费者若具有共同系数排序，可以共享同一个贪心见证；系数可以稠密、混号且满秩。它属于已知多面体贪心理论的应用，并非新定理的优先权主张。主源为 Bach 的 [Learning with Submodular Functions](https://www.di.ens.fr/~fbach/2200000039-Bach-Vol6-MAL-039.pdf)，§3.2 Proposition 3.2。我们给出了神经条件量接口、失配反例和完整显式编码账单。

第二，两层标量 ReLU 的完整旧 LP 可以精确消去中间连续值，并保留原两位相位。但普通非恒定旁路下，这套直接编码增加而非降低 nnz。它属于经典投影，不是新域。两项推导和限定均存入 [数学附录](THEORY.md)。

当前判断：不实现这两个已经显示成本劣势的直接版本；这不否定其他关系语言、已证冗余下的消元或精度与成本的合理折中。也不把“必须降低 nnz”加成全部能力候选的新硬门。更强关系可以有成本，只须完整计费并通过原能力晋级要求。

## 对真实网络有针对性的下一问题

[既有完整 large 局部证据](../d025_interval_capacity_20260930/RESULTS.md)中，320个接收行有314行容量系数严格收紧，两项外包跨零接收行均有收紧。这证明共同源关系在真实普通结构中有作用，但不证明 native 非冗余、性质改善或新增解；该三模型运行整体仍失败，Tiny 未进入。[真实共享源记录](../d054_shared_phase_cycles_20260930/RECORD.md)还确认下一混权消费者存在，但其中一父门稳定，不能算未决多相位收益。

据此，优先问题固定为普通 Conv→ReLU→混权 Conv→ReLU 及残差组合中的共同关系接口，先回答三件事：

1. 接口是否真正使用共同源，能区分独立幅度相同而相关性不同的结构？
2. 哪个具体后继需要该关系，原 native 与已有差分或容量规则为何尚未提供相同信息？不要求任意深度无损闭包或全局 ideal。
3. 原相位、全部消费者、可靠参数、传播、终端行及 nnz、输入重构和 GPU 完整代价分别是什么？数学可批处理不等于实测 GPU 加速。

只有先形成有用、健全、可反证的定义与结构命题，才另行预注册默认关闭的实现。不会为复现论文而新开禁止的求解路径，不会重跑消耗版本或事后调大预算。后续仍是数学、真实同结构、shadow、逐家族、同候选完整2413回放；独立E0另作400回放。

## 执行范围和记账

2026-10-01，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。配置为 paper_only、primary_literature、read_archived_documents；没有新运行依赖、模型执行、求解器、GPU、数值实验或生产编辑。三项只读子任务分别研究共同贪心、编译语义及真实证据优先级，根代理复核原文与推导。

本轮只创建当前隔离目录中的综述、数学附录和校验清单；不修改冻结文件、Goal正文或用户已有代码，不commit/push。原权威目标仍active。正式1870/2413（1063 CERT+807 validated ADV）、独立CIFAR100 25与TinyImageNet36（61/400）不变，formal_gain=0，未声称当前候选已经完成保旧回放。

使用 pages:write-page 将论文事实、项目迁移、解析结论与未证收益分开记录；只保存并读回本地文档，没有发布外部Page。不是完整的新颖性审查，尚无可宣称PLDI级贡献或满分保证。
