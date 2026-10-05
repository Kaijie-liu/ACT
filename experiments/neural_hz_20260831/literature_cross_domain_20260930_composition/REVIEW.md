# 从跨领域文献确定 Neural-HZ 的组合问题

需要继续借鉴 HZ 之外的研究，但本轮的交付不是更多论文名称，而是两个可用于筛选定义的准则：不能只看去掉相位标签后的输出投影；不能把共享源上的联合关系换成两个独立存在量词。优先研究普通混合 Conv、ReLU 和残差汇合中的共同赋值接口。它仍是研究假设，不是已经完成的新抽象域。

本文面向项目负责人，承接用户关于跨领域启发的要求。已只读核对[此前综述](../definition_first_20260928/d034_cross_domain_research_agenda_20260930/REVIEW.md)和[跨层幅值检查点](../definition_first_20260928/d036_descendant_amplitude_20260930/CHECKPOINT.md)。此前已覆盖 reduced product、perspective、控制理论、混合符号和知识编译，本次不将重读记为新发现，也不开展新的数值实验。

## 析取优化提示必须区分两种紧度

论文事实：Vielma 的《Small and Strong Formulations for Unions of Convex Sets from the Cayley Embedding》§6 Definition 5 区分 sharp 与 ideal。前者只要求连续松弛投影到原坐标后等于原集合的凸包；后者要求连同析取标签的提升空间也达到该文定义的凸包。Theorem 8 分别给出条件，两者不是同一要求。[作者全文](https://juan-pablo-vielma.github.io/publications/Small-and-Strong-Formulations.pdf)

项目推断：若下一个神经块仍依赖原相位，仅证明当前输出投影紧是不够的。需要说明源值、原相位和后代激活之间保留了什么联合关系。这是解释已有跨层反例的工具，不是该论文已经证明我们的 Neural-HZ 构造，也不要求全网络 ideal。其支持函数条件不能充当免费求界算法。

已知组件的定位：Günlük 和 Linderoth 的 perspective 结果说明如何利用 indicator 所控制的零面构造强 formulation；§2.3 也给出消去辅助量后需要指数规模不等式的例子。已有跨层幅值行应作为这个已知机制的神经结构实例，不能仅凭少变量便称作新域。[作者稿 §2.3、§3、§4.2](https://optimization-online.org/wp-content/uploads/2008/06/2014.pdf)

## 数据库理论提示消去共享源后仍有耦合

论文事实：Olteanu 和 Závodný 的《Size Bounds for Factorised Representations of Query Results》§3、Proposition 3.4 说明，投影掉连接属性后，剩余属性可能仍然相互依赖。§4 则区分因子化表达和共享子表达式的表示。[作者全文，TODS 2015](https://www.cs.ox.ac.uk/dan.olteanu/papers/oz-tods15.pdf)

对神经残差的迁移推断是：正确的共同源语义为 `exists s: R1(s,u) and R2(s,v)`，一般不能换成 `(exists s1: R1(s1,u)) and (exists s2: R2(s2,v))`。这里的 s 包含同一原连续赋值及相关原相位身份；只对齐两侧区间或平均值不足以替代共同赋值。

这与原 HZ 的共享 latent 原则一致，不能声称精确 HZ 原来没有该语义。值得研究的是前向摘要或连续辅助量精确投影后，如何显式保留必要耦合并供下一个算子使用。有限数据库的大小界不直接适用于连续非凸集合；本项目也不采用完整相位表、相位枚举或输入分裂。

本轮根代理读取 ICDT 2012 链接失败，改读上述作者的 2015 全文并核对对应段落；不把两版定理编号混用。

## 抽象解释和控制理论提供关系构件

论文事实：Cousot、Cousot、Mauborgne 的 reduced product 研究通过组件交换观察进行信息归约，其出发点是多个抽象描述同一个具体状态。[作者论文页](https://www.di.ens.fr/~cousot/COUSOTpapers/FoSSaCS-11.shtml)

项目推断：Neural-HZ 应先定义一份共同赋值如何解释所有原值、相位和新增观察，再定义 Conv、ReLU、Add 的传递规则。只把若干缓存并列在 HZ 旁边，或者只实现已知两两归约，不能自动成立为定义创新；局部一致也不等于全局精确。

论文事实：Fazlyab 等在 §III-C3 式 15 利用重复非线性的增量斜率。对 ReLU，令 d 为两个输出之差、delta 为对应输入之差，则有 `d*(d-delta) <= 0`。[作者全文](https://www.georgejpappas.org/wp-content/uploads/2022/01/Safety_Verification_and_Robustness_Analysis_of_Neural_Networks_via_Quadratic_Constraints_and_Semidefinite_Programming.pdf)

项目推断：可以借鉴跨神经元的关系语言，而不只建立独立标量界。但不能直接把二次关系当成现有线性终端已支持的谓词，也不移植 SDP、对偶救援或全网两两关系。任何采用都需要符合现有边界的健全构造和完整成本；不能再次以差分替换原绝对幅值及原 bits。

## 什么才值得进入下一项候选

优先问题是：能否为普通两跳混合残差结构定义有限的共同赋值关系接口，让原相位与后代激活的关联经过下一次非线性仍可被利用，同时控制关系增长和查询成本？“关系接口”只是待证明的设计问题，不是新名称或创新认证。

下一份数学候选应给出：

1. 域元素、具体化、原 HZ 嵌入和同一赋值下的见证重构；保留连续因子、全部原 bits、EQ/LE 和共享身份。
2. 常见 Affine/Conv、ReLU、Add 的显式前向规则，以及精确对应或健全外包的组合证明。
3. 普通结构上的可反驳比较，至少区别于原 HZ 加同一批有效约束、强单门 formulation 和既有多神经元关系；不以罕见恒等式或极端数值为主线。
4. 辅助量、行数、nnz、系数位宽、传播、终端转换与求解、输入重构的总成本；GPU 是实现维度，不代替定义创新。

PRIMA 已有多神经元凸抽象，Tsay 等已有输入求和分组的强 formulation，所以“多个神经元一起处理”或“分组后更紧”本身不是新颖性。它们是比较对象，不是要整体替换原非凸域；不连同其中的 split、搜索、OBBT 或 refinement cascade 一并移植。[PRIMA 原文](https://files.sri.inf.ethz.ch/website/papers/mueller2021precise.pdf)，[Tsay 等原文](https://papers.nips.cc/paper/2021/file/17f98ddf040204eda0af36a108cbdea4-Paper.pdf)

若得到的只是可逆换元、已有 cuts 的重排或成本被推迟到终端，应作为支撑组件或负结论存档，不继续包装为新域。也不增加“所有网络都必须全局 ideal”这样的新门。

## 状态与边界

日期 2026-09-30；分支 redu-hz；commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。配置为一手文献相关章节核对、历史研究只读比较与纸面机制分析。本轮新增本目录文档及校验清单，没有新增候选执行、数值实验或正式收益。此前准备任务交付了相邻 D037 的 amplitude.py、run_reference.py、archive_worker.py 三个草稿；已停止继续实现，它们未冻结、未执行，结果目录不存在，不具备测试通过或正式资格。

正式 baseline 仍为 1870/2413，即 1063 CERT 与 807 validated ADV，必须保住全部旧解及 13 家族。独立 E0 仍为 CIFAR100 25、TinyImageNet 36，共 61/400，不与正式分数相加。本轮 formal gain=0，不承诺满分或 PLDI 新颖性。

禁止退化为 CZ、Zonotope、纯区间或整体凸域替换；禁止原 bit 删除/pivot、PGD/attack、BaB、输入/相位 split、backward/dual rescue 及身份/LP 状态菜单。候选仍默认关闭，数学、真实同结构、shadow、逐家族及全量回放的原门不变。研究建议不改变 Goal 或授权范围。

生产代码、历史模型/日志/结果及冻结资料未改，未 commit/push。原九个 tracked 修改仍为 3806 insertions、57 deletions。按 pages:write-page 区分原文结论、迁移推断和未证假设；只保存本地 Markdown，未发布外部 Page。

本轮只读输入的 SHA256：

```text
0fd93920413f63ad6b3b65849ea4d857c416c6bf4ab56263c63ff73416ebe80c  GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md
3974aa6862864e24047f5ba115e7d595f0b39445a00a89309f3edb8a75495316  d034_cross_domain_research_agenda_20260930/REVIEW.md
66e08405734093cb72d13ed1089ab4bdbdcda680175437b3c4aef790179db266  d036_descendant_amplitude_20260930/CHECKPOINT.md
```
