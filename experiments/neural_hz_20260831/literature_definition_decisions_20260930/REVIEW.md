# 跨领域文献对 Neural HZ 定义研究的选择

需要借鉴 HZ 之外的研究，但已有综述不能反复当作新发现。本轮把论文机制与当前可执行关系组件作对照，结论是：优先研究共享输入和原相位下的关系如何跨普通混合权重、ReLU 与残差合流传递。差值、伴随量、共享因子身份、条件观察本身均有先例；创新还需要具体的组合规律和可承担的完整成本。

这是定向文献核验与研究建议，不是穷尽式新颖性检索、数值预注册或已完成的新抽象域。论文方法可以提供数学启发，不因此授权导入其中的搜索、分裂、反向细化或其他求解路径。

## 这次核验改变了什么判断

既有 [跨领域接口综述](../definition_first_20260928/d031_cross_domain_interfaces_20260930/REVIEW.md) 和 [研究议程](../definition_first_20260928/d034_cross_domain_research_agenda_20260930/REVIEW.md) 已包含抽象解释、析取规划、互补系统、差分验证、非凸可达集及知识编译。本轮不重建书单，也不把相同机制换名字。

当前 [前向关系组件结果](../definition_first_20260928/d044_relational_generator_20260930/RESULTS.md) 有实际数学进展：完整 3769 项测试通过，存在不要求激活支配的严格正控。但该证据不等于真实 CNN 普遍适用，也不证明胜过 PRIMA 全套方法或已完成定义创新。

这次最直接的定位修正来自 ReluDiff：不等权重下同时传播差值与伴随值已有明确先例。当前组件的潜在贡献应研究原相位条件信息如何在同一网络内部持续组合，而不是把差值加伴随本身当作创新。新增补读的重复 ReLU 完整二次约束研究则提示：关系语言的完整性与可计算性必须分别评价。

## 五项机制及迁移边界

### 抽象解释中的观察和归约

Patrick Cousot、Radhia Cousot、Laurent Mauborgne，2011，The Reduced Product of Abstract Domains and the Combination of Decision Procedures，§4 至 §4.1 讨论观察式归约；Theorem 4 指出反复两两归约一般仍达不到完整 reduced product 的最精确结果。[作者原文](https://www.di.ens.fr/~cousot/COUSOTpapers/publications.www/CousotCousotMauborgne-FoSSaCS11-LNCS6604.pdf)

项目推断：给 HZ 附加几个不交换信息的关系缓存并不够。需要规定观察在同一源赋值上的含义，以及 Conv、ReLU、Add 如何使用它们。载体写成 HZ 与关系域的组合是已有理论框架；贡献应落在可构造的关系语言和组合定理，不要求全网最佳抽象或免费闭包。

### 差分验证中的差值与伴随量

Brandon Paulsen、Jingbo Wang、Chao Wang，ReluDiff，ICSE 2020，§4.1 的仿射递推同时使用差值及一侧原值，§4.2 分析成对 ReLU。其完整流程还包含采样、梯度驱动的反向细化和输入分裂。[原文](https://arxiv.org/pdf/2001.03662)

项目推断：只借前向关系传播的数学，不移植完整工作流。对原输出 q、p，令 d=q-p，则 aq+bp=ad+(a+b)p；差值不能一般地替代伴随量。这个恒等式及思路不是新发现。当前候选仍须回答，带原相位的关系如何跨不同消费者和后续非线性保持有用，而不依赖两网络相似性或特殊权重等式。

### 多神经元与强混合整数建模

Müller 等，PRIMA，POPL 2022，联合多个神经元构造凸外包；其 §3.1 与 §5 的 Split Bound Lift 会拆分输入多面体。[原文](https://ggndpsngh.github.io/files/PRIMA.pdf) Anderson 等的强 MIP formulation，在本轮核对的 44 页版本 §5.2 Proposition 13 给出盒输入上单个仿射 ReLU 的 ideal formulation。[原文](https://optimization-online.org/wp-content/uploads/2018/11/6911.pdf)

项目推断：普通神经块和共同源约束是合理研究单位，不能只比较独立区间或弱 big M。借联合关系及强比较标准，不引入相位展开、解点驱动 cuts 或搜索。局部强 formulation 不是全网理想凸包；我们的有限正控也不能外推为超过这些完整算法。

### 非凸可达集中的依赖身份

Niklas Kochdumper、Matthias Althoff，Sparse Polynomial Zonotopes，本文核对 arXiv:1901.01780v2。Proposition 10 区分共享 dependent factor 身份的 exact addition 与丢失这种依赖的 Minkowski addition。[原文](https://arxiv.org/pdf/1901.01780v2)

项目推断：残差 Add 的数学语义必须使用同一输入及相位赋值；共享身份不是单纯指针去重。可借依赖管理原则，不以多项式域替换 HZ，不将近似误差冒认为原源，也不因此删除原 bits。该身份机制本身不能算我们的定义创新。

### 重复非线性的完整关系和成本

Sahel Vahedi Noori、Bin Hu、Geir Dullerud、Peter Seiler，A Complete Set of Quadratic Constraints for Repeated ReLU and Generalizations，本文核对 2024 年 arXiv:2407.06888v2。Theorem 1 用 2 的 n 次方个共正条件刻画完整 QC 集；Proposition 1 说明所用齐次二次形式不能单独排除 flipped ReLU；§V C 的计算采用共正性的充分松弛。[原文](https://arxiv.org/pdf/2407.06888v2)

项目推断：借鉴同一种激活在多个位置出现时的关系，而非只看独立标量界。完整 QC 不等于完整网络验证，且上述符号限制只针对该齐次 QC 语言，不能外推所有关系域。我们不整体转成 SDP，不把指数符号组合搬进运行时；原相位、方向和零点合法选择仍保留。

## 对定义研究的具体建议

精确 HZ 本来可以承载网络的分段仿射关系。研究问题不是宣称旧 HZ 没有表达力，而是能否定义适合神经计算的非凸关系对象及前向变换，在可承受成本内利用这些关系。

定义草案应明确区分两层：原连续因子、全部二元相位、EQ/LE 谓词、读出与 decoder 构成非凸语义；可认证的关系观察均解释在这一份共同赋值上。若新增观察均由原谓词蕴涵，具体化不变，但有限求界过程可能更有用。这种组织是已知 reduced product 思路，不是单凭写出集合式就成为新域。

接下来最值得检验的一个问题是：两个普通混合权重残差块合流时，能否用固定结构选择的有限观察，在不同原相位之间前向交换共同源信息，得到严格强于现有逐锚关系传播的物理输出约束？必须给出构造，而不是假设免费条件支持 oracle；必须保留全部原 bits，而不是对相位组合枚举。这里的“不同锚”指不同原相位身份，不允许把它们重命名成同一个 bit。

候选应先提供有适用范围的组合命题、普通结构正控与失败条件。若只是已有差分递推、已知有效行重排或可逆换元，就列入支撑组件；若仅在罕见权重恒等式下有效，不围绕它继续堆特例。可编译回 HZ 本身不是否决理由：应比较生成和利用关系的完整成本，而非要求逻辑上胜过 HZ 加完全相同事实。

当前同锚组件及区间系数草稿可作为这个问题的对照和支撑，不必丢弃，也不把完成其工程接入当作创新已经实现。真实适用性、CPU/GPU 传播、共享存储、谓词及证据规模、终端查询和具体输入重构均须计入。GPU 是设计目标，但本轮没有新 GPU 执行或性能证据。

## 本轮范围和保存状态

日期 2026-09-30，分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。本轮读取旧档、核验论文相关段落并记录建议，没有导入或执行候选、模型、LP/MILP、GPU、shadow 或回放。不是穷尽式系统综述，也未复现实验。ENS 论文主代理网页获取失败，由协作研究者通过作者原文只读核验；其余上述论文相关段落由主代理直接读取。

用户提出这轮文献问题时，前序已分配的区间系数工作仍有草稿写入。已要求在安全停点保留并停止扩展：d045_interval_relational_census_20260930 下仅有 THEORY.md、interval_relation.py、archive_worker.py 三份草稿，均未冻结、未导入或执行，不是通过的候选。新测试、监督器、完整预注册及接口和成本审查尚未完成。kernel 的一处构造前计费顺序、worker 早期认证异常的成本记录等仍待静态审查；无任何资源合格声明。不得据本记录启动或重跑实验，也不把暂停这一数值任务误记为整体 Goal paused；整体 Goal 仍 active。

正式 baseline 保持 1870/2413，即 1063 CERT 与 807 validated ADV；全部旧解和 13 家族不得回退。独立 E0 为 CIFAR100 25、TinyImageNet 36，共 61/400，不与 1870 相加。本轮正式增益为零。单一候选全量回放、invalid ADV=0、fail closed、默认关闭及原资源门均不变。

只新增本隔离综述；不修改旧档、生产默认、历史模型或结果，不 commit/push。工作区原有九个 tracked 修改仍为 3806 insertions、57 deletions。使用 write-page 文档技能将论文事实、项目推断、已有证据与未证假设分开保存，未发布外部 Page。

本轮只读核对的项目输入 SHA256：

```text
0fd93920413f63ad6b3b65849ea4d857c416c6bf4ab56263c63ff73416ebe80c  GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md
e44209a7dab24aa583f78869811a20837b62e57716e831e5d9ef4ab852cb8174  d031_cross_domain_interfaces_20260930/REVIEW.md
3974aa6862864e24047f5ba115e7d595f0b39445a00a89309f3edb8a75495316  d034_cross_domain_research_agenda_20260930/REVIEW.md
da575629ddd230a7e09d07b26c217568c9835efe1c8b655db326d5e74499c199  d044_relational_generator_20260930/RESULTS.md
```
