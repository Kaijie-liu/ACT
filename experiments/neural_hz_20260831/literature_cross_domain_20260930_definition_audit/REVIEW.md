# 跨领域研究如何改变 Neural HZ 的定义设计

需要借鉴 HZ 之外的研究。对本项目，综述应回答的是共同输入和原相位之间的关系如何经过普通神经算子继续被利用，而不是再罗列一批验证器。优先结合抽象解释的域设计、析取优化的带标签关系、数据库的共同见证，以及控制理论的重复非线性关系；保持原 HZ 的非凸语义和全部验证边界。

本文回应项目负责人关于跨领域启发的要求，承接[已有组合综述](../literature_cross_domain_20260930_composition/REVIEW.md)。这些文献方向此前已出现，本次不将重读算作新发现。新增交付是校准定义创新的判断标准、审查当前关系候选的缺项，并收敛下一项可检验问题。没有新数值实验或正式收益。

## 现有证据要求改变研究单位

[最近完整存档审查](../definition_first_20260928/d038_descendant_census_20260930/RESULTS.md)覆盖 102400 条真实连接。逐边判据留下的 267 条未决边均属于严格 active 的源；在保留这些稳定相位事实的比较系统内，余下行也全部冗余。实际生产相位列绑定尚未验证，结论只针对该存档块及指定比较系统，不外推整个 CIFAR100 或 TinyImageNet。

因此，继续铺开同一廉价逐边规则缺乏该块上的能力依据。值得研究的单位是普通混合卷积、ReLU、下一次混合卷积与 ReLU，以及残差汇合中的共同关系。不能仅因一个局部公式更紧就推断整网验证更强。

精确 HZ 本来就能表达丰富的分段仿射关系。新 Neural HZ 不必表达旧 HZ 无法表达的集合；能编译回 HZ 也不自动否定贡献。真正需要证明的是新的有限关系语言与可组合变换，或者在相同完整预算下可利用更多关系的结构性结果。换矩阵布局、可逆换元、将相同行改名，则不足以单独成立这种贡献。

## 四类外部机制及迁移边界

### 抽象解释提供从算子反推域的设计方法

论文事实：Giacobazzi、Ranzato、Scozzari 的 Making Abstract Interpretations Complete 在 §1.2 和 §1.4 讨论相对于指定语义算子与观察性质的完备性，并在相应前提下研究域的扩展或限制。[作者全文](https://www.sci.unich.it/~scozzari/paper/JACM00.pdf)

项目推断：应先问下一次 ReLU 或 Add 需要什么关系，再定义域元素和传递规则。这里借的是设计方法，不宣称其 complete shell 定理已适用于当前 Neural HZ，也不追加全网完备、最佳抽象或完整格要求，更不承诺满分。

### 析取优化提醒我们保留原相位关联

论文事实：Vielma 的 Small and Strong Formulations for Unions of Convex Sets from the Cayley Embedding，§6 Definition 5，区分只要求输出投影达到凸包的 sharp 和要求提升标签空间达到相应凸包的 ideal formulation。[作者全文](https://juan-pablo-vielma.github.io/publications/Small-and-Strong-Formulations.pdf)

项目推断：下游仍使用原相位时，当前输出集合紧并不足以说明跨块关系保留得好。域接口需要明确记录原相位和哪些幅值共享同一赋值。此处不要求全网 ideal，也不将论文的支持函数条件当成免费求界程序；perspective 或已有 guard 编码本身不算我们的创新。

### 数据库的联合投影提醒我们不能拆开共同见证

论文事实：Olteanu 和 Závodný 研究有限查询结果的因子化及投影后的依赖。独立复核定位到 §3 的联合查询投影与分别限制查询的比较；本次根代理直接访问作者 PDF 失败，来源核验由并行复核完成，不声称根代理本次完整阅读该文。[作者全文](https://www.cs.ox.ac.uk/dan.olteanu/papers/oz-tods15.pdf)

项目推断：残差两侧应满足 `exists s: R1(s,u) and R2(s,v)`，一般不能改为分别寻找 `s1`、`s2`。连续辅助量投影后，剩余消费者也可能仍有关联。这是原 HZ 已具备的共享 latent 原则；研究点是有限摘要如何保持并消费这种关联，不是把谓词换成 DAG 存储。有限表格的复杂度定理不能直接移植到连续非凸集合，也不引入相位表或枚举。

### 控制理论提供跨激活关系而非独立区间

论文事实：Fazlyab、Morari、Pappas 在 §III C.3 式 15 利用重复非线性的增量斜率。对 ReLU，记两个输入之差为 delta、输出之差为 d，有 `d*(d-delta)<=0`。[作者全文](https://www.georgejpappas.org/wp-content/uploads/2022/01/Safety_Verification_and_Robustness_Analysis_of_Neural_Networks_via_Quadratic_Constraints_and_Semidefinite_Programming.pdf)

项目推断：关系语言可研究同一种激活在多个位置的联系，而不只保存每门独立范围。但二次关系不能直接假装已被现有线性终端支持；采用任何派生关系都需单独证明健全性和完整成本。本项目不迁入 SDP、对偶救援或全网两两关系展开。

## 两个必须保留的对照

PRIMA 已研究多神经元联合凸抽象，§3 与 §5 还明确给出 Split Bound Lift 构造。因此“把多个神经元一起处理”不是新的研究结论；它是精度和新颖性的比较对象，不整体移植其分裂或 refinement cascade。[PRIMA 原文](https://files.sri.inf.ethz.ch/website/papers/mueller2021precise.pdf)

Tropical polyhedra 的神经分析使 ReLU 在其代数中精确，但普通线性映射仍带来抽象损失。这提醒我们同时核算混合权重层和激活，而不是只把某个算子变简单。[原文 §1、§3](https://arxiv.org/pdf/2108.00893)

上述机制已有明确先例。我们的候选必须说明新的组合定理或可构造方法在哪里，而不能把文献机制在 HZ 中重写一次就称为新抽象域。

## 当前候选还欠哪些定义义务

[共同关断面候选](../definition_first_20260928/d039_shared_off_face_20260930/THEORY.md)给出了同一原相位关闭时一组观察同时为零的关系。它已有普通混合权重正控及部分前向闭包，但目前仍是带证书的关系片段和有效行生成规则，不是已完成的新 Neural HZ。

最小下一项研究是：为常见两跳混合非线性与残差组合，定义有限的共同源关系接口，明确什么可传递、何时失效，以及怎样健全地不采用未获证关系。需要三类交付：

1. 域与变换契约。明确原连续盒、原二元域、EQ/LE、共同赋值、关系观察的生成与生命周期；逐算子给出 `F(gamma(d)) subseteq gamma(Fsharp(d))` 或范围明确的精确性定理。原连续源、全部原 bits 与 decoder 保留。
2. 有范围的组合证明。现有规则对同锚线性组合、符号可证旁路有部分闭包。应检验这些关系经过普通后续非线性后能否改善真实输出判断；跨零旁路尚未覆盖，不能暗建免费旁路网络或使用精确支持 oracle。未覆盖时不加入未证关系，保留原 HZ 语义。
3. 公平的完整成本比较。计入源证书、观察数量与生命周期、latent 展开、终端转换与求解、输入重构。与 HZ 加完全相同行相比，逻辑精度相同；潜在贡献只能来自新的可构造传递或成本定理。真正晋级仍需普通真实结构净收益及原全量保旧流程。

这些是对现有目标的落实，不是额外设置“所有网络固定宽度闭包”或“必须超过精确 HZ 表达力”的门。接下来应产出可反驳的定义和定理，而不是继续堆相似综述或测试框架。

## 当前定义片段的只读勘误

独立审查指出，旧候选 gamma 公式仅写满足 P_H，而前文把 P_H 描述为 EQ/LE 谓词；为了保证空关系集合精确恢复原 HZ，还必须显式保留原 latent 域。下一份候选应写为：

```text
gamma(H,Z) = { F_H(xi,b) : xi in B_H, b in D_H,
              P_H(xi,b),
              (1-alpha)*E(xi,b)=0 for every (alpha,E) in Z }.
```

这里 B_H 是原连续因子盒，D_H 是原二元域；所有读出均使用同一赋值。若原 bits 使用 {-1,1}，alpha 的 {0,1} 重编码必须定义并计费。任何讨论松弛的段落也应显式包含 `0<=alpha<=1`。此处只记录定义缺项，不修改旧 THEORY、冻结文件或历史结果，也不借此宣称新候选已验证。

## 存档范围与来源身份

日期 2026-09-30；分支 redu-hz；commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。配置为一手论文相关章节核对、既有项目证据只读对照和两项独立文献复核，没有数值候选执行。此记录不是穷尽文献综述、论文新颖性认证或实验资格。

正式 baseline 仍为 1870/2413，即 1063 CERT 和 807 validated ADV；必须保住每个旧解和全部 13 家族。独立 E0 为 CIFAR100 25、TinyImageNet 36，共 61/400，不与正式分数相加。本次 formal gain=0。既有 default off、预注册、shadow、家族与全量回放门不变。

保留连续因子、原二元非凸语义、EQ/LE、共享身份、具体输入重构和 fail closed。禁止整体替换为 CZ、Zonotope、box，禁止按身份或 LP 状态选路径，禁止 attack/PGD、BaB、输入或相位 split、backward/dual rescue。本次没有放宽任何权限、资源或验证要求。

只新增本隔离目录及校验清单，不改历史模型、数据、日志、旧文档、生产代码或 Goal。未 commit/push。原九个 tracked 修改仍为 3806 insertions、57 deletions。按 pages:write-page 区分论文事实、迁移推断、已证结果与未证义务，保存本地 Markdown，未发布外部 Page。

只读输入 SHA256：

```text
0fd93920413f63ad6b3b65849ea4d857c416c6bf4ab56263c63ff73416ebe80c  GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md
5e1809b7984b7b7d3b5c13476e576a614ce5a934eae2e9e3079b211a41b7b49f  d038_descendant_census_20260930/RESULTS.md
a45dcfeddeff8b64535f668f4bea6975b0fc79e8a949bac5755075636bde9ad3  d039_shared_off_face_20260930/THEORY.md
f6b1a5363240281a811fdc7a0c8d1de6cae7ce0659247f86f1fbd8626ce2c46e  literature_cross_domain_20260930_composition/REVIEW.md
```
