# 跨领域文献对 Neural HZ 定义与组合规则的启发

需要借鉴 HZ 之外的研究，但目标不是再扩大书单，而是回答一个具体问题：原始输入、激活相位和连续幅值的共同关系，怎样在混权卷积、下一次激活与残差汇合中继续被利用。综述应产出可检验的定义假设，而不是把已有符号方法、共享 ID 或有效不等式重新命名。

本记录回应用户提出的跨领域研究建议。它是有针对性的原文复核与研究取舍，不是穷尽式系统综述、新颖性认证或数值实验。此前的[研究聚焦](../d073_cross_domain_research_focus_20261001/RESEARCH.md)、[跨领域接口](../d031_cross_domain_interfaces_20260930/REVIEW.md)和[跨相位共同源研究](../d035_cross_phase_source_20260930/THEORY.md)保持只读，不把重读计为新成果。

## 已有研究可以借什么

| 研究方向与原文定位 | 文献中的机制 | 对项目的启发与不可直接迁移部分 |
| --- | --- | --- |
| 程序分析与符号数值域 | Miné，Symbolic Methods to Enhance the Precision of Numerical Abstract Domains，VMCAI 2006，§5.2 Definition 8、§5.3，将数值状态与符号关系放在同一具体化下，并在变换中利用符号代换。[原文](https://arxiv.org/pdf/cs/0703076) | 项目推断：关系必须参与下一算子的计算，不能只作为附属缓存。共享表达式本身已有先例；也不能把同一源的两次使用分别变成独立区间。 |
| 图像可达集 | ImageStar，CAV 2020，Definition 2、§4.1 Lemma 1，把图像张量读出与共同谓词分开，卷积保持谓词；§4.6 的精确 ReLU 分裂集合，近似 ReLU 增加变量和约束。[原文](https://arxiv.org/pdf/2004.05511) | 可借张量与谓词的组织。不能照搬相位 split，也不能用一个凸 ImageStar 替代原非凸 HZ。张量化本身不能承担本项目的定义创新。 |
| 非凸符号可达性 | Sparse Polynomial Zonotopes，Definition 1、§III-B.3 Proposition 10，用共同因子身份做 exact addition，与解除依赖的 Minkowski sum 区别。[原文](https://arxiv.org/pdf/1901.01780v2) | 项目推断：残差合流需要同一赋值，而非两份独立可行值。可以借依赖语义，不整体替换为多项式域，不引入消除原 bits 的降阶。 |
| 控制理论的重复非线性关系 | Fazlyab、Morari、Pappas，IEEE TAC 2022，§III-C，特别是式15至17，利用重复激活的增量斜率关联多个神经元。[原文](https://arxiv.org/pdf/1903.01287) | 可以研究跨神经元关系如何参与前向变换；二次关系本身已有先例。不直接改用 SDP，不移植对偶优化救援；如要在线性谓词中使用，构造与成本必须另证。 |
| 稀疏多项式优化 | Newton、Papachristodoulou，§4.3至4.6、§5.2，按变量与约束的共同出现组织耦合图，并利用弦稀疏结构。[原文](https://arxiv.org/pdf/2202.02241) | 项目推断：关系分组要看完整谓词，不只看一层权重稀疏性。宽层仍会扩大局部问题，残差还会改变作用域。只借结构选择思想，不导入 SOS、SDP 或其求解性能结论。 |
| 激活原生的代数表示 | Tropical neural analysis，§3至4，让 ReLU 在热带语言中容易处理，但普通仿射变换仍需抽象。[原文](https://arxiv.org/pdf/2108.00893) | 项目推断：不能只让一个算子简单，要同时衡量混合正负权重和合流。不是整体改成热带域，也不采用输入细分。 |

这些方向大多已进入旧档。因此，当前缺口不是此前完全没有跨领域阅读，而是尚未形成经过真实网络验证的 Neural-HZ 定义与组合规律。精确 HZ 本来能表达分段仿射网络；要改进的是在可承担成本下，这些关系能被传播和查询利用到什么程度。

## 本轮补充的操作组合视角

两篇概率电路论文让上述问题更明确：一个表示支持某个廉价操作，并不意味着连续执行多个操作仍然廉价或保有同样的信息。

Shen、Choi、Darwiche 的 Tractable Operations for Arithmetic Circuits of Probabilistic Models，NeurIPS 2016，§3 Theorem 3，证明同一 vtree 上的两个 PSDD 可以在 O(s1*s2) 时间构造乘积。§4 Theorem 5 则给出大小线性的表示，在求和消去一个变量后，对任何 vtree 都需要指数大小的实例。[原文](https://papers.neurips.cc/paper_files/paper/2016/file/5a7f963e5e0504740c3a6b10bb6d4fa5-Paper.pdf)

这不是 HZ 连续投影的复杂度定理，也不要求本项目围绕极端构造工作。项目启发只是：不能从“共享结构可快速组合”推出“以后消去变量也便宜”。实际候选仍应以普通神经块的完整计费判断，不以最坏情况否决所有有限范围的改进。

Wang、Kwiatkowska 的 Compositional Probabilistic and Causal Inference using Tractable Circuit Models，AISTATS 2023，§3.2 Definitions 8至11，区分完整变量上的确定性与投影到指定变量后的支持互斥性；§5.2 的 Forward Problem 沿操作序列传播输入输出结构条件，检查后续操作的前提。[原文](https://proceedings.mlr.press/v206/wang23o/wang23o.pdf)

项目推断：Neural-HZ 的关系语言应写清楚哪些后继变换保持该关系、哪些需要补充信息、哪些只能做健全外包。可以借这种前向操作合同的设计方法，不直接搬入概率语义、学习过程、相位条件化编译或论文的反向条件推导。

这里必须区分三件事：完整标签区分分支、投影后仍能区分支持、多个消费者存在同一个连续见证。前两项不自动保证第三项；保留相位标签或相位共现量，仍不能免费恢复相位内部的共同幅值。此前[共同见证研究](../d078_shared_witness_contract_20261001/DEFINITION_AND_BOUNDARIES.md)和 D035 已给出相关边界。本轮没有把概率电路的结论错误推广成神经网络精确闭包定理。

## 怎样把综述落实到定义研究

下一项有边界的主问题仍放在普通分叉与汇合块上：

```text
q  = ReLU(Wx + a)
r1 = ReLU(V1 q + U1 x + c1)
r2 = ReLU(V2 q + U2 x + c2)
y  = A r1 + B r2 + C x + d
```

这是研究结构，不是新候选预注册或已证明的结论。应优先研究有限共同关系如何被两条支路共同使用，而不是分别处理之后只匹配区间、均值或相位标签。真实宽 Conv 的其他输入项必须完整纳入；不能把未覆盖的残余独立区间化后宣称保留了原共同依赖。

研究交付应包括以下内容。

1. 明确域元素与具体化：连续源、全部原二元相位、EQ/LE、共同 frame 和 decoder 均存在；新增关系必须说明与原始赋值的联系。原 HZ 如何嵌入，哪些变换精确、哪些只是健全外包，分别证明。
2. 为关系写出前向使用条件：谁产生它、哪些消费者共享它、经过混权与下一激活后仍保留什么。先证一个普通块的有限范围命题，不要求任意深度全局闭包，也不以裸 DAG 或关系标签代替证明。
3. 作公平消融：固定相同源界与局部变换，比较共同接口与每个消费者独立接口；同时比较原生 HZ 和 HZ 加相同有效事实。终端约束完全相同时逻辑强度相同，但若能证明组合规则或完整成本优势，仍可能是有价值的算法贡献。
4. 对变量、原 bits、谓词行、nnz、源展开、证据、见证重构和终端求解统一计费。GPU 批量传播属于待验证实现方向，不能从公式可并行推断全路径已经 GPU 化。

若候选只能重述已知符号代换、RLT 或共享 ID，就作为支撑成果保存，不报告为新域。若为了覆盖下一普通块必须复制完整源或展开全部相位，也应明确限定或否决该闭包主张，不靠寻找特殊权重结构凑正例。此处不新增额外晋级门，不把“所有情形都精确”作为创新必要条件。

## 执行范围与保存记录

日期 2026-10-01，branch redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。配置为 read_only_history、primary_literature、paper_only、no_candidate_import、no_numeric_run。根代理核对相关原文与旧档；三路只读复核分别覆盖程序分析、知识编译和非凸可达性。以上只是所列相关章节的核对，不声称读遍每篇全文或证明新颖性。

本轮仅新增本目录的研究记录与校验清单，没有修改历史结果、生产代码、Goal 或默认配置，没有启动模型、LP/MILP、测试、GPU、shadow 或 replay，没有新后台实验，也没有 commit 或 push。已有九个 tracked 修改保持原状，diffstat 为3806 insertions、57 deletions。没有放宽资源、权限、验证人口或冻结预算。

正式 baseline 仍为1870/2413，即1063 CERT与807 validated ADV；13家族逐家族和逐旧解保全要求不变。独立 E0 为 CIFAR100 25、TinyImageNet 36，共61/400，不与1870相加。本轮 formal gain=0，未完成新 Neural-HZ 的正式回放，Goal 保持 active。

所有后续实现仍默认关闭，遵循数学测试、真实同结构样例、shadow、逐家族和完整2413回放；外部400独立回放。不引入 attack/PGD、BaB、输入或相位 split、backward/dual rescue 或实例菜单。write-page 技能用于将文献事实、迁移假设和未验证结果分开记录；仅保存并读回本地文本，不发布外部 Page。

核对的项目输入 SHA256：

```text
0fd93920413f63ad6b3b65849ea4d857c416c6bf4ab56263c63ff73416ebe80c  GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md
e44209a7dab24aa583f78869811a20837b62e57716e831e5d9ef4ab852cb8174  d031_cross_domain_interfaces_20260930/REVIEW.md
5ffe411a62b9f0713c5c14a756b8f698ffd018904ae572b7369ceade6eff7d83  d035_cross_phase_source_20260930/THEORY.md
b0e7d93cf4ee9dfd4de5a7b1a8ba017dd4474534c153fb59f26a6d63e8dc72c5  d073_cross_domain_research_focus_20261001/RESEARCH.md
913cf540ab52bd426bc96913bf8e7ee5584667a3506ed45cafa78b4345c46dfd  d077_coupled_relu_projection_20261001/THEORY.md
```
