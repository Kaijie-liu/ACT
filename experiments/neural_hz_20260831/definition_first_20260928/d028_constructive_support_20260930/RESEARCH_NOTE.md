# 跨领域研究对 Neural HZ 定义的具体要求

本次回应用户提出的跨领域研究方向，并保存此前尚未落盘的共享源支持推导。结论是：需要机制导向的文献综述，不能只在 HZ 的矩阵构造中找改进。现有跨领域阅读已经覆盖多个方向；本次不把重读或已有方法计为新颖性，重点明确哪些机制能改变研究问题，以及它们目前还没有解决什么。

## 最值得借鉴的机制

抽象解释中的观测归约为域设计提供组织方式：多个信息分量需要在共同语义上交换事实，而不是仅并列存储。Cousot 等的 Theorem 4 指出，迭代两两归约一般仍不足以达到完整 reduced product。我们的研究推断是：需要明确定义原相位、连续源与跨层关系之间的前向归约规则；只给 HZ 附加一个缓存没有解决问题。[原文](https://www.di.ens.fr/~cousot/COUSOTpapers/publications.www/CousotCousotMauborgne-FoSSaCS11-LNCS6604-proofs.pdf)

多神经元分析与强混合整数 formulation 提醒我们区别三个层次：标量门区间、保留原输入的单门关系、多个门及其消费者的共同关系。k-ReLU 已研究联合神经元约束；Anderson 等 Proposition 13 已给出盒源上的单个仿射 ReLU 的 ideal formulation。研究推断：新候选必须与这些强局部关系比较，不能只胜过弱 big-M 就称为新域；也不导入相位分裂或求解点驱动的 cut 路径。[k-ReLU](https://papers.nips.cc/paper_files/paper/2019/file/0a9fdbb17feb6ccb7ec405cfb85222c4-Paper.pdf)，[Anderson 等](https://optimization-online.org/wp-content/uploads/2018/11/6911.pdf)

非凸可达分析中的共享因子身份同样重要。Sparse Polynomial Zonotopes 的 Proposition 10 区分共同因子 exact addition 和会丢依赖的 Minkowski sum。研究推断：残差 Add 和 Concat 的共享源应进入数学语义，而非仅做存储去重。共享 ID 本身已有先例；不以多项式域替换本项目必须保留全部原二元相位的 HZ。[原文](https://arxiv.org/pdf/1901.01780)

Tropical 抽象域展示了从目标算子反推代数的办法：ReLU 在 tropical 代数中容易处理，但普通加权仿射映射一般不保持该集合类。研究推断：可以借这种域设计方法，但必须一起解决混合 Conv、ReLU 与残差，而不能把一个算子的困难转移给另一个算子，也不采用其 subdivision 流程。[原文](https://arxiv.org/pdf/2108.00893)

其他已读方向包括 ReluDiff 的共享输入差分、知识编译中表示大小与查询代价的区别，以及扩展 formulation。它们的来源和限制已在[既有综述](../d026_cross_domain_synthesis_20260930/REVIEW.md)记录，本次不重复计为新的阅读成果。

## 研究问题与当前证据

精确 HZ 本来就能表达网络的分段仿射关系。当前问题不是声称 HZ 缺少表达能力，而是能否用适合普通神经块的域元素和前向变换，在完整预算内更有效地利用共享关系。

[上一轮数学记录](../d027_signed_residual_closure_20260930/MATHEMATICS.md)给出一个跨层正控：分别满足完整父联合凸包和子门投影接口凸包的点，仍可被跨层共享源残差关系排除。这个例子支持研究共同源接口，不证明真实网络净收益，也不击败包含全部源与祖先相位的全局凸包。

本次[数学补充](MATHEMATICS.md)进一步区分两条路线。把其他门先替换成仿射上界、再加强一个门，不能超过相应的完整单门源 formulation；这条路线不作为联合关系创新实现。另一方面，固定两个正系数 ReLU 的共同源支持可以用四个完整盒上的仿射支持值构造，补上前述纸面例子的求界前提。后者是已知代数机制，不是新的抽象域；任意混合符号与多层组合仍未解决。

下一项有价值的交付应是：针对普通残差块，提出有限的共同源关系接口，定义其具体化、原 HZ 嵌入和统一前向变换，并证明组合时保留了哪类关系。随后才是默认关闭的实现、真实同结构测试和全量保旧增益验证。这里没有增加全局 ideal hull 要求，也没有承诺满分或 PLDI 新颖性已经成立。

## 只读源码核对带来的限制

现成 D015 有理区间内核可作支持求界 reference，但每次 affine_add、affine_scale 和 source_box_bounds 都会扫描完整 form；它不自动提供缓存 L1 的局部更新复杂度，也不自行绑定模型和 frame 身份。不能把复用解释为免费计算。

已有 D025 complete_0.json 保存完整图拓扑，但只解码了局部 Conv_0/BN_1 和 Conv_3/BN_4 的参数，没有 Conv_6/BN_7 的数值载荷。model_raw 是字节数和 hash，不是模型字节。因此不能声称已经用该旧包计算完整首残差块的数值支持；今后需要单独预注册的新隔离提取。本次没有导入或运行旧候选。

## 保存与成绩

日期 2026-09-30，分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。配置为一手论文相关段落复核、纸面推导及旧代码和证据只读检查；没有新增执行依赖或数值运行。生产文件与旧档未改，原有九个 tracked 修改保持不动；未 commit 或 push。

正式 baseline 仍为 1870/2413，其中 1063 CERT 与 807 validated ADV；13 家族保旧要求不变。外部 E0 仍为 CIFAR100 25、TinyImageNet 36，共 61/400，独立记账。本次没有新增正式解、GPU 性能资格、shadow 或 replay。全部原 bits、连续因子、EQ/LE、共享 frame、输入 decoder 与 fail-closed 边界不变；不引入 attack、PGD、BaB、split、backward/dual rescue 或实例菜单。Goal 保持 active，整体目标尚未达成。

权威输入为 GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md，SHA256 为 0fd93920413f63ad6b3b65849ea4d857c416c6bf4ab56263c63ff73416ebe80c；服务端当前目标同时要求 GPU，且已是 active。基线与研究依据来自该目标及冻结 D026、D027 记录，不把本次阅读当成基线复跑。

使用 pages:write-page 技能区分论文事实、项目推断和纸面证据，并检查本地文本。仅保存新的本地隔离文档，没有创建外部 Page。本记录不授权扩大实验或求解边界。
