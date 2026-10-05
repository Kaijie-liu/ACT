# 带符号跨层研究检查点

上一轮属于 progress：完成一手文献对照、纸面正负结论及隔离存档，改变了候选的新颖性判断。本轮开始时核对 D026 清单全部通过，分支仍为 redu-hz，commit 为 f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，没有需要恢复的本轮数值进程。

本轮也产生了改变下一项研究动作的证据：

1. [数学记录](MATHEMATICS.md)给出更强的跨层正控。一个满足完整父联合 hull 及子门完整投影接口 hull 的点，违反共享源残差行 11/400；排除了上一轮“只加 parent 联合行已足够排除该点”的消融缺口。
2. 加权残差规则覆盖任意固定正负系数；一个第二 ReLU 的纸面扩展保留该收益，真实上界为 4/5，而旧组合允许 321/400。原输入、全部原 bits 及零点选择保留。
3. 通用前向模板／indicator 支持规则有条件闭包和明确编码成本。在同样叶与观察量下，guard 可改写成已有 min/sum 构造器；这不否认新增连续源观察相对于旧 beta-only 接口的信息增益。固定轴及和差模板又有普通混合方向的明确遗漏。因此不为重复的 guard 语法启动 DAG 实现，不把这套已知机制命名为已创新的 Neural-HZ。
4. [真实结构核对](SOURCE_AND_GPU_SCOPE.md)确认 CIFAR large 和 medium 各有八个残差 Add；实际需要覆盖有符号 Add 后接混合 Conv/BN/ReLU，而不只是单门后直接 Add。Tiny 的旧拓扑只作线索，历史 BN 问题未被隐去。

独立审查复核了共同源混合见证、容量两行、多根 DAG 投影和下界条件，并提醒 binary 节点 bounds 与 relaxed-bit 电路可能不同。所有正控都是纸面证据，不是执行测试、benchmark 或实际 ADV。

## 研究决定

保留新跨层正控和加权规则作为后续比较基线，关闭“相位选择支持 DAG 本身带来新表达力”这项假设。接下来必须给出按实际神经消费者结构获取共享源支持的方法及完整成本；单纯增加名字、固定少量和差方向或保留全部网络图均未解决该问题。

这不增加全局 ideal hull、全局完备性或极端网络门槛。对任何真正候选，仍先明确定义、具体化、HZ 对应和组合定理，再进行默认关闭的数学测试、真实同结构、shadow、逐家族及完整回放。没有降低旧测试人口、资源边界或验证要求，也没有扩大求解权限。

引用的既有方法包括 [Cousot 等的观测归约](https://www.di.ens.fr/~cousot/COUSOTpapers/publications.www/CousotCousotMauborgne-FoSSaCS11-LNCS6604-proofs.pdf)、[k-ReLU 的联合关系](https://papers.nips.cc/paper_files/paper/2019/file/0a9fdbb17feb6ccb7ec405cfb85222c4-Paper.pdf)及[Vielma 的 formulation 大小与强度区分](https://juan-pablo-vielma.github.io/publications/Embedding-Formulations-and-Complexity.pdf)。本轮具体算术和反例是项目推导，不归因于这些论文，也不据此作完整新颖性声明。

## 状态与保存边界

2026-09-30；分支 redu-hz；commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。配置为纸面证明、独立只读复核、保存 JSON 的 metadata 读取和 hash 校验；没有新增运行依赖。

本轮仅新增此隔离目录的研究文件。未导入候选、跑数学数值测试、模型、solver、GPU、shadow 或 replay；未更改生产代码、历史模型或旧档，未 commit/push。原有九个 tracked 修改的 diffstat 保持 3806 insertions、57 deletions，本轮没有写入这些文件。

正式基线仍为 1870/2413（1063 CERT + 807 validated ADV），保全 13 家族。独立 E0 仍为 CIFAR100 25、TinyImageNet 36，共 61/400，不能与 1870 相加。正式新增收益为零，尚未获得新的定义创新、完整性能或 GPU 资格。Goal 保持 active，整体目标未完成；有后续可研究动作，不判为 blocked。

使用 pages:write-page 技能把证明、先例、保存拓扑和未验证假设分开记录；主代理做本地文本核对，没有发布外部 Page。新文档清单随后以 SHA256SUMS 封存，后续修订另存新版本。
