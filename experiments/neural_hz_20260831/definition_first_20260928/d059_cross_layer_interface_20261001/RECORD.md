# 跨层接口研究的真实结构证据与状态

本轮新增指定条件观察接口的秩差推导、跨层局部凸包失配的解析反例，以及真实残差拓扑的只读核对。它们改变下一候选的设计条件，但不授予定义新颖性、实现、GPU 或基准资格。数学正文见 [THEORY.md](THEORY.md)。

日期2026-10-01，分支 redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。所有新文件限于本新隔离目录。运行配置是 paper_only、read_saved_json_only、primary_literature_only；没有候选导入、脚本数值求解、模型解析/forward、LP/MILP、GPU、shadow 或家族回放。纸面 Fraction 数值是手工解析见证，不是实验次数。

## 当前状态重新核对

上一目标轮属于 progress：封存了已结束的三源失败实验，完成跨领域机制综述，明确将下一行动转为跨层关系接口的可证伪数学问题。本轮重新检查 D058 校验清单三项一致、目标文件 hash 一致，当前分支与 commit 未变。没有仍待轮询的 D057 运行；其 exit.json 已记录终态失败。没有因工具会话缺失而重启旧作业。

原九项 tracked dirty changes 仍为3806 insertions、57 deletions，本轮没有编辑其中任何文件。历史模型、日志、结果和冻结源码只读。目标权威仍为 GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md；其旧服务 paused 段落是历史状态，不据此暂停当前 active Goal。

## 原模型首块支持宽共享源而非独占接口

直接读取以下已保存字段，不重新解析模型或重跑数值：

- [D057 large 完整局部记录](../../results/d057_triangle_source_census_20260930_v1/complete_0.json)：source.model_relative_path 是 onnx/CIFAR100_resnet_large.onnx，model hash 为5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16。original_source_relu 是 Relu_2、port127；branches[0].consumer_relu 是 Relu_5、port130。weight_shape=[64,64,3,3]、group=1；五个窗口、320门、184320 canonical slots、102400 valid slots。
- [D047 medium 完整局部记录](../../results/d047_multiphase_source_census_20260930_v1/complete_1.json)：source.model_relative_path 是 onnx/CIFAR100_resnet_medium.onnx，model hash 为aba117ad0ad4abdd630c220beca70cd58825e72e7bada5dffdda10bb725cece4。原source为Relu_2、port121；接收为Relu_5、port124。weight_shape=[128,64,3,3]、group=1；五个窗口、640门、368640 canonical slots、204800 valid slots。

两个记录的 source_packet_ref 认证同一 extractor 版本，参数不在每份记录中重复复制。这些字段证明主分支未按通道分组，每门有576个 canonical 输入位置；不证明每个权重数值均非零，不证明全网所有卷积均group1，也不证明真正可行 source 是满维盒。因此既不能假定 depthwise/小独占接口，也不能从形状推断条件秩下界已经在模型上触发。

上述两份是各自失败三源运行中的完整单模型工件，不能拼成新的成功运行。D057 medium partial 未用于数学适用性认证；Tiny 在本轮没有新增完整证据，不能补齐三模型人口或参数资格。

## 实际残差后的下一门

若 x 是网络原输入，q 和 r 分别为第一、第二 ReLU bank，已有 packet 的参数覆盖普通两层式中 U=0 的情形。若将 x 改为第一bank、q取第二bank，并研究跨残差后的下一激活，则必须按真实图复合 Add 后的仿射层。

Large 的 [D025 packet.graph.nodes[6:12]](../../results/d025_interval_capacity_20260930_v1/complete_0.json) 是：Relu_5 port130经过Conv_6与BN_7到port132，与原source port127在Add_8合流为port133，再经过Conv_9、BN_10，到Relu_11 port136。不能把Add_8输出直接当下一ReLU输入。

Medium 的 [旧source证据中第1索引模型 packet.graph.nodes[6:14]](../../results/d015_source_shielding_20260928_v2/partial_source_evidence.json) 是：Relu_5 port124经过Conv_6、BN_7到port126；第一bank port121经过shortcut Conv_8、BN_9到port128；Add_10合流后再经过Conv_11、BN_12，到Relu_13 port132。

因此理论式 r=ReLU(Vq+Ux+c) 可以描述这些模式，但 V、U、c 必须包含后续Conv/BN的实际复合。上述记录尚未保存这些后继Conv_6及Add后Conv/BN的完整数值参数，不能宣布该式的真实参数已经认证或跨层规则已执行。

Medium shortcut 的旧packet.branches[1]有weight_shape=[128,64,1,1]、group=1及BN_9参数，但target_relu=null，stop={port:128,reason:side_boundary}。[旧extractor](../d015_source_shielding_20260928/source_packet_v1.py)的后继追踪在动态合流边界停止。这是原范围限制，不应修改原工件来补成成功。更深参数若确需取得，须新预注册，并保持原完整人口与资源门。

## 复核结果和研究决定

两名只读数学审查者分别推导条件方向定理和两层反例，随后交叉核对。审查补充了有效仿射空间的D与常数k、分数LP不是实际乘积、秩下界仅针对所选重构接口，以及局部凸包反例不否定精确separator join等限定。根代理核对全部手工算式和保存JSON字段，原文核查包括Anderson单门表述、分组lift、reduced product、bucket elimination以及神经验证中的RLT先行工作。

正面成果是一个准确、可构造的最小条件观察接口及两个普通强对照；负面结果是免费跨层继承和仅均值拼接的普遍主张均不成立。接口实现若只复用经典lift，仍是支撑库而非定义创新。没有将全网ideal凸包设为新的晋级门：有用、健全但非ideal的候选仍允许按原流程检验。

下一设计应面对宽共享源、混权、identity/projection shortcut及Add后仿射复合。只有条件方向能被完整消费者集合复用或精确省去，且完整代价确有改善，才值得实现新的候选。小treewidth、低秩、独占source或GPU速度都不是目前实证事实。

## 记分与保管

formal_gain=0；正式baseline为1870/2413，即1063 CERT和807 validated ADV。独立E0为CIFAR100 25、TinyImageNet36，共61/400，不与正式分数相加。未重新验证全部旧解，未新增CERT或validated ADV，未默认启用、commit或push。Goal保持active，定义创新、GPU与完整满分目标均未完成。

本轮仅创建数学正文、此记录及校验清单；D057、D047和其他消耗版本不重跑。SHA256SUMS绑定本轮文件、目标与实际读取的旧证据，记录旧文件hash不代表重授其失败运行资格。

write-page技能用于将数学定理、已知技术、真实拓扑、未知参数与正式记分分开呈现。全部本地文档读回核对，不声称外部Page渲染。没有因技能要求暂停研究或申请新的外部权限。
