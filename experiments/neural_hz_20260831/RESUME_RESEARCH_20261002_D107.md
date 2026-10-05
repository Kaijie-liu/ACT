# Neural-HZ 研究归档与续接入口

截至 2026 年 10 月 2 日，已完成的旧构造优化、跨领域文献、定义研究、正负结论及最新真实 ViT 图检查都有本地档案入口。本页补齐 D107 的收尾，使下一次续接不依赖聊天记忆；不是新实验启动命令。

## 目标与不可变边界

研究从 HZ 数学定义出发的非凸 Neural-HZ，面向神经验证、GPU、smooth 和 Transformer；不能用存储优化或计算图换名代替定义创新。完整范围见 [定义优先目标](GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md)和 [已保存的完整 Goal 快照](definition_first_20260928/d105_local_radial_comparison_20261002/GOAL_SNAPSHOT.json)。Goal 当前 active，未完成；更早的暂停状态和旧研究顺序不是当前指令。

正式基线 1870/2413，即 1063 CERT＋807 validated ADV；独立 E0 为 CIFAR100 25、TinyImageNet 36，共 61/400，不相加。此次研究没有新增正式分数。保住逐旧解与 13 家族成绩，新增 ADV 必须具体网络验证；完整同路径回放前不得晋级。

保留连续因子、原二元非凸相位、EQ/LE、共享 latent/frame、输入重构和 fail-closed。禁止退化 Z/CZ、删除或连续松弛 bits、按实例或求解状态选菜单、attack/PGD、BaB、split、backward/dual rescue。历史模型、数据和冻结源码只读；新工作只写本实验根下新隔离目录，保留 provenance。能力晋级与纯速度门解耦，不降低原测试、资源、完整成本和回放要求。

## 按需查阅的档案入口

| 内容 | 入口及资格 |
| --- | --- |
| 旧表示与构造优化 | [支撑成果索引](supporting_work_archive_20260928/README.md)，保留成功、失败和未认证草稿 |
| 跨领域文献与定义研究历史 | [D103 归档导航](archive_checkpoint_20261001_d103/README.md)，含更早入口和 smooth/Transformer 范围 |
| 文献综述与迁移限制 | [跨领域文献综合](literature_priority_synthesis_20261001/REVIEW.md)，最新 Attention 的文献和访问限制另见 D106 的 PRIOR_ART |
| 最近真实执行的数学组件 | [D098 结果](definition_first_20260928/d098_native_relation_bank_20261001/RESULTS.md)，3845 测试、188 文件，仅数学资格 |
| 径向关系与负结论 | [D105 总交接](RESUME_RESEARCH_20261002_D105.md)，包含 D104 原证明及不能扩大适用性的原因 |
| Attention 共享源纸面候选 | [D106 续接](RESUME_RESEARCH_20261002_D106.md)，定义、三 token 控制、先行研究、完整成本缺口 |
| 最新真实图检查 | [D107 结果](definition_first_20260928/d107_vit_graph_inventory_20261002/RESULTS.md)、[状态](definition_first_20260928/d107_vit_graph_inventory_20261002/STATUS.json)、[退出回执](results/d107_vit_graph_inventory_20261002_v1/exit.json) |

这些入口供恢复相关研究时按需读取，不新增所有任务必须通读全部历史的要求。旧文档保持当时状态；D106 的“尚未提取真实图”已由 D107 完成，但其候选并未因此取得 native 资格。

旧文献综合里 D091 的强比较已在后续研究中撤回，不能恢复为当前能力证据；其中“最近执行 D090”同样是当时状态，当前数学组件以 D098 为准。保留这些原文是为了追踪研究变化，不是再次认可已被否定的结论。

## 最新结论及下一步

D107 一次只读解码完成两个预注册唯一 ViT 模型。根据原图常量手推，IBP 为 17 tokens，PGD 为 5，均为 3 heads、每头 16 维；不是 shape inference 或网络执行结果。两者首块 Q/K/V 共享固定参数 BN 的图来源，首个 ReLU 在 Attention 之后；不能拿它们当成 LayerNorm 或前序相位证据。

真实 score 来自动态 Q×K。共同源不等于 D106 已认证的原输入仿射 score；下一步应先研究真实 QK 关系和共享概率如何以完整可支付的形式进入域定义，或者证明近似 score-HZ 的共同扩展见证。当前 token/head 已知，但实际 HZ latent、原 bits、稀疏支撑和转换后生产绑定仍未认证。

已讨论的 weighted-key/shared-hidden 收缩思路保存在 D107 结果末部，只是待比较的已知代数恒等式，未实现或预注册；不得省略中间乘积成本、宣称创新或直接运行旧候选。D099 静态费用否决、D105 局部化问题、D106 强对照和新颖性边界继续有效。下一次先完成定义及成本比较，再决定新版本实验，不重跑已消费 D107。

## 本次保存和核验

本轮补齐 D107 RESULTS、STATUS、本续接页和 SHA256SUMS，保留完整 RUN 的图 JSON、日志、依赖版本、预注册和退出回执。复核 freeze 的 8 个绑定文件与 exit 的 6 个证据文件，14 项均通过；不是全历史重审，也没有在本次收尾重新校验全部 1015 个解码依赖或重跑 D098。

D107 worker 已正常退出，结果完整；限定进程检查未发现明确相关的研究子进程。本次没有启动新的实验。分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；既有 tracked binary diff SHA256 仍为 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5，暂存区空。没有改生产、旧档和正式记账，没有 commit/push。

文档整理技能用于明确区分已证明、已执行、仅讨论及尚未认证的内容。这是本机持久归档，研究文件仍有未跟踪内容；哈希不是异机备份，也不保证未来新会话自动读取。下次可直接说“读取 experiments/neural_hz_20260831/RESUME_RESEARCH_20261002_D107.md 后继续”，即可恢复目标、依据、否决原因和下一步。
