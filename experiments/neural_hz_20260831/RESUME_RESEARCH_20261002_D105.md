# Neural-HZ 研究归档与最新续接入口

截至 2026 年 10 月 2 日，旧表示与构造支撑库、定义研究、文献对照、正负结果及最新 D105 推导均已有本地文件入口。本页是本次归档的恢复起点，不依赖聊天记忆；不是实验启动命令，也不更新正式成绩。

## 恢复时先确认目标与状态

当前 Goal active，完整范围见 [本次直接保存的 Goal 快照](definition_first_20260928/d105_local_radial_comparison_20261002/GOAL_SNAPSHOT.json)及 [定义优先目标文档](GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md)。目标是从 HZ 数学定义出发研发适合神经验证的非凸 Neural-HZ，包括 GPU、smooth 和 Transformer 的研究；不是仅优化存储、构造或将计算图改名。旧文档中的暂停和研究顺序只是历史状态，不覆盖当前 Goal。

正式基线仍为 1870/2413，即 1063 CERT＋807 validated ADV。独立 E0 为 CIFAR100 25、TinyImageNet 36，共 61/400，不与 1870 相加。D105 新增正式收益为 0。当前未完成全模型、GPU 或完整物理资格；研究目标未完成。

必须保留连续因子、原二元相位、EQ/LE、共享 latent/frame、输入重构和 fail-closed。禁止退化为 Z/CZ、删除或松弛 bits、按实例或求解状态做菜单，以及 attack/PGD、BaB、split、backward/dual rescue。历史 /data1/Kane/HyZor 与冻结源码、日志、结果只读；新工作仅放 experiments/neural_hz_20260831 的新隔离路径。所有正式晋级仍需同路径完整回放，保住逐旧解和逐家族成绩，不降低冻结测试或资源门槛。

## 按任务读取的档案导航

| 要恢复的工作 | 入口与状态 |
| --- | --- |
| 旧 HZ 表示与构造 | [支撑库索引](supporting_work_archive_20260928/README.md)，含成功、失败、证明和未认证草稿，均原地保留 |
| 定义研究与跨领域文献 | [D103 总交接](archive_checkpoint_20261001_d103/README.md)及其历史导航，不要求每次通读全部历史 |
| 最新实际测试 | [D098 结果](definition_first_20260928/d098_native_relation_bank_20261001/RESULTS.md)，3845 测试、188 文件；原日志、JUnit、退出及人口清单在其链接中；仅组件数学资格 |
| 径向定义与局部严格控制 | [D104 审阅后入口](RESUME_RESEARCH_20261002_D104_REVIEWED.md)，须包含原 bits 及相位视图澄清 |
| 最新同读出比较与负结论 | [D105 推导](definition_first_20260928/d105_local_radial_comparison_20261002/THEORY.md)，未局部化行在非零中心可能弱于简单乘积界 |
| 真实适用性和下一结构问题 | [D105 适用性审查](definition_first_20260928/d105_local_radial_comparison_20261002/APPLICABILITY.md)，真实 LN 来源包尚未建立，现有融合 Attention 必须作为强参考 |
| 本次核验与来源 | [归档审计](definition_first_20260928/d105_local_radial_comparison_20261002/ARCHIVE_AUDIT.json)及同目录 SHA256SUMS |

## 已否决路线与下一步

不把 D104 的零中心局部成功扩大成普通 LN 的已证优势。双标量归一化 Taylor 余项已有先行研究，不能重新命名为创新；h 依赖输入，余项也不是固定二维空间。D104 的原定理和控制保留，没有删除。

不直接恢复 D098 的旧 NEXT_SOURCE 执行方案：D099 的后续静态费用否决仍有效。D099 没有 freeze/RUN。D100 的低秩独立摘要、D101 的经典预算式、D102 的全方向展开和 D103 未完的强对照均维持原资格边界。

下一步先研究现有融合 Attention 之后的同源联合余项如何跨混权／残差保留，与已经存在的共享 Taylor、Q/K 上下文和概率质量界公平比较，给出定义、健全性、严格分离和完整成本；不是立即运行候选。LN 路线需补真实节点与输入域，不能拿 BN 或 Div 计数代替。GPU 和新家族目标不因当前负结论而取消。

## 本次归档做了什么

使用文档整理技能将已证明、已执行、尚未验证及否定性发现分开保存。补齐 D105 的同读出渐近推导、普通菱形源有限反例、已有 Taylor 关系的文献定位、真实适用性缺口及下一步强参考。未新建可执行候选，未启动模型、GPU、solver 或测试，也没有任务运行到一半而漏写本轮结果。

五份关键旧清单共 75 条哈希核验通过；条目有重叠，不是全历史逐文件重审。独立导航审查覆盖五个主要入口的 43 个本地链接，零断链。新文件另由 SHA256SUMS 固定身份。

分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；既有 tracked binary diff SHA256 为 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。本次不改生产或旧档，不暂存、不 commit、不 push。

这些是本机持久归档，不是异机备份。哈希能检测内容变化，不能防磁盘丢失，也不能保证新会话自动读取。下次直接说“读取 experiments/neural_hz_20260831/RESUME_RESEARCH_20261002_D105.md 后继续”，即可从这个入口恢复目标、依据、失败原因和下一步。
