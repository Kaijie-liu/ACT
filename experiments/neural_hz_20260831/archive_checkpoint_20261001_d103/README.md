# Neural-HZ 研究存档与恢复入口

2026-10-01 的最新归档补齐了 D103 双端预算纸面推导、smooth／Transformer 定义研究，以及含新范围的完整 Goal 快照。旧表示／构造成果和全部既有实验仍在原路径，不覆盖、不移动、不重新记分。

本页用于下次恢复，不依赖聊天记忆。它不是新候选预注册，也不触发任何测试或模型任务。

## 恢复顺序

1. 先读本页及 [Goal 快照](GOAL_SNAPSHOT.json)。快照从当前目标服务直接保存；Goal 是 active。旧目标文档中“服务尚未同步／paused”的说明只是历史记录，不能覆盖当前快照。新增 smooth／Transformer 和 GPU 要求不取消原八条限制。
2. 依据任务读取 [旧表示与构造支撑库](../supporting_work_archive_20260928/README.md)、[D101 归档导航](../archive_checkpoint_20261001_d101/README.md)及 [D102 研究续接](../RESUME_RESEARCH_20261001_D102.md)。不要求每次通读全部历史。
3. 最近纸面研究读 [D103 推导及未完成对照](../definition_first_20260928/d103_source_mixing_hull_20261001/THEORY.md)和 [smooth 与 Transformer 研究](SMOOTH_TRANSFORMER_RESEARCH.md)。
4. 真正已执行的最新组件状态读 [D098 结果](../definition_first_20260928/d098_native_relation_bank_20261001/RESULTS.md)及其原始日志／清单；不要把后续纸面推导当成执行结果。

## 必须记住的状态

| 项目 | 当前状态 |
| --- | --- |
| 正式 13 家族 | 1870/2413，1063 CERT＋807 validated ADV；本轮 formal gain 为 0 |
| 独立 E0 | CIFAR100 25、TinyImageNet 36，共 61/400；不与 1870 相加 |
| 正式 ViT | 90/200；旧未解模型的算子签名不代表标准 GELU／LayerNorm Transformer |
| 最近数值资格 | D098：3845 测试、188 文件；仅组件数学资格 |
| D099 | 静态否决草稿，无 freeze／RUN，不复活它替代定义研究 |
| D100 | 独立标量低秩摘要可能丢相关性；rank-only 包装未获支持 |
| D101 与 D102 | 经典预算支持／联合像有用，但不是新域；全方向展开尚无完整成本优势 |
| D103 | 纯 epigraph 四面凸包纸面证明；已知 MIR 的双端闭合，完整神经凸包及强对照均未完成 |
| Smooth 与 Transformer | 表达边界、现有实现及文献已核对；新关系块仅研究假设，无资格或成绩 |

总方向始终是从 HZ 数学定义出发的非凸 Neural-HZ，不是单纯存储或构造优化。保留原因子／bits、EQ/LE、共享身份、具体输入重构和 fail-closed；不能退化 Z／CZ、减少测试人口、混合不同路径分数或用禁用搜索取得收益。正式晋级仍需同路径完整回放、逐家族和逐旧解零回退。

## 下一步与不能重复的路线

下一步先研究共同归一化／余项关系如何跨普通 Attention、残差和非线性消费者保留，证明原 HZ 嵌入、健全 lowering、强对照分离及完整费用。现有 fused attention、共享多项式符号、MIR、simplex 或零和等式均不是此次新发明。ReLU 源绑定路线保留；D103 未完比较单独标注，不能当作强证据。

不重复把局部矩阵节省称为定义创新，不把 D091 已撤回的强比较、D101 被 D066 排除的控制恢复为新能力，不启动 D102 宽层全方向包装来替代实质研究。未知处保留未知，不以“全 ViT”愿景冒充保证。

## 保管与核验范围

归档配置为 documentation_only，branch redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。本轮未修改生产或历史模型／结果，没有新数值实验。四份关键旧清单重新校验共 71 条通过，条目可能重叠，不等于全历史逐文件或证明重新审计；详见 [审计记录](ARCHIVE_AUDIT.json)。新文件由本目录 SHA256SUMS 绑定。

原 tracked 差异仍是 9 文件、3806 insertions、57 deletions，binary diff SHA256 为 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5；没有暂存、commit 或 push。所有研究存档目前仍是本机文件，没有本次远端或异机备份。哈希能发现内容变化，不能保护磁盘丢失，也不保证未来会话自动读取。

本次使用 write-page 文档技能，将目标、已验证结果、纸面推导、负结论、未完成事项和备份边界分开。短入口为 [RESUME_RESEARCH_20261001_ARCHIVE_D103](../RESUME_RESEARCH_20261001_ARCHIVE_D103.md)。下次可直接说“读取这个续接入口后继续”。
