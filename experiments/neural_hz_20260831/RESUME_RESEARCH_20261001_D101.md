# Neural HZ 最新研究恢复入口

先读 [本次归档恢复页](archive_checkpoint_20261001_d101/README.md)，再按任务读取证据。这里是 D101 时快照，不要求每次通读旧研究；将来更新以用户明确修订和更晚记录为准。

目标是从 HZ 数学定义研发非凸 Neural-HZ 及域变换 GPU 加速，而非仅存储或构造优化。完整目标与八项硬限制见 [Goal 快照](archive_checkpoint_20261001_d101/GOAL_SNAPSHOT.json)。保留连续因子、全部原二元非凸相位、EQ/LE、共享身份与输入重构，保每个旧解；旧档只读，新候选先默认关闭、冻结及逐级验证。

最新实跑仍为 D098：3845 测试、188 文件通过，仅数学组件资格。D099 静态否决草稿未执行，不启动；D100 仅纸面，停止 rank-only 独立标量摘要方案。新补存的 [D101](definition_first_20260928/d101_phase_budget_projection_20261001/THEORY.md)也是纸面：经典单预算静态支持的有限前向改善，其后继例已被旧 D066 排除，不是新域或新的能力分离。没有 D101 候选、freeze、RUN、测试、模型或 GPU 执行。

正式 1870/2413（1063 CERT 加 807 validated ADV）和独立 CIFAR100 25、TinyImageNet 36 共 61/400 均不变，formal_gain=0。下次应研究公平强对照下真正可组合的联合关系贡献，不要重复误差坐标改名、rank-only 包装或把局部界改善冒充新解。

早先的表示和构造优化仍在 [支撑成果索引](supporting_work_archive_20260928/README.md)，未删除。关键旧清单 69 条检查通过，范围及一次工作目录误用均保存在归档审计。全部为本机文件，本次没有 commit、push 或异机备份，也不保证未来对话自动读档。Goal 读取时仍 active，本次未修改状态。
