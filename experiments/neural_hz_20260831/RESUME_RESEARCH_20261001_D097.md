# Neural HZ 研究归档与恢复入口

这是截至 2026 年 10 月 1 日 D097 完成后的续接入口，供用户和下一次会话恢复研究。最新结果是 [D097 数学组件通过](definition_first_20260928/d097_certified_observation_relay_20261001/RESULTS.md)：3841 项测试、187 个文件，测试子进程 48.54 秒。真实网络、GPU 和正式能力尚无本轮新增资格。不要将局部数学测试数当作 2413 例基准回放数。

最新用户要求是核对研究归档、防止上下文丢失。本次发现 D097 原始证据已自动保存但缺结果摘要与续接入口，现以新文件补齐；旧档案原地保留。本地文件帮助恢复上下文，但不等于永久聊天记忆、远端备份或全部历史逐文件重审。

## 目标与不可混淆的成绩

主线仍是从 HZ 数学定义研发适合神经网络验证的非凸 Neural HZ，研究域元素、具体化语义、生成元与相位及谓词的关系、前向抽象变换和精确对应，追求可证明的结构创新与 PLDI 级研究质量，并推进可靠 GPU 加速。不能把存储优化、构造提速、已有扩展表述或更换名称当作新域完成。

项目权威目标正文为 [定义优先修订](GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md)，当前 Goal 服务状态为 active，另含用户后来加入的 GPU 目标。旧文档记载的 paused 状态仅代表写作当时；本次没有暂停或完成 Goal。

- 正式基线为 1870/2413，即 1063 CERT、807 validated ADV。每一个旧解和全部 13 家族逐家族 solved 数都须保住。
- 独立外部 E0 为 CIFAR100 25、TinyImageNet 36，共 61/400，不与 1870 相加。新增解须在剩余 UNKNOWN 中独立计账。
- D097 的 `formal_gain=0`。只有同一候选源码、配置、预算和路径完成完整回放且保旧增益，才可更新成绩或默认启用。ADV 必须原具体网络验证，invalid 必须为 0。

## 恢复时必须保留的研究边界

保留连续因子、全部原 signed 二元相位、EQ/LE、共享 latent/frame 及具体输入重构；不退化为 Z、CZ、纯区间或其他凸域。零预激活处的两种合法原相位标签均保留。结构变换不能按实例、模型、家族身份、公开标签、margin 或 LP/MILP 状态做菜单；不得靠 attack/PGD、BaB、输入或相位 split、backward/dual rescue 冒充域收益。普通终端 LP/MILP 和独立具体见证验证边界不变。

候选默认关闭，依次经过数学、真实同结构、shadow、逐家族及完整回放。能力与纯速度门保持解耦，但不能降低继承的验证和资源门。所有未证前提、数值、资源或见证失败均 fail closed；不围绕极端特例偏离普通结构研究。

`/data1/Kane/HyZor` 历史模型、日志、表格、结果和此前冻结实验均只读。新工作只写 `experiments/neural_hz_20260831` 内的新隔离路径，保存分支、commit、配置、来源及基线 provenance；不覆盖、拼接、删除或重新记账旧档案。已消费版本不修改、不重跑；新的实质组合须新预注册。

## 研究导航

| 内容 | 入口 | 使用方式 |
| --- | --- | --- |
| 旧表示与构造优化 | [支撑成果归档](supporting_work_archive_20260928/README.md) | C1 至 C131 的检索入口，保留成功、失败、未执行草稿；是可复用支撑，不是新域完成。 |
| 早期实验与正式口径 | [试验日志](TRIAL_LOG.md)、[基线锁](BASELINE_LOCK.md) | 以原始来源为准，不拼接候选成绩。 |
| 跨领域综述与研究优先级 | [优先级综述](literature_priority_synthesis_20261001/REVIEW.md)、[复合变换综述](literature_composite_transformers_20261001/REVIEW.md) | 原论文事实、迁移假设与实验证据分开；不等于已认证新颖性。 |
| 定义研究早期导航与文件快照 | [阶段归档](research_handoff_20261001/README.md)、[当时的路径快照](research_handoff_20261001/DEFINITION_FILES.txt) | 这是历史快照，不是 D097 时全目录清单。 |
| D093 与 D094 | [稀疏共同观察](RESUME_RESEARCH_20261001_D093.md)、[出生和费用分析](RESUME_RESEARCH_20261001_D094.md) | 查稀疏界面、实际关系与条件性费用边界。 |
| D095 草稿 | [未执行草稿归档](RESUME_RESEARCH_20261001_D095_DRAFT.md) | 不将草稿或静审说成执行通过。 |
| D096 精确出生 | [结果](definition_first_20260928/d096_verified_phase_birth_20261001/RESULTS.md)、[对接义务](definition_first_20260928/d096_verified_phase_birth_20261001/NEXT_INTEGRATION.md) | 3837 项数学测试通过，在线生产接入仍未完成。 |
| D097 最新组合 | [合同](definition_first_20260928/d097_certified_observation_relay_20261001/CONTRACT.md)、[预注册](definition_first_20260928/d097_certified_observation_relay_20261001/PREREG.md)、[结果](definition_first_20260928/d097_certified_observation_relay_20261001/RESULTS.md) | 3841 项通过；实际出生证书组合保留共同观察收益。 |
| D097 原始证据 | [exit](results/d097_certified_observation_relay_20261001_v1/exit.json)、[inventory](results/d097_certified_observation_relay_20261001_v1/inventory.json)、[日志](results/d097_certified_observation_relay_20261001_v1/tests.log)、[JUnit](results/d097_certified_observation_relay_20261001_v1/tests.xml) | 只读结果，不重启已消费版本。 |

旧 `CURRENT_RESEARCH.md` 停在 D015，实验根 README 仍含 9 月 5 日状态；没有改写它们的历史内容。恢复时先读本入口，再按当前问题读取对应合同和原证据，无须每次通读全部历史。

## 最新进展和下一步

D097 解决了一个组合问题：D096 的实际出生关系同时供基础关系与后继共同观察使用，避免把由实际 EQ 固定的常量列误作自由盒余项。所有原非凸变量和谓词均保留。普通多层块上，继承的理想严格间隙 1/24 在可靠存储后仍大于 1/48；完整 64 后继、68 证书的结构控制也通过。这是已有结构关系在实际出生表示中的组合证据，不是独立新颖性或网络成绩。

下一步不是继续扩大通用测试或泛文献清单，而是新预注册的真实同结构接入：用统一结构规则选择完整组件，认证实际来源、原 phase 列与输入重构，验证组合关系的实际效用和完整成本。生产 `_sparse_relu_slots_for` 使用 frame 全局高水位及同层槽复用；D096 的局部追加不能直接替换。carrier 需独立生命周期，rebase 需明确身份映射，deferred 的部分物化不能冒充完整层，成功后发布登记须保持一致。

D096 单位盒推界可能扩大真实不稳定人口；同结构收益和回归必须实际检查。GPU 的可靠行界和模板批量装配尚是建议，未执行或授予资格。不得把省略终端、证据或 CPU/GPU 共存的成本当作提速。

不能重跑 D088/D090 的已消费归档：D090 已实际认证并加载，但在完整 root 检查中触及工作量和内存门，未进入结构发现。失败不是数学反例，也不是获得真实适用人口；不能再次以同一包装器或事后放宽门槛重复实验。D095 未执行草稿继续保留。

## 执行状态与可恢复性

D097 唯一 RUN 已结束，测试和监督器退出码均为 0；没有登记或启动其模型 worker。日志、JUnit、inventory、运行登记、exit 和两份测试保留记录已落盘。本轮只读进程检查未发现匹配本研究、pytest 或 shadow worker 的活动实验进程；这是检查当时的状态，不是长期监控服务。既有 runner 对成功或失败自动写结果，本次没有新建自动任务，也没有重新运行实验。

独立只读审计确认 D001 至 D097 的 97 个编号均有现存资料；D001 至 D003 位于定义目录顶层，不能仅以子目录计数判断缺失。抽查七份导航和综述的 63 条本地 Markdown 链接，缺失为 0。D 系列 31 个结果目录中 29 个有 exit.json，另外 D067 与 D075 使用 diagnostic.json，并有 RESULTS 解释完成或失败终态，不属于丢失结果。D087 与 D095 是明确记录的未执行草稿，不能当作仍在后台运行。

重新核验 D096 的 17 项、旧支撑成果的 28 项和 research_handoff 的 10 项 SHA 清单记录，最终 55 项全部通过；记录间有重复路径，不称为 55 个独立成果。旧支撑清单最初误从仓库根运行，因相对路径无法打开而失败，随后在其目录 28 项全部通过；这是核验工作目录错误，不是档案损坏或重新实验。

本轮重新核对 D097 六个冻结文件及 freeze 的七个 SHA256 身份，全部与执行时记录相符。随附 [SHA256SUMS](definition_first_20260928/d097_certified_observation_relay_20261001/SHA256SUMS)覆盖冻结文件、原始 RUN 的七个文件、本结果说明和本入口，共 16 项；核验范围不延伸为全历史或所有模型逐文件审计。可以在仓库根只读核对：

```bash
sha256sum --check experiments/neural_hz_20260831/definition_first_20260928/d097_certified_observation_relay_20261001/SHA256SUMS
```

分支 `redu-hz`，commit `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`。本轮开始时 tracked 差异为 9 个文件、3806 insertions、57 deletions；工作树 binary diff SHA256 为 `29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5`，暂存区为空。本次归档不改变这些既有改动。

这些是本机归档和完整性清单，部分文件尚未跟踪；未 commit、push、上传或异机备份。下次恢复依靠此入口和原证据，不假定助手具有永久记忆。文档技能促使本入口明确区分已执行结果、失败、草稿、未取得资格和下一步，未改变研究权限或验证门。
