# 原始卷积结构中的相位容量实验结果

统一的区间相位容量规则已经通过完整组件测试，并在 CIFAR100 的普通卷积结构中实际产生了系数收紧。但本次三模型实验未过整体资格门：第二个模型保存完整证据前耗尽了冻结的证据工作预算，TinyImageNet 尚未进入。因此本轮有局部适用性证据，没有完整三模型资格、新抽象域完成或正式新增解。

## 实际执行与保留结果

唯一执行命令为 `/data1/Kane/miniconda3/bin/python -B experiments/neural_hz_20260831/definition_first_20260928/d025_interval_capacity_20260930/run_census.py --enabled`。会话 60974 已返回退出码 1，无本轮遗留运行进程。首次执行前完成数学、source 语义、实现和预算审查，冻结了七个输入文件；执行后七项 SHA256 全部一致。没有修改冻结后源码、重跑或提高门槛。

完整继承的 3746 项测试加七项新测试，共 3753 项、168 个文件，全部通过，无 failure、error、skip，13 条 warning 均来自旧测试。收集与执行合计 57.823785 秒，低于原 60 秒门；pytest 自报执行为 44.68 秒。这不是网络验证提速数据。

真实结构保持每模型五个原定空间位置、每个直接 next-ReLU 分支的全部输出通道，以及完整 576 槽 fan-in。稳定源和 padding 未被筛掉，配对规则不依赖模型、实例、margin 或历史结论。

| 模型 | 数值完成的接收节点 | 容量系数有严格收紧的节点 | 普通外包仍跨零的节点中有收紧 | 证据状态 |
| --- | ---: | ---: | ---: | --- |
| CIFAR100 large | 320/320 | 314 | 2/2 | 完整逐行共享前提证据已保存 |
| CIFAR100 medium | 640/640 | 625 | 25/26 | 仅数值完成摘要和进度日志；未保存完整逐行证据 |
| TinyImageNet medium | 未进入 | 未测 | 未测 | 没有本版本数值结论 |

large 的两个跨零节点为位置 (0,0) 通道 56、位置 (16,16) 通道 55；这些身份只是输出审计信息，不参与规则选择。large 总计 3381 个正容量系数和 3376 个负容量系数降低。其 1600 个源中，认证严格正 727、严格负 863、外包跨零 10。medium 的日志总计正系数 6267、负系数 6243 降低。

这里的“收紧”是相对于同一认证区间上的未配对容量，至少一个系数严格降低。它不等于接收节点变稳定、不等于最终性质改善，更不等于新增 CERT/ADV。跨零只描述外包，不证明两种符号真实可达。新旧比较并非完整 ACT baseline 或同一终端 HZ 的求解比较，不能说 314 或 625 个节点被验证成功。

## 停止原因与资源范围

worker 在 54.670254 秒结束，停止原因准确为 `ValueError: evidence budget exhausted before operation`。完整 large 证据的遍历与编码用了 34,373,196 work；处理 medium 的物理对象账本时，全局证据计数达到 39,999,993/40,000,000，下一次操作被拒绝。medium 尚未开始发布完整 JSON；未进入 Tiny，也没有把剩余人口当作通过。

这不是总数值 work、240 秒或观察到的内存超限：whole work 为 170,283,793/256M，最大模型数值 work 为 76,684,744/200M，RSS 高水位增长 334,626,816 字节，tracemalloc peak 139,711,798 加 metadata 14,265,376 字节；两项再加 65,536 字节均小于 1GiB。`memory_gate_passed=false` 是包含完整性和所有资源条件的综合字段，不能据此误报内存耗尽。retained_entries=694,504 只来自完成账本的 large 模型上界，不代表 medium 的未完成账本也已认证。

large 完整证据为 12,062,002 字节，SHA256 为 `fbab84537df071153a7d9362161b2c9aa85b80c8b4c605ff1e1149170e0965e0`。证据保留原参数/输入、全部共享源和配对界、所有接收行及可重构的统一容量公式；没有把不收紧的行删掉。实际 HZ 未装配，行 nnz 只是保留 g、t 坐标时的理论支持计数，不是终端实测成本。

supervisor 结束时的源码、输入、生产 provenance 漂移检查分别为 []、[]、false。其 RSS high-water growth 字段为 0，并不表示零分配；VmRSS 从 20,148,224 增至 49,479,680 字节。CPU 测试进程、supervisor 和 source worker 的范围分开，不能宣称全系统物理资格。

## 对后续定义研究的影响

本轮排除了“这些相位关系只在手工小例子里有效”的过强怀疑：至少 large 的固定普通卷积结构有完整可复核的系数改善证据。但它未证明增益大小足以改变性质，也未证明加入全部谓词后有净成本收益。规则仍是已知关系的结构化组件，普通 HZ 加相同行的逻辑精度相同。

保留这个组件及失败版本，不为取得通过记录缩人口、重试或扩大证据预算。也不把下一轮主线改成 JSON 压缩。下一项定义候选需要回答当前单容量接口的跨层信息损失，并与新核对的 tropical 神经网络抽象先例比较，详见 [定义与先例审计](DEFINITION_AUDIT.md)。若只是 min/sum 电路或已知 cuts 的打包，继续作为支撑组件，不当作 Neural HZ 定义创新。

## 资格与存档

`component_tests_passed=true`；`source_census_completed=false`、`source_census_qualified=false`、`native_HZ_admitted=false`、`gpu_computation_completed=false`、`complete_physical_qualification=false`、`formal_gain=0`。本轮没有 GPU 初始化、原网络 forward、终端求解、shadow、逐家族或全量 replay。

正式 baseline 仍为 1870/2413（1063 CERT、807 validated ADV）；独立外部 E0 仍为 CIFAR100 25、TinyImageNet 36，共 61/400。不因组件测试通过声称候选已保住这些旧解，不把外部 61 加入 1870。Goal 保持 active；本轮属于 progress，依据是新数学适用范围、实现、实际测试和真实结构正负证据，而不是整体目标完成。

结果目录为 [独立执行记录](../../results/d025_interval_capacity_20260930_v1/exit.json)，完整 large 证据为 [complete_0.json](../../results/d025_interval_capacity_20260930_v1/complete_0.json)，medium 未完成状态见 [diagnostic.log](../../results/d025_interval_capacity_20260930_v1/diagnostic.log)。退出记录 SHA256 为 `634c09d1ffdeeafa4bedf3b16c7e1febb058301b69340f1b83f306c0e40be2aa`。

2026-09-30，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；原 dirty worktree 保留。仅新增隔离实验文件和本结果目录，没有改生产默认、历史结果或远端。pages:write-page 用于明确区分论文事实、数学推断、执行证据与未通过资格；仅写本地文档，未发布外部 Page。
