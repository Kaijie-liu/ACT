# 共享相位结构普查未通过证据预算门

数学组件检查通过，但本轮三个真实模型的完整结构普查失败，正式收益为零。失败发生在 CIFAR100 medium 的证据序列化阶段：共享的 40M evidence 工作预算耗尽。不能将已经计算出的局部关系、未发布的 partial 文件或全部数学测试通过，算成真实三源资格或新的 Neural HZ 验证能力。

2026-09-30，分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。唯一执行为本目录 run_census.py --enabled，解释器 /data1/Kane/miniconda3/bin/python -B。会话 38001 已结束，监督器和 worker 均 exit 1。原版本已消耗，不重跑、不补跑 Tiny、不提高冻结预算，也不改写 partial 或旧失败结果。

## 数学检查通过

完整收集和执行 3797 项、176 文件，pytest 报告 3797 passed、13 warnings、44.59 秒。collection 加 execution 为 57.91107290238142 秒，低于冻结的 60 秒限；监督器总时间为 101.9358916990459 秒，两种计时不能混用。

新增四项普通结构测试覆盖共享边与完整三门重算逐行一致、完整组人口与零相位、真实身份与区间参数，以及默认关闭和资源拒绝。其余 3793 项完整继承，没有减少人口。13 条 warnings 是一条旧 TypedStorage 弃用警告及 12 条旧 JUnit record_property 格式警告；没有据此跳过检查。

source_drift 和 input_drift 为空，provenance_drift=false。数学资格仅属于本次冻结组合；不使失败的真实结构阶段自动通过。

## 完成部分与未完成部分

范围仍是每模型冻结的一份性质、全部 admitted 第一 ReLU 到下一 ReLU 的直接 Conv 分支、原四角和中心窗口、全部通道及 canonical slots；不是整个网络所有层或全部家族性质。

- CIFAR100 large 发布了 complete_0.json，1104301 bytes。五个窗口、320 个接收门全部处理：80 严格 active、238 严格 inactive、2 open；每窗口均不足三个 open 门。因此总计 208320 个三元组全部由严格稳定规则省略，candidate_triangles、odd_triangles、row_count 和 nnz 均为零。此局部负结论不排除其他层或性质存在适用结构。
- CIFAR100 medium 的 worker 日志报告五个窗口、640 个接收门完成关系计算，累计 80 个候选三角、41 个 odd triangles、164 条生成行。但 complete_1.json.partial 在证据预算失败时保留，大小 4809952 bytes，没有发布完整文件，也未获得监督器的完整证据资格。这些数只作为未经完整归档核验的运行观察，不作为已认证真实新关系，更不证明它们对后继输出非冗余。
- TinyImageNet medium 未进入 source_parsed 阶段，没有本轮完整模型证据。不能继承别的运行或拼接前两模型声称三源完成。

complete_0 的 SHA256 为 0bceca1529633b1376f250b1e5b059cf111a2fb817996a88622a06208516a6e8；partial 的 SHA256 为 3173fe8486801abd6353d400576891669b0080126fda3eeadec4f8704a924b93。全部运行工件及其终态哈希见 [唯一运行目录](../../results/d057_triangle_source_census_20260930_v1) 与其中 exit.json。

## 停止原因与资源观察

worker failure.type=ValueError，reason=evidence budget exhausted before operation，evidence_work_used=39999996，原上限为 40000000。whole_work_used=191110889，低于原 256M；worker 时间为 33.403099277988076 秒，低于 240 秒。本次直接停止原因不是全局计算工作预算，也不是超时。

worker 记录 RSS high-water growth 为 456757248 bytes，tracemalloc peak 为 155845942 bytes，metadata 为 15192224 bytes；这些观察本身没有超过各自加 reserve 的 1GiB 门。但终态 memory_gate_passed=false，完整三源未完成，因此不授予内存或完整物理资格，也不把局部观测外推成主机设备总峰值。CPU affinity=[0]，AS=16GiB，无 CUDA。

所有模型 forward、diagnostic solver calls 和 new benchmark solves 都为零。没有 native HZ 行安装，没有真实 phase 列和 frame 绑定，也没有 GPU 计算或加速测量。局部相位盒的代数差不等于满足网络 guard 的点，不等于新的 CERT 或 validated ADV。

## 研究判断与下一步边界

本次失败没有推翻三角后果的数学健全性，但表明完整证据成本是该组合的实际开销，不能从代价中删掉。Medium 日志只能提示普通共享源结构可能值得研究，不能单凭行数认定方法有效。Large 的限定人口则给出明确无适用三角的结果。

经典布尔二次三角、原 HZ 添加相同行和窗口边缓存都不是新域定义。用户要求跨出 HZ 借鉴其他研究，因此后续先回到文献和定义层面：共同源关系的组合、激活原生语义与精确连续消元分别有什么已知机制和不可能边界。此报告不授权立即修证据格式重跑、不新设数值人口，也不以持续扩展测试代替定义创新。

正式 baseline 1870/2413（1063 CERT + 807 validated ADV）与独立 E0 61/400 不变，不能相加；formal_gain=0。source_census_completed、source_census_qualified、native_HZ_admitted、actual_phase_column_binding_verified、gpu_computation_completed 和 complete_physical_qualification 均为 false。Goal 仍 active，整体目标未完成。

本次只增加隔离目录中的此报告与校验清单；此前冻结源码、配置、模型、日志、partial、生产默认及既有 dirty changes 均不修改，无 commit 或 push。write-page 技能用于明确分开数学通过、局部运行观察、完整资格和正式记分。本地 Markdown 读回核对，不声称外部 Page 渲染。
