# HybridZ 小型实机控制与准入

2026-10-01。本阶段把现有同算法设备候选核接入有界生命周期，准备检查四个
原合成 HZ 的 CPU/GPU 兼容性。CPU 与模拟故障控制已通过；实机批次先冻结后
尝试一次准入，因设备被占用而未启动任何比较。不能把接线测试或拒绝记录称作
GPU 加速或完整 MoE 证明。

## 冻结与实现

[设计](hz_physical_device_design_20261001.md)先于实现提交；
[执行冻结](../configs/hz_physical_execution_20261001.json)绑定 32 个来源文件、
通过的控制记录和八个调用位置。四个原 HZ 各执行 CPU 与 GPU 一次，顺序交错；
每调用 30 秒，无预热、重试、参数搜索或真实模型。候选算法保持 float64、128 次迭代。

[生命周期](../scoped_proof/device_lifecycle.py)负责准入、提议、释放、检查和接收；
[策略](../scripts/hz_physical_device.py)绑定原始系数、查询、设备与成本；
[worker](../scripts/hz_physical_worker.py)只允许提议进程初始化 CUDA；
[批次入口](../scripts/run_hz_physical_device.py)保存启动收据、失败前缀与未执行位置。
CPU 独立精确检查每个原始目标，不要求 CPU/GPU 浮点轨迹逐 bit 一致。

批次先做一次至多 3 秒的设备准入。存在其他计算进程即拒绝，不因利用率为零
视作空闲；拒绝时八个位置均未启动，不循环查询。每次 GPU 提议前再次检查资源。
准入不是独占锁；释放仅检查自己的进程消失，不证明驱动正确性或下一次准入。
1 GiB 限制针对 PyTorch allocator，不是全部驱动与 CUDA context 显存。

## 已通过的控制

[R2 独立归档审计](hz_physical_controls_20261001_r2.json)通过八个测试方法和六次
固定 CPU/模拟调用。另有 67 项设备候选、支持、传播及工程导航回归通过。
R1 原样保留为加固前记录，最终执行冻结只绑定 R2。

| 调用 | 保存终态 |
|---|---|
| 正常 CPU | CHECKED_GIVEN_HZ_DEVICE_EXECUTION，四条界精确复核 |
| 初始准入忙 | RESOURCE_UNAVAILABLE |
| CUDA 初始化前资源变化模拟 | RESOURCE_UNAVAILABLE，未初始化 CUDA |
| 提议异常 | ERROR，仍检查清理与释放 |
| 检查延迟 | TIMEOUT |
| 释放延迟 | DEVICE_RELEASE_UNCONFIRMED |

控制还覆盖错误 UUID、身份、allocator 与成本字段、缺失及晚到证据、批次异常
后的已消耗位置与剩余清单。模拟 CUDA 收据接受数学有效但不同的下界，拒绝污染
字段；这些是收据测试，不是实机 CUDA 证据。R2 整套控制约 17.73 秒，其中六次
调用的观测耗时合计约 16.84 秒，不能当成 CPU/GPU 性能比较。

原始目录为 `/data1/Kane/MOE/baseline_runs/hz_physical_controls_20261001_r2`。
其 summary SHA-256 为
`bdc23ea58868fc7ae9aeab294b50453ce27253f97d5ec430a2b6694b0aa8d6c9`。
批次拒绝记录已写入冻结的新目录；离线审计成本单列，不伪装成请求内检查成本。

## 保证与下一门

目前仅建立给定 HZ 的连续松弛支持接口及其接线控制。没有新增网络转换证明、
真实请求、原生求解、完整 MoE SAFE 或 GPU 加速结果。G1 至 G6 仍为 OPEN。

## 实机准入结果

执行冻结在 `1ea20ebc8` 提交并推送后才使用。
[独立终态审计](hz_physical_execution_20261001_r1.json)为 PASS，执行状态为
NOT_ADMITTED：计划八次、已启动零次、待执行八次，CUDA 初始化意图和完成调用
均为零。一次准入耗时约 0.154 秒，清理确认；没有重试、轮询或调用替代模型。
PASS 检查的是拒绝记录及分母完整，不是设备兼容性通过。

该时刻设备总显存 97,887 MiB、已用 12,550 MiB，利用率 20%，存在两个计算 PID。
冻结规则要求没有其他计算进程，因此即使可用显存足够也不准入。此快照只是本次
准入证据，不是后续实时状态。原始记录保存在
`/data1/Kane/MOE/baseline_runs/hz_physical_execution_20261001_r1`，summary SHA-256 为
`3e2319023d028a2dafeaef0198154b10b499012bf5d57c50825374b5844d6f97`。

本批不继续查询设备。实机比较仍待另记的新鲜准入，不能覆盖这个拒绝目录或
声称已有 CPU/GPU 差分。可并行推进不依赖 GPU 的独立算法工作：审查 H2 端点目标
如何进入同一 guarded HZ 的支持接口，先写同域、同 gate、同性质的有限控制合同，
不混入本次设备兼容性冻结，不直接启动真实请求。

阶段结束的只读存储检查为整个 MOE 目录 225,777,500,160 字节，约 225.78 GB。
R1/R2 控制目录约 1.6/1.8 MiB；本阶段未删除证据、缓存、权重或环境。
