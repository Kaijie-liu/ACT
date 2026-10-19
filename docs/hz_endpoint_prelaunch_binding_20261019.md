# 端点有限批次的启动计费绑定

2026-10-19，本阶段仅实施已批准的早起点与计费绑定修正。接口默认关闭，
**静态检查通过，但真实启动方尚未绑定，不能执行**。没有增加监督进程、控制位置、
资源额度或候选查询，也没有运行控制、来源工厂、真实模型或 GPU。
起始版本为 `e6b323103be6fa0ee86ea8db28b2bee7e053146b`，分支与远端同步、工作区干净。

## 已完成的接口修正

启动方必须在本批请求准备及创建调用者进程之前记录 `CLOCK_MONOTONIC` 起点，
保留原始记录与其预期哈希，并在创建进程前施加原有 AS／FSIZE 硬限额。
调用者的 `execute()` 不再自行产生预算起点，也不在导入完成后补设限额。
当前限额不匹配时拒绝，不提升、修复或重试。

启动记录绑定冻结身份、真实启动方身份、请求目录、代码根目录、nonce、boot ID、
时钟命名空间及启动方 PID incarnation。它在校验和哈希之前脱离调用方可变字典，
随后只使用这份标量快照。PID／namespace 检查仅排除部分身份错配，
**记录的哈希不证明启动方可信，也不证明起点确实早于全部准备工作**。
这些事实仍由待绑定的真实启动方负责。

同一 `start_sha256` 贯通准入记录、worker 汇总、终态、返回 proposal、外部接收锚及审计。
`observer_entry` 只是观察点；`startup_seconds = observer_entry - started` 单列调用者启动成本，
所有累计截止时间仍从原始 `started` 计算。接收起点和进程身份检查完成后才记录接收时间，
不能把这些检查的耗时藏在更早的到达时间之后。

| 接口 | 本次改变 |
| --- | --- |
| [批次合同](../scoped_source/endpoint_batch.py) | V3 要求明确的 prelaunch 合同及启动方身份 |
| [观察与接收](../scoped_source/endpoint_batch_observer.py) | 必需外部起点、继承限额核对、启动成本及全链绑定 |
| [调用入口](../scripts/endpoint_midpoint_caller.py) | 显式转交原起点与预期哈希，不提供时钟回退或 CLI |
| [worker](../scripts/run_endpoint_midpoint_controls.py)与[审计器](../scripts/audit_endpoint_midpoint_controls.py) | V3 汇总绑定原起点，缺失成本不当零 |

不能用 `/proc` 进程出生时间补齐完整启动成本。Linux 的进程时间字段在部分创建工作后才设置，
导出的 starttime 又是 boot-time ticks；它既不覆盖调用方准备，也不能直接替换本合同的
monotonic 起点。本接口仅将该字段用于进程 incarnation 检查，不将它换算成预算起点。
参见 [Linux fork 实现](https://github.com/torvalds/linux/blob/master/kernel/fork.c)、
[proc 字段说明](https://man7.org/linux/man-pages/man5/proc_pid_stat.5.html)与
[时钟语义](https://man7.org/linux/man-pages/man2/clock_gettime.2.html)。

## 保持不变的资源和研究范围

七个受管位置、22 项组件名称、129 次评价／128 次 forward、全部来源／符号／残差／角色检查不变。
每位置 300 秒、8M Work、128 MiB 制品、8 GiB AS 硬上限及 2 GiB 采样 RSS 停止门不变。
整批仍为 2100 秒；worker、清理、接收分别截止于原起点后的 2098、2098.5、2100 秒。
一次准入的 60 秒窗口包括调用者启动，不另赠启动时间。七个位置不保证都能用满 300 秒。

新增启动记录上限为 4096 字节，放入原有 `admission.json` 的 1 MiB 上限内，
不新增文件或额度。整批四个文件的上界仍为 2,293,760 字节，小于 4 MiB；
native／component 元数据上界仍为 7,798,784／6,619,136 字节，小于原 8 MiB 预留。
候选搜索、HybridZ、精确数学检查及原生 60 文件依赖闭包没有改变。

## 静态检查与历史保留

仅运行现有环境中的两条静态检查及 `git diff --check`：

```text
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B scripts/check_endpoint_batch_static.py
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B scripts/check_native_integration_static.py
```

前者 PASS：11 入口、4 段 bootstrap、74 依赖，外部起点必需、无回退，
数值额度与旧候选一致。后者 PASS：12 入口、60 依赖、原 26 profiles。
只读 AI 复核检查了起点快照、计时顺序及跨文件身份绑定。
这不是时序／异常／污染控制的运行结果，也不替代独立人类审阅。

[旧候选](hz_endpoint_midpoint_freeze_candidate_20261019.json)的 SHA-256 仍为
`bb7bdadda356202f8dec7132782ac6e4e73dd445571094eb6179eac2e3faf107`；
[旧审查](hz_endpoint_midpoint_freeze_review_20261019.json)仍为
`b67e8ba8df9f69c9dbd0c50c9a664f4eda2abc007f727f3b93492277d27cccc7`。
候选、审查、其说明和旧候选检查脚本原字节保留。旧检查脚本须配合其历史代码版本，
不能在新 V3 接口上重建 V2 候选后将身份变化解释为旧结果损坏。
三份批准表原字节保留，endpoint 批次批准仍为空。

入口中的 V3 文件名与 r2 目录只是未来目标；本阶段没有创建它们或生成可执行冻结。

## 剩余依赖与停止边界

下一步缺的不是再次批准这项已完成的接口修正，而是**一个具体可用的启动方及其绑定**：
它须在请求准备前捕获起点，创建已限额的调用者，保持可核对的进程与时钟身份，
保留外部预期记录，并记录调用者尚未产生文件就失败的情况。
如果没有现成启动方，新建它是另外的实施范围，不能把本接口或一个哈希称为已存在的服务。

绑定完成后才可另行产生 V3 冻结并申请执行。当前文件最多支持一致性检查，
不会自行证明实际早起点、继承限额的历史安装时刻或最终接收后的持久化；
接收锚仍是明确的可信边界，不是完整操作系统或浮点计算栈证明。

本阶段没有新增算法效果、正证书或 GPU 结果。完整八专家来源的 Work／membership 拒绝仍在；
G1–G6 保持 OPEN。停止继续扩展本批包装，不以静态通过数代替后续有限算法对照。
