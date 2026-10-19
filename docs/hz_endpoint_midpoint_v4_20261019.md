# 端点中点 CPU 有限控制冻结

2026-10-19，起始提交 `e81339949242fb4442f169c6ac859eeea772fd9e`。
用户针对明确列出的有限阶段批准了测量边界、最小接线、静态核对、新冻结及一次执行。
“最完美的设置”不扩大该问题的范围：不提额、加时、增加来源、运行真实模型或 GPU。

## 研究问题与执行范围

检验已经实现的同 pair、同性质 gate 端点 best-dual 中点替换最后更新。
与原搜索保持同样的 129 次评价、128 次 forward；全部 proposed/zero 角色、精确残差、
来源与性质检查不变。不修改 HybridZ 表示或数值接受门。
[数学合同和停止条件](hz_endpoint_dual_static_decision_20261019.md)继续生效。

既定顺序为 components、guarded baseline、guarded midpoint、mixed midpoint、
mixed baseline、midpoint block、partial candidate。组件保留原 22 项，来源仅为已有
guarded、mixed、half；half 只用于组件。正常两臂独立付费，互不共享答案。
预期 midpoint block 的受管超时是故障控制，不是允许对意外失败重试。
任何非预期失败停止后缀，保留未执行位置。没有追加来源、候选比例或后备重复运行。

全部位置 300 秒、8M Work、128 MiB 制品、每进程 8 GiB AS、采样进程组 2 GiB RSS；
整批 2100 秒，只有一次最多 60 秒准入且计入整批。原位数、成员及元数据上限保持。
先 commit/push 冻结再唯一执行；旧候选、旧批准及失败结果不改写。

## 获批的测量根

[measurement host](../scripts/endpoint_measurement_host.py)是同进程的纯标准库测量根，
不是新增监督服务。只预置标准库、通信和原有限额，随后记录 `t0`；读取请求参数、
请求身份、项目代码和批准表，加载 launcher/caller，以及所有 worker 冷启动均在 `t0` 后。
冻结绑定宿主源码、解释器环境、启动模式与整个代码闭包；运行记录实际 PID、incarnation、
boot 和时钟 namespace。标准 Python、操作系统/时钟、测量入口及运行期间不可变 checkout
仍是可信基础，源码 hash 不是独立启动证明。

宿主预置的可观测前缀单列，解释器更早的冷启动和完整预置成本为 `null`，不填零。
这不是全冷启动总成本，不与历史不同边界求加速比。双方相同预置边界，没有跨请求
模型、来源、候选或已检查答案缓存。该边界不自动扩展到真实模型、GPU 或外部竞争。

原 launcher→caller→observer→受管 session 链不增加层级。V4 明确要求限额在请求 `t0`
前已安装，而不是虚称覆盖测量宿主自身冷启动。接收在原时钟下完成，超时只能撤销接受。
live 返回锚输出到工具接收方，随后在运行目录之外归档；导出编码/输出成本及离线审计成本
单列，不把后来落盘补成在线成功。运行根仍仅四个批次元数据文件及 `runs/`。

## 冻结与审计

[只读准备工具](../scripts/prepare_endpoint_v4.py)逐字节核对原来源/环境、重建当前代码
闭包和六份唯一 profile。native 表保留旧 26 项，仅追加这六项；endpoint 表只登记本次
V4 hash；resource 表不变。新冻结为 `hz_endpoint_midpoint_freeze_v4_20261019.json`，
新目录 `/data1/Kane/MOE/runs/hz_endpoint_midpoint_20261019_r2`，不覆盖旧 R1 候选。
全部 JSON 使用原规范编码，无末尾换行。

现有标准库 AST/compile 静态检查继续核对 22 项和七位置，增加对测量根的结构检查：
限额在 `t0` 前，项目导入在其后，没有新 spawn/监督器。激活后检查确切 V4 批次身份，
不是取消原空批准表门。旧 V2 检查工具仅配合其历史版本使用。

执行后独立读取保存记录与外部 live 返回锚；目录自身不能证明及时接收。分别报告结构/
故障控制、正常请求完整义务、逐性质 gain/loss、精确界、Work、时间、RSS 和制品。
若精确结果无增量，或损失与成本不合格，停止这一个机制，不扩次数/来源/预算追成功。
小控制成功也不关闭 G1–G6，不解除完整八专家来源容量拒绝。

## 缓存准备

既有来源检查拒绝未绑定的时间戳字节码。三个自有、可重建缓存已可恢复地移到
`/data1/Kane/MOE/cache_quarantine/endpoint_v4_20261019_NOPfoo`，
不是在 launcher 中偷偷删除或跳过验证：`scoped_proof/io`（5747字节）、
`source_enclosure/format`（4527字节）、`upstream_source/checker`（18079字节），均为
`__pycache__` 中的 `cpython-312.pyc`。不删除冻结来源、失败制品、checkpoint 或环境。
