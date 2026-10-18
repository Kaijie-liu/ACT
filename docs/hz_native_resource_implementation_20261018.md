# 联合资源接口 V2：已实施、默认关闭、尚未运行

2026-10-18。起始 `feat/moe-route-verification`，HEAD
`b6ce3eccc2f4ad3921ad262df8e7a254810f6bae`，干净且远端同步。
本阶段落实用户对[联合接口实施范围](hz_native_resource_interface_proposal_20261018.md)的批准。
仅实施及静态核查；没有批准或执行新控制、原生计算、真实模型、GPU、训练或安装依赖。
[机器记录](hz_native_resource_implementation_20261018.json)由
[只读核查脚本](../scripts/check_native_resource_implementation.py)重算。
它只读取源码、核对冻结锚、解析 AST 和内存编译，不导入这些新模块。

## 一、交付与范围

| 接口 | 已写入的职责 | 本阶段没有证明什么 |
| --- | --- | --- |
| `native_resources` | 实际有界读取；重复键仍拒绝；流式规范计长／哈希／写出；有限命名槽预留 | 不是完整 Python heap、分配器或通用别名安全证明 |
| `native_resource_protocol` | C/L/E/R/I 身份链、新审批表、来源布局与实现闭包 | 不自动继承旧执行审批；没有具体槽容量或完整模型准入 |
| `native_resource_production` | 固定来源预检查、原完整数学生产链、额外绑定 capsule | 未实测原生两臂；不改 HybridZ 数学表示／接受门 |
| `native_resource_portable` | 外部 capsule/bundle/E/I 锚、同一资源计划、原独立完整重算 | API 接线，未执行搬迁；尚无本版独立 CLI 冻结 |
| `owned_resource_v2` | 有界 proc 观察、明确不完整状态、只清理自身进程组 | 不是原子 OS 快照；不承诺回收孤儿 zombie 或逃逸子进程 |
| `native_resource_supervised` | 三阶段、统一原始 t0、失败前缀、终态／caller 回执及释放后接收 | 未运行截止／故障控制，不宣称实测完整成本闭合 |

旧 `native_protocol/native_supervised/native_production/owned_bounded`、数学消费器、
独立检查器和冻结结果未修改。新生产器保留两次独立 lowering、全部 pair/property、两角色
候选、精确符号／残差／补偿、完整聚合和最终 seal。新布局检查新增逐 tensor/chunk 的
`row_visits`；旧 Work 不删除、不按块重置。新元数据、读取、观察和归档耗时均在原请求
截止内，不能从旧记录继承一个免费的执行成本。

**本阶段是资源／可靠性接口实施，不是 G1 的算法增量，也不是 G2–G5 的新实验。**

## 二、明确收紧来源范围，而非掩盖读取窗口

V2 初版在 `profile` 和 `acquire` 两处拒绝 `declared_artifact`，仅支持原三种固定合成
声明的代码路径（mixed/clipped/guarded；本轮没有生成它们）。原因是外部 manifest 先预检、
再由旧 reader 重新打开，并不能把实际解析字节原子绑定到精细布局预留。不能靠复制后的
拒绝追溯撤销第一次分配。此缺口保留为后续独立入口工作，不修改旧 reader 来绕过冻结身份。

固定工厂只先生成字节；V2 在首次 `build→Snapshot/TensorRows` 前核对布局。
来源 raw manifest 仍最多 4 MiB；其规范身份表示的旧 grammar 上界是 32 MiB，二者分别
核对。32 MiB 是规范计长／哈希支持界，不是扩大 8 MiB 数学 header、128 MiB 制品或 heap
额度。opaque `trust` 中的 JSON 整数不再误套数学中间有理数的 8192 位门；后者原样保留。
深度及编码支持范围必须在后续资源计划中冻结，不能宣称接受所有任意深度的旧来源。

受管私有制品及搬迁副本在检查期间**不得有并发外部写者**。旧 reader 在 preflight 后
重新读取，身份／seal 仍检测变化，但这不是通用对抗式原子解析绑定。检查器代码与可信
bootstrap 必须从外部固定；运行过的恶意 Python 不能靠自报 hash 自证可信。

## 三、身份、资源与两臂

新审批文件 `hz_native_resource_execution_approvals.json` **为空**。入口先检查外部批准的
不可变 E 原文 hash，再解析其内容；E 不能自批。I 同时绑定实际 R 修订及 C/L/E，actor、
观察、capsule、终态与 caller 回执沿用该链。搬迁 API 核对传入资源 owner 的全部
capacity/joint_peak/formula/slots/binding 与 E 一致；不读取旧运行目录或审批历史。

资源计划没有默认数值。它须另行冻结 session、metadata、source、两个 observer、phase、
terminal、caller 的 requested/encoded/depth/role；同时保守覆盖 caller＋controller＋
一个活动 actor。相同槽名的嵌套借用是同一 backing 的内部约定，不会自动验证任意别名。
独立复制必须由另一槽或明确的 phase 合并公式计入。logical lease 释放不等于 RSS 归零，
返回对象仍由 caller 持有。当前只实现检查合同的接口，未把这些声明证明为完整 heap 上界。

registry 小型可信 bootstrap 读入、解释器及代码装载仍为明确的启动基础；不能声称所有
首次对象都已由运行时租约精确计量。它们必须进入后续冻结的联合对象账本。

两臂公共算法仍为 CPU/float64/单线程、128 更新／129 评估、profile 0、proposed＋zero、
候选 8／检查 2、production/replay 均 pair-scoped。仅表示允许 old COO/reference 与
source-row-block 的注册差别。单请求执行器不替代成对冻结器：后续必须核对两臂同 C/L、
各自独立付费且不借另一臂答案。本版未提供可执行配对配置。

## 四、终态与成本边界

- 外层必须在导入／输入准备之前取得原始 `started`，传给 `execute`，不得进入接口时重置。
  仍为原始起点后 299／299.5／300 秒的执行／清理／最终接收安排。
- 原工作与接受策略保持 8M Work、128 MiB 总制品、原位数、全部义务。OS 数值没有在本版
  获得新执行授权；旧 8 GiB AS／2 GiB sampled RSS 控制只属于旧冻结。
- controller 日志按实际旧 FSIZE 上限预留，而不是按较小 JSON cap 假装更小。实际 suffix
  读取不得超过 metadata backing；大日志可拒绝，不隐含扩大缓冲。
- 新 envelope、绑定 capsule、auxiliary code、terminal、caller 都计入制品。request-root
  元数据上界必须仍放在旧 8 MiB／16 文件预留内，不能把新增 caller 放到免费外部目录。
- proc 观察先预留完整字段／计数／错误码，再逐 PID 扣表示空间；原文读取有哨兵上限。
  扫描不完整保留 `complete=false, rss=null`，不是空进程组。进入 cleanup 前丢弃 running
  roster，只保留有限失败摘要；cleanup 最多旧／新两份 roster，不能暗留第三份。
- owned leader 不回收前先向自己创建的进程组发信号；回收后不再向可能复用的 PGID 发信号。
  运行失败被锁定；后来的完整清理不能把失败洗成成功。
- 准入之后 mkdir／封套／调用发布／启动失败进入同一失败路径；只向本次独占创建的目录
  写终态。I 未发布可记录 `invocation=null`，不伪造调用身份。deadline 后不能补写成功。
- 部分写出失败后尝试及时重数实际前缀；没时间或不完整则 `storage_complete=false`，不能
  把旧 census 写成完整最终体积。每个失败仍报告实际入口到返回耗时／overrun。
- terminal/caller 落盘内容只声明 pending。phase 与 session 显式释放后才观察最终时间，
  决定 `online_report_received`；`online_positive` 还需全部原正界聚合。完整 UNKNOWN
  报告可以被接收，但不会成为正证据。返回锚须由未来外层执行账本绑定回执 hash；文件本身
  不能自证包含其自身写完及返回时间。解释器最后返回／外部保存成本不能凭空视为零。

合作式截止不能保证 OS 阻塞时物理返回必早于 300 秒；超额必须记录并拒绝。离线完整
检查不能补成在线成功。本版不承诺错误时每个原生操作都可给出完整记录。

## 五、本阶段静态核查和下一门

三个只读 AI 复核分别检查来源解析、资源绑定和清理成本，修正了：首次来源重读范围、
opaque 整数／规范编码支持、搬迁 plan 绑定、controller 日志实际上界、观察三份重叠、
部分观察表示、初始化失败、发布失败后 census 以及释放前提前接受。它们不是独立人类审阅。

允许的复核命令（既有 act-py312，标准库，只读）：

```text
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B scripts/check_native_resource_implementation.py --check
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B scripts/check_native_resource_interface_proposal.py --check
git diff --check
```

本轮新控制／原生候选／模型／GPU 调用均为 **0**。静态 PASS 不表示搬迁、截止、异常、
部分证据或两臂运行已经通过。不能把这轮实现与旧 26 个控制拼接为新版验证结果。

下一步仅需一份独立、有限的控制冻结：先给出实际 slot/joint/失败归档账本、两臂同 C/L、
可信 launcher 原始 t0、搬迁入口及返回锚保存方式，再登记控制身份／次数／累计预算与
对应权限。没有这些就不填审批表、不启动计算；不重复请求已经完成的旧控制。
当前完整八专家仍因首 Snapshot 13,926,024 > 8M、展开 membership 至少 126,632 > 65,536
而拒绝，完整自有 heap 也未闭合。本接口不修复这些算法／容量问题。G1–G6 仍 OPEN；
历史 23 增益来源缺口、外部净优势与真实同源完整正证、GPU 实用收益均不升级。
