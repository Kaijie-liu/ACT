# H2 全尺寸合成请求的完整容量监督

本阶段把已经准备好的全尺寸来源身份接入创建、捕获、构造、候选、可搬迁检查与终态
接收的同一预算。微型控制与独立归档已经通过；本次文档先冻结全尺寸两臂，执行前
不宣称容量通过，也不增加真实模型证书或外部比较成绩。

## 对象与保证

[新执行协议](../configs/h2_capacity_execution_20261001.json)绑定
[来源准备归档](h2_capacity_identity_20261001_r1.json)的原始字节 SHA，固定声明、
分块来源、model state、物化中心、输入域与输出性质身份。配方仍是 seed 20261001、
输入 3×32×32、router 隐藏层 128、八个 256→128 隐藏专家、十类、selected-softmax
top-2。不读取 checkpoint 或数据集，不执行 forward，不按输出或路由选对象。

每个臂从工厂重新创建并捕获来源，匹配既有声明 `dda46f8c…` 与 1 MiB 分块身份
`57691658…`。不复制准备阶段的来源文件，不免费取得另一个臂的证据。
全部 28 个 unordered pair、252 条分类性质仍是义务；空 guard 也需要证据。
endpoint 最多 504 个 LP 是义务上限，不是已执行查询数。两臂 reuse 清单均为空。

新入口仍是固定合成对象，不接受任意模型或真实输入。声明实数语义的检查不等于
原生浮点执行证明；它也不是生产 HybridZ 的端到端差分。历史主表、23 个数值策略
增益的来源缺口、MetaMoE/CROWN 不利比较与封存输入全部保持原结论。

## 执行与接收

[监督器](../scripts/h2_capacity_supervised.py)与[worker](../scripts/h2_capacity_worker.py)
分别版本化，旧 `scoped_source` 文件不修改。API 从入口计费，每臂 300 秒、CPU 单线程，
父子进程合计抽样 RSS 阈值 2 GiB；这不是操作系统瞬时内存硬限制。完整 300 秒调用的
produce 截止为第 294 秒，check/receive 为第 299 秒，剩余用于终态发布；提前完成时
后续阶段获得实际剩余时间。沿用原监督的预留规则，不通过本轮搜索预算比例。

创建/捕获、分块、所有 source block 与 pair 构造、原生提议、精确下界检查、
序列化、独立 portable 检查、接收、父进程哈希与清理、最终发布全部计入 API。
批次控制门、实现快照、外部完成观察的落盘及后续行政归档单列，不算免费来源准备。
嵌套事件耗时不再加到所属进程耗时上。

父进程在 checker 退出后绑定实际 stdout，再允许接收器读取。即使 stdout 缺失或
读取失败，已执行的 check 阶段仍保留退出、清理、RSS 和耗时。接收器按原聚合规则
重新核对 pair/property/endpoint 覆盖、端点最小值、缺失数量、来源类型和严格
`>1/10000000` 正门。发布逾期覆盖先前正结果；不以离线补查修复在线超时。

报告明确分开：

| 字段 | 含义 |
|---|---|
| pipeline_complete | 三个阶段完整结束，包含完整的非正或缺证据结果 |
| all_obligations_checked | 所有必要义务都有已检查下界，但可能非正 |
| complete_declared_source_positive | 全部必要义务按原门为正且执行未超预算 |
| real_model_admitted | 本阶段始终为 false |

## 部分进度的解释

[构造观察版](../scripts/h2_capacity_build.py)与[提议观察版](../scripts/h2_capacity_native.py)
只添加标量事件；去掉观察语句后，函数 AST 与原版本相同。原 portable 数学检查器
保持不变。诊断文件在 proof/bundle 清单外，不保留矩阵引用，其写入也计费。

`property_record_complete` 可对应 `certificate=None`，不是证书计数；
`native_call_boundary` 是调用前边界，控制回调可以在真正调用前截止；
`native_solver_return` 才记录已返回的求解，仍不意味着成功或正界。
`candidate_independent_bound_complete` 包含非正下界，也不等于整个请求检查完成。
未发布 pair 文件不代表没有尝试性质，事件前缀将单列。被杀进程的最后半行只标为
截断诊断，不参与数学接受。

## 控制和复查结果

最终 R3 的 17 项控制通过，另一个无求解的观察控制覆盖四个旧微型来源、两臂共
8 个场景。新旧构造的全文件身份相同，覆盖并列、部分复用、非正和缺证据情况。
观察控制独立核对 pair、性质、端点去重次序与 origin；proposer 返回 None 时不计
原生候选或检查完成。没有新增模型或求解来选择更好控制。

[独立归档](h2_capacity_execution_controls_20261001_r3.json)核对固定 35 次调用：
1 次完整正、1 次完整非正、3 次缺证据、15 次拒绝、14 次超时、1 次资源限制。
微型完整两臂分别为 3/3 与 2/3 条正性质，base、guard、gate 与请求身份相同。
可搬迁包通过 `python -I -S`，不依赖模型、仓库或求解器；归档重新检查了所有五个
完整流程包。归档拒绝漏调用、删测试名、删源码绑定及空清单。

R1 的 13 项测试虽通过，只读复核发现阶段异常记账、资源/清理审计与归档清单缺口，
因此不是最终控制门。R1 全部记录保留；R2 补齐成本内层字段重签变异，避免测试只
在外层哈希处失败。测试没有改数学、范围、数值门或完整义务。
R2 的 16 项控制也通过，但后续发现 `Popen` 失败的有成本 `ERROR、pid=None`
终态未被审计器容许。R3 补该控制与明确异常合同；R2 归档保留，不作为最终执行门。

逐行/分块数学、静态容量与目录回归共 63 项通过；旧全尺寸身份归档独立复查通过，
说明新脚本未改变其冻结实现清单。控制归档时工程占用 225,026,695,168 字节，较上一
准备阶段约增 79 MB，主要为实现快照和控制证据；本阶段未删文件或安装依赖。

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -I -S scripts/archive_h2_capacity_execution.py /data1/Kane/MOE/baseline_runs/h2_capacity_execution_controls_20261001_r3 --check docs/h2_capacity_execution_controls_20261001_r3.json
```

## 有限全尺寸执行冻结

提交并推送本版本后，[批次入口](../scripts/h2_capacity_full.py)只运行两次：

| 顺序 | 臂 | 新目录 |
|---|---|---|
| 1 | endpoints | baseline_runs/h2_capacity_full_execution_20261001_r1/endpoints |
| 2 | mccormick | baseline_runs/h2_capacity_full_execution_20261001_r1/mccormick |

每臂独立重新创建/捕获，同一固定 300 秒/2 GiB，不并行污染成本。批次要求干净工作区、
已推送的研究分支、控制归档一致，创建排他的结果目录，并保留实际执行 HEAD 与实现
快照。不恢复覆盖失败目录，不因第一臂失败更换第二臂的对象或预算。

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -m scripts.h2_capacity_full --run
```

验收是明确实际容量停止点、完整终态和成本，不要求正结果。16 MiB member、4 MiB
manifest、256 MiB source、2 GiB bundle 等原限制不提高。若停在来源构造、pair
组装、提议或检查，就保留该失败并作只读诊断；不选择容易对象、扩大预算或把部分
文件升级为完整证明。全尺寸容量未过前不准入真实模型，且即使容量过门也须另行
冻结真实实验，不自动启动。
