# 端点对偶交换实施与有限控制草案

2026-10-19；起点 `6abbc75a0f8ca2d23cd4e1eb68416500f7834da2`，
`feat/moe-route-verification`，起始工作区干净。用户批准了默认关闭实现及控制协议起草，
**没有批准执行**。本轮只修改实现、做静态核查与只读代码审阅；不加载 Torch、生成来源、
执行候选或控制，不运行真实模型／GPU，不修改额度、数值门和 300 秒预算。

## 已实施的单一算法改动

[数学假设及限制](hz_endpoint_dual_static_decision_20261019.md)不变。
[method](../scoped_source/native_method.py) 新增 keyword-only `endpoint_midpoint=False`。
默认 `method(arm)` 返回的原方法字段不变；显式启用才产生以下独立身份：

| 字段 | 新策略 |
| --- | --- |
| schema | `NATIVE_SHARED_ENDPOINT_MIDPOINT_METHOD_V1` |
| scope | `HZ_ENDPOINT_DUAL_EXCHANGE_20261019_V1` |
| producer.kind | `CPU_SCALED_REFERENCE_ENDPOINT_MIDPOINT_LAST_V1` |
| 替换位置 | 第 127 号评价及 best 更新之后，替换最后一次更新 |
| 双端点更新 | 127 次梯度更新＋1 次固定中点更新 |
| 单端点更新 | 原 128 次梯度更新 |
| 每 tile 评价及 forward | 129 次评价、128 次 forward |

本次有意保留最后一次 forward，即使双端点更新不用其输出，以免同时引入省略算子调用的
因素；它仍完整计费。中点不是追加第 130 次评价，也不是给所有性质强制共用一个对偶。
每个端点保留本臂的数值 best，最终仍提交 proposed 与显式 zero 两角色，由原系数精确
检查器判定。它不保证优于旧搜索最后一次梯度更新，也不保证精确下界或完整证书增加。

[候选器](../act/back_end/moe/source_row_endpoints.py)的新增部分是：

- `endpoint_mates` 验证 live RootLease、同一 Driver／forest／Work、合法 pair 和当前
  Plan 的 gate；独立重算各任务的分类系数、offset、query 和 slot，要求完整且有序的端点组。
  不按相邻列盲目配对，缺少非退化 mate 不能冒充单端点回退。
- `midpoint_update` 先从两列 best 分别生成半值，再求和并写入新张量，不修改 best。
  采用 `left/2+right/2` 避免先求和的溢出，仍检查有限性和 inequality 对偶非正。
  配对必须覆盖当前 tile 的全部列且互不重叠；不保留跨 tile 数据。
- `Proposer.propose` 只在启用且第 127 号评价后采用上述更新；第 128 号评价仍执行，
  输出保持原 DualTile、完整行数和两角色格式。新方法身份在 propose 前、交换时及返回前核对。
- 新 `stats.search` 单列实际方法、已完成评价／forward／更新数量。旧算子的 algorithm
  标签在新臂明确记为 legacy label，不再把新搜索错误标成旧搜索。

没有修改 source／guard lowering、端点规划、精确残差、zero 角色、输出聚合、包检查器
或正界接受门。候选器不是独立证书的可信数学基础；任意数值候选都必须经过完整检查。

## 计费与内存含义

绑定扫描和每条分类系数重算继续使用原 Work。设本 tile 有 k 个端点、p 个不同性质、
类数 C、Plan 有 G 个 pair；绑定 helper 的显式新增计费为
`G+k+p` 个 row visits、`C*k` 个 coefficient visits、`1+k*(1+2*C)` 次精确算术。
哈希、类型／集合检查等仍消耗完整墙钟预算，不能因为未增加某项 Work 就算免费。

对一个有 r 行、k=2p 列的对偶组，中点 helper 先检查完整 best。每对端点额外收费：
两个半值的读写各 `2r`、求和 `3r`、两次输出复制 `4r` 个 coefficient visits；有限性
检查另收费，inequality 符号检查也单独收费。合计 equality 为 `14rp`、inequality 为
`17rp` 个 coefficient visits，各另有 p 个 row visits。原生浮点运算不冒充受检查的
有理数算术；相关读写按上述新代码的显式遍历计费，未改旧计量含义或上限。

`midpoint_helper_float64_payload_upper_bytes` 为 `8*r*(k+3)` 的局部组成上界：
新 `mixed` 及一对 `lo/hi/middle` 的 float64 payload。它**不是实测峰值或完整内存上界**，
不含输入 best、旧 z、另一对偶组、forward／残差、布尔检查临时量、Python／Tensor 元数据、
allocator 及异常栈寿命。这些仍须在未来执行冻结的完整对象账本中合并，不能据该字段准入。

## 旧身份与执行隔离

旧两臂默认配置的构造表达式与常量保持；新增策略只有封闭的新 schema／scope／kind。
`validate` 精确重建对应版本，未知字段、旧身份夹带新策略或 128／129 数量改变均不能通过。
这是配置语义的静态核对，不是已经执行了默认路径差分。

执行批准表未变。旧冻结的 implementation hash 不接受本轮新代码；旧结果应使用原提交或
已封装的原代码重查，不更新旧冻结 hash 迁就新版本。[V2 profile](../scoped_source/native_resource_protocol.py)
明确拒绝新 schema，因为其旧 common contract 尚未区别搜索策略。
V1 新运行也必须另有绑定当前代码、方法和来源的批准 profile。
本轮不扩展白名单或开启执行，不新建监督器；将来应复用已有硬监督及全程精确重算路径。

## 本轮已执行的静态核查

使用既有 `act-py312`，执行
`python -I -S -B scripts/check_native_integration_static.py`：PASS，解析／编译
12 个模块，静态遍历 60 个本地依赖；登记表仍有 26 个旧 profiles。
这里的编译仅构造 Python code object，没有执行模块或导入被测代码。

另以标准库 AST／文件字节对照起点 `6abbc75a0f8ca2d23cd4e1eb68416500f7834da2`：

- 原 `method` 的 dict 构造表达式及 ARMS／ALGORITHM／SCOPE 常量一致，新开关默认 False。
- 从新循环的 AST 移除显式 `if self._midpoint` 分支后，与原数值循环 AST 完全相同。
  这验证默认路径的源码一致性，不等于已执行浮点差分。
- 三个修改模块均解析／编译通过；十个数学检查、生命周期、包／策略及批准表文件逐字节未改。
- 本文四个本地链接存在，`git diff --check` 通过。

两项只读 AI 审阅分别核对实现／计费和有限控制设计。复核确认 helper 计费公式、退化 gate
来源及局部内存字段的限定；没有进行独立人类审阅。所有运行控制、来源工厂、模型及 GPU
执行次数均为零。

| 本轮修改模块（未执行） | SHA-256 |
| --- | --- |
| `scoped_source/native_method.py` | `48dfa13904a26cc609f6adf48f472b29245a032aacde0a640334b4c36aa637be` |
| `act/back_end/moe/source_row_endpoints.py` | `d1b6d907f5c5ee08a740e0b31d8843ddab02c6fe7b2c83b28a2082a8c3521d5c` |
| `scoped_source/native_resource_protocol.py` | `a6808a6366f4005cac3de9d6201c17d5f0b3fbcbbeb5e63761003ca62d7bd0a5` |

这些 hash 记录已审查的实现身份，不构成新的执行批准。

## 有限控制草案

以下是待实施／冻结／批准的范围，不是可执行 freeze，也没有新 profile 被加入批准表。
拟限一次准入、零重试、七个受管位置、每位置至多 300 秒；批次累计硬截止 2100 秒，
包括批次初始化及收尾。剩余批次预算不足时不启动下一位置，不额外给予最后位置 300 秒。
准入等待至多 60 秒并单列。失败、超限和未执行位置全部留在清单，不换来源或升限重试。
离线记录审计另记成本，不能补成在线成功。

| 顺序 | 待冻结位置 | 作用 |
| --- | --- | --- |
| 1 | 一批 22 项组件控制 | 先核对绑定、算术接缝及错误路径 |
| 2 | guarded 原搜索 | 付费强参考 |
| 3 | guarded 端点交换 | 同来源、同表示，只改搜索 |
| 4 | mixed 端点交换 | 另一固定来源，反转两臂先后 |
| 5 | mixed 原搜索 | 不重复计时或选最优次序 |
| 6 | 交换接缝非合作阻塞 | 共同硬截止、清理与不接收 |
| 7 | 部分候选输出后故障 | 截断／异常不形成完整接收 |

四个正常请求固定 `native_source_row_blocks`、CPU float64、单线程，双方从同一冻结声明
各自完整构造、检查和求解，不共享另一臂的来源检查或候选。组件可比较同一候选的精确
接受，不要求两种搜索生成相同候选。禁止全局 monkeypatch；故障驱动只能是另行冻结的
有限故障入口，不能由任意回调注入生产配置。

拟沿用 8 GiB 每进程 AS、2 GiB 进程组采样 RSS 停止门、8M 累计 Work、128 MiB 制品及
原位数／候选流额度；**这些数值曾用于旧批次，不等于批准了本批运行**。
运行前仍须闭合七个位置的源码／环境／来源 hash、自有内存、归档与部分终态成本，完成
故障驱动和记录审计器并提交执行冻结。当前不预告必能在这些上限内完成。

组件来源限定已有 `guarded`、`mixed`、`half` 声明；half 仅供组件中真实 checked Plan 的
退化 gate 检查，须作为本批独立来源明确绑定，不修改 native 正常请求的来源白名单。
guarded 与 mixed 的两个正常请求臂均保留完整义务。fixed Task／数组只用于中点算术与
输入故障，不冒充完整同源请求。不得再引入新样本、更多来源或八专家完整模型。

| 组件 | 必须核对的行为 |
| --- | --- |
| 1 | 旧默认方法字节身份不变；新方法显式启用；未知／夹带策略拒绝 |
| 2 | guarded 的宽 8 tile：四个同性质端点组识别完整 |
| 3 | 不同 best 列的同步中点；不顺序读取已改写值 |
| 4 | half 的实际退化 gate：单端点保持原末次梯度更新 |
| 5 | mixed 的宽 2 tile 不越界、不丢端点 |
| 6 | 非退化性质缺 mate 拒绝，不采用 singleton 回退 |
| 7 | 重复 slot 拒绝 |
| 8 | 错误竞争类别／性质绑定拒绝 |
| 9 | 跨 pair 绑定拒绝 |
| 10 | offset／query／slot 不一致拒绝 |
| 11 | 跨请求 roots 或 forest 拒绝 |
| 12 | 不同来源／因子框架身份不能套用；同框架任意数值 dual 必须按原行序重算 |
| 13 | 撤销 roots、旧 token 和旧 tile 拒绝 |
| 14 | 修改 helper 输出不污染 best 或 checked source；别名边界明确 |
| 15 | 正 inequality dual 不能通过完整精确接受 |
| 16 | 非有限 best／中点拒绝，极大／极小有限数仍经精确检查 |
| 17 | 交换接缝异常后清理，不接受部分证据 |
| 18 | 协作式 Work／deadline 超限，累计计费不重置 |
| 19 | 同一候选逐 proposed／zero 与独立展开 scalar oracle 一致 |
| 20 | 一个完整包搬迁后 fresh `python -S` 重算，不读历史运行来源 |
| 21 | 搬迁包 method／invocation 错绑拒绝 |
| 22 | 129 评价／128 forward、双／单端点更新数及终态成本完整 |

第 12 项不能要求拒绝任意数值行置换：非正对偶换序仍可构成另一份合法候选，只是下界
不同。应拒绝的是声明／来源绑定错误，而非质量较差但合法的候选。第 19 项也不能把新臂
候选包改标旧 method 做差分；保留原身份，对同一候选采用独立数学重算。

## 验收与停止

本阶段验收为实现默认关闭、静态源码核查及协议草案完成，不是控制通过。
未来控制要分别报告结构／故障正确性、各端点精确下界、全部性质闭合、gain／loss、四类
未完成状态及完整成本。若仅浮点 best 改善、精确结果无增益或净收益不足，停止此机制，
不增加比例、次数、来源或时间。mixed 专家函数相同，不能用它单独支持真实 MoE 优势。

完整八专家对象的首 Snapshot 至少 `13,926,024>8M`、展开 membership 超限以及真实来源
准入问题仍在。新搜索发生在它们之后，无法解除这些门。G1–G6、同源完整正证、实机 GPU
净收益和外部领先均保持未完成。不得以本阶段代码或日后小控制数量升级这些主张。
