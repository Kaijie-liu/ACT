# V2 阶段所有权合同与停止决定

2026-10-18，起点 `cbe6bc07dbb76d742a41c62d64031d50d9f8ba6d`，
`feat/moe-route-verification`，工作树干净、远端同步。
[静态记录](hz_native_phase_contract_20261018.json)由
[只读脚本](../scripts/assess_native_phase_contract.py)重算并绑定源码。

## 决定

**停止为当前 V2 追加小控制或新的资源包装。当前仍不准入运行。**
成功／失败终态的编码需求已能写成有限式；完整自有内存和返回结果的持久接收尚未闭合。
这次给出既有接口的对象—槽—阶段合同，不宣称已得到可执行的具体槽额度。
旧有限 CPU 控制已经完成，不复用其批准启动 V2；全部检查、8M Work、128 MiB 制品及
300 秒不变，执行审批表为空。

资源接口是执行可靠性工作，不是 G1 算法成果。完整八专家对象仅两次逐参数检查的条件
Work 下界就为 13,922,736，大于 8M，展开 membership 还独立超限。
**再完成一轮合成资源控制也不能解除该拒绝。** 后续静态研究应回到当前表示和支持算法的
可检验改动；若保留现有独立逐参数检查计量，完整实模必须另行决定资源政策，不能靠缓存、
重命名计数、每 tile 重置或漏检查绕过。这里不提出新数值额度。

## 终态编码合同

令 J 为规范 ASCII JSON 长度，O 为 observer 编码限额，C13／C14／C15 为对应时刻的
完整 census 编码上界，Q 为 caller 资源回执长度。时钟长度保留为
`τ=max(24,J(t0),J(t0+299),J(t0+299.5),J(t0+300))`；仅当冻结 launcher 使用有限
float 单调时钟时取 τ=24，不能从当前允许 int／float 的入口无条件推出。

| 对象 | 成功与失败共用编码上界 |
| --- | --- |
| owned_run | O + 646 + 8τ |
| terminal | C13 + O + 1624 + 9τ |
| caller | C14 + Q + 666 + 2τ |
| 返回锚 | C15 + O + 1895 + 12τ |

这些式子适用于 Session 构造成功后的当前绑定执行链。错误可以与完整 summary、Work、
owned_run 同时存在，不按全 null 缩减。成功清理允许 zombie 清单非空，故保留 O。
Linux PID、returncode 和 RSS 的位数使用受限进程接口的保守界；不是通用平台定理。
入口 t0／opt-in／Session 异常在主 try 外，仍需外层固定代码接收，不能算已有完整终态。

Q 的四个计数不是无界循环的经验猜测。记 s,p,m,o,n,t,c 为 caller 各槽 requested：

```text
live     <= s+p+c
peak     <= s+p+max(m,o+n,t,c)
acquired <= s+p+6m+o+n+t+c
released <= 6m+o+n+t
Q        <= 292 + 上述四个整数十进制位数之和
```

observer 两槽在整个观察循环外各获取一次；六次 metadata 获取来自固定 caller 调用点。
此式不推广到任意角色或任意 Resources 用户，也不能换成四倍 capacity 的位数。

controller log 仍预留 1 MiB、metadata 至少 1 MiB 时，原 8 MiB 根元数据预留要求：

```text
T <= 1 MiB, C <= 1 MiB, e+T+C <= 1,703,936
```

在来源已转移、原成功文件命名及 τ=24 条件下，令 L=LP+LG、n=nP+nG，充分条件成为：
`e+O+Q+4L+166n+5645 <= 1,703,936`。它说明不存在“终态必然放不下”的结构性结论，
但没有实际 E/I、O、完整文件计划和 requested 数值，不能声称已满足。
失败前缀必须独立使用其实际名称上界，不能套成功 L。

若今后选用请求根目录第 16 个预留文件保存返回锚，额外要求
`e+T+C+A_return <= 1,703,936`；同一成功命名条件下为
`e+2O+Q+6L+249n+9533 <= 1,703,936`。这只是合同选项，尚未实施。
锚保存 publication 前的 census，不能递归包含自己的 hash；写入、fsync、读回与接收均
使用原 t0 的 300 秒，不采用保存前的 online_positive 宣告完整计费成功。

## 一个接口内的对象所有权

下表为所需合并合同，不把租约计数当作已经验证的 heap 或别名证明。

| 对象／时段 | 所需所有者与同时存活项 |
| --- | --- |
| 外层输入、审批表、启动参数 | bootstrap／Boutside，早于 Session 租约；原 E、原表与新表的交叠计入 |
| Session 封套和资源表 | session；初始化旧／新结构重叠后持续到 close |
| controller 和当前 actor 的 bootstrap | Boutside；raw I、decoded I、raw E 在 actor_main 中仍被模块全局持有 |
| 普通 read／JSON 临时量 | metadata 负责 raw、解码、规范化临时量；返回树须明确转交仍活跃的 phase |
| 前序回执 | phase；replay 至多一份、reception 至多两份旧回执加当前新树；controller 旧 r 在下一 actor 等待和新 read 时也仍存活 |
| 生产、候选和精确检查 | phase；forest、规划、原生 operator／best dual、frame、残差临时量按同时存活求和 |
| 来源布局检查 | source 负责临时解析；生产数学对象仍由 phase 持有 |
| 搬迁重查 | metadata 覆盖整个 capsule 检查；数学对象归 phase；共五份独立 profile 树，不是一个别名 |
| 观察 | observer＋observer_next 覆盖相邻观察；最终 roster 转交 phase，再转返回接收所有者 |
| 终态发布 | terminal／caller 负责新包装、编码和读回；引用的原内容仍由原所有者负责 |
| execute 返回后 | 外层返回接收所有者；resources.live=0 不表示 result／storage／owned 已释放 |
| 失败诊断 | 失败栈仍持有的对象＋新 census／终态临时量；不是成功最大值与诊断最大值二者取最大 |

当前 `parse()` 文档要求同一槽覆盖解码对象全寿命，而 `read()` 释放 metadata 后返回树，
主调用链依赖外层 phase。应明确这种所有权转交并核验目标仍活跃；observer 返回和最终
return 同理。不能宣称当前通用 helper 已强制执行该转交。无需新增一套监督器。

联合要求仍为 `Jslot+Boutside<=joint_peak<=capacity`。现有 AS／RSS 停止门约束原生
失败，不替代自有对象账本。完整自有 heap、具体槽值与返回接收实现继续记为 null／未实现。

## 三个固定来源的阶段合并参数

由冻结工厂拓扑直接计数，不调用工厂、不依赖一次运行的零系数或稀疏对偶。N／F 为保守
节点／因子上界；H 为展开 membership，V 为 view 数。

| 来源 | N | F | H | V | 最大端点 | pair 行数上界 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mixed | 92 | 155 | 108 | 19 | 6 | 32 |
| clipped | 77 | 131 | 93 | 28 | 6 | 23 |
| guarded | 25 | 37 | 33 | 13 | 24 | 2 |

必须补进 phase 的三项，而不是增加 wrapper：

1. Driver 先建立 RootTable，再建立 Session 的 gate 规划。后一个 pair 的 pullback
   求值时旧 r 仍在：checked forest＋RootTable＋累计 Plan＋旧 F 因子字典＋新 N 节点
   字典＋新 F 因子字典同时存在；当前 gate 规划不是 arena 内的工作区计算。
2. 来源检查保留各 view 可见因子集合，长期条目界为 VF，即 2945／3668／481，另加
   两个各至多 F 的临时集合；这不是可以省略的原生 scratch。
3. workspace 关闭后，source seal 期间 endpoint values、issued 和 receipts 仍存活。
   同一 Receipt 对象只算一次，但各自索引和列表仍计入。异常栈可能延长其寿命。

这些数量不等于完整 Python 字节数。已有整数分配容量、容器及 Fraction 来源公式继续
适用；不能将它们用 `2*encoded+1` 或 arena.live_bytes 替换。受损来源的拒绝路径还需
使用冻结原始长度／深度上界，不能借用成功图的 N／F 作为恶意输入解析上界。
同样，Driver.receive 在检查帧内维度之前允许 4 MiB 原始帧并复制／seal，故障路径可能
有 8 MiB Owner 帧缓冲重叠，另有 caller 原帧；不能用合法小帧的容量式覆盖它。

## 后续边界

本次完成静态接口合同与停止判断，**没有完成可执行资源冻结**。不足项已明确，不再以
新增一轮合成控制、重复 schema 统计或重写监督器推进此分支。具体返回接收及转交约束应
沿现接口处理，完整 heap 未闭合前不启动原生控制；原实施批准不因这份审查自动撤销。

研究主线下一步只允许静态评估一个具体的 MoE 关系／支持机制：说明相对已有多目标支持、
共享／私有块和尺度分离到底新增什么，保留全部义务与独立检查，列明可证工作节省及下界
仍不能消除的部分。若无这样的增量，就停止该候选机制，不能拿接口工作顶替创新。
真实模型／GPU、额度与新控制均另行冻结授权。G1–G6、历史来源缺口及外部竞争结论不升级。

下轮只读候选已定位：V3 `PairWorkspace.pullback` 仍扫描全局 N 个节点，尚未接入绑定
当前 root／refs 的依赖锥执行计划。H1 `sparse_build.closure` 和多目标 pullback 已存在，
不能将依赖锥或批处理本身重新命名为创新。只评估计划生成／检查是否抵消减少的节点与
map 访问；当前零权节点已经跳过系数算术，不能重复计为节省。完整来源、全部 dual 字段
与符号、全因子残差／范数、逆拓扑运算顺序及位数门均须保留；该候选不消除 2P 下界。

本轮三项只读 AI 复核和 schema／AST 算术不替代独立人类技术审阅。重算命令：
`/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B scripts/assess_native_phase_contract.py --check`。
