# V2 控制冻结准入与完整模型阻塞

2026-10-18，核查起点 `feat/moe-route-verification`，HEAD
`d6455b676f6b139d110e9a3abdf1f4cd2d5bc7ad`，工作区干净且远端同步。
[静态记录](hz_native_resource_freeze_readiness_20261018.json)与
[只读重算脚本](../scripts/assess_native_resource_freeze_readiness.py)绑定当前实现。

## 结论与研究方向

**V2 尚不具备新控制的完整冻结条件；也没有解除注册八专家模型的准入拒绝。**
目前不需要再造一套监督包装，而需要在已实现接口上绑定一张完整对象与阶段账本，尤其是
启动阶段保留的副本和最终返回锚。下列参数化公式给出具体约束，不是新额度或执行批准。

本阶段只完成源码、文件名和小型 JSON 模板的静态核算，没有生成来源、调用候选器或运行
模型／GPU。V2 审批表保持为空，旧有限 CPU 控制不重跑。所有数学检查、8M Work、
128 MiB 制品额度和 300 秒保持不变。

研究主线仍是 MoE 专用的关系表示、条件传播和支持求解，以及完整输出证书和公平净收益。
**接口数量、归档检查或缓存节省不构成 G1 算法增量。** G1–G6、历史 23 增益的来源缺口、
真实同源完整正证、GPU 实用收益和外部净优势均保持未闭合。

## 元数据必须在原预留内重新分配

令 `M=1 MiB`、`L=64 KiB`，`m=metadata.encoded`，
`t=min(M,terminal.encoded)`，`c=min(M,caller.encoded)`，`e=|E|`，
`f=min(M,file_bytes)`。这里 E 为不可变封套，I 为调用记录，长度均指规范 JSON 字节。

[metadata_caps](../scoped_source/native_resource_supervised.py) 的 15 个根文件预留之和为：

```text
R15 = 5 min(M,m) + 6 min(L,m) + f + t + c + e <= 8 MiB.
```

生产端 capsule、manifest、bundle 的 `materialize(..., cap=M)` 要求 `m>=M`。
实际内容较短不改变这个接口检查。因此，当 `f=M` 时必须有：

```text
t + c + e <= 1,703,936 bytes.
```

若 terminal、caller 均沿用 1 MiB，上式总量将是 `8,781,824+e`，比原预留至少多
`393,216+e` 字节。**这不证明必须提额**：可以先按成功及失败记录的实际 schema 推导
terminal/caller 的需求，再判断能否在原预留内分配；不能随意缩小 cap 假装需求已满足。

## 新归档公式与失败边界

Ns 为来源文件数（含 manifest），Nf 为 forest 页数，Nt 为候选 tile 数，Nr 为结果页数。
nP 包含 bundle 本身，LP 是所有 package 文件的**相对请求根目录**文件名长度和。
当前新增四个辅助代码文件共 43,921 字节，连同 bindings.json 增加 5 个文件、270 个名称字节。

```text
nP = Ns + Nf + Nt + Nr + 42
LP = 2185 + 42 Ns + 33 Nf + 32 Nt + 34 Nr
Kcap = e + i + 1129                         i=|I|
Bbundle = LP + 79 nP + D_nonbundle - 33
```

D_nonbundle 是除 bundle 自身之外每个成员字节数的十进制位数之和。名称式使用当前固定
ASCII 命名及编号宽度；不能套给任意新来源路径。capsule 必须仍小于等于 1 MiB。
对相同旧 package，只加入这五个成员时：

```text
新增 package 字节 = Kcap + 44,605 + digits(Kcap)
                 <= e + i + 45,741.
```

这个差量不包括 V2 根元数据的其他变化；不能直接沿用旧完整请求的 87 MB 常数。

固定工厂还有包外模板，不能遗漏，也不能把 rename 的同一来源重复算成两份：

| 来源 | 留存模板位置 | 文件数 | 相对根目录名称长度和 | 模板字节 |
| --- | --- | ---: | ---: | --- |
| mixed | generated | 10 | 266 | J_template + 360 |
| clipped | generated/template | 10 | 356 | J_template + 360 |
| guarded | 无 | 0 | 0 | 0 |

维度静态核算中，实际来源分别为 mixed `Ns=16, S=Jmixed+736`、clipped
`Ns=32, S=Jclipped+856`、guarded `Ns=10, S=Jguarded+296`。
这些是源文件与二进制大小，模板的十个文件不能代替 Ns；manifest 实际长度尚未绑定。

令这些模板的文件数、名称和、字节分别为 nG、LG、G，则完整目录为
`nP+15+nG` 个文件、`BP+G+actual_metadata_bytes` 字节。
一般 census 的旧上界 `40+2Lnames+83nfiles` 给出以下新阶段式；字节字段仍按原
128 MiB 制品限制的最多九位十进制数界定。

| census 所在位置 | 已有根元数据数 | 编码长度上界 |
| --- | ---: | --- |
| reception | 11 | 1267 + 2(LP+LG) + 83(nP+nG) |
| controller | 12 | 1378 + 2(LP+LG) + 83(nP+nG) |
| terminal | 13 | 1491 + 2(LP+LG) + 83(nP+nG) |
| caller | 14 | 1600 + 2(LP+LG) + 83(nP+nG) |
| 返回对象 | 15 | 1705 + 2(LP+LG) + 83(nP+nG) |

前四份在成功路径落盘，共不超过 `5736+8(LP+LG)+332(nP+nG)`。第五份仍被返回对象
保留；引用共享不能免除未来保存它的序列化成本。
这些名称式限于来源转移完成后的成功顺序；失败时来源可能仍在 generated/source 等位置，
不能照抄成功路径的名称长度作为失败 census 的界。

设 OB、ON 为启动 archive 前、来源之外已留存的字节和成员数。当前 Archive 还保留
8192 字节内部 reserve，所以成功需要同时满足：

```text
BP <= 128 MiB - OB - 8 MiB - 8192
nP <= 4096 - ON - 16
```

单文件、候选流、Work 等门仍独立生效。对当前已列举的独占写入路径，失败文件是计划文件
的前缀且不覆盖重试，可写出 `Bfailure<=BPmax+Gmax+R15` 和
`Nfailure<=nPmax+nGmax+15`。这里的 max 必须来自完整生成计划，不是一次运行的已完成
前缀。原生非预期文件或并发外部修改不在此保证内；晚期失败可能同时保留结果和错误，
不能只套成功正文。截止后 census 不完整仍是未决，不以静态上界代替实际终态清单。

## 现有八槽需要的完整生命周期绑定

session、metadata、source、observer、observer_next、phase 由多个角色使用，现 schema
须用 `all`；terminal/caller 可限定 caller。当前保守槽和为：

```text
Jslot = 3 (session + metadata + source + observer + observer_next + phase)
        + terminal + caller.
```

这不是精确峰值；phase 必须覆盖内部同时保留的对象与临时量之和。`requested>=2*encoded+1`
只是 raw/fill 的最低条件，不是完整解析树、JSON encoder 或所有对象的需求证明。

还必须有 `Jslot + Boutside <= joint_peak <= capacity`，或经核验的更紧阶段合并式。
现接口允许 joint_peak 留出差额，但并没有自动证明 Boutside：

- caller 在租约前读取审批表，且外层持有原封套、启动参数。
- controller／当前 actor 的 bootstrap 模块全局保留 I 原文、解码 I、E 原文；进入
  actor_main 不会释放它们。replay 还持有独立 session E、actor I 和 capsule 中的 E/I，
  合计五份独立 profile 树，不能当同一别名只计一次。
- `execute` 释放租约后返回的 result/storage/owned_execution 仍存活。当前未保存最终返回锚；
  外层需要计入其编码、文件、fsync、最终观察和接收时间，保持同一原始 t0。

因此完整自有 heap、实际槽数值和返回锚需求继续为 null，不是零。8 GiB AS 与 2 GiB
采样 RSS 是不同的原生失败控制，不是可以直接代入的自有容量。

## 为什么不能只继续做缓存优化

注册完整拓扑有 P=6,961,368 个参数值、A=3,288 行仿射输出。当前源码的计数下界为：

| 保留的遍历 | Work 下界 |
| --- | ---: |
| 第一份 Snapshot | 2P+A = 13,926,024 |
| 当前完整路径已列举必需循环 | 12P = 83,536,416 |
| 理想地省去一份同进程 Snapshot，其余循环不变 | 69,610,392 |
| 仅保留 production 与 fresh replay 各一次逐参数检查 | 2P = 13,922,736 |

最后一行以原逐参数 Work 规则和独立重查为条件，不是所有可能算法的复杂度下界。
它已超过 8M；展开 membership 至少 126,632，也独立超过 65,536。
这说明**同进程不可变 bank 加独立 cursor 的复用，单独不能解除当前完整模型拒绝**。
该方向已有设计基础，不是本轮新发明，也不改变松弛精度。首层共享系数、多目标支持、
尺度分离与批宽八等已有机制也不能再次命名为新贡献。

## 后续有限动作与停止条件

继续的接口工作限定为：复用当前 V2，为三份固定来源给出一张对象—槽—阶段合并表，覆盖
成功／失败、bootstrap、返回锚；同时绑定实际 E/I 长度、两臂同 C/L、外层原始 t0 和
完整归档计划。能满足既有限额后，才提独立有限控制冻结。没有新增包装层的依据。

完整真实模型则是另一道门：需要针对共享／私有关系表示与完整支持检查提出真正减少工作
或表示需求的方案；若保留当前逐参数独立检查计量，则必须单独说明资源政策决策，不能靠
缓存、改计数单位或每 tile 重置绕过已知拒绝。新政策仍须有完整界、两臂共同规则和授权，
不得从这些下界倍数推导额度。本轮不产生这种额度，也不启动实模／GPU。

停止重复旧小控制、旧上限完整形状重试，以及把 bank 缓存微基准当成 G1 证据。新接口控制
即使以后通过，也只支持执行可靠性。研究目标仍需要完整请求的算法直接消融、真实同源
证据和与合理强基线的公平比较，不能用控制数关闭。

## 复核

三个只读 AI 分工核查了归档、生命周期和来源遍历；不是独立人类技术审阅。
脚本只核对冻结 hash、AST、名字长度和小型 schema 算术，不导入生产模块或调用来源工厂。
实际 JSON 差分纠正了新增 bundle 公式的一字节误差：四个新增代码文件大小的十进制位数
之和为 19，不是 20。下列三项静态核查与 whitespace 核查均通过，不能当作运行时控制。

```text
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B scripts/assess_native_resource_freeze_readiness.py --check
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B scripts/check_native_resource_implementation.py --check
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B scripts/check_native_resource_interface_proposal.py --check
git diff --check
```
