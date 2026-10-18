# 当前原生两臂的计数账本：补齐接线增量，不生成新额度

2026-10-18，基线 `d370e98f8ada51429c0bf55aa707344590c16a1a`，干净的
`feat/moe-route-verification`。本轮只读源码、冻结记录及 Git 历史对象，计算整数；
不运行来源构造、原生候选、模型、求解器或 GPU。前轮
[完整拓扑拒绝](hz_native_full_shape_decision_20261018.md)及
[有限 CPU 控制](hz_native_controls_results_20261018.md)保持原样。

交付：[可重算静态脚本](../scripts/assess_native_counted_envelope.py)、
[两臂分项及代码绑定](hz_native_counted_envelope_20261018.json)。

## 本轮实际补齐什么

**当前已接线的两臂均使用 pair-scoped 精确检查**，只有原生 local 表示不同：
`native_coo_reference` 对比 `native_source_row_blocks`。旧账本的 dense consumer
不是当前 COO 臂，不能把两个不同消费者的数值搬成这次原生比较。

旧来源／精确消费循环可在核对实现后继承，但旧 fixed proposer、字符串出口与
通用 frame 编码必须替换。新增声明制品复制也必须计入，不能假定 `os.read`
永远填满 64 KiB。重新对账后的**完整遍历逻辑 Work 条件上界**为：

| 当前路径 | 声明制品获取之后 | 加入声明制品获取 |
| --- | ---: | ---: |
| 原生 COO＋pair-scoped 检查 | 165,120,150,476 | **165,180,064,388** |
| 来源行块＋相同检查 | 165,120,350,016 | **165,180,263,928** |

这是对注册 seed 1/2 拓扑、良构已物化来源、全部候选／义务及原计数语义的松上界。
现行 8M／membership 等门会更早拒绝；本表不是实际执行记录、FLOPs、时间预测、
建议配额或成功保证。尤其不能据 1.65e11 的松上界推断真实请求需要同等工作量，
也不能因上界很松就删去检查或私自重定义 Work。

## 来源与消费者的继承依据

相对于旧[共享执行账本](hz_shared_execution_envelope_20261015.json)，22 个来源、
方程、格式和 workspace 文件逐字节一致。对已修改的 `direct_candidates.py` 与
`scaled_endpoints.py`，脚本还逐个比较 18 个实际准备／pullback／消费函数或类的
AST，与旧 Git 对象一致。`finish` 的 sink 改动另行核对：保留原逐端点、逐性质
及九字段计费，新结果模块没有直接 `visit/exact` 调用。

因此可继承：

| 部分 | 条件上界 |
| --- | ---: |
| 一次构造＋encode＋两次独立来源检查 | 216,424,736 |
| 每遍 pair-scoped consumer core | 82,442,004,780 |
| fresh replay frame | 756 |

两次来源 seal 已在两遍 consumer finish 中各按最多 4,096 chunks 计费，**不再
重复增加**。新来源 intake 最多 4,088 个文件，所以该 seal 界仍保守。
AST／哈希一致性支持旧推导的版本适用性，不是自动证明任意 Python 程序的成本。
旧数学循环公式仍是经过源码复核的推导，未声称机器检查过其复杂度证明。

## 原生部分的实际替换

令节点 N≤19,496、因子 F≤32,048、local L≤6,732,720、引用 R≤277,280，
首层权重 W1=6,684,672、首层行 S=2,176、自然 bank B=9。完整目录最多
E=504 端点、T=84 tiles、每 pair M≤2,700 dual rows，非空引用层 J≤2。
该 J 界仅适用于注册架构，不推广到任意深度。

| 计费部分 | COO rows / coefficients | 行块 rows / coefficients |
| --- | ---: | ---: |
| setup | 19,496 / 13,726,720 | 24,744 / 13,726,720 |
| optimize | 281,316 / 228,816 | 475,608 / 228,816 |
| RootLease 附加访问 | 3,024 / 11,088 | 相同 |
| owned frame＋Tile header | 1,176 / 5,443,788 | 相同 |
| **小计** | **19,715,424** | **19,914,964** |

行块 setup 的 coefficients 是 `Q+R+F+2W1`，其中 Q=L−W1；每位置读取及
copy 分别收费，不能将转码视为免费。每 tile 共 257 次 forward／adjoint，
额外 bank 行访问是 `257*T*B`，再加 setup 的 `n+S`，使上界相差 **199,540**。
这不是表示算法的真实耗时胜负。

新 Proposer 返回 owned DualTile，不再生成旧字符串 records，所以去掉旧
`2E+EM=1,361,808` 字符串提取计费。新 frame 保留完整 proposed 和 zero 两角色，
两个 advisory lower 由零填充缓冲区提供，因此不再收旧 generic encode 的
`2E=1,008` 次 advisory put。完整 frame 的公式是：

`rows=2T+2E; coefficients=T+E+4EM`。

这些项的 `arithmetic=0` 只表示没有调用 Work.exact；Torch 计算、同步、解析和
编码当然不是零成本，它们仍受完整墙钟和硬监督约束。

## 声明制品复制：保留短读的严格上界

当前 `capture_declared_owned` 每个文件收一次 row，每次正长度 `os.read` 收一次
coefficient。若总文件数为 Ns、总字节 Bs，严格可用的是：

`copy_rows=Ns; copy_coefficients<=Bs`。

不能用 `ceil(Bs/65536)` 替代：即使请求 64 KiB，成功读取也可能短于它。
注册张量 payload 为 55,715,520 字节，manifest 上限 4 MiB，所以
`Bs<=59,909,824`；成员准入给出 `Ns<=4096−8=4088`。本表使用
**59,913,912** 的复制计数上界，不假设旧满块 95 chunks 的来源布局。
transfer、census、manifest、结果与终态封装本身只 tick，没有直接增计数；
其调用的构造／检查已分项计入，真实 I/O／JSON／哈希／清理成本仍然非零。

完整式为：

`source_core + 2*consumer_core + replay_frame + native_join + copy`。

它没有包含 checkpoint 加载、模型／BN／预处理对应及实时捕获；当前接口接收的
已经是物化声明制品。因此不是 G3 全来源闭合或训练模型端到端成本的完成声明。

## 不能继续沿用旧成功元数据总数

当前格式有三 actor、controller、terminal，而非旧 worker／parent 组合。
完整成功归档与自有对象峰值仍为 **null**，缺项已定位为：

1. 新执行 profile 尚未固定 root/source 路径、environment 字段等的编码长度。
   写出前的 1 MiB 拒绝门不能替代完整生成需求上界。
2. reception、controller、terminal 分别保存 census；每个 census 都有完整
   `files` 与 `hashes` 映射。内存共享不能抵消不同文件中的重复编码。
3. owner/driver 计数与 phases、stats 需按新 schema 计长。本轮可支持累计 Work
   最多 12 位十进制数，但不因此覆盖全部其他计数。
4. 成功清组要求 live 为空，但允许 zombies 非空；不能自动假设完整清理记录是
   空列表。需绑定可信无派生进程范围或另给目录上界。
5. JSON 原对象、编码字符串／bytes、读取 raw 与 canonical 重编码可能共存；
   还需自有容器、Fraction 临时量及分配器寿命账本。8 GiB AS 是超限拒绝，不是
   这些对象必然容纳的静态证明。

所以本轮关闭的是**当前声明路径的计数接线缺项**，不是完整资源合同。下一步只
处理上述新元数据／对象寿命合同，不重新推导已核对的下界、不重复小控制，也不
从这个松上界设新额度。若需新的表示或政策选择，先提出一个联合范围；保持
300 秒、原数学门、全部义务和历史结果，具体实现／额度／执行仍另批。

## 复查

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B scripts/assess_native_counted_envelope.py --check
```

只读取小型代码／记录及指定 Git 对象，不写缓存，不导入生产／数值包；不是新
控制实验或数学端点复证。独立只读 AI 复核不替代人类技术审阅；G1–G6保持开放。
