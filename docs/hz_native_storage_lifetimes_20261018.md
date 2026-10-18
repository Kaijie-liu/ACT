# 原生完整路径：归档结构与自有存储生命周期

2026-10-18，干净起点 `91b661ec55741a1bd07a87a4295d927b2d801f55`，分支
`feat/moe-route-verification`。继承[当前计数账本](hz_native_counted_envelope_20261018.md)，
本轮只读源码／冻结记录／指定 Git 对象并计算整数，没有构造来源、原生候选、模型或 GPU。
[静态记录](hz_native_storage_lifetimes_20261018.json)和
[重算脚本](../scripts/assess_native_storage_lifetimes.py)绑定当前实现。

结论：**已把几项“未知”收敛为具体数量和生命周期，但完整归档生成需求与自有 heap
上界尚未闭合。** 本轮不能生成新额度或准入。尤其 8 GiB AS 只提供超限拒绝，
不能证明这些自有对象、检查与全部义务能在 300 秒内完成。

## 1. 三次完整遍历，不是三份矩阵同时驻留

同一注册八专家拓扑：参数 P=6,961,368，首层权重 W1=6,684,672，首层行
S=2,176。一次 Snapshot 的 packed 参数为 **55,690,944 B**。生命周期是：

| 阶段 | 同时可达的主要对象 | 结束后仍存活的对象 |
| --- | --- | --- |
| source build／encode | 一份完整 Snapshot＋Builder 图＋已编码 Packet 页＋行临时量 | Packet；Builder／Snapshot 不再被返回对象引用 |
| 独立 source check | Packet＋一份完整 Snapshot＋解析图＋检查临时量 | CheckedSharedForest、首层 SourceLocal 引用的 PackedRows |
| native setup／消费 | CheckedSharedForest＋首层来源行＋当前 native operator／workspace | operator 跨 tiles 保留，消费结束才关闭 |
| fresh replay | 新进程重新来源检查；不含生产的 native operator | 重算结果；不共享前一进程的 heap |
| reception | 新进程校验包、汇总完整输出；不再 lowering | 三阶段绑定及完整清单 |

checker 返回后，首层 PackedRows 为 **8(W1+S)=53,494,784 B**，其余 packed
参数可解除引用；不是继续保留整个 Snapshot。`packet=None` 也已在当前 production
与 replay 中存在，不能沿用更旧版本“生产全程保留 Packet”的结论。解除 Python
引用不等于分配器立刻退还 RSS。

按当前序列化布局，forest 二进制部分上界为 **5,329,156 B**；若 header 已通过
现行 8 MiB 门，页 payload 至多 **13,717,764 B**。这是条件格式范围，不是已经
证明任意完整来源生成的 header 都能通过。成员、引用及 Work 拒绝仍独立存在。

Checked forest 的持久 Fraction 位置上界为 `Q+R+2N+F=396,368`，其中
Q=L−W1=48,048，R=277,280，N=19,496，F=32,048。norms 至多 N 项采用
8192-bit 工作界；其余存储项来自 binary64，分子／分母至多 1075 bit。另需计算
Python 容器、索引及名字，不可把“位置数”直接写成字节。
checker 尚在运行时另有至多 N 项 `local_norms`。输入范围的文字解析发生在 Work
检查之前，不能把这些解析临时整数一概套用 8192-bit 界。Session 仍保留至多
504 个 endpoint receipt 和 504 个最终 Fraction；分页只约束结果页，不会释放
全部证明记录。Driver／Session 对同一 Receipt 的引用不重复算记录内容。

检查期 visibility set 总 membership 上界为 82F=2,627,936；规划下一 pair 时旧
pullback 结果尚可能存活，临时 Fraction 位置采用 N+2F=83,592，另加 scale groups。
非输入辅助 scale 被逐项验证为 1，故不同 scale 最多 n+1=3073；这不是假设所有
输入 scale 相等。生成器切换行时调用者旧行仍存活，采用两行至多 6146 个 Fraction，
不采用“一行”的过强生命周期假设。

## 2. 两臂显式数组：行块少存索引，但还不是实测内存优势

所有值均是**命名 tensor／buffer 的条件载荷上界或组成小计**，不含完整 Python
heap、Torch 内部 scratch、tensor wrapper、分配器和全部元数据。对 local 与
transpose 保守各计一份完整 COO，不依赖 Torch 是否共享底层索引／值。

| 常驻组成 | COO | 来源行块 |
| --- | ---: | ---: |
| local＋transpose | 48L | 48Q |
| reference 索引及系数 | 24R | 24R |
| scales | 8F | 8F |
| 自有 source bank 及 node 索引 | 0 | 8W1＋8S |
| **常驻命名 tensor 小计** | **330,081,664 B** | **62,712,192 B** |

旧静态行块小计 62,694,784 B 漏了 `self.ix(nodes)` 的 **17,408 B**，本轮显式补上，
不修改历史文件。来源 bank 不可写别名原始 PackedRow，所以上表之外仍需加共同的
53,494,784 B 来源行。

构造期 COO 的 `rr/cc/vv` 三个 Python list 各最多 **6,732,720 项**；行块对应
各最多 **48,048 项**。这些 list 与转成的 tensor 重叠，float／部分新整数及
reference tuple 也占空间，不能只计指针，更不能把它们归入未知原生 scratch。
行块现在逐行转码，只多一行最多 3072 个 Python float 及 24,576 B tensor；
不再用旧“双份最大自然权重块”解释当前转码。

保持全部实际对象，阶段组成小计为：

| 阶段（未加共同来源和 Python 对象） | COO | 来源行块 |
| --- | ---: | ---: |
| setup 公开 tensor 组成松上界 | 498,708,992 B | 65,036,800 B |
| 优化迭代＋常驻 operator＋已备 roots/scope | 379,256,296 B | 112,099,816 B |
| 精确消费 arena＋operator＋best dual | **440,813,552 B** | **173,444,080 B** |

setup 表是各构造子阶段的保守包络，不声称各极值实际同时出现。迭代保留旧
`points`／`residual` 和更新临时量；其额外显式数组公式保存在静态脚本，不能只算
一份 alpha/residual。Torch 内部 workspace 仍未知且另受硬限制。

关键接线是：`Proposer.propose` 返回后优化函数栈退出，只有 best dual 被移交；
**exact consume 完成后才 release DualTile**，而 operator 到全部 tiles 完成才 close。
所以精确检查时还活着 172,800 B best dual 和常驻 operator，不能只报
110,559,088 B arena；也不能反过来把整个优化栈峰值加进 exact 峰值。

fresh replay 没有 operator／DualTile，但 caller 的 raw frame 在 owned intake／seal
期间仍存活：至多三份 frame payload；精确消费小计为 110,922,032 B。下一次 read
赋值时旧 raw 与新 raw 也可能短暂重叠。

本轮首次脚本检查拒绝了旧 arena 文件哈希：版本确实不同。随后核对 13 项分配／
消费 AST 未变，并明确重推两项差异：RootTable membership interning 只减少 buffer
数量（另计新增 Python dict），shared objective 仍是至多四根的 SparseRoots。
没有删除版本检查或以文件名相同冒充实现相同。

## 3. 完整文件拓扑与清单重复现在可直接计长

令 Ns 为来源成员数（含 manifest），Nf 为 forest 页数，Nt≤84 为候选 tile 数，
Nr≤32 为结果页数。当前包固定包含 32 个检查器文件，则：

```
nP = Ns + Nf + Nt + Nr + 37
LP = 1915 + 42*Ns + 33*Nf + 32*Nt + 34*Nr
```

nP 是 package 文件数，LP 是其全部文件名相对请求根的 ASCII 长度和。代码路径
长度和为 1758，最长 64；来源 chunk、forest、tile、result 名均由当前格式绑定。
因此 **census 无需为外部绝对 root/source 路径重新猜限额**。

设一个非空 census 有 n 项，名字长度和 L，文件大小十进制位数之和 D，总字节
十进制位数 d，则 compact JSON 长度精确为：

```
C = 31 + d + 2*L + D + 74*n
```

当前字节字段范围给 `C≤40+2L+83n`。三份持久清单分别位于 reception、controller、
磁盘 terminal，文件数为 nP＋10／11／12，名字长度为 LP＋144／158／173；清单
编码总上界为 **3809＋6LP＋249nP**。这是真实重复序列化，不能按内存引用去重扣除。

完整请求目录有 **nP＋13** 个文件。磁盘 terminal 发布时不含其自身后补 hash／
完整 storage 标记，不能单凭它宣称 online acceptance；调用返回值才具有完整状态。
现有 batch 将返回值另存 caller receipt，若讨论联合交付归档，需要再计 **1 个文件
及其字节**。旧表的 request bytes 是请求目录范围，未包含外置 receipt；**其写出
时间原 runner 已计入 300 秒**，本轮不改历史计时或胜负。

## 4. 计数字段和 progress 不再是笼统未知

每次 owner 分配／复制均收费且单次不超过 B=128 MiB，故累计 allocated/copied/
released 各≤BW，allocations≤W，live/peak≤B。两种范围必须分开：

| 条件 | W 位数 | 累计字节上界／位数 | 全部成功 progress 上界 |
| --- | ---: | --- | ---: |
| 现行成功门 W≤8M | 7 | 1,073,741,824,000,000／16 | 16,432 B |
| 条件完整遍历 W≤165,180,263,928 | 12 | 22,170,119,734,856,515,584／20 | 17,992 B |

后一行不准入、不提额，其累计字节上界还超过 uint64，不能把旧 uint64 假设直接
套用；Python 整数可编码也不等于全部报告已闭合。production phases 至多 Nt＋15
项，编码≤`1+(Nt+15)*(140+3*digits(W))`。invocation 编码≤
`362+Pprofile+Qroot`，其中 Pprofile 是完整可信冻结 profile 编码长度，Qroot
包括引号和转义。路径、环境文件声明与版本字符串应在具体冻结对象上一次绑定计长，
不是每次补一个拍脑袋额度。

## 5. 还剩什么，以及下一步最小联合决策

目前不再缺“清单到底几份”或“哪份矩阵同时存在”。剩余联合合同需一次覆盖：

1. 可信 profile/root 的完整编码长度、来源 manifest／chunk 布局及生成的 forest
   header；不能沿用 95 个满块的旧假设，或用序列化后拒绝门替代生成需求。
2. 清理 PID／zombie 目录界与数值宽度。`live=[]` 不推出 `zombies=[]`；原生日志
   上限仍仅是拒绝合同，不保证任意原生库一定不超限。
3. 自有 CPython 对象与临时分配合同。对象数量已定位，但容器 capacity、整数／
   Fraction、字典／集合扩容及 JSON decode/encode overlap 仍要统一计费。
4. 完整新报告 schema、返回值与外置 caller receipt 的归档边界，避免再遗漏重复。

还须保留两个易漏项：bootstrap 的 `raw/inv/roster` 是顶层全局变量，在
`actor_main` 返回前仍活着，后者又解析一份 invocation；pinned manifest 的
`parts` 在合法短读下最多有 **4,194,304 个 bytes 元素**，join 时还有最多 8 MiB
payload 重叠，不能假定只有 64 个满块。来源 acquire 的 reader/copy/refs/raw 在
返回 token 后不再保留；相反 production 的 owned.inventory、outside census 及
Archive 的 entries/prefixes/events/attempted 会一直保留到函数退出。
后续对象合同还须计入 acquire 期间两个 TensorRows 各自的 2n 输入范围，以及
复制构造期间额外 n 个中心值；这些输入字段不属于上述 P 个网络参数载荷。

**继续方向是联合冻结这些表示／所有权参数，而不是增加又一层监督器或重跑三份
合成来源。** 先完成参数化的生成体积与自有 heap 合同；若必须改实现、资源数值或
执行范围，明确提出最小联合方案再审批。若不愿承担原生 COO 全规模 Python 列表
的构造成本，可另提一次有界构造方案，但必须保留强 COO 语义与完整检查，不能
把对手故意留在更弱配置。本轮没有实施该方案。

本轮说明了来源行块减少首层显式 COO 索引和构造列表的具体机制，但这是既有实现
的静态适用性分析，不是新发现的数学能力或实测收益。完整 source→checkpoint、
路由变化见证、外部净领先和 GPU 收益仍未建立；G1–G6 保持 OPEN。

## 验证

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B scripts/assess_native_storage_lifetimes.py --check
```

脚本只核对代码身份、AST、数量／字节代数和小型 JSON 计长差分，不导入生产模块。
33 项绑定及只读 AI 复核是工程证据，不是完整容量证明或独立人类技术审阅。
归档复核曾拒绝一个经过中间 JSON 数值工具舍入的大整数；现保留标准库整数重算的
原值 `22170119734856515584`。精确记录不得经 binary64 中间转换，也未以容差放行。
最终静态 `--check`、前一计数账本的 `--check`、精确 JSON 往返及 `git diff --check`
均通过；三路只读 AI 复核覆盖来源生命周期、原生载荷和归档计长。没有新增原生控制。
