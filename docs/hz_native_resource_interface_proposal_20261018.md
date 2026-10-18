# 统一自有资源接口提案：保持旧额度，尚未实施

2026-10-18；起始分支 feat/moe-route-verification，HEAD
61c6782131ef7380eeea606f1b0ef5b517312c6f，工作树干净、远端同步。
本提案落实[联合账本的下一决策](hz_native_joint_contract_20261018.md)。
[机器记录](hz_native_resource_interface_proposal_20261018.json)与
[只读检查器](../scripts/check_native_resource_interface_proposal.py)绑定当前源码。
**它们不是执行配置，不会填入审批表，也不批准实施、控制、提额或运行。**

## 一、这次做什么，不做什么

只提出一个共同、默认关闭的接口版本，使来源／profile接收、元数据构造、失败观察和
最终接收都能在构造前有明确的自有资源边界。两臂共用这一接口，不再分别修四个小包装。
已有共享来源、精确检查、原生COO／行块、8宽候选／2宽检查均不重新命名为创新。

这解决**资源合同与接收可靠性**，不是新的MoE松弛或支持算法，也不能使八专家形状
在旧额度内可运行。完整自有heap、成功／失败归档联合上界仍未闭合。后续要研究完整
模型，仍需单独的联合表示／资源政策决定；不能靠本接口绕过现行Work和逻辑成员门。
所有数学检查、数值门、全部义务和同一300秒保持。实模、checkpoint读取、Torch候选、
GPU、训练、后端搜索、封存输入均不在本轮范围。

## 二、当前源码复核新增的具体依据

| 当前位置 | 已有检查 | 尚缺的构造／读取保证 |
| --- | --- | --- |
| native_protocol.approval_registry | stat及读后65,536字节门 | 中间read_bytes无读取长度；文件增长时，事后拒绝不限制首次读入 |
| native_protocol.closure／verify_code | 128 KiB代码文件及身份检查 | stat与全量读取分离；AST前读取也需有界 |
| native_supervised.command中的bootstrap | invocation 1 MiB、代码128 KiB及hash | 同样先stat后全量读取；actor AS不能替代每个自有输入的读取合同 |
| validate_profile／write_json | 授权hash、最终编码cap | 首次identity／compact发生在部分形状／容量检查之前 |
| owned_bounded.observe／parent_rss | 截止、采样RSS和错误拒绝 | proc逐项读取没有完整性与构造前容量合同；外层caller不继承actor AS |

这不是声称已经发生了文件增长或内存攻击；是源码允许的路径，不能由正常控制结果排除。
只读检查器用AST定位上述读取，包括嵌入bootstrap字符串，没有导入生产模块或读取模型。
旧控制结果原样保留，不追溯改判。

另外，当前check_result_stream.aggregate允许完整记录中出现已检查的UNKNOWN。
**完整执行／完整报告接收，不等于全部输出性质为正。** 新接口必须保留这种语义：
不能为简化监督把UNKNOWN改成SAFE，也不能把完整、合法的UNKNOWN报告错称损坏包。

## 三、不可变身份链：不能由包或调用者自批

使用下列单向依赖；箭头表示依赖左侧已有身份，不表示免费共享另一臂的工作。

~~~text
共同合同 C → 来源布局 L → 冻结封套 E → 外部审批表 R
                  C + E + R → 在线调用 I
                  I → 所有权账本／观察 → 终态与caller回执
~~~

E还直接绑定C。**E不能包含审批它自身的R的hash**，否则形成自引用；I才绑定实际读取的
R修订hash。实施／数值政策／执行权限分开登记。新schema默认拒绝，旧审批不能跨版本
继承。旧运行目录、冻结文件及实现身份不覆盖；不全局monkeypatch POLICY。

### 1. 共同合同 C

固定完整原POLICY及计数语义、源码闭包、数学接受合同、资源公式版本、绝对截止安排、
两臂公共算法和唯一表示差异。文件、逻辑成员、唯一root、requested对象、观测RSS和
虚拟地址空间用不同单位，不相互抵扣。所有权租约结束不等于分配器RSS立即归零。

当前公共数值只引用原门：8M Work、128 MiB总制品、4 MiB单tile、16 MiB−8192累计
候选、4096 stored bits、8192 work bits、300秒。引用边65,536、展开membership
65,536、local nnz 1,048,576、active factors 16,384仍独立检查。
**128 MiB不是heap额度。** 新own-heap数值保持null；已有8 GiB AS／2 GiB采样RSS
只保留其已批准控制身份，不据此批准新执行。不得用165B条件Work上界作新额度建议。

### 2. 来源布局 L

只引用原始声明和有限布局摘要，不复制参数、不另造网络表示。至少绑定：

- manifest原始长度／hash、原规范来源identity／编码长度、request身份／长度；
- 实际tensor/chunk清单及连续覆盖、K与rank、文件名长度、成员及总字节；
- 严格整数tensor shape与原语义允许同值bool／float的request shape分别绑定；
- JSON容器／条目／字符串／数值token／深度账本；optional trust的存在、类型、
  长度与身份不能省略、剥离或默认浅层；
- 生成／捕获来源的成本范围：本入口仍是声明来源，不自动证明checkpoint、BN/eval、
  预处理或原生浮点程序对应。

L来自可信冻结只允许预留，不替代在线验证。请求内仍读取、核对hash和必要词法统计，
保留全部原schema、值合法性、两遍独立lowering、输入包含、guard、残差、聚合与seal。
扫描器自身也必须使用有界缓冲／深度与计时；不能先完整解析再声称已提前预留。
不能将请求特定的规范化、传播、范围求解或证明生成移到冻结阶段免费执行。

### 3. 冻结封套 E 与在线入口

入口只接精确类型的有限引用／不可变字节，不先对任意dict、循环对象、自定义对象调用
identity、str或repr。E绑定C、L、方法、实现／环境／ABI，以及profile、root、run-id、
version、property的原文／编码长度和hash。身份核验必须从外部审批锚开始。

来源路径与输出目录仍检查规范性、所有权、symlink/hardlink及身份变化。审批表、
invocation与代码文件采用**带长度的读取、哨兵超额检查、同fd类型／稳定性核对**，
沿用各自原有cap；失败即拒绝，而非先读任意大小再报错。不能为了减少重复读取，
取消必要的执行前身份验证或来源seal。

## 四、构造前预留、别名与完整成本

元数据不能仅提供write_json(obj)事后查长度。拟议接口先接受封闭schema／字段描述符，
对编码及自有对象需求预留，再创建容器和序列化；未知schema、超额、别名污染或描述与
实际不符均拒绝。计长与最终编码使用同一规范字节语义，并独立核对输出；不能先通过
compact生成完整副本再称其为“计长”。有效旧数学结果及身份语义不改变。

所有权账本按typed block记录，不为每个Fraction再生成一份巨型证书：owner/generation、
来源／调用／方法身份、阶段、公式、requested bytes、live／peak、wrapper／lease与释放。
共享backing按身份去重，但wrapper和复制计费。Snapshot、两份输入、旧／新两行、
RootTable、候选、精确arena、结果、metadata与Fraction哈希LRU分别归类；仍被引用的
对象不得记为已释放。账本一致性不是完整内存证明或数学证明。

完整模型准入前还须逐相位合并caller＋controller＋active actor，包括成功与异常路径。
新封套／布局／账本本身的生成、读取、编码、hash、文件与对象也收费；不能直接继承旧
归档常数。额外metadata遍历的计时／字节／对象费用必须显式列出，原Work计费点与累计
语义不删除、不缩成每块、不中途重置。任何新增计费项应在实现审计和后续冻结中列明。

未知库scratch沿用已采纳的硬限额拒绝合同，但项目自有对象不能归入该例外。没有来源
布局、阶段联合上界或所需reservation，就不准入完整形状；不以一次运行未OOM替代它。

## 五、有界观察与失败终态

观察结果至少包含调用／pgid／阶段身份、时间、complete/reason、已观察条目和字节数，
以及live、zombies、rss。proc枚举、逐文件读取和名单追加都必须增量有界并检查截止。
不预设no-fork，不假设live为空就没有zombie。保持原先条目消失时的处理边界；权限、
解析、超额、未扫完或截止不能通过默认空列表／rss=0来绕过清理判断。

观察所需字段的上界从可信合同与剩余表示预留导出；不在本草案任意指定PID数、
字符串长度或内存新额度。若不能在原cap内完整表示，就保留complete=false并拒绝。
完整一次扫描也不是全OS原子快照；不逃逸假设、OS阻塞及bootstrap可信范围继续披露。
运行超时后仍只清理本次受管进程；不能因观察失败放弃终止，也不能误杀其他任务。

**完整观察本身不足以接收。** 保留原`run.status == COMPLETED`和
`LEADER_REAPED_NO_LIVE_GROUP`门：leader必须已回收，完整且及时的最终观察必须
确认`live == []`。观察完整但仍有活进程，仍拒绝。zombies须完整记录，不能据此
声称孤儿zombie已被回收；原接收门不要求把此类记录伪装为空。

异常使用有限类型码和阶段码；不调用任意异常的str、repr、args或traceback格式化来
构造终态。已有有界日志可以单独绑定，日志超额和截断不得伪装成完整诊断。
诊断信息减少不允许减少原数学／身份检查。部分文件和失败前缀仍在分母和成本中。

状态转换与拒绝锁定：

~~~text
UNBOUND → BOUND → RUNNING → ACTORS_COMPLETE → CLEANUP_VERIFIED
        → RECEIVED → TERMINAL_DURABLE → CALLER_RECEIPT_TIMELY
任何失败：锁定online拒绝；已启动则继续有界cleanup；不重试、不迟到升级。
~~~

分开保留execution_status、原evidence_status、online_report_received和online_positive：

- 完整、及时接收的UNKNOWN报告可以是执行COMPLETE，但online_positive=false。
- 完整义务记录明确标记missing endpoint，与传输缺行／少pair不同；后者不能接收为完整。
- 正结论须额外满足原全正聚合、来源范围和全部数学接受门，不能靠metadata的COMPLETE。
- 外层caller回执单独计费，不在原13个request-root文件中。仅execute返回true还不够：
  还需持久化caller回执并完成最后截止观察。保存失败、缺失或迟到均不能是在线成功。
- 先有正报告、后清理／保存失败，仍是失败；可以保留部分result与error，但不覆盖原证据。

所有步骤从入口前t0共用300秒，保留原299／299.5／300的执行／清理／终态安排。
等待执行的准入有界并单列；执行内等待、预留、失败、释放和序列化均计入请求。
合作式检查不承诺OS延迟时物理返回必在300秒内；超额完整报告并拒绝，离线重查不补成功。
回执不能通过循环自hash声明包含自身写完时间：保存结束后的最终观测由上层执行账本
记录并绑定回执hash，未取得该观测就不是已确认完成。

## 六、两臂公平边界与仍然拒绝的完整拓扑

两臂引用**同一份**共同合同hash；仅representation、相应实现hash、独立目录和调用身份
不同。各自构造来源与事实、各自提议／检查／归档，不能复用对方答案或免费历史census。
公共算法保持CPU float64、单线程、128次更新／129次评估、profile0、proposed＋zero两角色、
候选宽8、production/replay均pair-scoped宽2。同政策不要求实际Work或耗时相等。
没有在这里加入domain-only视图、零角色新编码、cursor复用、GPU或新松弛。

| 旧静态拒绝／缺口 | 仍成立的结论 |
| --- | --- |
| 首次Snapshot至少13,926,024 Work > 8M | 当前八专家接线必然拒绝 |
| 两遍逐参数合法性13,922,736 > 8M | 少做部分重复构造仍不能通过 |
| 展开membership 126,632–136,232 > 65,536 | 当前布局必然拒绝；这不是唯一root数 |
| nnz／refs／active及22,868,160字节候选流上界超过各自门 | 不能保证完整容量，不表示实测每次必超 |

shared_format同时保存唯一roots和展开逻辑计数；历史inventory的mandatory_roots命名
在这里明确解释为**展开membership**，不改写冻结数据。metadata修复不解决这些门。
原完整遍历165,180,064,388／165,180,263,928 Work上界也不是推荐配额或时长预测。

## 七、实施前决定与有界验收目录

当前产物只完成设计和源码适用性检查。后续最小请求是批准**本联合接口的默认关闭实施**，
不顺带批准新资源值或运行。实施后先静态审计完整调用范围；控制身份、次数与累计预算
另行冻结，不能沿用旧26个控制身份重新刷数。未来只需一次有界验收覆盖以下六类：

1. 封套身份／别名／超大与循环对象；调用者不能自批或跨臂／跨请求借用来源。
2. registry、invocation、代码及来源的增长／替换／短读；旧schema与身份门保持。
3. Unicode计长、构造前容量、部分写出／fsync异常；result与error同时存在仍有界。
4. proc消失项、权限／解析错误、过长记录、live／zombie名单与扫描截止；不完整不冒充空。
5. actor截止、异常、部分数学证据与metadata超额；cleanup继续，所有原义务仍要求覆盖。
6. 终态与caller回执的缺失／迟到／旧身份、成本遗漏；不能由落盘正报告恢复失败。

这些是**未执行的验收类别，不是已经通过的控制，也不是完整批次冻结**。
尚未绑定的参数／来源／授权保持null，控制准入次数为0。

实施批准不等于完整形状准入。若所有权合并仍缺项或旧门仍拒绝，停止在该阶段，明确
提出完整联合表示／资源政策问题；不再做“只修一个字段，再试一次完整模型”的循环。
没有算法增量、同源完整正证或公平净收益，就保持G1–G6 OPEN，不以控制数量关闭目标。

## 八、本轮核查与交付范围

三个只读AI复核分别检查来源／ownership、候选／公平政策、观察／终态。
本轮新核查定位了registry与bootstrap等stat后全量读取窗口，并厘清UNKNOWN报告和正
证明的不同接收条件。只读检查器验证文件身份、原拒绝算术、共同配置及无自引用的身份链；
它不执行提案中的接口，也不是完整程序验证、人类技术审阅或网络证明。
复核指出的“观察完整不等于清理成功”已补正：保留原完成状态、leader回收和live为空
的联合门，不以新增complete字段取代它们。静态记录同时保存原POLICY全部15个字段，
从源码AST读取，不执行生产模块；提案正文也纳入hash绑定。

~~~bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B scripts/check_native_resource_interface_proposal.py --check
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B scripts/assess_native_joint_contract.py --check
git diff --check
~~~

两项静态检查及diff检查均通过；修正后观察／终态复核无剩余阻塞。
“通过”只指本草案与现有源码／约束一致，不是运行控制通过或完整资源上界已闭合。

改动只限提案、只读检查脚本和导航。未改审批表、生产模块、资源值或历史结果；
没有新来源构造、候选、模型、GPU、依赖安装或运行控制。
