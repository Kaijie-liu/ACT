# 分页结果实施后的资源核算

**分页结果接口已经完成；完整八专家工作负载仍不准入。** 本次绑定实现
`f87e4c7ef958035a89cd87aecb50fc8885e82206`，核对[171 项实施控制](hz_paged_result_controls_20261015.md)
中的 50 个文件身份，另核对来源布局的 `capture.py` 及继承账本哈希，补齐实现后的归档、读取和存活对象账本。未改运行代码、额度、
检查、接受门或 300 秒预算，也未再运行来源构造、候选、worker、模型、求解器或 GPU。

[核算脚本](../scripts/assess_paged_execution_resources.py)与[机器记录](hz_paged_execution_resources_20261015.json)
保留旧账本的条件，不重写旧结果。九项静态算术／身份测试通过；只读代码复核不替代
独立人类审阅，也不是这些程序的机器检查证明。

## 已解决什么

分页模式下，生产与重算结果均为摘要＋引用，不再在 worker report、父收据中复制
完整结果行。全部原始候选仍独立消费，全部端点仍重算，全部行仍比较。新增模块只调用
`Work.tick()`，不增加旧 `visit`／`exact` 计数；`Session.finish` 显式保留旧九字段计费。
所以原[计数公式](hz_shared_execution_envelope_20261015.md)在原假设下仍成立，**不表示新增
解析、编码、检查和 I/O 没有实际成本**，也不是总计算成本上界。

原条件化源子路径 Work 上界仍为 216,424,736；完整 post-capture 的 dense／pair-scoped
计数上界仍为 89,433,531,656／165,108,619,256。它们是使用全局闭包的宽松上界，不是
运行时间比较、建议新额度或可完成性预测；分页不会消除来源检查的既有下界。

### 完整成功归档

以下沿用[原固定来源布局](hz_shared_archive_contract_20261015.md)：注册拓扑、每 tensor
1 MiB 满块、固定 ASCII trust、有限输入域字符串，无额外 provenance；不能泛化为任意
来源或任意失败输出。生产监督器仍只有固定小源入口，没有注册全尺寸源适配。

| 成分 | 条件化字节上界 |
| --- | ---: |
| 来源＋forest | 62,759,937 |
| 便携代码，28×65,536 | 1,835,008 |
| 新实际 schema 的完整成功元数据 | 1,702,743 |
| binary64 候选流 | 22,868,160 |
| general-rational 候选流 | 2,831,559,360 |
| **binary64 完整请求** | **89,165,848** |
| **general-rational 完整请求** | **2,897,857,048** |

文件数为 96 来源＋108 forest＋84 候选＋28 代码＋33 结果＋4 合同／清单／终态＝353；
bundle 排除自身及内部终态，包含 351 个成员；再加外层五个文件，共 358。
代码按原每文件 64 KiB 上限计，而不是沿用旧 26 文件的实际字节数。

binary64 的**整体**上界低于 128 MiB，但候选流上界超过原独立 stream cap 16,769,024。
这表示缺少统一容量保证，不证明每个实际候选都会超限。general-rational 的整体及部分
单文件上界同样不获保证。不能只检查总归档就宣布成功准入。

### 在线结果读取与实际检查

结果页和结果 index 都完整读取十次：child 五次、parent 五次。逐页解析发生在
fresh replay 和 parent aggregate，其余八次是 inventory／archive／census 哈希读取。
Writer 生成时也校验每条行，因此每条非缺失行有三次新增的 Fraction 解析、规范形式、
位数和精确符号检查；不是十次解析，也不能因未计入旧 Work 就省略真实耗时。

全部页上界 1,261,868，加 index 5,431；这部分在线逻辑文件读取上界为
`10 × (1,261,868 + 5,431) = 12,672,990` 字节。不是物理磁盘流量，也不包含其他元数据、
代码、导入、内存 hash、未来捕获和原生候选。旧 core I/O 公式已包含 parent，不再重复相加。
离线终态审计另外四遍，搬迁重算另计，均不能救回在线超时。

## 仍不能作出的内存与截止保证

“Writer 最多保留 8 行”只是一个组件的持久行缓存限制，不是整条证明链只有一页内存：

- 绑定私有 roster 时仍临时建立最多 252 条 duty、504 个 slot 引用及其规范编码。
- Session 仍持有全部端点值与 receipt；这些没有因结果分页被删掉。
- Reader 换页时旧解析页会存活到新页解析返回；另有 raw、JSON 字符串／bytes、Fraction
  和规范编码临时对象。不能以一页文件字节数替代整个 Python 堆容量。
- 生产 Packet 保留到消费结束；replay 在消费前释放 Packet，但最后候选 frame 会与
  Comparator 存活期重叠。源图、规划字典、visibility sets、norms 和 allocator 保留均仍在。

完整 heap／RSS、原生捕获和候选工作区尚未闭合。2 GiB 是现有 child 采样规则，不是
整个进程树及父接收的 OS 内存限制。分页固定错误码消除了任意异常 `str`／`repr`，
但不能据此推出任意故障的完整内存上界。

300 秒仍是统一的**及时接受截止**。child 有外层终止监督；父层最终 receipt、census
和异常 postmortem 没有硬抢占，可能在截止后返回并拒绝／降级 TIMEOUT。因此不能
宣称整个监督调用必在 300 秒内返回。无 durable receipt 的终态仍是
`INCOMPLETE_TERMINAL`，`cost_complete=False`；仅存储核对完整不能冒充完整成本审计。

## 停止点与下一决策

本接口阶段到此完成，不再追加同类分页／复制微控制，不尝试全尺寸准入或只移动 dense
所有权。当前两项确定拒绝仍为：两次完整源检查至少 13,916,160 Work＞8,000,000；
mandatory expanded memberships 至少 126,632＞65,536。它们与候选字节的**上界缺口**
是不同性质的证据。

仅减少序列化、缓存或对象副本不能解决这些下界。下一项有意义的运行时研究必须先
说明将改变哪一个算法／表示机制，以及如何保留完整独立检查和旧计数；或单独提出
完整、版本化的资源合同并获批准。不能重定义计数后假装仍沿用旧配额，也不能拿这里
宽松的 Work 上界直接定新配额。本次没有提出新额度，也没有启动这类新实现。

G1–G6、真实同源证据、外部竞争、GPU 收益及历史来源缺口仍开放。该账本完成不等于
新的 HybridZ 算法效果或真实证书。

复查命令（仅标准库静态核算）：

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -S -m unittest scripts.test_paged_execution_resources
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -S -m scripts.assess_paged_execution_resources --check-report docs/hz_paged_execution_resources_20261015.json
```
