# 端点对偶交换控制驱动与成本准备

2026-10-19；起点 `cccda4f9d15a157acc27a851794122779c6c2d56`，
`feat/moe-route-verification`，起始工作区干净。用户要求继续，沿用默认关闭实施及控制准备
范围。本阶段没有执行组件、来源工厂、原生候选、模型或 GPU，没有修改资源额度或批准表。

**已补组件驱动与封闭故障接缝，尚未完成整批执行冻结。**
本阶段是[有限控制草案](hz_endpoint_dual_implementation_20261019.md)的实施准备，
不是控制结果，不增加 G1–G6 的科学验收成绩。

## 已接通的控制路径

[组件函数](../scripts/endpoint_midpoint_components.py)没有 CLI，也不会在导入时运行。
未来只能由验证过冻结身份的、尚未导入 Torch 的独立受管进程调用；它首先检查批准 profile、
方法、来源和代码绑定，再确认 8 GiB AS／128 MiB FSIZE。完整 RSS 监督仍属于外层。
V2 继续拒绝新方法；V1 的旧 profile 因实现 hash 不匹配，不能授权这些新控制。

组件包含原草案的 22 个具名位置，按同一 Work 与绝对截止计费，具体覆盖为：

- 旧默认方法字节身份、宽 8／宽 2 和真实退化 gate；缺 mate、重复 slot、错 pair、
  性质、目标、来源、frame、token 与撤销租约的拒绝。
- 同步读取两个 best 后写新中点，以及输出修改不污染 best。错误来源绑定必须拒绝；
  同框架合法对偶数值换序则按原始系数精确重算，不把另一份合法候选误标为非法。
- 正 inequality 对偶拒绝，非有限值拒绝；极大／极小有限中点进入原完整精确消费者，
  包含原系数、残差、位数及 zero 角色检查。
- 第 127 号评价后的异常、早期／中途拒绝后的输入不变与累计计费；固定候选逐角色
  与独立展开 scalar oracle 差分；本地新包搬迁及 method／invocation 错绑拒绝。
- 已完成评价／forward／更新计数：正常双端点 `129/128/127+1`，单端点
  `129/128/128+0`，交换前异常前缀 `128/128/127+0`。

第 18 项的拒绝由有限测试替身注入，实际发生的访问仍转交同一个 live Work。
它覆盖早期和 helper 中途的失败处理，**不是实际消耗完 8M 或实测 OS 截止**。
原 Work 门不改，真实硬截止另由草案第六位置检验。不能把组件 PASS 汇总成所有 OS／
资源故障都已实测。

组件精确限制三份来源：`guarded`、`mixed`、`half`。guarded／half 使用已有
`direct_supervised.fixed_source`，不经历史实验脚本生成；mixed 沿用原固定声明。
有三次正常单 tile 搜索（guarded 宽 8、mixed 宽 2、half 退化），以及一次 guarded
交换接缝异常搜索。寻找 half 的退化 tile 时，前缀义务使用固定候选完整消费，不额外搜索。
所有这些次数须进入将来的绑定清单；它们不是四个正式两臂请求的计时成绩。

## 首位置不借用后续或历史包

第一个组件位置本身生成 mixed 的完整固定候选包：全部六端点、每端点 proposed／zero，
共 12 个角色。构包沿用 `Packet.publish`、`Driver`、原精确消费者、`Writer`、
`Archive` 和原包 schema；不调用优化器生成这份证据夹具。

随后保留原包，复制到新位置，使用复制后的检查器运行 `python -I -S -B`，不读取
checkpoint／训练数据／历史目录。复制、代码、重新检查和子进程成本继承本位置预算，
不会另给 300 秒或 8M。包里的方法身份不能被用来声称 fixed dual 实际由 midpoint 搜索生成。

[原生故障链](../scoped_source/native_production.py)沿 V1 production → Lifecycle →
Proposer 传递封闭的 `midpoint_block`／`midpoint_error` 枚举。它仅适用于新方法及固定
合成来源，不能通过真实来源 profile 开启，不接受任意函数或全局 monkeypatch。
故障位于第 127 号评价、best 更新及 forward 之后，先输出具名到达标记；旧
`proposal_block` 在搜索之前，不能替代这个到达证据。
部分候选位置继续复用已有 `partial_candidate` 截断发布，不新建故障框架。

## 候选侧计量

设一个非退化 tile 有 `k=2p` 列、`I/E` 条 inequality／equality 行、`C` 类和
`G` 个 Plan pair。mate 绑定与两个中点 helper 的显式新增计费合计：

\[
\Delta R=G+k+3p,\qquad
\Delta V=Ck+p(17I+14E),\qquad
\Delta A=1+k(1+2C).
\]

零行组仍遍历 mates，因此其 `p` 个 row visits 不省略。退化 tile 没有中点 helper，
对应 `G+k+p`、`Ck` 和同一个算术项。组件为断言 mate 会先调用一次绑定 helper，
候选器内部再调用一次，这两遍均计费，不能把组件开销套成生产仅一遍的公式。

这不是端到端 Work 差公式。候选变化会影响精确残差的非零项、位数检查和停止前缀；
后续比较必须报告实际完整累计计费，不预先承诺节省或新增证书。

## 对象寿命与制品合并

中点的 `8*r*(k+3)` 仍只是局部 float64 payload。优化循环还持有两组 best／迭代、
alpha、a、value、residual、points、good 及算子临时量。特别是梯度循环局部 `z` 可在
替换 group 的迭代后继续保留旧 equality 张量，不能忽略到 propose 返回之前的寿命。
本阶段没有为了减少该引用而顺便改动搜索行为。

组件与生产的寿命不同：`sources` 保留三份 packet 与 checked forest；oracle 另有
mixed 展开矩阵、frame、候选和 receipt；归档保留 entries／prefixes／events。
搬迁检查时父组件仍持有这些对象及 Torch，子进程另建 mixed 来源、forest 和检查器。
因此搬迁阶段应合并**父组件＋子检查器＋外层调用／监督对象**，不能取单 actor 最大值。
原生分配器／scratch 仍采用硬限额、超限拒绝合同，不伪造完整 heap／RSS 上界。

令 `A` 为三个来源工厂全部目录（含模板），`B` 为完整 mixed 包（含代码及 bundle），
`X` 为外部 binding，`Lo/Le` 为搬迁 stdout／stderr，`Rc` 为组件报告，`Mw` 为
外层 log／receipt／元数据。一个组件位置必须满足：

\[
A+2B+X+L_o+L_e+R_c+M_w\le128\text{ MiB},
\]

\[
N_A+2N_B+1+2+N_{R_c}+N_{M_w}\le4096.
\]

代码对两份包累计归档；partial progress 至多 64 KiB，搬迁两条日志各至多 1 MiB，
组件报告预留 1 MiB。**外层 `Mw` 尚未绑定，现有 8 MiB 预留不是成本闭合的证明**。
失败前缀也要保留、计费；无法恢复的 Work 为缺失或观测下界，不是零。

## 执行冻结前仍需完成

不增加草案的七个位置、22项组件、来源或搜索配置。下一步集中完成：

1. 新批次只读验权／终态审计接口，绑定本组件、正常及故障请求的代码、来源和环境。
   不运行旧 freeze 脚本：它会生成来源并修改旧批准表。
2. 单位置起点涵盖准备、生产／组件、检查、回执持久化及清理；批起点涵盖初始化与收尾。
   复用既有 V1／owned 监督；不能把 `execute` 返回后的回执写入当成免费开销。
   七位置累计 2100 秒、每位置 300 秒；剩余不足完整位置及收尾时不启动。
3. 明确组件与搬迁子进程的硬截止、部分输出／累计 Work 接收、整个受管组清理与终态，
   并闭合 `Mw`、成员数和联合对象账本。`subprocess.run` 的局部超时不是整批监督已完成。
4. 静态核对后单独提交不可变冻结，获得执行批准与新鲜资源准入。当前没有任何新 profile
   登记，不能因组件函数存在就调用它。

本阶段仅进行了 AST／编译不执行检查、旧路径源码差分和只读 AI 复核。没有运行22项
控制，也不以控制代码数量关闭 G1。注册八专家的来源 Work／membership 准入拒绝仍在，
这项候选搜索假设不能消除来源构造阶段的拒绝。

静态检查记录：既有 `check_native_integration_static.py` 通过（12模块、60本地依赖）；
五个修改／新增 Python 文件单独解析编译通过；22个具名完成位点与固定清单一一对应。
剔除新故障分支后的搜索循环 AST 与起点一致，十个方法／数学检查／包／原监督／V2及
批准表文件逐字节未变。两项只读 AI 复核发现的拒绝文案、完整精确消费及来源依赖问题
已在草稿中修正。这些均为静态结果，没有调用测试函数或被测模块。
