# 同 pair 检查映射生命周期有限控制协议

用户已批准默认关闭的实现及有限标准库差分／故障控制。本协议从
`2b95b29c99b603d36638fb7ddbee2e0175bd1715` 开始，不将自动续行视为实模或 GPU 批准。
先提交冻结代码／协议／测试哈希，再执行一个新目录内的批次；失败留档，不升限重试。

## 单一实现变化

`retain_pair_scope=False` 为 Driver 和 PairWorkspace 的新显式只读选项。启用时，仅同一
Session 私有 Plan 下连续 tile 可以持有一个 pair Scope。scoped 首次完整检查 nodes／
factors、canonical 顺序、ancestry 后封存；global 保留原全局遍历和逐次数学检查，只
增加不可变映射所需的四次封存，不人为加入 scoped 专用 canonical 复核。

PairLease 绑定原 Scope、owner、forest、累计 Work、Session／Plan、pair、四个 buffer
及原 sealed bytes 的对象身份。续入、消费和 pullback 前检查，全部 visit 计费。
每 tile 累加器、目标 roots、候选 token 照常失效；无活跃 tile 禁止使用保留映射消费。
最后一个 pair tile 由私有 Plan 决定，在消费末尾释放，先于新 pair 和终态来源封存。
Lease 撤销同时清除原 bytes 强引用；拒绝 owner／size 污染后按原分配元数据清理自有块。
失败终止整个 Driver，不重新构造、降级回退或清零累计成本。

这是现有进程内 API 信任边界，不是防任意私有内存篡改的 Python 沙箱。未修改原生
production／portable 配置，不自动启用新选项。原 source、dual／zero、符号、精确残差、
位数、范数、全部输出聚合与正裕度门不变。

## 固定来源和四臂

仅两份标准库合成声明：

- 原 `guarded` 不变，5 类、12 性质、24 端点、3 tile；无跨 tile 摊销机会。
- 新 `eight_private_six_outputs`：沿用旧 eight_private 的 router／隐藏层，最后 Linear
  保留原两行并追加四份第二输出行，真正生成 6 个输出。新来源及独立 hash，不修改旧
  来源或只伪改 declaration。28 pair、140 性质、280 端点、56 tile，每 pair 为 `[8,2]`。

四臂为 original-global、original-scoped、retained-global、retained-scoped。guarded 使用
已有 binary64／有理数两种固定候选编码；新六输出来源仅完整运行 binary64，合计
12 次完整运行。编码0按正序、编码1按逆序执行四臂。保留未决，不按正界选来源；
没有优化器、训练、真实数据或 GPU。

冻结前静态容量审查拒绝新六输出来源的完整有理数编码：旧 R2 的 eight_private
该编码共 3,618,496 字节（28 tile、56 端点），每端点步长
`(3618496-28*128)/56=64552`；新增输出不改变隐藏约束数，因此新完整编码需要
`280*64552+56*128=18081728` 字节，超过原候选累计门 `16769024`。
不提额、不执行这条完整配置；仅保留既定首 tile 有理数算术轨迹控制（516,544 字节），
不称为完整有理数运行。此决定发生在来源生成／执行前，不是失败后删除结果。

独立 expanded exact oracle 在四臂计时之外、共同批次时钟之内运行，3 个记录、共
656 次角色界比较；另做 2 次新位置来源复制及原始候选 receive（guarded 编码1、
六输出来源编码0），以新的来源检查器
重算。它们属于控制成本，不并入某一比较臂使其变慢。12 个冻结测试方法覆盖默认关闭、
全部义务、真正跨 tile 的 missing、原 bytes／owner／size／mutable／release 污染、
Plan／pair／ordinal／旧 token、旧 roots／map 失效、初次 roster 污染、半封存／续租／
末 tile 的截止异常、原 Work 上限、较小 arena 拒绝、发布失败、后块候选错误、搬迁和
固定首 tile 的精确算术轨迹。测试源码是有限 case 列表，不追加来源或循环搜索。

## 成本与资源合同

四臂记录 source 检查、prepare、候选编码／接收、发布 stub、精确消费、释放、终态及
总耗时；来源捕获／构造与 oracle 另列并计入整个批次。运行耗时包含控制断言和相同
记录处理，不作为真实请求加速比。Work 独立核对：

```text
component = scope_work + global_seal_work + lease_work + plan_work
          + setup_work + selector_work - planned_iteration_saving
```

同来源／同候选四臂的 `total_work-component` 必须相同；额外生命周期费用不默认抵消。
报告 original-global→retained-global 与 retained-global→retained-scoped，不能把第一项
重建节省称为遍历创新。保存全部精确 receipts、候选身份、完整义务、终态和成本。

每请求保持 8M Work、128 MiB arena／制品、8192 中间位／4096 存储位；整个控制批次也
最多 300 秒，包含启动、导入、捕获／传播、检查、oracle、搬迁、终态发布。worker 使用
8 GiB 虚拟地址硬限额、2 GiB 进程组采样 RSS 停止门；不新增额度。外层监督器仅终止其
自有进程组，超限／截止保留终态，迟到报告不算 PASS。部分或结构非法的 worker 报告
保留为失败制品并记录解析错误；报告解析控制纳入原截止／异常测试方法，不将它称为
操作系统级强制终止实测。自有进程已退出的 kill 竞态仍进入终态记录。样本、尝试和额度均不追加。
arena payload 与字节强引用清理不等于 CPython／原生全堆证明；新 tuple 元数据和
生命周期增长须如实描述。独立记录审计在执行后单列，不能补成在线成功。

## 继续和停止

验收是同候选结果和全部义务不变、故障关闭正确、差量费用可解释，以及观察有无组件
收益。不是必须新增正结果，也不要求所有来源获益。无论结果如何，本批结束后保留旧
R2；不得外推注册完整拓扑准入、G1完成、真实同源证书、外部领先或 GPU 能力。
两次源参数检查至少 13,922,736>8M 与 membership 限制仍在，真实执行仍需单独决策。
