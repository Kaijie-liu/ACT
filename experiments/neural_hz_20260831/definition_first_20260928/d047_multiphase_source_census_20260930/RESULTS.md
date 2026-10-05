# 多相位关系的真实三源尝试与限制

本轮完成区间参数种子的数学实现并通过完整 3781 项测试，随后实际处理原三模型。CIFAR100 large 和 medium 的完整源关系证据保存成功；TinyImageNet medium 在第四个固定窗口内耗尽原 256M whole-work 预算，故整个三源研究失败，不取得 source census 资格。版本只执行一次，失败状态及全部已有工件已封存，没有减少人口、提高预算或重跑。

这轮的研究进展是把数学关系接到了实际原参数与性质，并发现其首块非线性信号远少于表面上的包络数量。它不是正式解题、GPU 成果或完整 Neural-HZ 创新；正式增益为零，整体 Goal 继续 active。

## 定义候选的实际扩展

[THEORY.md](THEORY.md) 给出共同原激活的区间参数种子定理：对 q=ReLU(g)、g∈[l,u] 严格跨零、原 active 指示 alpha，若实际系数 c∈[cl,cu]，则

```text
min(0,cl) u alpha <= c q <= max(0,cu) u alpha.
```

原 BN 系数通过已有可靠开方和区间复合取得，不用中点。对两个实际消费者先在共同源身份上形成系数差，再将差值和伴随值的四个包络交给 D046。稳定贡献用数值界，padding 保持原零值，全部原 HZ bits 和零点合法相位仍留在 backbone。本轮实际程序没有修改 HZ 或安装 native 行。

四个新测试覆盖非点及异号系数、稳定源/零点/padding、共同源身份、自配对条件、D046 正控对应及默认关闭/拒绝语义。固定小维枚举只存在于单元测试，不是运行时相位搜索。

## 一次执行的证据

新 RUN 为 [d047_multiphase_source_census_20260930_v1](../../results/d047_multiphase_source_census_20260930_v1/exit.json)。统一会话 84914 已返回 exit_code=1，worker_exit=1；进程已终止，没有待轮询或后台继续跑的本轮作业。

完整继承 3777 tests、171 files，加四项得到 3781 tests、172 files。全部通过，13 条旧 warning，无 failure/error/skip。收集加执行 56.851330 秒，满足原 60 秒门；pytest 自报 43.95 秒。四项新增 JUnit 记录和全部精确 node IDs 均匹配。`mathematical_component_gate_passed=true`；这不覆盖失败的真实三源阶段。

worker 运行 90.338854 秒，supervisor 总时长 158.206522 秒。whole_work_used 恰为 256000000，下一项操作在预付时以 BudgetExceeded 拒绝。最大已消费单模型 work 为 82688377，证据 meter 用了 32168434/40000000；失败不是证据预算或 240 秒墙钟耗尽。

worker 的 RSS high-water growth 为 370892800 bytes，tracemalloc peak 为 148764198、metadata 为 15098944 bytes；分别加原 65536 reserve 后均在 1GiB 内。记录的 retained_entries=1609157 是已完成模型 ledger 的最大值，不是未完成 Tiny roots 的完整 entry 资格；失败后没有额外未计费遍历来补发证书。`memory_gate_passed=false` 包括整个研究未完成，不应误读为物理 RSS 超限。supervisor 的原两项观察门通过，仍无 aggregate 物理资格。

冻结后六份来源文件没有修改。结束后只读复核 freeze 的六份 SHA 和 exit 中全部工件 SHA 均一致；原 source/input/provenance drift 均为空或 false。原三源身份、完整旧依赖、decoder、GPU dependency 核查均未删减。

由于 worker 返回失败，监督器没有进入后续完整三源 geometry/evidence checker。两份 complete 文件的 SHA 与各自 receipt 已核对，事后审查遍历了其中全部 pair；这些都不补成三源成功证书，也不声称已做独立原拓扑重抽。

## 两份完整来源记录

以下是失败研究中已保存的 per-model 记录，不是三模型整体通过，也不是正式新增解。

| 项目 | CIFAR100 large | CIFAR100 medium |
| --- | ---: | ---: |
| 固定窗口 | 5 | 5 |
| 接收行 | 320 | 640 |
| 固定消费者对 | 160 | 320 |
| 真实源坐标 | 1600 | 1600 |
| 源外包严格跨零 | 10 | 53 |
| 四类输出 bound 中非恒定者 | 136 | 476 |
| clamp 输入 phase cube 严格跨零者 | 0 | 4 |
| 固定最后单锚比较的充分 gap 阳性 | 54 | 425 |
| 普通接收界跨零行 | 2 | 26 |
| 四条关系的物理坐标 nnz 上界总和 | 1196 | 4441 |
| 完整证据大小 | 1129502 bytes | 3056139 bytes |

large 全部 184320 canonical 接收槽包含 102400 真实槽和 81920 padding；medium 全部 368640 槽包含 204800 真实槽和 163840 padding。全部固定通道和无改善记录均保留。medium 的一个 direct 分支进入接收 ReLU，另一个原分支止于局部边界，连同 side consumers 原样记录，没有假造动态残差闭包。

TinyImageNet medium 已实际读取原模型和性质，输入为 [1,3,56,56]。日志确认完成窗口 (0,0)、(0,13)、(7,7)，累计 192 对，然后在 (13,0) 窗口内失败；该窗口及后续窗口不能记作完成。没有 Tiny complete 文件，不能从部分日志推断其关系计数或能力增益。

## 执行后证据审查

[POST_RUN_AUDIT.json](POST_RUN_AUDIT.json) 是对两份已保存 complete 文件的描述性审查，不导入或重执行候选、模型、LP/MILP。主代理用 jq 遍历全部 480 对的 1920 个输出 bound；跨零分类只读取有理端点分子的严格符号，不用浮点容差。独立审查者复核了四个 hinge 记录及以下稳定性证明。

对 difference bound，“相关接收门跨零”指 h 或 w 普通界跨零；对 companion bound 只指 w。large 的 640 个 bound 中有 6 个关联跨零接收门，其中 2 个非恒定、0 个 hinge 跨零；medium 的 1280 个 bound 中对应数目为 76、41、3。这是诊断关联，不是下一次按坐标挑选的执行菜单。

medium 的四个 hinge 记录如下。坐标仅用于复查原证据，不能当预注册选例规则。

| 位置及通道 | 关系 | 原接收界状态 | 目前可断言的结论 |
| --- | --- | --- | --- |
| (0,0)，38/39 | companion lower | h 严格负，w 跨零 | 尚未被简单稳定性审查否决，非冗余未证 |
| (0,7)，126/127 | difference upper | h 跨零，w 严格正 | 尚未被简单稳定性审查否决，非冗余未证 |
| (4,4)，52/53 | difference upper | h、w 均严格负 | r=t=0，且 G_plus>=0，故此行冗余 |
| (4,4)，96/97 | difference lower | h 跨零，w 严格负 | t=0、r>=0、-G_plus<=0，故此行冗余 |

这些 G_plus 符号结论对完整连续 phase cube 也成立，所以后两项否决不要求搜索某个整数相位。没有一项的两个接收门同时跨零。前两项仍只是待强比较信号，不是已证明有益的两条约束，更不是两个新 CERT。其余 hinge=0 的包络也不能据此统一宣布冗余；本审查没有完成所有关系相对原 HZ 的强比较。

以上冗余证明以比较系统保留已认证接收门稳定事实为前提；实际 production LP 是否已正确绑定这些事实尚未验证，不能外推为已测试 native 安装后一定无效。

由此得到的研究选择是：保留多相位定义和区间种子组件，但暂不将 first-bank 全部条件 gap 数量当作大规模部署依据。后续应按同一数学稳定性规则作化约并进行强比较，重点检查跨零接收端所保留的共同源关系；不能只挑上述两个坐标，也不能用再压一点构造预算代替关系本身的价值证明。仍需完成原三源人口，才谈 source 资格；当前失败版不补跑。

## 未完成事项与存档

`source_census_completed=false`、`source_census_qualified=false`、`all_stages_passed=false`。实际 HZ 相位列绑定、native 编译、GPU 算术/终端、完整物理成本、shadow、逐家族及全部 2413 回放均未完成。

数学桥梁已核对：现有 sparse exact/compact ReLU 均采用 alpha=(1-z)/2，终端又使用 z=2t-1。原 frame rebase 会清理 slot 表，跨层观察必须绑定真实生命周期及 decoder，不能把 ONNX 标签直接当终端列号。本轮没有绕过这些缺项。

正式成绩保持 1870/2413（1063 CERT+807 validated ADV），13 家族保旧要求不变。独立 E0 仍为 CIFAR100 25、TinyImageNet 36，共 61/400，不与 1870 相加。默认验证路径和历史结果未改，formal_gain=0；不能用组件通过声称已回放确认逐例零回归。

日期 2026-09-30；分支 redu-hz；commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。仅新增当前实验目录和唯一 RUN；旧冻结实验、未执行草稿、历史模型和 /data1/Kane/HyZor 保持只读，没有 commit/push。原九项 tracked 修改仍为 3806 insertions、57 deletions。依据 write-page 技能，将数学贡献、实测数据、事后条件证明与未获资格分开保存，没有发布外部 Page。

本轮属于 progress：新增了区间系数数学组件、完整数学门、两模型完整真实关系证据、Tiny 明确资源停止点，以及会改变后续部署判断的冗余证据。整体目标没有完成，也没有需要用户解除的全局阻塞。
