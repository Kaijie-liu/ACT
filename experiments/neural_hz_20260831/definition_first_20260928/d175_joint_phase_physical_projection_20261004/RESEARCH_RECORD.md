# D175 研究记录：共同相容性必须落到物理输出

日期 2026-10-04，时区 Australia/Sydney；本轮于 09:41:35 UTC 开始。Goal 保持 active，定义优先、GPU、13 家族、CIFAR/Tiny、smooth/Transformer 与最终同路径全量提升的范围不变。困难不是外部阻塞，本轮有新的纸面推进。

相对 D174 的变化：D174 四门控制被普通原 HZ 四行 LP 蕴含，只是恢复弱代理路径丢失的关系。本轮给出完整单门 source hull、各对 D035 源因子及一致纯三相位分布都允许的明确物理点；已知 I3322 经神经读出投影排除它，并使下一个带 live skip 的 ReLU 稳定。没有把自由辅助矩不相容冒充物理输出增益。

主代理和两位独立代理从二十个等号点的均值分别推出同一分数点；主代理及代理核对单门内部源见证、共 delta 区间、纯三相位分布、非零偏置版本和后继读出。同轮稍后还得到三个完整两门联合 guards hull 的显式四格内部源见证，另存 PAIR_HULL_STRICTNESS.md，升级比较范围但不宣称超过三门或全网络理想 hull。另一代理给出固定线性商的相位乘法闭包条件及普通混权负控。全部是精确有理纸面推导和人工交叉复核，不是机器证明或执行测试。

新颖性状态明确为未成立：Bell 不等式、RLT 关系和线性换元都是已知机制。本轮有用之处是可直接生成物理行、无需九乘积、严格旧端延拓和明确跨一层消费；它们是后续新域定义的支撑，不是给旧 HZ 加 helper 后宣布突破。任意深度闭包、真实权重上的收益、统一三元组选择、GPU 与完整端到端费用均未解决。

主代理读取 BBQP 原文 Theorem 2.3、式 (2.1)、Corollary 2.11 和 §3.2，并核对本地 D035、D052、D126、D174。代理补查 Tsay 等输入分组工作：该分组是神经元求和项分组，不是输入域 split，但不能据此搬入其 OBBT/搜索流程或声称新原理。参考：[Tsay 等 NeurIPS 2021 原文](https://papers.nips.cc/paper/2021/file/17f98ddf040204eda0af36a108cbdea4-Paper.pdf)。主代理还读到 WraAct 2025 的引言，未完成全篇技术对照，不作其算法已全面审查的声明。

本轮没有候选代码、import、AST、编译、collection、数学脚本、pytest、LP/MILP、模型前向、GPU、shadow 或正式回放。最后已执行组件仍为 D158 的 4032 tests/212 files；D172 仅此前一次只读系数诊断，没有重试。本轮无新后台实验。任何后续候选执行必须先独立预注册、冻结，并保留原完整人口、数值、资源与逐级回放门。

正式成绩仍为 1870/2413=1063 CERT+807 validated ADV；独立 E0 CIFAR100 25、TinyImageNet 36，共 61/400。两套账均新增 0，不能相加，也没有本轮全量零回归实测。所有旧 CERT/ADV 保留要求、零无效 ADV、fail-closed、原二元身份与 decoder、no attack/BaB/split/backward/dual rescue 均不变。

分支 redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；tracked binary diff SHA256 在本轮复核仍为 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。配置 paper-only、primary-source-review、default-off、no-candidate-execution。只新增本目录及 RESUME_RESEARCH_20261004_D175.md；生产 dirty changes、历史模型/日志/结果和此前冻结源码均未改，无 commit/push/default 改变。

write-page 归档技能用于区分原生语义、已知关系、项目内推导、比较范围、资格与正式成绩；采用仓库本地新文档，不发布外部 Page。保存后读回并校验新旧哈希，无渲染预览。ANCHOR_SOURCE.sha256 保存只读依据，ARCHIVE.sha256 保存本轮文档。

下一步：以“同一源见证的可组合相容性”而非变量数量为切口，证明一个真正的 Neural-HZ 域元素及前向组合规律。当前静态三门投影可以作为强正控和可复用支撑；在差异性与完整代价未明确前，不因此扩张玩具测试或启动全量实验。
