# Definition-first Goal activation — 2026-09-28

Branch: `redu-hz`; commit: `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.

The user explicitly deleted the previous goal and requested a new Goal-mode
objective. `get_goal` returned `goal: null`; `create_goal` then accepted the
definition-first objective below and returned status **active**.

`createdAt`: `1790566184`; thread: `019fa371-dbaf-70b1-99cf-d7b3290e8aa6`.
No token budget was requested or imposed.

This synchronizes the goal service with the user-authorized definition-first
research direction. It supersedes ONLY the historical unsynchronized/paused
service-status notes in the sealed September 28 amendment and D001 checkpoint;
those documents and their hashes remain unchanged. Their research constraints
and the supporting-work archive remain in force. No numerical run, production
change, benchmark promotion, commit or push is part of this activation.

## Exact service objective

在分支 redu-hz 上，从 Hybrid Zonotope（HZ）的数学定义出发，研发适合神经网络验证的新型非凸抽象域 Neural-HZ，而不是只优化现有 HZ 的存储、矩阵生成、构造成本或求解器实现。核心研究对象是域元素、具体化语义、生成元/相位/谓词关系，以及仿射/卷积、ReLU、残差/Add/Concat 等抽象变换；证明与原 HZ 的嵌入或精确对应、算子健全性及适用范围，形成能在常见网络结构上带来实质验证收益的定义和定理创新。以 PLDI 级研究质量为目标，但不能把换名、已有符号执行或未简化的计算图包装宣称为新抽象域。研究优先精确分段仿射结构，不围绕极端特例或测试框架扩张偏离主线。

能力目标：在唯一正式 baseline 1870/2413（1063 CERT + 807 经具体网络验证的 ADV）之上逐类获得可复现净提升，长期正式终点为一条候选路径上的 2413/2413；同时优先推动 CIFAR100 和 TinyImageNet 的大规模能力提升，并检验其他尚未纳入既有 VNNCOMP 实验的家族。数学定义、组件优化和局部实验只是里程碑，不能据此宣布整体目标完成。

硬限制：
① 保住每一个既有 CERT/validated ADV 及全部 13 家族逐家族 solved 数；新增 CERT 必须健全，新增 ADV 必须通过原始具体网络与性质验证，invalid SAT/ADV=0。UNKNOWN、TIMEOUT、ERROR、中间层停止、断开的能力实验均不计正式收益。仅当同一候选源码清单、配置、预算和执行路径完成全部 2413 回放，保留全部1870且增加新解，才更新正式成绩或默认启用；不得拼接不同路径结果。
② 保留连续因子、显式二元非凸相位、等式/不等式谓词、共享 latent/frame 身份和具体输入重构；不得为了压缩 pivot/删除二元因子或将其连续放松，不得退化或整体替换为 Zonotope、Constrained Zonotope、区间或其他凸域。任何未证前提、数值/资源/见证失败必须 fail closed。
③ 算子和化简统一按可观察数学结构执行，不按实例/模型/家族身份、公开标签、历史结论、terminal margin 或 LP/MILP 状态选择菜单。禁止靠 attack/PGD、BaB、输入/相位 split、backward/dual rescue 或 LP 对偶/不可行证书修复冒充域创新；普通终端 LP/MILP 决策与独立具体见证验证的原边界不变。
④ /data1/Kane/HyZor 中历史模型、日志、表格和结果，以及此前冻结实验源码/证据均只读。所有新工作仅写入 experiments/neural_hz_20260831 下的新隔离文件/目录，记录分支、commit、配置、依赖及基线 provenance，绝不覆盖、混合或重写旧结果。此前表示/构造优化成果完整保留为可复用支撑库，失败版本和未认证草稿保持原状态，复用不免除新组合的证明。
⑤ 以定义创新为主线，按重复的普通神经网络结构逐类开展预注册研究：先提出数学定义、与现有 HZ/ImageStar/混合符号或多项式域的区别、可反证的结构定理及完整端到端代价，再实现默认关闭的候选。明确算子哪些精确、哪些尚未覆盖；不把前端变量少、惰性引用或局部存储少等同于全局收益。完整计算传播、谓词/相位、终端降级与查询、共享存储、证据及见证重构成本。保留原有数值、资源、全量验证与 fail-closed 要求，不降低测试人口或事后放宽冻结门槛。旧“必须先攻死连续因子投影/完成C131构造”的研究顺序已被定义优先修订取代。
⑥ 每个候选默认关闭/显式 opt-in，依次经过数学/等价性测试、真实同结构目标、同结构 shadow、逐家族回放和全部2413回放。能力与纯速度晋级保持解耦：能力须保旧解并通过原四并发不回退门；1.5x单请求/2.0x四并发/1.8x bootstrap仅用于纯速度晋级。保持冻结的完整物理存储、资源和验证边界，不能以不计终端转换/守卫开销掩饰退化。
⑦ 独立外部 E0 为 CIFAR100 25、TinyImageNet 36，共61/400，经验证的历史来源ADV不算Neural-HZ新增收益，也不与1870相加。新候选必须保留全部61，新增解仅在其余339 UNKNOWN上独立计账，并经同一路径完整400回放、两家族零回退和零无效ADV；其他外部家族先冻结独立基线与见证协议。
⑧ 所有定义、证明、假设、正负实验和停止原因都要记录。新颖性、健全性、组件资格、实际速度和正式成绩分别报告。不将困难等同于阻塞，不用反复压构造预算替代定义研究，不因一个局部成功提前完成整体目标。

当前数学起点为 D001：相位依赖生成元及守卫关系驱动的精确商化简。它只是待检验候选，不是必须坚持的答案；优先证明其相对已有混合符号域和现有HZ简化的实质区别，检查保留全部二元相位与守卫后完整终端求解是否仍有净收益。若仅是重命名/转移成本，记录负结论并修改定义假设。项目目标权威文本为 experiments/neural_hz_20260831/GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md；此前成果索引为 supporting_work_archive_20260928/README.md。

