# 非凸 Neural HZ 相位传播研究恢复记录

Goal 保持 active。用户明确要求强大的定义优先 Neural-HZ；不要把 helper、存储或凸子系统查询优化当成主线完成。正式 1870/2413 和独立 CIFAR25/Tiny36 共 61/400 均零新增，13 家族逐例保旧和全部验证门不变。

本回合的[数学记录](definition_first_20260928/d139_phase_effective_closure_20261004/THEORY.md)与[研究决定](definition_first_20260928/d139_phase_effective_closure_20261004/RESEARCH_RECORD.md)优先于 D138 的“接着实现小反馈块”下一行动；旧档保持只读，不改写。

新结果一：q=R(x+y/10)、r=R(x-q/2-1/4)、J=q-x/4-2r 在 [-1,1]^2 的真实上界为 11/20；完整父层 labelled hull 却有 (x,y,q,beta)=(1/2,0,3/4,3/4)，下一精确门 r=0 给 J=5/8。接 s=R(J-3/5)，真实 s=0，假点 s=1/40。所有线性支持也不能替代非凸关系跨 ReLU 的保留。假 beta 是分数，因此这不否定完整整数 HZ，不是生产运行或新 ADV。

新结果二：D126 相位条件 caps 与 D138 支持结合，自己的 beta^2 可消，但 beta1*beta2 仍在。含活 skip 的正确方向必须先合并为 d+gamma*w，需 k(d+gamma*w)^T c(beta)，不能独立界各支路再称联合精确。共同 e 也仍可能与真实原源不一致；完整父图 P 不可被忽略后又当作已消费。

新结果三：每个原相位若都具有源内非空开纤维，则定义代入后的 Boolean-affine 恒零式只能逐相位系数为零。普通两父门、混权后继和活 skip 的全八相位控制已给出；不再把共享源自动当成新等式商的证据。此边界不排除守卫不等式或其他可认证结构。

另存 SUPPORT_LEMMA.md：常数-cap convex feedback fiber 上，一个后继 hinge 加活 skip 的任意 signed query 可直接求；负 hinge 经经典凸 minimax 变为一维参数支持，小2SCC的全共享k曲线最多4n内部断点，朴素O(n*(n+nnzM))算术、O(n²)存储。不是未知相位全集合的费用，也不是非凸前缀的递归闭包；负hinge等强于同F原生单门LP。lambda是数学对偶参数，无运行或新solver路径。此项为支撑原语，未选择实现，不要让它再次取代主线。

下一研究：明确如何统一消费相位依赖的共同方向及源守卫，并在原非线性后保持有用关系；与已有 D027/D091 和同证据 HZ 比较完整成本。不可只加另一条位吸收规则、条件表或矩阵优化器。D137 的完整 Conv/Add 前沿和所有存活消费者继续必须覆盖，不把小控制或五窗口当成真实网络资格。

没有候选源码、运行/冻结、测试、模型、GPU、solver、后台 job、生产修改或 commit/push。D136 的 3965/208 仍是最后执行人口。此前真实资格、浮点/GPU、完整物理成本、shadow、逐家族及全 2413/400 均未完成。下一数值候选仍须新预注册/冻结、保全人口和原预算，不运行消费过的旧版本。

归档日 2026-10-04 Australia/Sydney；redu-hz；HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。只增新的隔离文档，原九个 tracked 修改与历史成果保留。
