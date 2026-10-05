# Neural-HZ 三相位物理关系研究恢复入口

完整 Goal 保持 active：定义优先、非凸原 phase/source/decoder、GPU、CIFAR/Tiny、13 家族、smooth/Transformer 及同路径全量目标不变。正式 1870/2413 和独立 E0 61/400 均新增 0。

本轮入口：[物理投影与正控](definition_first_20260928/d175_joint_phase_physical_projection_20261004/THEORY.md)、[完整两门 hull 强对照](definition_first_20260928/d175_joint_phase_physical_projection_20261004/PAIR_HULL_STRICTNESS.md)、[相位消费的商闭包边界](definition_first_20260928/d175_joint_phase_physical_projection_20261004/CLOSURE_BOUNDARY.md)、[研究记录](definition_first_20260928/d175_joint_phase_physical_projection_20261004/RESEARCH_RECORD.md)。旧 D174 等所有档案保持原状态。

有用的新项目内结果：固定 K=[[1,1,1],[1,1,-1],[1,-1,0]]，对任意三个共同源预激活 g，认证 ell<=K^-1*g<=u，t=max宽度，b_star=K*ell。已知 I3322 直接生成一条实际 q/g/原 beta 行：sum(q)-3*(g1+g2)/4-g3/2 <= (b_star1+t)*beta1+b_star2*beta2+b_star3*beta3-2*ell1-ell2。无需创建九个 beta*source 产品，也不新增 bit；同源支持、全部源展开、可靠算术和三元组选择成本仍需付。

严格正控：z∈[0,1]^3，g=Kz+(-1,.01,.01)，beta=(.6,.6,.5)，zbar=(.4,.4,.5)，Z=1/40*[[15,14,16],[14,14,8],[12,4,10]]，q=(.525,.506,.205)。每门有两点内部真实源见证；所有对的 D035 共 delta 可取(.35,.30,.30)，且有一致全正三 bit 分布。旧点 F=sum(q)-2z1-z2=.036；新行证明整个盒 F<=.02。后继 ReLU(F-.025)新为0，旧父域允许.011。此差异在实际幅值上，重选Z不能恢复。

比较升级：同轮补出了各对包含联合 guards 的完整两门 source-labelled hull 的四格内部源见证，三对共用同一 z/各门 beta/Z/q 及相容 delta。故这条物理行严格强于所有两门理想 hull 的交，不只是自由盒 pair 因子。各 pair 自己的源分布不能拼成共同三门源分布；不能扩大为超过三门或整个上游具体网络的全局 hull。已知 Bell/RLT 换元本身不是新域；本轮没有新颖性、真实 CNN 匹配、GPU 或数值资格。不得改称 formal gain。

辅助边界：固定线性观察商 Q 能精确消费原 mask M，当且仅当 ker Q 对 M 不变。对全部 k 独立相位普遍闭合并保留 affine source，可需 (d+1)*2^k 函数维；只限这种固定精确线性商，不是所有非线性/有损/因子化 Neural-HZ 的下界。普通三源混权 parity 见证说明相同父低阶矩和新 phase 均值仍缺 gamma*source 观察，不是新精度阳性。

下一步仍应提出可组合共同见证的新域定义并核区别与完整费用，不能把三门 Bell helper 或原计算图包装成答案。当前正控可支撑定义研究，但未授权候选执行。之后任何 numerical/code 活动先新预注册冻结，再数学、真实同结构、shadow、逐家族与全2413/独立400门；所有旧人口、四并发不回退及默认关闭边界保持。

本轮 paper-only，无后台实验。最后组件 D1584032tests/212files；D172只读系数诊断不重试。分支 redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。只新增隔离档案，历史及生产文件未改，无 commit/push/default 变更。
