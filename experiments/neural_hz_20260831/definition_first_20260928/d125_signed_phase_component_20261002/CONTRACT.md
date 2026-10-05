# 原相位条件幅值外包的最小数学组件

本组件实现 D124 的有符号条件误差定理及共同乘子族，在既有精确有理 System 上真正产生删除父连续幅值后的目标系统。它不是生产接入、完整 Neural-HZ 完成或真实网络收益。本轮只取得数学组件资格，不执行新的来源 census、GPU、shadow 或全量回放；后续这些阶段不被缩减或取消。

## 抽象变换与身份

使用既有 D112 Form/System 和 D119 Gate 类型。输入是一个完整数学系统、同一前沿上的父门、完整目标门和全部当前可见 readouts。从 target.g 自动抽取全部父幅值权重及 rest；按固定 tau 分解 h，不接受调用者伪造的 rho 或遗漏 fan-in 的 rest。源系数、tau 和有限 kappa 是结构参数，不由标签、terminal margin 或 solver 状态决定。

父门和目标门必须具有原四条 literal 约束及有效独立原 bit/output 身份。相对于这些原关系、变量界定义的输入 System 证明 sound；不能仅凭 Gate 描述就认证真实网络或原界来源。特别是有效关系给出的 tight 目标界可以比裸箱界紧，不额外要求所有 gate bounds 都覆盖忽略谓词的 box。

只移除已认证父门及目标门的旧四行，生成父符号守卫和目标新四行。所有未替换 EQ/LE 和保留列的 bounds 均保持并重映射。删除父 q 的变量 bounds 若产生额外限制，也须转到源上：q_lo>0 要求 f>=q_lo，q_hi 比门上界更小时要求 f<=q_hi。不得在零误差情况下遗失这些条件后误报精确。

父 f 和 target rest 必须不依赖待删 q、target.q 或参与本块的相位，从而保持无环同源前沿。未处理 EQ/LE 或任一可见 readout 仍引用父 q 时拒绝整个变换，不部分提交。readouts 是调用者声明的整个数学可见前沿，不是对真实网络外部消费者完整性的证明；native 接入仍须绑定所有实际端口。

所有原 bit 保留为 signed binary。新位置到旧 column ID 的 retained_ids 明确记录共享语义身份，不能以位置重排冒充删除 bit。对返回系统作 Affine/Add/Concat 时继续引用相同列；解码时按照原 source 恢复被删除的 q=ReLU(f)。诊断 decoder 只接受返回系统的整数可行点，并再检查完整原 System；原系统不满足时 fail closed。通过此数学检查仍不是 validated ADV，后者必须验证原具体网络。

## 证书与前向读出

对 t_i=(1-2*tau_i)f_i，按同一源 box 算 rho_i=max(0,upper(h+kappa_i*t_i))。T_plus<1 时使用 D124 的 E_plus、E_minus 和 R_minus 生成四行，保留原 beta、所有零点合法标签及目标原变量界。非零 rho 是有损外包；零 rho 才有相应精确带标签商语义。

零 rho 的目标关系等价不自动等于整个输入 System 投影等价：旧目标门的上下界可能额外限制源。receipt 分开记录 rho_zero、exact_target_relation_when_rho_zero 和 exact_when_rho_zero；最后一项还要求可靠源箱和父 ReLU 区间证明重构的完整 g 落在原目标四行使用的扩展界内。未证时只授予 sound outer，decoder 仍检查全部原限制；不把目标关系定理扩大成整个系统的精确性。

从目标两条上行还可前向导出一个源 affine upper。令 U>=r、L_h<=h、C=-E_plus-L_h：U=0 时 upper=0；U>0 且 C>0 时 upper=U/(U+C)*(h-L_h)；C<=0 时 upper=h+E_plus。同时保留 r<=U。下界 h-E_minus 与 r>=0 同时有效。普通正控可以用这个源 affine upper 和残差读出直接证明下一 ReLU 关闭，无新 LP。

共同 kappa 子族按支持函数的系数断点和零交点求 E_plus 下确界，区分有限达到、A 右边界及无穷远；常值段保留合法有限 witness。无穷远不是有限证书。该优化不保证 E_minus、最终输出界或总成本最优。模块只执行显式有理运算和参数扫描，不运行求解器、不枚举网络相位或输入区域。

## 成本与资格边界

精确有理数沿用 512-bit 门，返回条目数上限 64M；失败不修改原 immutable System。条目门不等于完整 transient 内存、GPU 或 whole-work 资格。

Projection 为了诊断 decoder 保留 original System。必须在 receipt 中明确保留旧系统并记录它与新发出系统的条目费用；不能只数新矩阵就声称净内存节省。测试所见 old/new 对照、映射、证书、decoder 和其他引用同样不是免费物理存储。后续实际接入需避免永久保存旧 HZ 矩阵，或支付其全部费用。

本次不新增 LP/MILP 查询。继承的固定组件 LP 控制原样保留。新控制直接代入旧 LP 的有理可行点及新整数伪点；数学小例的有限测试点/零点标签覆盖不是验证器中的 phase/input split。数学证明、机器测试、真实来源、native 绑定、GPU、完整物理成本和正式成绩分别报告。

## 原证据和完整目标

证明与公平参考是 ../d124_source_phase_fiber_20261002/SIGNED_ENVELOPE.md；旧 HZ 加相同证书必须列为强参考，不能宣称新外包在同证据下更精确。继承最后完成的 D120 全人口 3869 tests/194 files，不重跑其来源 worker或修改任何旧文件。

本候选默认关闭，必须显式 opt-in；正式 1870/2413 与独立 CIFAR10025/Tiny36=61/400 均不改变。完整 13 家族、CIFAR/Tiny、其他家族、smooth/Transformer、全面 GPU 及单路径满分目标保持 active。此前 D124 的 Attention 方向不在本次数值人口中，不因本组件而取消。

2026-10-02 Australia/Sydney；redu-hz；HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。所有文件只写新隔离目录，历史模型、结果、源码与冻结工件只读，无 commit/push。
