# 混合相位关系的原生终端运输

本候选把已证明的 D244/D245 同源混合相位关系运输到实际 `SparseHZono`
矩阵，而不重建 D243 的 owned Source，不调用 D064 的旧 bound generator，
不调用 Bank、支持搜索、攻击、相位搜索或 solver rescue。这是已有数学关系的
接入工作，不是又一个新抽象域，也不宣称新定义或新颖性已经完成。

## 原生语义

对原 H 的连续因子 xi、原 signed binary 因子 z，保持其所有 EQ、LE、输出和
具体输入读出。门描述符必须逐项匹配实际原谓词。extended 门满足
`f=L(s+z)+Q(1-eta)`，L<0、Q>0，守卫 `s+z>=0, eta-z>=0`。
整数 z=-1 时 f=q=Q(1-eta)，z=1 时 q=0、f<=0，因此 q=ReLU(f)，
f 属于 [2L,2Q]。零点两种原标签都保留。compact 门仅在 D064 提取误差严格
等于零时接受；不得把真实舍入差异设成零。所有四个原门位须不同。

父门取 S=max(-2L,2Q)、x=f/S、q=Q(1-eta)/S、alpha=(1-z)/2。
不把 q/(2Q) 误当 q/S。父源不得含所选门的内部槽，child 源不得含本批
两个 child 的内部槽；这一保守结构门不读取实例身份或 LP 状态。
两 child 的完整均值 zbar 和差 d 均从实际谓词提取。用 selected eta 的
实际系数消去归一化 q 部分，所有偏置、其他连续/二元来源仍在余项内。

在完整 native latent 坐标上固定使用半径一的二乘二 Gram，提取 a1,a2；
秩不足直接拒绝。它与 D245 定理相容，但不假称各种 source 参数化下得到
相同系数。tau 固定为 child difference 的两个 q 系数差的一半，须正。
完整 r/e 按 native cube 界估计，cube 仅支撑证明而不替代原 H。固定 carrier
若未认证其 EQ，不偷偷消去；由此可能拒绝本可成立的关系，记录为覆盖缺口。
随后执行 D245 同一 mixed mismatch guard，没有 X/Q 或其他失败回退路径。

## 保留整数具体化的扩展

每个关系使用六个连续产品 c12,c21,d12,d21,v1,v2，原 phase 不变。
每个 w=alpha*v 的可靠范围为 [min(0,Lv),max(0,Uv)]，按 midpoint+radius*zeta
嵌入一个 native [-1,1] 因子。零半径时 w 恒定、该预约槽自由，不增加相位。
四条 MC/产品、两条 defect、三条 X capacity，共 29 条精确 LE。

将常数移到 RHS 后，复用 D086 的逐系数 float64 舍入补偿：误差绝对值总和
加入 RHS，再向上舍入。旧 EQ/LE/输出只复制补零，不重新舍入。每个原整数
状态有 canonical 产品延拓，且新增谓词不能添加旧 H 之外的旧坐标，因此
忘却六个辅助坐标后整数具体化恰为原 H。舍入后的任意辅助解不保证仍等于
精确产品，不能将其当成真实条件矩继续传播。输入 decoder 只读取原坐标。

## 本阶段的生命周期边界

`capture` 建立深复制、只读的 detached terminal snapshot，记录整个 global
高水位并补零。它不是 live TF cache，不向原 allocator 或生产 cache 发布，
不能与未来出生的分支重新合并。所有关系从一个固定完整 population 同时
构造、同时返回；一项不合格则整批不返回，没有半批收益或按失败改选人口。
结果只通过原 `_lower_hz_milp` 的正常终端入口消费；不进行额外求解救援。
完整原 binary 保留，lowering 禁用删除/固定相位和连续投影。
旧 lowering 的 signed 到 0/1 平移并非自动精确，即使 .1/.2 这样普通系数，
RHS 的 float64 求和也可能改变旧 EQ。新组件必须逐项精确核验唯一正常
lowering 返回的系数、EQ/LE RHS、输出及变量域；任何不精确则 fail closed，
不调用第二个求解路径，也不修改原 solver。这一拒绝覆盖缺口不能隐瞒。

这不是把总目标缩减为 terminal-only。在线整批接入仍需独立 role namespace、
全 frame lease/commit、所有预计算消费者更新和可靠 rebase/release 代际。
生产 Add/Concat 只识别整数 frame_id，generation++ 本身不能防 slot alias；
旧 skip 仍存活时不得回收旧列。当前快速前向 bounds 不消费谓词，因此本
阶段也不声称增强下一层 bounds。这些未完成项必须保留到后续真实模型阶段。

## 物理正控和完整费用

固定 A∈[0,1]^4,t∈[-1,1]，x=(A1-A2,A3-A4)，q=ReLU(x)，
zbar=(13/16)(x1+x2)+(1/16)(q1+q2)+(3/32)t，
y=ReLU(zbar±(q1-q2))，F=2q1-x1+2q2-x2+(3/8)(y1-y2-(x1-x2)/2)。
真实上界 527/256，取等于 x=(1,-1),t=1；mixed guard 为 [-29/32,31/32]。
u=255/256、x=t=0、alpha=(1/2,1/2)、q=(u/2,u/2)、
y=(129u/160,97u/160) 的完整强参照点给 F=4233/2048，超过真界17/2048。
新测试必须从实际存储行而不是仅纸面定理证明排除该点，并确认原生正常
终端保留新增 LE。证明点与分布是固定数学控制，不是输入搜索或 ADV。

每关系增加六连续槽和29 LE，原 native box 已编码产品范围，无需重复12界行。
nnz 必须按实际 native 展开报告，不沿用 D245 的96。完整原六CSR扫描、复制、
补零、stack、所有 Fraction 运算、Gram、余项、产品、编译、终端降级及 decoder
共存另计。沿用共享生命周期 Budget：work256M、单公共操作200M、entries64M、
512位；不能按 pair 重置。逻辑计量及保留数组字节不等于完整物理峰值资格。

全部候选显式 opt-in。即使数学测试通过，三个完整实际模型、在线前向、GPU、
完整物理、smooth/Transformer、新家族、shadow、13家族、2413/400 和四并发
仍未通过；native_HZ_admitted、新域和新能力资格均为 false，新增 solved 为零。
正式1870/2413及独立 CIFAR10025+TinyImageNet36=61/400不变。
