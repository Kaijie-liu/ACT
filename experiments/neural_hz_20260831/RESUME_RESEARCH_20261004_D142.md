# Neural-HZ 联合状态替换的研究交接

Goal active，未完成。用户主线是强大的定义优先非凸 Neural-HZ，而非 helper。
上一轮 D141 的局部关系归档属于 progress；本轮得到
[联合状态替换合同与成本边界](definition_first_20260928/d142_joint_reabstraction_20261004/DEFINITION_AND_LIMITS.md)。

核心建设性结果：对当前整个状态 v=C(beta)z，若有证覆盖 E(beta)z，
就可用 Ctilde(beta)z 加同一份共同残差替换旧输出方程。全部 live 输出、
skip、原 guards、EQ/LE 同时使用该实现，decoder 精确保留。新 ReLU 在
当前扩大状态上精确构造，再作有证替换，有限次组合健全。不能只对原真实
轨迹证明误差，也不能让各消费者各自选误差。该包含定理本身不证明新颖性。

一个明确残差族是 v=Ctilde*z+L*eta、norm(eta)<=eps*norm(R*z)，对应
同一有界矩阵误差作用于同一源。它可以真正改变具体化且保留非凸性；但
固定源/相位下的球支持并未解决带原 guards 的完整查询，普通线性 terminal
也不自动支持这种非线性耦合。重复压缩会累计连续误差和谓词，未证固定宽度。

TT/separation rank 压缩不同于 D121 的 phase-degree 截断，但 TT 本身已知。
精确改存 TT 只是表示优化。两个具体费用结果：所有 live q/guards 放在
末端 readout 时可使 bond rank 增长；原逐 gate 的开关矩阵本来 rank 1，
固定 phase 顺序的常数双边投影仍非零，且 bit 未固定、连续投影非常数、
无其他已证消项时，标准 lowering 仍引入一次 bit×连续投影乘积。
不能凭更小 bond rank 宣称正式能力或终端节省。

原 gate mask 只保护 off-state，不自动保护 active value；需要声明精确
连接或明确近似合同。同时保留所有原 affine/gate 方程会恢复原图，不能
又称严格有损新域。D141 的 k=min(...,q11) 也有内部假点，不能替代第四门。

下一研究应解决共同 source/phase/guard 的可付统一查询或再抽象，并在
普通 mixed Conv/ReLU/residual 上给出完整收益依据。不启动只优化凸 fiber
的 helper，不搭未解决 terminal 的 TT 框架，不用合成低秩模型替代真实前沿。
smooth/Transformer、GPU、其他家族目标均仍在，尚无本轮资格。

正式 1870/2413、独立 CIFAR100 25 + TinyImageNet 36 = 61/400 不变；
13 家族逐解保全、原 bits、共享源、fail closed、禁止 attack/BaB/split/
backward/dual rescue、历史只读及完整回放边界不变。只有纸面与只读检查，
无数值候选、测试、模型或后台执行；最近执行人口仍 D136 3965/208。

分支 redu-hz；HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；
tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。
仅新增隔离记录，旧源和旧结果未改，无 commit/push。
