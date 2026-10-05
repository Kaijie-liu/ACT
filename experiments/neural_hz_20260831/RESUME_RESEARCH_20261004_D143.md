# Neural-HZ 整组缺口研究交接

Goal active，未完成。用户强调强大的非凸 Neural-HZ 本身，不是 helper。
上一回合仅确认方向，属于 no progress；本轮 D143 得到可改变下一动作的
[精确公式及双向能力边界](definition_first_20260928/d143_aggregate_gate_gap_20261004/THEORY.md)。

在 q>=0,q>=g 下，正权 sum a_i(q_i-beta_i*g_i)=0 等价整组原 ReLU，
保留全部原 bits 和零标签。g=A z+b 时可改写为 a^T q=theta^T z+(a*b)^T beta，
theta=A^T diag(a) beta。这是经典零互补 gap 的源侧参数化，不是已证新颖性。
theta 不是 binary；d 个 theta_j*z_j 的四行 McCormick 不会因 beta 整数而精确。

本轮重要比较：使用自由 bit 盒 theta 范围时，全部 source-labelled 单门
完整凸包的交包含于该 aggregate 外包，双方显式共有原谓词、幅值界、guards
与 off-mask；证明由 perspective 见证、负项补码及四行非负加总给出。
若用额外共同 guards 收紧 theta 界，需另证，不能
扩张该定理。强参照的实际生成成本也不免费。

同一三门两源非平行、有偏置、全 crossing bank 给双向物理 LP 分离：新形式
可改善弱四行查询，但也会失去旧父层已能证明的 mixed+live-skip 预激活界。
旧 bound=-1/40，新外包允许 +1/20，且新点的原 beta 全整数、源在内部、g 非零。
这是纸面回退控制，不是正式 benchmark 已测回归或 ADV。下一 ReLU 恒零结论
依赖精确变换或正常前向稳定识别，不适用于同用宽界的 fractional 子门 LP。

当前真实 Conv frontier 的 d/m 是 65536/65536、14400/8192、46656/25088，
不满足假想 d<m。保存 theta/p 的版本仍保留全部 q，并新增 2d 连续量、
5d+1 行及矩阵系数；去掉 theta 只是把其系数重复展开。点积 GPU 计算不是
完整集合查询加速。恢复旧门行加上该式则又变成辅助约束，不推进实现。

研究决定：关闭这个简单聚合包络的实现入口，保留公式与正反控制供以后
对照；不要重复启动正权 gap、经典能量或独立 product 包络为新域。
主线回到能处理共同 source/phase/amplitude 与全部 live consumers 的
跨层域规则或可付联合再抽象，不能只压符号或代数表达。

正式 1870/2413、外部 CIFAR100 25 + TinyImageNet 36 =61/400 均不变。
所有旧解保全、fail closed、完整回放、历史只读与禁止 rescue 边界不变。
没有候选代码、数值实验、模型/GPU运行或后台任务；最新执行人口仍 D136。

新工作只有 D143 隔离文档。分支 redu-hz；HEAD
f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；tracked diff SHA256
29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。
没有改默认路径、旧源码或旧存档，没有 commit/push。
