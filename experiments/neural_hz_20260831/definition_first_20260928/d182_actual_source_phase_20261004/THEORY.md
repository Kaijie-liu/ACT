# 非参考相位必须绑定当前源

本轮直接检验 Neural-HZ 的具体化定义。结果改变了下一研究动作：不能把同一个代理见证、可靠模型系数或更紧的能量预算当成已经保住当前源与门幅值的关系。一个普通五门例表明，即使同时加入原源盒可实现性、正确相位、分支范围、当前点精确能量和任意有限二次距离惩罚，双观测代理仍会产生错误的消费者包。

这是定义候选的联合反例与查询边界，不是新抽象域已经成立，也不是真实网络反例或新增 ADV。全部计算为纸面有理推导并经独立复核；没有导入或执行候选、模型、GPU 或求解器。

## 保持不变的语义

父元素保留原连续源、全部原整数二元相位、EQ/LE、共享 latent/frame、所有旧 bank 和输入 decoder。以下 beta 只是原 signed bit 的 0/1 视图，原零门两个合法标签不变。仿射、Conv、Add、Concat 必须按同一赋值读出，不能因消费者不同而复制源。

沿用 [D178 的定义](../d178_reference_observed_fiber_20261004/THEORY.md)：g=b+d，C 覆盖全部实际消费者且含 mass，tau_i=1[b_i>0]，H=C D_tau，K=[C;H]。原 guards 约束实际 g。共同代理满足

```text
Kt=Kd,  ||t||²<=E,
delta=C(D_beta-D_tau)t,
packet=C D_beta b+Hd+delta.
```

真实扩展取 t=d，因此健全；假包不说明健全性失败，而说明精度损失。新旧包之差为 C D_beta(t-d)。原精确 HZ 已有真实门图，目标只能是同信息、同完整预算下更强的可认证能力，不能要求外包比精确图更精确。

## 本轮排除的加强假设

在下面 d=A*x 的仿射盒父域上，考察一条比原实现更强的假想定义，不将其当成已经实现的候选：除双观测与原 caps 外，要求

```text
t=A*eta, eta belongs to the same original input box,
b+t has phase beta under the original weak-sign convention,
b+t lies in the original L/U,
||t||²+lambda*||t-d||² <= ||d||²,   finite lambda>=0.
```

这里右侧已使用当前源点的真实能量，而非原全域 E；代理由同一个仿射 producer A 生成，不只是落在 range(A)。但 eta 仍是另一个存在源，不能与当前输入 x 偷换。原输入 decoder 始终返回 x。

此假设仍有真实扩展 t=d、eta=x，零门继续接受原来的两个标签；下文反例的所有相关符号则都是严格的。它是否高效可查询尚未解决；本轮先检验即使免费给予这些更强前提，源绑定是否已经恢复。

## 普通五门的联合反例

令 P 循环左移五个坐标，A=I+P/4，x 属于 [-1,1]^5，且

```text
b=(1/2,-1/2,1/2,-1/2,1/2), tau=(1,0,1,0,1),
C=[(1,-1,-1,1,1/2); (1,1,1,1,1)].
```

两个读出都是真实消费者：混权输出和 mass。所有五门均 crossing、法向非平行、偏置非零。原完整盒给 d_i in [-5/4,5/4]，故正偏置门的 L/U 为 [-3/4,7/4]，负偏置门为 [-7/4,3/4]。

选取严格内部的当前输入

```text
x=(-3,12,-48,192,-768)/1025,
d=A*x=(0,0,0,0,-3/4), beta=(1,0,1,0,0),
k=(3,0,1,0,-4).
```

直接代入有 Ck=Hk=0、||k||²=26、d^T k=3。对任意有限 lambda>=0，取

```text
epsilon=1/[100(1+lambda)], t=d-epsilon*k.
```

能量差严格为负：

```text
||t||²+lambda*||t-d||²-||d||²
 = -6 epsilon+26(1+lambda)epsilon²
 = -287/[5000(1+lambda)] < 0.
```

所以即便改用点值精确能量并加距离惩罚，这个代理仍合法。它还有同 producer 的真实源见证：

```text
eta=(-3-3120 epsilon, 12+180 epsilon, -48-720 epsilon,
     192-1220 epsilon, -768+4880 epsilon)/1025,
A*eta=t.
```

0<epsilon<=1/100 时，x 与 eta 都严格在原盒内部。代理预激活为

```text
b+t=(1/2-3 epsilon,-1/2,1/2-epsilon,-1/2,-1/4+4 epsilon).
```

它严格具有同一个 beta，并严格处于对应 L/U。故同见证 branch box、由该盒推出的 aggregate caps、真实非负幅值的 mass dominance 都通过。旧全域能量与双观测推导的有限行也通过，因为这里已经满足更强的原生关系。原 x 的 guards 同样严格成立。

然而消费者输出不同：

```text
C ReLU(b+d)=(0,1),
C D_beta(b+t)=(-2 epsilon,1-4 epsilon),
delta=(3/8-2 epsilon,3/4-4 epsilon).
```

普通 lambda=1 时，epsilon=1/200，假包为 (-1/100,49/50)，点值能量左侧为 2669/5000，小于 9/16，间隙为 287/10000。它不依赖极端数值、零激活标签、边界输入或仅有全局能量松弛。

这是固定源诊断，不是整盒安全性质。后继读出若为 ReLU(-packet_1)，该当前真实源给 0，而假包给 2 epsilon；这里只说明失真可被后继消费，不宣称全盒该门稳定，也不计 CERT。

## 为什么有限二次惩罚仍留自由度

记 W=ker(K)，Q 为 W 的正交投影，n=t-d。配平方给裸关系的精确等价式：

```text
n in W,
||n+Qd/(1+lambda)||²
 <= (E-||d||²)/(1+lambda)+||Qd||²/(1+lambda)².
```

当 E=||d||²，有限 lambda 只是把球半径缩至 ||Qd||/(1+lambda)。忽略附加 caps 时，对方向 v 的输出误差支持为

```text
[-v^T C D_beta Qd + ||Qd||*||Q D_beta C^T v||]/(1+lambda).
```

这些是已知球切片几何的应用，不是新的支持算法。加入 caps 后该支持仍为上界，未必可达；上面的具体反例已另行核对全部相关 caps，没有利用这个豁免跳过证明。

局部原因更直接：沿可行核方向，||d+n||²-||d||² 有一阶项 2d^T n，而有限二次罚只有二阶费用。若 n 方向使一阶项严格下降，充分小的非零扰动仍合法。不能把该论证扩大成“任何有限惩罚都不可能精确”；例如非光滑线性范数罚不满足这个二阶前提。也不能由一个伪点推断所有有损域都没有验证价值。

## 与旧研究的区别和新的研究决定

[D159](../d159_full_interface_definition_boundary_20261004/THEORY.md) 已有点值能量丢失方向与固定相位 LP 凸化的边界；[D160](../d160_source_subspace_fiber_20261004/THEORY.md) 已研究源子空间；[D170](../d170_capped_joint_fiber_20261004/CAPPED_COMMON_FIBER.md) 已要求盒与能量共用见证；[D174](../d174_source_coupled_energy_20261004/THEORY.md) 已研究全域预算下的距离项。本轮新增证据仅是：在 D180 的双观测、非参考相位上，上述加强条件连同任意有限二次 tether 可同时成立，仍不能绑定当前源的输出。

全相位观测闭包和相位签名秩条件已由 [D158](../d158_joint_forward_support_20261004/THEORY.md) 与 [D179](../d179_preterminal_domain_20261004/DEFINITION_TEST.md) 记录，不再当作新发现。Box、更多固定参考或能量尺度的修正，只有证明实际查询与完整成本有用，才值得实现。

因此本轮不实现新的 tether、capped-fiber 或 loader 变体。转向[条件实际源加载关系](SOURCE_LOADING.md)：它明确指出需要保留什么，但完整逐列展开太贵，仍只是强对照而非选定的新域。下一定义应研究怎样以可付的共同结构承载这些相位—当前源作用，并证明它经过混权、live skip 和下一 ReLU 后仍可查询。若只能恢复完整旧图或得到比旧图更贵的行，不宣称创新完成。
