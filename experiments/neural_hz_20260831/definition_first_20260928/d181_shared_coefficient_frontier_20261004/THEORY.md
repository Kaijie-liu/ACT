# 真实系数共享的 Neural HZ 绑定合同

本轮把D180的固定有理系数原型推进到真实网络接入所需的数学接口：同一BN系数必须在源观测、相位掩码、输出和所有后继中保持同一身份。以下是纸面定义扩展及推导，不是已实现或已通过测试的新组件。D180的4056项资格只属于原冻结实现，不能转授。

这一扩展旨在避免真实模型绑定时丢失刚建立的共同关系，不将区间系数、参数化域或外向舍入本身宣称为新颖性。强Neural-HZ仍须以实际查询能消费的关系和同预算新证明体现价值。

## 真实链与系数身份

对于D179的完整链，以列向量记原父skip为a，新ReLU输出为q。由已认证的Conv、BN、Flatten和Gemm属性，后继前激活为

```text
h = B q + S a + c
B = W F D_s K
S = W F
c = W F t + b_fc
s_channel = gamma_channel / sqrt(var_channel + epsilon)
t_channel = bias_channel - s_channel * mean_channel
```

这里K为无bias的conv2，F为NCHW单样本Flatten。BN按单输出test模式解释，同一个通道scale在全部空间位置共享；Gemm使用已记录的transB=1和alpha=beta=1。[ONNX BN9模式与参数](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html#batchnormalization-9)、[官方test模式公式](https://raw.githubusercontent.com/onnx/onnx/main/onnx/reference/ops/op_batch_normalization.py)、[Gemm11约定](https://onnx.ai/onnx/operators/onnx__Gemm.html#gemm-11)支持这些算子语义；它们不是本项目浮点执行已获认证的证据。

原FLOAT常量可精确解释为有理数，但正平方根及除法生成的scale未必有理。一个可靠区间只包住真实scale，不将真实B变成区间中点矩阵。实际后端舍入误差还须另行认证；实数公式不能自动认证ORT、Torch或CUDA的具体执行。

令theta包含每个这样的固定模型系数身份，Theta是已认证含真实theta_star的有界集合。t_channel必须由同一s_channel导出，不创建独立shift副本。同一C(theta)、H(theta)、g_theta和skip读出贯穿整个状态；前后层也不能重新选择已经存在的theta。

## 共享参数具体化与健全性

取拥有原source、原二元相位、EQ/LE、旧残差和decoder的父族P_theta。定义新元素的具体化为

```text
exists one theta in Theta, one parent member in P_theta,
and one witness t_k for each bank k,
such that ALL predicates, observations and readouts use that same theta.
```

这等于按共同theta对各确定系数域的具体化取并集。不是让每个消费者或每条行各取一个参数，也不是将原二元因子放松成连续量。原输入decoder不把模型系数当成可控制的输入。没有新bank且没有参数不确定性时，原HZ嵌入不变。

对一个新bank，按统一规则预先选固定有理参考b0和固定tau_i=1[b0_i>0]；不能依据terminal状态或某个有利theta重选。令d=g_theta-b0，C(theta)覆盖所有消费者并含mass，H(theta)=C(theta)D_tau。使用整个联合父域上可靠的E，要求

```text
C(theta)t = C(theta)d
H(theta)t = H(theta)d
||t||^2 <= E
delta = C(theta)(D_beta-D_tau)t
packet = C(theta)D_beta b0 + H(theta)d + delta
```

原guards仍约束实际g_theta；分支caps等加强项必须同时对整个Theta及父域有效。对任意真实theta_star和父成员，取t=d，即得packet=C(theta_star)ReLU(g_theta_star)。由每步同一个theta的归纳，真实模型的前向像始终包含在新域中。这是健全的参数绑定定理，不证明该域可高效查询。

固定theta时，beta=tau仍强制delta=0，包精确等于C(theta)D_tau g_theta。对Theta取并后，它仅对每个参数切片精确，不等于对单一真实网络全域精确；参数不确定性与原代理的核失真分别存在。单门、确定C=1时保留原非凸ReLU图，故这一扩展并非整体替换为凸域。

固定theta时的Gram数学等价式仍成立，但未知区间theta还带来全局参数可行性问题。不能把C换成中心矩阵，再调用D180有理contains并把结果称为这个并集的精确成员判定；此扩展的native membership、数值算术及查询均未取得实现资格。将共享theta复制成独立变量通常会进一步放松集合，未必立即不健全，但会丢失本合同要求的身份与精度，不能混称为同一域。

## 必须补上的参考常数漂移

固定b0后，即使源在reference、旧相位取reference且旧delta为零，也可能有

```text
k(theta) = g_theta(reference) - b0 != 0.
```

所以新d的能量分解必须包括这个常数漂移、源变化、旧相位列及旧bank残差，再在整个Theta乘以原生父域上认证。沿用D180“参考常数已经精确扣掉”的零常数预算会遗漏合法变化。若采用固定等权Cauchy，k是一个正常计费项；不能只加系数半径，却不把它的值及交叉分解费用算入。

参考误差delta归零仍不意味着E归零。没有可靠的全域E、L/U或系数区间时，接入失败关闭，不用样本能量替代，也不改用独立hidden box。

## 有限行的可靠降级

假设已在全域证明某行a(theta)z<=b(theta)，并有固定有理中心a0、b0以及

```text
|a_i(theta)-a0_i| <= r_i
|b(theta)-b0| <= s
|z_i| <= M_i.
```

则同一域的一个可靠有限外包行是

```text
a0 z <= b0 + s + sum_i r_i M_i.
```

证明为加减a(theta)z，再逐项取绝对值。这不要求参数独立，但M_i必须覆盖整个合法原生状态，不可从正在证明的同一行循环取得。等式需两侧分别外包。它只生成外包：有限可行仍不代表存在共同theta、共同t或真实输入。

不能将该通用降级不加区别地用于参考相位range。它会把本来关联的常数与beta项分开，给beta=tau留下伪残差。应先保留结构式

```text
|delta_j| <= sum_i kappa_bar_ji * |beta_i-tau_i|,
kappa_bar_ji >= sup_theta R(theta)*|C_ji(theta)|.
```

原整数flip仍为beta_i或1-beta_i，因此有限行保留正确负系数，且beta=tau时右侧严格为零。相同符号表达式先按身份合并/抵消，再做区间估计；不得以独立副本替代共享源和系数。

平方证书对任意预先固定(l,r)有效。新系数合同应统一冻结有理方向尺度，例如固定s=1，或事前按可靠范围决定一次；不能未经证明使用可能穿过零分母的s(theta)=sqrt(Emax/||c(theta)||^2)。这是未来新版本的算子选择，不回写D180，也不将不同尺度的精度或资格混记。

## 面向真实后继的精度审计量

在相同theta、相同父赋值和原beta上，候选与真实后继之差为

```text
e = h_hat - h
  = B(theta)D_beta(t-d)
  = B(theta)(D_beta-D_tau)(t-d).
```

共同skip及偏置在差中精确抵消。若||t||和||d||均不超过sqrt(Emax)，则逐行Cauchy给出

```text
|e_j|^2 <= 4 Emax sum_i B_ji(theta)^2 |beta_i-tau_i|.
```

用同一L/U确定可能翻转集I：tau=0且U>=0，或tau=1且L<=0，都必须计入，以保留零点双标签。令Bbar_ji可靠覆盖全部theta的|B_ji|，则2 sqrt(Emax sum_{i in I} Bbar_ji^2)是统一全父域误差上界。这里只审计误差，不固定/删除bits、不搜索相位，不将旧HZ的证明转交候选后伪记为独立能力。上界大仅表示不足以证明有用，不自动成为失真反例。

## 不展开B的一个预算规则及其代价

令B=A K，其中A=WFD_s，D_flip=D_beta-D_tau且f_i=|beta_i-tau_i|。对任意读出矩阵M，Frobenius乘积界给

```text
||M B D_flip||_F^2
 <= ||M A||_F^2 * sum_i kappa_i f_i,
kappa_i = sum_o K_oi^2.
```

因此该界可作为原残差能量规则的健全上界：系数kappa只需一遍完整卷积stencil归约，而非对每个消费者展开B。Theta不确定时还须统一上界||MA(theta)||，原phase符号仍保留。mass行及它与其他读出的组合不在B因子中，必须单独保留并支付共同组合界；不能从计数中漏掉。

这是可实现预算公式，但本轮不选它作为已经足够强的替代：例如A=(1,1)，K的两行为(1,1)和(1,-1)，仅第二位flip时真实B=(2,0)给能量系数0，而该乘积界给4。普通混合权重的抵消会丢失，实际能力可能下降。即使这个预算较便宜，guards、Cd/Hd、有限行和终端仍须有完整可用接口；不能由一个廉价norm数值宣布全路径可扩展。

## 能力接受条件

同一原生父域下，原精确HZ后继集包含于候选外包。因此不能把“集合精度超过精确图”设为目标。真正接受证据是同一可靠信息、普通终端和完整预算下，新表示实际完成旧路径未完成的全域证明或更有用的认证界，且这种收益跨下一ReLU到达原性质查询。只排除一个Gram伪点、只在参考切片精确、只减少变量、或只绑定B均不够。

原HZ对照也必须获得相同的BN区间、E、L/U及有效后果，并计算相应成本。稳定门按双方同一可靠界处理，不再拿未简化4m当实际基线。此合同尚无数值资格；下一实现必须默认关闭，在新的预注册下保留全部4056/213数学人口，再开展完整真实前沿对照。
