# 共同连续源与原相位的混合关系包络

本候选把 D046 的纯原相位仿射包络扩为原连续源与原相位的共同仿射包络。它直接回应 D048 的信息不足证据：形成差式时不先把真实共同源压成各门幅度。新增的是可实现的混合上下包络和前向闭包；固定链、符号仿射、差分和 reduced product 均有先例，尚不声称已完成新 Neural HZ 或 PLDI 级创新。

## 原非凸语义与观察语言

固定原 HZ frame H，其连续因子、全部原 bits、EQ/LE、原物理读出和 decoder 不变。连续源 x_j 是同一原 frame 中已认证的既有坐标或读出，带可靠盒界 l_j<=x_j<=u_j；它们不是独立噪声的新副本。盒可包含原相关性，所有额外原谓词仍保留。b_i 是原 gate active bit 的认证 0/1 视图，零预激活处两个原合法选择均保留。

混合形式为 V(x,b)=c+sum_j v_j x_j+sum_i w_i b_i。每个 x_j、b_i 有真实源身份和固定结构序号；数值或标签相同不是同一身份的证据。观察 (E,L,U) 声明既有仿射物理读出 E 在同一赋值上满足 L(x,b)<=E<=U(x,b)。

候选 N=(H,source_context,observations) 的具体化是 H 的原整数具体化，再要求所有观察在同一个原赋值上成立。只加入有证观察则具体化不变，空观察嵌入 H。按共同 frame 上的具体化包含定义预序，再按相等取商得到偏序；不声称可计算最佳抽象、完整格或全网最精确闭包。原 HZ 加相同有效行有相同逻辑强度。

这个语义仍为非凸，因为原二元域和门关系始终保留。连续盒与下面的查询凸包只是证明外包的工具，不整体替代 HZ。source token 和形式检查不构成原 ONNX、HZ 列、active 方向或 decoder 的认证；调用方仍承担这些证明。

## 统一 literal 规范化

对连续项 v_j x_j，先合并同源系数。若 l_j=u_j，把该项精确计入常数。否则，v_j>0 时令 z_j=(x_j-l_j)/(u_j-l_j)，v_j<0 时令 z_j=(u_j-x_j)/(u_j-l_j)。权重 a_j=abs(v_j)(u_j-l_j)>=0，常数增加 min(v_j l_j,v_j u_j)。

对二元项 w_i b_i，正系数使用 z_i=b_i，负系数使用 z_i=1-b_i 并补偿常数。删除观察中的零系数不删除原原因子或 bit。这样

```text
V = c0 + sum_{j in C} a_j z_j + sum_{i in B} a_i z_i,
a>=0, z_C in [0,1], z_B in {0,1}.
```

固定顺序为连续源按原结构序号，再接原 bits 按其结构序号。不同类型不允许因序号相同而合并。该顺序不使用实例身份、属性、LP 点、margin、终端状态或事后收益。

## 混合上包络

按上述固定顺序，定义每项的 prefix 边际

```text
m_i = min(a_i, max(0, c0 + sum_{j<=i} a_j)),
G_plus(V) = max(0,c0) + sum_i m_i z_i.
```

D046 的超模增量证明给出所有二元顶点上的上界。ReLU(c0+a*z) 是凸函数，G_plus 仿射，因此将任意连续 z 表为盒顶点的凸组合可得

```text
ReLU(V) <= G_plus(V)
```

于完整连续 cube 上成立。证明中的顶点讨论不是运行时枚举；构造只做一次固定扫描。恢复 z 的源坐标表达得到同一 x、b 上的仿射形式，所有偏置和尺度补偿必须保留。

当 V 只含原 bits 时，该公式与 D046 完全一致；当仅含连续源时，它是已知单个仿射 ReLU 的盒凸包有效上面，而不是新的凸化定理。所有这类已知上面不等于一条固定顺序面的紧性。

## 保留连续源的混合下包络

不能把连续源直接塞入 D046 的 binary singleton minorant。改用以下有限规则。令 S_C=sum_{j in C} a_j z_j，固定

```text
theta = 1 if c0 + (sum_{j in C} a_j)/2 > 0 else 0,
n_i = min(a_i, max(0,c0+a_i)), i in B,
G_minus(V) = theta*(c0+S_C) + sum_{i in B} n_i z_i.
```

逐个加入取值为 1 的 binary literal。由于正部的非负增量随已有基值单调不减，基值含 S_C>=0 及此前 bits，故每次真实增量至少为 n_i。于是

```text
ReLU(V) >= ReLU(c0+S_C) + sum_{i in B} n_i z_i
        >= theta*(c0+S_C) + sum_{i in B} n_i z_i.
```

最后一步只用 ReLU(t)>=theta*t，theta 为固定的 0 或 1。中点仅选择一个有效切面，不替换原参数或固定任何原相位。整个规则只读取结构和已认证系数/盒界。

因此 G_minus 对原 binary bits 和所有连续 source 值健全；不声称对 fractional bits 逐点低于 ReLU。无连续源时 theta*c0=max(0,c0)，恰好恢复 D046。无 bits 时退回一个普通 ReLU 下支撑面；原门的 t>=0,t>=w 两行不删除。固定连续源可先精确代入，不以此改变原 factor 身份。

## 仿射和神经算子的前向闭包

对 E=c+sum_j t_j E_j，同一源和读出先合并，再按 t_j 的符号组合其 L_j、U_j。新的上下式仍是同一 source context 上的混合仿射式。这是已有符号仿射传播；稀疏源支持的并集与实际合并成本都需支付。

令 r=ReLU(h)、t=ReLU(w) 为原门，delta=h-w。有证观察 L_delta<=delta<=U_delta、L_w<=w<=U_w 给出

```text
lower(r-t) = -G_plus(-L_delta),
upper(r-t) =  G_plus( U_delta),
lower(t)   =  G_minus(L_w),
upper(t)   =  G_plus (U_w).
```

前两式由 ReLU 增量界成立，后两式由单调性及上述混合上下包络成立。差值和真实伴随值均保留，所以后续混权可继续使用 q=d+p 的固定读出恒等式。不同原相位不重命名成同一 bit。所有原门谓词及其 bits 仍由 H 保留，本观察变换不替代原门构造。

Conv 使用实际稀疏权重和原 padding/stride/dilation；Add 必须在同一 frame 的同一个赋值上合流，Concat 仅作身份保持的坐标嵌入。对已覆盖算子的有限拓扑组合可归纳得到健全性，不是无损闭包或统一精度保证。没有认证原算子的部分仍是缺项。

非点 BN 参数需要可靠 L/U 源表达式或完整显式误差界；不能用中点代替真实网络。本组件只处理已认证的精确有理包络系数，不负责模型解析、BN enclosure 或原门绑定。

## 比较控制和应拒绝的捷径

CONTROL.md 将固定普通混权、非零偏置的两门正控，证明一个混合 source 上面能排除同时满足两个完整独立 source labelled hull 及旧 D020 四行的点。它不证明超过任意 D029 观察或完整 joint hull，不把普通已知理论重新计作新颖性。

只有把连续源加到语法中并不够。如果先逐源坐标求绝对值上界、再相加，可能已被完整各门 source hull 蕴含；另有数学审计记录该否决理由。也不能把多门残差加一个聚合伴随量当作任意后继混权的无损基：不同权重方向需要足够的真实伴随读出。

## 完整成本和实现范围

一个 k 项混合形式的 literal 变换、prefix、binary singleton 和恢复需 O(k) 精确标量工作；未经规范化的身份检查、合并与排序另计，最坏 O(k log k)。连续尺度恢复含除法和常数补偿，512 位界适用于输入及每步结果。上面不需要运行时顶点、相位子问题、LP 支持 oracle 或新 bits。

仿射/Conv 的合并成本为实际每个非零权重所触及的 source/bits 支持总数，不只是权重 nnz。多层支持可能扩成整个 root 维度；有限观察不是免费压缩。四条终端 LE 保留 readout、连续 source 和相位系数；实际 latent 展开、RHS、归一化 slack、峰值共存、所有旧谓词、证据、A/A^T、终端查询与 decoder 都付费。CPU 数学资格不等于完整物理或 GPU 资格。

GPU 候选形式为 typed gather、分段 prefix/reduction 与稀疏系数恢复。但有理规则不能直接当浮点实现：源尺度、prefix、负 literal 回写和转置累加均需可靠向外误差分析。当前只有数学组件，不新增 GPU 初始化或重跑旧失败版本。

实现默认关闭，只接纳上下包络在完整 source box 和 bit cube 上相容的保守有限子类；依赖原额外谓词才相容的包络可能被拒绝，不据拒绝宣告 UNSAT。形式 token 不是安全边界或原网络证书。共享 context 不匹配、未声明身份、位长、支持、数值或其他前提失败均 fail closed，不删除原因子。

数学执行前须完成 PREREG、CONTROL、实现及测试的冻结。本页本身不授权真实网络、native、GPU 或回放；全部原人口和预算保持。正式 1870/2413 及独立 E0 61/400 不变，formal_gain=0。
