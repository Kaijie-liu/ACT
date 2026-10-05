# 多门相位关系的直接物理表示与宽度分离

上一轮三门构造可推广为一般宽度定理：一个已知 Boolean facet 能直接成为带原相位标签的 ReLU 联合图凸包的物理 facet，且不能由所有少一门的完整同源 hull 推出。每组只需一条含 O(m) 个物理系数的行，不必建立 m² 个相位源乘积。

这不是新不等式，也还不是新的 Neural-HZ 定义。它给出了一个可用于评估候选域能力的非平凡层级，以及解析的关系生成原语；实际网络适用性、GPU 性能和正式分数均未建立。

## 定义与已知前提

宽度 m>=2。固定矩阵 K=K_m：第一行全为 1；其余行满足

```text
K_ij = 1   当 i+j<=m+1，
       -1  当 i+j=m+2，
        0  否则。
c_j=m-j。
```

对 beta in {0,1}^m、z in [0,1]^m、Z_ij=beta_i*z_j，已知 Imm22 为

```text
-beta1-c^T*z+sum_ij K_ij*Z_ij <= 0。                    (B_m)
```

其为 bipartite Boolean quadric polytope 的 facet，且该多面体维数为 D=m²+2m；连续盒一侧具有相同凸包。这些前提来自 [Sripratak、Punnen、Stephen，Proposition 2.2、Theorem 2.3、式 (2.1)、Corollary 2.11](https://www.sfu.ca/~tstephen/Papers/bbqp.pdf)，不是本项目的新原理。以下神经图投影、费用与比较界限是项目内推导。

## 任意同源门组的无乘积编译

保留全部原 HZ 连续因子、二元因子、guards、EQ/LE、frame 和输入 decoder。对一组同源真实预激活 g 和原幅值 q，有 q_i=ReLU(g_i)=beta_i*g_i。

先按共同 latent 精确合并 w=K^-1*g，再在完整当前父域上认证 ell<=w<=u。令

```text
t=max_j(u_j-ell_j)>0，z=(w-ell)/t，b_star=K*ell。
```

此时 g=t*K*z+b_star；代入 (B_m) 得到

```text
sum_i q_i-lambda^T*g
 <= (K*ell+t*e1)^T*beta-c^T*ell，
lambda^T=c^T*K^-1。                                    (P_m)
```

同一真实赋值解释所有量，所以此行对整个父域成立，不需要按相位求子问题，也不看 LP 结果或性质 margin。t=0 表示全部预激活已认证为常量，走已有常量结构处理而不是除零。界不能认证时，不生成本行；没有额外求解路径。

K^-1 不必作为稠密矩阵显式求逆。记 S_m=g1，对 r=m-1,...,1 递推

```text
S_r=(S_(r+1)+g_(m+1-r))/2，
w_(r+1)=S_(r+1)-S_r，最后 w1=S1。
```

由 (K*w)_1=sum w 和 (K*w)_(m+1-r)=2*S_r-S_(r+1) 直接验证逆变换。权重为

```text
lambda1=1-2^(-(m-1))，
lambdai=1-2^(-(m-i+1))，i=2,...,m。
```

因此 sum(lambda)=m-1，所有权重位于 [1/2,1)。各 K^-1 行的绝对系数和为 1，dyadic 系数位长随 m 线性增长；这不免除真实模型系数和源合并的可靠算术。

## 神经图上的物理 facet

考虑具体网络

```text
z in [0,1]^m，g=K*z-e1，q=ReLU(g)，原标签 beta。
```

先证明所有 Boolean 输入顶点都满足 sum(q)=c^T*z。令 s_r=sum_(j<=r) z_j。此时 g1=s_m-1，其他 g 依次为 s_r-z_(r+1)。因为 z 为 Boolean，

```text
sum_(r=1)^(m-1) min(s_r,z_(r+1)) = max(s_m-1,0)。
```

左侧恰数出第一个 1 之后出现的 1；全零时两侧也为零。再用 ReLU(s_r-z_(r+1))=s_r-min(s_r,z_(r+1))，以及 sum_r s_r=c^T*z，得到结论。这是纸面计数恒等式，不是运行时输入或相位枚举。

固定 Boolean z 时，(B_m) 左侧对 beta 的最大值为 sum ReLU(g)-c^T*z=0。因此其 facet 上每个整数顶点必逐项满足 beta_i*g_i=ReLU(g_i)：非零 g 强制正确标签，g=0 保留原来的两种合法标签。所有这些 facet 顶点都是真实 labelled NN 图点。

定义线性投影

```text
pi(beta,z,Z)=(beta,z,q)，
q_i=sum_j K_ij*Z_ij-delta_(i1)*beta1。
```

pi 的秩为 3m：每个 q 都有独立的非零 Z 行可用。Bell 法向完全来自物理法向 F=sum(q)-c^T*z，所以 ker(pi) 包含于该 facet 的切空间。已知 facet 的像维数为

```text
(D-1)-(D-3m)=3m-1。
```

其像位于真实 NN 图凸包 P 的 F=0 面内。另一方面，取 z=ones/2，有

```text
sum ReLU(g)=(m-2)*(m+1)/4，
c^T*z=m*(m-1)/4，F=-1/2。
```

故 P 为 3m 维，F<=0 确为其物理 facet。严格点可小幅扰动避开零预激活，但这不表示上面的全部 facet 顶点都已替换成非零标签见证。

## 不能由所有更小门组替代

令 P_S 为完整 P 在 (z,beta,q_S) 上的投影，其中 S 为 proper gate subset。这里甚至保留所有原 beta 均值和其 guards 的联合可实现性，只丢至少一个幅值 q_i；比仅保留 S 自己标签的局部 hull 更强。

因为 F 中每个被丢 q_i 的系数均为 1，F=0 的仿射超平面投影到 (z,beta,q_S) 后是整个相应仿射空间。因此上述 facet 的像在该投影中满维。

取 Bell facet 全部顶点的均值，投影为 p_star。线性映射将相对内部映到像的相对内部，所以对每个 proper S，pi_S(p_star) 位于 P_S 内部。只需考虑有限 m 个 leave-one-out 最大组。因而存在 epsilon>0，使

```text
p_star+epsilon*e_(q1)
```

仍属于所有 P_S 的共同交，却有 F=epsilon>0，违反联合物理行。不含 q1 的投影根本不变；更小组自动包含在这个比较中。

这是每个 m>=2 的严格宽度层级。一般证明是存在性证明，没有给出统一数值 epsilon，也没有授权展开全部 facet 顶点。三门时 D175 已给明确有理点、普通非零偏置、内部见证和实际下一门分离。

若另存只约束 beta 均值的纯 phase 分布，以上扰动不改它；但若要求每个 P_S 的见证进一步共享一份指定的完整高阶 phase-slot 质量，则增加了新的共同坐标，本维数证明不覆盖。不能自动把这个更强要求附到定理上。

一般 m>=4 的证明合法保留零标签，但尚未证明一般系数扰动鲁棒性或训练模型上的适用率。用户要求的普通网络能力目前仍只能由真实结构实验回答，不能把这个存在性家族算成 benchmark 收益。

## 完整费用与 GPU 边界

已有物理 q、g、beta 时，每组新增一条 LE，至多 3m 个变量系数；零新连续乘积、零新 bit。源读出展开后的实际 nnz 另计。对组内源并集维数 d，合并、逆变换与盒支持的稠密工作量为 O(m*d)，须生成 2m 个可靠有向界；串行递推可用 O(d+m) 临时空间，但并行扫描的工作区不能按这个串行峰值记账。

递推是仿射 prefix scan，原则上能以 O(log m) 并行深度和 O(m*d) 工作映射到 GPU；这只是代数并行性，不是已有 GPU 实现、速度或数值认证。主机设备共存、证据、全部历史谓词、terminal 转换及解码成本仍须支付。

全层任选 m 门的组数为组合数量，不能用每组 O(m) 掩盖组选择成本。本轮没有选择全枚举、动态最优分组或由性质结果驱动的策略。任何具体规则必须结构统一且预注册。

加入所有这些逻辑后果仍不改变原 HZ 的整数具体化；提升的是查询表示。只把 P_m 当作一个 helper 或缓存行目录，不构成用户要求的新非凸域。本定理保存为强对照与关系生成支撑，不据此启动实现或宣布创新完成。
