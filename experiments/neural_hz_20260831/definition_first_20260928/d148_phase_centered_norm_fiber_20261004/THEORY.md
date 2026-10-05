# 原二元相位参与共享余项中心的 Neural-HZ 候选

本轮修正了上一版源仿射范数域的一个实质缺口：新激活的原二元相位现在直接进入生成元，而不是只留在 guards 中。得到一条可重复的健全 ReLU 替换规则，以及同一父状态和同一预算下的整数语义包含定理。配套普通 biased affine、ReLU、共享 skip 和后继 ReLU 控制，给出严格的源与输出投影分离。

这些是纸面定义与证明，不是已实现的新域、真实网络新增解或 PLDI 新颖性结论。反射等距是基础代数；同信息的 centered perspective-energy 已经支配新球。尚未证明整网保旧、完整成本优势或 GPU 收益。正式新增收益为零。

本研究继续执行[定义优先目标](../../GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md)。与仅给原完整 HZ 图添加 helper 不同，候选替换新门的精确幅值关系；因此它也必须承担有损近似的保旧验证责任。精确 HZ 本来就拒绝不真实的门幅值，不能声称本候选在集合精度上超过精确 HZ。

## 域元素和原 HZ 嵌入

沿用[源仿射范数候选](../d147_source_affine_norm_fiber_20261004/THEORY.md)的共享载体，现明确让每次新 ReLU 的原相位进入 G：

```text
y = c + C xi + G beta + sum_k E_k e_k
xi in [-1,1]^p, beta in {0,1}^b
||e_k||2 <= R_k(beta)=a0_k+a_k^T beta, R_k(beta)>=0
P(xi,beta,e_1,...,e_K).
```

P 保留父状态的 EQ/LE、源与输入 decoder、原相位 guards、输出 masks 和其他已有谓词。每个连续因子、二元因子和 e_k 块都有唯一共享身份。具体化使用同一满足全部条件的赋值读取所有活消费者及输入。语义偏序是相同保留坐标上的集合包含；没有声称最佳抽象、完整格或廉价包含判定。

原 HZ 的 y=c_H+G_c xi+G_b sigma 通过 beta=(sigma+1)/2、c=c_H-G_b*1、C=G_c、G=2G_b、K=0 精确嵌入，P 与 decoder 同步作该双射。原 bits 不删除、不 pivot、不改成连续语义。一般固定相位切片是凸的，但全体相位纤维的并仍可非凸；保留的相位、mask 和中心共同参与具体化，不整体退化为 CZ、zonotope 或 ellipsotope。

新门采用下面的有损规则时，不保留它的完整 active-value 上侧关系，不再同时保留 e_new 的精确非线性映射。否则只是旧图附加球。此前父状态已有的条件仍原样保留；现阶段没有用“投影”名义删除旧 bits、父图或共享读出。

## 相位中心反射变换

对整个当前域上的 g=Wy+d，选在查询前固定的结构参考 tau in {0,1}^b。tau 不能依实例身份、标签、margin、求解状态或 dual 信息选择；全部取零是一个合法统一选择。定义

```text
gbar = W(c+G tau)+d
S_gamma = diag(2 gamma-1)
q = ReLU(g), gamma 为这些原门的原相位。
```

gbar 不必是可达网络值，不能把它当实际界。对每个真实门及其两个合法零标签，S_gamma*g=|g|，于是

```text
q = (1/2)g + (diag(gamma)-(1/2)I)gbar + e_new
e_new = (1/2)S_gamma(g-gbar)
||e_new||2 = (1/2)||g-gbar||2.
```

最后一个等号来自对角正负一矩阵的等距，不是新的范数原理。它也适用于符号不变的对角加权范数，但不能不加证明推广到任意非对角度量。旋转一个自由 Euclidean 球本身不会变强；这里有用的变化是中心依赖实际相位，并与 epigraph、同源 guards 共同约束幅值。

假设可靠认证

```text
K0 >= sup_(xi in [-1,1]^p) ||WC xi||2
Kk >= ||WE_k||2
d_j >= ||(WG)_:j||2
B(beta)=K0+sum_k Kk R_k(beta)+sum_j d_j |beta_j-tau_j|.
```

三角不等式给 ||g-gbar||2<=B(beta)。只保留 ||e_new||2<=B(beta)/2，删除它的精确映射，即为健全外包。由于 beta、tau 为二元，|beta_j-tau_j|=tau_j+(1-2tau_j)beta_j，半径仍是相位仿射的。这个界对整个父抽象状态成立，不只对原网络真轨迹成立，所以可递归组合。

更新为

```text
c' = (1/2)(Wc+d-gbar) = -(1/2)WG tau
C' = (1/2)WC
G'_old = (1/2)WG
G'_gamma = diag(gbar)
E'_k = (1/2)WE_k
E'_new = I
R_new = B/2.
```

固定 gbar 使新原相位线性进入中心，无需保留 gamma*beta、gamma*xi 或 gamma*e_k 乘积。次数保持一不等于维度或成本固定：G 的列数和共同余项块数继续增长。如果 gbar 为零，这一步不会凭空产生新的中心耦合，不能保证所有结构都有收益。

保留可靠 L<=g<=U、原 sign guards、q>=0、q>=g 和 q<=max(U,0)*gamma。在普通 crossing 情形，guards 可写成 g<=U*gamma、g>=L*(1-gamma)。真实零的两个相位标签都保留。抽象可行点不是反例；从同一原 decoder 取输入后，仍须在原具体网络与性质上验证 ADV。

Affine/Conv 精确左乘当前所有共享系数；Add/Concat 按身份合并或堆叠。若 Sy 同时保留作 skip，新 e_new 在 skip 的系数为零，旧 e_k 在两端保持同一身份。不能独立重采样 skip 的噪声。稳定门可在已有可靠稳定证书下作其精确常斜率变换；不能把未证稳定或参考跨区间当成消除余项的依据。

## 同父状态整数语义包含

比较上一版的固定中心球与本轮相位中心球，父状态、原源、全部旧谓词、g、预算 B 和新门的其余保留约束完全相同。令 a=2q-g，则 q>=0、q>=g 蕴含 a>=|g|>=0。

```text
old: ||a-|gbar|||2 <= B
new: ||a-S_gamma*gbar||2 <= B.
```

对于整数 gamma，每个参考符号匹配坐标的残差相同；每个不匹配坐标的新旧残差平方差为 4*a_i*|gbar_i|>=0。因此新球蕴含旧球。

这是保留父坐标、物理输出和原 bits 的投影包含，不把两版不同定义的新余项坐标强行当作同一个变量。可令

```text
e_old = e_new + (1/2)(S_gamma*gbar-|gbar|)
```

延拓出旧子状态。这里新增余项没有其他同名外部谓词；全部原有父谓词不变。

[控制记录](CONTROL.md)给出严格包含和后继门分离。但这个定理只适用于同一个父状态与同一个 B。两版独立递推后 G、参考和半径证书不同，尚不能推导整网层层支配，更不能证明完整 1870 个旧解保留。

## 分数相位不继承该包含

取源 x,y in [-1,1]，

```text
g=(1/2+3x/4, 1/2+3y/4), gbar=(1/2,1/2)
B=11/10, B^2=121/100 >= 9/8
L=(-1/4,-1/4), U=(5/4,5/4).
```

在严格内部源 (x,y)=(-3/5,0)，令

```text
g=(1/20,1/2), gamma=(1/2,1), q=(1/20,209/200)
a=(1/20,159/100).
```

源、guards、epigraph 和 mask 均满足。新残差平方为 5953/5000<=121/100，旧残差平方为 6953/5000>121/100。因此新连续 conic 松弛不自动包含在旧连续松弛中，更不能由整数包含推出任意有限 LP 降低方案的保旧。

对固定实际方向 v，新球发布有证线性行

```text
v^T(2q-g) <= v^T S_gamma*gbar + alpha_v B(beta)
alpha_v >= ||v||2.
```

v>=0 时，S_gamma*gbar<=|gbar| 对全部 gamma in [0,1] 成立，新上侧行蕴含同 v 的旧上侧行。混合正负方向不能套这个比较。整数语义已证明的旧球后果可以作为额外有证终端外包行，但其构造、储存和查询开销都要计入；不能因此虚报某个默认 lowering 已经支配旧版本。

普通终端 LP/MILP 只消费有证线性外包，不引入新 SOCP、QP、SDP、phase split、backward 或 dual rescue 路径。递归数学传播继续使用完整球语义，不把有限支持平面组成的更大多面体误当成原球。

实现时任何共享身份、预算、舍入、输入重构或证据前提未认证，以及数值或资源失败，都必须 fail closed 为 UNKNOWN 或原协议错误状态，不能记 CERT/ADV，也不能临时接入其他算法补救。

## 与已有关系的准确区别

项目 [D009](../d009_bounded_phase_energy_20260928/D009_DOMAIN_AND_RESULTS.md)已有 m=(2 beta-1)g=|g|、平方等距和共同源 Gram 能量。[D012](../d012_phase_budget_20260928/D012_RESULTS.md)已有相位错配量及固定参考 residual。若 eta 是 gbar 的合法参考相位，则真实门满足

```text
|| |g|-|gbar| ||2^2
  + 4 sum_i |gbar_i| (q_i-eta_i*g_i)
  = ||g-gbar||2^2.
```

括号就是原有参考错配费用。新球把这种费用保留在相位中心里；不能把它宣称为新出现的几何原理。旧档未给出的具体组合，是保留 C*xi 的共享多球域、每层新 gamma 线性进入 G 的有损替换闭包，以及本轮同父整数比较。

更强参照是 centered individual perspective-energy。对相同 gbar 与 B，定义

```text
u_i = q_i-gamma_i*gbar_i
v_i = g_i-q_i-(1-gamma_i)*gbar_i
u_i-v_i = 2q_i-g_i-(2gamma_i-1)gbar_i.
```

0<gamma_i<1 时，Cauchy 直接给

```text
(u_i-v_i)^2 <= u_i^2/gamma_i + v_i^2/(1-gamma_i).
```

端点采用 perspective 闭包：分母为零的项仅在分子也为零时有限。于是 sum_i[u_i^2/gamma_i+v_i^2/(1-gamma_i)]<=B^2 蕴含新球，分数相位也成立。这个参照仍是数学比较，不是要安装或调用新优化算法。

[Gunluk 与 Linderoth 的 perspective reformulation](https://jlinderoth.github.io/papers/Gunluk-Linderoth-09-TR.pdf)第 1.1、2.1 节提供已有 indicator 与凸函数 perspective 构造；这里的 neural 两支表达和上述不等式是本记录的直接推导，不声称该论文提出了本域。[Ellipsotopes 定义 2](https://arxiv.org/pdf/2108.01750)已有连续范数块及线性约束，[typed-symbol 研究](https://arxiv.org/pdf/2009.07387)已有有类型的连续与二值符号及依赖跟踪。因此混合符号、范数块和仿射闭包均不是新原理。

目前不能称为已经创新性足够的 Neural-HZ。必须进一步证明该表示与算子组合在完整代价下的实用区别；本轮正控只区分控制记录中精确定义的旧规则族，不区分所有已有能量/perspective 方法。

## 完整成本和实施条件

每个 m 宽新非线性块仍增加至多 m 维共同余项，保留所有原 bits、父谓词和活消费者。WC、WG、WE_k、后续 SG、半径相位系数都会增长或填充。source box 的 K0 不能当免费二次优化 oracle；可靠 Gram、对角优势或其他固定结构上界都必须支付计算、存储及向外舍入费用，松界可能消灭控制中的收益。

若 g、q、e 已物理化，新定义通常为 m EQ、4m nnz；上一版相同口径为 m EQ、3m nnz，本轮多约 m 个相位系数。两条 sign guards 加两条 epigraph 和一条 mask 通常共 5m LE、9m nnz，另计变量 bounds、源连接、旧状态及证据。

若直接在 q/g/gamma 上发一个方向的上下支持行，每对通常约 2[3*nnz(v)+b_R] nnz，b_R 是半径所涉旧相位数；若在物理 e 上发行为 2[nnz(v)+b_R]，但定义 EQ 不可漏账。常数零、重复列、稀疏布局可改变实际计数，不构成未测的速度声明。

当前没有 GPU 实现或速度资格。固定稀疏乘法与归约是潜在并行计算对象，终端、身份合并、范数认证和证据工作同样计费。没有固定宽度或渐近优于精确 HZ 的定理。平滑激活仍只有上一版的中点斜率候选合同；本轮 ReLU 反射不自动覆盖 GELU、Softmax、LayerNorm 或完整 Transformer。

下一项有价值的工作是冻结一套结构统一的参考、预算认证及有限查询规则，检验这个实际表示在真实同结构层上是否仍有可用界与完整代价优势。若只能重新保留全部旧门图再附加这些行，不算定义创新。任何候选执行仍先预注册、冻结，再经过原完整数学测试、真实目标、同结构 shadow、逐家族和全量回放；本纸面结果不放宽预算或人口。

## Provenance 和正式记账

记录日期 2026-10-04 Australia/Sydney。分支 redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。配置为纸面推导、只读独立审查、历史文档与一手论文核对；无可执行候选或新依赖。

无 candidate import/AST/compile、数值搜索、测试、模型、solver、GPU、shadow、replay 或后台作业。最新既有组件执行仍为 D136，资格不继承到本轮。生产与旧档未改，没有 commit/push。

正式 1870/2413=1063 CERT+807 validated ADV，独立 E0 为 CIFAR100 25、TinyImageNet 36，共61/400；均未更新，新收益均为零。Goal 保持 active、未完成。归档技能只用于本地新文档及证明、比较和能力边界的清晰区分，未创建外部 Page。
