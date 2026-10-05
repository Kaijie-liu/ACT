# 共享源上的原生 Attention 生成元与关系查询

这里给出一个 smooth 方向的非凸域候选及专用查询，而不是把逐坐标界命名为新域。对固定 query、逐 token 仿射 score/value 和独立来源，可精确求单头线性方向的实数上界；另外可以认证保留原输入斜面的线性外包。它与本目录的 PWA 幅值外包是两个分别核验的构件，没有声称组合已经通过实验。

## 具体化与原 HZ 的关系

元素 E 保存原连续源 z、全部原 bits beta、线性谓词 P、共享身份的非线性生成元表 A、线性读出 L 和原 decoder。其具体化是：

```text
gamma(E) = { L(z,beta,eta) :
             原源范围和全部二元类型成立,
             P(z,beta,eta),
             eta_j = A_j(z,beta,eta_<j) 对每个生成元成立 }.
```

生成元表是无环的；本轮具体增加的是一个联合 Attention 向量，不是为每个方向复制输入：

```text
A(x) = sum_i exp(s_i(x_i)) V_i(x_i) / sum_i exp(s_i(x_i)).
```

所有输出共用这些 x_i、score 和分母。A 为空时精确嵌入原 HZ；不同输出方向的最大化见证不能拼成一个联合输入。具体化相对于已认证 frontier，不能恢复 frontier 在此前丢掉的相关性。语义包含定义于相同可见变量，没有声称最佳抽象或完整格可计算。

有限线性 HZ 固定二元因子后为多面体；有限辅助变量及投影仍只得到有限多面体之并。两 token 可得到保留源的图 `(x,sigmoid(x))`。在 [-1,-1/4] 上 sigmoid 严格凸，图上任何两个不同点之间的弦都不在图上，故一个凸多面体子集至多包含一点。有限并不能表示整条曲线。这里证明的是共享输入输出图的表达差异，不是声称单个输出区间非凸。

不过，具名函数图和 typed functional sets 已有先例；上述一般定义本身不能算新贡献。需要由下面可计算的结构原语、关系编译和实际收益支撑研究价值。[先例和真实源范围](SOURCE_AND_PRIOR_ART.md)分别记录。

## 算子闭包与终端边界

Affine/Conv 精确组合 L；同 source/frame 的 Add/Concat 作自然联接，不复制源。对真实下一 ReLU 的预激活 g，认证当前域上 l<=g<=u 且 l<=0<=u 后保留其原门 bit alpha，并加入一个幅值 q 及四行：

```text
q>=0, q>=g, q<=u alpha, q<=g-l(1-alpha).
```

与 native fibers 同时解释时，这对 g 精确，零点两标签都保留，没有相位枚举。每门新增一原 bit、一连续幅值、四行和实际 nnz，不能因生成元被延迟而不计费。一般后续块不再满足下面的低成本查询前提；表示闭包不等于高效查询闭包。

原 LP/MILP 不能精确消费指数图。实际终端只能接收从 native 图证明的线性上下界、source-slope 行和保留整数位的门关系，这是 sound outer compilation，绝不是 exact exponential MILP。可将行加到旧 lowering 上保其外包强度，但须支付全部旧成本；若替换 lowering，则必须重新验证逐例保旧。所有候选反例都经原输入 decoder 和具体网络验证，不能使用独立的 relaxed eta/p/product 当作真实输出。

## 独立 token 上的精确方向原语

固定一个 head 与输出方向，令 Xi 为紧非空盒，真实来源恰为其 Cartesian product，且：

```text
s_i = a_i*x_i+c_i,   v_i = b_i*x_i+d_i
Z(x) = sum_i exp(s_i) v_i / sum_i exp(s_i)
F(t) = sum_i max_(x_i in Xi) exp(s_i)(v_i-t).
```

分母严格正，且分子差对独立 token 可分。因此 `max Z<=t iff F(t)<=0`。对有效 score 界 L_i<=s_i(x_i)<=U_i，令 `m=sum exp(L_i)>0`、`M=sum exp(U_i)`，则对 delta>0：

```text
F(t)-M delta <= F(t+delta) <= F(t)-m delta.
```

故 F 连续、严格递减，唯一零点等于真实最大值。可用各 token value 的最小/最大范围作初始零点括号。若来源有跨 token 谓词或共享额外因子，product-box 计算仍可作上界，但不再具有 exact 性。不同 heads 共享输入且分母不同，分别最大值相加也仅 sound。

每个 Xi 到 (s_i,v_i) 的像是二维 zonotope，这只是原源的临时精确投影，不是把整个 Neural-HZ 替换成凸域。函数 exp(s)(v-t) 的 v 偏导恒正，满维像上的最大值在边界。至多 2d_i 条边可由生成元角序建立；沿一条边：

```text
s=s0+lambda a, v=v0+lambda b, 0<=lambda<=1
phi'(lambda)=exp(s)[b+a(v0-t)+ab lambda]
lambda_star=-1/a-(v0-t)/b, 当 ab!=0.
```

只需端点以及落在边内的驻点；只有 ab<0 才可能给内部最大值。退化线段使用同公式，点直接计算。这是有限投影几何上的优化，不是把输入/相位拆分后运行验证器。构造每方向 O(sum d_i log d_i)，一次 F 评估 O(sum d_i)；实数 exp 操作和所需数值精度不包括在这个算术计数内。

构造边界时可携带原 Xi 的端点 preimage，边上 lambda 对应两端点的凸组合。只有在实数零点 t_star=max Z 处，各 token 的最优见证组合才给出 ratio 的全局最优见证；任意固定阈值或有限精度 high 只产生待验证候选。不同方向/head 的见证不可合并，实际 ADV 仍需具体网络验证。无需为每条边复制整个 d_i 维源向量，可保留角序翻转和原生成元身份；若实现选择复制，就必须计入其二次存储。

## 超过独立盒的普通正控

固定 token 为 (s,v)=(0,0)，另一个 token 为 `s=x,v=-x+y/4`，x,y∈[-1,1]。这是满二维像。先取 y=1，导数符号为 `1/4-x-1-exp(x)`，在整个区间为负，故真实上界是 `5/[4(1+e)]`。独立 score/value box 给出 `5e/[4(1+e)]`。下一门 `ReLU(Z-1/2)` 前者可证恒零，后者不能。

这只区别独立盒。下面给出更强且明确限定的参考，以免把该简单比较夸大。

## 精确概率图仍不足以修复独立乘积的正控

仍保留固定零 token，改用 `s=x,v=3/4-x+y/4`。真实最大值为 Omega，其中 `Omega exp(Omega)=1`：y=1 时 `Z=sigmoid(x)(1-x)`，导数符号为 `-(x+exp(x))`，唯一最大点 x=-Omega，最大值 Omega。

考虑保留精确 p=sigmoid(x)、独立 p*x、p*y、p*v 的全部标量范围 McCormick 行、补概率的同类行及 simplex 产品守恒，并加统一参考的二次 energy 行。这个明确的外包仍允许：

```text
x=-1/2, y=1, p=1/(1+sqrt(e))
T_x=-1/4+4(p-1/2)^2, T_y=p
T_v=(3/4)p-T_x+(1/4)T_y=p-T_x
补概率产品分别为 x-T_x, y-T_y, v-T_v.
```

这里 score/probability 点本身是真实的；只改了乘积。设 l=1/(1+e)、u=1-l。T_x 的 McCormick 下界为 `max(l/2-p,p-3u/2)`，上界为 `min(u/2-p,p-3l/2)`，该 T_x 位于其间；T_y 在 y 的上端点精确。v∈[-1/2,2]，取值 3/2，其 product 行同样容纳 T_v。补产品行由对称性和守恒成立。

energy 恰取等：`E=T_x-x/2=4(p-1/2)^2=2*||(p,1-p)-(1/2,1/2)||^2`。然而 `T_v=-4p^2+5p-3/4>0.5675541936>5673/10000>Omega`。可用正项 Taylor 级数及几何尾界证明 `37754/100000<p<37755/100000`；而 `t*sum_(k=0..5)t^k/k!>1` 对 t=5673/10000 直接证明最后一个严格不等式。因此下一门阈值 5673/10000 也能区分。

这个参考不是所有 affine-combination 的完整 RLT，更不是精确联合 product 图。也没有证明优于现有完整 Taylor/ZonoGPT 外包。该控制定位的是保留 native score/value 同源图相对指定独立乘积 lowering 的价值，不是实际 benchmark 新解。

## 保留输入斜面的外包编译

仅输出逐坐标界会再次丢失输入关系。为认证 `Z<=R(x)+b`，令 `R=sum_i r_i(x_i)`，选择固定 theta>0 和 rho。记 `w_i=exp(s_i),D=sum w_i,N=sum w_i v_i`。有恒等式：

```text
N-D(R+b)
 = sum_i [w_i(v_i-b-rho)-theta r_i]
   +theta rho-(D-theta)(R-rho).
```

用可靠 D/R 范围认证 `kappa>=max -(D-theta)(R-rho)`，例如区间矩形四角的最大值。则

```text
H(b)=sum_i max_Xi [exp(s_i)(v_i-b-rho)-theta r_i]
     +theta rho+kappa <=0
```

足以推出所需线性斜面。H 的严格递减率仍夹在 m、M 之间。对交叉项的区间处理损失相关性，因此这是健全 intercept 编译器，不是最佳 intercept 的精确 oracle。

廉价族 `r_i=lambda_i v_i+mu_i s_i+nu_i` 仍只使用二维像。局部目标的 Hessian 行列式为 `-exp(2s)<0`，内部不能取最大值。沿边的导数为 exp 乘一次函数减常数，非退化时至多两个根；常数函数直接取端点。需支付认证标量求根或 Lambert W 实分支的成本，不能继续沿用上一简单根的一次闭式费用。

此斜面族包含名义输入切线：`grad_(x_i)Z=p_i b_i+p_i(v_i-Z)a_i`。也可以选择任意固定有理 lambda/mu/nu；这些是自由证书参数，不是把不确定网络系数擅自取 midpoint。即使和 Taylor 用相同斜率，当前 intercept 由原 score/value 几何认证；尚未证明更紧。

## 斜面能穿过残差和下一门的正控

固定零 token，另一 token `s=x/4,v=1+y`，x,y∈[-1,1]。令 `B=exp(1/4)<9/7`，选择 `R=(1+y)/2,theta=2,rho=1/2,kappa=(B-1)/2,b=1/4`。在矩形 (s,v) 像上，H 的两个可能最大值为：

```text
H_1=(2B-3)(B+1)/(4B)<0
H_2=(7B-9)/4<0.
```

所以 `Z<=3/4+y/2`。真实 output-only box 的最佳上界为 `2B/(1+B)>1`；新斜面却能证明普通残差 `g=Z-y/2-4/5<=-1/20`，从而后继 ReLU 恒零。这一正控仅与 output-only box 比较；上一节的乘积控制是另一个问题，不能将两者拼成胜过所有强参考的结论。

## 数值和 GPU 尚未取得资格

精确性目前指实数定理。有限精度只能在认证的 F_upper(t)<=0 或 H_upper(b)<=0 时发布界；不确定时保留既有可靠外包或 UNKNOWN。所有 token 可用同一个 score 平移，不能各自平移后丢失相对权重。角序、近共线判定、指数、边内根、求和和前端 BN/sqrt 都必须有舍入证明。

BN 系数的区间外包只能给 (s,v) 的可靠外像，不能假称 exact image。关系行本身的源系数也须认证补偿。批量投影、排序、边扫描及 root 查询具备 GPU 并行结构，但目前未实现、未测量，不能宣称全面 GPU 或速度提升。完整人口、source 绑定、证据内存和旧终端费用见下一文件；本记录不授权未冻结运行。
