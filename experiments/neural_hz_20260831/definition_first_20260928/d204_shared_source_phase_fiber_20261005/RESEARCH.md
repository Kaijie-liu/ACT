# 共享源尺度与原相位耦合的 Neural HZ 候选

用户再次明确，核心是提出强大的非凸 Neural-HZ，而不是改善 helper、加载器或存储。本轮据此收束到一个域内问题：怎样让共同输入依赖与跨神经元相位关系共同参与后续传播。得到一个保留共享尺度的局部精确凸包定理，以及把该信息直接用于域查询的健全规则。这些是候选原语，不是已完成的新抽象域或真实网络突破。

本文只记录纸面证明和只读核查，没有新候选代码、数值实验或运行预注册。原始连续源、整数相位、EQ/LE、共享身份、活消费者和输入重构均不删除。正式 1870/2413 与独立 CIFAR100 25 加 TinyImageNet 36 即 61/400 均不变，新增均为零。

## 定义创新的判据

候选必须同时交代域元素与具体化、与 HZ 的关系、普通网络算子的健全传播、可直接消费关系的域查询，以及全部构造和终端费用。一个新容器、追加几条已有有效不等式或另接一个支持器，均不足以证明定义创新。

本轮对象可写成 N=(H,B,R,D)：H 保留同一源状态、原二元相位和全部旧谓词；B 是共享源与原相位耦合的幅值关系块；R 是全部消费者的共同仿射读出；D 是原输入 decoder。具体化是同一个完整 latent 赋值同时满足 H 和全部 B，再由 R 读出的集合。无 B 时嵌入原 HZ。原生相位保持二元，本文中的 [0,1] 仅用于分析局部线性外包。预激活为零时两个原标签都保留。

对真实 ReLU 图加入本页已证有效关系，整数具体化不变。因此这里没有证明一个超越精确 HZ 的新集合类；潜在价值是更适合神经网络的结构化表示和组合演算，能否用可承受费用保留并消费更强信息。若旧 HZ 加同样信息也能证明，应如实列为同信息参照，不能靠较弱参照宣称创新。

## 保留共同源尺度的局部关系

设 0<l<u，k1,k2,a,b>0，ab<1。一个已认证的同父源仿射量 s 满足 l<=s<=u。定义

```text
P = { e>=0 : e1<=k1+a*e2, e2<=k2+b*e1 },
F = { (s,e,beta) : l<=s<=u, e in s*P,
      beta in {0,1}^2, beta_i=0 implies e_i=0 }.

U1=(k1+a*k2)/(1-a*b), U2=(k2+b*k1)/(1-a*b).
```

s 是一个共享量，不是给两门各复制一个独立区间。beta_i=(original_signed_bit_i+1)/2 只表示原标签的视图，不引入新 bit 或相位乘积列。若 s 是更大父元素的仿射读出，仍保留它与全部旧源、谓词及消费者的联系。

完整局部 conv F 恰由以下十条 LE 加 s 和 beta 的区间界表示：

```text
e1>=0, e2>=0,
e1<=U1*u*beta1,             e1<=U1*(s-l+l*beta1),
e2<=U2*u*beta2,             e2<=U2*(s-l+l*beta2),
e1<=k1*u*beta1+a*e2,        e1<=k1*(s-l+l*beta1)+a*e2,
e2<=k2*u*beta2+b*e1,        e2<=k2*(s-l+l*beta2)+b*e1.
```

没有辅助连续列、beta*s 列或运行时相位枚举。整数标签上的这些行与 F 相同。将它们与任意额外源谓词相交仍健全，但不能由此断言交集是整个共同源网络的理想凸包。

## 共同生成质量的精确消元证明

P 的四个顶点是 0、(k1,0)、(0,k2)、(U1,U2)。下标依次为 0、1、2、12。根据 D203 的带标签顶点生成表示，conv F 等价于存在质量 lambda 和尺度质量 eta，使

```text
lambda>=0, sum(lambda)=1,
l*lambda_j<=eta_j<=u*lambda_j, sum(eta)=s,
e1=k1*eta1+U1*eta12, e2=k2*eta2+U2*eta12,
lambda1+lambda12<=beta1, lambda2+lambda12<=beta2.
```

必要性由对各个固定 s 的顶点分解得到。充分性中，在 lambda_j>0 的每个顶点使用尺度 eta_j/lambda_j，位于 [l,u]；零质量项的 eta 也为零。正坐标强制标签为一，其余自由标签按剩余概率分配，即可同时补全两个 beta 均值。共同 s 均值仍为 sum(eta)，没有分别选择尺度。

先固定 eta，消去 lambda。令

```text
lambda_j=eta_j/u+delta_j,
d_j=(1/l-1/u)*eta_j,
B1=beta1-(eta1+eta12)/u,
B2=beta2-(eta2+eta12)/u.
```

需要 0<=delta_j<=d_j、sum(delta)=1-s/u，且 delta1+delta12<=B1、delta2+delta12<=B2。B1,B2 必须非负。在这两个容量约束下，后三项可达到的最大总量恰为

```text
min(d1+d2+d12, B1+d2, B2+d1, B1+B2).
```

这可直接固定 delta12=t，再最大化 min(d1,B1-t)+min(d2,B2-t)+t 得到；四项也是四种容量覆盖的上界。加上自由 delta0<=d0，最大值不小于目标是充分条件，因为该可行集含零且向下闭合，可同比缩放到所需总量。四个条件化简后为

```text
s>=l,
eta1+eta12<=s-l*(1-beta1),
eta2+eta12<=s-l*(1-beta2),
eta0/l-eta12/u>=1-beta1-beta2.
```

B1,B2>=0 另给 eta1+eta12<=u*beta1、eta2+eta12<=u*beta2。这一步保留了共同尺度产生的联合容量行，不能只证明各门边际可行就省略它。

接着令 t=eta12，r_i=min(u*beta_i,s-l+l*beta_i)，E=e1/k1+e2/k2，T=U1/k1+U2/k2-1。则其余 eta 唯一为

```text
eta1=(e1-U1*t)/k1,
eta2=(e2-U2*t)/k2,
eta0=s-E+T*t.
```

全部条件等价于两个上界 t<=e1/U1、t<=e2/U2，以及下列下界：

```text
t>=0,
t>=(E-s)/T,
t>=(e1-k1*r1)/(U1-k1),
t>=(e2-k2*r2)/(U2-k2),
t>=(E-s+l*(1-beta1-beta2))/(T-l/u).
```

分母均为正。两个 r 下界分别与两个上界比较，恰给十行中的四种上界 e_i<=U_i*r_i 和 e1<=k1*r1+a*e2 及其对称式。质量下界 (E-s)/T 与上界比较只给 plain feedback；因为 r_i<=s，它们已经蕴含。

最后的联合容量下界与 t<=e1/U1 比较，化成

```text
e2-(b-k2*l/(u*U1))*e1 <= k2*(s-l+l*(beta1+beta2)).
```

它由 e2-b*e1<=k2*(s-l+l*beta2) 及 e1<=U1*u*beta1 相加得到。另一上界对称。因此每个下界都不超过每个上界，可取下界最大值作为 t；再按前述容量构造 lambda，得到共同见证。必要性与充分性均成立。

## 从同父 ReLU 生成与直接消费

设 q_i=ReLU(f_i)，f1、f2、s 都是同一父元素上的完整读出。若整个父元素上已认证

```text
f1-a*f2<=k1*s, f2-b*f1<=k2*s, l<=s<=u,
```

则真实图满足 q1<=k1*s*beta1+a*q2 及其对称式，因而属于上述 F。证明使用 q_i=beta_i*f_i 和 q_i>=f_i、q_i>=0，包括零点的两个标签。认证必须在完整当前抽象父域成立，不能只在具体网络样本或选中相位上成立。优先合并共享源系数后认证，不能把独立区间差当作已保留共同依赖。

十行 hull 已经消去了尺度与原 bit 的乘积，但仅把这些行存入 predicates 还不够。只读检查发现 D136 的 support 忽略 predicates；其 relu 又硬编码返回原 Fiber。因此追加行或简单子类不会自动取得下述传播收益。不能把测试中手写的外部证书算成该组件自身的能力。

可以内生消费这项关系。设原全域下界 f_i>=-ell_i，ell_i>0。把原图行 q_i<=f_i+ell_i*(1-beta_i) 与新行 q_i<=k_i*(s-l+l*beta_i)+a_ij*q_j 作正系数组合，得

```text
q_i<=h_i+M_ij*q_j,
h_i=k_i*(l*f_i+ell_i*s)/(k_i*l+ell_i),
M_ij=a_ij*ell_i/(k_i*l+ell_i).
```

h_i 在同一父域仿射且非负，因为 l*f_i+ell_i*s>=ell_i*(s-l)>=0。两门 M12*M21<ab<1。于是它属于 D138 已证明可直接查询的非负收缩反馈幅值块。对同一个联合读出方向求得的查询系数 v 与实际父赋值无关；先形成 v1*h1+v2*h2，再与活源或 skip 的仿射读出合并，最后才对父域求上界，能够保留源抵消。这里 v 不是定义关系块的常数 k1、k2。

这是固定的结构规则，不依赖属性 margin、LP 状态、实例或公开标签，没有失败后补救。数学上它复用 D138 的支持公式及线性证书，并不是一个新的优化原理或独立辅助验证器。该公式对选中的两行幅值子系统给精确 support；恢复全部 H 和其他关系后仅是健全上界，不是完整 Neural-HZ 的精确查询。

多个块的组合仍须确保依赖图、共享父读出和查询成本满足合同。不能未经证明将任意循环或任意相位相关尺度纳入此闭式查询。所有未被消费的旧关系仍然保留；本页不授权删除原图或二元因子。

## 带活尺度和后继 ReLU 的严格正控制

采用一个普通三维多面体父域，不切分输入或相位：

```text
1<=s<=2, -s<=x<=s, -s<=y<=s,
f1=x+y/10+1/4,       q1=ReLU(f1),
f2=x/2+y/10+1/4,     q2=ReLU(f2).
```

两门有偏置且法向不平行。取 a=1、b=1/2、k1=1/2、k2=7/40。整个父域上 f1-f2=x/2<=s/2，f2-f1/2=y/20+1/8<=7*s/40。门一的全局下界是 -39/20。

新行 q1<=((s-1+beta1)/2)+q2 与原图上行合并，直接得到

```text
J=(49/20)*q1-(39/20)*q2-f1/2-(39/40)*s<=0,
child=ReLU(J-1/20)=0.
```

同一域内消费公式给 h1=(10/49)*f1+(39/98)*s、M12=39/49。对完整方向 (49/20,-39/20)，D138 的系数为 (49/20,0)；活源和活尺度同时抵消，得到同一全父域结论。这个方向只需第一条消元行，不能据此宣称已验证所有块查询。

强参照保留同一父域、完整 source-labelled 单门凸包、原 bits、最紧的 D203 常数差分块 c1=1、c2=9/40，以及 q1<=s/2+q2、q2<=7*s/40+q1/2 两条 source plain feedback。它仍允许

```text
s=11/10, x=y=0,
q=(29/40,1/4), beta=(1/2,1).
```

第一门的单门凸包见证是在同一个 s=11/10 上，源 (x,y)=(219/200,21/20) 及其负值各半，严格位于源边界内。f1 分别是 29/20 和 -19/20，故 q1 均值 29/40、beta1 均值 1/2。第二门取真实源 (0,0,11/10)。两个源均值完全一致，常数块与 source plain 行也全部满足。但 J=73/800，精确后继允许 child=33/800>0。

这证明保留共同尺度的关系严格强于所列强参照，不是原网络反例或新增 CERT。完整双门联合 hull、或旧 HZ 加同样十行，也可以证明新结论。这个 cone 父域是合法的带谓词 HZ 控制，不声称真实 CIFAR 输入盒已具有该结构，更不声称本例建立了外部新颖性。

## 成本和真实适用范围

局部关系从 D203 常数块的六行变为十行；在独立坐标 (s,e1,e2,beta1,beta2) 上，十行有 26 个 LE nnz，不含共同 s 区间和原 bit 端点。逐项为非负行 2、幅值上界 10、耦合上界 14。没有新增幅值或 bit 列。如果 s 只是已有源的仿射读出，实际 lowering 必须计其每行展开；不能按一个免费 s 标量计费，也不能免费增加别名等式。

源尺度发现、差分认证、原相位绑定、全部历史谓词、所有消费者、原图、终端转换、具体输入重构、证据和 host/device 共存均另计。所选子系统的闭式查询可批处理不等于 GPU 或端到端提速已经实现。没有新模型加载、GPU 运行、shadow 或正式回放。

本轮只读来源核查再次确认，D025 large 工件保存原参数包、3072 维输入盒，以及 1600 个首层稀疏 forms，共 34752 项。D015 medium 部分工件第 1 项保存原参数包和输入盒，没有已展开 forms，不能继承该失败运行的资格。两 CIFAR 的数值包覆盖首两次 ReLU 与活 skip，但不是完整模型；更晚 Conv/BN 数值不完整。Tiny 没有找到同等持久化参数包。

D120 三份 complete 文件只保存 source bounds、消费者和原模型/extractor 引用，没有完整 source forms 或权重数组。三个文件的 1600 源界均不代表完整首层人口，更不能代替同源系数。ONNX 节点与通道坐标也不是已认证的原生 HZ bit 列号。

后续若进行真实试验，应复用原冻结 extractor 路径，对三个原 model/spec 统一绑定，只形成冻结结构所需的完整稀疏源支持和差分证书。不能新建泛化 loader、按模型挑缓存捷径或重跑旧 D120 全消费者算法来替代本体研究。D067 的旧 global-even 人口没有双未定 pair；保留这个负结论，不事后只挑 crossing 门。

## 本轮结论与下一决策

本轮将 D203 的常数相位块扩展成共同源尺度块，并补全了一条域内消费途径。它比仅追加约束更接近有用的 Neural-HZ 演算，但还没有建立一个新的强大抽象域。先前 D138 已允许源仿射反馈 cap；本轮新增的是原相位与共同尺度如何共同生成更紧 cap 的精确推导，而不是首次提出源仿射 fiber 或新的 support 算法。

下一次选择实现前，必须检验该共同关系能否在普通真实结构中统一生成，并证明可消费精度与完整费用的优势。没有这样的证据，就记录局部结果并调整数学假设；不能继续用 adapter、重复测试规模或孤立正控充当本体突破。smooth、Transformer、全 GPU、所有家族与正式新增解仍是未完成目标，不把 ReLU 局部结论外推过去。

任何数值候选依然默认关闭，先独立 prereg/freeze；最新已通过数学人口为 D180 的 4056 tests/213 files，不能缩小或隐性替换。然后才是同结构真实比较、shadow、逐家族和完整 2413，以及独立 400 的同路径回放。数学正确、创新性、运行资格、性能和正式成绩分开记账，全部原安全与资源边界不变。

## 来源与不可变性记录

本轮日期为 2026-10-05 Australia/Sydney；分支 redu-hz；HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。原工作区改动完整保留。归档前 tracked diff SHA256 为 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5；该值不代表未跟踪源码清单。

以下为只读锚点，路径以 experiments/neural_hz_20260831 为根：

```text
0fd93920413f63ad6b3b65849ea4d857c416c6bf4ab56263c63ff73416ebe80c  GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md
100d6bc045e4e19a473001929063a7166799278fe11d2000d2a650477ba65d1d  definition_first_20260928/d203_phase_incident_pair_20261005/THEORY.md
cd2335196d122348a5994fe9aa01895d784ffef7f8196c4eac7201f873cf1f91  definition_first_20260928/d138_block_feedback_candidate_20261003/DEFINITION.md
7d95ae20a663133877d2812468381360c0b88e0fc3bc71c314f1abf50fbc1943  definition_first_20260928/d136_shared_fiber_component_20261003/fiber.py
fbab84537df071153a7d9362161b2c9aa85b80c8b4c605ff1e1149170e0965e0  results/d025_interval_capacity_20260930_v1/complete_0.json
bbf0d17e48439edc11b352ad4d38ea8fc3ad0e9e0bf99108802b51fd7faac456  results/d015_source_shielding_20260928_v2/partial_source_evidence.json
8bb8a5075e5575df40bda7ab629f9d2c0f904b73a5aac04b96fc8b39bf2b0f9d  results/d120_mixed_consumer_source_20261002_v1/complete_0.json
fb236df05faafd5f2403326fb5306a1d2cac5d563b7f6d011286e374e4143007  results/d120_mixed_consumer_source_20261002_v1/complete_1.json
a891450794e736132c5e357cec1d8fc2071fa4e322d72de3811cce7f3622b6b4  results/d120_mixed_consumer_source_20261002_v1/complete_2.json
```

范围交叉参考：[D150 完整前缀范围](../d150_stable_phase_charts_20261004/REAL_NEXT.md)、[D181 来源与实际接口审计](../d181_shared_coefficient_frontier_20261004/REAL_BINDING_AUDIT.md)。本轮未作新的外部文献新颖性结论，已知原理沿用 [D138](../d138_block_feedback_candidate_20261003/DEFINITION.md) 和 [D203](../d203_phase_incident_pair_20261005/THEORY.md) 的局限说明。

本轮研究方法为纸面推导、独立并行复核、源码/历史证据只读核查；没有候选 import、AST、编译、collection、求解器、模型或测试执行，没有后台数值任务。只在本新目录归档，未改生产、旧实验、历史模型、日志或正式结果；没有 commit/push。pages:write-page 技能仅用于本地记录，将证明、局限与成绩分开，不发布外部 Page，也不声称页面渲染资格。Goal 保持 active。
