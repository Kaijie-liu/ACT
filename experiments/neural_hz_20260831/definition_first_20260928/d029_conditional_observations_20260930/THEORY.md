# 条件观察的精确投影与共享源求界

本轮得到两个可组合的数学结果：一类条件观察提升可直接投影成四条线性行；所需的混合权重支持界可以由固定正项配对和共享源仿射支持构造。它们补全了一个前向关系组件，不构成已经确立的新 Neural HZ 域。所有结论均为纸面证明及独立复核，没有执行测试、网络、求解器或 GPU。

## 语义载体与适用范围

保留原 HZ 的连续坐标、全部原二元相位、EQ/LE、共享 frame 和输入 decoder。额外观察记录实际前向表达式、共同源身份、有效上下界和证明来源。辅助盒只是包含真实源的求界工具，不替换原非凸源谓词；没有把原二元因子改成连续因子。

这是已知 reduced-product 型框架。新增事实若由原整数语义推出，则整数具体化不变，LP 查询松弛可能变紧。空观察列表嵌入原 HZ；观察事实通过共享 Add/Concat 使用时，必须先对齐同一源身份。仅有该定义不构成创新，尚需实际组合规律与完整成本收益。

以下支持编译适用于同一已知盒内的仿射源表达式经过一层 ReLU 后的带符号读出。它不自动覆盖任意嵌套非线性表达式；不能在没有证明时把后层神经值当成原输入的仿射式。所有观察应由原前向结构统一产生，不按性质、标签、margin 或求解状态选择。

## 六条提升行的四条精确投影

设 q=ReLU(g)，alpha 为该门原 bit，且整数语义中 q=alpha*g。a 为任意固定实系数，包括负数和零。实际观察为 z=a*q+v，共同源证书给出

```text
L <= v <= U,
A <= a*g+v <= B,
L <= U, A <= B.
```

考察引入辅助 t 后的六条线性行：

```text
t >= L*alpha,
t <= U*alpha,
t >= v-U*(1-alpha),
t <= v-L*(1-alpha),
A*alpha <= a*q+t,
a*q+t <= B*alpha.
```

前四行为 t=alpha*v 的 McCormick 行；后两行由共同源区间乘以非负原 bit 得到。在 0<=alpha<=1 和 L<=v<=U 下，消去 t 恰好得到

```text
(A-U)*alpha <= a*q <= (B-L)*alpha,
L+(A-L)*alpha <= a*q+v <= U+(B-U)*alpha.       (P)
```

证明：t 的下界为 L*alpha、v-U*(1-alpha)、A*alpha-a*q，上界为 U*alpha、v-L*(1-alpha)、B*alpha-a*q。九个上下界交叉条件中，四个是 (P)，其余由已有 v、alpha 和区间端点界推出。反向取

```text
t = max(L*alpha, v-U*(1-alpha), A*alpha-a*q)
```

就不超过任何上界，给出同时扩展。任何不涉及 t 的原约束可原样保留；多个互不耦合且无额外消费者的 t 可同时消去。

分数 alpha 下，这是六条线性行的精确投影，不是非线性等式 t=alpha*v 的精确投影。整数 alpha 下 t 分别等于 0 或 v，结合原门与有效源界，每个旧整数状态都有扩展。g=0 时 q=0，但 v 可能非零；两种合法原 bit 分别对应 t=0 和 t=v，不能错误地一律取 t=0。

该规则不删除原 q、g 或 bit。如果 t 被其他门、证书或消费者使用，不能直接套用此消元。也不能把它声明为整个源图或完整 RLT 层级的理想凸包。

一般非零系数、q/v 已有独立坐标时，新增系统由一个辅助变量、六行、16 个矩阵非零，变为零辅助变量、四行、10 个非零；若 z=a*q+v 已有坐标，后者为 8 个非零。定义 z 的成本不能免除。界、RHS、规范化 slack、共享身份、证据和终端查询另计。这里相对的是假设的六行提升，不是现有生产路径实测降本。

只提供上界 B、未提供下界 A 时，四个 McCormick 行加 a*q+t<=B*alpha 的投影为两行：a*q<=(B-L)*alpha 和 z<=U+(B-U)*alpha。只留下后一条残差行，不能声称保住这五行的全部 LP 投影强度。

## 构造混合权重的共同源支持

令 x 属于同一盒 D，所有 f_j(x) 仿射，考虑实际前向读出

```text
F(x) = c+d*x+sum_j w_j*ReLU(f_j(x)).
```

本节先假定系数确定。负项使用固定且有效的 ReLU(f)>=f/2，形成仿射上界部分

```text
v(x) = c+d*x+sum_{w_j<0} (w_j/2)*f_j(x).
```

正权重按原 slot 次序相邻配对，奇数剩余项单独成组。令 P=ceil(n_positive/2) 为组数。P>0 时，每组分配一次 v/P，故 F 不超过各组函数之和；常数 c 总共只计一次。

双门组 k=v/P、a,b>0 的支持为

```text
U_group = max(sigma_D(k), sigma_D(k+a*f),
              sigma_D(k+b*h), sigma_D(k+a*f+b*h)).
```

这是逐点恒等式 k+a*ReLU(f)+b*ReLU(h)=max(k,k+a*f,k+b*h,k+a*f+b*h) 的结果。单门组使用 max(sigma_D(k),sigma_D(k+a*f))，不制造新的零门或相位。若 P=0，直接使用 sigma_D(v)。所有 sigma 都在同一完整盒上求值，没有固定原 bits 或求解相位子区域，也不推广成 k 门的指数展开。

定义 U(F) 为各组支持之和，得到可靠上界 F<=U(F)；对 -F 运行完全相同的代数规则，得到下界 -U(-F)。单组支持对所列组函数精确，但负项线性化和组间求和一般不精确。P=0 时只对仿射 majorant 精确，不能称原 F 的精确支持。

与同一分组、同一负项规则的单门对照比较：双门组把 k 等分给两个单门，则

```text
sup_D[k+a*ReLU(f)+b*ReLU(h)]
 <= sup_D[k/2+a*ReLU(f)]+sup_D[k/2+b*ReLU(h)].
```

未配对单门在两侧使用相同的完整单门支持，配对增益记零，不通过给 dummy 分配残差削弱对照。这不宣称胜过任意更优的残差分配、完整多门 hull 或所有已有分析方法。

在 D027 的 f=x+y/4、h=x-y/4、x,y in [-1,1] 上，F=1/10+ReLU(f)+ReLU(h)-x，联合组上界为 11/10；上述同预算单门分配给出 8/5。D028 还提供满足两个完整单门源凸包、却违反联合界的点，因而联合信息不只是相对弱区间的改善。另一个前提 3*ReLU(h)/2-x 的单门支持为 1。它们构造出原跨层正控所需的上界，没有免费 oracle。

## 区间系数必须带显式误差

真实 source packet 包含 BN 导致的有理端点区间，不能把中点当成原模型。以下只是求界证明的参考表达式，不创建替代网络或参考相位。

为每个系数区间取固定中点，得到 F0、w_j0、f_j0。设 rho_j 为 w_j 的半宽，且

```text
epsilon_j = radius(bias_j)
          + sum_i radius(coeff_ji)*max(abs(D.lower_i),abs(D.upper_i)),
M_j = max(0, sigma_D(f_j0)+epsilon_j).
```

则 |f_j-f_j0|<=epsilon_j，ReLU 的 1-Lipschitz 性给出对应输出差界，且 ReLU(f_j)<=M_j。令 epsilon_affine 为 c+d*x 的同类误差界，并设

```text
E = epsilon_affine
  + sum_j (rho_j*M_j + abs(w_j0)*epsilon_j).
```

恒等式 w_j*ReLU(f_j)-w_j0*ReLU(f_j0)=(w_j-w_j0)*ReLU(f_j)+w_j0*(ReLU(f_j)-ReLU(f_j0)) 给出 |F-F0|<=E。因而

```text
-U(-F0)-E <= F <= U(F0)+E.
```

即使权重区间跨零，此证明仍成立；按中点符号分组只作用于数学参考 F0，不用于推断原相位。原谓词、原 bits 和原网络验证全部不变。系数误差有相关性也不破坏外包，但逐项误差可能保守；没有承诺 BN 的误差足够小。

若 paired 与相同 single 对照使用同一个 E，前述非回退不等式保持。区间解析、误差半宽、原 frame 绑定、每次向外算术及有理数位长都要付账。BN 折叠、归一化、bias 和 padding 必须先有正确包络；误差公式不能补救遗漏项或错误 loader。此误差定理不授权把最终谓词中的不确定系数也直接替换成中点；(P) 的系数及其数值落地仍需独立认证。

## 共享源支持的计算代价

对 D=m+[-d,d]，sigma_D(k0+k*x)=k0+k*m+sum_i |k_i|*d_i。缓存同一组基底 k=v/P 的系数、中心值和 sigma_D(k) 后，稀疏增量 a*f 可按

```text
sigma_D(k+a*f) = sigma_D(k)+a*f(m)
  + sum_{i in support(f)} (abs(k_i+a*f_i)-abs(k_i))*d_i
```

求值。双增量在两个源支持的并集上合并，不能把相同 source ID 当成独立项。这样单个观察的支持归约算术量可为 O(n+nnz(v)+sum_positive nnz(f_j))，其中 n 包含逐门遍历和常数工作；另加构造负项 v、误差界、索引及证明的成本。不应逐组复制完整 v，否则可能变为 P*nnz(v)。现有 D015 API 尚未实现这种缓存局部修正。

这些 gather、绝对值与 reduction 具有设备并行形式，但不是 GPU 已实现或速度测量。证明针对精确算术；浮点大数相消时的向外误差、并行 reduction、完整矩阵作用及转置、设备内存和传输均待认证。

对同一真实消费者的很多被观察门，不能把上述单个观察成本免费复用。每个不同的 v 或 a*g+v 都要计算或提供已证明的共享编译。尚未证明全层全门查询具有同样的线性总成本。

## 与已有方法的关系

McCormick、RLT 和 Fourier–Motzkin 消元都是已知方法。[Sharp Hybrid Zonotopes 的第二节和 Theorem 7](https://arxiv.org/html/2503.17483v2)是直接对照；不能把选择性乘积或线性消元本身作为新颖性。固定 ReLU 下界也已有明确先例，参见 [CROWN 第三节](https://papers.nips.cc/paper/7742-efficient-neural-network-robustness-certification-with-general-activation-functions.pdf)，本研究不采用其反向传播或斜率优化。正项配对的联合支持与 [k-ReLU](https://papers.nips.cc/paper_files/paper/2019/file/0a9fdbb17feb6ccb7ec405cfb85222c4-Paper.pdf)属于同类机制，不作全面新颖性声明。

本轮在项目内新明确的是：完整双侧、任意固定 a 的投影与反向扩展，构造混合读出前提的统一规则，以及不替换原模型的区间误差条件。是否能组成有实际优势的新 Neural HZ 变换体系，仍需真实结构、完整代价及强对照实验；不能把这组局部定理宣称为整体目标达成。
