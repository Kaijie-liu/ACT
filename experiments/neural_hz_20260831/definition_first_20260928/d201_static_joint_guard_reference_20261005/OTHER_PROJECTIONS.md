# 跨守卫相位量和事件容量的投影对照

本文件记录本轮另两项纸面尝试。它们均是已有乘积提升和线性消元的实例，作为强参考保留；没有实现或执行，也不作新域声明。

## 跨守卫的共同相位量

设 g_i、g_j 的可靠 crossing 界分别为 [-A_i,B_i]、[-A_j,B_j]，四常数严格为正。原 bits 为 alpha、beta；h_i 表示 beta*g_i 的查询矩，h_j 表示 alpha*g_j 的查询矩，p 表示 alpha*beta。

在 p 的四条 McCormick 行之外，加入

```text
-A_i*(beta-p) <= h_i <= B_i*p,
-A_i*(1-alpha-beta+p) <= g_i-h_i <= B_i*(alpha-p),
-A_j*(alpha-p) <= h_j <= B_j*p,
-A_j*(1-alpha-beta+p) <= g_j-h_j <= B_j*(beta-p).
```

这是对同一原守卫分别乘另一个 bit 及其补 literal 的线性化，不运行条件子问题。所有原图点规范扩展 p=alpha*beta 都满足；矩、父谓词及源绑定不能省略。

当 p 没有其他消费者时，令

```text
Lower = {
  0, alpha+beta-1,
  h_i/B_i, alpha+beta-1+(h_i-g_i)/A_i,
  h_j/B_j, alpha+beta-1+(h_j-g_j)/A_j
},
Upper = {
  alpha, beta,
  beta+h_i/A_i, alpha+(h_i-g_i)/B_i,
  alpha+h_j/A_j, beta+(h_j-g_j)/B_j
}.
```

存在 p 的充要条件为每个 Lower 元素不超过每个 Upper 元素，即 36 条行。反向取 p=max Lower 即可重构。若原 BQP、其他事件或消费者也使用 p，必须把其所有界一起纳入，不能只消本地 12 行后删掉 p。稳定门和零界不能套这里的除法公式。

按 g_i、g_j、h_i、h_j 都已有坐标且不计 RHS，保留 p 是 1 个连续量、12 LE、36 nnz；投影是 0 个连续量、36 LE，按支持并集计的安全上界为 116 nnz，系数抵消后可能更少。源展开、h 定义或 g 的物化另付。因此省一个量不等于更小完整系统。

这与 D035 的共同相位区间消元同属 Fourier–Motzkin 机制，不值得仅为改名重复实现。

## 固定合取事件的输入容量行

设 x_j in [0,1]、w_j>=0，结构固定事件 E 上能认证 sum_j w_j*x_j>=b。令 lambda_E 是事件质量，则共同输入容量给

```text
b*lambda_E <= sum_j w_j*min(x_j,lambda_E).
```

对任一静态源子集 J，记 C=b-sum_(j not in J)w_j，得到

```text
C*lambda_E <= sum_(j in J)w_j*x_j.
```

若 E 是 k 个原 bit literal 的合取，C>0，利用 lambda_E>=sum_i literal_i-k+1，可以进一步消去 lambda_E：

```text
C*(sum_i literal_i-k+1) <= sum_(j in J)w_j*x_j.
```

每个固定选择一条 LE、无新连续量，约 k+|J| 个非零项。C<=0 时不能直接用上述下界替代 lambda_E。所有结构前提、共同源绑定及消费仍须保留；完整选择族可能很大。

D014 的源屏障/mixing-set、D191 的 AND 事件和 D195 的条件守卫已有相关先例。本轮公式澄清了一条廉价静态比较方式，不把它作为 Neural-HZ 原创成果，也不从查询点反馈挑选 J。

## RLT 层级口径

对已有线性系统 A*v+B*beta<=d，令 T_i 表示 beta_i*v、S_i 表示 beta_i*beta。将原行分别乘 beta_i 和其补 literal 后得到

```text
A*T_i+B*S_i <= d*beta_i,
A*(v-T_i)+B*(beta-S_i) <= d*(1-beta_i).
```

EQ、源界、S 的对称性及 S_ii=beta_i 同样保留。这是一次 literal 乘法、总单项式次数二；不能因次数二而含糊称为 SA 第二层。m 个 bits、n 个连续量、r 条原行的全量形式有 O(m*n+m*m) 个辅助量和 O(m*r) 条行。再乘两个 literals 会引入 v*beta_i*beta_j 等更高阶量，不享有同一费用。

这些式子足以解释为何上一轮正控并不要求完整 2^k 相位表，却不证明固定低阶松弛能解决完整网络。它们是纯静态代数参考，不授权额外优化器、LP 状态修复、相位搜索或新的正式记分。
