# 从原生谓词认证共同源观察

本候选把已测试的四平面参考界连接到实际SparseHZono的原EQ/LE、原signed相位和真实接收读出。要证明的是追加行不改变原native整数具体化，而不是用形式token宣称原模型端口已经认证。它是Neural-HZ关系变换的接入组件，不是新的域定义或经典ReLU编码的新颖性主张。

## 域语义与参考观察

原状态H的输出为c+Gc*xi+Gb*b，其中xi在[-1,1]、b在{-1,+1}，并满足全部原EQ/LE。候选保留H、其原frame、全部连续量及bits、输出和输入decoder，仅追加由这些原谓词推出的LE观察。空观察嵌入原HZ；有效观察不改变整数具体化。保留旧行使LP外包不扩大，但不证明求解时间或全量旧解已经保住。

共同源z_j不是新增独立噪声，而是同一个(xi,b)上的实际仿射读出。其声明盒必须包含该仿射读出在完整latent立方体上的可靠范围。用立方体计算一个证书并未将原二元变量的域改为连续；原HZ仍完整保留。声明区间参数仅用于构造中点参考表达式，真实读出的完整残差另行认证。

## 从实际存储行提取门关系

所有浮点存储值先解释为精确二进有理数。不通过重复同一浮点运算或容差来确认实数恒等式。令Q>0、K<0，原输出q=Q*(1-eta)，所有源与原slot来自同一HZ。

普通两连续量编码的实际行是：

```text
K*s-Q*eta+K*b-P=rho
-s-b<=0
-eta+b<=0.
```

由这些行和原因子域，定义g_native=P+Q+rho，精确有q=ReLU(g_native)。b=-1时s=1且g_native=q>=0；b=+1时eta=1且q=0、g_native=K*(s+1)<=0。这个证明保留两种合法零点相位，不执行相位搜索。

如果原前端曾用浮点rho=fl(c-Q)，则g_native与P+c不必完全相等。即使c=float(0.1)、Q=1/2这种普通数值也可能有差异。本候选不改写旧谓词或据此判定历史网络验证有误，而是认证实际存储集合上的关系；原模型语义需另行证明。

compact编码的实际三行是：

```text
-eta+b<=0
P+Q*eta<=R1
-P-Q*eta+K*b<=R2.
```

必须精确核对两条后侧行除原bit项外的系数互为反号。令gL=P+Q-R1，Delta=R1+R2+K，g_native=gL+Delta/2，epsilon_graph=abs(Delta)/2。则对每个原可行整数赋值都有

```text
abs(q-ReLU(g_native)) <= epsilon_graph.
```

当b=-1时gL<=q<=gL+Delta，且q>=0；可行时Delta>=0，所以中点误差成立。当b=+1时q=0、gL<=0，同样成立。Delta<0时原active切片本已不可行，inactive中点<=0；统一abs(Delta)/2仍保守有效。这里没有删除bit、修复旧模型或声称它精确对应原模型；新增观察只是旧集合的推论。

两类模板由实际原行结构触发，不根据实例、公开标签、margin或求解状态选择。原上游binary项完整保留在P中。错误guard、不同slot、系数不成对、缺失行、非规范数据或资源失败一律拒绝绑定，不降级成成功。

## 从参考表达式到真实接收读出

令四平面参考式为F_ref=v_mid(z)+sum_i w_mid_i*ReLU(g_mid_i(z))。D063的upper-error与lower+error恰是该中点式的参考上、下界U_ref和L_ref。本候选只使用这两个数学量；其参考token和参考门并不替换原门，D063原返回行也不直接安装。

所有参考仿射式通过z_j的原latent读出展开。对仿射残差e=e0+ec*xi+eb*b，完整立方体给出abs(e)<=abs(e0)+sum abs(ec)+sum abs(eb)。据此认证：

```text
d_i >= abs(g_native_i-g_mid_i(z))
e_i = epsilon_graph_i+d_i
e_F >= abs(F_native-v_mid(z)-sum_i w_mid_i*q_i)
E = e_F+sum_i abs(w_mid_i)*e_i.
```

ReLU的1-Lipschitz性和三角不等式推出L_ref-E<=F_native<=U_ref+E。e_F必须覆盖实际receiver的完整bias、连续列、二元列、权重差和遗漏项，不能仅比较几个参数。此完整native残差账替代D063的区间参考误差；不是忽略实际参数误差，也不依赖未经认证的“参考中点就是原网络”假设。

## 安全安装与代价

实际receiver是H的第j个输出，其原读出为c_j+Gc_j*xi+Gb_j*b。只追加原Gc_j/Gb_j及其负行，RHS分别为向正无穷舍入的U-c_j和c_j-L。系数精确复制存储float64值；负号对有限float64精确。舍入后以Fraction反检RHS未向内偏移，非有限结果拒绝。不能先浮点移项后只对舍入后的差作检查。

新SparseHZono保持原EQ、全部原LE前缀、输出、原因子宽度、frame和exact标志；原exact=False不能被新观察变成True。新增行是原整数集合的推论，保留旧行的LP可行域只会缩小或不变。此处exact标志保留不等于新增原模型端口正确性证书。不得调用无条件exact=False的一般intersect_bounds，再未经证明改回标志。

若实际receiver支持为s_f，新增LE系数最多2*s_f，另有两个RHS；没有新连续或二元因子。源展开、全部选定graph行、参考输入、原receiver及残差中间支持均计费，合并前按出现数受65536限额；每个有理输入和算术结果受512位限额。按实际索引收集原行也有成本。原HZ、证据、临时SparseHZono/CSR拼接、完整列和decoder均继续计入将来的完整物理账单，不能仅以两行成本宣称全系统过门。

本CPU有理参考为未来可靠GPU归约提供语义对照，不证明已经GPU加速。它也没有自动把界安装到下一ReLU；上一轮BINDING中显式后继行的义务仍在。

## 研究定位和当前范围

原HZ加同样有效行具有相同逻辑强度。本轮贡献是从实际存储谓词到共同源观察的可审计语义桥，不是完成PLDI级域创新。数学资格使用真实native算子生成的定向SparseHZono，不读取真实模型、不进行LP/MILP、GPU、source census、shadow或正式回放。必须在后续真实统一结构接入中证明原端口、稳定项、共享别名和全消费者对应，不能把fixture当真实网络收益。

2026-10-01，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。原D063及全部历史证据只读。正式1870/2413、独立E0 61/400均不变；本候选默认关闭，formal_gain=0。
