# 共同源关系生成器的数学实现范围

本组件实现冻结 D052 的静态匹配定理及非点参数扩展。它是面向 Neural HZ 的支撑生成器，不是完成的新域、原网络验证器、native 编译器或 GPU 实现。完整证明与先例边界见 [D052](../d052_static_source_certificates_20260930/THEORY.md)。

## 输入与保留的原语义

复用 D049 的共同 source context 和 D046 的原 value、phase 身份。输入 f、g 是同一可靠原 source 盒上的区间仿射形式，q、p 是原已有 ReLU 输出，alpha、beta 必须分别绑定这两个不同的原输出。相同标签不证明相同身份；区间相同也不证明参数或源可合并。

原连续因子、全部原二元相位及零点合法选择、EQ/LE、frame 和 decoder 全部不改。输入的模型对应关系、原 phase active 方向、参数 enclosure、source 盒及真实 gate 图由调用方认证。形式 token 检查不能代替这些原网络事实。本组件不安装行、不返回 SAFE/ADV，也不删除旧谓词。

区间 affine 构造器拒绝重复 source 条目、错误上下界和非 Fraction 参数；同源系数应由调用方先可靠合并。固定源仍保留在输入形式和 context 中，求界会计入其真实区间系数贡献。仅零系数可从表达式中省略，不删除原因子。直接将本 pair 输出当作自身局部源的声明拒绝；更完整的原拓扑认证仍是调用方义务。

## 统一生成与投影

取每个系数区间中点，仅用于选取 D052 的正反 source 坐标和固定幅值匹配，构造确切仿射 P、N。二者在完整原 source 盒上非负；非对称盒和反向坐标的 bias、尺度补偿均保留。f 或 g 独有的源、区间跨零及固定源不得遗漏。

随后对原区间形式先逐系数减去或加上确切 P、N，计算可靠完整盒支持：

```text
kappa0 >= sup(f-P), kappa1 >= sup(f-N), U >= sup(g-P+N).
```

不是对中点网络求界。每个区间项与 source 区间的四端点乘积均被覆盖；忽略未知参数相关性只会使界保守。原 gate 的整数语义蕴含 q<=P+kappa0*alpha 与 q<=N+kappa1*alpha，因此 F=q+p-P 满足

```text
F <= kappa0*alpha+U*beta+(kappa1-kappa0)*alpha*beta.
```

记 Delta=kappa1-kappa0，生成完整单消费者 McCormick 投影：

```text
Delta>0: F<=kappa1*alpha+U*beta,
         F<=kappa0*alpha+(U+Delta)*beta;
Delta<0: F<=kappa0*alpha+U*beta,
         F<=kappa1*alpha+(U+Delta)*beta-Delta;
Delta=0: F<=kappa0*alpha+U*beta.
```

没有创建或丢弃一个实际共享 overlap，原两个 bits 始终保留。若相位乘积有多个消费者，本规则不能代替其共同投影；也不声明所得两行是整个双门理想 hull。

## 行语义与前向消费

每行是 (value_terms, phase_terms, rhs)，语义为两类项之和 <= rhs。P 的 source 项使用对应 original_value 的同一个 namespace，不制造独立副本。原 bit 的 0/1 到 native ±1 编码和 RHS 转换尚不在本组件内。

返回证书保留 f、g、原 phases、P、N、kappa0、kappa1、U、Delta 与行。若 P=P0+sum P_i*x_i，输出行完整保留 RHS 的 P0 补偿。

仿射/残差可继续使用已得到的关系，D052 的共同相位系数规则给出跨下一 ReLU 的健全消费。本轮测试对一个明确混权后继核对这种组合，但不新增通用前向证书搜索 API，也不声称该 API 已实现任意 Conv、Add、Concat 的原模型绑定。那些算子仍须以同一 frame 上的真实读出和已付费源证书接入。

## 数值和完整成本边界

输入和每步 Fraction 结果限制为 512 位。公开输入组合在构造 union、dict 或行前检查稀疏出现次数不超过 65536；三次移位支持分别检查其实际输入出现次数。某个宽形式可能因证明中间式超限而被保守拒绝，这不构成 UNSAT。所有 public 调用默认关闭，disabled 返回 None 且不读取 task 输入；坏 enabled 类型也拒绝。

匹配与支持为实际共同支持上的线性标量运算，加排序和身份检查；三次支持会创建临时映射，返回的最多两行有独立 tuple 存储，证明中 P/N 与原参数也保留。数学内核没有宣称总存储等于最终行数，也没有获得 whole work 或完整物理峰值资格。

真实 HZ 展开、所有旧谓词、source 证据、终端 A/A^T、slack、并发主机设备存储、原网络见证验证及 decoder 仍全部付费。GPU 需要另证向外算术、归约及完整执行链；本 Fraction 实现不是 GPU 候选。

## 新颖性与资格

本规则仍属于 D052 已明示的 Anderson、条件观察与 RLT 组合；现有 HZ 加相同行逻辑等价，D049 也已能表示输出。贡献定位为统一、可构造且覆盖区间参数的关系生成支撑，而非新的域语义。数学测试通过也不意味着完成定义创新、真实模型净收益或正式分数提升。

全部执行人口、预算、一次性版本和证据要求在 PREREG.md 中冻结。未通过时 fail closed；不改旧失败结果，不减少测试或实源人口。形式原相位绑定检查不等于 actual_phase_column_binding_verified。
