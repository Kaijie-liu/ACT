# 共同负项在全盒同号时的精确支持化简

本页只证明D070已定义支持的计算等价，不提出新域或新强度。保持全部原HZ连续因子、原二元相位、EQ/LE、共同frame和decoder；名义聚合H不是一个新增网络门，也没有新相位。

令D为完整原源盒，H=b+sum a_l*z_l。其精确盒界为L=b+sum min(a_l*l_l,a_l*u_l)、U=b+sum max(a_l*l_l,a_l*u_l)。原锚贡献移除后Hrest=H+Delta；先按同一个源身份合并Delta，改变坐标集合为J。精确增量为

```text
Lrest=L+Delta_bias+sum_J [min((a_l+Delta_l)*l_l,(a_l+Delta_l)*u_l)
                         -min(a_l*l_l,a_l*u_l)]
Urest=U+Delta_bias+sum_J [max((a_l+Delta_l)*l_l,(a_l+Delta_l)*u_l)
                         -max(a_l*l_l,a_l*u_l)].
```

证明是把全盒支持中未改变的项逐项抵消；包含系数翻号、变零、重复源抵消、固定和未使用坐标。输入仍为完整D，未使用源不从身份人口删除。此公式不能从另一个更紧相关域界或相位条件界开始，也不能用interval(H)-interval(anchor)冒充精确结果。

对任一同源仿射A及R(t)=max(0,t)：Urest<=0时，R(Hrest)恒为0，所以sup_D(A-R(Hrest))=sup_D(A)；Lrest>=0时，R(Hrest)=Hrest，所以支持等于sup_D(A-Hrest)。Lrest=Urest=0时两式相同，单列zero避免重复分类。只有严格跨零时，该证明不能把它化为单一平面，保留原D070支持问题。

在D070固定锚i,j和方向s中，Hrest=sum_{k not i,j} max(-s*wbar_k,0)*gbar_k，不依赖四个锚状态。每个表方向只需一次分类，四槽共享其结论。对-F要重新使用方向s=-1，不能把上方向Hrest取负。

这里的gbar和wbar是原D066证明使用的区间中点，不是新的参考网络激活或实际模型参数。实际原bits、原参数误差E、原生图误差及receiver残差照旧处理，聚合的符号不能用来约束原门相位。分类不涉及LP状态、margin、求解器、输入或相位split。

一个重要比较边界：单侧时该支持等于min(sup_D A,sup_D(A-Hrest))，但未必等于原D066为某槽固定选的那一平面。因此单侧仍可能改善旧表；crossing也仅是需要进一步非线性支持的潜在人口，不保证strict gain。诊断不计算A或支持值，所以不报告任何精度改善。

每个consumer/sign建立合并H及盒界的算术为O(S+d)，每pair增量为O(|J|)，固定不交原对的总更新按真实源支持出现数计费。区间解析、规范化、512位运算检查、来源认证、保留对象、编码和保存均另外计费。线性次数不是已通过256M/200M/40M或实际速度门的证据。

本定理经两路独立纸面复核。完整人口、执行范围与停机条件由同目录PREREG规定；本文没有数值实验结论。2026-10-01，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。
