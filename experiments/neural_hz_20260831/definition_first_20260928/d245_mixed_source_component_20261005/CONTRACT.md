# 共同非凸状态的混合源相位组件

本默认关闭数学候选实现D244的统一mixed容量，不是新增验证helper。原H由冻结D243的Source拥有，保留完整仿射/Conv/BN外包图、原signed bits、EQ/LE、共享frame、全部消费者与原输入decoder。不重新声明四个独立门，不调用D243的X attach或任何Bank、支持搜索、LP/dual、攻击、BaB、backward或phase split。

复用D243不可变图及精确算术服务不转移原模型或GPU资格。当前仅是声明图数学；D244 THEORY.md中的定理、整数扩展证明及先例边界完整适用。旧源码和证据只读。

## 确定性的完整源提取

入口要求同一个已封存owned H、匹配frame、四个不同原门句柄及enabled=True。两父必须crossing，沿用原位；s_i=max(-L_i,U_i)，x_i=f_i/s_i、q_i=Q_i/s_i，不平移零点或pivot位。

沿完整定义DAG检查父来源不能依赖选中父输出，不能只看相消后的系数。父f与child f展开到共同输入、误差、原位及其他ReLU输出边界；child在选中q处停止，不在父f处停止。不同stop策略的memo分开。所有偏置、skip、其他幅值、BN误差和列身份保留，只作已有literal alias，不按数值相等合并来源。

取完整z=(g1+g2)/2、d=(g1-g2)/2；b_i为z的归一化q系数，A/B为d的系数。固定tau=(A-B)/2>0，e=d-tau(q1-q2)完整保留，不在失败后改tau、方向、anchor或方法。

将x_i=k_i+w_i*u、扣q后的z=c+v*u，以原可靠半径R_j=(U_j-L_j)/2计算G_ij=sum(w_i[j]w_j[j]R_j^2)、h_i=sum(w_i[j]v[j]R_j^2)，精确解二乘二G*a=h。det须严格正，否则拒绝。完整r=(c+v*u)-a1*(k1+w1*u)-a2*(k2+w2*u)，常数及余项不得丢失；r/e界使用整个H的可靠列界。盒是证明支撑，不替换H。

Gram不是最优L1残差，也未证明任意基变换不变；系数不确定性须有同源carrier。它是固定EQ代数，不读取property方向或求解状态。来源展开、Conv连接及重复消费全部收费，不声称解决已知大模型规模障碍。

## 唯一混合容量及扩展

对z=a1*x1+a2*x2+b1*q1+b2*q2+r，用唯一解析判据：

~~~text
L10=min(0,a1+b1)-max(0,a2), U10=max(0,a1+b1)-min(0,a2),
L01=-max(0,a1)+min(0,a2+b2), U01=-min(0,a1)+max(0,a2+b2),
[Lr+min(L10,L01),Ur+max(U10,U01)] subset [-tau,tau].
~~~

闭符号区间保留零点全部合法标签，不执行相位子问题。只替换h=alpha1-alpha2乘clip的共同差，不逐项误换alpha_i*clip。

六连续观察按(c12,c21,d12,d21,v1,v2)排列，分别等于(alpha1*x2,alpha2*x1,alpha1*q2,alpha2*q1,alpha1*r,alpha2*r)。每个产品登记完整四行MC。K=tau*h+a1(q1-c21)+a2(c12-q2)+b1(q1-d21)+b2(d12-q2)+v1-v2。登记D244两defect与三X容量，共29新增LE，无新相位。

这些产品引用同一H，不是各自私有源。原位整数时MC确定唯一真实产品；D244证明保证每个原H点可扩展，反向保留旧谓词，原整数投影不变。分数点用于松弛比较，不是ADV。

Relation只读持有parent H、source_forms、Gram/det、a/b、完整r/e界、mixed_bounds及实际新增行，提供canonical_extension、satisfied、decode、materialize。decode读取原输入。普通只读/结构拒绝保留已封存H、不退款；实际预算失败sticky，内存或非法值fail closed，不发布部分关系。多个Relation尚无全网辅助列合并资格，不凭frame相同拼接。

## 物理证书与完整费用

主控制是A in [0,1]^4、t in [-1,1]，f=(A1-A2,A3-A4)，子mean=(4/5)(A1-A2+A3-A4)+(3/50)(q1+q2)+t/10，子difference=q1-q2。shortcut及实际F用A，不插入f直边。实际存储LE的非负组合加原f/F EQ必须给F<=258/125，真点取等。u=199/200的2587/1250点须满足完整强参照及旧X/Q所有合格tau解析证书，不用solver搜索。

关系增量6列29LE96nnz，显式产品界另12LE12nnz，终端完整增量41LE108nnz。原H、EQ展开、Gram、r/e界、重复消费、存储共存、decoder和终端均另计。普通正例共用一个原Budget：work256M、单公共操作200M、累计逻辑entries64M、512位。这些不是完整物理峰值。

完整数学回放通过仅授予声明组件资格。actual_model_binding_qualified、native_HZ_admitted、complete_physical_qualification、gpu_computation_completed、new_domain_qualified、new_capability_qualified保持false。真实三个完整模型、shadow、13家族、2413/独立400及四并发门未通过；GPU、smooth、Transformer及新家族目标不缩减。正式1870/2413和独立61/400不变，新增记账0。
