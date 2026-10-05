# 相位仿射共同误差闭包及其能力限制

本轮推导出一条可重复的共同误差外包规则：中心与误差预算对原相位都保持一次，不必在这一新层保留相位乘积。独立审查认可下述纸面健全性，但它仍完全可编码回一般 HZ，目前只能称 HZ 兼容的抽象正规形与传播候选，不能称强大的新 Neural-HZ、表达能力突破或已经获得验证收益。

用户再次明确：重点是提出强大的 Neural-HZ。本记录保存未完成的数学工作，不把它升级为 helper 实现任务。当前正控仅击败逐门四行 LP；已有两门关系取得相同界。完整成本也没有优势证明。因此本候选尚未达到实现或能力晋级条件，正式收益为零。

## 候选元素和共同状态

在同一个当前状态 Gamma 中，令活读出满足

```text
y = c0 + G beta + r
sum_k v_k |r_k| <= R(beta) = a0 + a^T beta,   v_k > 0.
```

beta 为原二元相位的 0/1 记法，与原 signed bits 一一对应，不新建替代标签。c0、G、a、v 均为固定系数，不能偷偷让 c0 带连续源，否则后面的新中心会重新含二元连续乘积。R 必须在整个当前允许状态上非负，不只在原网络真实轨迹上成立；a 的单个系数可以为负。

Gamma 保留原连续因子、全部原二元身份、EQ/LE 谓词、共享输入与 decoder。本记录采用保守的具体合同：旧状态及其全部消费者仍保留；live skip 使用旧状态的精确 Sy，不另造独立摘要。这样没有删除旧 source/guard 依赖的隐含步骤，也不声称历史宽度下降。当前前沿作为一个整体递推，所有使用同一物理读出的消费者复用相同残差身份。

## 一次相位中心的前向闭包

令新实际算子为 g = W y + d、q = ReLU(g)，gamma 为这些原门的二元相位；需要继续使用的线性读出为 Sy。选固定结构参考 tau 属于 {0,1}^b，不依实例标签、LP 状态或性质挑选。tau 可以不可达，只用于代数，不能把参考值冒充可达界。

```text
ybar = c0 + G tau
gbar = W ybar + d
center_next = (diag(gamma) gbar, S(c0 + G beta))
delta_q = diag(gamma) [W r + W G(beta-tau)]
delta_s = S r
```

这组 delta 是每个当前状态经过真实新算子的共同见证。对固定正权重 w_i、t_l，定义

```text
kappa = max_k (sum_i w_i |W_ik| + sum_l t_l |S_lk|) / v_k
d_j   = sum_i w_i |(W G)_ij|
Rnext(beta) = kappa R(beta) + sum_j d_j |beta_j-tau_j|.
```

由加权 L1 诱导范数及三角不等式，整个堆叠残差满足

```text
sum_i w_i |delta_qi| + sum_l t_l |delta_sl| <= Rnext(beta).
```

证明中先合并 WG 再取绝对值；误差中偏置 d 已由中心完整承接。由于原 beta、tau 为二元，右侧仍为相位仿射式：

```text
a0_next = kappa a0 + sum_j tau_j d_j
a_j_next = kappa a_j + (1-2 tau_j) d_j.
```

新 gamma 的半径系数为零。center_next 对全部旧 beta 和新 gamma 也为仿射，因此这一正规形可以重复使用。该证明覆盖整个当前外包状态的真实算子像，不是仅重新包住最早一层的精确网络状态。

实际有损变换丢弃新 delta_q 的精确乘积方程，以共同预算替换；保留 delta_s=Sr、原 g=Wy+d、原相位 sign guards、q>=0 和可靠的 q<=U_positive*gamma。可靠门界必须对整个当前状态成立。旧 source、input decoder、全部旧 predicates 不删除。若也保留新 q=diag(gamma)g 的完整精确图，这就重新成为旧 HZ 加辅助约束，不再是此处的替换。

这是 sound outer approximation，不是精确 ReLU 图：新层 active-value 等式已放松。所有真实零点的原合法标签都有见证，但 active-zero 的外包可能多出伪幅值，不能称零纤维精确。新增 ADV 仍必须通过具体原网络检查。原 bits 从未连续化；用于分析的 LP 放松不是域的整数具体化。

## 完整费用尚未显示优势

令新门 m、live readout 数 l、p=m+l、半径相位系数非零数 bR。若 delta 已物理化，L1 epigraph 增加 p 个辅助量、2p+1 条 LE、5p+bR 个系数非零项。其外还要支付残差定义：门部分通常 m 条 EQ、3m nnz；live 部分 l 条 EQ、2l+nnz(SG) nnz。实际零系数可删，这些不是测量值。

物理 g 已有时，原两条 sign guards 加 q>=0、q<=U_positive*gamma 通常为 4m 条 LE、7m nnz；若另保 q>=g，再增 m 条 LE、2m nnz。g 的物理生成、原 bit bounds、旧 predicates、旧 r、源连接与 decoder 全部另计。直接代入 residual 可以省部分列和 EQ，但会增加展开行的系数，不能只报节省的一侧。

原精确四行门也只需四行，因此新预算并没有自动减少行数。WG、SG 可能填充，半径系数随历史相位增长，旧 guards 仍引用旧状态。一次相位次数不等于固定宽度或更低总费用。系数可靠计算、向外舍入、GPU 实现及终端 MILP 的完整成本均未验证；没有 QP、SOCP、支持 oracle 或其他 helper 被允许作为隐藏代价。

## 普通盒源正控和强比较失败

取原输入 x,y 属于 [-1,1]：

```text
eta1=(x+y)/2, eta2=(x-y)/2
g1=1/4+eta1, g2=-1/4+eta2
q_i=ReLU(g_i), s=y/10=(eta1-eta2)/10
J=q1+q2+s.
```

源的精确关系为 |eta1|+|eta2|<=1，因为 |x+y|+|x-y|=2 max(|x|,|y|)。不是把独立盒输入偷偷替换为更小的不确定集合。取输出和 skip 权重为 1，新预算为

```text
|q1-beta1/4| + |q2+beta2/4| + |s| <= 11/10.
```

故包括分数 bits 的终端 LP 也有 J<=1/4+11/10=27/20。真实源 x=y=1 取到等号。下一原门 ReLU(J-3/2) 因而恒为零。

同源、同紧界的旧逐门四行 LP 则允许严格内部输入及以下点：

```text
x=y=9/10, g=(23/20,-1/4), s=9/100
beta=(19/20,1/2), q=(19/16,3/8)
J=661/400 > 3/2.
```

原标量界分别为 [-3/4,5/4]、[-5/4,3/4]；两门的两条上侧行都在此取等号。该点的共同残差范数为 77/50，大于 11/10，因此被新预算排除。这是解析 LP 分离，不是一次执行、CERT 或具体网络 ADV。

但强比较不通过：已有恒等式

```text
ReLU(g1)+ReLU(g2) = max(0,g1,g2,g1+g2)
```

对共同原盒计算 s、g1+s、g2+s、g1+g2+s 的支持，分别得到 1/10、27/20、13/20、11/10，也直接证明 J<=27/20。[已有共同源块研究](../d062_mixed_block_support_20261001/THEORY.md)已覆盖这种构造。完整整数 HZ 也不承认该分数相位假点。因此不能据此宣称比已有方法更强，不能用弱参照的正控启动大规模实现。

## 定义创新的准确边界和后续主线

中心、预算的绝对值辅助变量、guards 和 masks 都是一般 HZ 的 EQ/LE 加原二元因子。这是一个 HZ 兼容的限制正规形和抽象变换，不因换了术语就成为表达力更强的新域。可编码回 HZ 不会自动否定任何有用的域设计，但这里尚缺可证明的结构优势、完整代价优势与真实能力增益，外部新颖性也未建立。

与 [有限阶相位展开](../d121_source_coherent_definition_audit_20261002/PHASE_SERIES.md)相比，本式用一次中心加相位仿射半径替代新高阶乘积；与 [共同替换合同](../d142_joint_reabstraction_20261004/DEFINITION_AND_LIMITS.md)相比，给出一个有线性终端编码的具体健全构造。这是项目内可保留的数学进展，不是已完成的 Neural-HZ 突破。

下一核心问题仍是 Neural-HZ 自身如何在普通混合仿射、ReLU、活旁路及后继非线性中，保留有用的跨神经元和跨层非凸关系，且避免把全部原图或昂贵查询藏回定义。候选要在同源、同预算、包含已有低成本关系的参照下展示实质收益；不是要求有损域在集合精度上超越同信息的精确 HZ，也不是不断添加 helpers 来制造成绩。

本轮不再扩展正控或启动实现。后续实际候选仍须原有预注册、数学测试、真实同结构、shadow、逐家族和全量回放；不改变用户的任何禁令或资源门。smooth、Transformer 和 GPU 是研究目标，不是此 ReLU 推导已覆盖的能力。

## 来源和执行状态

记录日期 2026-10-04 Australia/Sydney。分支 redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；记录前 tracked diff SHA256 为 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。继承 [D145 续接记录](../../RESUME_RESEARCH_20261004_D145.md)与[定义优先目标](../../GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md)。依赖仅为列明的冻结文档及纸面推导；无可执行候选、运行配置或新依赖安装。

本轮未执行模型、数值候选、测试、求解器、GPU、shadow 或 replay，未启动后台作业。新写入仅本隔离目录及新续接记录。生产路径、既有修改、所有历史模型和冻结证据保持原样。正式 1870/2413 与外部 CIFAR100 25、TinyImageNet 36，共 61/400 均未更新；本候选正式与外部净收益均为零。Goal 保持 active，未完成。
