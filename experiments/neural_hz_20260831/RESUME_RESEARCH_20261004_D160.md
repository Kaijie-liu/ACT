# Neural-HZ 源关系定义研究续接

重点保持从数学定义研发强大的非凸 Neural-HZ，不是旧图加 helper。D159 与本轮 D160 均归类 progress，但没有实测能力新增。最新[定义与证明](definition_first_20260928/d160_source_subspace_fiber_20261004/THEORY.md)和[完整研究记录](definition_first_20260928/d160_source_subspace_fiber_20261004/RESEARCH_RECORD.md)均为 paper_only。

本轮候选令 d=A(x-xref)+e、S=range A，e 是同一旧状态的共享 affine readout。新 proxy 满足 dhat-e in S、Cdhat=Cd、||dhat||²<=E；保存 r=C(D_beta-I/2)dhat，读出 Cq=.5Cg+C(D_beta-I/2)gbar+r。这个 canonical 形式保留下一层的原 source 系数；全部旧关系与消费者同步代入 a=r+Cd/2，不另设独立 a。

健全性用 dhat=d，覆盖整个当前 native 父域、全部原 bits 与零标签。新代理恰为 d+W，W=S intersect ker C；比旧 ambient 代理更紧。||Mr||²<=.25||MC||²E 可用于规定的 Cauchy 递归，但所有源/相位/历史bank项均需收费，不保证总预算收缩。中心化本身早已在 D147/D148 出现；不能再算新原理。

固定 h in S-perp 给出 phase-affine 的 2L<=E+||v_beta-h||²+2hᵀe；J_tau=[CA;CD_tau A] 的完整左核给出两侧固定 phase-flip 行。两类均不用新乘积或新求解器。平移后的 h_tau 行即使在参考相位也未必更强，故固定保留 h=0。原生 translated Gram 本身仍含 phase-parent 乘积，线性 lowering 只是外包；若丢 native 却继续用其范数给下一层预算，就缺少覆盖整个有限父多面体的健全性依据。没有解决廉价完整查询。

四门三源控制证明新域仍有损、且自动左核规则排除 D157 伪点。但对应后继性质已被旧逐门 epigraph q>=g、q>=0 全盒证明。成本也为46行，对比D157的32、旧图16，未有优势。因此不实施此版，不扩展 helper 正控，不把单点细化当新验证能力。三门注入控制和未认证 B4=3/4 草稿均已记录，勿重新包装成新进展。

下一研究动作应针对完整普通网络结构的可消费关系与完整费用，允许更换定义，不要求继续本候选。旧精确 HZ 不可被真实外包在集合精度上超过；有意义比较是同精度更便宜，或同资源预算下新增实测能力。完整接口身份、真实系数误差和 native/terminal 边界不能省略。不要再反复做相同 kernel/能量小控制，也不要把 GPU 计算给定点冒充集合验证。

本轮无候选执行、模型/GPU作业、shadow、replay或生产集成。最近4032项/212文件资格仍只属于D158。将来执行须新预注册、源码冻结、一次版本和完整继承门；没有待保存后台作业。

正式1870/2413=1063 CERT+807 validated ADV，独立E0 CIFAR25/Tiny36=61/400；新增均0。13家族旧解、GPU、smooth/Transformer、新家族与满分目标不变，goal active、未完成、未阻塞。分支redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。旧文件只读，新档案哈希见该轮 ANCHOR_SHA256SUMS 与 ARCHIVE_SHA256SUMS。
