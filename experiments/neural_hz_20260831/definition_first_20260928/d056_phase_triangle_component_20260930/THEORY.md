# 共享原相位的静态三角合成

本组件为原非凸 HZ 增加有证明的关系，不替换其具体化语义。经典布尔二次三角行来自已有组合优化；文献范围见只读 [D054 研究](../d054_shared_phase_cycles_20260930/RESEARCH.md)。这里实现的是从已认证共同源门关系直接生成物理 LE 行的组件，不宣称其本身已构成创新 Neural HZ。

## 原域与健全性范围

原域元素继续保留连续 latent、原二元 bits、共享 frame、EQ 和 LE 谓词及输出和输入 decoder。每个原门满足 qi=ReLU(fi)，原 active bit 为 bi；零预激活的全部合法 bits 保留。fi 是同一个原 source box 上的可靠区间仿射 enclosure，实际网络参数只需被包含，不能拿中点网络替代。所有外部真实模型绑定仍是独立证明责任。

对每个固定方向边 ij，D053 给出如下整数语义后果：

```text
Lij = qi+qj-Pij(source)
Lij <= Aij(b)+dij*bi*bj
Aij(b) = k0ij*bi+Uij*bj
```

Pij 包含常数和共同原 source 坐标。它不是新独立噪声。同一三门的三个输出和 bits 必须分别不同；只检查标签相同不够。若 source 包含任一目标门输出则拒绝，避免循环宣称同一输入前沿。本组件按原相位 ordinal 排序并固定 ij 方向；不假定非点参数证书正反向相同，也不按效果选方向。

记 thetaij=bi*bj 仅用于证明，不创建运行时 theta。所有原 bits 满足：唯一负边 uv、余点 w 时 theta_uw+theta_wv-theta_uv<=bw；三个负边时 -theta12-theta13-theta23<=1-b1-b2-b3。第一式在 bw 为 0/1 时分别化为非正乘积和 1 减一个非负乘积。第二式的差为 1-k+choose(k,2)，k 为三个 bits 的和，在 k=0,1,2,3 时分别为 1,0,0,1。这是有限数学证明，不是验证器运行时 split。

## 固定代数投影

设所有 dij 非零且负边数奇数，tau=min(abs(dij))。上式给出带符号三角 T(b)：唯一负边时 T=bw，三负时 T=1-sum(b)。将每条边绝对值减 tau 得到非负剩余量 rij：

```text
sum Lij <= sum Aij + tau*T(b)
            + sum_positive rij*min(bi,bj)
            + sum_negative min(0,rij*(1-bi-bj)).
```

每个边乘积的剩余项由其普通 McCormick 上/下界包住；负边方向必须使用 lower product bound，再乘负系数。负边第二仿射选项的常数为正 rij，不能遗漏。至少一个剩余量零，最多两个二选一项。全部展开最多四条仿射 LE，并全部保留，等价于上述凹分段仿射上界。一般不等幅时这只是健全后果，不宣称联合凸包或完整加权投影。

实际行左侧为 2*sum(qi)-sum(Pij 的线性 source 部分)-sum(Aij 和三角平面的线性 bit 部分)，右侧补回所有 Pij 常数与三角平面常数。source 使用原 value namespace，与原 latent 展开共享身份；不发明独立列。保留所有旧关系以及任何别处消费的辅助量，不利用本局部规则删除它们。

当 dij 为零或负边数偶数时，此纯上侧乘积图是森林或 balanced 三角；本候选不增加三角行。这不排除 source/guard/异号消费者等其他机制的收益。任一门可靠界 lo>0 或 hi<0 时也不生成本候选行；原 phase 不删除或 pivot。触零不能筛除。已固定相位下的 LP 冗余证明见 [D055](../d055_cycle_applicability_audit_20260930/PLAN.md)；仅获得语义稳定性还不能假设原 native LP 已安装对应固定事实。

## 实现代价和边界

当前一个调用处理一个完整三元组，重建三条 D053 pair，而不是接受可能被外部改造的证书结构。每条输入 form 出现两次，输入稀疏发生数在构造前统一预检；所有 Fraction 算术仍用继承的 512 位检查。输出最多四行，按实际总行支持预检。组件不修改原 frame registry，返回记录保留输入、bounds 和 pair 证据引用。

令 S 为三门共同来源支持并集。当前界、匹配和稀疏合并约 O(S log S) 的精确有理操作，另受整数位长影响；新增行最坏 O(4S) nnz，证书与中间支持也占存储。若在 m 个未定门上调用所有三元组，当前实现会重复构造 pair，实际约 O(choose(m,3)*S log S)，绝不能假装已有边缓存。完整边缓存需要另证生命周期与证据计费，目前没有实现或计入收益。

真实 source census 必须先完整处理原三个模型、全部 admitted 分支、固定窗口、通道及 canonical slots 的普通界，再统计 C(n,3)、严格稳定省略数和其余包括触零的全部 C(m,3)。每组三角的无效果、失败和所有输出行均需记录。旧 Large 0 / Medium 80 不是白名单，更不能用来跳过 Tiny。D047 原 seed/clamp 工作可被新结构算法避免，但不能把未调用旧算法当作已经支付了新完整算法成本。

旧 ledger 不接受 typed dataclass、MappingProxy 与 phase token。新真实研究必须对 plain 证据及 typed 窗口暂存分别计费，不能把后者算零。原网络、源参数、主机/设备共存、所有证据、完整终端转换、求解与 decoder 成本仍在。

形成证书后的 gather、标量 min、固定平面组合和稀疏合并适合研究 GPU 批处理；源端匹配、四端点乘积支持及归约也可批处理。但 GPU 外向舍入、可靠归约、传输和完整物理峰值尚未证明或实现，本轮无 GPU 加速声明。第三 Conv/残差后继的实际参数解码、native phase/frame/decoder 认证与真正的后继利用仍缺。

本组件保持原精确网络集合并增加整数语义后果，可能加强终端连续松弛；没有精确简化域定义或删除旧信息。原 HZ 加相同行有相同逻辑强度。要成为整体目标要求的定义创新，仍须产生可组合关系语言及非平凡表示/算子定理，并证明完整查询净收益，不能用这张组件通过证书替代。
