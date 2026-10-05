# 续接从强参照反例修改 Neural-HZ 定义

最新 [D151 理论](definition_first_20260928/d151_exact_comparator_audit_20261004/THEORY.md) 已改变下一行动：不要按 D150 续接文件直接把逐门范数替换移植到完整 CNN。相同父状态、相同 e/q/phase、相同范数预算下，标准四行精确 ReLU 比 D150 的五行关系更强，每个 crossing 门少一行，至少节省 g 的旧变量系数非零数那么多个系数；RHS 也不增加。它是比较对象，不是要把全精确图换个名字作为新域。

另有普通非平行两门混权残差反例：g1=x-y/4+1/4、g2=x+y/4+1/4。D150 在同源 x=y=0 允许 q1=q2=3/4，真实为1/4；继承到下一 ReLU 后可产生1/8的虚假输出，而真实全盒恒零。所有差分边都不能排除此共同幅度误差。先修改定义假设，不继续加已知边关系或堆数学小测试。

D149/D150 已完成源码和结果只读保留，作为支撑资产；本轮未执行新候选。强参照只证明局部语义与表示费用，不是实测运行时间或独立多层支配。不得将其扩大成所有范数域不可能有效。

下一定义必须在普通神经块中同时保留共同幅度和相对关系，提出可计费的实质状态/关系替换。原源、全部原 bits、EQ/LE、共同 latent 和 decoder 不变；按消费者压缩须包含全部活 skip，identity skip 不允许随意删除。当前没有已通过强参照的新定义，完整目标仍然是强大非凸 Neural-HZ、GPU及各家族真实能力提升，不是证明组件兼容。

[真实来源核查](definition_first_20260928/d151_exact_comparator_audit_20261004/RESEARCH_RECORD.md) 明确 large 跨 Add8 到 Relu11 和 medium 跨 Add10 到 Relu13 所缺参数；原 ONNX 存在，但需新预注册绑定。Add8/10 的输出还进入更后 Add14/16，不可丢弃。不能只凭 graph 节点名声称已有完整模型参数。

本轮是纸面研究 progress，无数值重跑、模型、solver或GPU执行；无后台任务。正式1870/2413、独立61/400均不变，formal_gain=0，goal active。分支redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked diff 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5 不变。新档案均在隔离实验树，旧档案不改。
