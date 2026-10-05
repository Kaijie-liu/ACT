# 三个真实残差前缀的来源与接入边界

只读核查确认三个目标的拓扑与参数来源已有 D241 记录，但没有可直接转移给 D249 的完整 native 状态资格。本文件记录实际代码证据和来源覆盖定理的适用前提，不把几何覆盖数当作数值拒绝率。

## 固定来源没有改变

权威人口仍为 results/d241_residual_source_binding_20261005_v1/preregistered.json 的 selected_sources：CIFAR100 large 的 row 117、CIFAR100 medium 的 row 30、TinyImageNet medium 的 row 73。模型、性质和哈希沿用原记录；不是按当前结果挑选新目标。三份 source_0/1/2.json 的 summary 均明确 bounds、完整 child affine map、native phase binding 尚未取得资格。

本轮读取完整节点的 shape、stride、padding 和消费者元数据，没有新解码原 ONNX 权重或执行模型前向，没有载入 pickle。source_1/2 的 Relu2 前沿分别为 64*15*15、64*27*27；Conv3 后的父层分别为 128*8*8、128*14*14。

## stride2 残差的来源覆盖证明

Medium/Tiny 都具有 Conv3 的 3*3、stride2、pad1，shortcut Conv8 的 1*1、stride2、pad0，再经 Conv11 的 3*3、stride1、pad1。记 Relu2 空间前沿为 A。

一个内部 child 位置 p 经 shortcut 可以读到 A 的九个粗格点 2*(p+Delta)，Delta 在 {-1,0,1}^2。一个父 Conv3 的感受野为 2*r+{-1,0,1}^2，其中双坐标都为偶数的位置只有 2*r。因此每个父最多接触九粗格点中的一个；两个父最多覆盖两个不同粗格点。

包含 padding 后，n*n child 网格有 4 个角位置、4*(n-2) 个非角边位置、(n-2)^2 个内部位置；潜在 shortcut 粗格点数依次为 4、6、9。Medium 的 n=8 对应 4、24、36 个位置；Tiny 的 n=14 对应 4、48、144 个位置。这些是完整网格的解析几何数，不是已传播后的 crossing 人口。Large 的父 Conv3 为 stride1、shortcut 为 identity，不能套用上述七点论证。

THEORY 中的 Gamma 只能来自完整规范来源合并后仍独立存在的列。稳定 Relu2 折叠、跨位置 alias、路径相消和零系数均可能改变它；原 initializer 非零不证明复合 mean 系数非零。边界零 padding 不能按输入像素继续外延。保留 Relu2 crossing 幅值作为规范列时有机会使用该分组；若 native 基底已经改变，须先证明相同身份/盒合同，不能仅凭端口和几何建立 Gamma。

## 原生门内部正确不等于原网络来源已经认证

生产 tf_mlp.py 的 sparse_hz_apply_relu_exact 保存 L=lb/2、Q=ub/2，并使用 stored(c-Q) 作为出生 EQ 的 RHS。D064 从实际 EQ 恢复的原生输入常数是 exact(stored(c-Q))+exact(Q)，因此相对于出生前读出的常数差为

```text
Delta_birth = exact(stored(c-Q))+exact(Q)-exact(c).
```

提取器给出 graph_error=0 证明的是实际存储 EQ/LE 内部的图关系，不单独证明 Delta_birth=0，也不证明它与原 ONNX 的逐层语义一致。这里没有实际三模型的差额统计，不能据此声称三模型或历史成绩不健全。

Compact 出生的两行 RHS 记为 R1、R2，D064 实际检查 delta=R1+R2+L，记录 error=abs(delta)/2；非零时不能擅自解释成精确 ReLU 图。D249 只接收其已证明支持的情形，这项限制没有被撤销。

solver_hz.py 的 sparse_hz_fast_bounds 仅读 c、abs(Gc)、abs(Gb)，没有读取 EQ/LE。故新增关系被正常终端消费不等于下一 ReLU 的前向界自动变强。原 _sparse_merge_constraints 还会以最长分支为基底合并谓词，出生行号不保证沿 DAG 保持；需要原槽身份及完整行内容的再认证。

这些是普通来源对应和算子消费问题，不是要求为极端浮点特例增加救援。输入盒、Conv/BN 舍入、rebase、shared/signed-shared 读出和 decoder 仍必须连成完整证据链。

## 旧快照与可复用入口

旧 C34 确有 native_state.pickle，不能说历史从未保存完整状态；但它是另一个 Tiny143 性质。其 restore_guard.json 明确失败为 final affine replaced actual source RHS storage，exit.json 中 restore_exit_code=1。它不是当前三目标的合格恢复入口。本轮只读这些文本，不反序列化该工件，也不改写旧失败。

正常 shadow_worker_dtype_v2.py 的 tf.apply 循环以及 verifier.py 在 release_intermediate_hz 前的状态可作为新捕获入口，但旧 worker main 含其他历史配置和写入，不能直接调用后声称本合同已认证。新捕获必须保留完整三来源、全部旧相位、所有消费者、可靠范围及输入映射，而非只取有利窗口。

D241 的 246412868 work 是那次实现的总账，不是所有未来方法固定税；也不授权把这些来源处理免费重用或重置预算。真实阶段仍保持原 256M 总 work、200M 每模型、40M evidence、64M entries、512 位、240 秒、AS16GiB、单线程 CPU0 和双 1GiB 宿主观察门。新的 native/前向候选还须继承完整 4225/225 数学人口与 60 秒门。

## 下一研究判断

不把未经可靠范围认证的普通浮点前向数值直接输入新守卫，不重跑已证明超账的 D243 逐 tap 构造，不以旧快照或小窗口替代完整来源。接下来需要获得完整共同前沿上可支付的条件来源容量及多消费者共享证明，检验本轮覆盖证书是否真正排除当前两父规则；若排除，应修改该数学假设或使用有证更紧来源信息，而非不断更换父对或终端路径。

尚未运行三来源 native 前向、数值适用率普查、shadow、全家族回放或 GPU；本记录没有新增 CERT/ADV。
