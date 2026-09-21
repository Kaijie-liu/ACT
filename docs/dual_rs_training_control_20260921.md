# Dual RS：原生选择器训练一步与新进程恢复控制

## 执行前冻结 R1

固定源、隔离环境与公开数据沿用[部署记录](dual_rs_author_deployment_20260921.md)。
本阶段没有长训练、fresh Monte Carlo认证或论文性能结论。
[配置](../configs/recent_moe/dual_rs_training_control_r1.json)绑定作者所有跟踪文件、
训练/测试sigma标签、class weights、CIFAR文件、基础权重、环境清单与执行源码。
不修改作者checkout，不改变旧环境，不接触封存的ACT输入98或holdout。

作者batch256实际切成两块128，每块复制两份噪声，再分别更新AdamW。
控制保留batch256、noise vectors2、原生增强和原生`train()`损失/反向路径。
仅通过optimizer post-hook在一次更新后停止；恢复时从同一物化batch的第二块继续。
native train函数、softCE、consistency loss、denoiser不重写。

1. 按原训练集顺序取通过作者`max_radius != 0`标签过滤的前256条。
   该控制选择不是未来全训练的抽样规则，也不是认证效果选择。
2. 第一更新完成，保存model（含BN）、AdamW moments、epoch scheduler、四类RNG、
   物化batch、chunk cursor和完整来源绑定。checkpoint受限tensor-only加载。
3. 同进程继续第二更新；另起新进程重新构造相同对象、恢复上述状态后执行同一第二更新。
4. 保存张量独立审计，要求两端全部状态及非计时指标**逐位一致**，不设浮点容差。
   AdamW step分别为1/2；scheduler在batch内部不能前进；denoiser无更新无梯度。
5. 任一步ERROR/TIMEOUT停止后续阶段，保留partial，不重试、不换输入、不降低batch。

兼容执行约束：Python3.11/Torch2.11/CUDA13；启用deterministic algorithms和
`CUBLAS_WORKSPACE_CONFIG=:4096:8`，cudnn benchmark关闭。作者原代码未强制这些设置，
故明确属于可复现执行配置，不冒称作者原软件栈。CPU2线程、workers0、GPU空闲至少32GiB、
allocator不超过全卡40%，不抢占其他任务。

**reference、resume及保存张量审计共用300秒外层期限**，含来源/环境/输入哈希、
加载、增强、计算与保存。三阶段继承同一专属进程组，超时只终止本次组。
退出后的来源postflight与终态整理另行计时，不藏进训练时间；这不是正式性能benchmark。
`outer_terminal.json`保留未开始/中断/失败状态，不以缺失或迟到文件升级成功。

执行前33项测试通过；新增7项覆盖exact续步、缺失状态/错误绑定/批数据污染、
原子保存/拒绝覆盖/partial保留、29→30epoch学习率恢复、外层异常与嵌套子进程截止。
CPU toy的milestone控制不是在真实模型上跑30epoch。旧论文表格重建不变。

通过真实控制后才冻结90epoch最终checkpoint规则与双阶段认证流程。
作者`--resume`未实现，且原始checkpoint缺scheduler/RNG；本控制的two-chunk schema
不冒充完整长训练恢复器。长训练需另行实现epoch监督、终态审计与执行身份。
本次控制输出在仓库外，原始张量不提交Git。

## R1 实际失败（执行冻结8fa8d4e34）

第一步原生训练及保存完成：loss1.8691837584，gradient L2=56.4901123，
335个参数张量发生更新；256/256输入经过实际增强后不同于原始ToTensor。
但连续执行第二步时检测到非有限梯度，在AdamW更新**之前**停止。
外层执行5.011s，含postflight5.051s；resume/audit均NOT_STARTED。
`after_step1.pt`及错误日志保留，不能把第一步成功当成恢复控制通过。
**长训练和认证的可执行冻结被此门阻止。** 不降低LR、不改样本、不静默改损失。

另行固定[失败诊断](../configs/recent_moe/dual_rs_failed_step_diagnostic_r1.json)：
仅从保存的一步状态重现第二步前向/反向；强制pre-hook禁止任何AdamW更新，
记录logits、概率零值及非有限梯度；对同一组保存logits分别检查softCE与consistency
的输出空间梯度，并用float64作诊断参照。这不是double训练修复，也不是第二次控制尝试。
单独300s目录、错误即停，保留R1；是否修改兼容路径需根据诊断另行定版。
