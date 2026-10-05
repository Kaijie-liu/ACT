# 下一真实结构试验的输入与范围

本轮只读核查确认，可复用的两个CIFAR包覆盖完整空间的首两次ReLU及活skip，但不是全模型参数包。当前不执行新的来源试验，也不把数学组件的通过转成真实网络资格。

[D025 large工件](../../results/d025_interval_capacity_20260930_v1/complete_0.json)与[D015 medium部分工件第1项](../../results/d015_source_shielding_20260928_v2/partial_source_evidence.json)都包含d015_raw_first_bank_v1 packet，外层box保留原输入盒。两者输入形状1x3x32x32、pre_ops为空，pre_affine是恒等。packet含完整Conv0和BN1；large含Conv3的36864权重、BN4和Relu5；medium含Conv3的73728权重、BN4和Relu5，另含Conv8的8192权重与BN9 shortcut。

因此可在新的预注册中覆盖large全部Relu5及65536维identity skip、medium全部Relu5及8192维投影skip，不必将旧五窗口当作完整源。完整空间、原source身份、全部原bits、全部通道和实际BN区间误差必须共同计入，不能独立复制patch。

两个packet的graph虽列出Conv6/BN7以及Add后的Conv9或Conv11等节点，但这些后续算子的数值参数没有同等完整保存。若试验要求穿过残差Add及之后真正ReLU，必须新绑定缺失参数；仅有节点名称不够。Tiny仍缺同等完整参数包。旧失败或部分运行的数据可以只读使用，不代表原候选资格通过。

当前小型稠密API不能直接承载该前沿：原source为3072，large首两层原门总数131072，medium为22592。仅large第一层dense source map已有65536x3072=201326592槽。下一可扩展实现必须保同一全局latent关系，并计算全部传播、相位、谓词、范数、输入重构及终端费用；提高小测试上限或拆成独立局部域不解决问题。

旧CNN来源门仍为240秒、global256M工作量、per-model200M工作量、evidence40M工作量、retained64M数值条目、512-bit、CPU1/thread1、AS16GiB及原worker内存观察。不同模板的计费范围不能任意互换；这些资源要求尚未被用户修改。

GPU历史需准确续接：D017首次初始化失败；[D020](../../results/d020_gpu_init_trace_20260930_v1/exit.json)跟踪在ptrace权限处失败，未得到CUDA trace；[D043](../../results/d043_cuda_error_log_20260930_v1/exit.json)的唯一一次后继CUDA可用性调用仍失败，[driver日志](../../results/d043_cuda_error_log_20260930_v1/worker.stderr.log)明确为cuInit返回CUDA_ERROR_OUT_OF_MEMORY。这没有定位资源根因，不能把AS上限当作已证原因，也不能声称GPU已经恢复。本轮未启动CUDA或重跑旧诊断；其余只读快照不是GPU计算资格。

下一来源试验应检验完整普通前缀及共同消费者的精度和费用，不能只测一个norm数值，也不能把中间层停止算新解。正式1870/2413和独立61/400仍不变；GPU、smooth、Transformer和全部家族目标保持完整。
