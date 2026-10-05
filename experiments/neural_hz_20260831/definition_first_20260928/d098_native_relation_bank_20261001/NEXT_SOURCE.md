# 新鲜残差块上的完整关系接入

D098 数学阶段已经完成。下一步应新预注册真实同结构检查，而非继续把小型目录测试或源码包装当成主线。本文只给接入口与资格边界，不登记新的执行，也不把旧来源统计当本次运行结果。

## 保留生产出生并选择完整当前结构

从原 INPUT 新鲜加载 Tiny corrected graph，到第一个包含三组原 ReLU 的普通残差块：ACT ReLU5、ReLU9、ReLU20。ReLU9 经过 Conv10 和通道仿射，与 ReLU5 经过 Conv13 的 shortcut 在 Add16 合流，再经过 Conv17 到 ReLU20。所有旁路、原相位和谓词必须保留。旧记录在 ReLU20 有 1054 个原 bit、1054 EQ、2108 LE；这只是历史规模参考，新运行必须从实际当前对象重新认证。

源模型和性质沿用已登记的原始来源及 corrected graph 身份，测试对象的预注册选择不能进入变换规则。D098 一律按实际谓词结构选择全部组，不读实例、标签、margin 或 solver 状态。CIFAR100 large 的对应块含 identity Add_8，可在第一条真实接入获得可靠证据后，以同一规则建立独立同结构检查；不得将 Tiny 结果当作 CIFAR 通过。

新 worker 借鉴 shadow_worker_dtype_v2.py 的原 dtype、VNNlib、VerifiableModel、TorchToACT 和从 INPUT 正向传播流程，并执行已有 BatchNorm producer graph 修复与图身份审计。不能直接调用旧 c5_corrected_prefix 或 c5_integrated_prefix 的 main：它们包含旧 hooks 和大 pickle 快照写入。不能从缺分配器注册的旧 snapshot 假装新鲜续跑。

出生、deferred 和原 frame allocator 均保持原样。D098 在已完成的当前 HZ 上认证实际存储关系，不需要把 partial 物化解释为未舍入的原网络预激活。它的认证不免除全模型来源、共享 latent 与 decoder 资格；实际 HZ 的数学健全性仍建立在原前端边界之上。

## 闭合组件与后续传播不能混称

第一项真实检查可在预注册边界闭合：完整认证全部原门、原生依赖、全部源组、所有后继、未覆盖及歧义，并在新注册的完整资源门内判断全组安装。预算失败整体失败，不挑少量有利门；若仅完成普查，结果明确是普查，不能说关系已安装或性质已验证。

闭合层20后即停止不算新增 CERT/ADV，且不是全部模型能力。D098 添加新的连续列而没有发布生产 frame 高水位；不得把其输出直接塞回 tf 缓存继续出生。真正在线继续传播须把追加列、共享缓存、分配器、rebase 和具体输入重构的生命周期一致性独立证明并测试，不能以同一个 frame_id 代替证明。

## 执行前必须完整支付

保留 whole 256M、branch 200M、evidence 40M、numeric 与 proved temporary 合计 64M、512bit、原 1 GiB 物理观察边界、worker 240 秒与 AS16GiB 的原作用范围；新银行的作用域和单位须在预注册时写清。C5 的 256M 原本是 selected affine island，不要追溯扩大为全模型工作门，也不能把新算法已登记费用挪到范围外。

登记原 model bytes 与读取/解析、转换后模型参数、ACT net、tf、after/before、完整 HZ 与 expression/precomputed/allocator 缓存、input decoder、关系银行、精确 Fraction 与行回执，以及输入/中间/输出数组的同时存活。不把只看单一 HZ buffer 的报告叫完整物理资格，不重走 D090 的无效全 pickle 读取路线。不能在预算检查前先生成巨大关系银行或最终矩阵。

测试基线已经是 D098 的 3845 项、188 文件。新执行须保持其源码和失败历史身份，不能重跑已消费 D098，也不能给原 RUN 追加未登记 worker 阶段。任何新代码按原先冻结和资格顺序处理。

## GPU和正式晋级仍是未完成项

当前银行和共同观察是 CPU Fraction/SciPy 实现；不能给模型 `.cuda()` 就宣称域已 GPU 化。D017/D043 的初始化 OOM 和 D020 的 tracing 权限拒绝仍保留，根因未定，无资源或权限放宽授权。本源接入检查不能预支未来 GPU 提速。

真实同结构完整成功后，才进入同规则 shadow、逐家族与完整 2413 回放；CIFAR100/Tiny 的 400 例独立。每个旧解、13 家族和外部 E0 全保留，invalid 为 0，新增 ADV 具体网络验证。新颖性、健全性、物理资格、速度和正式成绩分开，不将现有有效关系与完整新域创新等同。
