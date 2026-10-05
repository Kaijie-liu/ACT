# 共同非线性解码的研究结论与后续门槛

上一回合只汇报状态，为 no progress。本回合完成一个普通 ReLU bank 的共同非线性充分统计、构造性唯一解码证明、核空间一维的闭式，以及源绑定和终端成本反例；这些新证据改变了对下一候选的判断，属于纸面研究 progress，不是数值资格或正式增益。

## 相比此前研究的新信息

[定理全文](THEORY.md)证明 y=A^T ReLU(Ax+b) 唯一决定整组幅值，任意多个混权消费者可共享一个 decoder。它不同于保存几个独立支持上界，也不是 rank 分解后将误差逐项取盒。D126 的共同实现缺口在这个精确接口定义中有明确归宿，但并未因此获得便宜的抽象变换或一般 CNN 终端。

[D004](../d004_observable_interface_20260928/D004_RESULTS.md)、[D051](../d051_interface_sufficiency_20260930/PROOFS.md)和[D059](../d059_cross_layer_interface_20261001/THEORY.md)的线性接口下界没有否定非线性 decoder。相反，本轮三门例明确证明固定仿射 decoder 不存在，而共同 clip decoder 存在。不能把此前受限负结果扩大成所有低维接口都不可能。

另一方面，充分统计和唯一凸解码源于经典 firm nonexpansiveness 与凸共轭原理，不是新优化定理。一般函数图容器有现成先例，不能仅凭写出它就宣称 PLDI 新颖性。一维核空间闭式也是经典标量二次最小化；这里只获得新的项目级构造和准确适用范围。

## 不能忽略的成本反证

实际所有原源谓词、raw residual、原相位零标签和原 q bounds 必须保留或精确代入。源名称相同不代表共同 y 已绑定；必须显式约束 y=A^T diag(beta)(Ax+b)。解码器本身不会保证其隐含源满足原输入范围。

一般 decoder 仍包含 m 维凸优化；一维核空间版若在 terminal 恢复 clip 参数 t，原 m 幅值宽度就回来了。x 本身已经是 q 的充分输入，所以这个新接口必须进一步证明推理或全成本收益。当前没有这项证据；不以接口少一维、可在 GPU 扫描或手算正控代替它。

普通 CNN 优先仍不变，不为核空间恰好一维的特殊类启动大量适配工程。下一安全研究是围绕普通通道扩张中的共同非线性接口，构造明确可计算、可被下一原 ReLU 消费的关系算法，并与旧 HZ 加同样有效信息比较完整费用；也允许根据负结论更换定义假设。不能再次只调排序/缓存或修框架来代替该证明。

## 与 Attention 实验的分账

最近实际执行是 D130：3953 项数学组件测试通过；来源只完成 38/192 行、233/1152 个 roots，在 235 秒停止。38 行激活前上下界改善，11 个 ReLU 下界、13 个上界改善，但零新增稳定门，完整来源未过门。已补齐[D130 结果与失败归档](../d130_import_isolation_20261002/RESULTS.md)。它没有因本轮纸面研究获得 native、GPU、shadow 或回放资格。

普通 CNN 最新完整来源仍是[D120](../d120_mixed_consumer_source_20261002/RESULTS.md)：1600 个局部读出、下一 ReLU 改善为 0。本轮没有读取新模型参数、运行模型或改变这些历史记录。D128 的[CNN 共同斜面研究](../d128_bounded_attention_source_20261002/CNN_SOURCE_PLANES.md)仍受 D018/D052 强旧对照限制，不因新的充分统计定理被升格为能力成果。

正式 baseline 1870/2413（1063 CERT + 807 validated ADV）和独立 E0 61/400（CIFAR100 25 + TinyImageNet 36）都零新增。候选保旧尚未通过；未知、超时和数学正控均不计分。完整目标继续 active，没有默认启用、生产改动、commit 或 push。

## 可恢复边界

没有新的候选代码、AST/import/compile/pytest、source worker、solver、网络 forward 或 GPU 调用。本轮没有后台数值任务，D128/D129/D130 均为已结束的冻结历史，不重跑。未来实现仍需新预注册和 freeze、继承最新完整测试人口以及不变资源门，再经真实同结构、shadow、逐家族和完整回放。

2026-10-02 Australia/Sydney；redu-hz；commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；tracked binary diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。只写新的隔离文件，历史和生产只读。

write-page 技能用于分开定理、正负控制、先例、未获资格和恢复入口，并遵循现有本地 Markdown 文档格式。没有外部 Page；保存和读回不等于已验证网页排版。归档校验清单是事后文档封存，不是数值执行前 freeze。
