# 完整真实首 bank 的入场审计

本轮将已通过数学门的 D209 对接真实结构，并验证当前实现是否至少能完成完整首 bank。预期结论是其累计身份条目账必超上限；不得将这个预期改写为已经获得真实能力。另在 THEORY.md 证明普通重叠 Conv 何时保证传递共同的父非线性 graph slack。不是新增验证器、solver helper 或正式回放。

## 单次冻结

schema 为 d210_full_prefix_admission_v1。执行前冻结恰三文件 PREREG.md、THEORY.md、run_audit.py。freeze 同时固定九项 anchor：D209 preregistered.json/exit.json；D179 inputs.json/result.json/model_0.json/model_1.json/model_2.json；D209 和 D207 的 fiber.py。

唯一 RUN 为 experiments/neural_hz_20260831/results/d210_full_prefix_admission_20261005_v1。显式 --enabled，独占创建目录和产物；创建即消耗版本，失败不编辑或重跑。冻结前禁止本诊断 import、AST、compile、collection 或候选数值执行。

完整继承 D209 receipt 的7425源身份/14输入，前后流式哈希；认证其4100 tests/216 files完整通过与正式增益0。此次不重新执行数学人口，也不追加、替换、裁剪旧测试；因为没有修改域实现或运行候选，只是核对已认证源结构上的费用下界。最后数学资格仍来自 D209；本诊断成功不取得任何新数学组件、native、模型或正式资格。

## 全部真实来源

使用 D179 inputs.json 的全部三项，原顺序，不选性质、不读 sat/unsat 标签，不按 margin 选择实例。原ONNX模型只哈希，不导入、解析或执行；形状和完整消费者从认证 D179 JSON 读取。必要条件是完整结构标志、节点索引/类型/端口匹配、正的单样本 NCHW 尺寸、完整消费者身份。

对每个模型验证前六节点是完整 Conv0→BN1→ReLU2→Conv3→BN4→ReLU5，保留首 bank 的所有消费者，包括 CIFAR medium 的 shortcut。输出完整 input shape、两 bank shape、d/m/n、全部首 bank uses与节点身份。不是D025/D120的窗口子集，也不恢复缺身份槽位的旧pickle。

预注册尺寸为 CIFAR large d3072/m65536/n65536，CIFAR medium d3072/m14400/n8192，Tiny medium d9408/m46656/n25088。程序从完整记录计算并核对这些值；任何不符 fail closed，不临时改成更小人口。

## 审计公式及资格

按 THEORY.md 和已冻结源码认证 entries_lower=9*d*m、entries_cap=64000000。忽略此前全部其他费用时，最迟第 floor(cap/(9d))+1 门已经超账；该数必须不大于完整 m。三模型的当前实现入场结论必须均为 false。

这个下界只针对当前 D209 的累计逻辑条目，不是物理内存、GPU显存、速度、所有可能 Neural-HZ 定义的下界或某个性质难度。未执行完整首 bank，因此不能将证据写成一次实际 TIMEOUT/ERROR；应写成由真实形状与当前代码推导的预先入场否决。定理中的 rho/有效系数/正 cap 条件本轮没有从真实权重计算，条件性保证不得写成已观测收益。

## 资源与自动保留

只用标准库，不导入候选、ONNX、torch、pickle，不做AST、模型forward或solver。CPU [0]、RLIMIT_AS16GiB、总60秒闹钟，bytecode关闭。仍保留监督器RSS high-water增长加65536 bytes，以及trace peak加metadata加65536 bytes，分别不超过1GiB的原门。完整前后流式核验统计时间和字节数单报，不当作候选执行性能。冻结的候选 whole256M/branch200M/entries64M/512bit 均不改变；本诊断没有新候选物理资格。

自动独占写 preregistered.json、admission.json、exit.json及其hash，失败也有回执。源码/输入/生产branch、commit、tracked diff 前后核验；任何证据不符都只得到诊断失败。metadata_only=true、candidate_execution_registered=false，所有 candidate/native/GPU/physical/formal 资格 false，formal_gain=0、新解0。

2026-10-05 Australia/Sydney，redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。历史模型、实验和 /data1/Kane/HyZor 只读；新写入仅本新目录和唯一新RUN，无默认启用、生产改动或commit/push。正式1870/2413与独立E0 61/400不变，Goal active。
