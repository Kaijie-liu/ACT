# HybridZ 设备可选候选算法控制

本协议将[CPU 多目标支持](hz_batch_support_controls_20261001.md)的候选内核独立
版本化，CPU 与 CUDA 共用同一更新规则；原 V1、HZ→LP 检查器和数值接受公式不改。
先实现设备路径，运行 CPU 参考和模拟故障控制；**本阶段不授权实际 CUDA 执行**。
当前设备有其他计算进程，瞬时利用率为零不是独占或性能测量准入。

## 数学与执行合同

沿用一个给定 guarded HZ 的原始因子坐标，多目标共享 A/E 与各自的转置，不合并
专家私有因子。128 次投影次梯度更新、零初值、每列最佳候选和精确全零兜底不变。
只参数化设备，保持 COO、float64、单 CPU 线程及原步长；不增加 AMP、TF32、CSR
改写、迭代搜索或 native fallback。二元因子仍为显式连续松弛。

候选只提供有限乘子。CPU 使用原系数、原目标和原精确残差核重新评价，再由独立
检查器接受完整清单。CPU/GPU 候选轨迹不要求逐 bit 相同，数学有效性不能用近似
一致替代；CPU 新旧版本的固定控制需要差分。设备标签不改变数学 batch 身份。

GPU 入口要求调用方锚定的执行上下文，绑定 batch、绝对截止、GPU UUID、单设备
可见性及 allocator 上限；无上下文、错设备、错 hash、过期或初始化失败均拒绝。
这个上下文本身**不是资源预约或准入证明**，后续监督器仍须独立取得资源准入。
GPU 只能在新进程使用逻辑 cuda:0；所有浮点分配明确为 float64。

记录 CPU 验证、CUDA 初始化、张量构造／传输／同步、迭代同步、回传和精确候选
评价的费用。返回前同步并检查绝对截止，不把异步提交当完成。OOM、同步错误、
非有限值或部分列不返回先前 best，不静默转 CPU／低精度。allocator 统计不等于
整个上下文显存，memory fraction 也不是 OS 硬配额。

候选入口仍是合作式 deadline，不能中断阻塞同步。实际 CUDA、完整传输／检查／
序列化／清理计费与强截止，必须接入另行冻结的 owned-process 监督；不能把此阶段
当成已经完成 G2、真实模型容量或 GPU 加速。原 production fallback 不变。

## 固定控制

[配置](../configs/hz_device_candidates_20261001.json)绑定四个既有来源，加常数目标
和四输出共享／私有因子控制；后者含零宽连续因子、独立二元列、上下界及八条义务。
检查 min/max、offset、零行 A/E、换序、单列／多列、上下文污染、缺列、错误乘子、
非有限／溢出、显式 CPU 分配不受默认设备影响，以及截止与异常。

GPU 初始化、OOM、同步及回传后截止的故障模拟必须标为 stub，不得算实际硬件通过。
检查缺候选保持原义务数；所有正常包以原 checker 独立复读，不重新生成候选来冒充
证据检查。新目录保留实现、配置、日志、正常包和拒绝记录，失败不覆盖。
本阶段不运行模型、数据、真实 LP/MILP、GPU、训练或封存输入。

## API 依据与下一门

PyTorch 的 [sparse.mm](https://docs.pytorch.org/docs/2.9/generated/torch.sparse.mm.html)
支持 COO；可选 reduce 扩展限 CPU CSR，因此这里不使用它。
[synchronize](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.synchronize.html)
用于完成观察；[memory fraction](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.memory.set_per_process_memory_fraction.html)
只约束缓存分配器。代码和当前 CUDA/dtype 的实际可执行性仍须硬件控制，不靠文档推定。

本阶段通过后，下一门是设备版本的硬预算监督、资源门与冻结的微型 CPU/GPU 控制。
性能测试须另行预注册完整成本标准、冷启动与顺序，不能将小核快或高利用率视为
端到端收益；强原生 CPU 基线及可用外部 GPU 路径的公平机会仍保留。
