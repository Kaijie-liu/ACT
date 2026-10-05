# 两个真实 ViT 的只读图元数据检查

这是 D106 的来源诊断，不是 Neural-HZ 候选、数学资格测试或模型验证。上一轮有新的纸面分离证据，属于 progress。本轮要确认原图的概率轴、Softmax/MatMul 连线、显式 shape 与布局常量；不从文件名或算子计数猜测实际结构。

唯一输入是正式未解清单中的两个唯一 ViT 模型，按路径排序全取，不按标签、margin 或结果选择：

- vit_2023/onnx/ibp_3_3_8.onnx，303176 bytes，SHA256 9cc53b9edb35d40d70de1d816008a434c7e60b0404cf3cb38b9aefc189692883。
- vit_2023/onnx/pgd_2_3_16.onnx，325647 bytes，SHA256 246326387574617a274b6f73f7f52771df1195e03c99065ce4ad18c46b8e0984。

根为 /data1/Kane/data/vnncomp2025_benchmarks/benchmarks。模型原件仅以 rb 读取；大小和原 SHA256 必须一致。只用 ONNX protobuf ParseFromString 解码，不使用 onnx.load 的外部数据加载，不运行 checker、shape inference、simplifier、runtime、Torch、网络或 solver。任何外部 tensor、子图、局部函数或无法记录的属性类型拒绝本次完整诊断，不自动忽略。

新工具 graph_probe.py 默认关闭，只有 --enabled 启动监督器。源码、本文、正式 manifest、D106 seal、Python executable 及 ONNX/protobuf/upb/numpy/ml_dtypes/typing_extensions 的解码依赖清单在第一次 AST/import/compile/执行前以 freeze.json 和 DECODER_SHA256SUMS 冻结。执行前后认证，不修改已消费版本。清单生成只使用文件枚举和哈希，不导入候选或解码模型。

固定执行为 /data1/Kane/miniconda3/bin/python -B experiments/neural_hz_20260831/definition_first_20260928/d107_vit_graph_inventory_20261002/graph_probe.py --enabled。唯一 RUN 为 experiments/neural_hz_20260831/results/d107_vit_graph_inventory_20261002_v1，独占创建即消费。监督器先写 preregistered.json，再启动独立子进程；完整 stdout/stderr、逐模型图记录、exit.json 自动保留，超时/异常同样保留，不挑选成功子集。

父/子均单 CPU affinity 0、单线程、CUDA 空、AS16GiB；子进程启动至结束限60s，包含导入和解码，无网络推理。原始单模型≤64MiB、节点≤10000、initializer≤10000、单条静态整数张量解码≤64元素、结果单文件≤4MiB。只读取原注释 shape，不运行任何 shape inference；未知或符号维度保留未知。FLOAT/DOUBLE 权重不转 ndarray、不计算，只记 tensor 身份/shape/类型与序列化哈希；仅小型 INT32/INT64 布局常量可直接解码。超过边界拒绝，不调整上限。

父子 peak RSS 和 tracemalloc peak/metadata 各自记录并检查 RSS 与 tracer 合计≤1GiB；这些是这个小型解码诊断的观察，不冒充 D106 完整物理资格。共享库/native buffer 用 RSS覆盖，非单靠 tracemalloc。JSON 缓冲上限以及完整依赖、模型原字节/消息、证据暂存的共存仍受该进程界限制。这里不运行原数学 suite，所以没有缩减其人口，也不提升任何候选资格；D098 的3845项/188文件仍是最后已执行数学结果。原候选 whole256M/branch200M/evidence40M/numeric+tmp64M、512bit 和 worker 门均未改动，本诊断不证明满足候选全成本。

只输出 graph metadata 和诊断完成/失败，不输出 SAFE/ADV。opset版本与显式 axis 一起保存；opset≤12的Softmax flatten语义必须另读规范核对，不自动等同于最后一维归一化。真实 native HZ frame、源支撑、二元列绑定和概率误差来源仍未知，即使图检查成功也不能自动取得它们。

分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。tracked binary diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5 不变，暂存区空。所有写入仅本新隔离目录及唯一新 RUN；不修改生产、历史模型/日志/结果，不 commit/push。正式1870/2413、独立61/400不变，formal_gain=0。文档技能用于区分只读来源诊断、候选数学资格和实际验证成绩。
