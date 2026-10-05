# D152：完整活读出维度的结构证据检查

状态：执行前预注册。此文件与 `audit_metadata.py` 冻结后只允许一次执行；本文件不因结果修改。

## 问题与边界

目标仍是从数学定义研发强大的非凸 Neural-HZ。本次检查只决定下一定义是否存在真实适用结构，不实现抽象域、不计算新界、不运行网络、不调用验证器或求解器，也不宣称能力、速度、数学资格或生产资格。

固定误差加载 `q=A u+B z+C beta+d+E eta` 在具有独立可达折面的普通块上，可能必须支付完整幅度维度。只有完整活读出 `Vq` 确实低维，压缩输出幅度才有直接结构空间。检查固定三个原模型中的 ReLU 是否仅由最终 Gemm 消费，并记录该 Gemm 常量 B 的维度、转置属性与原连接。这是维度上界证据，不是实际矩阵秩、可靠界或健全投影证明。

## 固定人口、来源和身份

仓库 `/data1/Kane/FSE/ACT`，分支 `redu-hz`，HEAD `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`。`git diff --binary HEAD --` 的 SHA256 必须为 `29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5`，执行前后均检查。

继承 `results/d150_stable_phase_charts_20261004_v1/preregistered.json`，其 SHA256 为 `7f13f7860a3490aa757a0fef853147d35620c2fa599a10a6336404df737a2813`。完整认证其中 7,285 个 source 身份及 14 个 input 身份，不能选取子集代替；执行前后检查。首次导入 ONNX 必须在前置认证之后。

固定读取 `results/d120_mixed_consumer_source_20261002_v1/complete_0.json`、`complete_1.json`、`complete_2.json` 的 `source.model_path` 与 `source.model_sha256`。三个全部处理，不因匹配与否过滤报告：

- CIFAR100_resnet_large.onnx，SHA256 `5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16`。
- CIFAR100_resnet_medium.onnx，SHA256 `aba117ad0ad4abdd630c220beca70cd58825e72e7bada5dffdda10bb725cece4`。
- TinyImageNet_resnet_medium.onnx，SHA256 `234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776`。

模型必须为 regular file、每个不超过 64 MiB。先认证原字节，再由同一份内存字节解析一次 protobuf；禁止解析后再换来源。后置重新 hash 原文件只是身份检查，不再次解析。不得修改、覆盖或混写历史结果与模型；只读访问限上述认证和元数据范围。

## 只允许结构元数据

记录全部节点的 operator/domain/input/output，op counts，图输入输出，以及全部 ReLU 的匹配或不匹配原因。initializer 只读取名字、维度、data type/location 与序列化字节数；不取浮点数组、权重符号或数值秩，不调用 tensor-to-numpy、ONNX Runtime、checker、shape inference、任何模型 forward。

用 `onnx.load_model(BytesIO(authenticated_bytes), format='protobuf', load_external_data=False)` 或 `load_model_from_string(authenticated_bytes, format='protobuf')`。外部 tensor data、嵌套 graph/function、重名 initializer/输出、initializer 覆盖 graph input 等未覆盖情况 fail closed。模型逐个处理并释放；完整图关系保留于报告。

匹配条件：标准域 Relu、单输出、唯一消费者为标准域 Gemm 的 A 输入，该 Gemm 单输出且是唯一模型输出；Relu 幅度没有其他消费者或图输出旁路。Gemm 为 2/3 输入，B 是 rank-2 常量 initializer，`transA=0`、`transB` 为 0 或 1。所有其他 ReLU 均保留不匹配理由。

只读 `transA/transB` 的整数属性。`alpha/beta` 仅记录名称、类型和是否缺省，不读取显式 FLOAT 值；缺省可记录规范默认值。未知属性、重复属性、不支持类型/参数拒绝匹配。B 的有效形状为 `(K,N)`：transB=0 时依原形状，transB=1 时交换。报告 hidden dimension K、output dimension N 与 rank upper bound `min(K,N)`，不报告实际 rank，不声称非零 alpha、bound shape 或已绑定数值仿射映射。C/bias 的 metadata 不替代数值/广播资格证明。

解释依据为 [ONNX Gemm 规范](https://onnx.ai/onnx/operators/onnx__Gemm.html) 与本机已认证 ONNX loader；这不是新的 Neural-HZ 规则。

## 执行、预算和自动保留

默认关闭，必须显式 `--enabled`。冻结文件 `freeze.json` 的精确结构为 `{"schema":"d152_live_amplitude_metadata_v1","source_sha256":{绝对PREREG路径:hash,绝对audit_metadata路径:hash}}`；冻结前不得 import、AST parse、compile 或试跑新脚本。

唯一 RUN 为 `results/d152_live_amplitude_metadata_20261004_v1`。必须排他创建，一经占用即消耗本版本尝试；失败、超时或部分结果不能删后重跑，不能调整冻结预算或源码补跑。

单 CPU、线程数 1、CUDA 隐藏，地址空间上限 16 GiB。单次总阶段外部 timeout 60 秒；内部 alarm 58 秒，留终止和写出时间。RSS high-water 增量加 65,536 字节、tracemalloc peak 加 tracemalloc 自身 metadata（tracer_metadata_bytes）与 65,536 字节，分别不得超过 1 GiB。两个指标分别报告，不能冒充完整候选物理内存资格。运行缓存和临时目录仅在新 RUN 内，禁止写 bytecode 到历史源码。

保留 source/input 认证摘要、三模型逐项证据、完整节点关系、资源数据、成功/失败原因、退出状态与 artifact hashes。超时或异常应自动保存已得到的证据；如果被外部强制终止，保留目录和外部退出记录，不伪造完整结果。失败即无结构资格，不能推断没有该结构。

## 不变的能力与验证口径

不改生产、不启用候选，不引入 helper、attack、PGD、BaB、split、backward/dual rescue。所有连续源、二元非凸相位、谓词、共享身份和具体输入重构要求不变。

最近完整数学人口仍为 D150 的 4,000 tests / 210 files。本次只是元数据证据检查，既不运行或替代该人口，也不授予任何新候选数学资格。未来候选仍需完整原人口及其新增定向证明测试，不能把此审计视为减少验证的先例。

正式基线仍为 1,870/2,413；独立 E0 仍为 CIFAR100 25 + TinyImageNet 36 = 61/400。新正式收益为 0。任何资格/晋级仍须经过原同结构 shadow、逐家族与完整同路径回放，以及全部原门槛。
