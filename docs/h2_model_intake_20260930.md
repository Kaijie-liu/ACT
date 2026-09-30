# H2 模型对象接入与容量准入

本阶段从 `feat/moe-route-verification`、`a39ec0f0935752f0ba8b0623a6084406789dab0c`
继续此前已登记的未提交接入工作。H2 已能从一个实际 CPU float64、eval 模型对象捕获
声明来源，在统一硬预算内生成和独立检查全部输出义务。这个对象使用原已固定的合成
系数，不是训练模型或新增验证输入；本轮没有真实模型收益或外部胜负结论。

另一个决定性结果来自只读元数据核查：三个已注册 bal010 模型的源编码下限已超过
当前接收上限。因此不能直接启动真实请求。下一门是同一数学对象的有界分块表示和
完整义务检查，不是扩大时间、调读取上限或重新选择容易模型。

## 同一合成对象的新来源

[接入协议](../configs/h2_model_intake_controls_20260930.json)在调用前固定，canonical
SHA 为 `c496b003c786eadbf3c2dac4f03ddabe00e57cf9336e26ccf2dbb86bde0735ca`。
[薄适配器](../scoped_source/endpoint_intake.py)把原 weighted_sign 系数直接装入私有
Linear/ReLU 专家和 affine router；不经过 float32 初始化值转换、不训练、不加载
checkpoint、不执行 forward 来提取来源。创建过程恢复调用者 RNG。

现有 [capture](../scoped_source/capture.py) 与
[capture_bound](../scoped_source/sparse_intake.py)未修改。新控制拒绝 hook、实例方法或
序列化覆盖、输入/参数变更、错误域/性质、dtype、训练模式及不支持的算子。固定工厂的
专家 module 和参数存储彼此独立；这不等于捕获器已通用拒绝所有共享存储别名。

独立声明与实际捕获逐项相等。捕获器增加已有的 `trust` 字段，故来源身份从
`b50e6ac79b92ac1d25e6a2ad8d83d1958353e90eeba869663e3d9d7c9eff0de2` 变为
`d4d33ce1e9c56cb264e27b5325f51559423f9797deb75315a624641e85b38a2e`。
其余图、输入和系数相同；两臂各自生成新矩阵、新候选和证明，不拼接旧正界。
有限 native probes 仅检查四个点的组件与 selected-weight 计算，位于控制测试中，
不是预算内网络证明，也不能建立全域浮点等价。

## 完整调用和证据接收

[worker](../scoped_source/endpoint_worker.py)把对象创建、数值库导入、捕获、验证和
capture receipt 发布放入原 produce 截止时间；后续来源构造、候选求解、序列化、
独立检查、接收、最终发布和清理继续全部计费。接口最多 300 秒；本批普通控制预算
30 秒，捕获截止控制总预算 8 秒，其中 produce 在第 5 秒截止以保留后续阶段时间。

[receiver](../scoped_source/endpoint_supervised.py)将 capture receipt 与实际已检查源、
请求、模型状态、输入、执行身份及 producer 文件逐项核对，其内容哈希进入最终收据。
父进程仍锚定 checker 的实际 stdout；伪造一套内部自洽文件不能替代已观察到的调用返回。
捕获模式新增六个 producer 文件绑定，完整归档绑定三十个实现/配置/测试文件。

[九条调用归档](h2_intake_controls_20260930_r1.json)结果如下。所有受控失败保留，
没有因错误或超时删除分母。

| 调用类别 | 数量 | 结论 |
|---|---:|---|
| 正常端点臂 | 1 | 全部 3/3 义务正，检查 6 个端点下界 |
| 正常 McCormick 臂 | 1 | 2/3 义务正，检查 3 个下界，完整请求未决 |
| 缺一个端点证书 | 1 | 检查 5 个界、缺 1 个，UNKNOWN_MISSING_EVIDENCE |
| 参数/输入改变、捕获异常、删性质、篡改 checker stdout | 5 | ERROR |
| 捕获延迟 | 1 | TIMEOUT，部分捕获记录保留 |

两正常臂逐值相同的 source/P/gate/reuse 已核对，复用清单为空。正臂整体最小下界
约 0.1；MC 本次候选的已检查下界约 −0.275。后者不是模型反例，不是 LP 最优值，
也不是[前一来源控制](h2_source_controls_20260930.md)中约 −0.025 的精确可行点。
本批没有重新归档可行负点或独立路由变化见证，不能把旧见证写成本次新增结果。

| 正常臂 | 生成阶段秒 | 检查秒 | 接收秒 | 全 API 秒 | 可搬迁包字节 |
|---|---:|---:|---:|---:|---:|
| 端点 | 1.1740 | 0.0942 | 0.0561 | 1.3320 | 83,338 |
| McCormick | 1.1577 | 0.1002 | 0.0553 | 1.3205 | 82,037 |

阶段外发布/监督开销仍包含在 API 总时间。单次控制不是计时实验，不由这些数字声称
速度胜出。删性质及篡改 stdout 的 ERROR 发生在候选生成之后，原始 generation 与
候选仍保留，不能按 ERROR 计为零求解。捕获错误/截止则未到候选生成阶段。

搬迁控制把完整包复制到新位置、隐藏原包，用 `python -B -I -S` 运行，无 checkpoint、
数据、模型对象、仓库或求解器依赖。它是十三项接入测试之一，不是第十条监督调用。
只读归档重检独立重建声明图、全部 pair/性质及有理数下界；不信任原求解器状态。

保证仍是 `CHECKED_DECLARED_SOURCE_POSITIVE`：声明实数输入、源图到完整输出闭合；
源解析器/检查器实现及原生程序与声明图的对应关系仍受信任。不是 PyTorch/CUDA
逐运算浮点 SAFE，不是生产 HybridZ lowering 差分，也不是外部工具优势。

## 已注册真实模型的静态容量

[容量脚本](../scripts/audit_h2_model_capacity.py)仅读取固定源码与注册元数据；
[结果](h2_model_capacity_20260930_r1.json)绑定这些文件和 checkpoint 身份，但没有打开
checkpoint、加载模型、读取数据、选择样本或运行求解器。三模型参数数均为 6,961,368，
张量数为 52。seed 1/2 架构依据训练配方；seed 0 的注册规模一致，未重新加载其拓扑。

每个参数以 binary64 存储需 8 字节，原始参数至少 55,690,944 字节；base64 编码
下限为 74,254,592 字节。逐张量填充、JSON、输入及身份字段只会增加该值。
脚本从 portable reader 和 receiver 的源码 AST 提取当前 67,108,864 字节限制。
因此源文件本身就不被接收，与输入或是否容易证明无关。

按已注册 `3072→128→8` router 与 `3072→256→128→10` 专家配方：28 pair ×
9 性质＝252 条义务，最多 504 个端点。每 pair 有 4,892 个变量。假设所有权重非零，
单 pair 的仿射等式贡献 2,036,124 个系数项；当前逐性质复制基础 P，会重复存储
513,103,248 项。这是**稠密架构下的仿射贡献**，不是实测 nnz、总 LP 大小、内存或
时间预测；ReLU、guard 行另计，实际权重零值没有读取。

第二家族卷积模型保留已归档 checkpoint 身份及 67.06% test accuracy；其 Conv2d、
pool 算子不在当前捕获支持范围。不能将它改成 MLP、删池化或换 top-k 后称为同对象。
历史所有封存输入与 holdout 不变；旧 529 索引排除清单不是未来最新排除清单。

## 继续门与停止线

模型对象接口控制通过，真实模型容量准入未通过。先研究新版证据表示：将原始源系数
分块保存、来源约束各保存一次，pair 引用有序来源块与 guard/local map，性质引用 pair
并保存独立目标与证书。一次只处理一个 pair。只把 252 份 P 改成 28 份仍不够，源文件
上限及跨 pair 的共同来源重复都要显式处理。

这不改变 LP 数学或 gate 范围，不删除检查。检查器仍从源独立重建 affine/ReLU、
范围、guard、全部 pair/性质及下界；哈希只负责身份。小控制必须证明展开后的新旧
精确矩阵/目标相同，并拒绝错映射、漏/重义务、外来块、污染与截止。MC 的额外变量、
差值范围和四条乘积行仍须逐性质检查，不能仅共享一个已知 hash。

新格式必须重新通过可搬迁检查及完整硬预算监督，再检查静态容量。若仍需要全量复制
或只能检查局部输出，保留未通过状态；不能据本报告自动冻结真实请求。不重开 H1
稠密依赖剪枝、缓存计时、输入 98 或旧 holdout，不修改 25%、数值门或模型分母。

## 验证与复查

本阶段 13 项接入控制、18 项既有监督控制、69 项来源/端点/工程维护回归及 4 项容量
控制均通过，共 104 项。只读协作审查核对来源身份、完整终态链、共同 P/gate/reuse 和
成本；它不替代独立人类技术审阅或干净环境实证复现。

[新版监督回归归档](h2_supervision_intake_regression_20260930_r1.json)保留另一个目录的
31 条调用：5 正、3 非正、4 缺证据、10 ERROR、8 TIMEOUT、1 RESOURCE_LIMIT。
旧 R1/R2/R3 监督目录及归档不修改。因 worker/receiver 增加对象模式，其源码 hash
改变；重算旧监督成本归档需旧提交或对应快照，不能拿当前 hash 改写旧结果。

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -I -S scripts/archive_h2_intake.py /data1/Kane/MOE/baseline_runs/h2_intake_controls_20260930_r1 --check docs/h2_intake_controls_20260930_r1.json
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -I -S scripts/archive_h2_supervision.py /data1/Kane/MOE/baseline_runs/h2_supervised_intake_regression_20260930_r1 --check docs/h2_supervision_intake_regression_20260930_r1.json
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -I -S scripts/audit_h2_model_capacity.py --check docs/h2_model_capacity_20260930_r1.json
```

这三条只读重检不会启动求解。阶段盘点 MOE 占用 224,670,552,064 字节，较上一阶段
224,656,019,456 字节增加约 14.53 MB，主要是新控制/回归证据。本轮未删除文件、
安装依赖或下载数据；后续仍按独立缓存维护规则，不把失败证据当缓存清理。
