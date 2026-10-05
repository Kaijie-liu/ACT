# 残差系数几何一次性只读诊断预注册

状态为执行前合同。先冻结本页、DEFINITION_AND_SCOPE.md、inputs.json 与 audit_geometry.py，再进行唯一一次执行；此前不导入、AST 解析、编译、收集或数值试跑新代码。失败版本及其预算不改、不重跑。此诊断决定定义研究下一动作，不构成域组件、模型验证或性能测试。

## 固定来源与结构人口

仓库 /data1/Kane/FSE/ACT，branch redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked binary diff SHA256 必须为 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5，执行前后检查。

继承 results/d158_joint_forward_support_20261004_v1/preregistered.json，其 SHA256 为 64cd535e4ae10bfaeb8204d21841f991781aa981e2aeddb359a87ca8050103d9。完整认证 source_sha256 的 7327 项和 input_sha256 的 14 项，不能筛选子集；前后都检查。该继承是身份与已有测试人口保留，不授予新诊断任何旧数学资格。旧 4032 tests/212 files 不在本次重跑，也不被本次取代或减少。

inputs.json 固定两份 ViT 原模型与已存原图，全部 ReLU 分别为 2 和 3 个，五块全部报告。模型名中的 PGD 是历史训练名称，不执行攻击；模型身份只用于来源人口和认证，算子匹配只由实际端口、常量维度和消费者决定。不得依据公开标签、历史状态、margin 或观察到的常数筛选。输入性质仅在继承身份检查中读取字节，不解析性质或生成输入范围。

每个模型及 graph JSON 必须为普通非符号链接文件且不超过 1,000,000 字节；原模型只解析一次认证后的同一份内存字节，禁止 external tensor data、嵌套图/函数、重名或歧义连接。使用已认证 ONNX protobuf loader，不运行 checker、shape inference、ORT、torch 或任何模型 forward。

统一识别 q→Transpose→BN→Transpose→MatMul(W1)→Add(b1)→ReLU→MatMul(W2)→Add(b2)，末端与同一个 q 相加。核对每个内部端口的全部消费者、图输出旁路、常量维度、opset 和属性；任何未覆盖结构 fail closed。五个 ReLU 必须完整匹配，不能跳过失败块留下已成功块当完整结果。每块完整保存结构身份和区间系数证据；未完成时保存已有部分并明确不完整。

## 数值范围与固定诊断

只解码这些完整块的 FLOAT inline W1/W2/b1/b2 及 BN scale/bias/mean/var、epsilon；其他系数不作数值解码。按 DEFINITION_AND_SCOPE.md 的列向量方向折叠 BN，保留末端 bias d。FLOAT 值用精确 Fraction，正平方根用 64 位 dyadic 外向区间，全部有理中间量检查 512 位上限。不得用浮点中点代替真实系数、未带证的 SVD、幂迭代或额外优化器。

每块统一计算 K、λ=1/(1+K)、α、Q(U)、Q(E)、ν、ε、ρ、L_sector、L_triangle。原系数为可靠区间时按整个区间计算，不挑有利端点。常数之间的比较是纸面定义适用性证据；没有候选输出界、ReLU 稳定性判断、反例、CERT 或得分。五块处理完且前后身份/预算均通过，才可标 diagnostic_complete；无论数值好坏都不是 candidate_pass。

## 固定资源与执行边界

默认关闭，仅 --enabled 显式执行。freeze.json 的 schema 为 d172_residual_source_geometry_v1，source_sha256 精确列出四个冻结文件的绝对路径及 SHA256。唯一输出目录为 results/d172_residual_source_geometry_20261004_v1，排他创建，一经创建即消耗此版本。所有 cache、临时文件与证据仅写入该新目录，禁止写旧目录或历史 bytecode。

沿用轻量来源审计的 CPU 0、一个数值线程、CUDA 隐藏、地址空间 16 GiB、外部 60 秒及内部 58 秒工作/后置截止。内部截止用于留下失败回执；外部截止仍有效，不能因写证据而延长。RSS 高水位增量加 65,536 字节、tracemalloc peak 加 tracer metadata 及 65,536 字节分别不超过 1 GiB；两者不是完整候选物理内存资格。

同时保留既有来源计费界：whole_work≤256,000,000、branch_work≤200,000,000、evidence_prepaid_work=40,000,000、retained_entry_cap=64,000,000、rational_bit_cap=512。证据预付计入 whole，不能从报告中再扣除。文件哈希、解析、所有区间运算、范数归约、结构遍历、序列化和保留条目均要明确计费；稀疏/密集布局的未计成本不能假定为零。

原输入与完整源码身份在首次第三方导入前认证，结束时重查。依赖版本、Python、CPU/线程环境、墙钟、内存观察、计费和完整/部分终态均自动留档。任何未知值、坏输入、身份漂移、异常、资源或时间失败均保留部分结果并使诊断不通过；不触发第二条算法、不同参数或同版本重试。

## 不变的完整目标与资格

这不是 helper 验证路径，也没有生产接入；新候选仍须默认关闭并依次通过原数学人口、真实同结构、shadow、逐家族及完整同路径回放。no model forward、no solver、no GPU、no benchmark solve；native_HZ_admitted、source_component_qualified、actual_model_verification_qualified、complete_physical_qualification 均为 false。系数诊断完成不更改这些字段。

正式 baseline 1870/2413=1063 CERT+807 validated ADV，13 家族保旧；独立 E0 CIFAR100 25、TinyImageNet 36，共61/400，不相加。本次 formal_gain=0。历史模型、结果、冻结源码及现存生产 dirty worktree 只读，不 commit/push、默认不变。完整 Goal 保持 active。
