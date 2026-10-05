# 残差后继来源组件：完整测试通过，真实模型资格尚未取得

冻结的 D081 默认关闭组件完成一次性测试：3817 passed，181 个测试文件，零 failure、error、skipped。它支持在普通残差后继上继续研究联合关系，不是 Neural-HZ 定义创新、实际网络能力或正式成绩提升。

## 本次唯一执行

命令为 `/data1/Kane/miniconda3/bin/python -B experiments/neural_hz_20260831/definition_first_20260928/d081_residual_source_qualification_20261001/run_math.py --enabled`。唯一结果目录为 `../../results/d081_residual_source_qualification_20261001_v1`；会话 94470 已终止，exit_code=0，没有本次遗留运行任务。原 D072 的3813项/180文件全部保留，追加4项，不缩减人口或改序。

六文件在候选导入、AST、编译、收集及执行前冻结。静审中修复监督器对 Python 可执行文件符号链接的认证，并将最终身份检查与证据封存拆为独立收尾；这些修复发生在冻结前。本次没有失败后修改重跑，也不修改或重跑旧 D079 草稿。

| 观察 | 本次结果 | 范围 |
| --- | --- | --- |
| pytest | 3817 passed，13 warnings，47.96秒 | 警告完整保留；不是性能基准 |
| 测试子进程总时间 | 49.439174519851804秒 | 含启动、收集、测试、JUnit收尾，低于原60秒门 |
| 监督器总时间 | 61.00170190818608秒 | 另含身份认证和证据收尾；不宣称该总数小于60秒 |
| 人口 | 3817项、181文件 | 执行前清单检查通过，零skip/error/failure |
| 漂移 | source=[]，input=[]，provenance=false | 含生产身份检查 |
| 主机观察 | traced_peak=14141586，metadata=4673584，reserve=65536字节；RSS高水位增量=0 | 监督器观察低于原1GiB门，不是子进程完整物理内存认证 |

13项既有警告为1项 TypedStorage 弃用和12项 record_property/xunit2 不兼容提示；没有隐藏、忽略失败或将测试标为跳过。CPU1、库单线程、CUDA不可见、AS16GiB均保持原注册配置。认证6732个历史source身份、9个input、4417个GPU依赖和1011个decoder依赖，不等于运行这些模型或GPU。

## 通过的语义范围

四项新增测试覆盖默认关闭与batch绑定；Conv/BN/投影shortcut/Add/后继Conv/ReLU完整参数顺序；全部消费者、共享常量缓存及普通预激活旁路；未知算子、外来依赖、形状/dtype、cycles和opset范围拒绝。

普通结构 `g=Conv(z), q=ReLU(g), a=q+g, r=ReLU(a)` 暴露了旧草稿将已有 q 误列为后继 ReLU 的问题。新组件验证原端口身份，将其记录为 frontier_relu_boundary，真正的 r 才是 next_relu。测试不通过拒绝该普通结构来规避错误。旧 D079 源码保持原哈希。

component_tests_passed、mathematical_component_gate_passed、source_component_qualified 为 true；all_stages_passed 仅指本注册的来源组件阶段。actual_model_binding_qualified、actual_phase_column_binding_verified、native_HZ_admitted、source_census_qualified、gpu_computation_completed、complete_physical_qualification、candidate_physical_gate_evaluated 全为 false。

本次只构造内存合成 ONNX 图，没有真实三源 worker、LP/MILP、GPU、shadow 或2413/400回放。raw bytes、protobuf、Fraction、缓存、块metadata和证据的同时存活尚未接入完整费用门，不能据本次通过启动无计费真实普查。原256M whole、200M branch、40M evidence、64M retained、512位及240秒worker门不变，既有失败保留。

## 保管与下一步

2026-10-01，branch redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。配置和依赖见 PREREG.md、freeze.json及唯一RUN的preregistered.json。exit.json SHA为 `971a56e5578a65895853d684e3f16f5358cdab9135b8708bcd7ebf06a9745964`。SHA256SUMS保护本次来源、结果和旧草稿锚，不是新候选预注册。

新工作只写本隔离目录和唯一RUN；生产dirty changes、旧档、历史成绩、Goal及远端不改。正式1870/2413（1063 CERT +807 validated ADV）和独立CIFAR100 25、TinyImageNet36即61/400不变，formal_gain=0。

跨领域数学研究另存于D082；这3817项测试不授予D082的数学实现或native资格。来源组件用于支撑真实后继块研究，不能替代非凸域定义、组合变换和完整收益证明。write-page技能用于区分已执行证据与尚未取得的资格；仅本地存档并读回，不发布外部Page。
