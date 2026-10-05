# 原生谓词绑定组件验证结果

唯一预注册运行通过 3805 项数学组件测试，覆盖 178 个测试文件。此次新增的四项检查使用实际 SparseHZono 和原生 ReLU 算子构造的定向状态，验证共同源参考界能够安全绑定到实际存储的 EQ/LE 与接收读出。这是接入组件资格，不是新域定义完成、真实模型资格或正式提分。

## 实际执行

运行目录为 [d064_native_predicate_binding_20261001_v1](../../results/d064_native_predicate_binding_20261001_v1)。执行命令、冻结来源与依赖见 [PREREG.md](PREREG.md)、[freeze.json](freeze.json) 和运行目录的 preregistered.json。版本只执行一次，没有失败重试或修改冻结文件。

- 完整 collection 和 execution 共用 60 秒，实际 57.27382577210665 秒。
- pytest 报告 3805 passed、13 warnings，用时 44.16 秒；无失败、错误或跳过。
- 监督器总耗时 65.45038856007159 秒，包含门外的来源认证与存档工作，不能把这个总耗时混同于 60 秒测试门。
- source_drift 和 input_drift 均为空，provenance_drift=false。
- 测试使用单 CPU、单库线程、CUDA 不可见及原 AS 限制；没有启动 worker、LP/MILP、模型或 GPU。

监督器 tracemalloc 峰值 14327404 字节，metadata 4830992 字节，另保留 65536 字节 summary reserve。初始 VmRSS 24522752 字节，末尾 VmRSS/VmHWM 49278976 字节。退出记录另报 rss_highwater_growth_bytes=0；不将该计数解释成实际进程没有内存增长。此处内存检查只覆盖监督器规定的口径，不是 pytest、模型、HZ 与设备的完整物理资格。

## 得到与尚未得到的证据

本轮通过普通和 compact 两种实际存储模板的提取、完整 latent 残差、向外 RHS 舍入、非法输入拒绝、默认关闭及旧状态保留检查。原连续因子、全部原相位、旧 EQ/LE、frame 和 exact 标志保留；新增行的健全性证明见 [THEORY.md](THEORY.md)。浮点行与原模型实数公式之间的差异通过残差保守处理，不修改旧模型或历史结论。

exit.json 的 mathematical_component_gate_passed、component_tests_passed 和 native_predicate_fixture_exercised 为 true。actual_model_binding_qualified、actual_phase_column_binding_verified、native_HZ_admitted、source_census_qualified、gpu_computation_completed、complete_physical_qualification 均仍为 false。all_stages_passed=true 仅指本次登记的数学组件阶段，不授予未运行阶段资格。

既有 D063 数学成功记录以及 D057、D047 的 source census 失败原因原样继承。没有重启这些失败版本，没有缩减测试人口、改变来源人口或放宽预算。

下一研究步骤仍须证明与实际模型端口、稳定项、共享别名及全部消费者的对应，说明关系如何被下一 ReLU 消费，并验证完整端到端成本。原 HZ 加同样有效行具有相同逻辑强度，不能以本组件通过宣布新的抽象域或 PLDI 级创新。

## 基线和保存

2026-10-01，分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。正式基线仍为 1870/2413（1063 CERT 加 807 validated ADV）；独立 E0 为 CIFAR100 25、TinyImageNet 36，共 61/400，不能相加。formal_gain=0；未进行保旧或全量回放，候选默认关闭，Goal 保持 active。

运行器已自动保存日志、JUnit、冻结人口、来源摘要与 exit.json。本文件在运行结束后作为新的结果说明补充保存，不修改任何冻结输入、源码、历史证据或生产文件；没有 commit/push。使用 write-page 技能区分实际测试、语义结论和未验证收益，交付前读回本地文本。
