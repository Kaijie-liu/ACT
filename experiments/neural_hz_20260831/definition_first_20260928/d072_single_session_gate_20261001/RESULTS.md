# 共同负项候选通过完整数学组件检查

本次隔离执行成功：原封不动的 D070 数学候选及全部 3813 项测试、180 个文件，在同一个 pytest 进程中完成收集核验、执行与 JUnit 收尾。授予的是本执行配置下的数学组件资格，不是新抽象域、真实模型能力或生产准入。

## 完整执行证据

唯一命令为 `/data1/Kane/miniconda3/bin/python -B experiments/neural_hz_20260831/definition_first_20260928/d072_single_session_gate_20261001/run_math.py --enabled`。唯一 RUN 为 `results/d072_single_session_gate_20261001_v1`，会话 21198 已返回 exit_code=0，没有本次遗留的运行会话。

三份新执行文件经根代理通读和独立静态审查后冻结。冻结前只作 AST、哈希和源码检查，没有导入、编译、收集或预跑候选。数学候选及四项新测试仍使用 D070 原路径、原哈希。新契约不缩减人口、不调序、不删检查、不放宽预算，也不重跑已消费的旧版本。

- pytest 报告 3813 passed、13 warnings、45.65 秒。警告记录保存在 tests.log，未隐藏或改为跳过。
- 父监督器计得收集加执行完整子进程 46.97283185273409 秒，低于原 60 秒上限；包含身份认证和证据收尾的总时间为 55.59134741872549 秒。
- 根代理独立核验 JUnit：3813 个唯一测试，零 failure、error、skipped；与冻结 manifest 逐项一致。inventory 的有序人口与 manifest 完全相同，3813 项、180 文件，manifest SHA 一致，执行前核验标志为 true。
- 原四项新测试均有完成记录：支持与稀疏更新 0.002 秒、共同负项前向控制 0.002 秒、混权区间参数与宽层支配 0.019 秒、身份及默认关闭和限额 0.001 秒。这些是测试用时，不是实际验证器性能。
- 6732 个 source 身份、9 个 input 身份、4417 个 GPU 依赖及 1011 个 decoder 依赖纳入认证。source_drift=[]、input_drift=[]、provenance_drift=false。GPU 依赖认证不等于执行 GPU。
- 监督器双主机观察门通过：tracemalloc 峰值 18294319 字节，metadata 8223728 字节；加原 reserve 后仍在原上限内。这不授予候选或整个 pytest 的完整物理存储资格。
- 根代理重算并核对全部七项 artifact SHA，均一致。exit.json SHA 为 `65825e670fd591fb54b6bb9c923c5c04d4926f5b8a39df335fd3324905214fed`。

CPU 单核、库单线程、CUDA 不可见、AS16GiB 及全部原资源门未改变。只有重复启动与收集被移除，不把这一变化报告为 Neural HZ 提速。

## 旧失败和当前资格分开保留

D070 v1 的完整数学门仍为 TimeoutExpired、FAILED。其 exit.json SHA 仍为 `2211f568b60801bc98b259b924c92972bcc8a096fc69f7bafdfae1a569c504cf`；本次 manifest 和 exit 明确嵌入该失败，不将旧进度点数补写为成功。D066、D064 的既有数学资格，以及 D047、D057 的普查失败，均保持各自原范围。

本次 component_tests_passed、mathematical_component_gate_passed、joint_negative_fixture_exercised 为 true。actual_model_binding_qualified、actual_phase_column_binding_verified、native_HZ_admitted、gpu_computation_completed、source_census_qualified、complete_physical_qualification 全为 false。没有执行真实网络性质、shadow 或全量回放。

下一有效实验应评价真实归档上的种子界改善，并区别稳定事实兼容状态与不可能状态；不能重新用已证无双未定锚的归档追逐共享 delta 额外收益。适配、实际模型绑定及完整费用仍须独立证明，不再为相同数学候选扩张测试框架。

2026-10-01，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。正式 1870/2413（1063 CERT 和 807 validated ADV）、独立 CIFAR100 25 加 TinyImageNet 36 即 61/400 不变，formal_gain=0。生产代码、旧档和远端不改。

本轮属于 progress：完成此前缺失的完整数学组件验证。write-page 技能用于分开本次成功、旧失败和未取得的资格；文档读回核对，仅本地存档。整体 Goal 保持 active，定义创新、GPU 和正式增益仍未完成。
