# 联合激活组件的完整数学预注册

唯一候选是本目录 bank.py 的精确两父两后继联合关系。CONTRACT.md、PREREG.md、bank.py、test_bank.py、run_math.py、collection_contract.py 必须在任何候选 import、AST、compile、pytest collection 或数值执行之前写入 freeze.json 的 SHA256 清单。默认关闭。首次冻结执行失败后不修改或重跑该版本，不覆盖任何旧档。

## 固定人口与检查

从最后成功 D214 继承全部4116项、217文件及其完整有序 nodeids、7473源码身份、14输入身份、来源/资源检查。追加以下16个无参无装饰器 plain tests，得到4132项/218文件。在同一 pytest 进程 collection 完成后认证全部人口，再进入 test loop；不选择性重跑或减少旧人口。

1. test_01_default_off_and_foreign_frame
2. test_02_spec_types_and_bounds_fail_closed
3. test_03_no_residual_matches_independent_phase_cells
4. test_04_one_residual_matches_independent_phase_cells
5. test_05_two_residuals_match_independent_phase_cells
6. test_06_interior_zero_intersection_retained
7. test_07_every_vertex_is_true_graph
8. test_08_signed_rows_and_zero_labels
9. test_09_integer_phase_compatible_mixture_is_true_graph
10. test_10_physical_and_phase_support
11. test_11_shared_bank_control_and_d062_reference
12. test_12_source_binding_without_vertex_source_atoms
13. test_13_constant_residual_and_dependent_zero_planes
14. test_14_compiled_counts_and_nnz
15. test_15_shared_budget_and_bit_limit
16. test_16_asymmetric_permutations_and_summary

第三至第五项以独立完整原相位 cell 的线性顶点枚举作精确参考，分别覆盖0、1、2余项。参考仅在测试侧，不向候选提供点池、界或执行选择；这不授权候选运行 phase 子问题。用有理消元认证完整顶点集合，不能只采样几点估计完备性。参考也是本轮冻结测试源码的一部分。

定向检查还包括：父真边而非凸包弦；D227 面内部 (11/20,7/20) 必须存在；全部顶点为实际图点；原 signed bits、八个标签行和零标签自由度；同一整数相位中的凸组合仍是真图；物理及原bit读出的支持与达到点；D227控制和D062同样给零界的事实；原输入极点有局部 lambda 扩展却不能强行逐顶点条件原子绑定；常量余项物理坐标不丢、相关零式处理；稀疏行/列/nnz上界；持久累计预算和512位拒绝；普通不对称混权及置换。

最后一项独占创建本 RUN 的 summary.json，保存前15项及本项实际检查记录。记录候选点数、真实顶点数、稀疏行费用、支持值、source extension 和预算观测；测试侧记录不能被当成原生全网物理计费。任何前项失败都不能伪造完整 summary 或通过数学门。

## 来源与隔离

唯一 RUN 为 experiments/neural_hz_20260831/results/d228_joint_bank_component_20261005_v1，必须从不存在开始。唯一入口为 /data1/Kane/miniconda3/bin/python -B 本目录/run_math.py --enabled。运行自动保存 preregistered.json、inventory.json、tests.log、tests.xml、summary.json 及 exit.json；失败仍保存已产生的输出和原因。不会后台重试。

旧数学证据重定向仍只允许模块局部 writer/目录绑定：D112四测试RUN、D207第44行_record三文件、D208第39行_record两文件、D209第35行_record三文件，追加 D214第24行_record四文件到新RUN/inherited_d214_controls。验证原源hash、函数身份和目的地，不更改测试逻辑、数学计算、断言或候选。所有历史目的地只读。

D214完整成功 receipt、六源、freeze 和 D227理论/归档作为只读锚。新全部 source/input 并集在执行前后核验；生产branch/commit/diff、解释器、原项目导入闭包和依赖人口照旧认证。仅有新6源hash不足以通过。

## 资源与门

CPU0，全部线程1，CUDA隐藏，AS16GiB；完整 pytest 导入、collection、测试和JUnit在60秒内完成，监督器仍执行原1GiB和65536 reserve检查。继承所有原工作/条目/证明费用，不借本组件降低旧ledger。新 Budget 的累计256M work、200M单操作work、64M entries、512位按CONTRACT执行。

summary、完整 ordered collection、JUnit全部4132无fail/error/skip及资源/漂移检查共同决定数学门。单个新测试通过、统计上的总数一致、退出码零或旧receipt均不能单独通过。任何失败数学资格为false，不能事后放宽60秒或改人口。

schema=d228_joint_bank_component_v1；required_tests=4132；required_test_files=218；new_evidence_files=[summary.json]；mathematical_stage_only=true；worker_stage_registered=false；fixed_component_lp_controls_registered=true（仅继承原测试对照）；new_component_solver_free=true；solver_rescue_registered=false；negative_audit_only=false；domain_definition_changed=true（算子实现，不是新集合类）；new_set_class=false。native、原模型绑定、完整物理、GPU、新域完成和新能力资格均false，formal_gain=0。

## 之后的真实目标不被缩小

已有 CIFAR100-medium ReLU2/port121、channel44 相邻父门及 Conv3/BN4 全128通道消费者来源。最小完整双bank为14400→8192、原3072输入且保全部shortcut；这只是后续来源锚，本轮不解码/执行真实模型、不挑有利channel报告收益。指定child的原phase column/frame、完整576-slot余项、可靠界与decoder仍须认证。

数学通过并不证明真实净收益。下一阶段按结构统一规则认证和评估完整来源；强参照、source正确性及费用失败均保留负结论。之后的shadow、逐家族、全2413和独立400门不变，正式1870与E0 61不相加，任何本轮测试都不更新它们。文档技能仅用于新本地预注册与结果分离，不发布外部Page。
