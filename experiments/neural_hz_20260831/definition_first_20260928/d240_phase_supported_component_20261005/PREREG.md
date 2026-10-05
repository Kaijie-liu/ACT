# 相位作用范围组件的唯一数学验证

本预注册覆盖 CONTRACT.md 的默认关闭数学组件，不覆盖真实模型或 native HZ 绑定。六个新源为 CONTRACT.md、PREREG.md、factor.py、test_factor.py、run_math.py、collection_contract.py；在任何候选 import、AST、compile、collection 或数值执行前冻结 SHA256。允许冻结前静态审阅，不执行试跑。冻结版本只运行一次；失败保留原状，不修改或重跑。

## 固定新增测试

新增 test_factor.py 的16个普通测试，函数顺序如下，不用参数化隐藏测试人口：

1. test_01_default_off_and_identity
2. test_02_integer_graph_extension
3. test_03_symmetric_certificate
4. test_04_asymmetric_certificate
5. test_05_strong_prefix_witnesses
6. test_06_amplitude_anchor_contract
7. test_07_complete_residual_payment
8. test_08_unsupported_guard_rejected
9. test_09_original_zero_labels
10. test_10_retained_source_predicates
11. test_11_difference_only_alias
12. test_12_exact_arithmetic_rejection
13. test_13_sticky_resource_limits
14. test_14_shared_state_immutable
15. test_15_registered_parameter_family
16. test_16_complete_record

具体有限输入、真图状态、分数比较点和固定参数由冻结测试源码确定，不运行搜索或事后挑例。原 signed bits、完整源读出、r/e 偏置与剩余系数、原 EQ/LE、输入重构和辅助产品唯一扩展分别核验。测试中的有理网格只检查已证明的数学性质，不作为候选采样或验证算法。

关键精度测试必须独立以非负有理权重相加编译的有限行，得到物理证书 F<=51/25，从而排除所有辅助变量取值，不以“某组猜测产品不满足”冒充严格分离。非对称控制 F=25641/12500 的两个完整单子门前缀和联合 child-over-convexified-parent 见证、全部原标签按 D239 核对。后继阈值41/20的数学差不记作网络 CERT/ADV。

X 采用统一21LE合同，包含两条 T 容量和一条 U<=D，因此对称与非对称主控制均支付76个增量nnz。Q不偷用没有声明的X产品。完整组件包含原源图和新增行的真实列、行、nnz另行记录。错误frame、非法数值、unsupported guard、原零标签、完整余项、输入谓词与sticky资源拒绝均不得跳过。

## 完整继承与隔离证据

继承 D231 的全部4153 tests/221 files、有序nodeids、7615来源身份、14输入及完整依赖/旧证据闭包。加入新文件后必须4169 tests/222 files。实际总来源身份数由最终manifest记录。历史成功receipt只是来源门，全部继承测试必须重新执行，不删、skip、替换或降低人口。

D112、D207、D208、D209、D214、D229、D230的独立writer重定位与D228内联Path.open窄facade原样保留。新增D231 test_birth.py第52行_record_file，只迁其summary.json到本RUN/inherited_d231_controls；原函数、字节码、常量、断言和数学输入不变。其他继承证据由原ACTIVE_COMPONENT_RUN/TMPDIR约定写入本独占RUN。D231按自身positive数学flags验收，不误套D230负诊断语义。

重跑D228/D231旧数学测试不授权候选使用其几何算法。D240只复用D228预算、精确算术、Row工具；候选不存在Bank、点池、geometry、support调用或新solver。测试监督器的身份/证据工具不是新的能力路径，其全部时间和来源仍计账。

## 唯一入口与原资源门

唯一输出目录为 experiments/neural_hz_20260831/results/d240_phase_supported_component_20261005_v1，首次独占创建即消费版本。唯一运行命令为

```text
/data1/Kane/miniconda3/bin/python -B experiments/neural_hz_20260831/definition_first_20260928/d240_phase_supported_component_20261005/run_math.py --enabled
```

CPU0、单线程、CUDA隐藏、AS16GiB。包含导入、collection、执行及JUnit的完整pytest阶段仍为60秒；监督器观测1GiB、reserve65536。候选的同一终身Budget仍work256M、单公共操作200M、entries64M、算术512位，不用新对象重置旧支出。测试各独立拒绝案例可创建其声明的新预算，但不得将其当生产重复重试。

监督器独占保存preregistered.json、inventory.json、tests.log、tests.xml、summary.json、exit.json及全部继承新证据。错误时保存已生成工件和失败状态，不自动重跑。后验核对完整有序population、JUnit所有项无failure/error/skip、源码/输入前后SHA256、工作区provenance以及封存工件。若观察调用超时，只轮询原进程，不重新启动。

## 资格与 provenance

schema=d240_phase_supported_component_v1，required_tests=4169，required_test_files=222。mathematical_stage_only、fixed_component_lp_controls_registered、domain_definition_changed、new_component_solver_free为true；domain_definition_changed只表示本次新增关系语义和组件，不表示新集合类或新域原创性。negative_audit_only、new_set_class、solver_rescue_registered、worker_stage_registered为false。

source_component_qualified、actual_model_binding_qualified、actual_phase_column_binding_verified、native_HZ_admitted、complete_physical_qualification、gpu_computation_completed、new_domain_qualified、new_capability_qualified均为false。没有真实模型前向、额外求解器、GPU或回放阶段。formal_gain=independent_e0_gain=new_benchmark_solves=0。

分支redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked binary diff SHA256应保持29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。仅在本新隔离目录及本RUN写入，不改生产、默认配置、旧模型/测试/结果，不commit/push。

正式1870/2413和独立CIFAR10025+TinyImageNet36=61/400不变，也不声称新路径已实测保住旧解。即使数学门通过，仍须真实同结构绑定、shadow、逐家族、完整2413和独立400回放、四并发不回退及零无效ADV。完整Goal保持active，不以本组件成功宣告完成。
