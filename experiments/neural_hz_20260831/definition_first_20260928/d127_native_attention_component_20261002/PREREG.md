# 原生 Attention 查询的单次数学实验预注册

最近完成的 D126 研究产生数学证明、强反例和图范围修正，属于 progress。随后一轮仅向用户核验状态，属于 no progress，没有新实验。D126 否决了把标量预算与差分直接当共同域的捷径。本轮并行继续 CNN 定义研究，同时完成已证明的原生 Attention 方向查询；不是用小例取代完整 Neural-HZ 或宣布 Transformer 已解决。

## 冻结和唯一执行

任何候选 AST/import/compile、数学脚本、pytest 收集或执行前，冻结八个文件：CONTRACT.md、PREREG.md、attention.py、exp_interval.py、test_attention.py、test_exp_interval.py、run_math.py、collection_contract.py。冻结前只允许编辑和静态审查新文件；不得先试跑小测试或 filtered collection。消费后不可修改或重跑本版本，失败也完整保存。

唯一 RUN 为 experiments/neural_hz_20260831/results/d127_native_attention_component_20261002_v1，独占创建即消费。命令为 /data1/Kane/miniconda3/bin/python -B 对应 run_math.py --enabled。schema 为 d127_native_attention_component_v1；freeze 指明 mathematical_stage_only=true、worker_stage_registered=false、solver_rescue_registered=false。父 manifest 使用 NEURAL_HZ_D127_MANIFEST_SHA256。

认证 D125 最新数学成功与完整人口，以及 D120 历史来源成功及其全部已继承身份、失败档案和资源限制。D125 的数学资格不转授给本候选；D120 来源资格也不转授。原输入、解释器、GPU/decoder 依赖身份和完整项目 import closure 按原合同认证。

## 全部有序测试人口

完整继承 D125 的 3881 项和 196 个文件，添加十二个无参无 decorator 的普通函数，总计 3893 项和 198 个文件：

1. test_attention.py::test_attention_joint_polygon
2. test_attention.py::test_attention_interior_stationary_control
3. test_attention.py::test_attention_strong_product_reference
4. test_attention.py::test_attention_common_score_shift
5. test_attention.py::test_attention_shared_source_scope
6. test_attention.py::test_attention_opt_in_identity_and_limits
7. test_exp_interval.py::test_exp_zero_and_known_brackets
8. test_exp_interval.py::test_exp_negative_reciprocal
9. test_exp_interval.py::test_exp_monotone_enclosures
10. test_exp_interval.py::test_exp_dyadic_rounding
11. test_exp_interval.py::test_exp_range_reduction
12. test_exp_interval.py::test_exp_fail_closed

它们验证二维联合像边界、退化像、内部驻点、同源 score/value 正控及明确定义的强参考、公共 score shift、共享源与谓词的精确性降级、原 bits/frame/System 身份、默认关闭、无认证时不发上界、位长与资源限制，以及指数区间与有向舍入。数学固定控制不是运行时采样、attack 或 phase/input split。

不得跳过、减少或重排继承项。D112 四项旧控制只有证据输出位置在认证后内存重定位到本次 inherited_d112_controls，旧测试文件、断言、输入和预算均不变。其他旧测试通过 NEURAL_HZ_ACTIVE_COMPONENT_RUN 写新 RUN，不写历史目录。

## 不变的执行门

单一 pytest 进程包括启动、收集、执行和 JUnit 的上限仍为 60 秒。CPU1、单线程、CUDA 空、AS16GiB；监督器 1GiB 观察门及 65536 字节 summary reserve。独立 tmp/cache/evidence 目录；不因为指数认证较慢提高预算或缩减人口。

只运行数学阶段，不启动新 source worker、具体模型 forward、GPU 或 solver。继承的固定组件 LP 控制保持，不新增任何求解器查询。原来源 whole work 256M、per model 200M、evidence 40M、entries 64M、rational 512-bit、worker 240秒等后续限制不变；未运行的阶段不得报告通过。

自动保留 preregistered.json、全量 inventory、tests.log、JUnit、继承控制证据和 exit.json，失败与 timeout 也保存。强乘积正控若通过，另存 native_attention_control.json，记录认证查询与原参考伪点的有理区间；不事后重跑求值。观察超时不授权重启。source/input/provenance drift、错误或遗漏、skip、数值或资源失败均使本版本数学资格失败。

## 实施后的判断

只有全人口和身份门通过才授予数学组件资格。实现必须保留一个共享 native fiber，阈值查询不能把不确定/共享源的结果误称 exact，旧模型和标签不参与选择。算法的实数证明、认证数值实现、小例强参考、实际模型、GPU、全量保旧与正式成绩分别记录。

若成功，下一阶段应针对已保存的真实 ViT 结构完成数值参数与全部同结构方向绑定，同时继续普通 CNN 的共同关系问题；不得直接默认启用或把数学控制记为新 CERT。所有失败、新实现和结果只写本次隔离目录。Goal active，正式及独立新增均为零。
