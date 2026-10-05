# 原生共享幅值变换的一次性组件测试

本版本只申请默认关闭的原生共享上行组件资格，不申请新域新颖性、真实模型、GPU或完整物理资格。数学范围、舍入证明和未覆盖部分见 CONTRACT.md。

## 冻结人口与配置

继承D081成功的全部3817有序node IDs及181文件，追加 test_native_shared_transfer.py 的四项固定测试：test_shared_native_positive_and_projection、test_full_residual_and_binary_preservation、test_binding_rejection_and_default_off、test_outward_rows_and_cost_accounting。总3821项、182文件，旧项不筛选、改序、删除或跳过。

第一项检查原生产算子生成的两父混权后继、完整有理赋值扩展、零点双标签，以及旧LP点被联合行排除3/16的纸面控制。第二项检查完整原连续及二元残余、旧谓词和输出、frame/exact和decoder列前缀保留。第三项检查禁用时不读输入、假slot/phase、非零compact图误差及非法参数拒绝。第四项检查非二进制有理斜率/系数的向外补偿、精确行与存储行关系及全部矩阵buffer/nnz计数。

唯一RUN为 experiments/neural_hz_20260831/results/d086_native_shared_transfer_20261001_v1，--enabled 首次创建即消费。先冻结 CONTRACT.md、PREREG.md、native_shared_transfer.py、test_native_shared_transfer.py、run_math.py、collection_contract.py 六文件及指定4测试名，再允许候选AST、导入、收集、执行。没有单测预跑；失败完整保留，不修后复用该RUN。

一次pytest进程，执行测试体前检查完整有序人口与路径，结束核对JUnit，零skip/error/failure。CPU1、库单线程、CUDA不可见、AS16GiB。启动、收集、执行及JUnit收尾合计60秒不变。监督器RSS增量加65536 reserve，及tracemalloc峰值加metadata加reserve分别不超过1GiB，保持旧作用范围，不冒充对子进程完整内存认证。

继承D081全部source/input、4417 GPU依赖身份、1011 decoder依赖身份及historical_contracts，认证其manifest/exit/inventory/artifacts。仅调用原辅助函数，不运行旧main/worker，不修改旧模块globals。执行前后source/input/production零漂移；缓存、日志、临时与结果仅写新RUN。

## 本轮不开放的阶段

没有真实网络执行、模型selector资格、GPU、LP/MILP、三源worker、shadow或2413/400回放。原256M whole、200M branch、40M evidence、64M retained、512位与240秒worker门保持原范围，本轮不运行worker，不能据组件通过声称这些完整费用已过门。

模型身份、具体输入decoder来源与 native HZ 模型级接入继续标为未认证。测试使用实际生产算子但仅合成小块；不能称为CIFAR/Tiny收益。formal_gain=0，基线1870/2413和独立61/400不变，默认不启用。

2026-10-01，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。本目录仅新增文件，历史结果与冻结源码只读。write-page技能用于区分证明、执行证据与未获资格，不发布外部Page。
