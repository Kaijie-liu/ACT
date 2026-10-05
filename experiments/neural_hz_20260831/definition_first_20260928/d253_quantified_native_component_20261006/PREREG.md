# 定量守卫原生组件的执行预注册

日期2026-10-06 Australia/Sydney；分支redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，原tracked binary diff SHA256为29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。

唯一新RUN为experiments/neural_hz_20260831/results/d253_quantified_native_component_20261006_v1。只在本源码目录中静态编写，freeze.json完成并核验前禁止候选import、AST/compile、pytest collection、数值运算或模型执行。所有源与依赖身份在运行前认证。失败也完整保留源码、清单、日志和实际生成证据，不原地修补重跑。

冻结并复核后唯一一次执行命令：

~~~text
/data1/Kane/miniconda3/bin/python -B experiments/neural_hz_20260831/definition_first_20260928/d253_quantified_native_component_20261006/run_math.py --enabled
~~~

观察超时只轮询同一进程，不重新启动。RUN独占创建即消耗本版本，成功或失败都不复用该目录。

完整继承D249有序4225 tests/225 files，包括全部旧writer证据隔离；追加本候选16个plain test函数、一个文件，总计4241 tests/226 files。具体名称冻结于run_math.py及freeze.json。单次pytest、同进程collection gate在任何测试执行前认证整个有序人口，零失败/错误/skip且完整通过才取得数学组件资格。不可先跑子集试错，不改变旧人口或测试断言。

数学阶段固定CPU0、单线程、CUDA隐藏，地址空间16GiB，完整pytest wall time<=60秒。保持原256M/200M work、40M证据预付、64M entries、512-bit以及1GiB host观察门，summary reserve65536。监督器与pytest内存观测的不同范围须如实报告；不将监督器峰值当成完整模型/GPU资格。

预注册覆盖：默认off；旧guard内精确行等价；guard外普通dyadic强对照；完整带源单child前缀及convex-parent joint-child见证；实际存储行非负证书；两个不同事件的上下尾和非对称界；原位与全部合法零标签延拓；decoder/旧谓词/完整宽度；整批失败；非零compact误差、秩与非法依赖拒绝；舍入RHS；身份污染/资源失败；完整局部账及固定普通终端LP对照；最终资格与独立记账。

主dyadic结构取a=7/8、b=1/16、epsilon=3/32、tau=1、lambda=3/8，mixed范围[-31/32,33/32]，eta_minus=0、eta_plus=1/32。新有限证书目标F<=265/128；强参照见证u=127/128、A/C各半，F=8509/4096>265/128。后继阈值1061/512位于两者之间。本结构由D252的同一通用公式作有理数推导，不从试跑结果挑选；它不是声称属于D252固定b和epsilon的那一条单参数族。普通终端仅在预注册固定正控、固定方向调用两次LP作新旧对照；主要证据是独立精确存储行证书，而非solver成功状态。

旧D249数学源码、结果和D252证明只读；全部新证据写入唯一新RUN，旧writer只重定向目的地、原测试代码不改。没有实际模型worker或GPU任务，不转授三来源、online、完整物理、新域或新能力资格。新组件自身不调用solver；测试普通终端边界不扩张成救援算法。

正式baseline1870/2413=1063 CERT+807 validated ADV，独立E0 CIFAR10025+TinyImageNet36=61/400，各账新增固定为0。只有后续同一路径完整回放满足全部用户条件才可更新成绩或默认。固定三个真实来源及所有后续门保持不变；本次组件不能代表整体Goal完成。
