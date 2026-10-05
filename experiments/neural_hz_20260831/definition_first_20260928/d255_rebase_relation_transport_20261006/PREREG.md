# 重基底关系运输的执行预注册

分支redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。日期2026-10-06 Australia/Sydney；tracked binary diff SHA256必须保持29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。本候选为默认关闭的同H精确读出运输，不是新域或实际网络能力声明。

唯一新RUN为experiments/neural_hz_20260831/results/d255_rebase_relation_transport_20261006_v1。六源为CONTRACT.md、PREREG.md、native_rebased.py、test_rebased.py、run_math.py、collection_contract.py。freeze.json完成并核验前禁止候选import、AST/compile、pytest collection、数值试跑或模型执行。只允许静态文本编写及独立审阅。

唯一命令为

~~~text
/data1/Kane/miniconda3/bin/python -B experiments/neural_hz_20260831/definition_first_20260928/d255_rebase_relation_transport_20261006/run_math.py --enabled
~~~

RUN独占创建即消耗此版本。观察超时只轮询同一会话，失败保留所有证据、不修改冻结源重跑。D253收集失败、D254成功以及全部旧源/结果只读。

## 人口与资源

完整继承最后成功D254的有序4241项、226个文件及全部证据写入隔离；新增固定16个plain test函数、一个文件，共4257项、227个文件。名称和顺序冻结于runner及freeze。单个pytest进程，同进程collection在任何测试执行前认证全部人口；零失败/错误/skip且执行后身份不漂移才通过。没有子集试跑、挑选窗口、删旧测试或延长时限。

CPU0单线程、CUDA隐藏、AS16GiB、完整pytest墙钟60秒。保留原1GiB host观察门、65536 summary reserve、256M/200M work、40M evidence预付、64M entries和512-bit。监督器观测不代表pytest/GPU完整物理资格。普通正控共用Budget；预声明的非法/资源负控可以使用较小独立Budget以检验sticky rejection，不调整默认门来过测。

## 固定覆盖与数值对照

测试以实际sparse_hz_rebase_image_exact完成两父输出重基底，再在保留其完整谓词的H上追加两个child，检验D254旧eta提取拒绝的tau=0与新规则恢复。主结构固定沿用D254的a=7/8、b=1/16、epsilon=3/32、tau=1、lambda=3/8，不依据数值运行选择参数。

完整覆盖：默认off与空bindings保留旧关系；真实rebase内容凭据；主关系恢复；原整数点/全部零标签/decoder；普通非零stored偏差−2^-55；多epoch；原binary来源和无pivot常量link；全部旧谓词/输出/全局高水位；行重排后的内容重绑；外来/篡改凭据拒绝；隐含private依赖拒绝；预算与位长sticky失败；向外舍入和旧terminal；实际存储行证书及完整成本；共享预算整批原子性；最终资格分账。

固定精确证书目标仍为归一化读出F<=265/128，再由实际link EQ拉回完整显示读出。只有第14项调用两次固定方向的普通终端LP，不增加第三条求解路径或依据结果选规则，不要求较粗精确上界必然可达。主要数学证据是存储行与EQ组合，浮点LP仅辅助对照；不把分数点称作ADV。

局部展开硬上限由实现的预付及静态审计覆盖，小型测试检查预算增加、受限budget及位长失败；不为触碰65536边界专门扩大极端fixture，也不声称已经数值跑过那个边界。

## 真实来源及不可转授范围

本次不启动真实模型worker/GPU，不将构造rebase输入等同于三个完整来源。rebase_native_transport_passed初始false，只能在全门后置true；旧quantified_native_transport_passed仅保留历史收据，其资格不转授到新run。所有实际模型/online/GPU/完整物理/new_domain/new_capability资格仍false，domain_definition_changed=false，所有gain=0。

并行只读审查确认：后续CPU完整前缀捕获并不以strace/CUDA成功为先决条件，但普通tf.apply没有完整work计费，现有cache账本也不含全部模型/图/allocator。新的真实worker必须先冻结固定三来源、实际传播费用和全部生命周期成本；不能因库调用快就免除原费用。这里不授权重跑已经超预算的direct-tap或稠密reader组合。

正式1870/2413（1063 CERT+807 validated ADV），独立CIFAR10025+TinyImageNet36=61/400不相加。原全部晋级门和完整Neural-HZ/GPU/smooth/Transformer/新家族目标保持，组件通过不结束Goal。
