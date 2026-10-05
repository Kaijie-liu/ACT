# Neural-HZ joint forward support continuation

目标仍是定义优先的强大非凸 Neural-HZ，包含实际家族能力提升和 GPU、smooth/Transformer 目标；未完成、未阻塞。用户最新强调不能转向 helper、存储优化或测试框架扩张。上一回合 D157 分类为 progress；本回合也取得了可复核的算子进展及定义反证。

本回合完成 [D158 结果](definition_first_20260928/d158_joint_forward_support_20261004/RESULTS.md)。保留 D157 原生域，增加统一的域内联合前向支撑公式。旧 norm 与新 joint 证书每次都计算，标量上界取较紧者，不按失败择路，不逐项混合相位系数。新算子没有 LP/QP、网络 backward rescue 或其他外置验证器。

关键正结果不再是手填组合权重：直接调用 bounds 得到下一预激活上界 -3/55，再构造下一 ReLU，整域输出 bounds=(0,0)。原 17 个 bits 全保留。非均匀偏置、混合符号相位系数、共享 skip 和两 bank 传播亦通过。它是 D157 算子的补全，不是新的域定义或论文级新颖性结论。

完整唯一冻结运行通过 4032 项测试、212 文件，保留全部 4020 项旧测试。合并测试时间 49.092110965400934 秒，小于原 60 秒；13 项旧警告、零失败/错误/跳过，source/input drift 空且 provenance 不变。[运行目录](results/d158_joint_forward_support_20261004_v1)保存清单、JUnit、日志、退出和哈希。没有剩余作业，已消耗版本不得编辑或重跑。

必须保留负证：旧三门 source-binding 假成员仍被 native 域接纳，新查询只能对它 sound，不能谎称消除了损失。新研究的 kernel contraction（dhat=Pd+theta*Nd）在 native d=0 上可修复假点，但两个严格同相位内部成员按 5/9、4/9 混合就恢复该假点；即使完整 fixed-phase 凸包也丢修复。该定义只归档，不实施。THEORY 还记录固定线性源商的精确性条件，以及“凸关系包含完整 affine graph，在一个 relint 源点 fiber 精确则全域精确”的小定理及其严格范围，不是所有 Neural-HZ 的不可能性声明。

下一轮不要再反复扩 Householder 控制、测试包装或尝试仅在一个源点修补凸相位域。集中处理普通混权/跨层结构的有效精度与预算增长、真实完整消费者/源绑定以及全端到端成本。任何真实模型同结构尝试仍先单独冻结合同与配置；本次数学成功不授予真实模型、GPU 或全量资格。

正式 baseline 仍 1870/2413 = 1063 CERT + 807 validated ADV，新增正式解 0。独立 E0 为 CIFAR100 25、TinyImageNet 36，共 61/400，不与 1870 相加。所有原禁令、逐家族保旧和全 2413/独立 400 回放要求不变。本回合没有实际网络、GPU、shadow、replay 或生产集成。

后继数学组件继承完整 4032 项/212 文件，不删除替换；先预注册并冻结再执行。当前 manifest 的 source_count=7327、input_count=14、CPU=[0]。6 冻结源为 THEORY.md、PREREG.md、joint_support.py、test_joint_support.py、run_math.py、collection_contract.py，schema=d158_joint_forward_support_v1。

分支 redu-hz；HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5 前后相同。freeze SHA256 b053962da9b59a48d0cdff6587ecc02c5f77300bdf362f315c1cfae63a6525a3；manifest SHA256 64cd535e4ae10bfaeb8204d21841f991781aa981e2aeddb359a87ca8050103d9；inventory SHA256 7526f53a18410f3108e384c78b5c9226a430262d7bce4cb8cfe24c361e4573bb；exit SHA256 1001c78199c0df081576d2630630ca1d6c36fb2e43515c33d20cd6fdd5e92b23。历史模型与存档仍只读，所有新增文件均在隔离实验树内。
