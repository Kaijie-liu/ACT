# 续接：目标是强大的 Neural-HZ 自身

用户再次明确主线：从 HZ 数学定义出发，提出适合神经网络验证的强大非凸 Neural-HZ，而非 helper、存储优化或框架扩张。权威目标仍为 [定义优先目标](GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md) 与 active goal。正式 1870/2413、独立 E0 61/400 均未增加；保持逐家族零回退及全部既有解的要求。

最新完成 [D150 结果](definition_first_20260928/d150_stable_phase_charts_20261004/RESULTS.md)：稳定门精确化、共同仿射范数和精确余项投影。原连续源、二元相位、EQ/LE、共享身份和输入 decoder 保留。原两零标签保留。相同父域下的新关系包含于 D149 加稳定等式的关系；跨层全面支配、强参照优势及论文新颖性没有建立。稳定门处理本身不算创新。

唯一冻结运行已完成且不可重跑或改动：4000/4000 测试、210 文件，组合阶段 48.272818114608526 秒；前后源、输入、provenance 无漂移。新基准解和正式收益为 0，来源/model/GPU/native/完整物理资格均 false。退出记录和全部 12 工件哈希复核通过；相关来源哈希也已复核。没有本轮后台运行。

下一步只针对普通 Conv→ReLU→混权 Conv＋活 skip→ReLU 做强参照、同预算、完整费用比较，不再以局部检查数量代替能力进展。已有 packet 的实际范围与缺失参数见 [REAL_NEXT.md](definition_first_20260928/d150_stable_phase_charts_20261004/REAL_NEXT.md)。需保全全局共享源和所有空间位置；当前小型稠密 API 不是可扩展实现。若增益只对弱对照成立，则记录本结构能力增益为零并改变域假设，不添加救援路径。

GPU 仍无成功计算资格：D017 初始化失败，D020 ptrace 在 CUDA worker 前失败，D043 cuInit 仍报告 CUDA_ERROR_OUT_OF_MEMORY；不能声称已定位 AS cap 为根因。没有新的 GPU 运行或权限/资源边界改变。smooth、Transformer、完整 CNN 与正式回放仍需工作。

分支 redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5 未变。所有新工作仅在隔离实验目录，旧源和结果只读。研究目标保持 active；本轮仅数学组件取得进展。
