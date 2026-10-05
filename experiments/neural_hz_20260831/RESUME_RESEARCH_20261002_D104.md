# 共同源径向 Neural-HZ 研究续接

最新材料为 [D104 候选定义](definition_first_20260928/d104_radial_source_relations_20261002/DEFINITION.md)、[支持定理与严格控制](definition_first_20260928/d104_radial_source_relations_20261002/THEORY.md)、[先行研究及决定](definition_first_20260928/d104_radial_source_relations_20261002/PRIOR_ART_AND_DECISION.md)和[完整费用及状态](definition_first_20260928/d104_radial_source_relations_20261002/COST_AND_STATUS.md)。完整 Goal、限制及旧工作索引沿用 [D103 归档入口](archive_checkpoint_20261001_d103/README.md)。

本轮从数学定义添加同源径向关系：原 HZ／bits／EQ/LE／输入 decoder 不丢弃，LN 的整组输出共享一个受范数约束的 rho。共同仿射分母被确认是已有改写，不当新域；“输出像精确”也不等于“原输入—输出共同图精确”，不能据此简化掉残差关联。

正结果是统一的前向支持行。一个普通三维控制严格超出“共享原源一阶展开＋最紧逐坐标余项＋精确输出像＋未耦合共同分母”的明确参考，并将 7/200 分离传给两相位均真实出现的后继 ReLU。不是所有 Taylor／CPZ 方法的下界或实际模型成绩。

下一步针对真实非零中心 LN／残差，做同事实强对照和完整代价判据；旧 ViT 未解结构以 Softmax、动态 MatMul 和 ReLU 为主，不能把 LN 控制直接当成其适用证据。Softmax／GELU 和 GPU 完整支持仍未完成，原目标不缩减。

无新代码、freeze、RUN、数值、GPU 或回放。D098 仍是最新组件执行，3845 测试／188 文件。正式 1870/2413 及独立 61/400 不变，formal_gain=0，Goal active。文档仅本地归档，旧档不改写，未 commit／push。
