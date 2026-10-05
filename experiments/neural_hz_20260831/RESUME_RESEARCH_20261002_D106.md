# Attention 共享源联合约束研究续接

最新为 D106 纸面候选：在保留原非凸 HZ、原 bits、EQ/LE、共享输入和 decoder 的基础上，研究同源概率乘积与 Softmax 联合关系。定义见 [候选定义](definition_first_20260928/d106_attention_source_energy_20261002/DEFINITION.md)，此前完整目标和历史成果入口仍见 [D105 总交接](RESUME_RESEARCH_20261002_D105.md)。

新的证据是 [三 token 双通道控制](definition_first_20260928/d106_attention_source_energy_20261002/CONTROL.md)：一个假点通过 simplex、全部逐产品 McCormick、列守恒、精确概率图、最紧坐标 Taylor 余项和真实输出像，却违反源单调性行 1/50；该信息可生成一个真正不稳定后继 ReLU 的更紧界，控制差 1/100。不是完整联合余项图、精确产品或全部 IQC 的分离。

单调性、共享乘积及 RLT 本身已有先例，不记为新发明；新域创新和真实收益尚未成立。不能将当前生产 exact=False 的全部旧伪赋值都称作被保留。候选统一按认证的 Softmax 数学结构触发，没有实例菜单、split、attack、dual 或新非线性求解器。

下一步依据 [真实绑定与费用](definition_first_20260928/d106_attention_source_energy_20261002/COST_AND_BINDING.md)预注册局部来源检查：当前 fused 未显式保留概率 HZ 全部状态；真实 token/head/源支撑人口未知。必须同时支付 p、共享产品、原谓词、原 bits、终端、GPU 缓冲和见证，不以单条最终行的成本代替整套成本。不恢复 D099，不重跑冻结版本。

本轮仅纸面推导、原论文核读和代码静审；没有候选代码、freeze/RUN、模型/solver/GPU 执行或测试。最新已执行组件仍为 D098 的 3845 测试、188 文件，仅数学资格。正式 1870/2413、独立 CIFAR100 25＋TinyImageNet 36 共 61/400 不变，formal_gain=0。Goal active，尚未完成。

分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；既有 tracked binary diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。本轮只写新隔离档案，不改生产、历史模型、日志或成绩，不 commit/push。文档技能用于分开已证比较、先行研究、未实现事项和不可扩大的声明。
