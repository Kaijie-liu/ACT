# Neural-HZ 定义主线的研究交接

用户再次明确：重点是提出强大的 Neural-HZ。现有目标已经规定从 HZ 的
数学定义出发，不必重建目标，更不能把辅助关系或存储优化当作目标完成。
服务端目标保持 active；没有修改权限、验证限制或默认路径。

本轮[四门共同关系记录](definition_first_20260928/d141_rectangular_relation_support_20261004/RESEARCH_RECORD.md)
保存了一个严格超过全部六对完整 source-labelled hull 的纸面控制、
原相位保全、缺陷补偿以及进入下一原 ReLU 的有限有效界。其基础是已知
increasing differences，项目 D046 已有该引理；packet 上界和网格预算
多数也由成对关系蕴含。它有支撑价值，但仍是原 HZ 加有效关系，未形成
新的统一非凸域。没有启动 cuts/helper 实现或候选实验。

下一份核心成果应回答：Neural-HZ 的一个元素如何经过普通
Affine/Conv -> ReLU -> live residual 统一得到下一个元素；原相位、
连续幅值与共享源之间的耦合如何被直接使用，而不是只留给终端整数求解。
同时给出具体化、原 HZ 对应关系、精确和近似的适用范围以及完整成本。
不要求不可实现的全局最佳抽象或处处精确，不再只枚举还能加哪些不等式。
GPU 服务于这些域操作；smooth 和 Transformer 仍是待覆盖目标。

D140 的正控已被旧 D052 更便宜地覆盖，不能重新启动为新能力。
旧表示/构造成果、全部失败与证明档案保持只读，不删除，也不重复记收益。

正式成绩不变：1870/2413，1063 CERT 加 807 validated ADV；独立
CIFAR100 25、TinyImageNet 36，共 61/400。无新正式解，无全量晋级。
保住每个旧解及 13 家族、保留二元非凸性与 shared latent、fail closed、
禁止 attack/PGD/BaB/split/backward/dual rescue 的边界不变。
任何新增数值候选仍须预注册和冻结，不减少 D136 的 3965 tests、208 files
及继承预算。本轮仅纸面与只读核查，无数值、模型、solver、GPU 或后台实验。

分支 redu-hz；HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；
原 tracked diff SHA256
29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。
只新增隔离存档，未 commit/push。目标尚未完成。
