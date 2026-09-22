# 六篇作者基线：实际执行、同对象比较及剩余阻塞

更新于2026-09-22（悉尼）。本表替代旧状态页中的“仍在训练／尚未认证”描述；
原失败、冻结结果和论文报告值不修改。**六行状态齐全，不等于六篇复现及六组公平成绩已完成。**
设计与论文实验细节仍见[研究记录](recent_moe_baselines_20260921.md)和
[A/B/C分表协议](recent_moe_comparison_protocol_v1.md)。

## 当前可引用的实际结果

| 工作 | 本地真实执行／独立检查 | 论文规模作者复现 | ACT同对象公平比较 |
|---|---|---|---|
| Dual RS |90epoch selector落地；绑定最终权重后，原序输入0、1均正确，统计L2半径0.8304935215、1.1060614554；43.719s含postflight|未完成全测试认证曲线；这是命名的log-domain数值兼容变体|尚无同函数比赛；原生RS证明平滑函数，不是ACT确定性F，不能横比SAFE百分比|
| MetaMoE |原生PyTorch作者后端绕过ONNX后，CIFAR/MNIST组件均正；完整20类smoke中，两臂均复放CIFAR0误分类，作者臂MNIST0正|尚未复跑全部router/expert表；旧ONNX不等价失败保留|R2正式执行门未过：ACT在router首层ReLU丢失共享HZ，触发64M容量限制。未冻结／启动新20输入比较|
| Robust Experts |两臂200epoch/12600更新＋全10k终评完成，独立存盘复核通过；dense clean/PGD/APGD=37.27/18.99/18.74%，ConvMoE=22.18/12.88/11.70%|完成两配置的论文规模训练，未复现全部论文表或其MoE优势；另查明源码entropy目标不同于正文公式，保留命名兼容变体|有原始语义接口及解析全历史控制，无已训练全尺寸全域成绩；不能用经验RA或编译通过填认证栏|
| RoME |4输入×3范数已全部终止并独立审计：5完成、7超时；完成中3条是已有clean错误、2条是同一输入的Linf/L1扰动误分类|不是全10k RA；s4/b6等代码默认与论文身份差异明确保留|原始连续多层LoRA混合仍不在ACT完整全域入口范围；不得替换成离散top-k|
| J-TLAT |指定公开作者仓库本次仍只有README；论文称有匿名补充代码，尚未取得可执行包|未完成；不是“复现得零分”|未运行；不能用自编同名方法冒充作者工具|
| Feature Noise |已读作者实验设计；v2论文与作者组页面仍未定位完整实验制品|未完成；TEAL/MiniMind链接只是依赖，不是本文完整实现|特征／文本噪声任务不等于像素L∞盒；尚无同对象协议或成绩|

## Dual RS：本次请求第1项已完成

- 最终epoch090权重SHA256：`5bcf3fd0ad3b28cab728fa3e56c77f3300cae572f864ff8b3d78bcf342ad07cc`。
- 90epoch／33,300更新；35.692%的训练遥测是sigma-label accuracy，不是CIFAR分类准确率。
- 两输入、每执行阶段N0=100/N=10,000、alpha=.0005；两阶段合成每输入失败概率上界.001。
- 结果等级`PROBABILISTIC_RS_NATIVE_NUMERICAL`；不升级成确定性SAFE，也不以2/2估计总体CRA。
- [训练落地](dual_rs_training_landing_archive_20260922.json)、
  [真实认证与计数审计](dual_rs_certification_archive_20260922_r1.json)。

## MetaMoE：MNIST作者入口已通，ACT规模阻塞不能藏在部署成功里

R3取消BN折叠仍有MNIST探针差1.6701e-4，高于原1e-4门，原失败保留。
R4改走固定作者αβ-CROWN的native PyTorch入口，保留eval BN及原float32组件，
两控制正结果且移除wrapper的有限探针差为0。这是明确标注的前端适配，不是证明旧ONNX等价。
R4原始terminal误沿用了旧parser的ONNX等级字符串；绑定backend.yaml实际为native
`Customized`入口。已追加[元数据更正](metamoe_native_grade_erratum_20260922.json)，
不覆盖原文件或改变数值等级；完整模型两臂入口本来就使用单独的numerical-filter标签。

完整模型比较另用显式float64快照、相同物化归一化盒、全20类性质、300s总预算。
作者臂包含route-invariance、所选score非零和未选类别零块义务，非只查选中专家分类。
ACT R1的函数式ReLU错误已由独立R2等价拼写适配修复；R2仍无法保留完整router HZ。
90s内、无求解器查询的诊断指出第一层ReLU20480输出触发`sparse_relu_size_limit`。
**这说明当前ACT配置无法完成该对象的关系表示，不说明性质不安全或下界为负。**

后续先做单独的容量／存储策略控制，再冻结新的同对象执行身份；不在现有结果上
调高上限、扩大预算或选另一个容易输入。不通过正式门就不消耗新20输入集合。

证据：[native控制](metamoe_native_control_review_20260922_r4.json)、
[完整R2终态](metamoe_paired_smoke_review_20260922_r2.json)、
[原模型反例复放](metamoe_paired_replay_20260922_r2.json)、
[容量诊断](metamoe_hz_intake_diagnostic_20260922_r1.json)。

## Robust Experts：完整训练与独立落地复核已完成

最新结果、完整曲线、论文—代码—执行对照与证据边界见
[落地报告](robust_experts_landing_20260922_r1/README.md)。两臂原序全10k评估完成，
独立重读498冻结输入文件、400epoch checkpoint哈希、最终权重/optimizer、全轨迹与评估
汇总后通过，48.060秒；不是独立重跑全部攻击。原执行总26,208.253秒。
本次ConvMoE未超过dense；不改epoch200选模，不把最高validation替换为终评。
尤其需要保留目标身份：论文的−H(batch mean)对应源码`column_entropy`，本次冻结的
源码`entropy`还含mean H(sample)。不能把“跟随源码”与“精确复现论文公式”混写。
Top2有效STE=false；原始源码自动关闭，非本轮改动。

以下为已完成部署与启动过程的历史说明：

| GPU控制 | 实际批量／训练攻击 | 含启动及存盘耗时 | 峰值Torch allocated | 存盘状态／动量 |
|---|---|---:|---:|---:|
| dense ResNet18 |640／PGD7|10.0845s|3.017GiB|122张量／62缓冲|
| ConvMoE E4/k2/layer4 |640／PGD7|12.3486s|4.165GiB|147张量／86缓冲|

实际执行在Blackwell依赖overlay中，CUDA基底只读且安装前后未变；act-py312未装包。
R1元数据callable序列化错误、R2临时socket路径错误（人工停止自有失败组）都保留；
R3仅修复这些部署问题，不更改batch、PGD、optimizer或模型。不是择优训练重跑。
GPU一步存盘不等于GPU精确续训；另有CPU两epoch fresh-process续训及下一批增强数据完全一致。

科学配方：CIFAR100，dense与ConvMoE，固定seed12345、200epochs、lr.01、batch640、
native SGD/PolyLR/augmentation/PGD7，finalepoch200唯一模型选择。
新增完成：长训练外层监督、完整epoch checkpoint/RNG绑定、终评与预算内存盘审计，
8项截止／异常／身份控制；R2两个架构的完整短流程分别20.402／27.464s通过。
原R1共用CSV造成训练曲线覆盖已保留；R2仅隔离目录，两份曲线均核验存在。
正式执行另行冻结24h/架构、串行、最终200epoch唯一模型选择，无隐式重试／GPU精确续训声明。
当时剩余的实际完整训练、全10k终评和最终归档**现均已完成**；不能再列为尚未启动。
**没有把短控制checkpoint当成训练完成权重。**
正式启动记录绑定执行commit `acd27d19f`、config SHA256
`7731f7ae144b3816d28b0dde7ee59dfcb4fac90f2c73b6e47cd36ceeca67954d`。
已观察dense真实首epoch完成63个更新；这不是最终成绩，终态和各epoch以服务器新目录为准。
监督流程不自动Git写入；本次已另行独立归档，不回填或覆盖原始记录。
证据：[CPU续训](robust_experts_resume_archive_20260922_r1.json)、
[GPU独立存盘检查](robust_experts_gpu_step_archive_20260922_r3.json)、
[完整短流程](robust_experts_pipeline_archive_20260922_r2.json)、
[正式启动记录](robust_experts_paper_training_launch_20260922_r1.json)、
[监督与长运行协议](robust_experts_training_supervision_20260922.md)、
[科学配方](../configs/recent_moe/robust_experts_paper_training_recipe_r1.json)。

## RoME：冻结三范数执行及独立复放已归档

固定seed0索引6044、2890、9399、1917，每项CPU2线程／600s，Linf8/255、L1=12、L2=.5。
全部12条保留；5条完成、7条TIMEOUT。总执行4252.323s，含postflight4252.397s，
另行独立复放审计11.107s；不能把7个未完成攻击计作鲁棒，也没有自动扩到100输入。

| 输入 | Linf | L1 | L2 | 可解释的结论 |
|---:|---|---|---|---|
|6044|完成，clean错误|完成，clean错误|完成，clean错误|本来label7/pred4；零扰动，不是三个新攻击|
|2890|完成，5→7|完成，5→7|超时|两个范数有复放的经验扰动误分类；不是两个不同输入|
|9399|超时|超时|超时|没有完整攻击结论|
|1917|超时|超时|超时|没有完整攻击结论|

2890的Linf距离0.03137257695比精确8/255约大2.79e-8，符合冻结1e-7数值攻击容差，
但不能升级为精确盒UNSAFE；L1距离11.9999999069在原半径内。这里也没有形式SAFE。
原始union标签`ATTACK_FOUND`包含clean错误，保留原字段，并用派生解释区分原因。
声明的攻击列表是AutoAttack配置，不代表每个请求均实际执行完所有攻击。

证据：[完整终态与独立复放](rome_multinorm_archive_20260922_r1.json)、
[clean错误／新增扰动／未完成分解](rome_multinorm_interpretation_20260922_r1.json)。
下一阶段若改用GPU或扩大样本，必须另行冻结，不能修改这批CPU结果。

## 缺失制品的最新核查与停止边界

2026-09-22重新读取[J-TLAT作者仓库](https://github.com/DeepSota/J-TLAT)：
master列出的文件仍仅README。[作者OpenReview论文](https://openreview.net/pdf?id=8voly42rKo)
说明匿名代码作为补充材料提供；本次公开论坛遇浏览器验证，notes API返回403。
因此状态应为“公开仓库不完整、补充制品尚未取得”，不能断言作者从未提供代码。
不绕过访问控制，不自动联系作者；PI可提供合法下载的补充包。

Feature Noise的最新[arXiv v2](https://arxiv.org/html/2601.14792v2)（2026-06-09）
和[作者组论文页](https://relationalml.github.io/publications/)经核查，尚未定位其完整
实验实现。文中的TEAL、MiniMind和NLPAug是组件来源，不能当成已复现本文全部实验。
这不是“证明不存在代码”；后续若取得制品再绑定版本、数据和任务。

没有相同函数／输入域／性质／预算时，“全部对应ACT公平成绩”只能保留为未完成项，
不能将RS半径、PGD准确率、CROWN数值过滤和HZ接受结果塞进一列排行。
