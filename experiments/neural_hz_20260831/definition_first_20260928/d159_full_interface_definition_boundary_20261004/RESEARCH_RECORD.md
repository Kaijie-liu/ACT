# Neural HZ 完整接口研究记录

2026-10-04 Australia/Sydney；分支 redu-hz；HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。

用户最新要求是“重点是提出强大的 Neural-HZ”。紧接该要求的上一轮主要核对目标和重述方向，归类 no progress，而不是已完成的研究推进。本轮没有沿用这个状态停留，完成了[完整接口与点值能量的定义审查](THEORY.md)，并据证据否决两项直接工程推进。

## 本轮完成与决定

1. 从旧 D120 JSON 静读三模型的 first_shape、branches 的 shape/stride/padding 和 side_consumers；核对 D149 来源边界、D156 定义、D157 合同及 D158 证明。没有重新运行模型或读取新的模型参数。
2. 确定 D157 的字面完整接口在三个首块上都比原门数更宽。large 的 identity-q skip 还使原生域及相应有限 caps 恢复旧门图。其结果不能作为直接部署该候选的依据。
3. 区分 medium/Tiny 的投影 shortcut，证明其结构秩上界严格低于原门数，并构造不遗漏消费者的 C0=[B_main;S;ones]。证明裸原生接口细化包含关系；不把较少接口行等同于新域、保精度或真实收益。
4. 补全精确源能量变体的核球、椭球、单点条件和点值支持。查重确认 D156 已有损失、D158 已有同相位凸化反例，因此不把它命名成新候选，不实施或重跑。
5. 核读三项一手文献的有限相关范围，分开既有数学、项目内推导与未证新颖性。没有引入 SDP/QP/SOCP、backward/dual rescue、相位枚举或新的求解路径。

这些工作提供了改变下一研究动作的证据，归类 progress；不是“创新 Neural-HZ 已完成”。真实验证能力仍没有新增。

## 执行边界

本轮配置为 paper_only，candidate_execution=false、model_execution=false、gpu_execution=false、shadow=false、replay=false。仅使用静态文本/旧JSON读取、git 状态与哈希、主文献浏览和新文档写入。没有候选 import、AST、编译、pytest 收集、数值实验或 source worker，因而没有新实验 prereg 或伪造的 run manifest。

最近完整数学资格仍属于冻结 D158 的4032项测试/212文件，不重跑、不减少，也不授予本轮纸面变体。后续一旦实施，仍须新预注册、源码冻结、继承完整人口和原资源门；旧资格不因复制接口而转移。

所有新文件均在本隔离目录及新的根级 resume，旧模型、表格、日志、实验源码、失败结果和生产文件没有编辑。输入哈希记录在 ANCHOR_SHA256SUMS；最终档案哈希另记 ARCHIVE_SHA256SUMS。D158 的三项档案校验已通过。

## 独立核对和证据等级

current_helper_audit 核对完整消费者、形状、秩上界、接口分解与行数；aligned_domain_equivalence 独立核对核球几何、单点条件、接口包含和旧反例继承；fiber_kernel_review 核对点值支持与固定相位线性外包的范围。主代理逐项复核并整合。它们是纸面审查，不是机器证明或外部同行审稿。

文档归档技能用于保持来源范围清楚、区分推导与实验，并将记录写入既有本地研究树；没有创建外部 Page。检查的是保存的 Markdown 文本与链接目标，未声称外部页面渲染资格。

## 分数与完整目标

正式 baseline 1870/2413 = 1063 CERT + 807 validated ADV；全部旧解及13家族逐家族保全义务不变。独立E0为 CIFAR100 25、TinyImageNet 36，共61/400，不与1870相加。本轮新增正式解0、外部新增解0，没有新见证或无效ADV计账。

目标继续 active。定义创新、真实同结构收益、GPU、smooth/Transformer、新家族及最终单一路径完整回放都没有完成。没有设置更小替代目标，也没有把研究困难归为需用户解除的阻塞。
