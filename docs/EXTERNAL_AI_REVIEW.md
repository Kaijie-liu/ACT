# ACT／HybridZ MoE：独立 AI 评审交接

用途：让没有旧聊天上下文的评审者，仅凭论文、源码和可获取证据，质疑并
核查主要主张。**这是审阅协议，不是新实验协议，也不是评审完成的证明。**

直接转发 [启动指令](EXTERNAL_AI_REVIEW_PROMPT.md)。科学材料基线为
`806443470b3eeda3e601510ea8a6fc71570f96bf`，不是随后生成交接包的提交。
固定文件清单见 [manifest](external_ai_review_manifest_20260921.json)。
本文中的数字都是待复核的历史记录，不是要求评审者认可的结论。

## 1. 访问、权限和审阅独立性

服务器目录 `/data1/Kane/MOE/ACT`；唯一工作分支
`feat/moe-route-verification`。先读 [AGENTS.md](../AGENTS.md)，再读
[最新交接](CODEX_HANDOFF.md) 顶部；后者是导航，不是原始证据。
如起始工作区不干净或分支不符，停止并报告，不 reset、不 checkout、不
清理现场。不要与实现会话同时写同一 checkout。本轮只读，因此不需要
生成提交；报告直接给 PI，落盘须另外授权。

有三种可行访问层级，必须在报告中写明：

| 访问条件 | 可以核查 | 不能声称 |
|---|---|---|
| 只有别人粘贴的摘要 | 逻辑与表述一致性，提出所需材料 | 已读源码、已核验 raw evidence |
| 有对应 Git checkout／可读仓库 | 正文、冻结配置、代码、提交 JSON；无模型表格重建 | 已重放未提交 checkpoint／大包／服务器日志 |
| 有服务器只读访问与旧 raw 包 | 再加保存输入、逐包身份、精确输入包含核对 | 因 raw 包可读就已独立证明全部网络 SAFE |

缺少材料时给出准确文件、影响哪条主张、最小补交物；不要猜路径内容、
编造运行成功或把“未访问”写成“文件不存在”。不要下载未授权权重或自动安装
依赖。不同 AI 会话可以减少作者自证偏差，但共享模型盲点仍可能存在；
不得把它标成已完成人类专家审阅。若要查最新文献／会议规则，应自行查
原论文、作者制品或官方来源；旧聊天和 Advice 文件中的引文也需要核验。

## 2. 阅读次序：先形成自己的理解，再核对作者账目

### 第一遍：论文主线

先读 [阅读入口](../paper/README.md)、[摘要](../paper/abstract.md)、
[引言](../paper/sections/01_introduction.md)、
[方法](../paper/sections/03_path_conditioned_method.md)、
[可靠性](../paper/sections/05_soundness_engineering.md)、
[评估](../paper/sections/08_evaluation.md)、
[局限](../paper/sections/09_discussion_threats.md)。
先用自己的话写明目标、对象、前提、贡献和你认为最关键的未闭合环节。
不要把 proof-checking 基础设施的完成数当作核心研究成果。

### 第二遍：当前主张最敏感的证据

1. [主表来源适用性报告](main_table_source_applicability_20260921.md) 及
   [逐盒／逐增益 ledger](main_table_source_applicability_20260921.json)。
   审查其推理，而不是只看 `PASS`；它的总状态明确为包含性存在缺口。
2. [主表](../paper/results/main_tables.md) 与下表的原始提交汇总。
3. [旧条件化正证明](portable_conv_proof_v1_results.md)、
   [新完整来源结果](full_source_v1_results.md)、
   [最终有限诊断与停止决定](property_diagnosis_v1_results.md)。
4. [制品入口](../paper/artifact_quickstart.md)：区分重建表、运行新请求、
   检查旧包；私有／未发布的大型制品不是已经交付给审稿人。

### 第三遍：定向追源码，不从所有历史日志顺序读起

| 待核查问题 | 定向入口 |
|---|---|
| 合法 top-k、并列、候选完整性 | `act/back_end/moe/schema.py`、`hz_routing.py`、`route_a.py` |
| Tier 1、F0 和终态 | `act/pipeline/moe/staged_verifier.py`、`act/back_end/moe/weighted_top2.py` |
| 共享／私有因子、加权联合模型 | `act/back_end/moe/weighted_topk.py`、`monolithic_f0.py` |
| 作用域复用、共同事实、公平调度 | `act/pipeline/moe/scoped_f0_proofs.py`、`route_complexity_schedule.py` |
| 输入盒→HZ 与版本 | `scripts/audit_moe_main_source.py`、`scripts/review_moe_main_source.py`；按其冻结 Git 路径追入口与 `solver_hz.py` |
| 独立 LP 下界推理 | `act/back_end/solver/lp_certificate.py`、`act/pipeline/moe/check_request_lp.py` |
| 新来源检查是否自证 | `router_source/checker.py`、`full_source/check_lift.py`、`full_source/check_obligations.py` |
| 总预算、完整分母、统计单位 | `act/pipeline/moe/schedule_confirmation.py` 及冻结100输入配置／选择清单 |

检查器不能仅调用被检查构造器再认定两者相同。尤其沿“源对象→构造→范围
证据→下界→完整聚合”追至少一个实例，列出没有独立检查的箭头。
数学文件在 `act/back_end/moe/proofs/`：先审 normalized top-k 分解、
conditional support 单调性，再按需审 η 和 Lagrangian 充分归约。
不要把后两个封存适配器当成当前主要性能贡献。

## 3. 主要实证定位表

完整路径与 SHA-256 已固定在 manifest；大 JSON 可按 key 定向解析，不必把
整文件倾倒进上下文。以下不是不同证据等级可直接混算的总榜。

| 材料 | 待复核记录 | 优先追问 |
|---|---|---|
| `schedule_confirmation_100_review_20260914_r1.json` | 三固定模型×100输入×3臂；SAFE179/156/141，主比较23 gains、0 loss | 是政策接受结果还是某个网络域的证明？跨模型相关性、净增和成本是否正确？ |
| `relation_ablation_review_20260914_r1.json` | 10个已观察输入×3模型；共享侧新增3 SAFE | 哪些是完成松弛后的差异，哪些仍有求解预算混杂？ |
| `external_pair_comparison_review_20260914_r1.json` | ACT11政策 SAFE，ACT前端＋plain CROWN13数值正过滤 | 8交集／3 ACT-only／5 CROWN-only；便宜外部路径是否限制实用性主张？ |
| `conv_full_v2_review_20260915.json` | 卷积30输入：两ACT臂0 SAFE；CROWN1正过滤 | 第二家族适配完成是否被误说成跨家族正证书完成？ |
| `docs/general_evidence_execution_v1_results.json` | 新20输入证据模式没有完整正请求 | 搬迁、缓存、控制通过是否被误说成覆盖提升？ |
| `docs/portable_conv_proof_v1_review.json` | 旧 supplied-HZ input98条件化9/9正 | 上游专家传播等仍信任什么？这些矩阵有没有与新来源混用？ |
| `docs/full_source_v1_review.json`、`docs/property_ranges_v1_review.json` | 新来源链更完整；新输出仍非正／缺失 | 同一来源上有没有真正闭合完整正证明？ |
| `docs/main_table_source_applicability_20260921.json` | 100/100请求盒包含失败，98/100重建HZ内缩，含全部23 gains | 仅是输入公式的精确反例还是全网不安全？对论文措辞该作何处置？ |

前四行文件在 `act/pipeline/moe/results/`。已保存 raw 主表目录为
`data/moe/results/schedule_confirmation_100_full_20260912_r1`；Git 本身不包含
全部 raw 包、checkpoint 和数据。所有23 gains 的包／请求／模型／性质／
配置身份和接受来源均在 source-applicability ledger 的 `gain_records`。

## 4. 需要独立裁决的六组问题

### A. 保证对象和数学前提

- 请求实数盒、物化端点盒、浮点生成HZ、声明网络图、原生浮点程序分别是什么？
  哪些包含关系已证，哪些只是配置中的声明？不要只写“有数值误差”。
- `exact=True` 是否被当成网络到集合的包含证明？允许所有tie-legal路径
  的理想语义，与根据未核验router转换排除路径的实现之间是否存在缺口？
- 对原始网络而言是否完整覆盖？Tier1凸组合、pair域包含、F0外松弛、
  性质复用在哪些前提下成立？失败能否被错误升级成 SAFE／UNSAFE？

### B. 输入缺口对正主张的影响（最高优先级）

- 自行判断 source-applicability 审查是否正确，包括精确 `2/255` 与
  binary64 epsilon 的区别、截止到输入重构的范围、半径删列阈值。
- 误差微小不等于无害；也不能仅从一个输入内缩推导模型不安全或所有SAFE
  结论必假。局部检查没有恢复旧的完整传播轨迹。
- 共享同一缺陷能否支持工程比较？**不能因此取消健全性义务。** 请明确
  哪些结果只能保留为观测行为，哪些“新增证书”表述应撤回，是否仍存在
  未加限定的标题、摘要、表名或结论。不要求认可作者当前的措辞方案。

### C. 证据链与独立性

- 结构审计、求解器状态检查、精确对偶检查、源转换检查分别检查了什么？
- 旧正证明和新来源检查绑定的输入域、矩阵、因子、gate范围、性质一致吗？
  不一致时，禁止把两条链的成功部分拼成一条证书。
- 新源码里的检查者是否独立重算必要关系，还是复用同一可能错误的函数？
- `PASS` 是否只是制品自洽？记录点是否已精确可行？非正下界是否被误解为
  LP不可能认证或网络不安全？重复解析优化不能替代这个判断。

### D. 经验比较与归因

- 三个seed是同家族固定训练运行，不是三架构；300 model–input pairs
  不是300独立图像；100输入也不是整个测试集 certified accuracy。
- 样本／配置是否先冻结？30与100是否分开？预算是否包括失败、前处理、
  跨环境开销？超时与缺失是否保留？任何对手是否免费或被削弱？
- 23 gains 的来源、两条 legacy-only SAFE、旧monolithic覆盖更好的实验
  是否都可见？不要只审作者赢的表。
- 调度对照不等于关系消融；记录了reuse不等于每条增益都因reuse而产生。
  CROWN数值过滤与HZ-policy接受不是同一种形式保证。

### E. 新颖性与论文可成立的最小范围

审查 MoE合法路由、共享因子、性质导向weighted义务、作用域复用与完整
聚合的组合是否有可辩护贡献，不把通用工具本身叫首创。若进行文献审阅，
从正式论文和作者制品核对 MetaMoE、proof sharing/incremental verification、
constraint-aware verification、source semantics 相关工作；列出确实读到的
来源与任务差异。不要把模型攻击准确率、router认证率、竞赛积分直接当作
当前任务的对照，也不要复述未经核实的“首个”“SOTA”或会议信息。

### F. 可复现性与投稿门槛

区分获取现成结果表、重跑真实请求、独立检查真实包三条路径。指出缺的
数据、权重、statement identity、许可证、环境和干净安装证据；不要把
no-download控制代替真实实证artifact。最终把问题分成“必须修”“可准确
声明的限制”“可延后”，而不是要求解决通用MoE、整个CUDA栈后才能投稿。
不能以“要冲A”为理由保证录用，或自动要求继续追input98正例。

## 5. 允许的只读检查与禁止的动作

在服务器已有环境下，以下命令不训练、不做网络前向或求解，也不修改历史
结果。设置 `PYTHONDONTWRITEBYTECODE` 避免额外字节码；其他机器使用已有
适用Python，不为本轮审阅擅自安装依赖。

```sh
git status --short --branch
git rev-parse HEAD
PYTHONDONTWRITEBYTECODE=1 /data1/Kane/miniconda3/envs/act-py312/bin/python -I -S scripts/rebuild_moe_main_tables.py --check
PYTHONDONTWRITEBYTECODE=1 /data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/test_review_handoff.py
PYTHONDONTWRITEBYTECODE=1 /data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/test_moe_main_source.py
PYTHONDONTWRITEBYTECODE=1 /data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/test_manuscript_source_contract.py
PYTHONDONTWRITEBYTECODE=1 /data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/test_rebuild_moe_main_tables.py
```

**仅当原739包可读时**，可以追加保存数据检查：

```sh
PYTHONDONTWRITEBYTECODE=1 /data1/Kane/miniconda3/envs/act-py312/bin/python scripts/audit_moe_main_source.py --check
```

最后一条只解码旧 `request.pt`、查Git冻结版本并做精确输入算术，仍不证明
全部输出。无raw包则标“未执行”，不要为了通过测试生成替代包。别运行不带
`--check` 的生成入口，也别把所有名称含 `audit`／`review` 的旧脚本都当成
只读无求解工具：有些会加载模型、复放反例或写输出。

本轮禁止：新 solver／前向／传播、加时扩样、阈值修改、依赖安装、训练、
自动优化修复、holdout重开、input98重试、外部投稿／联系／发布。
建议实验可以写出来，但必须说明假设、资源、验收和停止条件，等待PI另行决定。

## 6. 评审输出格式

先列实际HEAD、与固定材料版本的关系、访问层级、读过的主要文件、执行的
命令／返回值、缺失材料。没有执行的操作不要用“复现／复验通过”描述。

然后按以下结构交付：

1. **独立摘要**：用你自己的话说明问题、方法、贡献与当前最大风险。
2. **问题清单**，每条使用下列字段：

   `编号 | 严重度 | 已确认/疑点/未核验 | 文件:行号或JSON key |
   观察与推理 | 影响的主张 | 最小修正 | 是否需新计算`

   严重度建议：S0健全性／证据链或主要结论失效；S1重大过度主张／比较
   缺口；S2可复现性或有界工程问题；S3措辞／组织。不要用级别代替证据。
3. **主张—保证矩阵**：主张原文、实际证据等级、仍信任的环节、
   保留／收窄／撤回／材料不足，以及建议替换表述。
4. **论文判断**：现有范围能否形成有贡献的稿件？必须解决的前三项是什么？
   可明确声明的限制有哪些？不能给虚构录用概率。
5. **下一步最多三项**：每项注明目的、为什么已有数据支持、最小交付、
   通过／停止条件和是否需PI授权。若仅需改稿／补材料，请明确说不需要新实验。
6. **审阅局限**：AI会话独立性、不可用raw制品、未独立检查的数学／数值环节。

理想的反馈是能够被作者逐项复查、接受或以反证回应的有限清单；不是另一份
越来越长的路线图，不以多写checker或多跑样本代替关键科学判断。
