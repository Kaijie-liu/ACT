# 共同相位关系研究的证据与保管

日期 2026-09-30。分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。Goal 服务为 active；权威目标文本仍为 GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md，未更改目标或资格门。

本轮由用户要求跨 HZ 文献寻找启发，以及后续 Goal 继续指令授权。前次中断前的重复来源核对没有新增候选或结果，按 no progress 处理；更早 D053 的完整数学执行仍是该轮 progress，不重新记成本轮运行。

本轮新增证据是共同上侧耦合范围、三原门到真实物理读出的严格分离、无需等幅的至多四行合成，以及普通三门链的二次 root 接口闭包限制。这些改变下一结构假设，属于 progress；它们仍来自已知 BQP、包络与离散差机制，不构成新颖性认证或正式收益。

## 原文核验

- He 与 Tawarmalani，Tractable Relaxations of Composite Functions。出版信息核对出版社 https://pubsonline.informs.org/doi/10.1287/moor.2021.1162：2021 年 online，2022 年卷期。https://par.nsf.gov/servlets/purl/10382117 的 web 打开超时后，使用只读 curl 管道与 pdftotext 核对 §3.2 Lemma 4、Theorem 3、Corollary 6，以及 §3.3 Theorem 4。没有把搜索摘要当最终数学证据，没有下载写入旧目录。
- Boland 等，https://arxiv.org/pdf/1507.08703，核对 Theorem 4 及 §2.4；区别单侧与完整图凸包的圈符号条件。
- Gupte 等，https://www.pure.ed.ac.uk/ws/portalfiles/portal/137016756/1702.04813.pdf，核对 Theorem 3、§3 的凸组合方法及 McCormick 定义。
- 只读复核此前热带分析与互补系统论文的相应段落；不把这些已经存档的方向重复算作发现。未开展穷尽式文献检索。

## 独立数学审查

根代理推导三门控制与不等幅合成；一个独立代理手工复核六个有向双门投影、三对旧差界、单门凸包见证、物理读出上界与取等源，确认 23/171 及严格内部版 824/7695 两个缺口。另一个代理独立复核共同阈值耦合与原 guard 限制，并提供三门链闭包反例，根代理重新逐式检查。

这些是纸面复算，没有执行新候选、导入被冻结的候选、枚举程序、LP、数值搜索或单元测试。未把静态审查写为数值资格。

## 真实来源的只读定位

只读文件 results/d025_interval_capacity_20260930_v1/complete_0.json：

- result.source_forms["(0, 16, 16)"].form[1] 和 result.source_forms["(1, 16, 16)"].form[1]，source "495" 的两套区间端点均正；source "497" 的第一套均正、第二套均负。
- box["495"] 和 box["497"] 均有严格正宽度。result.windows[2].source_slots[4] 与 [13] 分别是上述两个原 first-bank 门。
- result.receivers["0"][0].weights[4] 两端点正，[13] 两端点负，因此确有下一混权 Conv 消费。这不是动态 Add 的完整证明。
- 第一个原门的 bounds 跨零，第二个严格正。故这些字段只证明结构前提存在，不证明新双门或三门关系非冗余；没有运行 D053 生成器计算新界。

根代理读回了这些具体字段、区间端点符号、窗口引用与 receiver 权重。它们是旧失败研究的 per-model 完整证据，不补成全三源通过。Tiny 没有相应 D047 complete，未从部分日志补造数据。

如果未来在 D047 原固定消费者 pair 上使用共同源规则，它们已是第二个 ReLU bank，源应是第一 bank 的实际激活值。继续传播还需第三 Conv 的真实参数；旧 extractor 只保存后续图端口，不能拿第一 bank 的参数证据冒称第二 bank 的两层消费已验证。不得按本节坐标挑选运行目标。

## 本地输入身份

```text
0fd93920413f63ad6b3b65849ea4d857c416c6bf4ab56263c63ff73416ebe80c  GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md
5ffe411a62b9f0713c5c14a756b8f698ffd018904ae572b7369ceade6eff7d83  ../d035_cross_phase_source_20260930/THEORY.md
ce51a4a3f34444aa33325d6e897431fc51ed9c39155d30163cc1c32d3fbcbff8  ../d052_static_source_certificates_20260930/THEORY.md
a11cede907f3fae5bcc8db8da87d4c148d2c2406bc0f0e96688d54ffc2ecc69b  ../d053_static_source_component_20260930/RESULTS.md
77efd4e1d774ab3ebdeafa6db4d50f3a6ad0aafc831264c430a5461ab541edd2  ../d053_static_source_component_20260930/SHA256SUMS
fbab84537df071153a7d9362161b2c9aa85b80c8b4c605ff1e1149170e0965e0  ../../results/d025_interval_capacity_20260930_v1/complete_0.json
```

目标文本路径在实验树根，其余路径相对本目录。没有新的运行配置、依赖包或模型副本；配置是 paper-only、no-candidate-import、no-model-execution、no-GPU、no-solver。

## 实际写入与未完成事项

仅新增本目录的研究、记录和校验清单，并为此前已终止 D053 运行新增 SHA256SUMS。D053 的六份冻结源码、freeze、结果及日志没有修改；17 项清单验证均通过。没有后台新实验，没有重跑 D047、D049、D053。

保留原九项 tracked dirty changes，仍为 3806 insertions、57 deletions。没有生产修改、commit 或 push；历史模型、结果和 /data1/Kane/HyZor 保持只读。

正式 1870/2413 与独立 E0 61/400 保持原记账，formal_gain=0；没有重新确认逐例旧解，也没有新的 CERT 或 validated ADV。目标远未完成，不能用本轮数学正控宣布 Neural HZ 创新、全网可组合性或 GPU 成功。

按 write-page 文档技能把来源事实、项目推导、反例范围、旧资格和未完成项分开存档。检查了本地 Markdown 内容与哈希；没有外部 Page 或渲染预览。
