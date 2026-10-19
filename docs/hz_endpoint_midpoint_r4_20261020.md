# 端点对偶交换的 R4 有限算法对照

2026-10-20 用户明确授权：重新冻结并一次执行原七位置 CPU 批次，随后独立审计归档。
起始提交 `d3362aa89c3795795b7ab8276b57c97ba5c6a7e7`，分支 `feat/moe-route-verification`，工作树干净。
本轮研究问题是端点对偶交换是否带来**经精确检查的下界或完整义务增量**，不是增加工程控制数量。

## 唯一算法因素与保留条件

继承 [R3 的研究问题、七位置顺序和停止规则](hz_endpoint_midpoint_r3_20261020.md)。
双端点以同 pair、同性质的 best-dual 固定中点替换最后一次梯度更新；保持 129 次评价、
128 次 forward、候选宽 8、检查宽 2、proposed／zero 两角色及全部义务。
不修改 HZ 外包络、来源、原系数精确检查、数值接受门、位数或候选搜索比例。

顺序仍为：22 项组件控制，guarded 原搜索，guarded 中点，mixed 中点，mixed 原搜索，
中点接缝非合作阻塞，部分候选发布故障。half 只用于组件控制。
新批次内重新核对完整控制，不借用旧 actor 的结果补在线接收；不另跑已完成的 pidfd／观察器控制。

每位置至多 300 秒，整批包括初始化和收尾共 2100 秒；一次准入至多 60 秒并计费。
8M Work、128 MiB 制品、每进程 8 GiB AS、2 GiB 采样 RSS 及全部检查保持。
意外失败立即停止后缀，不重试、换来源、加时、提额；不运行真实模型或 GPU。

## 新执行身份

目录：`/data1/Kane/MOE/runs/hz_endpoint_midpoint_20261020_r4`。
[冻结](hz_endpoint_midpoint_freeze_r4_20261020.json) SHA-256：
`f34f87f1ecbe46e0c4d3cff90460d60171dfa5f8e6337629bdd7b6f2f496cf4a`。
[静态复核](hz_endpoint_midpoint_freeze_r4_review_20261020.json)绑定当前代码、既有三份来源与环境。

相对 R3，batch 依赖仅改变已通过四组控制的 `owned_session.py` 和 launcher／caller 的
R4 路径常量；native60、六份 profile、组件目录、host 文件、资源政策和预算逐值保持。
不新增监督层。[R3 失败](hz_endpoint_midpoint_r3_results_20261020.md)和
[观察器修复结果](hz_owned_observer_results_20261020.md)原样保留。
本冻结只登记一个新 hash，尝试后消费，不恢复旧批准。

## 科学结果与成本

四个正常请求按同 pair／property 比较规范有理数下界，报告提高／不变／降低、正性质
gain／loss、全部义务是否闭合。已保存的是端点聚合后的性质下界；没有保存的逐端点细节不虚构。
位置 PASS 只表示完整执行接收，组件 PASS 不是算法有效性结论。

成本用每位置含收据接收的完整耗时、最终累计 Work、全部制品和采样内存；累计 Work
不把 production、replay 和 reception 的前缀反复相加。宿主预置、请求计时、首行导出、
离线审计分栏。更早冷启动及第二成本行／最终退出未精确测量时仍为 unknown，不填零。

若没有精确下界／完整义务增量，或只有浮点 surrogate 改善，则停止该中点机制，不追加
比例、迭代、来源或样本。保留全部损失和成本，不事后发明原协议没有的成本倍数门。
局部精确提升最多支持有限合成信号；mixed 的专家相同，不能单独证明真实 MoE 优势。
本批不关闭 G1–G6、完整八专家容量、外部领先、实机 GPU 或真实同源路由变化证据。
