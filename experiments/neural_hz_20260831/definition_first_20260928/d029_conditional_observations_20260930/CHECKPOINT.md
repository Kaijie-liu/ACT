# 条件观察研究检查点

上一轮为 progress：D028 补上双门支持的构造前提，排除单门替换路线并完成隔离存档。本轮开始核验 D028 两个文件 hash 均通过，当前分支仍为 redu-hz。本轮也为 progress：完成了会改变实现选择的投影定理、混合读出及区间误差构造，以及两个有明确范围的否决结论。

[数学记录](THEORY.md)给出六条线性提升约束到四条行的精确投影，保留原 gates、全部 bits、源谓词与 decoder，并明确分数相位下只是线性系统投影。共同源混合读出的上下界有统一构造，不要求 LP、对偶或免费支持 oracle；辅助参考中点带有完整误差预算，不冒充原模型。[边界记录](LIMITS.md)说明旧相位矩不能自动跨新 ReLU 复用，固定负门斜率也不是新的联合强度。

这些仍是已知数学机制组成的候选前向组件，不能据此声称新域新颖性、全网闭包、GPU 资格或正式成绩。新颖性应检验组合定理与完整成本下的能力，不额外要求超越“原 HZ 加完全相同行”的逻辑精度；后者本来就是同一线性系统。

## 真实证据可复用的范围

只读 schema 核对表明，冻结的 D025 large 完整 JSON 含 3072 维原输入盒、1600 个源 affine forms、1057 个唯一有序 pair 界，五个原窗口、每窗口 576 个 canonical slots 和 64 个 receiver rows。旧 canonical pairing 每窗口有 288 对，故旧 receiver-pair 出现数为 92160。这不是本轮新正权重配对规则的实际组数，后者要重新按完整系数解析计算，不能混算。

该文件可支持未来预注册的 large 存档组件检查，但不能替代原三模型人口或完整残差块。medium 没有完整逐行记录，Tiny 当时未进入；第二 Conv_6/BN_7 的参数也未保存。新研究若需要它们，须受控读取原始只读模型并保存新的独立证据，不能重跑冻结 D025 或把缩小人口记为它通过。

保存参数为有理区间，reader 必须完整保留端点、原 source ID、phase 和 frame；不得 eval 字符串 key 或经浮点重建有理数。THEORY.md 的显式误差桥接尚未实现和数值验证。还没有新的 reader、求界编译器或 GPU kernel。

## 验证与成本要求

新实现仍需独立预注册及默认关闭。D025 组件层 3753 tests、168 files 已通过，但整体 source census 未通过；不得混淆两者。后继资格实验继承完整组件人口并添加新测试，保留 collection 与执行合计 60 秒、worker 240 秒、256M 全局及 200M 每模型工作、64M entries、512-bit、AS16GiB、两项 1GiB 内存门，以及原全局 40M evidence 预算和 65536 reserve。具体原文保留在 D025 PREREG.md；本轮未修改门槛或执行新资格实验。

下一项实现问题已明确：认证稀疏共同源支持及误差，用四行投影输出原相位关联；核算多个真实消费者上的共享与重复工作。不能把单个观察的线性算术量宣称为整层线性总成本。实际残差支持的 source extractor、强单门及已有联合关系对照、所有模型和完整晋级验证仍待完成。

## 状态与保存边界

2026-09-30，分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。模式为纸面数学、独立只读复核、一手论文相关段落及旧证据 schema 和 hash 核验，无新增执行依赖。没有导入候选、跑测试、网络、solver、GPU、shadow 或 replay；没有确认需等待的数值进程。

只新增当前隔离目录。原有九个 tracked 修改保持 3806 insertions、57 deletions；未修改生产、历史模型或冻结实验，未 commit/push。以下输入 hash 本轮已重新核验：

```text
0fd93920413f63ad6b3b65849ea4d857c416c6bf4ab56263c63ff73416ebe80c  GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md
fbab84537df071153a7d9362161b2c9aa85b80c8b4c605ff1e1149170e0965e0  results/d025_interval_capacity_20260930_v1/complete_0.json
c23b0541c696375224a6aae6b530ecce9609cece8806b16d53e1bda517860a4a  results/d025_interval_capacity_20260930_v1/preregistered.json
32a0b157cda8cb4c0bb4947bbb9a5b8264290cc8d50172c98e08e446dd623239  results/d025_interval_capacity_20260930_v1/inventory.json
634c09d1ffdeeafa4bedf3b16c7e1febb058301b69340f1b83f306c0e40be2aa  results/d025_interval_capacity_20260930_v1/exit.json
```

正式基线仍为 1870/2413（1063 CERT、807 validated ADV），逐家族和逐旧解保全条件不变。外部 E0 仍为 CIFAR100 25、TinyImageNet 36，共 61/400，独立记账。正式新增收益为零。Goal 保持 active，整体目标未完成；存在可继续的构造与验证工作，不构成 blocked。

使用 pages:write-page 技能把定理、先例、反例和未验证范围分开存档；主代理检查本地文本，没有发布外部 Page。所有 split、attack/PGD、BaB、backward/dual rescue 和实例菜单禁令保持不变。
