# Robust Experts：完整训练落地、独立存盘复核与配方差异

2026-09-22，**无新增训练、模型查询、攻击或模型选择**。原执行 commit
`acd27d19f`，配置 SHA256
`7731f7ae144b3816d28b0dde7ee59dfcb4fac90f2c73b6e47cd36ceeca67954d`。
作者源码固定为 `ed22e81fabc3c3196b6bcd352ee83042473cdfbf`，使用已披露的
R3 optional-SyncBN / APGD eval-mode 兼容补丁及 Blackwell 环境。
原始运行目录、配置、200 个逐 epoch checkpoint/臂和旧失败均保持原样。

## 1. 可以引用的结果与证据等级

| 最终 epoch 200 模型 | Clean | PGD-20 | APGD-20 | 训练／终评／运行内审计秒数 | 全流程秒数 |
|---|---:|---:|---:|---:|---:|
| Dense ResNet18 |37.27%|18.99%|18.74%|5228.868 / 28.339 / 6.921|5264.203|
| ConvMoE E4/k2/layer4 |22.18%|12.88%|11.70%|20798.652 / 113.290 / 31.859|20943.859|

每个评估均为原序 CIFAR-100 test 的 10,000 张；clean 16 批，攻击40批。
两臂各 200 epochs / 12,600 更新；每训练 epoch 实际 62×640+320=40,000 张。
全 launch 含前后检查 26,208.253 秒，约7小时17分。
这是 **EMPIRICAL_ATTACK_EVALUATION**，不是 ACT SAFE、认证准确率或全论文复现表。
一次固定训练运行中 ConvMoE 的 clean/PGD/APGD 分别低15.09/6.11/7.04pp；
不据此推断所有 ConvMoE、所有作者配置或所有 seed 都劣于 dense。

最终权重绑定：

- Dense：`975b1962e327b2ee7cce41c7ce23cf4a767e9860c3746f9a05787b6745f02d9d`
- ConvMoE：`056de314a1b48e9243a060b8e537f1d5939b678d56f41e106f3ad8f1bf7b5d75`

[独立复核 JSON](review.json) / [全部400条派生 epoch 数据](trajectory.csv) /
[训练曲线](trajectory.svg) / [作者源语义控制](source_semantics.json)。
这里的 JSON/CSV/SVG 是紧凑派生制品；权重、数据和作者仓库不提交 Git。

## 2. 独立复核到底检查了什么

新检查器不调用训练 supervisor、运行内 auditor、作者模型或攻击函数。
在 ACT `act-py312` CPU 环境重新读取：498 个冻结来源/数据文件；400 个
逐 epoch checkpoint 的实际文件哈希；两个最终 checkpoint 的全部122/147个
状态张量及62/87个动量缓冲；optimizer/scheduler/RNG与身份字段；全部 epoch
journal、两份训练CSV、两份评估CSV、逐攻击来源、阶段终态和完整成本。
200个epoch的原生 polynomial 学习率轨迹、每epoch40,000张、63更新及最终零LR均一致。
没有发现模型替换、缺epoch、短批次控制残留、CSV覆盖、评估间保存状态变化或预算越界。
独立复核48.060秒，另计于原训练流程之外；作图时间不包含在此复核计时内。

**边界：** 保存的指标是聚合值及运行时覆盖计数，没有持久化逐输入预测／攻击张量。
检查器验证其多份记录一致、可对应整数计数，不独立重算每个预测。
因此不能写“独立重跑全10k攻击通过”，更不是形式证明或人类技术评审。
保存CUDA RNG不代表已经建立GPU精确续训等价性。

## 3. 训练轨迹：并非停训；总loss不能直接比较

下表 epoch 为**已完成epoch数**，不是原日志从0开始的编号。
训练准确率在增强＋PGD7批次上测量，验证准确率为 clean validation；不是逐epoch test RA。

| Epoch | Dense clean val | ConvMoE clean val | Dense PGD7 train | ConvMoE PGD7 train |
|---:|---:|---:|---:|---:|
|20|11.16%|7.80%|5.81%|3.91%|
|50|20.80%|12.41%|9.99%|6.01%|
|100|28.48%|16.72%|14.78%|7.95%|
|150|33.80%|19.86%|17.69%|10.05%|
|180|35.32%|21.05%|19.05%|10.76%|
|200|35.98%|22.05%|19.31%|11.23%|

两臂持续学习到后期；有波动，不声称逐epoch单调。日志最高validation分别
36.09%@e197、22.23%@e199，只是描述，**没有替换预注册的e200模型**。
末20epoch平均validation为35.762%/21.802%。没有证据把差距解释成中途崩溃或未完成训练。

最终训练分类loss为3.4256/3.9261；ConvMoE辅助loss为−3.3385，总loss仅0.5876。
因此“ConvMoE总loss更低”不代表分类拟合更好。该负项由原生entropy目标与五个gate的
0.5系数求和产生。负loss本身不是数值错误、路由坍缩或某一专家负载的证明。
没有逐gate entropy/route-share/梯度分解遥测，不能进一步确定性能差距的唯一根因。

## 4. 论文—发布源码—本次执行，分开对照

论文来源：[CVF正式全文](https://openaccess.thecvf.com/content/ICCV2025W/STREAM/papers/Pavlitska_Robust_Experts_The_Effect_of_Adversarial_Training_on_CNNs_with_ICCVW_2025_paper.pdf)
§3.4、§4.1–4.2；[作者arXiv全文](https://arxiv.org/html/2509.05086v1)；
[固定作者代码](https://github.com/KASTEL-MobilityLab/robust-sparse-moes/tree/ed22e81fabc3c3196b6bcd352ee83042473cdfbf)。
CVF文件已有本地冻结副本；本次网页读取CVF返回403，arXiv和作者仓库可读。
不从散点图猜测高精度论文表格成绩，也不宣称匹配某个未公开运行的完整配置。

| 项目 | 论文／作者发布内容 | 本次实际执行及分类 |
|---|---|---|
|模型、数据|ResNet18/50、ConvMoE/BlockMoE、CIFAR100|只做ResNet18 dense与ConvMoE E4/k2/layer4，两次固定运行，不是全组合复现|
|训练轮数|正文200；experiment默认100；shell覆盖200|200，与正文一致，预先覆盖默认值|
|优化器|正文SGD、momentum.9、wd5e-4；源码另有Nesterov|一致；final optimizer实际状态核对通过|
|初始LR|正文.01；experiment默认.1，shell未覆盖|预先冻结为.01；与正文一致、与发布launcher默认不同，不能说逐字执行默认命令|
|LR schedule|正文polynomial；源码PolyLR指数.9|每epoch下降，e200归零；不是常数LR或后程阶梯衰减|
|数据规模|正文只说CIFAR100；原生Bolts默认20%validation、split seed42|40k训练＋10k验证，另有10k test；不是50k全部用于训练。模型seed12345不等于split seed|
|增强与归一化|正文未给完整细节；源码Crop32/pad4、Flip、ColorJitter(.3)、ToTensor、CIFAR100 Normalize|prepared配置和源数据管线保留五项；不凭smoke推断增强。本轮未重新物化随机增强样本|
|训练PGD|论文PGD7；源码eps.03137、alpha.00784313725、随机起点|原样；训练半径比精确8/255小约2.55e-6，不偷换为精确相等|
|终评攻击|正文PGD20/AutoPGD；`src/run.py`硬编码8/255、PGD步长2/255|全10k PGD20及APGD20；后者是torchattacks APGD-CE、1 restart、seed0，不是完整AutoAttack ensemble|
|k、gate|正文较好组合top2、GAP-FC；experiment默认k1|冻结E4/k2/GAP-FC；原代码对k≠1自动关闭STE，5次warning与constructor控制一致|
|**entropy目标**|正文§3.4：−H(batch mean)；源码`entropy`：mean H(sample)−H(batch mean)|**继承源码`entropy`，不等于论文公式**。源码`column_entropy`才是−H(batch mean)。这是语义差异，不是已证实的性能根因|
|MoE插入范围|论文比较单层／整stage、两种MoE|源码`layer4`匹配其中所有Conv2d，共5个gate，包括downsample；不能称“只插入单个MoE层”|
|门控输出|源码对softmax scores选top2后除以选中和＋1e-5|保留epsilon分母；不是ACT输出层严格和为1的selected-softmax函数|
|checkpoint选择|本次预注册固定finalepoch；作者完整图表选择细节未完全定位|保留全部epoch但仅终评e200，不切换到最高validation权重|
|设备／环境|正文RTX4090；公开依赖不直接适配本机|Blackwell＋隔离依赖overlay；optional SyncBN导入与APGD模式恢复为显式补丁。不是原环境逐比特复现|
|指标|正文有average precision措辞、图标accuracy；源码`Accuracy(multiclass)`|报告top-1 accuracy而非PR曲线面积；loss为原生batch均值的均值，不能称严格样本加权平均|

最重要的新核查是entropy目标。设每个样本概率为p_t，P=mean(p_t)：

\[
L_{\text{paper}}=-H(P),\qquad
L_{\text{code:entropy}}=\frac1T\sum_t H(p_t)-H(P).
\]

两行均匀四专家概率的原生CPU无训练控制：`entropy=0`，
`column_entropy=-1.3862943611198906`。两者不是相差全局常数，不能按名字视为同一个目标。
原生`top2,use_ste=true`构造控制得到effective STE=false。见[source_semantics.json](source_semantics.json)。
不修改已训练源文件或把本次权重倒称为`column_entropy`模型。

APGD比PGD耗时短也不能单凭比例定为故障：已安装APGD代码在`cheap=True`下只攻击
当前尚分类正确的输入，而PGD入口遍历全批；两条终评路径均覆盖10k并统计原本错分类样本。
这提供执行结构解释，但不是独立攻击强度评测或耗时因果分解。

## 5. 验证与重建

ACT环境的9项新复核控制通过：缺epoch、训练覆盖、LR、loss/CSV矛盾、缺攻击、
权重绑定/测试分母错误、评估CSV篡改、状态张量变化/NaN拒绝。
原监督/日志隔离/长配方的10项控制在已有作者CPU环境通过。
第一次把19项一并放ACT环境运行，有3项旧测试因缺OmegaConf报错（非实验失败）；
未安装依赖，改回其既有隔离环境执行。语义检查器首轮错误地要求未执行的作者shell
launcher也在执行清单中，因缺绑定而拒绝；修正为“只作对照且核对作者commit”，未放宽执行源绑定。

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -m unittest tests.test_robust_experts_landing_review
/data1/Kane/MOE/envs/robust-experts-workflow-cpu-20260921-r2/bin/python -m unittest tests.test_robust_experts_pipeline tests.test_robust_experts_pipeline_r2 tests.test_robust_experts_training_recipe
```

独立复核需本机已保存的权重／原文件，纯clone不能重跑它；输出必须是新目录：

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python scripts/review_robust_experts_landing.py --config configs/recent_moe/robust_experts_paper_training_execution_r1.json --output /data1/Kane/MOE/baseline_runs/robust_experts_landing_review_NEW
```

没有修改被冻结的旧archive脚本。它的固定scope文字仍指“limited-batch control”，
不适用于长训练；本次使用另一个独立脚本，避免把完整结果套入错误范围标签。

## 6. 结论与下一步边界

本批完整训练、全测试终评、独立存盘复核与归档已完成。诚实名称是
**paper-hyperparameter / source-entropy compatibility variant**；“完整论文结论复现成功”未建立。
对日志可作的判断是ConvMoE学习较弱且本次未获经验优势；训练轨迹没有隔离原因。

若继续归因，先另行固定paper-column-entropy与source-entropy的**单因素语义控制**及梯度合同，
而不是同时把LR改.1、增epoch、改增强、换seed再比较。此次没有批准或启动新长训练。
也不能把此结果作为高准确率目标完成或训练后多层ACT全域认证成绩。
