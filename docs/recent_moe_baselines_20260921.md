# 六项 MoE 工作：实验设计、制品核查与部署（2026-09-21）

这是 PI 指定的**新外部基线工作线**。不重开 input98、旧 holdout 或后端搜索，
不修改已冻结结果。起点 ACT `06e0b25fed0a301b2cec238fa060a2b7fb357aaa`。
配套：[比较协议](recent_moe_comparison_protocol_v1.md)、
[实际部署及来源记录 R2](recent_moe_deployment_20260921_r2.json)。
首批14次尝试的[R1](recent_moe_deployment_20260921.json)保留，R2增加一次作者权重控制。
[本阶段验收](recent_moe_baseline_validation_20260921.json)：10项控制测试、归档重建、
原14条尝试不变与历史主表核对通过；这些不是完整网络证书审计。

阅读范围：取得六篇全文；核对方法定义、实验正文、可获得的实验附录与作者代码入口。
这不是对六篇全部定理的独立证明审查。文中的原论文数值均为作者报告；
只有“本地运行”部分是本轮实测。成功导入、图表复算、训练完成、认证完成、同任务比较
是五个不同状态，不能互相替代。

## 先更正两个重要的旧判断

1. **MetaMoE 有作者制品**。从 VeriVITAL 实验室的全文第 18 页找到
   [PMQ9/Mixture-of-Experts_Research](https://github.com/PMQ9/Mixture-of-Experts_Research)，
   已取得 `research_paper` 分支 `6aed3606e4b226e18e1c9d249485d99bb453f488`。
   仓库含模型、训练脚本、router/expert αβ-CROWN 验证入口及 `paper/artifacts` 权重。
   旧的 MetaMoE-style 比较仍是重实现；不能追溯更名为作者工具复现。
2. **RoME 已发布代码及权重链接**。实际克隆
   `4c691ccaeb7dfb613015d825435d4247a5198f9c` 含完整训练／评估／模型实现，
   README 给出 Google Drive checkpoint 链接。
   先前缓存页面的“待发布”文字已过时，不能再据此排除实验。

## 总览：应该比较什么

| 工作 | 作者主要评估对象 | 主要数据 | 与 ACT 的比较层级 | 当前可执行制品 |
|---|---|---|---|---|
| Dual RS | smoothed router + smoothed classifier/experts，统计认证 | CIFAR-10、ImageNet | 认证系统定位与任务级 trade-off；不是同一原始确定性函数 | 训练、认证、结果数组、图表代码；router/fine-tuned 权重需定位或按协议训练 |
| MetaMoE | disjoint-class hard top-1 的组合验证 | CIFAR-10/MNIST；GTSRB/PTSD | 最直接的形式验证对照，但必须保留其类别空间语义 | 作者脚本、权重、αβ-CROWN 子模块引用均已取得 |
| RoME | Transformer 内部多层低秩加权专家的多威胁训练 | CIFAR-10、ImageNet-100/1K | 公开高准确率模型、经验鲁棒性；当前 ACT 入口不支持原始多层对象 | train/evaluate、AutoAttack 入口、权重链接已确认 |
| J-TLAT | 视频 hard top-1 MoE 的联合攻击／训练 | UCF-101、HMDB-51 | 攻击有效性／模型外部效度，不是现成 verifier | 指定作者仓库当前仅 README |
| Feature Noise | 特征噪声、模块化与稀疏计算 | 合成、SST-2/CoLA/MNLI/AG News、WikiText2/MiniMind | 理论与机制定位；不与逐输入 certified accuracy 混比 | 未在本次全文、机构页、精确标题检索中定位作者复现实验仓库 |
| Robust Experts | CNN 中间层 ConvMoE/BlockMoE 的对抗训练 | CIFAR-100 | 第二个公开模型家族；不能冻结内部路由后冒充原模型认证 | 实际训练／评估代码；缺失 vendor 模块阻断完整架构导入 |

## 1. Dual Randomized Smoothing: Beyond Global Noise Variance（ICLR 2026）

来源：[全文 v3](https://arxiv.org/abs/2512.01782v3)，§4–6、附录 C–E；
[作者代码](https://github.com/eth-sri/Dual-Randomized-Smoothing)，
commit `6f83aeb7f47466b1dee6295baad8d59a8c94eceb`。
不要误认成 ICLR 2024 同样简称 DRS 的另一篇工作。

### 作者如何设计实验

- 验证对象是经过 Gaussian smoothing 后的分类器。先认证噪声选择器在邻域内保持选择，
  再认证选定 classifier/expert；最终半径取两者较小值。使用 L2 半径及统计失败概率，
  不是 weighted top-2 原始网络的确定性 L∞ 证明。
- CIFAR-10：50M diffusion denoiser、87M ViT classifier、ResNet-110 variance estimator；
  候选噪声 `{0.25,0.5,1.0}`。ImageNet：552M denoiser、305M BEiT、ResNet-50 estimator，
  候选 `{0.5,1.0}`。
- 主认证 `N=10,000`，总体 uncertainty `0.001`；作者 CLI 显式两部分各用 `0.0005`。
  CLI 默认值未必等于主实验值：例如 sigma-estimator 入口默认 N 为 100,000，必须显式传参。
- CIFAR router：90 epochs，batch256，AdamW lr0.01/wd0.01，每30 epochs乘0.5；
  consistency λ40、η0.5、两份噪声。classifier fine-tune：15 epochs，batch128，lr2e-5。
  ImageNet router 9 epochs，classifier 在2%训练集上 fine-tune 1 epoch。
- Table2/Fig3：分别比较固定全局 sigma 的 Carlini diffusion smoothing、Multiscale、
  Dual RS 的 off-the-shelf 与 adaptive fine-tuning；不能把训练差异当成纯认证算法差异。
- **真正的多专家比较是 Fig4**：sigma0.25和1.0的两专家，分别构造 weak/strong 组合。
  Table2 的三噪声单 classifier 配置不能自动命名成三独立专家实验。
- 消融考察 consistency loss/λ、候选 sigma、训练标签估计 N、estimator 架构及
  classifier fine-tuning；主指标是整条 certified-accuracy–radius 曲线。
- 作者报告 RTX4090 下每输入22.58秒（标准RS14.07，Multiscale20.21）；
  CIFAR完整训练流程约19.5 GPU小时，ImageNet标签构造使用128 GPU。
  这些不是本机耗时，也不是未测量训练的 ETA。

### 本轮实际运行及边界

作者 `reproduce/code/tab3.py` 原样运行成功：ImageNet fine-tuned Dual RS 在
L2 `{0,.5,1,1.5,2}` 上为 `{74,60.6,48,33.6,17}%`，与 Table3 一致。
独立重算 CIFAR fine-tuned 行的11格也与 Table2 一致：
`70.53,57.48,45.27,34.15,24.68,17.84,12.46,8.83,6.65,4.73,3.14`。
保存结果行数为 CIFAR10 10,000、ImageNet 500，严格使用作者的 `radius > threshold`。

**保留不一致**：off-the-shelf CIFAR 零半径作者脚本和我们的独立重算均为69.34%，
v3 Table2写68.34%；其余10格一致。不得改阈值消除差异。
独立重算 R1/R2 因这格不一致以非零退出码结束；R2仅修正报告中 NPY 后附 oracle/label
列被误称专家列的问题，最终半径未变。NPY行没有独立输入ID，仍信任作者的按位置对齐。

作者图脚本尝试完整保留：fig3a/fig4 缺本地 TeX `type1cm`；fig3b 还缺仓库内
`reproduce/data/multiscale/0.25_0.50_1.00.tsv`，不能宣称整张图已复现。
原始 ResNet110 estimator 架构 CPU 前向和输入梯度通过，但为随机初始化、未含 normalization
wrapper，无新认证结果。认证 CLI 在当前 ACT 环境缺 `statsmodels`。

部署还需：独立环境、diffusion/classifier权重、实际 estimator checkpoint 或冻结训练。
README虽列 `logs/`，当前克隆并不等于已有训练权重。发现 `core.py` 的 `predict()` 引用
未导入的 `binom_test`（文件导入的是 `binomtest`）；只有实际涉及的入口才作单独兼容修复，
不把此静态问题冒充已经解释所有认证失败。

## 2. MetaMoE（SAIV 2026）

来源：[作者全文](https://www.verivital.com/research/pham2026saiv.pdf)，§3–6、附录A–E；
[DOI](https://doi.org/10.1007/978-3-032-32357-6_8)；作者仓库见上。
事件是SAIV2026，出版社正式引用元数据年份为2027，书目需注明而非自行改DOI记录。

### 作者如何设计实验

- 不相交的类别空间：router识别数据域，hard top-1选择专家；选中专家填入对应输出段。
  在它的任务语义下，误路由会选错类别空间，因此路由保持具有不同于同类专家的必要性。
  不能把 ACT 的共享标签空间路由变化成功直接用来反驳该定理。
- 数据对：CIFAR-10/MNIST（视觉差异大）与 GTSRB/PTSD（相似交通标志域）；
  论文附录另列BTSD、ETSD、TSRD作为扩展域，不能都算主验证实验。
- CNN：4卷积20→28→40→56，前三层AvgPool，896→64→C，专家BN，router无BN。
  专家训练Adam lr1e-3、batch128、最多200 epochs/early stop、label smoothing0.1；
  RT用PGD7，epsilon8/255、step2/255。router最多50 epochs，wd5e-4，专家冻结。
- 8种router/expert RT/NRT组合，正文称40 runs（各5次）；正式验证40个router输入
  （20 CIFAR+20 MNIST），每专家20个clean-correct输入，epsilon `{2,4,8}/255`。
  PGD7在完整测试集上评估，不是同一小集合的认证准确率。
- αβ-CROWN正文版本`d4c79e3`、PyTorch2.1/CUDA12.1、RTX4060 8GB，expert timeout300秒。
  作者仓库子模块实际指向`58bb93f4886eea7cd1a3dfeb303695f20f61473b`。
  两版本尚未作代码差分，不能假定相同。
- Fig3路由：2/255、4/255两种训练均100%；8/255 RT100%、NRT60%，后者为超时，
  不是检测到错误路由。Fig4专家：CIFAR-RT为90/54/0%，MNIST-RT为100/99/95%；
  因此摘要中的router100%不能填进完整MoE SAFE率。
- Table1是PGD攻击准确率：both RT clean87.98%、AA41.93%，both NRT91.32%/28.78%；
  ±值及5次重复必须保留。增量添加数据域的训练时间与验证时间是另一个比较。

### 制品落实及必须核对的接口

已克隆代码和仓库内13MB `paper/artifacts`，有 CIFAR/MNIST/GTSRB/PTSD 的NRT/RT专家
及两种数据对的MoE权重。原 `UltraVerifiableCNN(10)` 前向／输入梯度通过（94,310参数）；
`verify_expert_abcrown.py --help` 原样通过。没有执行实际CROWN验证或声称复现Fig3/4。

进一步实际加载作者CIFAR-RT专家checkpoint `e256e3a4...`：限定7种已审查类型，
`weights_only=True`，没有不受限pickle回退；CPU随机输入前向／输入梯度均通过，
启动含preflight约1.90秒。该权重为作者已训练制品，但本控制未加载数据集、未计算准确率，
也未验证完整MetaMoE。外部子进程使用现有rt-er-blackwell环境，未改动依赖。

作者 `run_expert_formal_verification_experiments_5_times.py` 对同一模型文件重复调用验证，
本身不训练五个独立seed。论文“5 independent runs”和固定seed42，需要保留解释边界；
不可将这五次自动当成训练seed样本。正式比较也要区分训练变化与求解重复。
仓库内checkpoint包含序列化作者类；后续须限制安全加载的允许类型，不能无检查地
把任意pickle交给 `weights_only=False`。

下一步优先复跑作者**同一checkpoint的组件义务**，再组合完整请求；
输入归一化、BN folding、VNNLIB十位小数导出、tie policy须显式记录。
作者清理脚本会删除运行目录下旧VNNLIB；必须在独立运行副本使用，绝不在历史目录运行。

## 3. RoME（ECCV 2026）

来源：[全文](https://arxiv.org/abs/2607.06109v1)，§5、附录A1–A4；
[已发布作者代码](https://github.com/wkim97/RoME)。

- CIFAR-10、ImageNet-100/1K，ImageNet预训练ViT-B/DeiT-B/Swin-B，另有WideResNet扩展。
  四个rank16低秩专家放在Transformer各块QKV/O；patch/global门与多威胁diversity loss。
  实际适配器按softmax混合所有低秩专家，不能简化成 ACT 的输出层 sparse top-2。
- 威胁 `(L1,L2,L∞)`：CIFAR `(12,.5,8/255)`；ImageNet `(255,2,4/255)`。
  PGD训练步数分别20/20/10和40/20/10；APGD训练每威胁15步；测试是各norm的AutoAttack。
- 对照 RANDOM/AVG/MAX/MSD/MORE/E-AT/RAMP；未知威胁另比PAT/VR，包含corruption、patch、
  perceptual/spatial/recolor等，不是统一单一epsilon实验。
- AdamW，CIFAR20 epochs/batch64/lr1e-3，ImageNet5 epochs/batch128/lr1e-4；
  warmup+分段线性衰减。fine-tuning配置另为3/1 epochs。公平比较区分PGD/APGD训练组。
- 同时报告clean、每威胁RA、均值及**同输入三威胁成功集合交集**（union robustness）。
  union不能用三个RA的最小值代替。
- Table1 CIFAR RoME+MAX/APGD：82.5% clean，48.9/67.3/44.2%各norm RA，53.5%平均，
  43.7%union。这些是经验攻击结果，不能与我们的policy SAFE率排列为认证冠军榜。
- 消融拆开低秩结构、local/global gate、diversification，另量参数/FLOPs/训练代价。

本轮：完整 CIFAR10 ViT RoME 未训练架构（145,547,434参数）CPU前向及输入梯度通过，
没有缩小层数或token数、没有修改作者代码。训练／评估CLI尚缺`autoattack`等独立环境依赖。
作者提供6个MAX/RANDOM×三数据集checkpoint的Drive位置，但本轮未下载／评估这些权重。

必须冻结的分歧：论文§5.1写门参数`b=2`，当前train/eval脚本和adapter默认`b=6`，
且说明s/b不存于checkpoint；不能按哪边成绩好来选。先采用**code-default reproduction**
并明确标记，论文配方作为单独协议，不把两者混合。checkpoint loader的`strict=False`
会打印缺键；正式复跑须在外层检查非允许缺键并拒绝，不能把随机初始化残余层当成作者模型。

## 4. J-TLGA / J-TLAT（ICLR 2026）

来源：[会议论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f206871468dc89cb20fefbc75f2de861-Abstract-Conference.html)，
[全文](https://arxiv.org/abs/2602.01369)，§3–5；
[作者仓库](https://github.com/DeepSota/J-TLAT)，`a8f2059ac83f4634c992223ad2082a3db5c829bc`。

- UCF-101/HMDB-51视频分类；3D ResNet18、TSM、SlowFast、R(2+1)D，4专家hard top-1。
  训练中的Top-2弱专家目标不等于推理架构改成Top-2。
- TLGA针对router，J-TLGA联合router/experts；Lipschitz目标、时序步长、专家mask为关键组件。
  epsilon为 `{8,10,12,14}/255`。论文正文未在已读实验段完整给出所有采帧、split和训练
  随机种子细节，不能从其他视频论文默认值补造。
- 攻击对照FGSM/PGD/AutoAttack/TT及router-only/expert-only/overall/joint；防御对照
  AT-S、AT-D、AT-M、OUD-M、AAT-M、TLAT、J-TLAT。展示clean/RA/GFLOPs/Lipschitz/IoU。
- UCF101+3DResNet Table2：J-TLAT clean54.29%，8/255 PGD36.37%、AutoAttack34.29%、
  J-TLGA33.96%；这是不同攻击下的经验RA，不是逐输入形式证书。
- 消融内层攻击、正则项、时间自适应和联合目标，另报告迁移攻击／训练动态。

当前克隆仅README，无Python实现、checkpoint或评估入口：标记`BLOCKED_UPSTREAM_CODE`。
不得用自写PGD改个名字充作J-TLGA。代码获得后应优先复跑攻击表与威胁约束、动态全模型
反例复放，再考虑视频MoE验证扩展；不在本轮训练巨大视频网络。

## 5. Robustness of Mixtures of Experts to Feature Noise（ICML 2026）

来源：[作者全文v1](https://arxiv.org/abs/2601.14792v1)，§3–5、附录B1–B5；
[机构登记](https://cispa.de/en/research/publications/213129-robustness-of-mixtures-of-experts-to-feature-noise)。

- 理论控制**总参数数相同**，研究带模块结构的特征及噪声下泛化、收敛与样本复杂度；
  正确／错误路由的风险不同。不是每个L∞输入盒子的安全判定工具。
- 合成数据验证模块假设；LLM激活图使用Llama2-7B/Llama3.1-8B、WikiText2 validation、
  长度1024及TEAL阈值。各层重排／聚类不同，图不能逐格当同一坐标比较。
- T5-base MoEfication：96专家选择20/40/60/80，SST-2 validation；word swap和keyboard noise；
  100个噪声seed0–99的均值/区间，不是100个独立训练网络。
- T5-small线性probe：SST-2、CoLA、MNLI、AG News；Fisher-score约束聚类，专家logistic
  regression，oracle loss最小专家作为router监督。对照global Lasso。
  Gaussian sigma `{.2,.5,1,2}` 与文本扰动分开；指标clean−noisy drop，CoLA用weightedF1，
  其他用accuracy。论文称搜索正则强度后报告最佳值，复跑需明确选择集以防test调参。
- MiniMind：8层、hidden512、8heads；4routed+1shared、top2，每expert中间层1024；
  dense中间层5120实现FFN总参数匹配，active参数不相同。gongjy/minimind语料，1epoch约5000steps，
  batch32、AdamW5e-4、MoE balancing0.01；比较train/validation loss，不是认证率。

全文中的TEAL、NLPAug、MiniMind链接是依赖／基础框架，不是此文完整复现包。
本次未找到明确绑定本文实验的作者代码，标记`AUTHOR_REPOSITORY_NOT_IDENTIFIED_IN_SEARCH`，
不声称作者没有发布。若自行实现合成控制，必须命名paper-based reimplementation并单独冻结。

## 6. Robust Experts（ICCV 2025 Workshop STREAM）

来源：[CVF全文](https://openaccess.thecvf.com/content/ICCV2025W/STREAM/papers/Pavlitska_Robust_Experts_The_Effect_of_Adversarial_Training_on_CNNs_with_ICCVW_2025_paper.pdf)，
§3–4；[作者仓库](https://github.com/KASTEL-MobilityLab/robust-sparse-moes)，
`ed22e81fabc3c3196b6bcd352ee83042473cdfbf`。

- CIFAR-100，ResNet18/50；ConvMoE换中间卷积，BlockMoE换残差块；稀疏k与专家数、
  GAP-FC/Conv-GAP、entropy/switch/KL balancing作组合实验。论文比较dense baseline及
  普通训练／对抗训练，不只是拿自己最好一行比别人的未经训练模型。
- 正文：SGD momentum.9、wd5e-4、lr.01、polynomial decay、200epochs；PGD7训练，
  PGD20/AutoPGD评估，RTX4090。源配置PGD训练epsilon0.03137、step≈2/255。
- 图4是clean–attack accuracy trade-off，另测FLOPs/参数随E/k；§4.3按固定专家路径检验
  特化和routing collapse。固定路径评估是诊断，不是完整动态MoE认证。
- 数值主结果多为散点图；本轮不凭图手抄高精度成绩，正式重跑需原CSV或模型评估。
  正文用“average precision”而图标accuracy；须核对源torchmetrics实际定义再定列名。

部署核查发现三个具体障碍：

1. `requirements.txt`需要Lightning、Hydra、torchattacks等；ACT环境不具备。
2. **作者完整架构实际导入失败**：`resnet_block_moe.py`依赖仓库未包含的
   `src.models.nn.deeplabv3plus.sync_batchnorm`。失败堆栈已保留；不伪造模块让测试假通过。
3. 默认experiment写100epochs/lr.1；batch launcher把epochs改成200但仍继承lr.1，
   与论文lr.01不一致。eval默认通过W&B定位checkpoint；未确认无需账号的作者训练权重下载。

另须保留真实权重语义：源TopKGate先经router输出分数，归一化分母有`+1e-5`，
故不能声称与当前ACT精确selected-softmax凸组合语义完全一致。
BN/容量及train/eval路由差异也必须在模型接口合同中披露。

## 本阶段部署结论与下一步

本轮已完成：六篇实验方法梳理、五个作者仓库定位/固定、实际原脚本图表尝试、
三项架构前向/梯度控制、一项作者预训练专家加载控制、四个公开入口探测、失败记录与紧凑归档。
**尚未完成**：六项完整作者成绩复现、任何新的ACT–作者工具同任务胜负表。

优先顺序调整为：**MetaMoE作者权重与正式验证路径 → Dual RS认证pipeline →
RoME作者权重/AutoAttack → Robust Experts兼容修复/训练**。J-TLAT与Feature Noise
缺失制品独立挂账，不用假重实现占位成绩。具体冻结条件见比较协议。

本轮没有安装依赖、上传数据、联系作者或占用他人GPU。需要PI明确允许在MOE目录创建
专用隔离环境及安装作者依赖；不能更改`act-py312`或现有冻结环境。作者联系由PI管理。
