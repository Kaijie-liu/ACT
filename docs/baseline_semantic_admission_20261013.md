# 六篇 MoE 工作：逐对象语义与比较准入

2026-10-13；只读核对的 ACT HEAD：`38478a1379e44775616d7ce6334e5111ef7fb4c2`。
本页完成 G6 的**比较合同子项**，不关闭 G6，更不替代 HybridZ 算法、GPU 和真实证明目标。
无模型加载、求解、训练、攻击或 GPU 调用。只使用已冻结的本地论文／作者源和归档；
不是本日线上可用性调查。旧 NEXT 不是新许可，本文不冻结或准入任何新执行。

已有成绩不重算、不换等级，见[执行账本](author_baselines_status_20260924.md)。
本页补充的是旧[三类比较协议](recent_moe_comparison_protocol_v1.md)未落实到对象的条件。
源码 SHA、精确几何／概率算术见[紧凑记录](baseline_semantic_admission_20261013.json)，
核查器为 [analyze_baseline_semantics.py](../scripts/analyze_baseline_semantics.py)。
它检查算术与文件身份，**不是函数等价、Gaussian RNG 或网络证明检查器**。

## 1. 准入结论与最小缺口

| 工作 | 实际对象／保证 | 可报告的比较 | 新 ACT 完整比较的缺口 |
|---|---|---|---|
| Dual RS | diffusion＋选择器／ViT 的平滑组合，统计 L2 保证 | 两输入原生统计 pilot；不是确定性 ACT 胜负 | 同一平滑函数及噪声／置信合同，或明确改为不同函数的系统级比较 |
| MetaMoE | 类别分离 raw-score top-1，其他类别填零 | 已完成10输入同对象 operational 比较，保证分栏 | 新版本来源闭合与独立冻结；不能要求此对象产生跨域路由变化正证书 |
| Robust Experts | 已训练 E4/k2、五处中间卷积混合，带分母偏移 | 两配置全10k经验攻击；非ACT认证比赛 | 架构特定非零性、原权重的联合转换、完整历史覆盖 |
| RoME | token相关、全专家连续 LoRA 混合与完整 ViT | 4输入×3范数经验终态；非认证率 | 连续全网络语义与转换；离散top-k适配不适用 |
| J-TLGA/J-TLAT | 视频top-1 MoE的联合攻击／训练 | 论文定位；完整作者执行仍未取得 | 实现、权重、采帧／归一化／域；攻击不是验证器对手 |
| Feature Noise | Gaussian特征噪声、文本编辑、风险理论等不同任务 | 理论／机制参照 | 对象制品与任务拆分；有界特征认证须另立合同 |

新配对必须逐项固定函数／权重／补丁、物化域、性质与 tie、证据等级、全集与选择规则、
总预算和硬件。启动、加载、转换、候选、同步、回退、检查及终态接收均计费。
同对象不是 manifest 两个 `same=true`；有限探针也不能证明全域等价。
未执行指标保留 null，不把缺制品、错误、超时或攻击未找到反例写成安全。
既有比较可保留数值等级进行覆盖—成本分析，不能等严格证明全部闭合才承认它存在，
也不能把数值比较升级成严格证书支配。

## 2. Dual RS：概率在认证抽样上，函数不等于原确定性网络

固定作者源 `6f83aeb7f47466b1dee6295baad8d59a8c94eceb` 的 `code/core.py:76–93`
只调用 `base(batch,t)`；噪声来自 `DRM_sigma_est.py:50–91`、
`DRM_classifier.py:58–95` 的 `randn_like/q_sample`。设
\(a_t=\sqrt{\bar\alpha_t}, b_t=\sqrt{1-\bar\alpha_t}\)，作者先形成
\(a_t(2x-1)+b_tZ\)，再取单步 diffusion 的 clipped `pred_xstart`。
选择器接三分类 ResNet110；分类器接 bicubic/antialias resize224 和固定 ViT。
nominal σ 属于 `{.25,.5,1}`，第二阶段共用 diffusion／ViT 权重，**不是三个独立专家**。

令 S 为选择器输出分布的 population argmax，H_j 为固定 timestep 分支的
population argmax 分类函数；全局并列时固定取最小类索引，则目标为 \(G(x)=H_{S(x)}(x)\)。
正半径的置信成功事件上存在严格多数，认证开球内的结论不依赖这个 tie completion。
完整身份包括 denoiser、selector、ViT、timestep、预处理、argmax、eval、autocast 与补丁。
[冻结 driver](../scripts/dual_rs_certification_pipeline.py)取首个 \(b_t/a_t\ge2σ\) 的 t；
理想实数下原像素有效噪声 \(\tildeσ=b_t/(2a_t)\geσ\)，不能写成恰好 nominal σ。
原生 coefficient extraction／CUDA／RNG 尚未独立证明符合这个理想对象。

在已声明采样与概率数值前提下，保守的合同是：

\[
\Pr_\omega[\text{abstain，或}\ \forall\|x'-x\|_2<R_\omega,
                  G(x')=\hat y_\omega]\ge1-(\alpha_S+\alpha_H).
\]

它不是“99.9%的扰动安全”，也不是确定性原函数 F 的结论。分类器在选择后使用另一个
预定样本流；所选分支的条件置信有效时才能组合。Union bound 本身不要求独立，但不能
以此免除条件采样有效性。nominal每阶段 .0005，合成每输入≤.001，两输入同时成立失败≤.002；
记录中的1/2000等为协议理想有理参数，不是原生binary64(.0005)的精确值或尾概率误差界。
不无故乘三倍分类器预算，固定 seed 也不证明 IID。半径取两阶段最小值，正确预测才计
correct_radius；零半径／abstain不产生正半径证书。作者曲线使用 `radius > r`，本合同
保留开球；不把闭球边界默认为已证。[pilot及信任边界](dual_rs_certification_archive_20260922_r1.json)。

### 概率与确定性比较设计

首先区分两个不同实验，不得拼接胜负：

1. **同一 G 的认证算法比较**：ACT 必须真正支持上述平滑组合及有效统计合同。目前没有，
   暂不准入。仅验证 selector、ViT 或某个固定噪声样本均不能代替。
2. **不同系统的鲁棒性—成本比较**：可以另行固定同一原始图像、标签、部署协议、
   扰动域、选择规则和置信指标，分别报告 G 的统计保证和 F 的确定性/条件化保证。
   这是系统级比较，不能解释为同函数验证器谁更强；成本含随机预测与完整认证。

若一个有效统计证书给出开 L2 球，而待比较闭盒为 [l,u]、中心 z，则需检查

\[
\sum_j\max(|l_j-z_j|,|u_j-z_j|)^2<R^2.
\]

对未裁剪、原像素3072维、ε=2/255，其左侧为 **4096/21675**。通道 Normalize 后应使用
相应的坐标比例，裁剪盒用实际端点；不能只凭“CIFAR10”和相同ε判断同域。
这仅是几何包含，仍不证明 G=F、Gaussian实现正确或概率尾函数严格舍入。
本轮不将两个历史 pilot 半径重分类为新 ACT/闭盒结果。

## 3. MetaMoE：已能比较，但不是路由变化正证书的合适对象

固定作者源 `6aed3606e4b226e18e1c9d249485d99bb453f488` 的
`vision_transformer_moe.py:324–372` 返回 raw logits，不是注释暗示的 probabilities。
两个专家输出类别块 [10,10]。合法路由 i 的输出为
\(F_i=P_iE_i\,r_i/r_i=P_iE_i\)，**只在 r_i≠0 时可约去**；P_i 将其他块填零。
不能加 ε、改 softmax、填负无穷或只验证本专家9条 margin。

实际域是各数据集 Normalize 后物化的 `(1,3,32,32)` 闭盒，端点为保存的
`clip(x±binary64(2/255),-10,10)`。MNIST先Resize32→RGB，三个通道之后独立受扰。
这不是原28×28灰度域，也不是 raw-pixel 2/255 声明。
目标包括19条全局 `F_y-F_k≥m`，m为冻结的binary64(1e−7)，及 selected division 有定义。
ACT覆盖 `r_i≥r_j` 的全部 tie-legal 分支；作者充分适配固定中心路由，要求严格路由
优势、固定符号非零和严格输出 margin，共21条正不等式。
[对象/性质](../scripts/metamoe_paired_model.py)、[输入与执行](../scripts/metamoe_paired_execution_r2.py)。

**一个不需新求解的结构结论：**若路由选择不含标签 y 的类别块，y 及原块中另一类别 k
的输出同时为0，故 `F_y-F_k=0<m`。所以此 [10,10] 对象不可能在允许跨域选路的域上
满足全部严格正性质。MetaMoE 的 G4 比较研究组合验证覆盖与成本；G3/G5 的路由变化
正证据应来自共享输出类别的另行固定对象，不能从这个不可能任务追求成功。

[实际10输入配对](metamoe_las_followup_result_20260923_r1.md)已完整完成：ACT4、作者9，
ACT独有0；均为各自数值等级。双方同系数/物化域、CPU64和完整300秒，但来源对应、
functional rewrite、HZ/guard containment、后端数值及部署浮点语义仍非全部独立证明。
本轮未重开该集合，也未把作者数值过滤称为 formal SAFE。

## 4. Robust Experts：先纠正 definedness，再明确未实现的范围

已训练对象为 E4/k2/layer4 ConvMoE，最终权重 SHA
`056de314a1b48e9243a060b8e537f1d5939b678d56f41e106f3ad8f1bf7b5d75`。
作者源 `ed22e81fabc3c3196b6bcd352ee83042473cdfbf` 加已披露兼容补丁。
`routing.py:5–12` 的实际 router 以 Softmax 结束；`gate/topk.py:230–268`
用原 scores 计算 \(w_i=p_i/(s+\delta)\)，\(s=\sum_{i\in S}p_i\)，δ为正1e-5偏移。
另一次softmax用于选择，在实数语义下不改变排序。

有限实数logits给出 p_i>0、总和1；最大k项的平均不低于总平均，所以 s≥k/E。
此处 s≥1/2，选中权重严格非零、分母有定义，但

\[
\frac{1/2}{1/2+\delta}\le\sum_{i\in S}w_i
 =\frac{s}{s+\delta}\le\frac1{1+\delta}<1.
\]

因此，通用 raw-score dispatcher 的全零反例**不能用来断言此架构实数函数未定义**。
当前 [multilayer.py](../act/back_end/moe/multilayer.py) 一律拒绝非STE作者路径，准确含义
是架构特定支持未接入。此处没有取消拒绝门，也未证明原生浮点下溢、非有限值和
`nan_to_num` 分支。不能把 \(s+\delta\) 换成 s：中间卷积后还有偏置、BN、残差与非线性，
局部正缩放不等于最终 logits 的统一缩放；即使输出性质齐次也不能这样换对象。

五站点未剪枝历史目录为 \(\binom42^5=7776\)，超过默认4096。
这不是证明7776条都可达；需有证明的排除／共享覆盖，不能漏掉未决历史或直接加限额。
旧[全尺寸控制](../scripts/multilayer_author_history_control.py)是**未训练k1**、1024种目录、
两个具体探针，不是训练k2全覆盖。仍缺原softmax/加权联合传播及完整上游转换。

未来同对象请求须固定像素盒＋原Normalization，或有包含证明的逐通道normalized盒，
100类全部99条性质与tie。既有[200epoch、全10k成绩](robust_experts_landing_20260922_r1/README.md)
属于 source-entropy compatibility variant 的经验PGD/APGD评估；无ACT认证胜负。

## 5. RoME：不是需要更多 top-k 历史的模型

固定源 `4c691ccaeb7dfb613015d825435d4247a5198f9c` 的
`models/lora_moe_mlp.py:60–85` 计算 token均值的global softmax与逐token local softmax，
由层相关β混合，然后加权**全部**LoRA专家。没有top-k mask。
全ViT的attention、LayerNorm、GELU、patch embedding与残差都是对象的一部分。
两个softmax显式 `dtype=torch.float32`，`.double()`不会自动得到纯float64语义。
除权重外还需绑定s、b、alpha/r、层位置、eval/dropout、辅助头处理；不能只读state_dict。

所以不能冻结clean gate、删主干、改成top-2或只编译某条path，仍称同对象。
准入缺口是连续token相关的全网络转换，不是扩大离散history上限。
[旧12终态](rome_multinorm_archive_20260922_r1.json)使用每input×norm600秒，不能直接与
300秒ACT比较；5完成=3条同一clean错误＋2攻击记录，7超时。Linf记录依赖1e-7经验容差，
不在无容差的精确8/255盒内，不升级为精确UNSAFE。无新的执行准入。

## 6. J-TLAT 与 Feature Noise：实验任务不能由指标名称代替

J-TLAT 保存全文 §3–5 描述视频 \(x\in R^{C\times T\times H\times W}\)、Linf投影攻击、
UCF101/HMDB51、四专家推理top-1；训练选择弱专家的top-2不是推理top-2。
RA定义为攻击后正确比例，不能当成∀δ认证率。
冻结仓库 `a8f2059ac83f4634c992223ad2082a3db5c829bc` 仅README是**历史快照**，
不是本日线上或所有分发渠道无代码的结论。仍需实现/权重、split/video ID、采帧、
resize/crop、归一化、epsilon坐标及全部帧允许扰动的合同。时间自适应步长不是额外域约束。
作者攻击与ACT可以研究同模型，但只能比较反例/证明互补，不能把RA与SAFE率排胜负。

Feature Noise 保存全文的任务必须分列：

| 子任务 | 域与量词／指标 | ACT可能的有限适配，尚未执行 |
|---|---|---|
| 线性模块理论 | Gaussian特征噪声、部分理想gating假设、风险/概率理论 | 不能当具体模型证书 |
| T5-base MoEfication | 96选20/40/60/80；离散word-swap/keyboard，100噪声seed区间 | embedding盒须证明覆盖编辑及encoder；100seed非逐输入失败概率 |
| T5-small probe | Gaussian特征与文本扰动分开；clean−noisy，CoLA weighted F1、其余accuracy | 冻结probe的有界feature域是新任务，上游encoder不自动在证明内 |
| MiniMind | 等总参数dense/MoE、train/validation loss | 模型/训练参照，不是认证率 |

Gaussian支持无界；有限盒不能直接等同其噪声实验。若增加截断及概率质量，则须新合同。
文本编辑亦不等于连续像素/特征盒。依赖库不是完整作者制品；保存记录的未定位状态不代表
不存在。最佳正则结果的选择集仍需明确，不能从test选配置。
全文SHA见紧凑记录；行段为J `255–276,481–508`、Feature `1297–1309,1359–1388`。

## 7. 检查与下一边界

九组标准库控制覆盖裁剪/归一化与开闭球、极值角点差分、置信合成、并列top-k质量、
非归一权重与后续偏置反例、历史计数、Meta零类别块和身份篡改。
执行命令（ACT既有环境，`python -I -S`检查不加载第三方包）：

```text
python -S -m unittest scripts.test_baseline_semantics
python -I -S scripts/analyze_baseline_semantics.py --check-report docs/baseline_semantic_admission_20261013.json --verify-local-sources
```

几何包含、源码hash和这些控制**不授予**新实验或严格证书资格。
两项独立只读 AI 复核检查了 Dual RS/Meta 和 Robust Experts/RoME；本轮据复核补全
population tie 定义、nominal概率参数边界及双类别块控制。这不是外部人类技术审阅。
另有六项文稿／冻结证据保留回归通过，合计15项；所有15个固定本地来源SHA核对一致。
G6当前明确了比较问题与缺失前提；六篇完整复现/比较、人类技术审阅、干净环境实证复现仍未完成。
G1–G5仍开放，未产生真实模型、外部净收益或GPU加速新结果。
来源/残差容量路线按[联合裁决](hz_residual_template_capacity_20261013.md)停止；
新资源协议仅处于待用户批准的起草/评估提议，未改变8M、300秒、接受门或任何封存边界。
