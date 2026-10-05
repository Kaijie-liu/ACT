# 真实结构入口和已有方法的比较边界

本轮只读保存的两份原始 ViT 图，找到第一 CLS 行的固定 query 和固定残差入口。此前对一般 QK 的二次性判断仍正确，但不能用来否决这一结构子类。以下是元数据加算子语义的纸面推论，不是新模型解码、数值认证或执行结果。

## 固定 CLS 行在原图中的完整路径

证据来自同一已完成图诊断的 [IBP 图](../../results/d107_vit_graph_inventory_20261002_v1/ibp_3_3_8_graph.json)和 [PGD 图](../../results/d107_vit_graph_inventory_20261002_v1/pgd_2_3_16_graph.json)，不修改原记录。

共同节点链为 14 ConstantOfShape、15 加固定 cls_token、16 在 token 轴前置 CLS、17 加固定 positions。固定输入 shape/batch 后，首 token 与图像的数值无关。这里不需要猜 ConstantOfShape 的浮点填充值为零。

18 Transpose、19 五输入且四个统计/仿射参数均为 initializer 的 BatchNormalization、20 Transpose、24/25 query 仿射，故在 BN 参数有效且真实表达式认证后，首 CLS query 是常量。26/27 的 key 和 28/29 的 value 对每个 patch 的输入仿射。因此 51 的首 query 行、经固定 52/53 scale 后，是逐 patch 仿射 score；其他 patch-query 行一般仍是二次的。54 Softmax axis=3，55 是对应 value MatMul。

62/63 是 48 维输出仿射；64 的残差输入正是节点 17，所以首 CLS 的残差同样为常量。后续 65/66/67 为固定 BN 布局，68/69 是 48 到 96 的仿射，70 为真实首 ReLU。因此全部 96 个 CLS 前激活都是常数加三个 head 的固定线性读出，可以消费单头方向界。固定前向块内合并这些 affine 读出不是基于 terminal property 的 backward rescue。

73 加回的已经是非常量 attention 输出；后续层不能无条件沿用 fixed-query/separable oracle。其余 tokens 仍被下一 attention 消费，不得因只研究 CLS 而删除。

## 来源独立性及三头共享

首 patch Conv 的 kernel=stride，padding 为零：IBP 为 8，PGD 为 16。其间 reshape、position、固定统计 BN 和投影都逐 token 工作，没有 LayerNorm。因而各 patch 的原像素支持不重叠；只有原性质确为 product box、没有额外跨 patch 约束，且浮点常数和 BN 有效性认证后，才能使用连续盒上的 exact 单头定理。

原 bit 若实际影响此处的来源、或使用跨 token HZ 谓词，不能悄悄将离散/相关来源当成独立连续盒来声称 exact；盒外包最多保持 sound。没有任何 bit 被删除或放松到实际域中。

三头必须使用同一组 3072 个像素身份：

```text
y_h(x) = sum_(i=0..P) exp(s_hi(x_i)) V_hi(x_i) / D_h(x)
a_CLS(x) = c + W_O concat(y_1(x),y_2(x),y_3(x)).
```

i=0 是 CLS self-key/value 常量项，没有像素生成元或 polygon 边，但每次阈值评估必须消费；不能只遍历 P 个 patch 项。

对固定方向 d，`sup d*a_CLS <= c_d+sum_h sup d_h*y_h`；不是等号。分别求界只是查询外包，不在域里克隆三套独立源。界的差异必须在同一 headwise 聚合参考下比较。

当前 sparse MatMul 的中点公式以及共享 key-difference 公式，在某个 query 的可靠宽度真正为零时，双线性乘积余项已经消失。它不代表整个输出精确：sparse 路径仍有 256 项截断、omitted 补偿及向外舍入，dense 路径也不因单行固定而必然走全 tensor 常量捷径。因此不能把“发现 CLS 常量”本身称为旧实现遗漏的精度收益。新候选要检验的是保留 score/value 同源图的后续消费；当前也没有认证转换后 native query 的行宽实际为零。

## 真实人口和代价

| 原模型 | patch 数 | 每 patch 像素维数 | head 数和 value 维数 | 第一 ReLU 全人口 | CLS 子人口 |
| --- | ---: | ---: | --- | ---: | ---: |
| ibp_3_3_8 | 16 | 192 | 3 头各 16 维 | 1632 | 96 |
| pgd_2_3_16 | 4 | 768 | 3 头各 16 维 | 480 | 96 |

每个输出方向在三头合计最多 9216 个投影生成元、18432 条 polygon 边；二维系数展开最多 18432 个标量。若完整展开 3 个 scores 和 48 个 values 的仿射 bank，是最多 156672 个像素系数，仍只有 3072 个源身份；因式表示和物化费用不能同时按各自较小值记账。

两个模型各 96 个 CLS 前激活的上下界共涉及 1152 个单头标量根问题。若每根 q 次评估，每模型最多 3538944*q 次边访问；这不包括三个 head 的 CLS 常量项及其他算术。正负方向可以共用同一个二维多边形，但改变 value 方向通常仍需新角序。固定方向内的边端点指数可以缓存，驻点和 root 的实际开销仍要计入。

这不是已预注册执行人口。下一实际研究可以固定全部 192 个 CLS 前激活，保留所有稳定、零系数和无收益成员，同时明确它不覆盖其余 token 或整网。数学、来源、GPU 和正式回放资格仍逐层取得，不得为了源提取或性能失败减少此人口。

## 数值绑定仍缺少的证据

本轮没有读取原模型浮点 payload 并执行新算术。仍需认证所有原常量有限、BN 的 variance+epsilon 正、真实平方根/除法、固定 scale、原 VNNLIB box、batch 绑定以及转换到 ACT 后的模型和 frame 身份。中点折叠不能冒充原实数表达式。只通过保存的元数据无法给出这些证据。

算子规范采用 [ONNX BatchNormalization](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html)、[Conv](https://onnx.ai/onnx/operators/onnx__Conv.html)和 [Softmax](https://onnx.ai/onnx/operators/onnx__Softmax.html)；本轮沿用原 opset9 结构解释，不执行 opset 转换或 shape inference。

## 主文献的实际覆盖范围

[Bird 等的 HZ 论文](https://arxiv.org/abs/2106.14831)将有限 HZ 解释为有限多个 constrained zonotopes 的并。这里对 sigmoid 源图的非多面体证明是本轮自含推导；它不是一般非线性集合表示的新颖性证明。

[Combastel 的 functional sets](https://arxiv.org/abs/2009.07387)已有 typed continuous/discrete symbols 和函数像的语义。仅引入“原 HZ 加具名函数图”不足以构成本项目创新。

[Dinkelbach 1967](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.13.7.492)是分式目标零点转换的经典先例。本轮不能把阈值化本身作为新定理贡献；候选的具体增量是 correlated dynamic value 下的低维局部最大化和可消费关系编译。

[Vertex-Softmax](https://arxiv.org/html/2605.10974v1)的 v1 为 2026-05-08。第 2 节式 6 明确 value 系数相对 score 固定；第 3 节给独立 score box 的阈值顶点算法；第 4 节式 15 至 18 先界定 value，再调用该原语。式 24 已有逐头求界后相加。因此本轮的潜在区别只在 token 内 score/value 同源查询，不包括逐头聚合，也不代表已胜过其完整验证器。

[ZonoGPT](https://arxiv.org/html/2609.34457v1#S4.SS2)的 v1 为 2026-09-28。第 4.2 节式 14 保留多头 attention 和残差，式 15 至 18 使用共享源 Jacobian 与 Taylor 余项，式 20 和定理 2 给 structured-zonotope 外包。当前 native 图及受限几何查询与其不同，但尚无同信息端到端或完整外包比较，不能宣称精度/速度更优。

先例检索仅缩小了待比较差异，并不证明不存在相同方法，更不证明达到 PLDI 水平。PWA 侧也必须与既有 D015 分解、D123 零误差证书及旧 HZ 加相同行比较。

## 来源身份

日期 2026-10-02 Australia/Sydney；分支 redu-hz；HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。

IBP 原图 JSON SHA256 为 b4a4436858aa08f23fcc8791c3718367d383ed9c8896c56eec997103a6dac1d0；PGD 为 d5d0f9e7d273fb277e8c711fb44015a51a461e1267c28487adc517b8330c2d9f；原诊断 exit.json 为 26f6462208364a6a36696043f9cdd4e07061b8001f3c7516edd2f70dd5f3fa9c。原模型哈希和全部解码来源仍以该 RUN 的 preregistered.json 为准，不在这里重写。
