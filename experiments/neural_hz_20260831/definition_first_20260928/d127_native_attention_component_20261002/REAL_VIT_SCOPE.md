# 首层 CLS 查询的真实结构与系数误差桥

两个已存档 ViT 的首块 CLS query 在结构上符合固定 query、逐 token 仿射 score/value 的条件。当前 Attention 组件不能直接据此声称模型绑定：原 BN 的平方根系数尚未认证，也没有解码数值参数。本页记录已核对的结构、一个可证明的系数外包桥以及接入的完整范围，不是来源执行或新的候选冻结。

## 已核对的原图

本次只读 [D107 IBP 图](../../results/d107_vit_graph_inventory_20261002_v1/ibp_3_3_8_graph.json)、[D107 PGD 图](../../results/d107_vit_graph_inventory_20261002_v1/pgd_2_3_16_graph.json)及退出记录，重新计算三个文件 SHA256。图中的 network_executed、shape_inference_executed、native_hz_source_binding_verified 均为 false；该来源是已保存的原图结构，不是原生域资格。模型名称中的 PGD 是历史训练模型名称，本研究没有执行 attack。

| 结构 | IBP模型 | PGD模型 |
| --- | --- | --- |
| 来源 | vit_2023/onnx/ibp_3_3_8.onnx | vit_2023/onnx/pgd_2_3_16.onnx |
| 图文件SHA256 | b4a4436858aa08f23fcc8791c3718367d383ed9c8896c56eec997103a6dac1d0 | d5d0f9e7d273fb277e8c711fb44015a51a461e1267c28487adc517b8330c2d9f |
| 图记录的原模型SHA256 | 9cc53b9edb35d40d70de1d816008a434c7e60b0404cf3cb38b9aefc189692883 | 246326387574617a274b6f73f7f52771df1195e03c99065ce4ad18c46b8e0984 |
| 输入与patch | 3×32×32，8×8不重叠patch | 3×32×32，16×16不重叠patch |
| token数 | 16个patch及1个CLS | 4个patch及1个CLS |
| attention | 3 heads，每head 16通道 | 3 heads，每head 16通道 |
| 首MLP矩阵 | onnx::MatMul_308，[48,96] | onnx::MatMul_224，[48,96] |

D107 exit SHA256为26f6462208364a6a36696043f9cdd4e07061b8001f3c7516edd2f70dd5f3fa9c。本次没有重新读取或执行原模型，只对照存档与其已有 provenance；未来数值绑定仍必须验证原文件。

两图共同的实际节点关系为：节点3的Conv采用无padding、kernel=stride的patch投影；14/15生成与图像数值无关的CLS，16前置，17加入固定positions。18/19/20是Transpose、固定统计BN、Transpose，不是LayerNorm。24/25、26/27、28/29分别形成Q、K、V。51计算QK，52/53固定缩放，54为axis3的Softmax，55计算PV。

因此在固定batch和推理BN语义下，只有首CLS行的query不随图像变化；各patch的K/V在实数模型中是像素的仿射函数，patch来源不重叠。CLS自身的常量key/value项也必须保留。其他patch query通常产生二次QK，后层CLS已依赖输入；一般pre-LayerNorm模型的patch K/V也不是原像素的仿射函数，不能扩用这里的条件。

首CLS输出经62/63的output projection、64的原CLS residual、65/66/67的BN和68/69的MLP仿射，进入70的ReLU。完整首CLS人口为每模型96个预激活，共192个；不是选择几个正收益通道。73之后不再属于该固定query合同。三个heads共用像素，所以逐head界相加仅sound，不是多头联合精确。

## 系数不能用中点冒充确切值

FLOAT权重的存储值可表示为有理数，但BN包含sqrt(running_var+epsilon)，实数折叠系数一般不是有理数。D107图保存了参数形状和protobuf身份，没有保存FLOAT内容的求值认证。必须先认证正分母、平方根区间、CLS及positions、scale、全部projection和后继系数。任何框架浮点运算误差也须按原验证语义明确处理；本页实数公式不自动认证实际float Softmax。

以下为根代理推导、独立审查的标准误差桥，不宣称新颖性。假设一个真实固定query token的score/value系数都有可靠区间。选定确切有理中心形式 ŝ_i(x)、v̂_i(x)，若每个真实系数相对中心的最大偏差为δa_k、偏置偏差为δa_0，输入为x_k∈[l_k,u_k]，则

```text
ε_s = δa_0 + Σ_k δa_k max(|l_k|,|u_k|),
|s_i(x)-ŝ_i(x)| <= ε_s.
```

value方向同理给ε_v。因此真实联合点位于

```text
P_i = image_Xi(ŝ_i,v̂_i) + [-ε_s,ε_s] × [-ε_v,ε_v].
```

这给临时二维查询几何增加两个生成元，但仅针对一个已经合并的scalar value方向，不是给全部value通道永久共享两个误差变量。所有源/谓词/原bits仍保留在native元素内；临时外包只用于认证，不整体替换域。

对任意原真实轨迹，其每个(s_i,v_i)同时落入各P_i。故即便原BN等系数跨token相关，

```text
Σ_i max_(s,v in P_i) exp(s)(v-t) <= 0
```

仍足以认证真实单head输出≤t：左边上界该轨迹的分子差，而分母严格正。扩大为product polygons只放松，不要求误差彼此独立。所有token共用扩大后score上界导出的shift，CLS常量token的误差也计入。所构造外包的exact_product不能转授成原模型精确性，其正下界不能产生真实ADV，各方向的极值点也不能拼接。

## 后继混权和费用义务

对node69的第j个CLS预激活，先将output projection、第二BN及MLP的全部有符号系数合入每个head的scalar value，再做上述查询。所有偏置与CLS residual汇入一次常数区间。若分别认证U_jh，则

```text
g_j <= c_j^U + Σ_h U_jh,
R(g_j) <= R(c_j^U + Σ_h U_jh).
```

不能用负权直接乘某head的上界，也不能重复计算偏置。可靠下界按完整负方向同理生成。对照组必须使用同样的系数区间、源范围、误差桥和后继方向；不能把更精确的参数处理冒充native相关性收益。

两个模型每head覆盖3072个实际像素源；192个后继方向各需三head查询。一般每token的二维像可有2d条边；逐方向展开后，候选指数求值的上界已达数百万，而当前每次指数使用固定64项。没有证明这一全人口实现能落入原256M工作、240秒worker及完整物理门，不能因为结构匹配就忽略查询费用、提高预算或只留下有收益方向。应先把系数合并、几何复用、全部方向、数值证据和GPU方案的实际成本列入新的预注册，再运行数值绑定；不重跑旧D090大归档包装。

本页没有实现该误差桥，也没有执行新来源、模型、GPU、shadow或全回放。它使下一动作具体化，但不转授D127的数学资格。普通CNN共同关系仍为优先研究，完整13家族及独立CIFAR/Tiny目标不变。

2026-10-02 Australia/Sydney；redu-hz；HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。只读图核对与纸面证明；正式1870/2413、独立61/400，两边新增均为零。
