# 真实来源认证与前向消费的缺口

共享容量已有解析强对照，但现有真实来源证据不足以判断它是否适用。瓶颈不是普通显式 HZ 终端完全不读关系，而是缺少跨组的完整同源认证，以及把关系转成下一层可靠界的原生接口。

## 边际记录不能恢复跨组来源

最近真实普查 D261 覆盖 CIFAR100 large 64512、medium 8128、TinyImageNet medium 24892，共97532个位置，2853条 pair/mask 记录。其公平参照后17543个未被充分冗余条件排除的位置不是17543个严格增益，更不是网络新解；Tiny 注册前缀仅剩一个孤立角点，不能围绕该角点改变本轮结构规则。

[D261 observer 的记录构造](../d261_mask_matched_reference_20261006/source_observer.py:383)保存 tau、parent_scales、parent_u、selected_coefficients、e_bounds 和 child 上界，没有完整 rho 形式，也没有 e 的逐源系数。上游 [fused_offset](../d259_shared_difference_source_20261006/source_bounds.py:794)已经把未选择残余归约为标量区间。因此这份记录不能反推出跨组相消。

这不是只差一个字段名：两组取同一个 rho=(R/2)(s-t)，令 e_1=(R/2)(s+t)。e_2=e_1 与 e_2=-e_1 的各自 rho/e 边际范围完全一样，各自 Psi 最大值也同为2R；但两 Psi 之和的最大值分别为4R与2R。只看这些边际数据的算法不可能区分两种情况。

D265 没有实际 source RUN，不能借它认证真实 a/b 系数。它的 [_peel](../d265_joint_source_20261006/joint_pair.py:125)还按当前父对移除不同的两个 Q；不同 pair 的 rest 不能直接当作同一源轴。需要补回两组被移除项并对齐完整并集。

九种 mask 只是几何与界的复用。BN 误差的真实身份是 (model_sha,spec_sha,BN output,channel,row,column)，不同实际标量独立、同一标量由所有消费者共享，见 [BN carrier 记录](../d261_mask_matched_reference_20261006/source_observer.py:565)。相同误差幅值或相同 mask 不能代替这个身份。

## 固定组合先合并再展开的接口规格

后续可研究的接口是出生时固定的 FixedFourCombinations(forward_A,forward_B,theorem_matrix,budget)，不是接受任意 property_query 的支持 oracle。

两输入首先必须拥有实际读出、固定 a/b/tau/mu、父归一化、方向和来源认证。目前这类真实完整包尚未取得。四个组合由统一定理及已绑定实际前向读出确定，不按模型／家族身份、输出待证性质、margin 或终端状态选择。原 spec 身份、输入盒及其可靠界仍是必须认证的输入，不在此禁止之列。

将组合代为原 g/x/q 的线性表达式，按同一个 port 和实际位置先合并外 Conv/BN 系数，再展开仍非零的仿射 DAG 到共同 Q、A、BN epsilon 截面。四方向共享源轴索引并流式归约，避免生成四份完整 rho/e 矩阵。此过程不穿越 ReLU 做反向松弛，不运行 Gram 或乘子搜索。正守卫底值的一般情况应使用已证明的十面版本并如实支付费用，不能用四面签名伪装覆盖。

必须保留全部 Conv/BN bias、外层均值与差值对同一 epsilon 的系数、中层和 skip epsilon、父归一化项 -a*E/s，以及两父对剥除项的并集。未绑定、位长、误差、资源或身份失败时不发布证书。

固定跨 pair 邻接人口尚未注册；原逐 pair 的97532人口不能冒充跨 pair 覆盖。一个可审查的未来选择是仅按原 slot 顺序组织不共享父的邻接组，并把边界／未匹配项保留为明确记录，而非按历史收益选择；这只是提案，不是已获资格的扫描配置。

费用至少包括认证读取、完整仿射乘加、所有 source 合并、误差界、支持、存储和证据。若该展开实现保留 P 个乘加贡献，仅乘加至少2P算术工作；这不是所有算法的下界。D261 已用255549650/256000000 work，而 D265 准备下界68890624已超过可替换空间66679110。不能简单把新功能追加到旧账或因为少存矩阵就宣布过门；须有完整的替代路径账。

## 当前生产如何消费谓词

显式 SparseHZ 的仿射、Add 和 ReLU 保留 EQ/LE，普通终端 _lower_hz_milp 读取 Auc/Aub/ub。这条路径不能被误报为“新增行被丢弃”。

但 [sparse_hz_fast_bounds](/data1/Kane/FSE/ACT/act/back_end/solver/solver_hz.py:1875)只用 c、Gc、Gb 的绝对值和，不读取谓词；[_sparse_fact](/data1/Kane/FSE/ACT/act/back_end/hybridz_tf/hybridz_tf.py:504)直接调用它。只增一条 JG 行而不改变读出矩阵，前向界不会自行变紧。

最小消费边界如下，当前均未接入：

| 路径 | 必须使用认证界的位置 | 不能省略的条件 |
| --- | --- | --- |
| 显式 ReLU | [最终 lb/ub 与 crossing 分类前](/data1/Kane/FSE/ACT/act/back_end/hybridz_tf/tf_mlp.py:404) | 完整同 H 读出和可靠向外界 |
| lazy/selective | [N/P/U mask 产生前](/data1/Kane/FSE/ACT/act/back_end/hybridz_tf/tf_cnn.py:738) | core 与 stable-positive expression 全部包含 |
| deferred | [提前生成 ReLU 输入界时](/data1/Kane/FSE/ACT/act/back_end/hybridz_tf/tf_cnn.py:1270) | 出生界、precomputed 界及后续 Fact 一致 |
| lazy terminal | [ASSERT 分支](/data1/Kane/FSE/ACT/act/back_end/hybridz_tf/hybridz_tf.py:822)及[verifier 获取输出](/data1/Kane/FSE/ACT/act/back_end/verifier.py:652) | 完整物化或完整原生表达交接，不能只取 core |

当前 lazy ASSERT 明确 drop，verifier 只取得显式 sparse 或 dense HZ；D257 私有零值谓词载体未改写生产绑定。若只在后面收紧 Fact 而不更新 deferred 出生状态，还会触发 deferred_relu_interval_bounds_mismatch。上述均为静态源码发现，不是本轮运行失败。

## 文献与创新边界

[Sharp Hybrid Zonotopes](https://arxiv.org/html/2503.17483v2)已经区分相同整数集合和不同连续松弛的表述强度，并使用 RLT 强化 HZ。因此原整数具体化不变、新线性界更强，并不单独构成新集合域的证明。本地 D029、D062 已有共同 max-affine 支持，D209 已有非负组合；本轮不重记这些机制的新颖性。

[Mao 等的 Expressiveness of Multi-Neuron Convex Relaxations 第5.1节](https://arxiv.org/html/2410.06816v4)证明，在复制输入的等价网络变换下，理想逐层多神经元凸包可达到完整界；论文也明确指出计算这种理想凸包可能昂贵或不可处理。它提醒我们保留跨层来源的重要性，却不提供免费的高效凸包算法，也不让“复制共同输入”成为新颖贡献。其凸松弛不完备结论不能直接当作保留整数相位的 Neural-HZ 的全面不可能性定理。

本轮定义仍应标为候选观察域，不能宣传为独立创新已经成立。真正下一道判断是：原生同源接口能否以付清的成本在普通真实结构中产生旧接口得不到的可消费关系。若做不到，应修改域假设；不靠再加手工例子、测试数量或包装 helper 替代它。
