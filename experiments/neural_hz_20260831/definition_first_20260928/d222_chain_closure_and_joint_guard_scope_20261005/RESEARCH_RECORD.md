# Neural-HZ 联合表达能力的研究落点

本轮没有得到更强的新 Neural-HZ，也没有新增正式解。得到的是两项有用的数学边界：常数链的全部传递相位行已经完整，流图扩展没有精度收益；在符号平衡源盒上，全部源相位乘积的共同化也不超过完整单门关系。两者使下一步聚焦真正的共同源与多门神经 guard，而不继续搭建等价表示或辅助求解器。完整证明、费用和被否决路线见[理论记录](THEORY.md)。

## 进展和处置

上一轮完成了二维模板的覆盖必要充分条件、径向修正边界和已知真实 rank4 块的排除，属于 progress。本轮开始核验其 ARCHIVE.sha256，13 项均通过。

本轮属于 progress，限于纸面定理和明确处置：原本考虑的 O(n²) 静态链流图已被零辅助量的完整传递闭包取代。两个独立复核分别给出共同阈值构造和支持函数路线；采用前者归档，并复核其原标签、断点数及完整代价。没有把更贵等价物推进为候选，也不把局部理想性等同于整体定义创新。

FREE 的投影等价定理覆盖任意源维数，但具有明确符号条件和盒源前提；它不禁止 Neural-HZ 变强，只说明强度必须来自额外的真实神经联合约束，而非这些产品本身。D061 已有正控制仍是旧证据。源 packing 与混权超模快捷路线也已记录，不再反复当新想法测试。

## 真实重叠卷积仍是落点

现有 [D217 来源证书](../d217_source_phase_capacity_20261005/REAL_SOURCE_CERTIFICATE.md)对应 CIFAR100 medium 首层通道 44 的四个相邻 crossing 门，共同源到这四门预激活的线性映射 rank 为 4。[D194 接口](../d194_sparse_crossing_source_fiber_20261005/REAL_TARGET.md)保留 75 个原像素：48 个在四门内私有、27 个共享；每门涉及 12 个私有和 15 个共享坐标。这些不是新独立噪声，也未将原源降成四个独立量。

已知缺口不是 loader。[D199 的 private 容量定理](../d199_private_capacity_elimination_20261005/THEORY.md)能重分配每个私有标量的条件值，但没有证明能以同一混合权重恢复全部原像素均值、原界和其他消费者。[D194 条件纤维反例](../d194_sparse_crossing_source_fiber_20261005/CONDITIONAL_FIBER.md)已说明，仅保留旧定义等式与 decoder 不会自动恢复这份共同条件分布。即使标量 COVER 条件成立，这一缺口仍在。

现有来源可检验 COVER 的条件是 r_private_i>=r_shared_i+abs(center_i)，但本轮没有执行该计算，也不将其视为真实接入充分条件。来源在 ../../results/d015_source_shielding_20260928_v2/partial_source_evidence.json 的零基记录 [1]：原 box、first_conv 权重与偏置、first_post_ops[0] 的 BN 参数、model/spec 哈希均存在。有效核为 kappa*kernel，kappa=gamma/sqrt(variance+epsilon)；有效偏置为 kappa*(conv_bias-mean)+beta，不能漏掉 BN 或取未认证中点。

真实消费者必须完整保留：branches[0] 的 Conv3→BN4→ReLU5 含 128 个声明输出通道，branches[1] 的 Conv8→BN9 是 shortcut，side_consumers 连到 Add10。第二支 target_relu=null 是侧边界，不是没有消费者。这里给出的字段定位来自只读档案审查，不是新的数值提取或 Tiny 证书；该来源记录不包含 Tiny。

## 下一候选应回答的问题

下一候选从非凸域元素和具体化关系出发，保存普通重叠 patch 中完整多维输入、原相位与多门幅值的共同可行性。它必须说明 Affine/Conv、共享 Add/Concat 及下一非线性怎样消费这些信息，同时提供原输入重构；不能只展示人为挑选的有利读出。

比较至少包含完整单门关系、已有同源关系、完整 FREE 在其适用范围内的投影，以及传递相位闭包。若候选只是它们的重命名或更贵表述，就记录结论，不继续铺接口。这里不要求击败精确原网络集合，而要求在相同信息和可付完整代价下展示新的可消费精度或闭包优势。

当前尚无符合这些要求的新定义，不将计划写成成果。GPU、smooth activations、Transformer、其他家族和最终全分目标均继续保留；本轮没有取得任何相应资格。既有普通终端和独立具体见证验证边界不变，不引入 attack、PGD、BaB、输入或相位 split、backward/dual rescue 或外部 helper。

## 执行边界与成绩

本轮仅读文档、历史来源与论文相关段落，作纸面推导、独立数学和范围复核，并写入本新隔离目录。没有候选源码、数值预注册、import、AST、编译、collection、测试、模型、LP/MILP、GPU、回放或后台任务。没有修改生产、旧实验、历史模型、结果、默认配置、commit 或 push。

正式成绩仍为 1870/2413=1063 CERT+807 validated ADV；独立 E0 仍为 CIFAR100 25+TinyImageNet 36=61/400，两者不相加，新增均为 0。没有新回放，就不能声称本轮候选已验证保住基线。最近合格数学组件仍是 D214 的 4116 tests/217 files，本轮未重跑；它包含逆出生历史支持传播，不能误称完全没有 backward 计算，也没有在本轮作为 helper 调用。

2026年10月5日，Australia/Sydney；branch redu-hz；commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。已有 tracked diff 的 SHA256 保持 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5；这不是所有未跟踪源码的完整候选冻结证明。文件散列与研究来源见 provenance.json 和 ARCHIVE.sha256。Goal 保持 active。

pages:write-page 仅帮助本地档案区分定理、先例、未解问题和成绩，没有发布外部 Page。归档将读回文本并校验散列；未检查 Markdown 应用渲染，不影响所保存原文。
