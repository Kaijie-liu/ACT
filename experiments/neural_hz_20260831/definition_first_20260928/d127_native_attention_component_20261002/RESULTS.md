# 原生 Attention 查询的数学实现结果

共享源 Attention 的受限方向查询已完成实现与完整数学测试。它能在保留 score/value 同源关系的普通小例上认证一个指定独立乘积外包无法认证的阈值，但尚未绑定真实模型、GPU 或终端验证路径。这是数学组件进展，不是完整 Neural-HZ、新颖性资格或 benchmark 新解。

## 唯一执行和完整人口

八个文件先由 [freeze.json](freeze.json) 冻结，再唯一执行 run_math.py --enabled。会话 51942 已终止，退出码 0；这些冻结文件和此次 RUN 从此只读，不重跑、不修改失败条件。上一用户回合仅核验状态，属于 no progress；本次实现、执行及下面的来源范围审查产生新的证据，属于 progress。

完整继承 D125 的 3881 项并新增十二项，共 3893 项、198 个测试文件全部通过，零失败、错误、跳过，13 条继承 warning。单 pytest 进程包括启动、收集和 JUnit 的耗时为 47.87927108630538 秒，保持原 60 秒门；pytest 自报 46.69 秒。监督器连同前后身份认证共 61.11768810637295 秒，不是测试超时，也不是速度对比。

CPU1、单线程、CUDA 空、AS16GiB 和完整继承人口均未放宽。inventory 的前 3881 项与 D125 有序相等，末十二项为本次预注册控制；JUnit 的全部实际 testcase 与其对应。监督器 traced peak 为 18402818 字节、tracer metadata 为 7587056 字节，观察门通过；这些指标不等于候选完整物理存储或 GPU 资格。十二个结果工件和八个冻结文件均在执行后重新核对 SHA256 一致。

## 新实现保留和查询的语义

原 System、连续源、全部原 signed bits、EQ/LE 和 frame 原样保留，一个 native fiber 的全部输出共享 score、value 来源及分母。build 缺省关闭。公开元数据在查询时重新验证；假精确性标签、错 frame、直接 binary score/value、范围或预算错误一律拒绝。

每个方向先按同一源合并 value，构造 token 的二维联合像，再扫描边界端点与内部驻点。指数由全有理 Taylor、尾界、向外 dyadic 舍入和倒数认证，不使用 float exp 作 oracle。所有 token 只使用一个共同 score shift。

token 真正来自独立盒且没有源谓词时，实数定理把方向阈值等价地归约为 F(t)<=0；有限实现发布 F 的区间，只在上端非正时认证。共享源或谓词情况下仍是 sound product-box 外包，不把其正下端误当真实反例。临时二维 zonotope 是查询几何，不是整体域退化为凸域。

## 保存的严格正控

[实际控制回执](../../results/d127_native_attention_component_20261002_v1/native_attention_control.json)保存两 token、二维来源的小例：固定 token 为零，另一 token 的 score=x、value=3/4-x+y/4，x,y∈[-1,1]。在 t=5673/10000 下 F_upper 严格小于零，因而后继 ReLU(Z-t) 可认证恒零。

同一测试保留真实 p=sigmoid(-1/2) 的认证区间，构造符合指定标量 McCormick、补概率产品守恒及 energy 行的放松乘积点，其 T_v 下界仍大于 t。不是把任意有理 p 当成真实概率。比较只针对明确的独立乘积参考，不扩大为完整 RLT、Taylor 或其他 Transformer 验证器的优势。

查询实际检查五个多边形顶点和一个内部驻点。回执 work=2846、entries=53、entry_upper=458 是符号计数，不是 bit complexity、运行速度或净内存节省；原 System 仍保留。测试同时覆盖退化点/线、公共平移、联合通道抵消、共享源降级及 fail-closed。

## 接下来真正需要做的工作

[真实 ViT 来源范围](REAL_VIT_SCOPE.md)将该原语定位到两个已存档模型首块的 CLS query，并明确 BN 数值误差桥和所有 96+96 个 CLS 后继方向的接入义务。只认结构匹配，不冒称已读取系数、执行网络或取得来源资格。

普通 CNN/PWA 仍优先。[共同关系审查](CNN_RELATIONS.md)记录本轮的相位抵消证书、下一 ReLU 控制、三相位负例和联合能量投影的代价。它们提供支撑与反例，尚不足以选定完整共同消费者的新实现；不把旧机制重新编号后算作新域突破。

本组件没有实现 source-slope 编译、通用不确定 query、自注意力后层、多头联合精确查询或终端指数求解。候选仍默认关闭。后续须依次通过同结构真实来源、shadow、逐家族及完整 2413/400 回放和四并发不回退门，不能用数学资格替代。

## 资格和账本

[exit.json](../../results/d127_native_attention_component_20261002_v1/exit.json)的 all_stages_passed=true 只指本次 mathematical_stage_only。source worker 未启动；实际模型/原相位列绑定、native HZ、GPU、完整物理成本、shadow、全量回放和新颖性均未取得资格。最近真实来源结果仍是 D120 的 1600 个局部读出，下一 ReLU 改善为零。

正式 baseline 仍为 1870/2413=1063 CERT+807 validated ADV；独立 E0 为 CIFAR100 25、TinyImageNet 36，共61/400。两边新增均为零。历史账本未改不等于候选已完成保旧，二者不相加。Goal 保持 active，完整 CNN、smooth/Transformer、GPU、13 家族及新家族目标均未缩小。

## 保存与来源

2026-10-02 Australia/Sydney；redu-hz；HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；tracked binary diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5，与本次开始相同。只写当前隔离研究与结果目录及新的恢复入口；历史模型、结果、冻结源码和生产代码未修改，无 commit/push。

freeze SHA256 为 e58a5df0ee7ef6d3928b565ab5671915847dc9378b2354af401116874051a6cd；exit SHA256 为 863baec836336804aaee8d4ec3b14828bd31a0d46d8df0bb753368d55ce3d94c。pages:write-page 用于分开数学、执行、来源推断和正式成绩；保存于既定本地 Markdown，未创建外部 Page，网页渲染未验证。事后归档校验清单不冒充执行前 freeze。
