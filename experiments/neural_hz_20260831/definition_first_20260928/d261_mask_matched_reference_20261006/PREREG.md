# 同 mask 公平参照的完整测试与来源预注册

日期为 2026-10-06 Australia/Sydney。分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。生产 tracked binary diff SHA256 为 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。所有候选源码在首次 import、AST、compile、pytest 收集或数值执行之前由 freeze.json 一起冻结；冻结后不得修改、补跑或复用 RUN。

## 研究问题和固定范围

只检验 CONTRACT.md 的同一 H 上普通 mask 上界，不改候选 e、tau、parent 范围或相位关系。三来源固定为原 CIFAR100 large、CIFAR100 medium、TinyImageNet medium 模型及 D241 原 spec、input_box、prefix/initializer/consumer 记录。模型身份只用于来源认证与固定实验人口，不进入候选规则。原完整登记为 64512、8128、24892 位置，合计 97532；pair/mask 记录为 567、1143、1143，合计 2853。输入至第三个原 ReLU 的全部节点、通道及空间位置均保留；不裁剪模型或挑窗口。

## 首次执行之前

冻结八源为 CONTRACT.md、PREREG.md、source_bounds.py、source_observer.py、test_source_bounds.py、run_math.py、collection_contract.py、run_audit.py。D259 数学收据、全部冻结源码及 archive 只读认证；不从 D259 source JSON 读入候选参数或免费复用其计算结果。数学继承完整原序人口，并仅新增四个 plain tests：

1. test_01_masked_fraction_oracle
2. test_02_original_bn_carrier_preserved
3. test_03_fair_reference_and_population
4. test_04_default_off_budget_and_summary

## 两个一次性阶段

数学 RUN 固定为 results/d261_mask_matched_reference_20261006_v1。使用既有认证 Python、单进程 pytest、importlib 模式、禁插件自动加载和 bytecode、CPU0、单线程及 CUDA 隐藏，全部 4281 项与 230 文件必须有序一致且零 failure/error/skip。单 pytest 总墙钟上限 60 秒；原 AS16GiB 和监督器双 1GiB 宿主观察保留。旧证据 writer 只在各自模块中重定向到本 RUN 的隔离子目录，不更改旧函数代码或旧结果，不全局 monkeypatch Path。新 evidence 为唯一 summary.json。

来源 RUN 固定为 results/d261_mask_matched_reference_20261006_sources_v1。只有完整数学及所有后置检查通过才执行一次 run_audit.py --enabled。复用已认证 D241 监督器的私有模块实例，仅将其 RUN 指向本独占目录；原旧文件不变。完整读取三个原模型和 spec，验证全部 70 initializer、875072 标量、全部消费者及 97532/2853 分类人口。每个来源分别保存原/公平分类与共享 mask 上界。旧排除变成公平未排除视为失败。

来源资源门不变：全阶段 work 256000000、单模型 work 200000000、证据 work 一次预付 40000000、累计 entries 64000000、rational bits 512、完整墙钟 240 秒、AS16GiB、CPU0 单线程、双 1GiB 宿主观察、单 JSON 不超过 8MiB、report 和 exit 共用 64KiB 终态预留。原 evidence 额度在 whole work 中预付，不再重复相加；所有实际 evidence 仍独立计费且不能超过额度。

完整数学收据、旧源、原模型/spec、当前候选、输入及生产 provenance 都在执行前后认证。资源或任何阶段失败即 fail closed，保留异常与部分工件，不运行后续阶段，不在原目录修复或重试。

## 解释边界

本轮新代码默认关闭且没有 solver 调用。来源阶段的 wall time 不是完整网络验证时间；数学通过不转授原生或GPU资格。公平排除增加仅说明部分旧信号能被普通范围解释；公平未排除仍非严格强化或新解。固定正式 baseline 1870/2413 与独立 E0 61/400 全部不更新。

本阶段不注册原生关系安装、shadow、完整2413/400回放或 GPU 实验。后续必须在新的版本中证明真实同源严格强化和全部终端消费，再按原晋级门执行；本次诊断不替代这些工作。

## 前版入口失败与本版身份修正

D260 在唯一数学入口的上一版 freeze 身份检查中失败，未导入候选或收集测试。其源码和失败 RUN 已完整封存，本版不继承任何通过资格或测试人口。这里只修正抄写错误的 D259 freeze 哈希，保持相同算术、观察器、四个测试、全量人口和所有资源门；本版仍从最后成功 D259 的 4277/229 继承。新身份从现有文件的 SHA256 输出读取，不重写旧记录。D260 freeze、archive 和 exit 作为失败 provenance 只读锚定。
