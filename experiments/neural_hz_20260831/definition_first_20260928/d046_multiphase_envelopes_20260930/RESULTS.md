# 多原相位关系组件的验证结果

本轮把跨领域文献中的固定超模链上界与前向差分关系结合，形成保留多个原二元相位的仿射包络组件。唯一冻结执行通过全部 3777 项、171 文件的数学组件门；没有执行真实网络求解、GPU、shadow 或全量基准回放。正式增益仍为零，整体 Goal 继续 active。

这不是只压缩矩阵的工作：候选规定观察的共同赋值语义、不同原 bits 的关系语言及跨混权仿射和 ReLU 的前向变换。但载体的 reduced-product 组织、差值与伴随量、超模链本身都有先例；通过测试不等于已有 PLDI 级定义创新。

## 数学上得到的内容

[THEORY.md](THEORY.md) 保留原连续因子、全部原 bits、EQ/LE、共享 frame 和 decoder。读出的上下界同时依赖多个不同原 bits，不把这些身份合并成一个相位，也不在运行时枚举相位组合。零预激活的两种合法原相位均保留。

混权仿射按有符号系数传播同一赋值上的界；ReLU 正部上包络使用固定结构顺序的 prefix 边际，下包络使用 singleton 边际。一个原 bit 时对应已通过的单锚四端点规则。特别注意：singleton 下包络只保证二元语义，不是整个连续 cube 上 hinge 的逐点下界。由真实整数关系导出的线性行可以用于连续查询，但原域没有被替换成凸域。

现有 Fraction 实现只是有限子类：上下包络必须在整个独立 bit cube 上次序相容；它会保守拒绝仅依赖原谓词才有序的合法观察。形式 token 不认证实际网络、相位列或门前提。位宽 512、输入稀疏出现次数 65536 的拒绝保留，失败不代表 UNSAT。

[CONTROL.md](CONTROL.md) 的普通三输入、四 ReLU 混权网络给出新关系 `t-r<=alpha/4+eta/2`。结合原幅度行可证物理读出

```text
Z = t-r+(q-x)/4+(p-y)/2 <= 3/4.
```

预登记比较点满足完整真实前缀的标记凸包、该连续前缀上最后门的精确图、全部六对 D020 的 24 行及所列逐单锚行，却有 `Z=63/80`，分离为 `3/80`。真实输入 `(-1,-1,-1/2)` 达到 `3/4`。这仅证明相对于明确列出的比较系统的局部严格性，不证明超过完整 PRIMA、任意观察闭包或全网理想凸包。旧松弛点不是 ADV，纸面性质不是正式 CERT。

第二个实际混权块的测试确认相同原 alpha、eta 可继续向前携带，未宣称第二次严格分离。支持集可能随深度增长；尚未证明在真实 CNN 上能以可承受成本持续改善。

## 单次执行证据

冻结文件为 [freeze.json](freeze.json)，执行记录为 [exit.json](../../results/d046_multiphase_envelopes_20260930_v1/exit.json)。首次执行前已冻结六份来源文件和八个新测试名；没有先导入或单跑候选选择结果，没有失败重跑或放宽门。

| 检查 | 实际结果 |
| --- | --- |
| 完整继承人口 | 原 3769 tests、170 files 全部保留 |
| 新人口及合计 | 新增 8 tests、1 file；3777 tests、171 files |
| Collection 与 execution 合计 | 56.903497 秒，通过原 60 秒门 |
| pytest 输出 | 3777 passed，13 warnings，43.94 秒 |
| 监督器总时长 | 65.310893 秒，包含来源认证等，不冒充测试合计时长 |
| JUnit 与精确 node IDs | 完整匹配，无 failure、error 或 skipped |
| 源码、输入、生产 provenance | 前后无漂移 |
| 监督器资源观察 | 通过原门；不代表子进程聚合物理峰值合格 |
| 终态 | supervisor_exit=0，无额外 worker，进程已结束 |

CPU affinity 1、AS 16GiB、单线程、assertions 开启、bytecode 关闭及 CUDA_VISIBLE_DEVICES 为空均按预注册执行。原 whole/branch/evidence/retained caps 未改；本次数学执行未授予完整物理资格。监督器保存 collection、tests、JUnit、inventory、preregistered、exit 和工件 SHA256；终态后再次只读核对六份冻结源码及全部运行工件摘要一致。

八项新测试覆盖默认关闭及真实 token 区分、二元包络和连续下界反例、单锚对应、异号 literal、混权合流、四点前缀见证、六对旧行、物理取等、后续混权传播、终端行及拒绝前提。小维有限真值核对是单元测试，不是候选运行时相位搜索。

## 真实接入和 GPU 尚缺什么

`native_HZ_admitted`、`actual_phase_column_binding_verified`、`source_census_qualified`、`gpu_computation_completed`、`complete_physical_qualification` 均为 false。这里的通过仅是 mathematical_component_gate_passed。

下一步优先核对真实原相位列与普通混权消费者的统一来源，不再以扩大数学测试框架代替实际适用性。只读检查 [现有稀疏 ReLU 构造](../../../../act/back_end/hybridz_tf/tf_mlp.py) 显示：在 `sparse_hz_apply_relu_exact` 这条具体路径，原 z 为 -1 时 active，+1 时 inactive，故本组件的 active 指示应为 `(1-z)/2`，不能凭习惯写成 `(1+z)/2`。实际 slot 由 `(frame_id,layer_id,neuron)` 绑定；稳定门可能没有原 bit，不能给每个归档张量坐标擅自创造原相位。该静态推导不授予其他构造路径或完整接入资格。

后续候选需要在实际同结构源上证明幅度种子和消费者映射，统计非平凡多相位关系及支持增长，再检查真实终端收益。非点 BN 系数不得取中点；未通过的区间草稿不能直接接入。GPU 的分段扫描与 clamp 具有可并行构造，但认证舍入、读出展开、正向和转置、证据、重构及原 HZ 共存峰值仍须完整付费。本轮未重跑旧 GPU 初始化失败版本，也没有新增 GPU 性能声明。

## 文献定位和存档边界

固定链仿射上界的直接先例见 Iancu、Sharma、Sviridenko 的 [Supermodularity and Affine Policies in Dynamic Robust Optimization](https://dan-a-iancu.github.io/publications/supermodularity-affine-policies-dynamic-robust/supermod_robust.pdf)，2013，Lemma 1。这里仅使用一条固定结构顺序，不声称得到涉及所有排列的完整凹包络。差值加伴随的先例及跨领域选择已在 [定向综述](../../literature_definition_decisions_20260930/REVIEW.md) 记录。潜在贡献仍要落在适合神经计算的可组合关系语言、统一构造与完整成本，而非这些已知工具的换名。

日期 2026-09-30；分支 redu-hz；commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。正式 baseline 仍为 1870/2413，即 1063 CERT 与 807 validated ADV；13 家族及每个旧解的保全要求不变。独立 E0 为 CIFAR100 25、TinyImageNet 36，共 61/400，不与 1870 相加。本轮 formal_gain=0，未改默认路径、未 commit/push。

仅新建当前实验目录及唯一 RUN。旧冻结实验、D045 未执行草稿、历史模型、日志、表格和 /data1/Kane/HyZor 均未修改。原九项 tracked 修改仍是 3806 insertions、57 deletions。依据 write-page 技能，将已知文献、数学推导、实际测试与未完成资格分开记录；文档保存在本地，没有发布外部 Page。

关键摘要如下，其余见本目录 SHA256SUMS 与运行 manifest。

```text
e0a0569b0854a7c6cfa7b01a76d2e80075a13558ebdd816d2e08fa310cb33d42  freeze.json
a4a5eef3249b0161b0be4322370faba4c42bdfd79d77f4bde3720f5757b99904  RUN/preregistered.json
3e49d547c78940bb6b7428119701e950fc47fb21921eff8f4cd1b5fc16a9d231  RUN/inventory.json
267200d05c1aab7c1aa881cad3d73eaf24762f0596e3cdc8a581380fb1ce174a  RUN/tests.xml
1051d8d61e6d671938b215cfa86df7506ac8161d6bd11a42920a68c385727087  RUN/exit.json
```
