# 现有归档可支持的联合关系评估范围

只读核查确认：不必重解析模型才能取得一组真实source级输入。D067的身份索引绑定D025已完整保存的CIFAR100 large五窗口归档，其中有共同源区间仿射式、门界和实际接收参数。它足够作为新源级评估的输入材料，但不包含原native相位列、HZ读出或decoder绑定，不能据此安装候选或报告网络新解。

## 原始证据与字段

入口为 `results/d067_fixed_pair_applicability_20261001_v1/complete.json`。其 `archive_path/archive_sha256` 指向 `results/d025_interval_capacity_20260930_v1/complete_0.json`；本轮重算两文件哈希均吻合。具体SHA见本目录清单。

| 材料 | D025原字段 | 已保存范围 |
| --- | --- | --- |
| 输入盒 | `.box` | 3072个原输入坐标 |
| 共同源形式 | `.result.source_forms[key].form` | 1600个区间仿射形式，bias及原输入系数 |
| 门界与身份 | 同记录`.bounds/.original_phase` | D067的`.source_records[].archive_form_key`逐一引用，保存`bound_recomputed_equal` |
| 接收参数 | `.result.receivers["0"][channel]` | 64通道，每个576项weights，bias、alpha、shift |
| 窗口与读出 | `.result.windows[]`及`.rows[]` | 5窗口、320行，source_slots、source_bounds、original_phases、receiver_coefficients_ref和ordinary_bounds |
| 同一frame | `.result.frame_identity` | 模型SHA与`modelInput`；不是D025顶层同名字段 |

模型是CIFAR100_resnet_large.onnx，SHA为 `5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16`。其输入性质身份保存在`.source`。第一原ReLU端口为126到127；source phase形如`["127",channel,row,col]`，这是ONNX语义身份，不是SparseHZono的binary列。

归档的首个source bias和receiver weight已经是非点区间。不得把中点视为实际网络，也不能在中点预激活上沿用原bits而不证明误差。D083的可靠容量、共同c、MIR行及支持误差尚未在这些数据上计算；普通接收界是旧记录，非本轮新结果。

## 明确缺失的资格

D067的`diagnostic.json`明确保存：actual_phase_column_binding_verified=false、actual_model_binding_qualified=false、native_HZ_admitted=false。这些字段不在complete.json顶层；查到null不能解释为已通过或失败的另一次运行。

没有实际HZ的±1/0–1列映射、连续latent读出、原EQ/LE和输入decoder，也没有下一完整残差块数值传播。D081只通过合成来源组件测试，不自动补齐这些缺口。

旧索引保存10个外包crossing源，不证明其相位组合真实可达。92160是旧固定pair到消费者的映射数量，kernel_tables_evaluated仍为0。完整输入范围仅此large归档；不能将原三模型普查的预算失败改记为medium或Tiny成功。

## 下一实验的边界

若研究决定采用新联合关系，可以围绕这份不可变归档新建默认关闭的源级预注册，保留完整320读出及全部原源、参数区间和费用，不因只有少数crossing源而少报人口。先进行所需数学组件资格，再计算同一固定结构规则的正负结果；不能在旧D025/D067目录补写或重跑。

该评估只能判断真实共同源形式上的界与成本，不替代native、GPU、shadow或全量回放。它是一个现成的非空研究入口，不是新baseline，也不使D083成为已确认的新域。

2026-10-01，branch redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。本轮仅以jq读取已有字段、以sha256sum核对字节；没有重算门界、执行候选、解析原ONNX或修改任何旧材料。正式1870/2413、独立61/400不变，formal_gain=0。
