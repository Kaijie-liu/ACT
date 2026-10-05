# 下一真实关系算子的完整父状态

只读审计找到了比旧首层配对人口更合适、且不需要新建 loader 的原生来源：Tiny 的 ADD75、Flatten76、Dense77、ReLU78 这一完整 200 门 bank。它提供已有的共同父状态与原相位归属，适合检验 Neural-HZ 关系算子的数学适用性。这是一个来源候选，不是已经取得新接入或收益资格。

## 为什么选择完整末端 bank

D179 的三个 CNN 预终端匹配只提供图形状，不能将宽度当跨零人口。D120 的下一层跨零记录分别是 2/320、26/640、14/640，但均只覆盖旧五窗口，也没有认证完整原生相位列。

C5 的 Tiny ReLU44 整层有 6272 个输出，其中外包 crossing 为 157，稳定正为 124，稳定负为 5991；精化后新增 80 bits。这个接口曾在运行中得到完整认证，但 layer44.pickle 没有保存全局 frame_widths、relu_slots 和 aux_slots。出生前表达式、完整 bounds 和原槽的运行时证明不能自动转成“加载该快照即可复用”的证书。

C34 的 terminal_gate.json 则明确记录所选 ReLU78 新增 200 个原相位和 400 个连续槽，正负稳定人口均为零。其完整 native checkpoint 保存了该 bank 的 preactivation expression、views、entry slots、原 apply 参数与 facts，以及全局状态。因此优先检验整个 200 门 bank，不挑一两个有利门，不按历史 margin 或公开 SAT 标签选择。

该来源的历史 iid 标识仅用于 provenance。候选规则仍必须适用于同类完整数学结构，不能包含 iid 特判；一个来源正例更不能代替 CIFAR、Tiny 或家族 replay。

## 已保存的内容与恢复边界

工件根目录为 experiments/neural_hz_20260831/results/c34_changed_terminal_20260911_v1。

```text
native_state.pickle
bytes=506666963
SHA256=cb5170da0473c01f5304ed307105df0f4aa09c4115af330126fd062ee7bbce3a

terminal_gate.json
SHA256=f2b0729982efd1af933762b00eae4aee79b758b77353a8f4bf22bd30b0382906
```

主代理本轮重新读取生成代码与 gate，并重新计算上述两个文件的哈希，均匹配。未反序列化 pickle。

c34_changed_terminal_worker_v1.py 保存完整 net、HZ/expr caches、frame_widths、relu_slots、aux_slots、input_hz、final_hz、性质与 terminal kwargs；还保存 spliced_state_fields 和 runtime_numeric_roots。后者经 c9_live_runtime_v1.py 和 c32_splice_binding_v1.py 保存所选 bank 的原出生接口及 original_source_fields.hz。这个 pre-HZ 不是用独立区间替换出来的父状态。

已有 c34_portable_restore_guard_v1.py 的成功恢复先例。它首先认证冻结来源和 checkpoint，恢复完整 splice 关系，然后显式恢复 RHS 共享关系并核对 final/input/property 身份。裸 pickle.load 不等于取得这些资格。主代理本轮核对该 guard 源码 SHA256 为 b7bc697fe14499a800f7c6b68490be01dd4081198175c6204e23fb7067942eaa。

原网络来源为 /data1/Kane/data/vnncomp2025_benchmarks/benchmarks/tinyimagenet_2024/：

```text
onnx/TinyImageNet_resnet_medium.onnx
SHA256=234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776

vnnlib/TinyImageNet_resnet_medium_prop_idx_3553_sidx_3392_eps_0.0039.vnnlib
SHA256=812bec2c0362d92d123380df161e1da6d5addbc84a27304d0a079090e814f5c7
```

模型和原性质的这些哈希来自本轮来源审计读取的历史 provenance，本轮未重跑原模型。C5/C34 执行使用的完整转换性质另有 SHA256 d105a0c7ca711eb46b9f20ab772a0564444569d06f6206bca0e95d5a19990cca，不能与原 VNNLIB 字节混称。

## 下一步的有界问题

新运行前须冻结一个完整结构配对人口及其系数规则，并认证所有 200 门的共同父表达式、原 phase 端口、全部活消费者和 decoder。先检查 [共同松弛量规则](THEORY.md) 在这些完整父形式上是否产生非零、可消费的关系，再决定是否实现新生成器。稳定、无共同项、无收益、资源失败均保留，不更换人口或事后改耦合系数。

这是新的隔离工作，不重启旧 C34 terminal，不重跑旧 main，不覆盖旧 checkpoint，也不借用旧 solver 结果修复失败。已有普通 MILP 超时仍是 UNKNOWN/零收益。任何需要的恢复只在新的冻结运行中调用已认证的恢复函数，所有输出保存到新的结果目录。

成本是实际前提：旧 C34 generation-plus-incremental work 为 255056237，距 whole256M 仅余 943763，不能免费叠加新关系或把构造费用移到 checkpoint 之外。全部完整状态、hash/I/O、恢复、源关系计算、工作量分类、证据和终端开销必须如实列出；不能用独立诊断预算宣称新的端到端路径已经付费。若现存构造路径不足以支付，就需要在同一新域路径中证明代价改善，不能事后提高旧门。

同样不能拿 D149 的 64-phase/128-coordinate 小型 reference 包住这个 200 门来源，分批隐藏其余状态，或扩大上限冒充继承资格。当前选择是先解决实际共同父接口与关系的适用性，不再造断开的 adapter 或辅助 verifier。

## 本轮未执行的事项

没有新的数值普查，没有恢复 pickle，没有模型 forward，没有创建候选/runner，没有数学门或真实 replay。以上 evidence 改变了下一次来源选择，但尚未证明存在有利 pair、原生算子已可用、GPU 可用或正式分数提高。

全部旧资源、fail-closed、非凸相位和基线门保持不变。正式 1870/2413、13 家族以及独立 61/400 仍原样记账。这里不是运行预注册，不授权省略其数学、来源、资源或完整回放步骤。
