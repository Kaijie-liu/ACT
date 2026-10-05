# 共同谓词余量闭包：单次数学实验

本轮检验 Neural-HZ 元素如何保留、组合并向后继非线性传递自己的关系。不是外部 helper 的结果拼接，不以存储少或构造快作为能力收益。候选元素保留原非凸具体化，新增可认证余量身份和统一关系出生/消费规则；不据此宣称新的可表示集合类或已达到发表级新颖性。

## 冻结与完整人口

在本候选任何 import、AST、compile、collection 或数值执行前，冻结本目录六文件：CONTRACT.md、PREREG.md、fiber.py、test_fiber.py、run_math.py、collection_contract.py。schema 为 d209_owned_slack_closure_v1。唯一 RUN 为 experiments/neural_hz_20260831/results/d209_owned_slack_closure_20261005_v1；创建即消耗版本，失败也不改源码、断言、预算或重跑。候选默认关闭，监督器只接受 --enabled。

完整继承 D208 成功的 4084 tests/215 files、7399 source identities、14 inputs、解释器、CPU、decoder/GPU 依赖、project import closure 和生产 provenance。不执行旧 runner 的 main。保留旧有序 nodeids，在末尾追加十六个无参数测试，合计 4100 tests/216 files；新源、合同及 receipt 身份另行加入清单，不把 7399 当新总数上限。

1. test_default_off_and_hz_embedding
2. test_original_graph_and_phase_semantics
3. test_exact_alias_and_signed_eq_witness
4. test_query_witnesses_reconstruct_all_four_bounds
5. test_reject_modified_foreign_and_missing_proofs
6. test_root_box_slack_matches_source_pair
7. test_two_layer_order_closure_is_intrinsic
8. test_positive_cap_shared_slack_closure
9. test_three_bank_mixed_queries_are_sound
10. test_all_modes_are_paid_and_retained
11. test_complete_two_hundred_gate_accounting
12. test_full_interface_and_decoder_preserved
13. test_nonpositive_caps_keep_original_bits
14. test_zero_and_no_shared_slack
15. test_resources_and_exact_inputs_fail_closed
16. test_initial_predicates_and_sibling_identity

## 固定出生规则

CONTRACT.md 为权威数学定义。相邻 crossing gates 固定 a=b=1/2，从共同父域的完整支持查询取得两个差分 cap 及余量证书。四种查询始终全部计算；共同余量按原 LE/端点身份取归一化系数的逐项 minimum。尺度的 B 只由合并后 t0 的完整父因子自由盒求上界，不临时改用支持优化、逐 atom 上界和、四舍五入常量或样本统计。任一 cap 非正时统一 s=l=1，仍保留全部二元位与原图。

新行的健全性由原整数 ReLU 转移和父证书共同给出，不冒充旧连续 LP 的 Farkas 后果。余量不是新独立变量。alias 的 EQ 和共享身份必须保留；不额外声称任意等价证书或 alias 改写后的 B 和第四证书数值不变。

## 两层分离：不仅修复非正 cap

共同源 x,y,t 均在 [-1,1]。第一 bank 为

```text
f1=x+3y+1/4, f2=3x+4y+1/4, qi=ReLU(fi), s0=(3+y)/4,
J=(73/16)q1-(15/8)q2-(13/16)f1-(195/32)s0.
```

完整父支持必须自行证明 J<=0；不得添加 J<=0 输入谓词或外部 cap。第一控制沿用 D208：g1=t/2+J/16-1/8、g2=t+1/4、ri=ReLU(gi)，候选应从自己的非正差分 cap 推出 r1<=r2/2，证明 ReLU(r1-r2/2-1/8)=0。旧 D208 负例仍原样随完整人口运行。

第二控制必须保留严格正 caps。固定

```text
g1=t+3J/64+1/2, g2=3t+J/16+1/4, ri=ReLU(gi),
c=(7/8,5/2), t0=-J/64,
B=1481/4096, tau=1, s=1+J/64, l=2615/4096,
e=884539/610304, p=18305/32768, E=(7/8)s,
W=(e+p)r1-e*r2/2-p*g1-e*E.
```

theta=-J 的来源是第一门 source-phase upper slack 的 15/4 倍，加第一门 active-value upper slack 的 13/16 倍。第二差分证书分别为 (1+t)/2+theta/64 与 (5/2)(1-t)+(5/128)theta。归一化后的 theta 系数逐项 min 恰给 theta/64；不同端点身份不得错误合并。完整父因子盒 J=[-1481/64,1241/64] 给上述 B；不是用经验取值代替。

第四固定支持证书对 32W-1/32 必须恰为 -1/32；其两个正闭包系数为 (32(e+p),0)，所有父读出项精确抵消。于是下一 ReLU 稳定为零。前三项旧证书必须仍 >53/2400，而非删掉旧查询获得的假进步。

强对照使用同一合法第一父点 (x,y,t)=(9/10,-2/5,1/24)：第一 f=(-1/20,27/20)、q=(0,27/20)、J=-4129/640；第二 g=(29399/122880,-289/10240)。第二 bank 的分数诊断点固定为

```text
r1=(7/8)*(g1+e)/(7/8+e), beta1=r1/(7/8), r2=beta2=0.
```

它通过第二 bank 原八图行、旧 D207 自由盒 pair 的全部十行，以及使用紧常量 c 的 labelled 六行，却有 W>1/600、32W-1/32>53/2400。旧 pair caps=(4825/4096,26685/8192)，旧尺度=(4203+64J)/5444、下界1/2。测试直接按精确分数逐行验证这些诊断不等式，不调用 LP、搜索或相位枚举。该点不是 native 整数成员，更不是 ADV；新域的排除证据必须是新 source-phase 行违反，不能仅以 contains 拒绝分数 bits 代替分离证明。也不声称击败旧 HZ 精确整数求解或包含同样新行的 LP。

固定真成员点用于接口和健全性回归，不用有限样本推断全域定理。至少涵盖 D208 所有固定点以及第二控制的内部父点。三 bank 的原五 bits、连续因子、原 decoder 和所有 live readouts 保留。零点两个 signed 标签均须合法，非零点不应任意换相位。

## 其他数学与费用检查

测试 HZ 初始谓词/二元嵌入、Affine/Add/Concat/Select、EQ alias 正负系数证明、proof 完整恒等式、修改/异 frame/丢失前缀拒绝、初始与后生相位身份、零及负 caps、无共有余量、四查询全量支付、caller 收紧预算和精确输入拒绝。所有派生对象须保持新 Fiber 类型及出生 metadata。

完整 200-gate 控制固定为 100 次第一双门结构，不把重复控制称为异质真实网络。保留 200 个原 bits 和所有原图、旧 pair、新行、两条父查询、证书、共同余量、B、四查询、decoder、读出与累计 lineage 账。允许新行与旧行完全相等时复用谓词身份，但对象中保留的重复行也计费。完整成本由实际报告保存，不能只报新增行数或将 proof DAG 视为免费。512-bit 异质分母风险不在本轮偷偷放宽。

## 证据重定向、资源与保留

只重定向旧测试的输出目的地：D112 原四测试 RUN 到 inherited_d112_controls；认证 D207 _record 身份后，其固定三 JSON 到 inherited_d207_controls；认证 D208 _record 身份后，其 closure_counterexample.json 和 direction_family.json 到 inherited_d208_controls。保持固定文件名、独占 x 写入、有限 JSON 与原数学测试体。不得替换断言、域函数、Path 或 os。

新控制在同次执行写 closure_repair.json、positive_slack_control.json、complete_200_gate_control.json，包含实际证书、全逻辑账与未获资格标志。统一自动保存清单、inventory、log、JUnit、证据、pre/post source/input/provenance 检查、telemetry、artifact hashes 和 exit.json；失败亦保留。

解释器 /data1/Kane/miniconda3/bin/python，认证 CPU [0]、数值线程1、CUDA 隐藏、RLIMIT_AS 16 GiB。完整 pytest 子进程启动至 JUnit 的 60 秒门保持，不选择性预跑。监督器完整 pre/post 哈希时间单报；其 RSS high-water 增量+65536 及 trace peak+metadata+65536 均<=1 GiB。不是 child RSS 或候选完整物理/GPU 资格。逻辑 whole256M、branch200M、entries64M、512-bit 不变。零失败/error/skip/缺失/替代才通过数学门。

## 资格和研究边界

domain_definition_changed=true、negative_audit_only=false 仅说明本轮改变元素 metadata 与关系演算，不能推出新颖性或正式解数。全部 source census、真实模型、native phase、GPU、完整 physical、shadow、逐家族与完整回放资格保持 false；formal_gain=0、new_benchmark_solves=0。通过仅意味着这一数学组件及两层分离按合同成立，后续必须转向真实同结构能力实验。

非负线性组合证明和模板关系域已有研究基础，不作为本工作独立新颖性。参照 [Fouilhe 等的多面体正确性证书研究](https://arxiv.org/abs/1304.0864) 与 [Sankaranarayanan 等的固定模板线性系统分析](https://theory.stanford.edu/~srirams/papers/vmcai05.pdf)。本轮待检验贡献仅限原非凸相位/谓词身份下，确定性共同余量出生与下一非线性消费的组合规则；不把既有 Farkas 证书换名为 Neural-HZ。

2026-10-05 Australia/Sydney；branch redu-hz；HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；预期 tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。所有写入仅本新目录与唯一新 RUN，历史与 /data1/Kane/HyZor 只读，无生产修改、commit/push 或默认启用。正式 1870/2413 与独立 E0 61/400 不变。
