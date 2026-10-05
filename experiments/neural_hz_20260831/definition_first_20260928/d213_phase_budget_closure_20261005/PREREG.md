# 相位条件余量闭包的完整数学门

本轮只执行默认关闭的 D213 数学候选；先冻结本文件、CONTRACT.md、fiber.py、test_fiber.py、run_math.py、collection_contract.py 六项 SHA256，然后才允许候选 import、AST、compile、pytest collection 或数值执行。单次冻结版本不修后重跑；失败也保存全部日志、人口、前后来源核验和已有诊断产物。

继承最后真正通过的 D209 全部4100测试/216文件，追加以下固定12项，唯一人口4112测试/217文件。继承人口仍测试其原冻结实现；D213自身的新增能力由本轮定向测试直接执行，不能把继承人口通过夸大为 D213 已回放13家族。D212是纸面算术参考，不是数学人口资格。D211未冻结草稿不导入、不复活。

## 固定测试顺序

1. test_phase_budget_strict_complete_readout
2. test_d209_positive_cap_multilayer_preserved
3. test_d209_nonpositive_cap_multilayer_preserved
4. test_fifth_certificate_drives_next_birth
5. test_intrinsic_t_bound_identity_and_selection
6. test_small_box_bound_does_not_assume_domination
7. test_zero_caps_stable_and_zero_phase_labels
8. test_full_input_decoder_and_interfaces
9. test_owned_proofs_and_alias_equalities
10. test_complete_two_hundred_gate_members_and_cost
11. test_cumulative_budgets_fail_closed
12. test_all_five_proofs_sound_and_paid

每项为无参数、无装饰器的真实 test 函数。执行前在同一个pytest进程核对完整有序nodeids、源文件身份、继承证据函数绑定；不得选择性collect、删例或以新增断言替代旧人口。

## 必须得到或明确否定的能力证据

D212结构保留完整输入盒、读出、原bits，旧四界为1/3、309/5740、5419/177940、5419/177940，第五界为-1/140，后继ReLU界为[0,0]。fractional点只作连续关系诊断，不计ADV。

D209的正cap与非正cap多层结构在新Fiber内重新构造；新五模式共同决定所有bounds/caps，不注入旧界或挑模式。要求原固定读出能力至少保住-1/32和-1/8；允许进一步收紧，不因新出生变化要求原中间数值不变。

固定新跨层控制使用D212的首bank，额外保留独立根变量t，并记F=q1-(13/35)q2-(13/35)(1+x)。下一bank预激活固定为

```text
g1=t/2+F/16+1/4,
g2=t+3F/32+1/4.
```

两cap应由第五证书严格胜出为1/8、7/8；预期共同余量为-F/14。继续使用固定读出

```text
F2=q_second1-q_second2/2-(1/8)*(1+F/14),
child_pre=F2-1/140.
```

必须实际调用新域支持与后继ReLU，检查第五公式能否证明child_pre<=-1/140；不只查看metadata。所有原接口、全部源/相位、decoder与完整证明都保留，不能依据跑后margin改变这些系数。

内生T控制使用已证明的D209前缀J，固定g1=J+1、g2=J+t/8+1；检查本轮实际父证书、下界证明、剩余证明和T_int恒等式，以及T_int确比自由盒更小并被选用。不导入外部界，不能为通过断言硬编码出生caps。B<1场景保留旧关系与四查询，不假设单独新投影全范围支配。

完整200门仅是 supplied mathematical bank，保持全200原bits和输出，验证实际成员、完整方向的五份证书与累计费用。它不是CIFAR完整14400门、native模型或异质Conv资格，不计正式收益。测试中的固定成员是健全性检查，不是攻击或按样本选择表示。

## 资源与来源

原 whole256M、branch200M、entries64M、512-bit 不变，lineage累计不重置。完整pytest调用含导入、collection和测试，总60秒上限，CPU0、各线程1、CUDA隐藏、AS16GiB；监督器原1GiB与65536 reserve守卫保留。不得把监督器RSS解释成候选完整物理资格，不另加超时续跑。

完整继承D209的7425源码身份及14输入，再加入本轮六源、freeze、继承receipt/产物和D212四项参考锚；实际合并计数运行时记录。保持全部原源/输入pre-post核验与branch/commit/config provenance，不只校验新增测试。不会加载真实模型或启动native/GPU阶段；继承测试历史LP只作原数学对照，不提供候选bounds或救援。

冻结 schema=d213_phase_budget_closure_v1，mathematical_stage_only=true，worker_stage_registered=false，fixed_component_lp_controls_registered=true，solver_rescue_registered=false，negative_audit_only=false，domain_definition_changed=true，new_set_class=false。新域、能力、native、GPU、完整物理和正式收益资格仍须分别验证，不自动继承。

## 只重定向继承证据写入

为不覆盖历史产物，仅在认证后改变继承测试的证据目的地，不改测试正文、断言、导入域、参数或数学运算。

- D112：原四测试模块的RUN改成本次inherited_d112_controls。
- D207：只替换原文件第44行_record，限定原三个JSON；源hash为1c23e38befc50be621e7dacf863f20bddc9ff107b4ef2c3bc3742ee8d6627e02。
- D208：只替换原文件第39行_record，限定closure_counterexample.json、direction_family.json；源hash为4825d15388db02c02ab4918a915baa0501c41849e784b5556a10a949043007e8。
- D209：只替换原文件第35行_record，限定closure_repair.json、positive_slack_control.json、complete_200_gate_control.json；源hash为f150be4bb9e9fdcf4fe4117038ceacebf9410d39085ab86022dacaca0f72af77。

三个_record必须认证module/global/code来源、首行、源hash和会话前后绑定，成功时检查证据完整。历史文件和原RUN只读。

## 结果与后续门

唯一新RUN为experiments/neural_hz_20260831/results/d213_phase_budget_closure_20261005_v1，必须不存在才开始。新证据恰为phase_budget_separation.json、multilayer_closure.json、complete_bank_cost.json；各自独占创建，并自动保存完整tests.log、tests.xml、inventory.json、preregistered.json、exit.json。固定入口是/data1/Kane/miniconda3/bin/python -B本目录run_math.py --enabled。

任何失败保留版本不重跑，数学组件门为false；不得靠成功的少数新例掩盖多层回归。全部通过也仅数学组件资格，正式1870/2413和独立E0 61/400不变。接着必须面对完整真实结构及全端到端成本，再进入shadow、逐家族、完整2413与独立400回放，保每个旧解与零invalid。没有完整证据不得默认启用。
