# 作者基线继续部署与 ACT 语义扩展

PI本轮明确授权：逐项关闭部署阻塞，并扩展ACT入口接收类别分离top-1、
多层专家与中间层MoE。此授权不解除旧holdout/input98封存，不允许改变问题取得正例。
起点1a75e077b，原作者checkout与旧环境不修改。每项分开版本、控制、结果和冻结。

## 第一阶段：Dual RS log-domain 数值兼容

R1失败完整保留。仅替换选择器训练的consistency调用点：
`log p_bar = logsumexp(log_softmax(logits), noise_axis) - log(m)`，
`KL = mean(sum(exp(log p_bar) * (log p_bar - log p_i)))`。
保留目标梯度、lambda40、eta.5、原entropy的1e-20下限及总consistency的1e-10下限。
不detach、不新加概率下限、不改学习率/样本/精度；这是数学目标相同的数值兼容版本，
不标为字节相同的原作者执行。

39项控制通过：正常域float64/float32 loss与梯度差分、极端logit负控制、
double gradcheck、非有限输入拒绝，加上原保存/恢复/截止/污染控制。
R2继续原256输入、batch和统一300s；精确续步接受门不变。
通过后才冻结长期训练与认证；本阶段本身不启动90epoch任务。

## 第二阶段：原模型入口语义核查

已核查作者源，以下细节必须进接口合同，不能用近似模型代替：

- MetaMoE在不选中的类别段填**零**，并非负无穷。正确类别不仅要超过本专家的
  其他logit，还要超过其他段的零。top-1仍执行`selected_score / selected_score`，
  零分数会产生NaN；不能无条件改成常数1后声称原程序等价。
- RoME的实际LoRA adapter对**全部专家**加权，global/local softmax随输入变化；
  不是离散top-2。还需覆盖attention、LayerNorm、GELU、低秩残差与完整共享骨干。
- Robust Experts的eval top-k使用当前router分数，权重分母保留`+1e-5`，
  train模式容量/噪声语义不纳入逐输入eval验证。不能重新归一化为精确和为1的权重。

分开验收：原生前向接入/差分、完整输入域与路由义务、受支持后端求界。
缺算子或缺路由覆盖时必须返回明确unsupported/unknown，不把clean-path导出当完整证明。
最终性能实验仅在这些门和整批监督/审计通过后冻结。
# Executed update: Dual RS R2 and MetaMoE full intake preparation

R2 is now `CONTROL_PASS`, with independent saved-only re-audit. The two real
updates and fresh-process continuation agree exactly in all eight snapshot
components and next-step metrics (excluding time). Outer execution10.377s,
postflight-inclusive10.422s. See `dual_rs_training_control_archive_20260921_r2.json`.
R1's nonfinite-gradient failure remains unchanged. This validates the named
log-domain compatibility loss and state restore, not final training or accuracy.

`class_separated_top1.py` is a NEW ACT entry, not a widening of the frozen
selected-softmax top-2 entry. It consumes the pinned original `MetaMoE` class,
keeps its zero-filled class blocks, and separately checks selected router score
nonzero on each legal guard before using the top-1 reduction. A global class
margin includes competitors in unselected zero blocks. All legal ties must be
covered; incomplete route enumeration, NaNs and unproved division return no
positive. UNSAFE requires in-box finite original complete-model replay.

Nine controls cover positive/negative selected scores, zero score, a zero-crossing
score, global-vs-local classification, ties, changing class widths, invalid mode/
dtype/source, out-of-box/NaN witnesses and late conclusions. Run CPU controls
with `CUDA_VISIBLE_DEVICES=`; ACT initialization is explicitly CPU/float64.
Conversion/support still rely on existing ACT lowering and HZ numerical policy.
Neither the old input enclosure trust nor solver roundoff is independently closed.

One separately frozen full author checkpoint control uses CIFAR test index0,
no outcome filter, normalized-space2/255 and300s outer deadline including import,
load, materialization, conversion and solve. Author float32 forward and float64
snapshot reduction are compared at the center only, NOT claimed globally
equivalent. Full API verification uses the explicitly named float64 snapshot;
it does not relabel the failed MNIST ONNX R2 conversion. NOT executed at this
preparation entry. No new performance sample set or original paper replication.
