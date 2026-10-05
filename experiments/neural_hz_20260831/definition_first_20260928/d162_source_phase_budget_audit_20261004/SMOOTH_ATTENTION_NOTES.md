# 共享源 Attention 和 smooth 推导的支撑记录

本文件保存上一研究段落返回、此前尚未写入本地文件的纸面结果，避免把它们遗忘或在未来重复计算。它们不是当前 CNN 预算闭包的组成部分，也不是已认证的新 Neural-HZ 路径。所有结果均无本轮数值执行，formal gain=0。

## 动态 query 下可以精确消除的共同二次项

同一原连续源 xi 上，令 q=q0+Q xi、k_i=k_i0+K_i xi。真实 softmax 允许对每个 xi 同时减去同一个标量函数 s0(xi)，不只允许常数平移。取 s0=q^T k0，则：

```text
Delta k_i0 = k_i0-k_00, Delta K_i=K_i-K_0,
s_i-s_0 = q0^T Delta k_i0
          +(Q^T Delta k_i0+Delta K_i^T q0)^T xi
          +xi^T sym(Q^T Delta K_i) xi.
```

当源具有非空开集时，所有差分为仿射的充要条件是所有 sym(Q^T Delta K_i)=0。若已有可靠仿射包参数化 xi=xi0+T eta，则检查限制后的二次矩阵 T^T sym(Q^T Delta K_i) T。不能把该必要性推广到任意非线性源流形，也不能把原整数位偷偷当作自由连续坐标。

这不要求 q 常量，也不要求 Delta K=0。一个非退化代数例为：

```text
xi=(x,y) in [-1,1]^2,
Q=[[1,1/4],[-1/3,1]], q0=(1/4,-1/5),
Delta K=[[1/3,-1],[1,1/4]], Delta k0=(1/3,1/2).

Q^T Delta K=(13/12)[[0,-1],[1,0]],
s1-s0=x/20+17y/60-1/60.
```

可以取可逆 K0=[[1/2,-1/3],[1/4,1]]、K1=K0+Delta K，并让 values 同源变化，例如 v0=x/2-y/3+1/10、v1=-x/4+3y/4-1/5。query、keys 和 values 均非固定。

静读当前 `_attention_score_differences` 确认它已经先合并共享 key 差分，随后为逐坐标 query 半宽乘差分半宽增加非负余项。上述例中这一余项之和是 10/3，而精确仿射差分半径为 1/3。这只是该公式分支的代数比较，不是实际工具输出或完整 ACT 界的测量。

一般同源二次多项式归并也会消掉反对称项，因此这里不是新的数学或新域。普通独立 token 扰动、per-token LayerNorm、实际 trained QK 未被证明满足该条件。全矩阵检查也非免费，稠密朴素成本为每 token O(d_h d_source^2)，包括精确位长与全部 heads/query rows；浮点中点近零不能当证书。

消去后 tokens 仍共享 xi，不能向 D127 授予 independent-token exact_product 资格。它没有解决动态 QK/PV 的一般共同源查询和跨层关系，因此不为该特例启动候选实现。

## smooth 激活的势差关系不需要共轭求解器

[Gu、Askari、El Ghaoui 的 Fenchel Lifted Networks，第 4.1 节](https://proceedings.mlr.press/v108/gu20a/gu20a.pdf) 已用 Fenchel 关系表示激活图。本项目 D131 也已讨论共同 decoder 与源绑定。不能把图等式本身称为本项目创新。

下面是本轮从凸函数切线不等式直接得到的推导，不是该论文声称的验证算法。设 phi_i=G_i'，且在所需全部区间 phi_i'>=-kappa_i，kappa_i>=0。则 F_i(u)=G_i(u)+kappa_i u^2/2 凸。对任意预先固定 t>0 和完整混权 w，有：

```text
sum_i [G_i(g_i)-G_i(g_i-t w_i)]/t
  -(t/2) sum_i kappa_i w_i^2
<= sum_i w_i phi_i(g_i)
<= sum_i [G_i(g_i+t w_i)-G_i(g_i)]/t
  +(t/2) sum_i kappa_i w_i^2.
```

分别在 g_i+t w_i 和 g_i-t w_i 使用 F_i 的切线下界并整理即可证明。所有 g_i 仍读取同一源；live affine skip 可加到两侧后合并，不需新 bits、反函数、共轭求值或输入 split。

[GELU 原文第 2 节与第 4 节](https://arxiv.org/pdf/1606.08415) 定义 phi(g)=g Phi(g)，并指出其非单调性。以下凸化常数是本轮推导：phi'=Phi+g varphi，|g varphi(g)|<=1/sqrt(2 pi e)<1/4，所以 psi=phi+g/2 全局严格递增，其导数落在 (1/4,7/4)。可以使用非最小但简单的 kappa=1/2，原函数为：

```text
G(g)=[(g^2-1) Phi(g)+g varphi(g)]/2.
```

Sigmoid、tanh、softplus 可取 kappa=0。这里覆盖的是精确标准 GELU 的数学定义，不自动覆盖模型中的 tanh 近似、具体浮点 CDF 或其它实现。

未解决的是整个共同源上的势差之和怎样得到可靠且低成本的界。逐项独立取界可能丢掉所需抵消；保留整个函数又只是图包装。可靠 CDF、原函数求值、每个混权方向与终端外包的费用均未认证。这是支撑推导，不是新 helper 或已实现 smooth 能力。

## 已有真实 Attention 证据的准确范围

重读 [D130 结果](../d130_import_isolation_20261002/RESULTS.md)：真实 PGD 命名 ViT 的首 CLS 完成了 38/192 个预注册读出，相对同系数矩形参考上下界均改善，但没有新增稳定门，完整来源实验超时。PGD 是模型名称，不是本候选运行了攻击。不能说完全没有真实来源，也不能说已经完整验证模型。

[D124 原生 Attention](../d124_source_phase_fiber_20261002/NATIVE_ATTENTION.md) 的精确方向原语仍要求仿射 score/value 和独立 token 来源；共同源情况下 product-box 查询只是外包。source-slope 编译尚未在该实测中完成。新的动态 query 联合关系、跨层组合、GPU、正式收益均未获资格。

所有本文件结果仅作支撑记录。当前主线仍是定义层面的强大 Neural-HZ，不把这些零散定理或局部界相加成一个尚不存在的验证器。
