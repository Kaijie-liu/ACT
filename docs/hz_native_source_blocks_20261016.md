# 原生候选器的来源行块分解与成本边界

**首层全局 COO 可以由来源权重块与完整辅助项替代，保持同一个实数线性算子。**
本次静态推导明确了这个表示改动，并补上了“现有 CPU 候选器接直接消费路径”的条件化
Work 上界。尚未实现新算子；没有完整内存、300 秒可完成性或真实模型收益结论。

基线为 `5989618ff3c9db9c4aa164392b806c272d787cbc`。范围仍是注册的八专家 MLP、
252 条义务、最多 504 端点／1008 份两角色候选。见[整数核算与源码身份](hz_native_source_blocks_20261016.json)
和[纯静态脚本](../scripts/assess_native_source_blocks.py)。此前
[准入拒绝及优先级裁决](hz_full_path_decision_20261016.md)保持有效。

## 只改变 local 存储，不改变数学义务

现有 V3 系数算子是 \(C=(I-R)^{-1}LD\)。\(D\) 保存每个因子的尺度，\(R\) 保存
后层 affine、ReLU 和 guard 的引用。输入单位基、尺度分离和精确伴随不是本轮新增。

令 \(W\) 为所有首层 `SourceLocal` 的原始权重；\(P\) 选择原始输入因子；\(S\)
将这些首层行放回原始 node ID。令 \(E_{aux}\) 包含其余 local 项，**以及每条
SourceLocal 的全部误差／lift 尾项**。独立来源检查器已经要求首层 row、node、owner
和输入因子映射对应，并验证全部尾项。因此：

\[
L=E_{aux}+SWP.
\]

对不重不漏的首层自然块 \(B\)，可用

\[
LZ=E_{aux}Z+\sum_B S_BW_BPZ,
\qquad
L^\top A=E_{aux}^\top A+P^\top\sum_BW_B^\top S_B^\top A.
\]

这直接由矩阵分配律和转置规则得到。完整 forward 仍先施加 \(D\)，再计算 local 和
按层传播 \(R\)；adjoint 仍先逆序传播 \(R\)，再计算 local transpose，最后施加
\(D\)。所有专家块必须累加到**同一输入因子**后再求残差范数，不能分别取绝对值。
bias 和 center 不放进权重 GEMM，但仍保留在性质 offset 与约束 RHS 中。

自然块由注册结构决定：router 128 行、八个专家各 256 行，每行 3072 个权重。
没有引入可调 tile 参数，不剪除当前 pair 外的 native 节点；候选宽 8、检查宽 2、
128 次更新和全部独立检查均保持。将来 GEMM 与 COO 的浮点求和顺序可能不同，故要求
**同候选精确接受一致**，不要求优化轨迹逐位一致；候选数值正界本身不能授权接受。

已有[共享／私有块支持](hz_blocks_20261002.md)和 `store_endpoints.Native.blocks(pair)`
不能重新算作本轮创新。这里的增量仅是 V3 首层来源不再展开为全局 COO；它首先是
可检验的表示成本改动。是否增加完整证书或净成本收益，仍须后续同算法直接消融。

## 生命周期取舍

| 方案 | 明确载荷／代价 | 裁决 |
| --- | --- | --- |
| 每次乘法从来源重新转码一块 | 最大权重块 6,291,456 B；完整目录最多 21,588 次 forward/adjoint | 不作为默认方案 |
| 一次转码九个自然层块，随后驻留 | 原生权重 bank 53,477,376 B；与检查来源独立所有权 | 继续静态核算的候选，不是实施批准 |

第一种方案在完整调用目录中读取首层位置
\(21,588\times6,684,672=144,308,699,136\) 次，即最多
**1,154,469,593,088 B 逻辑 source payload**；不含写入、检查和转码，亦不是物理磁盘
流量或耗时实测。不能只报单块小内存而隐去重复工作。第二种仍有 native bank 读取，
只是免去每次重新解析原来源；不能用可写 Torch alias 指向检查器的 immutable bytes。

辅助 local 上界为 \(Q=n+2A+12U=48,048\)，其中包含最多 4,352 个首层私有尾项，
不能用零中心、无补偿或相同尺度缩小它。保留小 local COO、转置与原 refs 的条件化
**命名 tensor 载荷**为：

\[
8W_1+48Q+24R+8F=62,694,784\text{ B}.
\]

这里为 local 与转置保守预留两份 COO；不是已核实其底层 storage 别名行为。
旧同类公式为 330,081,664 B，公式之差不是已实现或实测内存节省。
正常来源检查返回、无异常 traceback 或外部引用时，`CheckedSharedForest` 仍经
`SourceLocal.row` 保留首层来源字节 53,494,784 B，两者合为 116,189,568 B；
来源检查期间的全参数 Snapshot bank 则为 55,690,944 B。对象不可达不保证 RSS 归还。

还要另计 Packet、图与 Fraction、索引映射、候选和精确 arena。一个 reference level
的 select＋product 载荷上界仍达 35,491,840 B；全局 alpha/residual、旧迭代值、
isfinite/boolean-index 临时量仍在。新 dense GEMM 的内部 workspace 也不能借旧 sparse
AXPY 的审阅结论。因此以上小计**不能与 256 MiB 作完整容量准入比较**。

## 现有 CPU 原生路径的条件化计数已补齐

本节不是新行块算子的 Work 预测。它只替换现有共享 Driver 中的 fixed producer，
使用原 `Operator`、现有 RootLease／SparseRoots、直接帧和分页重算；**不调用旧
`produce()` 的额外 Session／JSON proof 路径**。假设接线不添加其他计费操作。

注册图的非空 refs depth 只有 2、3：ReLU 正输出沿用节点，不稳定输出用新 local，
其 equality 引用 preactivation；最后专家 affine 最大 depth 3，router guard 亦不超 3。
该层数界不适用于任意深度模型。取 \(T=84,M\le2700,E=504,J\le2\)：

| 生产者部分 | Row visits | Coefficient visits | Work.exact |
| --- | ---: | ---: | ---: |
| 原生准备，含 SourceLocal 逐位置访问 | 19,496 | 13,726,720 | 0 |
| 原生生成，暂按普通 roots tuple | 282,324 | 1,589,616 | 0 |
| RootLease＋SparseRoots 的附加访问 | 3,024 | 11,088 | 0 |
| **合计** | **304,844** | **15,327,424** | **0** |

每目标有 \(h\le4\) 个 roots，lease 额外 rows 为 \(2+h\)，coefficients 为
\(3h+h(h+1)/2\)。后者包含底层 `Block.get` 的枚举和逐 key 线性查找；只数
RootMap 会漏计。该部分已包含在旧 fixed producer 中，所以必须替换而非叠加：

\[
U'_a=U_a-2,738,232+15,632,268.
\]

结果为 dense **89,446,425,692**、pair-scoped **165,121,513,292**。
这使“现有原生 Operator 假设接线”的 post-capture 计数从空缺变为有条件上界；
不改变旧归档，也不声称此接线已经实现。将来新增生命周期操作须再计费。
`Work.exact=0` 只表示没有调用该计数器，**不是原生计算、Fraction/string 或 I/O 免费**。
新行块的读取／转码计数须单独绑定实现；两遍独立源检查的 13,922,736＞8M 拒绝不变。

## 候选出口仍有独立的内存问题

当前 `Operator.propose()` 返回一整个 tile 的字符串 records。以每个有限 binary64
有理数字符串最多 342 字符计，8×2700 个 proposed scalar 可占 7,387,200 个字符；
再有零角色、列表、Fraction、临时 `.tolist()` 和编码副本。新版同 tile 二进制帧上界
362,944 B，**不能替代生成这个帧之前字符串 records 的存活成本**。
完整候选流上界仍为 22,868,160 B，超过原独立流 cap 16,769,024 B 的统一保证。

## 下一门与停止边界

保留一次转码的自然块作为有限静态候选；它有明确代数合同和原生构造动机，不再为
“省一个大数组”直接启动小控制。接下来的完整合同还必须处理**有界候选输出接口**、
新 GEMM workspace、来源／精确 arena 同时存活，以及真实捕获；只完成首层构造不能
宣称完整内存闭合。若这些界无法闭合，就报告具体缺口并停止实施准入。

同算法 COO 强参照、dense/scoped 检查因素与 GPU 因素分开，不把 native pair 剪枝、
H2 表示、预算或数值门混入。当前不选新配额，不运行真实模型、GPU、候选或新控制。
G1–G6、历史来源缺口及外部竞争状态不变。

静态复查只读源码／记录、解析 AST、计算整数；AI 复核不替代独立人类审阅：

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B scripts/assess_native_source_blocks.py --check
```

本记录与上一份准入记录重算通过；既有资源／导航共18项测试通过，619份冻结来源
未变，新增文档本地链接通过。两路只读复核分别检查完整计数和对象寿命，另一路推导
算子等式。没有运行来源构造、原生候选、监督子进程、模型或GPU，也未删除文件。
