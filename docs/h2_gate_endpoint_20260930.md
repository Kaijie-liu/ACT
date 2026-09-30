# H2 gate 端点支持控制与下一步

2026-09-30，从干净且同步的 `95a06d46a88e` 开始。H1 稠密隐藏依赖剪枝已经停止，
本轮没有重新调它的 fixture，也没有重启缓存、输入 98 或真实后端搜索。

**结论：发现并实现了一个值得继续的加权义务表示机制。** 同一合成请求的三个 pair、
六个端点下界全部检查通过；同一基础域的 McCormick 松弛有精确可行负目标点。
这支持“继续来源接口控制”，不支持真实模型收益、运行提速或外部竞争胜出。

## 为什么选择这一机制

当前 [F0 构造](../act/back_end/moe/weighted_top2.py) 会投影专家基准值 u 和差值 d，
求差值范围，再建立 gate/product 变量与四条 McCormick 平面。
[来源义务](../full_source/obligations.py)和[H1 来源 LP](../scoped_source/sparse_build.py)
也使用乘积松弛。此次在这些入口中没有发现 gate 区间端点支持路径。

H2 对给定合法分支域 P 和已证明的 gate 范围 [l,h]，将完整加权性质改为两个线性
支持义务 u+l*d、u+h*d；取两个已检查下界的最小值。它保留同一输入与私有专家因子，
消除独立区间 gate 的乘积松弛。这不是原始 sigmoid 的精确消除；真实 gate 与输入
的函数关系在这里仍仅由区间外包络处理。
证明、适用前提和反例见[端点恒等式](../act/back_end/moe/proofs/gate_interval_endpoint_support.md)。

这是基本仿射端点性质在 MoE 义务组织中的应用，不声称新的通用优化定理。
另一候选“精确反向证据”暂未实现：ACT 的
[DualSolver](../act/back_end/solver/solver_dual.py)和
[ReLU 反传](../act/back_end/dual_tf/tf_mlp.py)已经存在，不能重新命名为新算法。
CROWN 的论文及 auto_LiRPA API 也明确提供反向界传播与线性系数返回；这些系数可用于
提出证据，但 API 本身不是本项目的独立精确证明合同。
参考：[CROWN 原论文](https://proceedings.neurips.cc/paper/2018/hash/d04863f100d59b3eb688a11f95b0ae60-Abstract.html)、
[官方 API](https://auto-lirpa.readthedocs.io/en/latest/api.html)。两条候选不混合消融。

## 固定控制与结果

[协议](../configs/h2_endpoint_algebra_20260930.json)在运行前固定输入、三专家表达式、
gate、全部 pair、接受阈值和零原生求解调用。输入为一维有理数盒，模型是为分离表示
问题而构造的解析控制，不是随机抽样、训练 checkpoint 或真实效应检验。

| 项目 | 经检查结果 | 限定 |
|---|---|---|
| pair 01 两端 | 1/10、1/10 | 同一 P，保留私有 ReLU |
| pair 12 两端 | 1/10、1/10 | 同一 P，保留私有 ReLU |
| pair 02 两端 | 47/20、27/20 | guard 空域上的真空保证；仍保留义务 |
| 完整端点聚合 | 三条性质、六端点均正 | 给定 P 与 gate 的接口结论 |
| 同基础域 McCormick | 精确可行目标 −1/40 | 证明该 LP 无法给出正下界，不是网络 UNSAFE |
| 路由变化 | x=−1 选 12，x=1 选 01 | 两点同域，tie 点 −1/2 的两条路由同时覆盖 |
| 专家逐个验证 | A(−1/2)=−3/20 | 该点 pair 01 合法，完整混合仍有正保证 |

结果归档：[精确重算记录](h2_endpoint_algebra_20260930_r1.json)。原生求解调用为零，
手工指定有理数对偶，独立核验残差修正；没有运行时间或 solver search 改善结论。
不能把六个端点计为六个完整请求。旧真实主表和封存结果完全不变。

19 项控制通过，覆盖全部 pair/property/endpoint、tie、类别维度、空域 guard、
零宽 gate、零差值、[0,1] 退化为专家逐个验证、交换专家后的补 gate、错误身份、
重复/缺失义务、缺证据未决、变量别名、坏 dual、保留原矩阵、迟到候选与迟到检查。
`python -I -S` 子进程禁止导入构造器或数值栈，仍能独立检查。

只读协作复审指出 fixture 有若干描述字段未参与构造；归档前增加固定协议哈希准入、
实际阈值一致性及相应变异控制。没有改模型、接受门或观察结果后换样本。

另有 34 项维护回归通过，合计 53 项。维护首轮命令在隔离模式下直接 `-m unittest`
无法找到本地 scripts 模块，产生四个导入错误；显式指定本地模块搜索路径后通过，
没有修改测试或接受规则。导航检查 196 个链接、619 个冻结源码绑定与历史交接均不变；
七次已封存真实调用账本仍为 0 个完整正请求。阶段结束存储检查为 224,623,136,768
字节，较开始仅增加约 74 KB 的代码与文档；没有新增大型结果或缓存。

## 当前保证边界

通用接口明确返回 `source_complete=false`、`deployed_float_SAFE=false`、
`hard_budget_supervision=false`。它检查给定 LP 上的精确下界，不检查这些 LP 如何
来自网络、gate 区间是否覆盖真实权重，也不验证原始程序的浮点运算。
重新绑定一个错误但自洽的 P/gate，不会因此变成网络证明；这项前提不能用哈希代替。

本轮只有限定为最多 300 秒的协作式截止检查，没有新硬监督器、资源准入、完整搬迁包
或计时比较。不能借用旧 H1 监督控制来声称 H2 已通过这几项。

## 下一门是来源控制而非真实实验

1. 在现有声明源表示上，独立重建共同 P、性质及私有因子身份。先支持由已检查 router
   margin 符号得到的 [0,1/2]、[1/2,1]，无符号证明时用 [0,1]；不给未证的紧范围。
   如以后采用已有精确 sigmoid 检查器，范围来源仍须绑定同一 pair 和输入域。
2. 对同一新来源生成两臂，全部 tie-legal pair/性质不减少。先控制错源、向内 gate、
   私有因子合并、部分 scoped reuse、空域证明、遗漏端点、实际二进制系数和改变维度。
   基础 P 相同时要求相同矩阵/变量界；不要求不同表达的数值下界完全相等。
3. 来源控制通过后，才单独接入既有监督框架的新版本。捕获、来源检查、gate 支持、
   构造、两端候选生成、序列化、独立检查、聚合及发布计入一个 300 秒硬预算。
   验证截止、异常、内存、部分证据与迟到拒收；保持原有正 margin 接受门。
4. 原生完整合成比较必须记录两次支持查询及所有前处理，不将手工 dual 控制称为计时
   优势。若只有代数简化、所有 gate 都退化为 [0,1]，或没有完整证明/实际工作改善，
   记录并停止该因素；不能靠调 gate、增时或挑标签追正例。
5. 通过后另行决定有限 development，当前没有任何真实输入授权。H2 直接面向
   weighted top-2/CROWN 竞争线，不直接改善 MetaMoE 类别分离 top-1；后者仍是独立
   未达成目标。高准确率、跨家族、人类审阅与干净环境复现继续 OPEN。

复核命令均不加载模型、不调用求解器：

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -I -S -c "import sys,unittest; sys.path.insert(0,'.'); unittest.main(module='scoped_source.endpoint_tests')"
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -I -S scripts/check_h2_endpoint.py --check docs/h2_endpoint_algebra_20260930_r1.json
```
