# 完整共享来源实现的成本预检失败

本版真实来源worker未启动。冻结前静态审查已经给出其必做准备阶段的费用下界；该下界超过旧完整 pair 计算可替换出的空间，尚未计新联合关系本身。因此 freeze.json 将 source_static_preflight_passed 和 source_worker_authorized 均设为false，run_audit.py第107行显式要求前者为true才允许继续。数学通过不能覆盖这一失败。

## 比较范围

原 D261 完整三来源 whole work 为255,549,650，门为256,000,000。逐项静态审查可被新算法整体替换的旧 pair 主体上界为66,228,760；其中包括原逐对差值、offset、mask聚合、选中Q缩放、旧与公平分类及其记录，不包括未改变的完整前缀、来源/输入认证和一次预付40M证据账。

因此，用最有利的旧主体上界计算，新主体的可用空间至多

```text
256000000 - 255549650 + 66228760 = 66679110 work.
```

新增manifest与闭包等费用还应另外支付，不能先把这些视为节约。本比较保留原物理/数值/来源边界，不提高门槛、不裁掉任何pair/mask，也不返还没有用完的证据预付额度。

## 新实现的必做下界

三项均位于 joint_pair._prepare，在逐对 crossing 或 Gram eligibility 判断之前执行，故不能依靠之后的 ineligible 比例规避。

| 必做计算 | 完整元素数与最低单元素账 | work 下界 |
| --- | --- | ---: |
| 两个 projection center 复合缓存 | 2×128×128×64；interval乘法13、sum_axis输入检查4、两个归约各3 | 48,234,496 |
| 原 parent、middle、outer及projection Conv扫描 | 864,256；两次borrow各5、比较预付2 | 10,371,072 |
| middle selected-tap removal缓存 | 331,776；准备5、两次interval乘法各13 | 10,285,056 |
| 合计 | 未计常数费用 | 68,890,624 |

代码位置为 joint_pair.py 的 _conv、_removal_cache 及 _prepare 中逐输出的 center composition；算术账由 source_arithmetic.mul、sum_axis 和冻结 D259 的 _fast_mul、_check、_axis_sum_endpoint 给出。这里只使用源码已明确执行的最低账，不把全部候选费用估成这三项。

三项下界已比可用上界多2,211,514。父系数与gamma缓存、普通参照表、Gram、分组支持、每mask标量证书、来源记录等都还没有算入。故本版完整执行不能在原门下晋级。

## 结论边界

这是当前实现的静态费用否决，不是实测网络耗时，也不是证明所有联合残余或 Neural-HZ 设计都不可能。未来可靠的融合、共享身份认证或不同结构定理可能改变成本，但须新候选、完整计费和重新验证，不能在本版冻结后删费追认成功。

本轮停止对此实现做局部成本优化。结合 WIDE_RESIDUAL_THEOREM.md 中的能力负例，下一个研究问题应优先改变宽残差的关系推理方式；否则即便降低构造成本，也可能只更快地添加旧状态已蕴含的谓词。
