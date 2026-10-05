# 下一步落在真实跨层关系，而不是继续扩展 helper

D209 两层严格分离和完整数学门均通过，说明共有余量值得继续检验；它没有给出真实网络新增解。后续研究问题应是：真实普通网络中，这条同父域规则能否保住连续非线性之间的相关性，并在完整费用下带来新 CERT 或经过原网络验证的 ADV？不要把映射/loader/测试数量作为主成果。

## 本轮只读定位

在 results/c5_relu44_admitted_20260905_v1/events.jsonl 第80–100行，可以看到同次完整运行的真实 Tiny 结构：

```text
ReLU36 → Conv37 → Scale38 → Bias39 → Add40
       → Conv41 → Scale42 → Bias43 → ReLU44
```

已封存的 layer36.snapshot.json 声明 layer36 文件大小118231878 bytes，以下 SHA256 均直接摘自原 snapshot manifest。本轮未加载 pickle、未重新提取矩阵、未运行网络，也未将该数据加入 D209 数学输入。

完整原始 SHA256：`2e2cb7d9aa3a368803907993a8ffcc0888f4db2e35044fb83e353b0d47442959`。
layer44 的原 snapshot 声明 SHA256：`58f1dd7ab87f21ba1d608056e0a1fbef4ad005fbbfa9674703d1e6224ba04917`。

旧 evidence/c5_relu44_admitted_20260905_v1.json 的运行时发布记录覆盖完整6272门：157 crossing、124稳定正、5991稳定负，新增原bits80。这是旧路径的发布事实，不自动成为 D209 的数学或能力资格。

## 为什么不能直接恢复后记作新结果

c5_corrected_prefix_worker_v2.py 第36–42行的 snapshot payload 仅显式保存 net、hz_cache、expr_cache、bounds、terminal_bounds、precomputed_relu 和 provenance。它没有完整保存全局 frame_widths/relu_slots/aux_slots 以及完整输入 spec/decoder 绑定。本轮定位没有发现可直接作为新域完整父元素的已认证恢复接口。不能从 shape 或连续列位置猜测原相位身份，也不能把不足的列身份换成新独立因子。

另两个已知来源不适合作为本轮收益捷径：C34 的完整200门是最后 ReLU78→Dense79→ASSERT80，没有后继 ReLU；C5 的 ReLU44→55 中，55已被旧路径认证全负，不应重报为新增跨层能力。D025/D120的1600读出只是部分位置，不能冒充完整 bank。

## 下一版应预注册的决定

先在新的独立合同中固定真实完整结构、输入性质、出生相位身份、全部 live outputs/残差、decoder和原路径对照。若原快照不足，应从原输入重新建立完整拥有的元素及必要绑定；不补造缺失历史 metadata，也不污染旧 snapshot。只做能够支撑本规则实测的最小绑定，不重开通用 loader/helper 工程。

在相同完整父输入上，无条件执行冻结统一配对/共同余量规则，报告新增关系、全部后继查询和最终边界，不选几个好看的通道或已知 sat 身份。继续支付 cap 证明、四支持查询、全部谓词/相位、存储/terminal/具体见证验证费用，记录普通异质分母风险。若真实结构上没有相关性收益或费用失败，保留负结论并回到域定义，不用攻击、split或外部 rescue 补成成功。

任何新实现或诊断执行先冻结新版本；D209已消费，不改不重跑。后续数学人口从本次4100 tests/216 files、7425源身份/14输入完整继承；真实源证据新增认证，不继承旧算法资格。之后依次同结构shadow、逐家族及全部2413回放，另做独立400回放；通过各自门以前正式账均不动。

2026-10-05 Australia/Sydney，后续假设记录，不是已执行计划或新源资格。
