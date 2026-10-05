# Neural-HZ：真实残差系数诊断后的恢复入口

完整 Goal 仍 active：从 HZ 数学定义提出强非凸 Neural-HZ，保留连续因子、原二元相位、EQ/LE、共同 latent/frame、具体输入重构及 fail-closed。不是 helper、计算图换名、矩阵压缩或仅提速。redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；正式1870/2413与独立61/400均零新增，不能相加。完整用户限制仍适用。

本轮入口为 [D172研究记录](definition_first_20260928/d172_residual_source_geometry_20261004/RESEARCH_RECORD.md)。旧 [D171入口](RESUME_RESEARCH_20261004_D171.md) 保持原样；以下是其新增证据，不重写历史。

## 新证据

一次冻结系数诊断完整处理两份真实 ViT 的五个48→96→48 residual MLP，保存U/V/c/d及原BN与权重，保留末端bias。固定λ=1/(1+Q(V))和外向范数规则全部一致，不搜索。L_sector/L_triangle依次约0.997971、0.998576、0.999778、0.999679、1.000604；ρ约96–501。**本固定标量证书没有显示支撑升级的实质抵消，不据此推进完整候选。**

不要反推真实谱范数、真实defect或网络扩张；没有父域R、输出界或性质验证。5/5是完整系数人口，不是完整token传播，更不是ViT验证通过。两份训练名中pgd只是来源名称，没有执行攻击。

新 [源仿射查询推导](definition_first_20260928/d172_residual_source_geometry_20261004/FORWARD_QUERY.md) 是纸面正向：把原生源相关锥用平方弦转为一条固定前向行。普通三门混权控制的J界1059/640低于7/4；仍满足完整两球外包的假点给57/32。这是既有不等式应用，不单独构成新域；未跑数学测试。

同一控制证明固定原相位下native仍非凸，纯连续线性aux lowering无法精确保留；增加相同两球的方向数不能修复。对称裸盒锥的凸包退为全局球，但不要外推到与guards/caps相交的全域。行数成本修正：m=2n、k=n时，替换并支配旧行是10n；需保留更强旧R证书时可达12n，对比旧精确8n。幅值少不是已支付。

## 下一定义问题，不是已选候选

优先处理完整块里共同源、原相位与方向性残差的联合关系，以及native→terminal接口。当前宽块优势与粗标量预算的精度冲突已经清楚；下一步不应继续只加锥方向、不应为这次很小的比例变化启动模型回放，更不应修改solver或调用helper补解。

提出任何新关系时同时回答：具体化是否真不同于已有HZ/混合符号图；全部原bits及零点标签如何保留；混权、shortcut、全部消费者和decoder是否闭合；相对强同源参考（含中点恒等式）何时严格有效；完整终端降级与代价能否支付。若只剩旧图或藏起m个幅值，记录负结论，不包装创新。

CIFAR/Tiny仍优先。已有D161显示large首块identity同宽，medium/Tiny首块projection shortcut；不能把ViT的m>n proof直接套过去。本轮没有新CNN数值人口。不得按模型名、结果、margin或LP状态触发菜单；真实结构适用条件与来源人口选择分开。

## 执行与封存

唯一目录 results/d172_residual_source_geometry_20261004_v1 已消耗；外部exit0、内部19.4234秒，7327源/14输入和4新文件前后认证完成，五块证据完整且八个exit artifact哈希另经核对。诊断时间不是候选速度。无model forward/solver/GPU/native候选资格，formal_gain=0。所有新执行需要新预注册，禁止编辑重跑此版本；不放宽资源或验证人口。

最后执行数学组件仍为D1584032 tests/212 files，本轮不重跑也不替换。旧生产dirty文件保留，tracked diff SHA256仍为29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。历史HyZor/模型/结果只读，没有commit/push/default切换，没有未完后台实验。研究档内ANCHOR_SOURCE.sha256与ARCHIVE.sha256从仓库根目录校验；自动数值证据清单由原exit.json保存。
