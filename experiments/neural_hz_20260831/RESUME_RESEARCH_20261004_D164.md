# Neural HZ 输入盒组合研究续接

本轮[强参照比较](definition_first_20260928/d164_box_budget_comparison_20261004/THEORY.md)改变了 D163 的下一行动：从共同输入盒取得联合 L1 预算早在 D009 已建立，不能当新缺口重启。当前常中心 M 组合及半源中心组合都未通过独立实现的价值判断；不代表所有定向域失败或总体目标阻塞。

已知预算：G=A^T diag(w)A，B0=tr(G)+2sum_(j<l)|Gjl|，R^2>=sum(w)*B0。必须先合并 Gram 相消。只用 ||A||2²<=||A||1||A||infinity 或 Frobenius 导出的L1预算不小于 sum|Aij|，不比总IBP半径更紧。D009已有真实Conv形状、缓存费用与严格多门LP分离。

本轮盒正控：A=H8+11T/32，b=1/4，xi∈[-1,1]^8。非正交满秩，Gram B0=137/2，R=74/3通过548<(74/3)^2。M=0给sum|q-1/4|<=74/3。混权B=(1,...,1,-1/4)，live源skip=sumxi/16，J=Bq+skip，M证J<=163/6<55/2。旧完整source-labelled单门hull交允许source0、beta全1/2、q1=17/4其余33/8，J=895/32>55/2。但同一D009证书的一条2sumq-sumg<=80/3已经给J<=119/6，更强。局部完整逻辑账：旧4行+资源行16连续/8bits/33行/192nnz；消f的M为24/8/49/208；字面f为32/8/65/240。无GPU/速度/内存测量。

源插值：c_lambda=b+lambda Axi、半径(1-lambda)R0，在同源整数语义下lambda越大前像球越小，lambda1恢复精确图。lambda1/2的M预算被相位反射L1 sum w|q-g/2-(beta-1/2)b|<=R0/2蕴含：active成本同，inactive有(g+b)+<=|g-b|。普通严格例A=[[1,1/4],[-1/3,1]], b=(1/4,1/4), R0=25/12，source(1/4,-3/4)、bits10、q(1,0)，M成本23/32通过，reflection成本109/96>25/24拒绝，guards/epigraph/caps均通过。此包含不自动适用于分数bits的终端LP。一般M需2m连续、含guards7m+1行，reflection同2m但6m+1；没有新强度或完整成本证据。

不要重复研究：常中心球的逐门LP分离、半源中心、范数换名、已知Gram预算生成。下一定义必须面对跨层共同幅值和方向及域自身可支付的查询，而不是再加helper或在完整旧图上附加已知资源后换名称。D151同坐标强参照与本轮同证据比较应在候选设计时就纳入；不能要求有损集合超越精确原网络图，也不把本轮反例扩大到所有有损域。

本轮仅纸面推导/只读审查/新文档，无候选导入、数值、模型、GPU、shadow、replay或后台作业。最近执行人口仍D158 4032项/212文件。未来执行仍须新预注册/源码冻结/一次性版本及完整旧门；不复跑消费版本。D149/D150数学通过不继承到新组合。

正式1870/2413=1063CERT+807validatedADV，独立CIFAR25+Tiny36=61/400，两边新增0。所有原source/bits/EQ/LE/共享身份/decoder及fail-closed、禁止其他算法补解、13家族保旧、GPU/smooth/Transformer/新家族与最终满分目标不变。Goal active。本轮属于改变实施判断的数学progress，不是能力progress。

日期2026-10-04 Australia/Sydney，分支redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。仅新隔离目录及本续接有写入；历史只读。见[工作记录](definition_first_20260928/d164_box_budget_comparison_20261004/RESEARCH_RECORD.md)及目录内来源/档案哈希清单。
