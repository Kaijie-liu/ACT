# Neural-HZ源耦合研究恢复入口

完整Goal继续active，定义优先、GPU、CIFAR/Tiny、13家族、smooth/Transformer及最终同路径2413目标不变。正式1870/2413和独立61/400本轮均新增0，没有新的数值资格。

新入口：[共同源距离定义](definition_first_20260928/d174_source_coupled_energy_20261004/THEORY.md)、[被否决方案与边界](definition_first_20260928/d174_source_coupled_energy_20261004/BOUNDARIES.md)、[研究记录](definition_first_20260928/d174_source_coupled_energy_20261004/RESEARCH_RECORD.md)。此前D173及所有旧档保持原状态。

新增原生候选：完整C、全部原gamma及共同源d，存在同一个t满足Ct=Cd、a=CD_gamma t、||t||²+||t-d||²<=E。真t=d证明whole-parent健全，新关系蕴含旧norm fiber。任意固定lambda,mu,h可生成2L+2hᵀd<=E+||v_gamma+h||²+||h||²；右侧phase-affine，无新乘积变量。固定h=-v_nom/2和原六坐标方向即可构成统一规则，不看margin/solver状态。

四门完整C=[(1,-1,-1,1);ones]，源盒[-1,1]^4，b=(1/2,-1/2,1/2,-1/2)，E4。旧boxed共同代理在source0、phase1010取t=(4/5,-4/5,4/5,-4/5)，得到Q13/5。原mass-opposite方向加新h自动产生4Q-3d1-d2-3d3-d4<=10，故下一ReLU(Q-(3d1+d2+3d3+d4)/4-51/20)恒零；旧点允许1/20。全部phase项在这条物理行中相消，不是只有整数native点分离。未运行CERT，不是实际ADV。可靠sqrt尺度误差须认证；完整Bq/Q消费者复用mass幅值但计两条零dominance行，本例旧32/new44行。

最直接负控：原HZ普通四行LP已经给J<=2，比新5/2更强；新控制只是恢复弱proxy路径所丢能力。逐坐标source-centered perspective给同一行但少||h||²，也更强。新增6p行使计数2m+16p+2r，完整source fill/Gram/history/terminal还要收费。满列秩C时旧原生本已精确；当前真实CNN没有等权pool，不能转去不存在的结构。没有真实费用或新颖性资格，不因这个控制开另一轮玩具组件。

精确源锁dᵀt=||d||²仅作支撑：点态E恢复原图，global-E在普通完整C上仍有固定phase凸化损失。动态mean是D156预算改写，公共anchor被grounded+Jensen蕴含且其局部分离可在LP丢失；不要重复宣称突破。

下一研究应寻求真实普通混权结构上的可付源相位关系，或直接可消费的更强定义；保留完整幅值也允许，不以变量少作为创新门。禁止helper/攻击/分裂/后向或对偶救援，全部原phase、共享身份、decoder及fail-closed保持。所有数值候选另行预注册冻结并保留既有测试、资源、shadow及全量门。

本轮paper-only，无后台实验。最后组件D1584032tests/212files；D172只读系数诊断不重试。分支redu-hz、HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac、tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5；历史和生产文件未改，无commit/push/default改变。
