# 继续从联合域定义研究 Neural-HZ

最新 [D155理论](definition_first_20260928/d155_common_fiber_projection_20261004/THEORY.md)
与 [记录](definition_first_20260928/d155_common_fiber_projection_20261004/RESEARCH_RECORD.md)
已保存。目标仍是强大非凸 Neural-HZ，不是helper或构造优化；formal_gain=0，
正式1870/2413及独立61/400不变，goal active。

D155修复D154的shared witness丢失：对四门完整消费者，用
a=t1+f1=q1+q3,b=t2+f2=q2+q4,z=4q4-q3，h=q4，
q=(a+z-4h,b-h,4h-z,h)。共同h的max lower<=min upper精确保LP/整数图。
六条strip+8guards在LP源有order时52rows/178nnz，对旧16/32，幅度4→3。
order若只integer-valid不能删LP行；无order56/192，或两侧同装order另收费。
对D018强chain有8lower×8upper的64row直接消元，不偷用52/178。
固定分数fiber有12必要facet但不排除全系统16row候选，更非扩展表示下界。

共同区间fiber→多个mixed ReLU有精确公式：child四行中两行转成A_j<=eta<=B_j，
所有parent/child lower与upper交叉比较，取eta=max lower是共同见证。
新直接行数pn+s(p+n)+s²+s，旧p+n+4s。p=n=s1且parent非空恒真可6→4行。
保存了2→1幅度、20→12nnz正控，但它是一般HZ/已有有损余项，不是exact CNN。
关键新限制：theta固定完整输入且eta是真神经元值时，exact integer fiber必须
singleton；p=n1则为affine身份，LP是否也零宽另证。不要把自由宽度1/4正控
作为普通精确CNN适用性证据去建新普查。双children分别合法不等于共同见证。

共同正分母/齐次神经读出支线已查重：D104 DEFINITION与PRIOR_ART_AND_DECISION
已有Charnes–Cooper、原bits×scale、同denom Affine/ReLU与raw residual障碍。
固定阈值清分母是有用接口但不解决分母生成/跨denom合流，不重启为新域。

本轮不实现generic FM投影或上述自由fiber包装，无新选定数值候选。下一定义
必须在实际同源神经状态上提出有用联合传递，不靠自由隐藏幅值或独立乘积
凸包假设。精确LP包含不是新增硬禁令；真正晋级仍是原全量保1870及新收益。
不削弱安全、测试、资源、默认off与13家族/E0保旧门。

配置paper-only，无代码候选/import/测试/模型/solver/GPU/后台任务。D150完整
数学人口4000/210保留，D152已消费run只读不重跑。文档技能将已证与未证分开。
分支redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked diff
29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5不变。
GPU、smooth/Transformer及全家族突破均未完成；D154和D155均为paper progress。
