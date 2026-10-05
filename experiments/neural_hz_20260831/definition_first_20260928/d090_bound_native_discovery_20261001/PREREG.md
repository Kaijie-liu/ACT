# 原生依赖修正版的单次资格与归档预注册

先冻结七文件：CONTRACT.md、PREREG.md、project_import_closure.json、archive_probe.py、run_archive.py、run_math.py、collection_contract.py。freeze schema 为 d090_bound_native_discovery_v1，required_tests=3825、required_test_files=183、new_test_names=[]。未冻结前不对本候选AST解析、导入、编译、收集或数值执行。

唯一数学RUN为 experiments/neural_hz_20260831/results/d090_bound_native_discovery_20261001_v1；首次创建即消费本版本数学阶段。run_math.py --enabled 必须验证D088完整数学成功和其全部历史身份链，继承原183测试路径及3825有序nodeids。不能预跑单测、跳过测试或修改冻结版本后重跑。

数学成功后，唯一归档入口 run_archive.py --enabled 启动 archive_probe.py --enabled。监督和worker分别独占创建RUN/archive_probe_supervisor及RUN/archive_probe。240秒包含worker启动、导入、身份认证、解码、完整root计费、全结构发现、全部组应用及worker收尾；监督器自身终态封存另列，不宣称监督总成本免费。超时杀死并等待同一worker，保存已有日志及终态；不因观察等待超时重启。

唯一归档、全部root、全扫描/全组人口及不变预算由CONTRACT约束。修正只增加冻结的导入身份闭包以及解码后的同类检查，不改变结构规则、模板范围或候选core。D088归档失败receipt作为历史证据保留，不将其当作适用性负结论或数学失败。

任何前提、身份、数值、费用、存储或时间失败都fail closed并自动留存停止阶段。没有全模型forward、LP/MILP、attack、split、backward/dual rescue或GPU执行。即使归档转换成功也不产生CERT/ADV、不证明具体网络收益；正式成绩和默认配置保持不变。

2026-10-01，branch redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。全程只写本新目录和唯一RUN；旧冻结档案、历史模型及生产状态只读，未commit/push。
