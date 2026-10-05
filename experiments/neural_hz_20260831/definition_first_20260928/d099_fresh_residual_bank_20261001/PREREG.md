# 真实残差块检查的单次预注册

状态：这是保留的预注册草稿，已静态否决，没有生成 freeze.json，没有执行或创建 RUN。以下步骤未发生；见 [结果](RESULTS.md)，不得据此启动已知不合门的候选。

先冻结六文件：CONTRACT.md、PREREG.md、source_probe.py、run_source.py、run_math.py、collection_contract.py。freeze schema 为 d099_fresh_residual_bank_v1，required_tests=3845、required_test_files=188、new_test_names=[]。冻结前不得对新候选 AST 解析、导入、编译、收集或数值执行。

唯一 RUN 为 `experiments/neural_hz_20260831/results/d099_fresh_residual_bank_20261001_v1`。`run_math.py --enabled` 首次独占创建即消费数学阶段，继承 D098 的全部源身份、失败历史、188测试路径和3845有序nodeids；不复制或替换旧测试，不降低人口，不进行预跑。一个 pytest 子进程的启动、导入、收集、测试、JUnit 和退出合计不超过60秒，零失败、错误或跳过。数学监督器完成前后身份与生产 provenance 检查并自动封存。

只有完整数学成功后，`run_source.py --enabled` 才独占建立 `source_probe_supervisor` 并启动 `source_probe.py --enabled`；worker 独占建立 `source_probe`。其固定来源、完整结构与所有预算以 CONTRACT 为准。240秒从 worker 子进程启动前计至退出；内部235秒留给失败收尾。超时应终止并等待同一子进程，保留已有日志和终态，不因观察等待超时重启。

真实阶段不使用模型/实例菜单，不选局部成功关系，也不在资源失败后换另一路径。全组未安装或任一守卫失败，结果必须明确失败；完成诊断不等于完整物理、原模型健全性、decoder、GPU 或正式能力资格。无适用组是有用的普查结论，但不能算安装带来的验证收益。

所有运行自动保留来源清单、环境、阶段与费用、时间/内存、完整和未完成字段、错误及输出摘要。已消费版本不编辑、不重跑；后续实质改进须另建预注册。未执行草稿也保留，不把静态审查说成测试通过。

2026年10月1日，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。正式和独立外部成绩均不变；formal_gain=0。文档技能用于明确区分证明、实际诊断和未取得资格，不增加确认环节，不放宽门槛。
