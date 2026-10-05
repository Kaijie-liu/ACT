# 容器修复后的测试收集失败

本版在新目录修正了D128的protobuf计数接口，并加入实际NodeProto序列化/解析的合成控制。唯一会话54232已终止，supervisor exit=1。注册3933项/204文件，但pytest收集时三个新文件与继承的D128文件同名，导致import-file-mismatch；pytest exit=4，三项收集错误，没有取得完整测试资格，没有启动来源worker。

监督器数学进程观察13.295582642778754秒，pytest报告12.23秒。完整inventory尚未生成，所以总回执另报missing, linked or oversized JSON；真正首因保存在[tests.log](../../results/d129_source_container_compat_20261002_v1/tests.log)。不能把缺失inventory当作原始算法或数值失败。

三个执行工件与十二个冻结文件已SHA256复核；source/input/provenance无漂移。[退出回执](../../results/d129_source_container_compat_20261002_v1/exit.json)SHA256为31b90f187abe8daec8ca852080bc59e5a4008729e77819a8c99adc4fd66f777d。旧D128失败版与本版均保持只读，不重跑或删测试。

下一隔离版本D130固定pytest的importlib路径导入模式，并保留本版全部已登记测试；数学、来源人口、资源门及资格边界不改变。正式1870/2413和独立61/400仍零新增，没有默认启用、GPU/真实模型/保旧资格。2026-10-02，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；无生产修改、commit或push。
