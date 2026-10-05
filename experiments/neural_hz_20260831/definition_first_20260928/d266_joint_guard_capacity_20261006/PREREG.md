# 相位守卫与共同残余联合容量预注册

本轮仅注册默认关闭的标量数学组件和完整继承数学回放。源码静态编写可在冻结前进行；任何候选 import、AST、compile、collection 或数值调用都必须等 freeze.json 完整固定源码、理论、来源与测试人口后，走唯一 run_math.py 入口。禁止冻结后修改、补跑或减少人口。失败保留原状。

唯一 RUN 为 experiments/neural_hz_20260831/results/d266_joint_guard_capacity_20261006_v1。保留 D265 的 4289 项及 232 个文件的完整原顺序，仅追加 test_guard_capacity.py 的四项，共 4293 项、233 文件：test_01_guard_identity_and_valid_points、test_02_strict_open_family_and_consumer、test_03_wide_residual_and_old_dominance、test_04_default_off_and_evidence。无候选 solver、模型、GPU、source worker、随机搜索或数值试跑。

四项分别检查：精确守卫恒等式及真实图点；固定开参数族对完整六产品 MC 加 X 加 QG 加旧 JP 的严格分离及后继 ReLU 上界；宽残差原负控和新行不弱于旧行；默认关闭、原预算失败粘滞、非有限及 512 bit 拒绝、证据写入。网格只为固定测试 oracle，不是候选相位或输入 split；离散检查不替代理论健全性证明。

原 60 秒单 pytest 进程、CPU 0 单线程、AS 16 GiB、双宿主观察 1 GiB、完整依赖/输入身份和证据重定位保持。新逻辑共享原 256M work、64M 累计 entries、512 bit 有理数上限；这些只是组件计费，不能充当全网资源资格。十四个冻结输入继承但不运行模型。本轮无真实来源人口执行，D265 未通过的构造预审不因新数学成功转正。

CONTRACT.md、本 PREREG.md、THEORY.md、CONTROL.md、NOVELTY.md、两个候选文件及 runner/plugin 均入冻结。测试输出只能写新 RUN；历史源码、历史证据、生产文件只读。正式 baseline 1870/2413、独立 E0 61/400、全部收益 0 不变。通过本轮不更新默认配置或正式成绩。
