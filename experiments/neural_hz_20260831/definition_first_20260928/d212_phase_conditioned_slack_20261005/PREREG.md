# 相位条件余量的独立算术核对范围

本次执行仅核对 RESEARCH.md 的精确有理恒等式、完整旧局部关系上的诊断点和四个固定公式数值。它不是候选域执行、完整数学组件测试、模型运行或基准回放。最后成功数学人口仍是 D209 的4100 tests/216 files，既不减少、不重跑，也不替代该人口。没有新的测试资格或正式收益。

执行前冻结本文件、RESEARCH.md 和 check_algebra.py 的 SHA256，并固定四项只读数学来源的 SHA256、branch、commit 和 tracked diff。执行前后全部核对，不导入任何旧候选、测试、模型、求解器或 D211 草稿。不修改任何旧文件。

固定核对内容如下。

1. 两张 cap 证书的全部仿射系数恒等式，共同 t、B_box、T_int、T 和新 h、M。
2. 相位消费恒等式全部系数，包括 beta、q1、q2，不仅代入样本。
3. 固定诊断点通过所有8条原graph和10条旧pair/closure关系及放松的完整因子界。
4. 同一点违反新反馈及可消费关系，p的两方向严格分离；不当作具体成员或ADV。
5. 四旧查询的闭式逐项结果、新查询v=(1,0)，以及只改tau的消融。比较域内固定证书，不调用LP求最优。
6. 本控制的完整四投影行，无新增z或bits；检查通用逐行支配不能直接扩大到B<1的反例数值。

禁止增加随机搜索、攻击、抽样选控制或事后换参数；禁止通过失败后改同一版本重跑。失败如实留档，代码不输出CERT/ADV。

只使用标准库 Fraction，所有中间有理数检查不超过512-bit。进程绑定CPU0、线程环境1、CUDA隐藏，AS16GiB、CPU与墙钟60秒上限。这个微型算术诊断没有候选work/entries或physical资格，不改变其whole256M/branch200M/entries64M与原物理门。输出资源仅为诊断自身，不与候选性能比较。

固定启动使用 /data1/Kane/miniconda3/bin/python，PYTHONOPTIMIZE=0、PYTHONDONTWRITEBYTECODE=1，OMP_NUM_THREADS、OPENBLAS_NUM_THREADS、MKL_NUM_THREADS、NUMEXPR_NUM_THREADS 均为1，CUDA_VISIBLE_DEVICES为空。脚本拒绝被关闭的assert或不符的线程/GPU环境，算术失败也进行后置来源核对并独立记录结果。

唯一新结果目录为 experiments/neural_hz_20260831/results/d212_phase_conditioned_slack_20261005_v1，必须不存在才启动。输出 algebra.json 和 exit.json 以独占创建自动保留；仅运行一次。所有资格字段为false，formal_gain=0。

未执行的后续候选仍需另行预注册、冻结、完整4100测试继承及新增定向测试，随后真实结构、shadow、逐家族与全2413/独立400回放。本诊断的通过绝不能越过任何一道门。
