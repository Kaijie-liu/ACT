# Dual RS 隔离部署：依赖与公开 CIFAR 基础模型

PI已授权隔离环境、作者依赖、公开权重与必要数据下载。本阶段不更换ACT/MetaMoE环境，
不启动ImageNet标签生成或论文大训练，不回填已有作者结果复算表。

## 固定来源与环境

- 作者源：`eth-sri/Dual-Randomized-Smoothing`，`6f83aeb7f47466b1dee6295baad8d59a8c94eceb`。
- 新环境：`/data1/Kane/MOE/envs/dual-rs-author-blackwell-20260921`；独立venv，无system-site-packages。
- Python3.11、Torch2.11/CUDA13、torchvision0.26，适配本机Blackwell；作者建议Python3.9，
  未发布完整版本锁。因此这是**兼容环境**，不是作者原始软件栈逐版本复现。
- 顶层依赖固定于[requirements](../configs/recent_moe/dual_rs_blackwell_requirements_20260921.txt)，
  解析后的全部包/解释器清单另行冻结；不以只有requirements文件冒充完整身份。
- `pip check`、原始分类器认证／噪声选择器认证／选择器训练CLI `--help`均通过。
  这些检查没有加载权重或产生认证半径。

## 权重与有界原生前向控制

只获取CIFAR diffusion和ViT，不执行作者下载脚本中无关的大型ImageNet权重部分。
diffusion采用作者`prepare_models.sh`所指OpenAI公开URL；取得后计算SHA-256，
没有作者额外公布的digest时明确只标本地身份。ViT固定HF revision
`63acc43bab8617ad96b6a9cc35760802ba495fa1`，LFS SHA-256独立匹配。
临时文件和失败不覆盖；每文件上限1GiB、下载阶段总预算900s，未完成不标可用。

原生前向控制固定：CIFAR10 test index0、seed1、batch1、sigma `{.25,.5,1}`；
使用作者未修改的`DiffusionRobustModel`（真实可变噪声denoiser+ViT）。
预算180s含来源、哈希、受限权重加载、数据与前向；CPU2线程、GPU空闲显存至少12GiB、
本进程allocator上限总显存10%。不抢占别人的工作，不自动重跑失败。
所有pickle先用`weights_only=True`验证为有限张量state_dict，不允许任意类加载回退。
这不是N=10,000的Monte Carlo认证，也不是clean accuracy测试。

## 仍缺的关键组件与下一步

作者仓库提供了sigma标签数组和训练脚本，但没有随库提供实际训练好的sigma-estimator权重。
不能用随机router、已有TSV或三个固定sigma前向代替完整Dual RS认证。

下一阶段在原代码/标签上先做单独冻结的真实训练一步＋保存恢复控制，记录数据选择、loss、
梯度和预处理；通过后再冻结90epoch选择器训练与checkpoint选择，不按认证结果挑模型。
现有[比较协议](recent_moe_comparison_protocol_v1.md)中的双阶段前2图认证仍保持
N0=100/N=10,000、两部分alpha=.0005；不能为了快速通过而减少N后仍冒称该协议。

源码`Smooth.predict()`的`binom_test`未导入问题仍未修改；当前前向不经过它。
若后续实际路径需要该函数，应作为单独兼容修复与回归测试，不能只因help通过就忽略。
原始确定性MoE与平滑后的函数保证不同，后续ACT比较仍需分表并披露L2/L∞转换。

结果在执行完成后追加；权重和环境留在ACT仓库外，不提交原始模型或数据。
