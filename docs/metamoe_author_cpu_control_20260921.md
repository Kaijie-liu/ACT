# MetaMoE 作者制品独立环境与两条真实组件控制

授权：PI在2026-09-21明确允许在`/data1/Kane/MOE`建立隔离环境、安装作者依赖、下载公开权重与必要数据。
本阶段不启动整批表格、不修改ACT/旧环境、不重训或重开封存的input98/holdout。

## 身份与偏差

- 作者repo `6aed3606e4b226e18e1c9d249485d99bb453f488`。
- 作者绑定的backend `58bb93f4886eea7cd1a3dfeb303695f20f61473b`，
  auto_LiRPA `28da3d0148ce320142f4a50b485ac71bca7acc3b`。
- onnx2pytorch取该backend环境文件固定的`325959ed128200459a9634668149a722a6a74797`。
- 新环境`/data1/Kane/MOE/envs/metamoe-author-cpu-20260921`，Python3.11、
  PyTorch2.3.1+cpu/torchvision0.18.1+cpu/numpy1.26.4；无system-site-packages。
  Python基础解释器来自现有3.11环境，第三方包独立安装，不宣称独立操作系统复现。
  Gurobi Python包按作者依赖安装，未取得/激活付费许可证；本控制不使用MIP/cuts。
- 论文写PyTorch2.1.0/CUDA12.1、backend短hash`d4c79e3`；该hash在作者完整克隆中不能解析，
  官方上游commit页面也返回404。未建立它与制品后端的等价性。
  本次是**发布制品CPU兼容控制**，不是论文硬件/环境的精确复现。
- 作者实验入口未设seed，backend默认100；本次保持100，而非默改成论文描述的42。
- 作者导出opset11，论文附录写13；本次沿用代码11，记录差异。

## 已完成的安装/加载控制

Torch、固定依赖、固定源码包分三次有界安装，日志位于
`/data1/Kane/MOE/baseline_runs/metamoe_install_20260921`。`pip check`通过；
原生`abcrown.py --help`通过，尚不等于实际界查询完成。
CIFAR数据从现有公开原始数据独立复制；MNIST由torchvision官方镜像下载并执行完整性检查。
两者各10,000测试输入；数据文件hash清单独立保存，未按验证结果选择输入。

CIFAR-RT及MNIST-RT作者专家均已在受限加载器下检查，仅允许7种审查过的class，
`weights_only=True`、无任意pickle回退。较旧的作者兼容环境不具备相同安全API；
真实控制仅允许这两个预先检查过且精确hash绑定的原始pickle，沿用作者加载与导出函数。
这些加载检查未测测试准确率，也没有router/完整MoE结论。

## 冻结的真实控制（不是性能比较）

[机器配置](../configs/recent_moe/metamoe_component_control_r1.json)。

| 项 | 预定值 |
|---|---|
| 两个请求 | CIFAR10-RT专家、MNIST-RT专家，各原测试顺序index0 |
| 选择 | 不按clean正确、路由数或bound挑选；错误分类也不换样本 |
| 域 | 作者归一化空间2/255，按作者代码裁剪[-10,10]及十位小数VNNLIB |
| 性质 | 每请求完整9条分类margin，允许找到并列输出的违规条件 |
| 方法 | 作者完整αβ-CROWN入口；PGD skip、alpha100/beta20、batch1024、kfsb3、cut关闭 |
| 预算 | solver配置300秒，外层每组件360秒（含启动、身份/数据、模型、导出、检查、后端、终态） |
| 偏差 | CPU/2线程、显式默认seed100、独立输出目录；不是原RTX4060速度复现 |
| 停止 | 执行ERROR停止后续；TIMEOUT保留；不替换、不加时、不自动重跑 |

使用原作者模型／BN折叠／ONNX导出／VNNLIB及配置函数；外层替代原CLI的两处不闭合接口：
验证子进程错误未传到顶层退出码、原ONNX验证失败只打印而不拒绝。
我们在原图像、零输入及固定随机输入上检查**原始checkpoint到ONNX**误差<1e-4，
未沿用只比较折叠模型与ONNX的一次探针。有限探针不是全域等价证明。

所有请求独立付费加载和准备；安装、数据获取的一次性成本另外报告。
后端继承外层独立进程组，外层截止终止该请求及其子进程，保留阶段与原始日志。
原生正结果超过300秒后端墙钟或被外层截止时不接受为本控制的按时正结果。

## 保证边界

`BACKEND_POSITIVE`只表示后端对**已导出的ONNX＋十进制VNNLIB**返回正结果。
十位小数可能向内移动原float32盒边界，必须记录坐标数，不冒称证明原请求盒。
网络→BN折叠→ONNX等价、原生数值误差、完整router+专家组合均未由此独立证明。
`BACKEND_UNSAFE_UNREPLAYED`不是完整动态MoE反例，不能填整体UNSAFE栏。
审计只重核文件身份、完整性质、终态与成本，不重新求界；两条正结果也不等于Fig3/4复现。

## 执行及审计

冻结执行代码及配置、控制测试并commit/push后运行：

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python scripts/metamoe_component_control.py --config configs/recent_moe/metamoe_component_control_r1.json
/data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/audit_metamoe_component_control.py --config configs/recent_moe/metamoe_component_control_r1.json --output docs/metamoe_component_control_review_20260921_r1.json
```

准备状态：两条实际界查询尚未启动。结果须追加，不覆盖本冻结说明及首次失败。

## R1实际执行与R2兼容修复（追加，不覆盖）

R1在`c058caf2f`冻结后执行：CIFAR index0真实预测为3，与label3一致；
原始checkpoint→ONNX的三次探针最大差约8.83e-6，低于固定1e-4。
之后后端在ONNX解析失败：`count_include_pad`属性未实现；**尚未开始求界**。
按fail-stop，MNIST为`NOT_STARTED_AFTER_ERROR`。独立记录审计通过只说明这次失败记账闭合。
[R1审计](metamoe_component_control_review_20260921_r1.json)。

已定位最小上游修复：onnx2pytorch `8447c42c3192dad383e5598edc74dddac5706ee2`
的直接父提交恰为R1的`325959ed...`。差分仅加入`count_include_pad`到PyTorch布尔属性的
两行映射，以及一处空白修正。无需删除池化属性、修改模型或换CROWN策略。

R2在**另一个隔离环境**重新安装R1相同的全部包版本，仅更换这一个源码commit；
保留R1环境、图、数据、日志及终态。先测padding0/1×include_pad0/1四个ONNX/转换结果，
再转换R1实际失败图并检查相同输入上的误差，全部通过才冻结R2。
相同两个index0、checkpoint、epsilon、性质、迭代与预算，新结果目录；
这属于版本化前端兼容修复，不是发现负界之后调整认证策略。
R1原生子进程墙钟约2.04秒，整个worker约4.32秒；不得将R2覆盖R1或隐去部署失败成本。

R2冻结前检查：`pip check`通过；四种pool探针逐值一致，R1失败图的
ONNX Runtime/转换模型探针误差4.7684e-7。R2全部包版本与R1相同（源码commit单独绑定）。
[R2配置](../configs/recent_moe/metamoe_component_control_r2.json)绑定检查回执及结果，
冻结时未执行真实请求。23项控制测试通过，旧论文表格重建未变。

## R2执行完成（冻结提交0d842e060）

| 请求 | 终态 | 后端墙钟 | 全请求外层墙钟 |
|---|---|---:|---:|
| CIFAR10 RT index0 | `BACKEND_POSITIVE`，native `safe-incomplete` | 3.026s | 5.174s |
| MNIST RT index0 | `ERROR`，导出一致性未过，未运行求界 | 未启动 | 1.667s |

CIFAR原生后端在initial CROWN即验证全部9条性质，打印下界最小约0.22790134。
`safe-incomplete`是该后端“不完整验证阶段已证明”的名称，不表示漏掉性质；但本项目
仍只将它记作作者ONNX/VNNLIB后端正结果，**不是**完整MoE或source-complete证书。
两次CIFAR导出ONNX和VNNLIB哈希完全相同，R2没有改变模型/输入图以得到正界。

MNIST第二个预注册探针（零张量）误差2.79307365e-4，超过原1e-4门，自动拒绝。
仅在原保存图/原输入及零探针上作无求解诊断：

| 探针 | 原模型32 vs 折叠32 | 折叠32 vs ONNX | 原模型64 vs 重新折叠64 |
|---|---:|---:|---:|
| 实际index0 | 5.6744e-5 | 4.7684e-7 | 1.6787e-13 |
| 预注册零输入 | 2.8038e-4 | 2.4885e-6 | 3.2085e-13 |

所有12个折叠参数与保存ONNX逐值相同。该有限诊断支持主要观测差异在float32 BN折叠
及其计算路径；没有证明全域等价、唯一底层kernel原因或模型不安全。
64位仅用于定位，**没有**更改生产dtype、阈值、输入、权重或后端，也没有MNIST求界重跑。
零探针不在该index0的扰动盒内；它是事前注册的转换兼容控制。
失败并非该盒内等价错误的证明，更不是模型不安全或论文认证结果错误的证明。

[独立记录审计R2](metamoe_component_control_review_20260921_r2.json)通过；
[完整紧凑归档](metamoe_author_control_archive_20260921_r2.json)包含R1/R2原始文件哈希、
环境、安装和有限诊断成本。检查器核对记录，不重新证明网络界。
五个旧环境及R1环境的包/解释器清单不变；两个隔离环境各约1.6GiB，公开数据约242MiB。
安装和数据准备是另外计费的部署成本，不能加进或替代上述逐请求耗时。

科学状态：**一条真实作者组件验证已跑通，一条按源转换门拒绝**。
不得写“MetaMoE整表已复现”或“已完成和ACT的同任务竞争”。
