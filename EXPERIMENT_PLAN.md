# 神经网络验证器验证框架 - 详细实验方案

## 论文概述

**标题**: Who Guards the Guards? Soundness Validation for Neural Network Verifiers

**核心问题**: 如何确保神经网络验证器本身的正确性？验证器中的bug（如不健全的抽象转换器或数值错误）可能导致错误的认证结果。

---

## 可复现性保证 (Reproducibility)

**论文投稿要求**: 所有实验必须完全可复现。审稿人/读者使用相同的种子和配置应得到完全一致的结果。

### 随机种子管理策略

```
┌─────────────────────────────────────────────────────────────────┐
│                     可复现性层次结构                              │
├─────────────────────────────────────────────────────────────────┤
│  Level 0: 全局主种子 (Master Seed)                               │
│           ↓                                                      │
│  Level 1: 实验种子 = hash(master_seed, experiment_id)            │
│           ↓                                                      │
│  Level 2: 网络种子 = hash(experiment_seed, network_idx)          │
│           ↓                                                      │
│  Level 3: 组件种子 = hash(network_seed, component_name)          │
│           (权重初始化、采样、变异等)                               │
└─────────────────────────────────────────────────────────────────┘
```

### 种子记录要求

每次实验必须记录并保存以下信息：

```python
@dataclass
class ExperimentMetadata:
    """实验元数据 - 用于复现"""
    # 种子信息
    master_seed: int              # 全局主种子
    experiment_id: str            # 实验标识符 (e.g., "rq1_detection")

    # 环境信息
    python_version: str           # e.g., "3.10.12"
    torch_version: str            # e.g., "2.1.0"
    numpy_version: str            # e.g., "1.24.3"
    cuda_version: Optional[str]   # e.g., "11.8" or None
    platform: str                 # e.g., "Linux-5.15.0-x86_64"

    # 配置信息
    config_hash: str              # 配置文件的SHA256哈希
    git_commit: str               # Git commit hash

    # 时间戳
    timestamp: str                # ISO 8601 格式

    def save(self, path: str) -> None:
        """保存到JSON文件"""
        ...

    @classmethod
    def load(cls, path: str) -> "ExperimentMetadata":
        """从JSON文件加载"""
        ...
```

### 网络生成的可复现性

**NetFactory 种子派生机制**:

```python
def derive_network_seed(master_seed: int, network_idx: int, instance_id: str) -> int:
    """
    确定性地派生网络种子

    Args:
        master_seed: 全局主种子
        network_idx: 网络在序列中的索引
        instance_id: 网络实例标识符

    Returns:
        派生的种子（32位无符号整数）
    """
    import hashlib
    payload = f"{master_seed}|{network_idx}|{instance_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)
```

**生成记录格式 (manifest.json)**:

```json
{
    "master_seed": 42,
    "experiment_id": "rq1_detection",
    "generation_timestamp": "2025-02-04T10:30:00Z",
    "config_file": "configs/config_gen_act_net.yaml",
    "config_hash": "sha256:abc123...",
    "num_networks": 100,
    "networks": [
        {
            "name": "mlp_plain_6_32x64x64_12345",
            "seed": 12345,
            "family": "mlp",
            "variant": "plain",
            "input_shape": [1, 6],
            "file_path": "nets/mlp_plain_6_32x64x64_12345.json"
        },
        ...
    ],
    "tf_targets": ["interval", "hybridz", "dual"],
    "registry_mode": "intersection",
    "environment": {
        "python": "3.10.12",
        "torch": "2.1.0",
        "numpy": "1.24.3",
        "platform": "Darwin-24.6.0"
    }
}
```

### 设置随机种子的正确方式

```python
def set_all_seeds(seed: int) -> None:
    """
    设置所有随机数生成器的种子

    必须在任何随机操作之前调用！
    """
    import random
    import numpy as np
    import torch

    # Python 内置随机
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (如果可用)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU

        # 确定性模式 (可能降低性能，但保证复现)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 环境变量 (用于某些库)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
```

### 实验运行脚本模板

```python
#!/usr/bin/env python3
"""
RQ1 实验脚本 - 可复现版本

使用方法:
    python experiments/rq1_detection.py --seed 42

复现:
    python experiments/rq1_detection.py --seed 42 --verify-reproducibility
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Master seed")
    parser.add_argument("--config", type=str, default="experiments/config.yaml")
    parser.add_argument("--output-dir", type=str, default="results/rq1")
    parser.add_argument("--verify-reproducibility", action="store_true",
                        help="Re-run and verify results match")
    args = parser.parse_args()

    # 1. 设置所有种子
    set_all_seeds(args.seed)

    # 2. 记录实验元数据
    metadata = ExperimentMetadata(
        master_seed=args.seed,
        experiment_id="rq1_detection",
        python_version=get_python_version(),
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        cuda_version=get_cuda_version(),
        platform=get_platform(),
        config_hash=hash_file(args.config),
        git_commit=get_git_commit(),
        timestamp=datetime.now().isoformat()
    )

    # 3. 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. 保存元数据
    metadata.save(output_dir / "metadata.json")

    # 5. 运行实验
    results = run_rq1_experiment(args.seed, args.config)

    # 6. 保存结果
    save_results(results, output_dir / "results.json")

    # 7. 如果是验证模式，比较结果
    if args.verify_reproducibility:
        expected = load_results(output_dir / "results_expected.json")
        verify_results_match(results, expected)
        print("✓ Reproducibility verified!")

if __name__ == "__main__":
    main()
```

---

## 分层验证架构

本框架采用**两级验证架构**，从外到内逐层检查验证器的正确性：

```
┌─────────────────────────────────────────────────────────────────┐
│                        验证器 (Verifier)                         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Level 2: BCA                              ││
│  │         边界包含审计 (Bound Containment Audit)                ││
│  │                                                              ││
│  │    检查每层神经元的具体激活值是否落在抽象边界内                  ││
│  │    ┌──────────┐  ┌──────────┐  ┌──────────┐                 ││
│  │    │  Layer 1 │→│  Layer 2 │→│  Layer N │                  ││
│  │    │ v∈[lb,ub]│  │ v∈[lb,ub]│  │ v∈[lb,ub]│                 ││
│  │    └──────────┘  └──────────┘  └──────────┘                 ││
│  └─────────────────────────────────────────────────────────────┘│
│                              ↑                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Level 1: SCC                              ││
│  │          语义交叉检查 (Semantic Cross-Check)                  ││
│  │                                                              ││
│  │    Input x ──→ [Network] ──→ Output y                       ││
│  │         ↓                        ↓                           ││
│  │    input_satisfied(x)?      output_satisfied(y)?            ││
│  │         ↓                        ↓                           ││
│  │    检查验证器判决是否与具体执行结果一致                        ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Level 1: SCC (语义交叉检查)

**定位**: 输出级验证 - "验证器的最终判决对不对？"

**目标**:
- 检测 **false CERTIFIED** 结果（验证器说安全，实际存在反例）
- 验证 **FALSIFIED** 结果的正确性（验证器给出的反例是否真实）

**方法**:
1. 从输入规范提取可采样区域（BOX 或 LINF_BALL）
2. 在区域内采样具体输入点
3. 执行网络获得具体输出
4. 检查：`input_satisfied(x) ∧ ¬output_satisfied(y)` → 真实反例

**检查逻辑**:
```
如果找到真实反例:
    验证器判决 = CERTIFIED  → Level 1 FAIL (健全性失败!)
    验证器判决 = FALSIFIED  → Level 1 PASS (证人一致)
    验证器判决 = UNKNOWN    → Level 1 ACCEPTABLE (不完备但无矛盾)
否则:
    → Level 1 INCONCLUSIVE (未发现问题，但不能证明正确)
```

**局限性**:
- 依赖随机采样，高维空间下覆盖率低
- LIN_POLY 规范无法直接采样
- 只能检测输出级错误，无法定位内部故障

---

### Level 2: BCA (边界包含审计)

**定位**: 内部级验证 - "验证器的每一步计算对不对？"

**目标**:
- 检查抽象解释的**基本健全性不变量**：具体值必须落在抽象边界内
- **精确定位**故障到具体的层/神经元

**方法**:
1. 使用 forward hooks 收集每层的具体激活值
2. 从验证器获取每层的抽象边界 [lb, ub]
3. 对齐激活与边界（按执行顺序）
4. 检查包含不变量：`lb - τ ≤ v ≤ ub + τ`（τ为数值容差）

**违规度量**:
```python
gap = max(lb - τ - v, v - ub - τ, 0)
# gap > 0 表示边界过紧，存在健全性违规
```

**检查逻辑**:
```
如果存在违规 (gap > 0):
    → Level 2 FAIL，报告违规位置和严重程度
如果对齐失败 (非顺序网络、重复调用等):
    → Level 2 ERROR，保守地拒绝给出结论
否则:
    → Level 2 PASS
```

**优势**:
- 不依赖采样，检查确定性不变量
- 可精确定位到层/神经元级别
- 即使 Level 1 INCONCLUSIVE，Level 2 仍可检测内部错误

**局限性**:
- 对齐问题：非顺序网络（残差、多分支）可能对齐失败
- BatchNorm 等状态相关层需要特殊处理
- 只检查边界包含，不检查约束正确性

---

### 两级互补关系

| 场景 | Level 1 (SCC) | Level 2 (BCA) | 结论 |
|------|---------------|---------------|------|
| 1 | FAIL | FAIL | 确定性健全性失败，可定位 |
| 2 | FAIL | PASS | 输出级错误，边界正确但约束/求解器问题 |
| 3 | FAIL | ERROR | 输出级错误，内部无法确认 |
| 4 | INCONCLUSIVE | FAIL | 采样未发现但内部边界有问题 |
| 5 | INCONCLUSIVE | PASS | 未发现问题（覆盖有限） |
| 6 | PASS | - | Level 1 只能 PASS 当 FALSIFIED 正确时 |

**关键洞察**:
- Level 1 是"黑盒"检查，关注输入-输出语义
- Level 2 是"白盒"检查，关注内部抽象边界
- 两级结合可以同时检测**输出级错误**和**内部级错误**

---

## 实验设计

### 研究问题 (Research Questions)

| RQ | 问题 | 检验对象 |
|----|------|----------|
| RQ1 | 检测注入的健全性违规 | Level 1 + Level 2 综合检测能力 |
| RQ2 | Level 1 (SCC) 的有效性边界 | 规范类型、输入维度对采样的影响 |
| RQ3 | Level 2 (BCA) 的定位准确性 | 不同架构下的定位能力和对齐可靠性 |
| RQ4 | TF感知生成是否提高覆盖率 | 操作符覆盖与bug发现的关系 |
| RQ5 | 不同抽象域的行为差异 | interval vs HybridZ vs Dual |
| RQ6 | 验证开销 | Level 1 和 Level 2 的运行时成本 |

---

## 第一阶段：核心模块实现

### 1.1 Level 1: SCC 模块

**文件位置**: `act/pipeline/verification/validate_verifier.py`

> **注意**: Level 1 (SCC) 功能已集成到 `VerificationValidator` 类中，复用现有的验证基础设施。

```python
"""
Level 1: Semantic Cross-Check (SCC)
输出级验证 - 通过 VerificationValidator.validate_counterexamples() 实现
"""

from act.pipeline.verification.validate_verifier import VerificationValidator

# 使用示例
validator = VerificationValidator(device="cuda", dtype=torch.float64)

# Level 1: 反例验证
level1_results = validator.validate_counterexamples(
    networks=["network_name"],
    solvers=['torchlp']
)

# 结果包含:
# - validation_status: "PASSED", "FAILED", "ACCEPTABLE", "INCONCLUSIVE"
# - concrete_counterexample: bool
# - samples_tried: int
```

**VerificationValidator 的 Level 1 方法**:
- `find_concrete_counterexample()`: 在输入空间采样寻找反例
- `validate_counterexamples()`: 批量验证网络列表

---

### 1.2 Level 2: BCA 模块

**文件位置**: `act/pipeline/verification/per_neuron_bounds.py`

> **注意**: Level 2 (BCA) 功能已集成到 `per_neuron_bounds` 模块中，复用现有的验证基础设施。

```python
"""
Level 2: Bound Containment Audit (BCA)
内部级验证 - 通过 run_per_neuron_bounds_check() 实现
"""

from act.pipeline.verification.per_neuron_bounds import (
    PerNeuronCheckConfig,
    run_per_neuron_bounds_check,
    compute_abstract_bounds,
    collect_concrete_activations,
    compare_bounds_per_neuron,
)

# 配置
config = PerNeuronCheckConfig(
    atol=1e-6,      # 绝对容差
    rtol=0.0,       # 相对容差
    topk=10,        # 报告最严重的k个违规
)

# 使用示例
result = run_per_neuron_bounds_check(
    net=net,
    model=model,
    input_tensor=x,
    tf_mode="interval",
    config=config,
)

# 结果包含:
# - status: "PASS", "FAIL", "ERROR"
# - violations: List of layer violations
# - worst_gap: float
# - layers_checked: int
# - neurons_checked: int
```

**核心函数**:
- `compute_abstract_bounds()`: 使用指定抽象域计算边界
- `collect_concrete_activations()`: 通过 forward hooks 收集具体激活
- `compare_bounds_per_neuron()`: 逐神经元比较边界包含
- `run_per_neuron_bounds_check()`: 完整的 Level 2 检查流程

---

### 1.3 可复现性工具模块

**文件位置**: `act/back_end/validation/reproducibility.py`

```python
"""
可复现性工具模块
确保实验结果可以完全复现
"""

import hashlib
import json
import os
import platform
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess

import numpy as np
import torch


def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    """
    设置所有随机数生成器的种子

    Args:
        seed: 随机种子
        deterministic: 是否启用完全确定性模式（可能降低性能）
    """
    # Python 内置随机
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # 环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)


def derive_seed(base_seed: int, *components) -> int:
    """
    从基础种子派生新种子

    Args:
        base_seed: 基础种子
        *components: 额外组件（用于区分不同用途）

    Returns:
        派生的32位无符号整数种子

    Example:
        >>> derive_seed(42, "rq1", 0)  # RQ1实验，第0个网络
        >>> derive_seed(42, "rq1", 0, "weights")  # 该网络的权重初始化
        >>> derive_seed(42, "rq1", 0, "scc", 5)  # 该网络SCC的第5个采样
    """
    payload = "|".join(str(c) for c in [base_seed] + list(components))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def hash_file(path: str) -> str:
    """计算文件的SHA256哈希"""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()[:16]}"


def get_git_commit() -> str:
    """获取当前Git commit hash"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()[:8]
    except Exception:
        return "unknown"


def get_environment_info() -> Dict[str, str]:
    """获取环境信息"""
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }

    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["cudnn"] = str(torch.backends.cudnn.version())
        info["gpu"] = torch.cuda.get_device_name(0)

    return info


@dataclass
class ExperimentMetadata:
    """实验元数据 - 用于复现"""
    # 基本信息
    experiment_id: str
    description: str = ""

    # 种子信息
    master_seed: int = 42
    seed_offset: int = 0

    # 配置信息
    config_path: Optional[str] = None
    config_hash: Optional[str] = None

    # 版本控制
    git_commit: str = field(default_factory=get_git_commit)

    # 环境信息
    environment: Dict[str, str] = field(default_factory=get_environment_info)

    # 时间戳
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None

    # 结果摘要
    num_networks: int = 0
    networks: List[Dict[str, Any]] = field(default_factory=list)

    def get_experiment_seed(self) -> int:
        """获取实验种子"""
        return self.master_seed + self.seed_offset

    def get_network_seed(self, network_idx: int, instance_id: str) -> int:
        """获取网络种子"""
        return derive_seed(self.get_experiment_seed(), network_idx, instance_id)

    def record_network(self, name: str, seed: int, **kwargs) -> None:
        """记录生成的网络"""
        self.networks.append({
            "name": name,
            "seed": seed,
            **kwargs
        })
        self.num_networks = len(self.networks)

    def finalize(self) -> None:
        """完成实验，记录结束时间"""
        self.end_time = datetime.now().isoformat()

    def save(self, path: str) -> None:
        """保存到JSON文件"""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentMetadata":
        """从JSON文件加载"""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


class ReproducibleNetFactory:
    """
    可复现的网络工厂包装器

    确保每个网络的生成都有确定性的种子
    """

    def __init__(
        self,
        base_factory,  # NetFactory instance
        metadata: ExperimentMetadata
    ):
        self.factory = base_factory
        self.metadata = metadata
        self._network_idx = 0

    def generate_one(self, **kwargs) -> "Net":
        """生成单个网络，记录种子"""
        # 派生网络种子
        temp_id = f"net_{self._network_idx}"
        network_seed = self.metadata.get_network_seed(self._network_idx, temp_id)

        # 设置种子
        set_all_seeds(network_seed)

        # 生成网络
        instance = self.factory._sample_instance(self._network_idx)
        name = instance["instance_id"]
        spec = self.factory._build_network_spec(instance, dtype="torch.float32")
        net = self.factory.create_network(name, spec)

        # 记录
        self.metadata.record_network(
            name=name,
            seed=network_seed,
            family=instance["family"],
            input_shape=list(instance["model_cfg"]["input_shape"]),
            index=self._network_idx
        )

        self._network_idx += 1
        return net

    def generate_batch(self, n: int) -> List["Net"]:
        """生成一批网络"""
        return [self.generate_one() for _ in range(n)]
```

---

### 1.4 Bug注入（Mutation）模块

**文件位置**: `act/back_end/validation/mutations.py`

```python
"""
变异操作 - 向转换函数注入已知缺陷
用于评估 Level 1 和 Level 2 的检测能力
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
import torch

from act.back_end.core import Bounds


class MutationType(Enum):
    """变异类型"""
    M1_TIGHTEN = "tighten_bounds"        # 收紧边界 (健全性违规)
    M2_LOOSEN = "loosen_bounds"          # 放松边界 (负控制)
    M3_SWAP = "swap_lb_ub"               # 交换上下界 (严重违规)
    M4_ZERO_LB = "zero_lower_bound"      # 下界置零 (健全性违规)
    M5_SCALE_UB = "scale_upper_bound"    # 缩放上界 (健全性违规)
    M6_NOISE = "add_noise"               # 添加噪声 (随机违规)


@dataclass
class MutationConfig:
    """变异配置"""
    mutation_type: MutationType
    target_layer_id: Optional[int] = None  # None表示所有层
    factor: float = 0.1
    seed: Optional[int] = None


class TFMutator:
    """转换函数变异器"""

    @staticmethod
    def M1_tighten_bounds(bounds: Bounds, factor: float = 0.1) -> Bounds:
        """
        M1: 收紧边界 - 减小边界宽度

        这会导致健全性违规：具体值可能落在收紧后的边界之外
        预期: Level 2 应该检测到违规
        """
        width = bounds.ub - bounds.lb
        shrink = width * factor
        return Bounds(
            lb=bounds.lb + shrink,
            ub=bounds.ub - shrink
        )

    @staticmethod
    def M2_loosen_bounds(bounds: Bounds, factor: float = 0.1) -> Bounds:
        """
        M2: 放松边界 - 增加边界宽度

        这是负控制：不应导致健全性违规
        预期: Level 1 和 Level 2 都不应报告错误
        """
        width = bounds.ub - bounds.lb
        expand = width * factor
        return Bounds(
            lb=bounds.lb - expand,
            ub=bounds.ub + expand
        )

    @staticmethod
    def M3_swap_lb_ub(bounds: Bounds) -> Bounds:
        """
        M3: 交换上下界

        严重的健全性违规：lb > ub
        预期: Level 2 应该立即检测到
        """
        return Bounds(lb=bounds.ub.clone(), ub=bounds.lb.clone())

    @staticmethod
    def M4_zero_lower_bound(bounds: Bounds) -> Bounds:
        """
        M4: 下界置零

        对于负值激活会导致健全性违规
        预期: 当存在负激活时 Level 2 检测到
        """
        return Bounds(
            lb=torch.zeros_like(bounds.lb),
            ub=bounds.ub.clone()
        )

    @staticmethod
    def M5_scale_upper_bound(bounds: Bounds, factor: float = 0.5) -> Bounds:
        """
        M5: 缩放上界

        可能导致上界过紧
        预期: 当激活值超过缩放后的上界时 Level 2 检测到
        """
        return Bounds(
            lb=bounds.lb.clone(),
            ub=bounds.ub * factor
        )

    @staticmethod
    def M6_add_noise(bounds: Bounds, std: float = 0.1, seed: int = None) -> Bounds:
        """
        M6: 添加随机噪声

        随机性健全性违规
        预期: 根据噪声方向 Level 2 可能检测到
        """
        if seed is not None:
            torch.manual_seed(seed)

        noise_lb = torch.randn_like(bounds.lb) * std
        noise_ub = torch.randn_like(bounds.ub) * std

        return Bounds(
            lb=bounds.lb + noise_lb,
            ub=bounds.ub + noise_ub
        )

    @classmethod
    def get_mutation_fn(cls, mutation_type: MutationType) -> Callable:
        """获取变异函数"""
        mapping = {
            MutationType.M1_TIGHTEN: cls.M1_tighten_bounds,
            MutationType.M2_LOOSEN: cls.M2_loosen_bounds,
            MutationType.M3_SWAP: cls.M3_swap_lb_ub,
            MutationType.M4_ZERO_LB: cls.M4_zero_lower_bound,
            MutationType.M5_SCALE_UB: cls.M5_scale_upper_bound,
            MutationType.M6_NOISE: cls.M6_add_noise,
        }
        return mapping[mutation_type]

    @classmethod
    def apply_mutation(
        cls,
        bounds: Bounds,
        config: MutationConfig
    ) -> Bounds:
        """应用变异配置"""
        fn = cls.get_mutation_fn(config.mutation_type)

        if config.mutation_type == MutationType.M6_NOISE:
            return fn(bounds, std=config.factor, seed=config.seed)
        elif config.mutation_type in [MutationType.M3_SWAP, MutationType.M4_ZERO_LB]:
            return fn(bounds)
        else:
            return fn(bounds, factor=config.factor)
```

---

## 第二阶段：统一验证流程

### 2.1 两级验证协调器

**文件位置**: `act/pipeline/verification/validate_verifier.py`

> **注意**: 两级验证通过 `VerificationValidator` 类统一协调，复用现有基础设施。

```python
"""
两级验证协调器
通过 VerificationValidator 协调 Level 1 (SCC) 和 Level 2 (BCA) 的执行
"""

from act.pipeline.verification.validate_verifier import VerificationValidator
from act.pipeline.verification.per_neuron_bounds import PerNeuronCheckConfig

# 创建验证器
validator = VerificationValidator(device="cuda", dtype=torch.float64)

# Level 1: 反例验证
level1_summary = validator.validate_counterexamples(
    networks=["network_name"],
    solvers=['torchlp']
)

# Level 2: 边界验证
per_neuron_config = PerNeuronCheckConfig(atol=1e-6, rtol=0.0, topk=10)
level2_summary = validator.validate_bounds(
    networks=["network_name"],
    tf_modes=["interval", "hybridz", "dual"],
    num_samples=10,
    per_neuron_config=per_neuron_config,
)

# 综合验证（Level 1 + Level 2）
comprehensive_summary = validator.validate_comprehensive(
    networks=["network_name"],
    tf_modes=["interval"],
    solvers=['torchlp'],
    per_neuron_config=per_neuron_config,
)
```

**VerificationValidator 核心方法**:
- `validate_counterexamples()`: Level 1 - 反例搜索
- `validate_bounds()`: Level 2 - 边界包含检查
- `validate_comprehensive()`: 综合验证（两级结合）

---

## 第三阶段：实验执行

### RQ1: 两级检测能力评估

**目标**: 评估 Level 1 和 Level 2 对注入缺陷的检测能力

```python
#!/usr/bin/env python3
# experiments/rq1_detection.py
"""
RQ1: 两级检测能力评估

可复现运行:
    python experiments/rq1_detection.py --seed 42 --mode real
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

from act.back_end.net_factory import NetFactory
from act.back_end.validation import (
    set_all_seeds, derive_seed, ExperimentMetadata, ReproducibleNetFactory,
    MutationType, MutationConfig, TFMutator,
)
from act.pipeline.verification.validate_verifier import VerificationValidator
from act.pipeline.verification.per_neuron_bounds import PerNeuronCheckConfig


def run_rq1_experiment(master_seed: int, config_path: str, output_dir: Path):
    """RQ1: 检测能力评估 - 使用现有验证基础设施"""

    # ===== 1. 初始化实验元数据 =====
    metadata = ExperimentMetadata(
        experiment_id="rq1_detection",
        description="两级检测能力评估",
        master_seed=master_seed,
        seed_offset=1000,
        config_path=config_path,
    )

    set_all_seeds(metadata.get_experiment_seed())

    # ===== 2. 定义实验配置 =====
    soundness_mutations = [
        MutationType.M1_TIGHTEN,
        MutationType.M3_SWAP,
        MutationType.M4_ZERO_LB,
        MutationType.M5_SCALE_UB,
        MutationType.M6_NOISE,
    ]
    control_mutations = [MutationType.M2_LOOSEN]
    domains = ["interval", "hybridz", "dual"]

    # ===== 3. 创建验证器（复用现有基础设施） =====
    validator = VerificationValidator(device="cuda", dtype=torch.float64)
    per_neuron_config = PerNeuronCheckConfig(atol=1e-6, rtol=0.0, topk=10)

    # ===== 4. 生成网络 =====
    base_factory = NetFactory(
        tf_targets=domains,
        base_seed=metadata.get_experiment_seed()
    )
    reproducible_factory = ReproducibleNetFactory(base_factory, metadata)
    networks = reproducible_factory.generate_batch(n=100)

    # ===== 5. 运行综合验证 =====
    results = {"metadata": {...}, "detection_results": defaultdict(list)}

    for net_info in metadata.networks:
        # Level 1: 反例验证
        level1_results = validator.validate_counterexamples(
            networks=[net_info["name"]],
            solvers=['torchlp']
        )

        # Level 2: 边界验证
        level2_results = validator.validate_bounds(
            networks=[net_info["name"]],
            tf_modes=domains,
            per_neuron_config=per_neuron_config,
        )

        # 记录检测结果
        # ...

    # ===== 6. 保存结果 =====
    metadata.finalize()
    metadata.save(output_dir / "metadata.json")
    # ...
```

### RQ2: Level 1 (SCC) 有效性边界

**目标**: 分析 Level 1 在不同条件下的有效性

```python
# experiments/rq2_scc_effectiveness.py

from act.pipeline.verification.validate_verifier import VerificationValidator
from act.back_end.validation import set_all_seeds, derive_seed, ExperimentMetadata

def run_rq2_experiment():
    """RQ2: Level 1 有效性分析（使用现有基础设施）"""

    validator = VerificationValidator(device="cuda", dtype=torch.float64)

    spec_types = ["BOX", "LINF_BALL", "LIN_POLY"]
    dimensions = [4, 16, 64, 256]

    for spec_type in spec_types:
        for dim in dimensions:
            # 生成网络
            nets = generate_networks(input_dim=dim, input_spec_kind=spec_type, n=30)

            for net in nets:
                # 运行 Level 1（复用 VerificationValidator）
                result = validator.validate_counterexamples(
                    networks=[net.name],
                    solvers=['torchlp']
                )

                # 记录结果
                # - INCONCLUSIVE 率（特别是 LIN_POLY）
                # - 反例发现率随维度的变化
```

### RQ3: Level 2 (BCA) 定位准确性

**目标**: 评估 Level 2 的故障定位能力

```python
# experiments/rq3_localization.py

from act.pipeline.verification.per_neuron_bounds import (
    run_per_neuron_bounds_check, PerNeuronCheckConfig
)
from act.back_end.validation import MutationType, MutationConfig, TFMutator

def run_rq3_experiment():
    """RQ3: Level 2 定位准确性（使用现有基础设施）"""

    config = PerNeuronCheckConfig(atol=1e-6, rtol=0.0, topk=10)

    architectures = ["sequential_mlp", "sequential_cnn", "residual"]

    for arch_name in architectures:
        for _ in range(30):
            net, model = generate_network(arch_name)

            # 选择目标层注入变异
            target_layer_id = select_injectable_layer(net)
            mutation_config = MutationConfig(
                mutation_type=MutationType.M1_TIGHTEN,
                target_layer_id=target_layer_id,
                factor=0.1,
            )

            # 运行 Level 2（复用 per_neuron_bounds）
            result = run_per_neuron_bounds_check(
                net=net,
                model=model,
                input_tensor=sample_input,
                tf_mode="interval",
                config=config,
            )

            # 检查定位准确性
            if result["status"] == "FAIL":
                violations = result["violations"]
                top1_correct = violations[0]["layer_id"] == target_layer_id
                top5_ids = [v["layer_id"] for v in violations[:5]]
                top5_correct = target_layer_id in top5_ids
```

---

## 实验配置

```yaml
# experiments/config.yaml

# ===================================================================
# 可复现性配置 (CRITICAL FOR PAPER SUBMISSION)
# ===================================================================
reproducibility:
  master_seed: 42                    # 全局主种子 - 修改此值将改变所有结果
  record_seeds: true                 # 记录所有派生种子
  deterministic_mode: true           # PyTorch 确定性模式
  verify_on_load: true               # 加载时验证哈希

# ===================================================================
# 实验通用配置
# ===================================================================
experiment:
  num_trials: 30                     # 独立运行次数（统计显著性）
  output_base_dir: "results"
  save_networks: true                # 保存生成的网络文件
  save_intermediate: true            # 保存中间结果

# ===================================================================
# Level 1 (SCC) 配置
# ===================================================================
level1:
  sampling_budget: 20                # 每种策略的采样数量
  strategies: ["uniform", "boundary", "center"]
  # 采样种子派生: sample_seed = hash(network_seed, "scc", sample_idx)

# ===================================================================
# Level 2 (BCA) 配置
# ===================================================================
level2:
  tolerance: 1e-5                    # 数值容差 τ
  max_violations_per_layer: 10       # 每层最多记录的违规数

# ===================================================================
# 各 RQ 实验配置
# ===================================================================

rq1:
  description: "两级检测能力评估"
  seed_offset: 1000                  # RQ1种子偏移: rq1_seed = master_seed + 1000
  num_networks: 100
  mutations: [M1, M3, M4, M5, M6]    # 健全性违规变异
  control: [M2]                      # 负控制（不应被检测）
  domains: [interval, hybridz, dual]
  mutation_factor: 0.1               # 变异强度

rq2:
  description: "Level 1 (SCC) 有效性边界"
  seed_offset: 2000
  spec_types: [BOX, LINF_BALL, LIN_POLY]
  dimensions: [4, 16, 64, 256]
  networks_per_config: 30

rq3:
  description: "Level 2 (BCA) 定位准确性"
  seed_offset: 3000
  architectures: [sequential_mlp, sequential_cnn, residual]
  networks_per_arch: 30
  mutation: M1
  topk: [1, 5]

rq4:
  description: "TF感知生成覆盖率"
  seed_offset: 4000
  configs:
    basic_50:  {mode: random, n: 50, seed_offset: 0}
    basic_100: {mode: random, n: 100, seed_offset: 100}
    full_100:  {mode: coverage, target: 0.95, max_attempts: 100, seed_offset: 200}

rq5:
  description: "跨域比较"
  seed_offset: 5000
  num_networks: 100
  domains: [interval, hybridz, dual]

rq6:
  description: "运行开销"
  seed_offset: 6000
  sampling_budgets: [5, 10, 20, 50]
  model_sizes:
    small:  {input_dim: 16, hidden: [32, 32], output_dim: 4}
    medium: {input_dim: 64, hidden: [128, 128, 64], output_dim: 10}
    large:  {input_dim: 256, hidden: [512, 256, 128], output_dim: 10}
    xlarge: {input_dim: 784, hidden: [1024, 512, 256], output_dim: 10}
```

### 种子派生规则

| 组件 | 派生公式 | 示例 |
|------|---------|------|
| RQ实验 | `rq_seed = master_seed + rq.seed_offset` | RQ1: 42 + 1000 = 1042 |
| 网络生成 | `net_seed = hash(rq_seed, net_idx, instance_id)` | hash(1042, 0, "mlp_...") |
| 权重初始化 | `weight_seed = hash(net_seed, "weights")` | hash(12345, "weights") |
| SCC采样 | `sample_seed = hash(net_seed, "scc", sample_idx)` | hash(12345, "scc", 0) |
| 变异操作 | `mutation_seed = hash(net_seed, "mutation", layer_id)` | hash(12345, "mutation", 3) |

---

## 目录结构

```
act/
├── back_end/
│   └── validation/
│       ├── __init__.py           # 统一导出接口
│       ├── mutations.py          # 变异操作（bug注入）
│       └── reproducibility.py    # 可复现性工具（种子管理、元数据）
├── pipeline/
│   └── verification/
│       ├── validate_verifier.py  # Level 1 + Level 2 验证协调器
│       ├── per_neuron_bounds.py  # Level 2: 边界包含审计核心
│       └── model_factory.py      # 模型工厂
└── util/
    └── device_manager.py         # 设备管理（CPU/GPU）

experiments/
├── config.yaml             # 实验配置（包含 master_seed）
├── rq1_detection.py        # RQ1: 检测能力
├── rq2_scc_effectiveness.py # RQ2: Level 1 有效性
├── rq3_localization.py     # RQ3: Level 2 定位
├── rq4_coverage.py         # RQ4: 覆盖率
├── rq5_cross_domain.py     # RQ5: 跨域比较
├── rq6_overhead.py         # RQ6: 开销
├── run_all.py              # 运行所有实验
├── verify_reproducibility.py # 验证可复现性
└── analyze_results.py      # 结果分析

results/                    # 实验结果（论文提交时包含）
├── rq1/
│   ├── metadata.json       # 实验元数据（种子、环境等）
│   ├── manifest.json       # 生成的网络清单
│   ├── networks/           # 生成的网络文件
│   │   ├── mlp_plain_6_32x64x64_12345.json
│   │   └── ...
│   └── results.json        # 实验结果
├── rq2/
│   └── ...
└── ...
```

---

## 论文提交检查清单

### 可复现性要求

```
□ 所有实验使用固定的 master_seed (默认: 42)
□ 每个网络的派生种子记录在 manifest.json
□ 环境信息完整记录（Python/PyTorch/NumPy版本）
□ 配置文件哈希值记录
□ Git commit hash 记录
```

### 提交物清单

```
artifact/
├── README.md                    # 复现指南
├── requirements.txt             # 依赖版本（固定）
├── experiments/
│   ├── config.yaml              # 实验配置（master_seed=42）
│   └── run_all.py               # 一键运行脚本
├── results/
│   ├── rq1/metadata.json        # 包含所有种子
│   ├── rq1/results.json         # 预期结果（用于验证）
│   └── ...
└── verify.sh                    # 验证脚本
```

### 验证脚本 (verify.sh)

```bash
#!/bin/bash
# 验证实验可复现性

set -e

echo "Setting up environment..."
pip install -r requirements.txt

echo "Running experiments with seed=42..."
python experiments/run_all.py --seed 42 --output-dir results_verify

echo "Comparing results..."
python experiments/verify_reproducibility.py \
    --expected results/ \
    --actual results_verify/

echo "✓ All experiments reproducible!"
```

### 复现性验证代码

```python
# experiments/verify_reproducibility.py

def verify_results_match(expected_dir: Path, actual_dir: Path) -> bool:
    """
    验证两次运行的结果是否完全一致

    检查项:
    1. 生成的网络数量和名称
    2. 每个网络的种子
    3. Level 1 和 Level 2 的检测结果
    4. 统计指标（检测率、定位准确率等）
    """
    mismatches = []

    for rq in ["rq1", "rq2", "rq3", "rq4", "rq5", "rq6"]:
        expected = json.load(open(expected_dir / rq / "results.json"))
        actual = json.load(open(actual_dir / rq / "results.json"))

        # 比较元数据
        if expected["metadata"]["master_seed"] != actual["metadata"]["master_seed"]:
            mismatches.append(f"{rq}: master_seed mismatch")

        # 比较网络清单
        if expected["networks"] != actual["networks"]:
            mismatches.append(f"{rq}: networks mismatch")

        # 比较检测结果
        if expected["detection_results"] != actual["detection_results"]:
            mismatches.append(f"{rq}: detection_results mismatch")

        # 比较统计指标（允许浮点数误差）
        for metric in ["detection_rate", "localization_accuracy"]:
            if metric in expected:
                diff = abs(expected[metric] - actual[metric])
                if diff > 1e-10:
                    mismatches.append(f"{rq}: {metric} differs by {diff}")

    if mismatches:
        print("❌ Reproducibility check FAILED:")
        for m in mismatches:
            print(f"  - {m}")
        return False

    print("✓ All results match exactly!")
    return True
```

---

## 总结

本实验方案采用**两级验证架构**：

| 级别 | 名称 | 定位 | 检查内容 | 优势 | 局限 |
|------|------|------|---------|------|------|
| **Level 1** | SCC | 输出级 | 验证器判决 vs 具体执行 | 直观、快速 | 依赖采样 |
| **Level 2** | BCA | 内部级 | 边界包含不变量 | 精确定位 | 对齐问题 |

两级互补，共同检测和定位验证器中的健全性缺陷。

---

## 模块复用说明

本框架最大限度复用 ACT 现有基础设施：

| 功能 | 复用模块 | 位置 |
|------|---------|------|
| **Level 1 (SCC)** | `VerificationValidator.validate_counterexamples()` | `act/pipeline/verification/validate_verifier.py` |
| **Level 2 (BCA)** | `run_per_neuron_bounds_check()` | `act/pipeline/verification/per_neuron_bounds.py` |
| **综合验证** | `VerificationValidator.validate_comprehensive()` | `act/pipeline/verification/validate_verifier.py` |
| **设备管理** | `initialize_device()`, `get_default_device()` | `act/util/device_manager.py` |
| **种子管理** | `set_all_seeds()`, `derive_seed()` | `act/back_end/validation/reproducibility.py` |
| **Bug注入** | `TFMutator`, `MutationType`, `MutationConfig` | `act/back_end/validation/mutations.py` |
| **元数据** | `ExperimentMetadata`, `ReproducibleNetFactory` | `act/back_end/validation/reproducibility.py` |
