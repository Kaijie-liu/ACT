#!/usr/bin/env python3
"""Run the frozen gate40 additions as sequential fresh-child diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
DATA = Path("/data1/Kane/data/vnncomp2025_benchmarks/benchmarks")
CASES = (
    ("cifar100_medium_iid86", "cifar100_2024", "CIFAR100_resnet_medium.onnx", "CIFAR100_resnet_medium_prop_idx_4964_sidx_5266_eps_0.0039.vnnlib"),
    ("cifar100_medium_iid64", "cifar100_2024", "CIFAR100_resnet_medium.onnx", "CIFAR100_resnet_medium_prop_idx_5973_sidx_7841_eps_0.0039.vnnlib"),
    ("cifar100_medium_iid0", "cifar100_2024", "CIFAR100_resnet_medium.onnx", "CIFAR100_resnet_medium_prop_idx_7641_sidx_1041_eps_0.0039.vnnlib"),
    ("cifar100_medium_iid10", "cifar100_2024", "CIFAR100_resnet_medium.onnx", "CIFAR100_resnet_medium_prop_idx_7704_sidx_3701_eps_0.0039.vnnlib"),
    ("cifar100_medium_iid21", "cifar100_2024", "CIFAR100_resnet_medium.onnx", "CIFAR100_resnet_medium_prop_idx_8594_sidx_5815_eps_0.0039.vnnlib"),
    ("cifar100_medium_iid3", "cifar100_2024", "CIFAR100_resnet_medium.onnx", "CIFAR100_resnet_medium_prop_idx_9694_sidx_2810_eps_0.0039.vnnlib"),
    ("cifar100_large_iid180", "cifar100_2024", "CIFAR100_resnet_large.onnx", "CIFAR100_resnet_large_prop_idx_1216_sidx_6431_eps_0.0039.vnnlib"),
    ("cifar100_large_iid133", "cifar100_2024", "CIFAR100_resnet_large.onnx", "CIFAR100_resnet_large_prop_idx_3681_sidx_4987_eps_0.0039.vnnlib"),
    ("cifar100_large_iid160", "cifar100_2024", "CIFAR100_resnet_large.onnx", "CIFAR100_resnet_large_prop_idx_5162_sidx_8126_eps_0.0039.vnnlib"),
    ("cifar100_large_iid102", "cifar100_2024", "CIFAR100_resnet_large.onnx", "CIFAR100_resnet_large_prop_idx_5308_sidx_1650_eps_0.0039.vnnlib"),
    ("cifar100_large_iid182", "cifar100_2024", "CIFAR100_resnet_large.onnx", "CIFAR100_resnet_large_prop_idx_5546_sidx_4338_eps_0.0039.vnnlib"),
    ("cifar100_large_iid161", "cifar100_2024", "CIFAR100_resnet_large.onnx", "CIFAR100_resnet_large_prop_idx_8502_sidx_2893_eps_0.0039.vnnlib"),
    ("tinyimagenet_medium_iid64", "tinyimagenet_2024", "TinyImageNet_resnet_medium.onnx", "TinyImageNet_resnet_medium_prop_idx_4327_sidx_7050_eps_0.0039.vnnlib"),
    ("tinyimagenet_medium_iid47", "tinyimagenet_2024", "TinyImageNet_resnet_medium.onnx", "TinyImageNet_resnet_medium_prop_idx_4500_sidx_6717_eps_0.0039.vnnlib"),
    ("tinyimagenet_medium_iid100", "tinyimagenet_2024", "TinyImageNet_resnet_medium.onnx", "TinyImageNet_resnet_medium_prop_idx_4686_sidx_8792_eps_0.0039.vnnlib"),
    ("tinyimagenet_medium_iid158", "tinyimagenet_2024", "TinyImageNet_resnet_medium.onnx", "TinyImageNet_resnet_medium_prop_idx_5442_sidx_6087_eps_0.0039.vnnlib"),
    ("tinyimagenet_medium_iid193", "tinyimagenet_2024", "TinyImageNet_resnet_medium.onnx", "TinyImageNet_resnet_medium_prop_idx_5922_sidx_1989_eps_0.0039.vnnlib"),
    ("tinyimagenet_medium_iid34", "tinyimagenet_2024", "TinyImageNet_resnet_medium.onnx", "TinyImageNet_resnet_medium_prop_idx_6546_sidx_3168_eps_0.0039.vnnlib"),
    ("tinyimagenet_medium_iid191", "tinyimagenet_2024", "TinyImageNet_resnet_medium.onnx", "TinyImageNet_resnet_medium_prop_idx_6700_sidx_2455_eps_0.0039.vnnlib"),
    ("tinyimagenet_medium_iid1", "tinyimagenet_2024", "TinyImageNet_resnet_medium.onnx", "TinyImageNet_resnet_medium_prop_idx_6827_sidx_2613_eps_0.0039.vnnlib"),
    ("tinyimagenet_medium_iid105", "tinyimagenet_2024", "TinyImageNet_resnet_medium.onnx", "TinyImageNet_resnet_medium_prop_idx_6957_sidx_1178_eps_0.0039.vnnlib"),
    ("tinyimagenet_medium_iid84", "tinyimagenet_2024", "TinyImageNet_resnet_medium.onnx", "TinyImageNet_resnet_medium_prop_idx_7749_sidx_8253_eps_0.0039.vnnlib"),
    ("tinyimagenet_medium_iid2", "tinyimagenet_2024", "TinyImageNet_resnet_medium.onnx", "TinyImageNet_resnet_medium_prop_idx_7943_sidx_6299_eps_0.0039.vnnlib"),
    ("tinyimagenet_medium_iid130", "tinyimagenet_2024", "TinyImageNet_resnet_medium.onnx", "TinyImageNet_resnet_medium_prop_idx_9600_sidx_3536_eps_0.0039.vnnlib"),
    ("tinyimagenet_medium_iid167", "tinyimagenet_2024", "TinyImageNet_resnet_medium.onnx", "TinyImageNet_resnet_medium_prop_idx_974_sidx_2976_eps_0.0039.vnnlib"),
    ("tinyimagenet_medium_iid115", "tinyimagenet_2024", "TinyImageNet_resnet_medium.onnx", "TinyImageNet_resnet_medium_prop_idx_9875_sidx_1510_eps_0.0039.vnnlib"),
)


def main() -> None:
    results = []
    for case, category, model_name, spec_name in CASES:
        family = DATA / category
        env = dict(os.environ)
        env.update(
            {
                "ACT_PHASE_PROJECTION_ONNX": str(family / "onnx" / model_name),
                "ACT_PHASE_PROJECTION_VNNLIB": str(family / "vnnlib" / spec_name),
                "ACT_PHASE_PROJECTION_CATEGORY": category,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scratch_phase_projection_probe.py")],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=60.0,
        )
        records = [
            json.loads(line)
            for line in completed.stdout.splitlines()
            if line.startswith("{")
        ]
        if completed.returncode == 0 and len(records) == 1:
            record = records[0]
        else:
            record = {
                "status": "UNKNOWN",
                "reason": "fresh child failed or emitted a malformed receipt",
                "returncode": completed.returncode,
            }
        record["case"] = case
        results.append(record)
        print(json.dumps(record, sort_keys=True, separators=(",", ":")), flush=True)
    print(
        json.dumps(
            {
                "schema": "act.hybridz.phase_projection_fixed40_additions.v1",
                "attempted": len(results),
                "singleton_verified": sum(
                    item.get("status") == "singleton_verified" for item in results
                ),
                "unknown": sum(item.get("status") == "UNKNOWN" for item in results),
                "results": results,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

