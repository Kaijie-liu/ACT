"""Export two dynamic MoEs and probe α,β-CROWN's program front end."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

from act.util.typing_compat import install_typing_override

install_typing_override()

import numpy as np
import torch
from torch import nn

from act.back_end.moe import load_output_moe_checkpoint
from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256
from act.pipeline.moe.icml2025_route_telemetry import (
    CIFAR_MEAN_255,
    CIFAR_STD_255,
    OFFICIAL_COMMIT,
    OFFICIAL_REPO,
    _grouped_official_forward,
    _load_official_model,
)
from act.pipeline.moe.train import _load_dataset


DEFAULT_CONFIG = PROJECT_ROOT / "act/pipeline/moe/configs/verifier_parser_probe.json"
CROWN_PYTHON = Path("/data1/Kane/MOE/envs/alpha-beta-crown/bin/python")
WORKER = PROJECT_ROOT / "act/pipeline/moe/verifier_parser_probe_worker.py"


class VectorizedOfficialHardMoE(nn.Module):
    """Exact tensorized form of the official hard dispatch for export only."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.router = model.router
        self.experts = model.experts
        self.register_buffer(
            "mean", torch.as_tensor(CIFAR_MEAN_255, dtype=torch.float32)[None, :, None, None]
        )
        self.register_buffer(
            "std", torch.as_tensor(CIFAR_STD_255, dtype=torch.float32)[None, :, None, None]
        )

    def normalized(self, pixels: torch.Tensor) -> torch.Tensor:
        return (pixels * 255.0 - self.mean) / self.std

    def route(self, pixels: torch.Tensor) -> torch.Tensor:
        normalized = self.normalized(pixels)
        return torch.argmax(self.router.gate(normalized.flatten(1)), dim=1)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        normalized = self.normalized(pixels)
        scores = self.router.gate(normalized.flatten(1))
        routes = torch.argmax(scores, dim=1)
        expert_values = torch.stack(
            [expert(normalized) for expert in self.experts], dim=1
        )
        index = routes[:, None, None].expand(-1, 1, expert_values.shape[2])
        return torch.gather(expert_values, 1, index).squeeze(1)


def _repo_value(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _save_npz(path: Path, **arrays: np.ndarray) -> str:
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256(path)


def diverse_probe_slots(keys: Sequence[tuple[int, ...]], samples: int) -> list[int]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    selected: list[int] = []
    seen: set[tuple[int, ...]] = set()
    for slot, key in enumerate(keys):
        key = tuple(int(value) for value in key)
        if key not in seen:
            selected.append(slot)
            seen.add(key)
        if len(selected) == samples:
            return selected
    for slot in range(len(keys)):
        if slot not in selected:
            selected.append(slot)
        if len(selected) == samples:
            return selected
    raise RuntimeError("not enough probe inputs")


def _export_and_check(
    name: str,
    model: nn.Module,
    probes: torch.Tensor,
    route_keys: Sequence[tuple[int, ...]],
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort

    model = model.float().eval()
    probes = probes.float()
    onnx_path = output_dir / f"{name}.onnx"
    probes_path = output_dir / f"{name}_probes.npz"
    with torch.no_grad():
        expected = model(probes).cpu().numpy()
    export_started = time.monotonic()
    try:
        torch.onnx.export(
            model,
            probes[:1],
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=int(config["export"]["opset"]),
            dynamo=False,
        )
        export_error = None
    except Exception as error:
        return {
            "name": name,
            "overall_status": "EXPORT_REJECTED",
            "export_error_type": type(error).__name__,
            "export_error": str(error),
            "export_seconds": time.monotonic() - export_started,
        }
    try:
        graph = onnx.load(str(onnx_path))
        onnx.checker.check_model(graph)
        operators = Counter(node.op_type for node in graph.graph.node)
        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        runtime_output = session.run(
            None, {session.get_inputs()[0].name: probes.numpy()}
        )[0]
    except Exception as error:
        return {
            "name": name,
            "overall_status": "REFERENCE_ONNX_REJECTED",
            "export_seconds": time.monotonic() - export_started,
            "onnx": str(onnx_path),
            "onnx_sha256": _sha256(onnx_path),
            "reference_error_type": type(error).__name__,
            "reference_error": str(error),
        }
    maximum_error = float(np.max(np.abs(runtime_output - expected)))
    semantic_match = bool(
        np.allclose(
            runtime_output,
            expected,
            atol=float(config["export"]["semantic_atol"]),
            rtol=float(config["export"]["semantic_rtol"]),
        )
    )
    probe_hash = _save_npz(
        probes_path,
        inputs=probes.numpy(),
        outputs=expected,
        route_keys=np.asarray(route_keys, dtype=np.int64),
    )
    return {
        "name": name,
        "overall_status": (
            "EXPORTED_SEMANTICS_MATCH" if semantic_match else "SILENT_SEMANTIC_MISMATCH"
        ),
        "export_seconds": time.monotonic() - export_started,
        "onnx": str(onnx_path),
        "onnx_sha256": _sha256(onnx_path),
        "onnx_nodes": len(graph.graph.node),
        "onnx_operator_counts": dict(sorted(operators.items())),
        "dynamic_dispatch_operators": {
            key: operators.get(key, 0)
            for key in ("TopK", "ArgMax", "Gather", "GatherElements")
        },
        "onnxruntime_semantic_match": semantic_match,
        "onnxruntime_maximum_abs_error": maximum_error,
        "probes": str(probes_path),
        "probes_sha256": probe_hash,
        "route_keys": [list(key) for key in route_keys],
        "export_error": export_error,
    }


def _run_worker(record: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if record["overall_status"] != "EXPORTED_SEMANTICS_MATCH":
        return {"status": "NOT_RUN_EXPORT_FAILED_OR_MISMATCHED"}
    output = output_dir / f"{record['name']}_crown.json"
    log = output_dir / f"{record['name']}_crown.log"
    command = [
        str(CROWN_PYTHON),
        str(WORKER),
        "--onnx",
        record["onnx"],
        "--probes",
        record["probes"],
        "--output",
        str(output),
    ]
    with log.open("xb") as handle:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0 or not output.is_file():
        return {
            "status": "WORKER_FAILED",
            "returncode": completed.returncode,
            "log": str(log),
            "log_sha256": _sha256(log),
        }
    value = json.loads(output.read_text(encoding="utf-8"))
    return {
        **value,
        "output": str(output),
        "output_sha256": _sha256(output),
        "log": str(log),
        "log_sha256": _sha256(log),
    }


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    output_dir = _inside(output_dir, WRITE_ROOT)
    if output_dir.exists():
        raise RuntimeError(f"parser probe refuses to overwrite {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "PREREGISTERED_NOT_RUN":
        raise RuntimeError("parser probe config is not frozen")
    if _repo_value(OFFICIAL_REPO, "rev-parse", "HEAD") != OFFICIAL_COMMIT or _repo_value(
        OFFICIAL_REPO, "status", "--porcelain"
    ):
        raise RuntimeError("official RT-ER clone identity/cleanliness gate failed")
    crown_repo = Path(config["consumer"]["repository"])
    if _repo_value(crown_repo, "rev-parse", "HEAD") != config["consumer"]["commit"] or _repo_value(
        crown_repo, "status", "--porcelain"
    ):
        raise RuntimeError("α,β-CROWN clone identity/cleanliness gate failed")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.dont_write_bytecode = True
    dataset = _load_dataset("CIFAR10", train=False, download=False)
    pool = torch.stack([dataset[index][0] for index in range(128)]).float()
    output_dir.mkdir(parents=True)
    records: list[dict[str, Any]] = []

    bal_config = config["models"]["bal010_weighted_top2"]
    bal_checkpoint = _inside(Path(bal_config["checkpoint"]), WRITE_ROOT)
    if _sha256(bal_checkpoint) != bal_config["checkpoint_sha256"]:
        raise RuntimeError("bal010 checkpoint changed")
    bal_model, _payload = load_output_moe_checkpoint(
        bal_checkpoint, map_location="cpu"
    )
    bal_model = bal_model.float().eval()
    with torch.no_grad():
        bal_routes = bal_model.route(pool).indices.cpu().numpy()
    bal_keys = [tuple(sorted(map(int, row))) for row in bal_routes]
    bal_slots = diverse_probe_slots(bal_keys, int(config["export"]["probe_samples"]))
    records.append(
        _export_and_check(
            "bal010_weighted_top2",
            bal_model,
            pool[bal_slots],
            [bal_keys[slot] for slot in bal_slots],
            output_dir,
            config,
        )
    )

    rt_config = config["models"]["rt_er_epoch010_hard_top1"]
    rt_checkpoint = _inside(Path(rt_config["checkpoint"]), WRITE_ROOT)
    if _sha256(rt_checkpoint) != rt_config["checkpoint_sha256"]:
        raise RuntimeError("RT-ER epoch10 checkpoint changed")
    rt_model, payload = _load_official_model(rt_checkpoint, torch.device("cpu"))
    if int(payload.get("epoch", -1)) + 1 != 10:
        raise RuntimeError("RT-ER parser checkpoint is not epoch10")
    vectorized = VectorizedOfficialHardMoE(rt_model).float().eval()
    with torch.no_grad():
        rt_routes = vectorized.route(pool).cpu().numpy()
    rt_keys = [(int(value),) for value in rt_routes]
    rt_slots = diverse_probe_slots(rt_keys, int(config["export"]["probe_samples"]))
    rt_probes = pool[rt_slots]
    with torch.no_grad():
        normalized = vectorized.normalized(rt_probes)
        official_output, official_scores = _grouped_official_forward(
            rt_model, normalized
        )
        vectorized_output = vectorized(rt_probes)
        vectorized_scores = vectorized.router.gate(normalized.flatten(1))
    official_maximum_error = float(
        (official_output - vectorized_output).abs().max().item()
    )
    score_maximum_error = float(
        (official_scores - vectorized_scores).abs().max().item()
    )
    if not torch.equal(official_scores.argmax(dim=1), vectorized.route(rt_probes)):
        raise RuntimeError("tensorized RT-ER wrapper changed official hard routes")
    if not torch.allclose(
        official_output,
        vectorized_output,
        atol=float(config["export"]["semantic_atol"]),
        rtol=float(config["export"]["semantic_rtol"]),
    ):
        raise RuntimeError("tensorized RT-ER wrapper changed official outputs")
    rt_record = _export_and_check(
        "rt_er_epoch010_hard_top1",
        vectorized,
        rt_probes,
        [rt_keys[slot] for slot in rt_slots],
        output_dir,
        config,
    )
    rt_record["official_grouped_forward_crosscheck"] = {
        "outputs_match": True,
        "maximum_abs_error": official_maximum_error,
        "router_score_maximum_abs_error": score_maximum_error,
        "routes_match_exactly": True,
    }
    records.append(rt_record)

    for record in records:
        record["crown_frontend"] = _run_worker(record, output_dir)
    result = {
        "schema_version": 1,
        "status": "COMPLETED",
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "models": records,
        "conclusion_scope": config["claim_scope"],
        "official_clone_clean_after": not bool(
            _repo_value(OFFICIAL_REPO, "status", "--porcelain")
        ),
        "crown_clone_clean_after": not bool(
            _repo_value(crown_repo, "status", "--porcelain")
        ),
    }
    _write_json(output_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
