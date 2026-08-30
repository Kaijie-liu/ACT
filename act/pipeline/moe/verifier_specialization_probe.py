"""Show that Route A specialization removes the dynamic verifier-front-end gap."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import torch
from torch import nn

from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256
from act.pipeline.moe.icml2025_b3 import PixelNormalizedExpert
from act.pipeline.moe.icml2025_route_telemetry import (
    OFFICIAL_COMMIT,
    OFFICIAL_REPO,
    _load_official_model,
)
from act.pipeline.moe.train import _load_dataset
from act.pipeline.moe.verifier_parser_probe import _export_and_check, _run_worker


DEFAULT_CONFIG = (
    PROJECT_ROOT / "act/pipeline/moe/configs/verifier_specialization_probe.json"
)


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


def specialize_official_expert(model: nn.Module, expert_index: int) -> nn.Module:
    """Return the raw-pixel branch obtained by fixing official hard dispatch."""

    experts = getattr(model, "experts", None)
    if not isinstance(experts, nn.ModuleList):
        raise TypeError("official model must expose experts as ModuleList")
    if not 0 <= int(expert_index) < len(experts):
        raise IndexError("expert index is outside the official model")
    return PixelNormalizedExpert(copy.deepcopy(experts[int(expert_index)])).float().eval()


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config_path = _inside(config_path, PROJECT_ROOT)
    output_dir = _inside(output_dir, WRITE_ROOT)
    if output_dir.exists():
        raise RuntimeError(f"specialization probe refuses to overwrite {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "PREREGISTERED_NOT_RUN":
        raise RuntimeError("specialization probe config is not frozen")
    if _repo_value(OFFICIAL_REPO, "rev-parse", "HEAD") != OFFICIAL_COMMIT or _repo_value(
        OFFICIAL_REPO, "status", "--porcelain"
    ):
        raise RuntimeError("official RT-ER clone identity/cleanliness gate failed")
    crown_repo = Path(config["consumer"]["repository"])
    if _repo_value(crown_repo, "rev-parse", "HEAD") != config["consumer"][
        "commit"
    ] or _repo_value(crown_repo, "status", "--porcelain"):
        raise RuntimeError("alpha-beta-CROWN clone identity/cleanliness gate failed")
    dynamic_probe = _inside(Path(config["dynamic_probe"]["path"]), PROJECT_ROOT)
    if _sha256(dynamic_probe) != config["dynamic_probe"]["sha256"]:
        raise RuntimeError("dynamic parser-probe result changed")
    dynamic_result = json.loads(dynamic_probe.read_text(encoding="utf-8"))
    dynamic_rt = next(
        row for row in dynamic_result["models"] if row["name"] == "rt_er_epoch010_hard_top1"
    )
    if dynamic_rt["overall"] != config["required_outcome"]["dynamic_model"]:
        raise RuntimeError("dynamic RT-ER result no longer has the frozen rejected status")

    checkpoint = _inside(Path(config["checkpoint"]["path"]), WRITE_ROOT)
    if _sha256(checkpoint) != config["checkpoint"]["sha256"]:
        raise RuntimeError("specialization checkpoint changed")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model, payload = _load_official_model(checkpoint, torch.device("cpu"))
    if int(payload.get("epoch", -1)) + 1 != int(config["checkpoint"]["epoch"]):
        raise RuntimeError("specialization checkpoint epoch changed")
    dataset = _load_dataset("CIFAR10", train=False, download=False)
    indices = [int(value) for value in config["export"]["probe_dataset_indices"]]
    probes = torch.stack([dataset[index][0] for index in indices]).float()
    if len(probes) != int(config["export"]["probe_samples"]):
        raise RuntimeError("specialization probe sample count changed")

    output_dir.mkdir(parents=True)
    forbidden = tuple(config["required_outcome"]["forbidden_operators_after_specialization"])
    records: list[dict[str, Any]] = []
    for expert_index in config["checkpoint"]["experts"]:
        expert_index = int(expert_index)
        specialized = specialize_official_expert(model, expert_index)
        with torch.no_grad():
            expected = model.experts[expert_index](
                (probes * 255.0 - specialized.mean) / specialized.std
            )
            actual = specialized(probes)
        if not torch.allclose(
            expected,
            actual,
            atol=float(config["export"]["semantic_atol"]),
            rtol=float(config["export"]["semantic_rtol"]),
        ):
            raise RuntimeError("Route A specialization changed expert semantics")
        record = _export_and_check(
            f"rt_er_epoch010_expert{expert_index}",
            specialized,
            probes,
            [(expert_index,)] * len(probes),
            output_dir,
            config,
        )
        if record["overall_status"] != "EXPORTED_SEMANTICS_MATCH":
            raise RuntimeError("specialized expert did not export with matching semantics")
        present = {
            name: int(record["onnx_operator_counts"].get(name, 0)) for name in forbidden
        }
        if any(present.values()):
            raise RuntimeError(f"specialized expert retained dispatch operators: {present}")
        record["specialization"] = {
            "expert": expert_index,
            "dispatch_fixed": True,
            "forbidden_operator_counts": present,
            "direct_semantics_match": True,
        }
        record["crown_frontend"] = _run_worker(record, output_dir)
        records.append(record)

    required = config["required_outcome"]["all_specialized_experts"]
    all_accepted = all(
        record["crown_frontend"].get("overall_status") == required
        for record in records
    )
    result = {
        "schema_version": 1,
        "status": "COMPLETED" if all_accepted else "FAILED_REQUIRED_ACCEPTANCE",
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "dynamic_reference": {
            "path": str(dynamic_probe),
            "sha256": _sha256(dynamic_probe),
            "overall_status": dynamic_rt["overall"],
            "rejection_message": dynamic_rt["rejection_message"],
        },
        "experts": records,
        "all_specialized_experts_accepted": all_accepted,
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
