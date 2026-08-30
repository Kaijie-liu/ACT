"""Audit router gradient paths in three pinned, official MoE repositories.

This audit is deliberately source-level.  It does not install baseline
dependencies or execute training.  Conclusions are therefore about gradient
paths realized by the pinned released source, not about a checkpoint tensor
change unless a separate dynamic artifact is cited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable


MOE_ROOT = Path("/data1/Kane/MOE").resolve()
PROJECT_ROOT = (MOE_ROOT / "ACT").resolve()
BASELINE_ROOT = (MOE_ROOT / "baselines").resolve()

REPOSITORIES = {
    "rt_er": {
        "name": "Robust-MoE-Dual-Model",
        "url": "https://github.com/TIML-Group/Robust-MoE-Dual-Model",
        "path": BASELINE_ROOT / "Robust-MoE-Dual-Model",
        "commit": "30ef94d77b5451595b82e739aa8938e1f4c4521f",
        "license": "Apache-2.0",
    },
    "robust_moe_cnn": {
        "name": "robust-moe-cnn",
        "url": "https://github.com/optml-group/robust-moe-cnn",
        "path": BASELINE_ROOT / "robust-moe-cnn",
        "commit": "c50796fb8284512b6f6ad8e843f95182cec527cf",
        "license": "NOT_FOUND",
    },
    "vmoe": {
        "name": "vmoe",
        "url": "https://github.com/google-research/vmoe",
        "path": BASELINE_ROOT / "vmoe",
        "commit": "c07681241f81ba11421ba98e523e1499b2738a79",
        "license": "Apache-2.0",
    },
}


def _inside(path: Path, root: Path) -> Path:
    path = path.resolve()
    root = root.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"path escapes {root}: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def _source(repo: Path, relative: str) -> tuple[Path, str, list[str]]:
    path = _inside(repo / relative, repo)
    text = path.read_text(encoding="utf-8")
    return path, text, text.splitlines()


def _anchor(
    *, repo: Path, relative: str, key: str, needle: str, occurrence: int = 1
) -> dict[str, Any]:
    path, text, lines = _source(repo, relative)
    matches = [index + 1 for index, line in enumerate(lines) if needle in line]
    if len(matches) < occurrence:
        raise RuntimeError(
            f"missing anchor {key}: {relative!r} occurrence {occurrence} of {needle!r}"
        )
    return {
        "key": key,
        "path": relative,
        "line": matches[occurrence - 1],
        "predicate": f"line contains {needle!r}",
        "file_sha256": _sha256(path),
        "matched_line_sha256": hashlib.sha256(
            lines[matches[occurrence - 1] - 1].encode("utf-8")
        ).hexdigest(),
    }


def _hash_anchor(
    *,
    repo: Path,
    relative: str,
    key: str,
    line: int,
    matched_line_sha256: str,
) -> dict[str, Any]:
    """Create an anchor without embedding source from an unlicensed repository."""
    path, _, lines = _source(repo, relative)
    if line < 1 or line > len(lines):
        raise RuntimeError(f"invalid pinned anchor line {relative}:{line}")
    observed = hashlib.sha256(lines[line - 1].encode("utf-8")).hexdigest()
    if observed != matched_line_sha256:
        raise RuntimeError(f"pinned anchor line changed: {relative}:{line}")
    return {
        "key": key,
        "path": relative,
        "line": line,
        "predicate": "pinned line SHA-256 matches",
        "file_sha256": _sha256(path),
        "matched_line_sha256": matched_line_sha256,
    }


def _file_records(repo: Path, relatives: Iterable[str]) -> list[dict[str, str]]:
    records = []
    for relative in relatives:
        path = _inside(repo / relative, repo)
        records.append({"path": relative, "sha256": _sha256(path)})
    return records


def _repo_record(key: str) -> dict[str, Any]:
    spec = REPOSITORIES[key]
    repo = _inside(Path(spec["path"]), BASELINE_ROOT)
    head = _git(repo, "rev-parse", "HEAD")
    if head != spec["commit"]:
        raise RuntimeError(f"{key} commit drift: {head} != {spec['commit']}")
    status = _git(repo, "status", "--porcelain")
    if status:
        raise RuntimeError(f"{key} worktree is dirty")
    return {
        "name": spec["name"],
        "url": spec["url"],
        "branch": _git(repo, "branch", "--show-current"),
        "commit": head,
        "license": spec["license"],
        "worktree_clean": True,
    }


def _audit_rt_er() -> dict[str, Any]:
    repo = Path(REPOSITORIES["rt_er"]["path"])
    prior_path = PROJECT_ROOT / (
        "act/pipeline/moe/results/icml2025_rt_er/"
        "released_training_router_gradient_audit_20260830.json"
    )
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if prior.get("audit", {}).get("status") != "PASS":
        raise RuntimeError("prior RT-ER gradient audit is not PASS")
    for item in prior["source_audit"]["files"]:
        path = _inside(repo / item["path"], repo)
        if _sha256(path) != item["sha256"]:
            raise RuntimeError(f"RT-ER source hash drift: {item['path']}")
    return {
        **_repo_record("rt_er"),
        "router_update_class": "RELEASED_TRAINING_PATH_DOES_NOT_UPDATE_ROUTER",
        "route_semantics": "hard output-level argmax/top-k index dispatch",
        "gradient_mechanisms": [],
        "source_conclusion": (
            "All four released training entry points use integer route indices and "
            "place no router score in a differentiable loss."
        ),
        "dynamic_evidence": {
            "available": True,
            "scope": prior["dynamic_confirmation"]["scope"],
            "router_parameter_tensors_changed": prior["dynamic_confirmation"]
            ["router_parameter_tensors_changed"],
            "router_parameters_with_adam_state": prior["dynamic_confirmation"]
            ["router_parameters_with_adam_state"],
            "expert_parameter_tensors_changed": prior["dynamic_confirmation"]
            ["expert_parameter_tensors_changed"],
        },
        "prior_audit": {"path": str(prior_path), "sha256": _sha256(prior_path)},
        "files": prior["source_audit"]["files"],
        "claim_boundary": (
            "The conclusion concerns the pinned released training paths. Static "
            "routing is not intrinsically invalid."
        ),
    }


def _audit_robust_moe_cnn() -> dict[str, Any]:
    repo = Path(REPOSITORIES["robust_moe_cnn"]["path"])
    anchors = [
        _hash_anchor(repo=repo, relative="models/layers/router.py", key="router_parameter_layer", line=48, matched_line_sha256="7d70d99d612c97e8b2e628e22fd3c51ed2f3050ebedd32f6e0f1ed52cdb4f45d"),
        _hash_anchor(repo=repo, relative="models/layers/moe_layer.py", key="hard_argmax_dispatch", line=9, matched_line_sha256="7817c30762b69773e86aa874525dc114ce514aa1a0a3b5e4b4ec04319c9cd826"),
        _hash_anchor(repo=repo, relative="models/layers/moe_layer.py", key="straight_through_backward", line=16, matched_line_sha256="bc6d02226d4f26b949eefa1dee46ba8d5452eef5fa341ff7b17f44be083c1ab9"),
        _hash_anchor(repo=repo, relative="train_moe.py", key="main_optimizer_created_before_router_attachment", line=191, matched_line_sha256="d515584648f82510454b25e1802b9a3333323b31c1840ed9256cbf1d5949dc9c"),
        _hash_anchor(repo=repo, relative="train_moe.py", key="router_attached_after_main_optimizer", line=195, matched_line_sha256="2e66baa77c0090b7fc755ce7c6507974a62b2be273a8cb234a23edf6bc0b7095"),
        _hash_anchor(repo=repo, relative="train_moe.py", key="separate_router_optimizer", line=196, matched_line_sha256="31dde889cdfccb04319a546abadd218e4e69ab3338696ba34d81f970544a941f"),
        _hash_anchor(repo=repo, relative="train_moe.py", key="clean_router_scores", line=39, matched_line_sha256="e25380b286261325d19abd29c5d189a234ef6a080aea7baa2f9833b77bb3859a"),
        _hash_anchor(repo=repo, relative="train_moe.py", key="adversarial_router_scores", line=114, matched_line_sha256="c4b9f9e7aa1008400e9a281d910bb8af4f1f9f5afc02ddbc822208c9be3a81d5"),
        _hash_anchor(repo=repo, relative="train_moe.py", key="explicit_supervised_router_loss", line=143, matched_line_sha256="6b910c3039d3f67b2ead8bf9e4e2613a28eaed32aba2c3cb1134f47aa14e538a"),
        _hash_anchor(repo=repo, relative="train_moe.py", key="router_gradient_zeroed", line=155, matched_line_sha256="b69d0f0c995632e1c2835bc7a2d760030b70c42f2cfd84314de65cae7f0872cb"),
        _hash_anchor(repo=repo, relative="train_moe.py", key="router_optimizer_step", line=157, matched_line_sha256="da696ae7c1e3c072f26849fdbb407d5fcd33c48e9e4db0a8d6d9630678654b69"),
        _hash_anchor(repo=repo, relative="train_moe.py", key="router_optimizer_checkpointed", line=256, matched_line_sha256="a26dd994ce0f7eb57e4bbf55d2c73a75fd2f2943de97e7717cf08eefc34cb58b"),
    ]
    positions = {item["key"]: item["line"] for item in anchors if item["path"] == "train_moe.py"}
    ordered = [
        positions["main_optimizer_created_before_router_attachment"],
        positions["router_attached_after_main_optimizer"],
        positions["separate_router_optimizer"],
    ]
    if ordered != sorted(ordered) or len(set(ordered)) != 3:
        raise RuntimeError("robust-moe-cnn optimizer/attachment order changed")
    return {
        **_repo_record("robust_moe_cnn"),
        "router_update_class": "TRAINED_BY_EXPLICIT_ROUTER_OBJECTIVE",
        "route_semantics": "shared convolutional hard top-1 routing",
        "gradient_mechanisms": [
            "separate supervised router cross-entropy",
            "clean-versus-adversarial router KL",
            "explicit straight-through backward for hard one-hot selection",
        ],
        "source_conclusion": (
            "The released trainer directly differentiates a router-specific loss, "
            "updates a separate router optimizer, and checkpoints its state."
        ),
        "dynamic_evidence": {
            "available": False,
            "scope": "No baseline training or dependency installation was performed in this audit.",
        },
        "anchors": anchors,
        "files": _file_records(
            repo,
            [
                "train_moe.py",
                "models/layers/router.py",
                "models/layers/moe_layer.py",
                "utils/general_utils.py",
            ],
        ),
        "claim_boundary": (
            "This is a source-level gradient-path result. The repository has no "
            "located license, so no source was copied into ACT."
        ),
    }


def _audit_vmoe() -> dict[str, Any]:
    repo = Path(REPOSITORIES["vmoe"]["path"])
    anchors = [
        _anchor(repo=repo, relative="vmoe/nn/routing.py", key="router_dense_parameter", needle='name="dense")(inputs)'),
        _anchor(repo=repo, relative="vmoe/nn/routing.py", key="differentiable_gate_softmax", needle="gates_softmax = jax.nn.softmax(gates_logits)"),
        _anchor(repo=repo, relative="vmoe/nn/routing.py", key="importance_auxiliary_loss", needle="importance_loss = jax.vmap(self._importance_auxiliary_loss)(gates_softmax)"),
        _anchor(repo=repo, relative="vmoe/nn/routing.py", key="load_auxiliary_loss", needle="self._load_auxiliary_loss,"),
        _anchor(repo=repo, relative="vmoe/moe.py", key="selected_gate_combine_weights", needle='"SE,SEC->SEC", gates, dispatch_weights'),
        _anchor(repo=repo, relative="vmoe/nn/vit_moe.py", key="router_invoked_in_moe_block", needle="dispatcher, metrics = self.create_router()(inputs)"),
        _anchor(repo=repo, relative="vmoe/nn/vit_moe.py", key="auxiliary_losses_collected", needle="metrics['auxiliary_loss'] = sum("),
        _anchor(repo=repo, relative="vmoe/train/trainer.py", key="gradient_over_full_parameter_tree", needle="@functools.partial(jax.grad, has_aux=True)"),
        _anchor(repo=repo, relative="vmoe/train/trainer.py", key="main_plus_auxiliary_objective", needle="total_loss = metrics['main_loss'] + metrics.get('auxiliary_loss', 0.0)"),
        _anchor(repo=repo, relative="vmoe/train/trainer.py", key="full_state_parameters_supplied", needle="state.params, images, labels, state.rngs"),
        _anchor(repo=repo, relative="vmoe/train/trainer.py", key="gradients_applied", needle="state.apply_gradients_and_compute_global_norms("),
        _anchor(repo=repo, relative="vmoe/configs/vmoe_paper/pretrain_imagenet21k.py", key="published_e8_k2_configuration", needle="config.description = 'ViT-B/16, E=8, K=2, Every 2, 300 Epochs'"),
        _anchor(repo=repo, relative="vmoe/configs/vmoe_paper/pretrain_imagenet21k.py", key="positive_importance_weight", needle="config.encoder.moe.router.importance_loss_weight = 0.005"),
        _anchor(repo=repo, relative="vmoe/configs/vmoe_paper/pretrain_imagenet21k.py", key="positive_load_weight", needle="config.encoder.moe.router.load_loss_weight = 0.005"),
    ]
    config_text = _source(
        repo, "vmoe/configs/vmoe_paper/pretrain_imagenet21k.py"
    )[1]
    if "frozen_pattern" in config_text or "trainable_pattern" in config_text:
        raise RuntimeError("V-MoE paper pretraining config added a parameter freeze rule")
    return {
        **_repo_record("vmoe"),
        "router_update_class": "TRAINED_END_TO_END_BY_COMBINE_WEIGHTS_AND_AUXILIARY_LOSSES",
        "route_semantics": "hidden-layer noisy weighted top-k token routing",
        "gradient_mechanisms": [
            "selected softmax gate values combine expert outputs",
            "importance auxiliary loss",
            "load auxiliary loss",
            "full parameter-tree gradient with no router freeze in the pinned paper config",
        ],
        "source_conclusion": (
            "The published E=8, K=2 config creates Dense router parameters, uses "
            "gate values in expert-output combination and positive auxiliary losses, "
            "then differentiates and updates the complete parameter tree."
        ),
        "dynamic_evidence": {
            "available": False,
            "scope": "No V-MoE training or dependency installation was performed in this audit.",
        },
        "anchors": anchors,
        "paper_config_has_router_freeze_rule": False,
        "files": _file_records(
            repo,
            [
                "vmoe/nn/routing.py",
                "vmoe/nn/vit_moe.py",
                "vmoe/moe.py",
                "vmoe/train/trainer.py",
                "vmoe/train/optimizer.py",
                "vmoe/configs/vmoe_paper/pretrain_imagenet21k.py",
            ],
        ),
        "claim_boundary": (
            "This is a source-level path audit of the pinned official paper "
            "configuration, not a new checkpoint reproduction."
        ),
    }


def collect() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "COMPLETED_NOT_INDEPENDENTLY_AUDITED",
        "audit_scope": (
            "Pinned official repositories; source-level source-to-loss-to-optimizer "
            "gradient paths. No dependency installation or baseline training."
        ),
        "pipelines": {
            "rt_er": _audit_rt_er(),
            "robust_moe_cnn": _audit_robust_moe_cnn(),
            "vmoe": _audit_vmoe(),
        },
        "cross_pipeline_finding": {
            "static_released_router_training_paths": ["rt_er"],
            "learned_router_training_paths": ["robust_moe_cnn", "vmoe"],
            "external_validity_consequence": (
                "Learned routing is present in two official third-party pipelines, "
                "but their routing granularity and architecture differ from RT-ER."
            ),
        },
        "license_discipline": {
            "robust_moe_cnn": (
                "No license file was located. Only hashes, line predicates, and "
                "semantic findings are stored; no source is copied into ACT."
            )
        },
    }


def run(output_path: Path) -> dict[str, Any]:
    output_path = _inside(output_path, MOE_ROOT)
    if output_path.exists():
        raise RuntimeError(f"output already exists: {output_path}")
    result = collect()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
