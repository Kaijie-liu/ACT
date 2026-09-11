"""One frozen real-checkpoint guarded-router HZ export, no property search."""
import argparse
import json
import os
from pathlib import Path

import torch

from act.back_end.moe import load_output_moe_checkpoint, build_act_moe_program, condition_topk_set
from act.back_end.solver.hz_lp_export import export
from act.back_end.solver.check_hz_lp_export import check_export
from act.back_end.solver.lp_certificate import propose
from act.front_end.specs import OutputSpec, OutKind
from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256, _git_value, _propagate_component
from act.pipeline.moe.train import _load_dataset
from act.pipeline.moe.staged_verifier import _tensor_identity
from act.pipeline.moe.paired_followup import save, source_identity
from act.util.device_manager import initialize_device


def run(path):
    if _git_value("branch", "--show-current") != "feat/moe-route-verification" or _git_value("status", "--porcelain"):
        raise RuntimeError("clean feature branch required")
    config = json.loads(path.read_text())
    selection_path = Path(config["selection"])
    if _sha256(selection_path) != config["selection_sha256"]:
        raise RuntimeError("selection drift")
    selection = json.loads(selection_path.read_text())
    subject = selection["models"][config["hz_export_model"]]
    checkpoint = _inside(Path(subject["checkpoint"]), WRITE_ROOT)
    if _sha256(checkpoint) != subject["checkpoint_sha256"]:
        raise RuntimeError("checkpoint drift")
    output = _inside(Path(config["hz_export_output"]), WRITE_ROOT)
    output.mkdir(exist_ok=False)
    os.environ["ACT_TORCHVISION_DATA_ROOT"] = str(PROJECT_ROOT / "data/torchvision")
    initialize_device("cpu", "float64")
    model, payload = load_output_moe_checkpoint(checkpoint, map_location="cpu")
    model.cpu().double().eval()
    index = selection["samples"][config["hz_export_rank"]]["dataset_index"]
    image, _ = _load_dataset(payload["dataset"], False, download=False)[index]
    center = image.unsqueeze(0).double()
    epsilon = selection["request"]["epsilon"]
    lower, upper = (center - epsilon).clamp(0, 1), (center + epsilon).clamp(0, 1)
    with torch.no_grad():
        logits, route = model.forward_with_routing(center)
    pair = sorted(int(v) for v in route.indices[0].tolist())
    outsider = next(i for i in range(model.spec.num_experts) if i not in pair)
    program = build_act_moe_program(model, center=center, lower=lower, upper=upper,
                                    output_spec=OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=[int(logits.argmax())]))
    router = _propagate_component(program.router)
    hz = condition_topk_set(router.output_hz, pair).hz
    q = [0] * hz.n_out
    q[pair[0]], q[outsider] = 1, -1
    record = export(hz, q)
    save(output / "export.json", record)
    torch.save({"center": center, "lower": lower, "upper": upper}, output / "request.pt")
    manifest = {"git_head": _git_value("rev-parse", "HEAD"), "source_sha256": source_identity(),
                "config_sha256": _sha256(path), "checkpoint_sha256": subject["checkpoint_sha256"],
                "selection_sha256": config["selection_sha256"], "dataset_index": index,
                "center": _tensor_identity(center), "lower": _tensor_identity(lower), "upper": _tensor_identity(upper),
                "pair": pair, "q": q, "request_sha256": _sha256(output / "request.pt"),
                "hz_sha256": record["source_sha256"], "export_sha256": _sha256(output / "export.json")}
    save(output / "manifest.json", manifest)
    certificate = propose(record["lp"], time_limit=config["hz_lp_seconds"])
    save(output / "certificate.json", certificate)
    checked = check_export(record, certificate, expected_source_sha256=manifest["hz_sha256"])
    save(output / "check.json", checked)
    print(json.dumps(checked, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)
