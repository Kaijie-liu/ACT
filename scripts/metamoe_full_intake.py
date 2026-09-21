"""Frozen original class-separated top-1 intake under a hard outer supervisor."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

from recent_moe_deployment import git_identity, sha256, supervise


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def validate(cfg):
    identity = git_identity(cfg["repo"])
    if identity["head"] != cfg["commit"] or identity["status"]:
        raise ValueError("author checkout identity")
    for p, h in cfg["files"].items():
        if sha256(p) != h:
            raise ValueError("frozen file changed: " + p)


def load_original(cfg):
    import torch
    sys.path.insert(0, str(Path(cfg["repo"]) / "src/Vision_Transformer_Pytorch"))
    from small_expert import UltraVerifiableCNN, UltraVerifiableCNN_Features
    from vision_transformer_moe import MetaMoE, MetaGatingNet
    allowed = [UltraVerifiableCNN, UltraVerifiableCNN_Features, MetaMoE, MetaGatingNet,
        torch.nn.Dropout, torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d,
        torch.nn.AvgPool2d, torch.nn.ModuleList]
    names = {f"{c.__module__}.{c.__qualname__}" for c in allowed}
    unexpected = set(torch.serialization.get_unsafe_globals_in_checkpoint(cfg["checkpoint"])) - names
    if unexpected:
        raise ValueError(f"unreviewed checkpoint types {unexpected}")
    with torch.serialization.safe_globals(allowed):
        model = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)
    if type(model) is not MetaMoE or model.meta_top_k != 1:
        raise ValueError("unexpected author model")
    return model.eval(), sorted(names)


def run_worker(cfg, config_path, request):
    started = time.monotonic()
    validate(cfg)
    import torch
    from torchvision import datasets, transforms
    from act.back_end.moe.class_separated_top1 import (
        ClassSeparatedTop1, classification_rows, verify_class_separated_box)
    from act.util.device_manager import initialize_device
    initialize_device('cpu', 'float64')
    torch.set_num_threads(2)
    original, allowed = load_original(cfg)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    dataset = datasets.CIFAR10(cfg["data_root"], train=False, download=False, transform=transform)
    x32, label = dataset[request["index"]]
    x32 = x32.unsqueeze(0).float()
    # Source-native probe and a separate explicitly named float64 snapshot.
    original.float()
    adapter32 = ClassSeparatedTop1.from_metamoe(original)
    with torch.no_grad():
        y32 = original(x32)[0]
        a32 = adapter32(x32)
    if not torch.equal(y32, a32) or not torch.isfinite(y32).all():
        raise ValueError("original forward contract")
    model64 = copy.deepcopy(original).double().eval()
    adapter = ClassSeparatedTop1.from_metamoe(model64)
    x = x32.double()
    surrogate = adapter.reduced_components(x)
    with torch.no_grad():
        y64, scores64 = model64(x)
        reduced = surrogate(x)
    # Only a finite concrete control: no domain equivalence inferred here.
    if not torch.equal(y64, reduced) or not torch.isfinite(y64).all():
        raise ValueError("nonzero branch concrete reduction mismatch")
    eps = cfg["normalized_epsilon"]
    lower, upper = x - eps, x + eps  # no raw [0,1] clipping in normalized space
    rows = classification_rows(adapter.total_classes, int(label))
    threshold = torch.full((len(rows),), cfg["classification_margin"], dtype=torch.float64, device='cpu')
    root = Path(cfg["output_root"]) / request["id"]
    initial = {"config_sha256": sha256(config_path), "request": request,
        "checkpoint_sha256": cfg["files"][cfg["checkpoint"]], "allowed_globals": allowed,
        "source_native_forward_equal": True, "reduced_float64_probe_equal": True,
        "dtype_probe_max_abs": float((y32.double() - y64).abs().max()),
        "global_label": int(label), "class_counts": list(adapter.class_counts),
        "source_prediction": int(y32.argmax(1)), "float64_prediction": int(y64.argmax(1)),
        "scores_float64": scores64.tolist(), "output_float64": y64.tolist(),
        "numerical_object": "explicit CPU float64 snapshot of public state; not source-complete float32 semantics",
        "input_space": "author CIFAR normalization; epsilon in normalized coordinates, not raw pixel L_inf",
        "control_seconds": time.monotonic() - started}
    write(root / "intake.json", initial)
    import hashlib
    tensors = {"center": x, "lower": lower, "upper": upper, "rows": rows, "thresholds": threshold}
    initial["materialized_tensor_hashes"] = {
        k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in tensors.items()}
    torch.save(tensors, root / "request.pt")
    result = verify_class_separated_box(adapter, **tensors,
        total_seconds=max(.001, cfg["request_seconds"] - (time.monotonic() - started)))
    result.update(initial)
    result["worker_total_seconds"] = time.monotonic() - started
    write(root / "result.json", result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--worker')
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    validate(cfg)
    if a.worker:
        run_worker(cfg, a.config, next(r for r in cfg['requests'] if r['id'] == a.worker))
        return
    root = Path(cfg['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    terminals = []
    for request in cfg['requests']:
        receipt = supervise([cfg['python'], str(Path(__file__).resolve()), '--config', str(a.config.resolve()),
             '--worker', request['id']], str(Path(__file__).resolve().parents[1]), root / request['id'],
             cfg['request_seconds'], 'AUTHOR_COMPONENT_CONTROL', cfg['repo'])
        result_path = root / request['id'] / 'result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        # Outer terminal wins even if a late/partial positive file exists.
        status = result['status'] if receipt['status'] == 'COMPLETED' and result else receipt['status']
        terminals.append({'request': request, 'status': status, 'receipt': str(root / request['id'] / 'receipt.json'),
                          'complete_result': bool(result), 'evidence_grade': result['evidence_grade']
                          if receipt['status'] == 'COMPLETED' and result else 'NONE'})
        write(root / f"{request['id']}_terminal.json", terminals[-1])
    write(root / 'summary.json', {'config_sha256': sha256(a.config), 'terminals': terminals,
        'scope': 'frozen bounded original-class intake; not paper replication or source-complete proof'})
    print(json.dumps(terminals, indent=2))


if __name__ == '__main__':
    main()
