"""Restricted loading of a pinned author expert; no dataset/accuracy/certificate."""
import argparse
import json
from pathlib import Path
import sys

from recent_moe_deployment import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    a = p.parse_args()
    checkpoint = a.checkpoint.resolve()
    if not checkpoint.is_relative_to(a.repo.resolve() / "paper/artifacts"):
        raise ValueError("only pinned author artifacts accepted")
    if sha256(checkpoint) != a.sha256:
        raise ValueError("checkpoint identity mismatch")
    sys.path.insert(0, str(a.repo.resolve() / "src/Vision_Transformer_Pytorch"))
    import torch
    from model_wrapper import ModelWrapper
    from small_expert import UltraVerifiableCNN
    torch.set_num_threads(2)
    torch.manual_seed(20260921)
    allowed = [ModelWrapper, UltraVerifiableCNN, torch.nn.BatchNorm2d,
               torch.nn.Dropout, torch.nn.Conv2d, torch.nn.AvgPool2d, torch.nn.Linear]
    allowed_names = {f"{c.__module__}.{c.__qualname__}" for c in allowed}
    globals_in_file = set(torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint))
    if globals_in_file - allowed_names:
        raise ValueError(f"unreviewed globals: {sorted(globals_in_file - allowed_names)}")
    with torch.serialization.safe_globals(allowed):
        model = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if type(model) is not ModelWrapper or type(model.model) is not UltraVerifiableCNN:
        raise ValueError("unexpected model topology/root type")
    if model.num_classes != 10 or model.model.fc2.out_features != 10:
        raise ValueError("expected ten-class expert")
    # No unpickling fallback, model surgery, or strict=False state loading.
    model.eval()
    if any(m.training for m in model.modules()):
        raise ValueError("eval mode did not propagate")
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("nonfinite state")
    x = torch.rand(2, 3, 32, 32, requires_grad=True)
    logits = model(x)[0]
    if tuple(logits.shape) != (2, 10) or not torch.isfinite(logits).all():
        raise ValueError("invalid output")
    logits.square().mean().backward()
    if x.grad is None or not torch.isfinite(x.grad).all():
        raise ValueError("invalid input gradient")
    if sha256(checkpoint) != a.sha256:
        raise ValueError("checkpoint changed during control")
    print(json.dumps({"grade": "AUTHOR_CHECKPOINT_SMOKE", "checkpoint_sha256": a.sha256,
                      "checkpoint": str(checkpoint), "weights_only": True,
                      "allowed_globals": sorted(allowed_names),
                      "globals_in_file": sorted(globals_in_file),
                      "python": sys.version, "torch": torch.__version__,
                      "device": "cpu", "dtype": "float32", "seed": 20260921,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "finite_forward": True, "finite_input_gradient": True,
                      "trained_checkpoint": True, "dataset_loaded": False,
                      "fresh_accuracy_or_certification": False,
                      "scope": "author ten-class expert only, random input control"}, indent=2))


if __name__ == "__main__":
    main()
