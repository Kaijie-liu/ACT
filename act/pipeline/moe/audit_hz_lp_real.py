"""Recheck the saved HZ export and its bound without importing the exporter."""
import argparse
import hashlib
import json
from pathlib import Path

from act.back_end.solver.check_hz_lp_export import check_export


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root, config_path):
    config = json.loads(config_path.read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    for path, expected in ((config_path, manifest["config_sha256"]),
                           (root / "request.pt", manifest["request_sha256"]),
                           (root / "export.json", manifest["export_sha256"]),
                           (Path(config["selection"]), config["selection_sha256"])):
        if sha(path) != expected:
            raise ValueError("artifact hash mismatch")
    selection = json.loads(Path(config["selection"]).read_text())
    subject = selection["models"][config["hz_export_model"]]
    if sha(Path(subject["checkpoint"])) != subject["checkpoint_sha256"] or manifest["checkpoint_sha256"] != subject["checkpoint_sha256"]:
        raise ValueError("checkpoint mismatch")
    if manifest["dataset_index"] != selection["samples"][config["hz_export_rank"]]["dataset_index"]:
        raise ValueError("registered input mismatch")
    record = json.loads((root / "export.json").read_text())
    certificate = json.loads((root / "certificate.json").read_text())
    pair = manifest["pair"]
    q = [0] * len(record["source"]["c"])
    q[min(pair)] = 1
    q[next(i for i in range(len(q)) if i not in pair)] = -1
    if record["q"] != q or manifest["q"] != q or record["offset"] != 0:
        raise ValueError("registered query mismatch")
    checked = check_export(record, certificate, expected_source_sha256=manifest["hz_sha256"])
    if checked != json.loads((root / "check.json").read_text()):
        raise ValueError("saved result differs from independent check")
    return {"status": "PASS", "issues": [], "checked": checked,
            "manifest_sha256": sha(root / "manifest.json"),
            "certificate_sha256": sha(root / "certificate.json"),
            "export_sha256": manifest["export_sha256"], "raw_root": str(root),
            "scope": "Artifact identity, given-HZ lowering and rational LP bound; upstream clean-route/propagation correctness remains trusted."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.root, args.config)
    if args.output:
        from act.pipeline.moe.experiment1 import _inside, WRITE_ROOT
        from act.pipeline.moe.paired_followup import save
        output = _inside(args.output, WRITE_ROOT)
        if output.exists():
            raise RuntimeError("refusing audit overwrite")
        save(output, result)
    print(json.dumps(result, indent=2))
