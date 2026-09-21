"""Archive compact provenance and actual deployment outcomes; never invent success."""
import argparse
import hashlib
import json
from pathlib import Path

from recent_moe_deployment import git_identity, sha256


PAPERS = {
    "dual_rs": "https://arxiv.org/pdf/2512.01782v3",
    "metamoe": "https://www.verivital.com/research/pham2026saiv.pdf",
    "rome": "https://arxiv.org/pdf/2607.06109v1",
    "j_tlat": "https://arxiv.org/pdf/2602.01369",
    "feature_noise": "https://arxiv.org/pdf/2601.14792v1",
    "robust_experts": "https://openaccess.thecvf.com/content/ICCV2025W/STREAM/papers/Pavlitska_Robust_Experts_The_Effect_of_Adversarial_Training_on_CNNs_with_ICCVW_2025_paper.pdf",
}


def build(base, runs):
    papers, repos, attempts = {}, {}, {}
    for name, url in PAPERS.items():
        pdf, txt = runs / "papers" / f"{name}.pdf", runs / "papers" / f"{name}.txt"
        papers[name] = {"url": url, "pdf_sha256": sha256(pdf), "pdf_bytes": pdf.stat().st_size,
                        "text_sha256": sha256(txt), "local_pdf": str(pdf)}
        repo = base / name
        if not repo.exists():
            repos[name] = {"status": "AUTHOR_REPOSITORY_NOT_IDENTIFIED_IN_SEARCH",
                           "not_a_claim_of_global_absence": True}
            continue
        identity = git_identity(repo)
        if identity["status"]:
            raise ValueError(f"dirty author checkout: {name}")
        hashes = identity.pop("tracked_sha256")
        identity.update({"tracked_files": len(hashes),
                         "python_files": sum(k.endswith(".py") for k in hashes),
                         "tracked_manifest_sha256": hashlib.sha256(
                             json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                         "license_files": [k for k in hashes if "license" in k.lower()],
                         "weight_files": {k: v for k, v in hashes.items()
                                          if k.endswith((".pth", ".pt", ".onnx"))},
                         "readme_sha256": hashes.get("README.md")})
        repos[name] = identity
    for path in sorted(runs.glob("*/receipt.json")):
        receipt = json.loads(path.read_text())
        for stream in ("stdout", "stderr"):
            if sha256(path.parent / f"{stream}.txt") != receipt[f"{stream}_sha256"]:
                raise ValueError(f"changed {stream}: {path}")
        source = receipt.pop("source_before")
        receipt["source_head"] = source["head"] if source else None
        receipt["receipt_sha256"] = sha256(path)
        receipt["receipt_path"] = str(path)
        receipt["stdout"] = (path.parent / "stdout.txt").read_text()
        receipt["stderr"] = (path.parent / "stderr.txt").read_text()
        attempts[path.parent.name] = receipt
    return {"schema": 1, "date": "2026-09-21", "papers": papers, "repositories": repos,
            "attempts": attempts, "new_training_runs": 0, "fresh_certification_runs": 0,
            "dependencies_installed": False, "same_task_comparison_completed": False,
            "scope": "literature, source deployment, result recount and architecture/CLI probes only"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    value = build(a.base, a.runs)
    with a.output.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"papers": len(value["papers"]), "attempts": len(value["attempts"]),
                      "completed": sum(r["status"] == "COMPLETED" for r in value["attempts"].values()),
                      "all_experiments_reproduced": False}))


if __name__ == "__main__":
    main()
