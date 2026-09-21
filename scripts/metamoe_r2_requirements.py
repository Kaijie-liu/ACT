"""Pin R2 to R1 packages; the only source change is upstream AvgPool support."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    environments = json.loads(a.inventory.read_text())
    if len(environments) != 1:
        raise ValueError("expected one frozen environment")
    packages = next(iter(environments.values()))["environment"]["packages"]
    lines = [f"{name}=={version}" for name, version in packages
             if name.lower().replace("_", "-") not in ["auto-lirpa", "onnx2pytorch"]]
    with a.output.open("x") as stream:
        stream.write("# R1 exact package versions; VCS packages installed separately.\n")
        stream.write("\n".join(lines) + "\n")
    print(json.dumps({"pinned_packages": len(lines)}))


if __name__ == "__main__":
    main()
