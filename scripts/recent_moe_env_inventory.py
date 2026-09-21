"""Record/compare existing environment packages without changing them."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def inventory(executables):
    probe = ("import importlib.metadata as m,json,sys; "
             "print(json.dumps({'version':sys.version,'prefix':sys.prefix,"
             "'packages':sorted([(d.metadata['Name'],d.version) for d in m.distributions()])}))")
    return {path: {"binary_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                   "environment": json.loads(subprocess.check_output([path, "-c", probe], text=True))}
            for path in executables}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument("--check", type=Path)
    p.add_argument("executables", nargs="+")
    a = p.parse_args()
    value = inventory(a.executables)
    if a.check:
        if value != json.loads(a.check.read_text()):
            raise ValueError("existing environment inventory changed")
        print("PASS: existing executable/package inventories unchanged")
    elif a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        print(json.dumps({"recorded": len(value), "output": str(a.output)}))
    else:
        print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
