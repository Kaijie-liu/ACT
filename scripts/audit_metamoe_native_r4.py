"""Independent saved-record audit; no network inference or bound recomputation."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_spec(text, label):
    # Independent small S-expression parser for this fixed author VNNLIB grammar.
    text = re.sub(r";[^\n]*", "", text)
    stack, result = [], []
    for token in re.findall(r"[()]|[^\s()]+", text):
        if token == "(":
            stack.append([])
        elif token == ")":
            if not stack:
                raise ValueError("unbalanced specification")
            value = stack.pop()
            (stack[-1] if stack else result).append(value)
        else:
            if not stack:
                raise ValueError("tokens outside specification")
            stack[-1].append(token)
    if stack:
        raise ValueError("unclosed specification")
    declarations, lower, upper, properties = [], {}, {}, []
    for row in result:
        if len(row) == 3 and row[0] == "declare-const" and row[2] == "Real":
            declarations.append(row[1])
        elif len(row) == 2 and row[0] == "assert":
            expr = row[1]
            if len(expr) == 3 and expr[0] in [">=", "<="] and expr[1].startswith("X_"):
                target = lower if expr[0] == ">=" else upper
                if expr[1] in target:
                    raise ValueError("duplicate bound")
                target[expr[1]] = Fraction(expr[2])
            elif expr[0] == "or":
                properties.append(expr[1:])
            else:
                raise ValueError("unexpected assertion")
        else:
            raise ValueError("unexpected expression")
    if declarations != [f"X_{i}" for i in range(3072)] + [f"Y_{i}" for i in range(10)]:
        raise ValueError("declaration roster mismatch")
    if set(lower) != {f"X_{i}" for i in range(3072)} or set(lower) != set(upper):
        raise ValueError("incomplete box")
    if any(lower[k] > upper[k] for k in lower):
        raise ValueError("inverted box")
    expected = [["and", [">=", f"Y_{k}", f"Y_{label}"]] for k in range(10) if k != label]
    if properties != [expected]:
        raise ValueError("not the complete strict classification property")
    return lower, upper


def audit(config_path):
    cfg = json.loads(config_path.read_text())
    root = Path(cfg["output_root"])
    batch = json.loads((root / "batch_terminal.json").read_text())
    if batch["config_sha256"] != digest(config_path):
        raise ValueError("configuration mismatch")
    if [r["request"] for r in batch["requests"]] != cfg["requests"]:
        raise ValueError("terminal roster mismatch")
    results = []
    for terminal in batch["requests"]:
        item = terminal["request"]
        folder = root / item["id"]
        if terminal["status"] == "NOT_STARTED_AFTER_ERROR":
            if not results or results[-1]["status"] not in ["ERROR", "NOT_STARTED_AFTER_ERROR"]:
                raise ValueError("unexplained skipped request")
            results.append({"id": item["id"], "status": terminal["status"]})
            continue
        receipt = json.loads((folder / "receipt.json").read_text())
        if receipt != terminal["outer"] or receipt["deadline_seconds"] != cfg["outer_seconds"]:
            raise ValueError("outer receipt mismatch")
        for name in ["stdout", "stderr"]:
            if digest(folder / f"{name}.txt") != receipt[f"{name}_sha256"]:
                raise ValueError("outer log changed")
        row = {"id": item["id"], "status": terminal["status"],
               "outer_seconds": receipt["execution_including_preflight_seconds"],
               "postflight_total_seconds": receipt["total_with_postflight_seconds"],
               "receipt_sha256": digest(folder / "receipt.json")}
        work = folder / "work"
        if (work / "phases.jsonl").exists():
            phases = [json.loads(t) for t in (work / "phases.jsonl").read_text().splitlines()]
            values = [p["elapsed_seconds"] for p in phases]
            if values != sorted(values) or any(t < 0 for t in values):
                raise ValueError("invalid phase timing")
            row["phases"] = phases
        if receipt["status"] == "TIMEOUT":
            if terminal["status"] != "OUTER_TIMEOUT":
                raise ValueError("outer timeout upgraded")
        elif (work / "terminal.json").exists():
            worker = json.loads((work / "terminal.json").read_text())
            if worker != terminal.get("worker") or worker["request"] != item:
                raise ValueError("worker terminal mismatch")
            row["worker_seconds"] = worker["worker_wall_seconds"]
            if worker["status"] == "ERROR":
                if terminal["status"] != "ERROR":
                    raise ValueError("worker error upgraded")
                row["error"] = worker.get("error", worker.get("reason"))
            else:
                prepared = json.loads((work / "prepared.json").read_text())
                if prepared != worker["prepared"]:
                    raise ValueError("preparation changed")
                native = json.loads((work / "native_model_spec.json").read_text())
                if (native["checkpoint_sha256"] != item["checkpoint_sha256"] or
                        native["export"] != "NONE_NATIVE_PYTORCH" or
                        digest(native["loader"]) != native["loader_sha256"] or
                        digest(native["checkpoint"]) != native["checkpoint_sha256"] or
                        prepared["conformance_errors"] != [0., 0., 0.]):
                    raise ValueError("native source binding/conformance")
                onnx_files = [work / "native_model_spec.json"]
                for path, key in [(work / "input.npy", "input_sha256"),
                                  (work / "request.vnnlib", "vnnlib_sha256"),
                                  (onnx_files[0], "native_model_spec_sha256"), (work / "backend.yaml", "config_sha256")]:
                    if digest(path) != prepared[key]:
                        raise ValueError(f"prepared file changed: {path}")
                check_spec((work / "request.vnnlib").read_text(), prepared["label"])
                text = (work / "backend.stdout").read_text()
                pairs = re.findall(r"^Result: (.+) in ([0-9.]+) seconds$", text, re.M)
                if len(pairs) != 1 or worker["backend_exit_code"] != 0:
                    raise ValueError("missing native completion")
                raw = pairs[0][0]
                if raw != worker["backend_status"]:
                    raise ValueError("backend status changed")
                if terminal["status"] == "BACKEND_POSITIVE":
                    if raw not in {"safe", "safe-incomplete", "safe-complete"}:
                        raise ValueError("invalid positive")
                    if worker["backend_wall_seconds"] > cfg["solver_seconds"] or receipt["status"] != "COMPLETED":
                        raise ValueError("late positive")
                row.update(prepared=prepared, backend_status=raw,
                           backend_wall_seconds=worker["backend_wall_seconds"],
                           backend_stdout_sha256=digest(work / "backend.stdout"),
                           backend_stderr_sha256=digest(work / "backend.stderr"))
        elif terminal["status"] != "ERROR":
            raise ValueError("missing worker terminal without error/timeout")
        results.append(row)
    return {"audit": "PASS", "config_sha256": digest(config_path), "requests": results,
            "scope": "saved identities, complete property grammar, native status and cost accounting",
            "independent_bound_or_network_reproof": False,
            "full_dynamic_moe_verification": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    value = audit(a.config)
    with a.output.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"audit": value["audit"], "states": [r["status"] for r in value["requests"]]}))


if __name__ == "__main__":
    main()

