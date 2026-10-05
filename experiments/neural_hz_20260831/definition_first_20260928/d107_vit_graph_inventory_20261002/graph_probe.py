"""Default-off, one-shot graph metadata audit; no model evaluation or HZ code."""

import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import struct
import subprocess
import sys
import time
import traceback
import tracemalloc


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUN = ROOT / "experiments/neural_hz_20260831/results/d107_vit_graph_inventory_20261002_v1"
BENCHMARK = Path("/data1/Kane/data/vnncomp2025_benchmarks/benchmarks")
MAX_RAW = 64 * 1024 * 1024
MAX_JSON = 4 * 1024 * 1024
MEMORY_GATE = 1024 ** 3
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def bounded_read(path, cap):
    with Path(path).open("rb") as stream:
        data = stream.read(cap + 1)
    require(len(data) <= cap, "read exceeds cap: " + str(path))
    return data


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(name, value):
    require(Path(name).name == name, "output must be a leaf filename")
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    require(len(data) <= MAX_JSON, "JSON evidence exceeds cap")
    with (RUN / name).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def resource_setup():
    os.sched_setaffinity(0, {0})
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024 ** 3, 16 * 1024 ** 3))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for key in THREAD_KEYS:
        os.environ[key] = "1"
    tracemalloc.start()


def memory_report():
    current, peak = tracemalloc.get_traced_memory()
    tracer = tracemalloc.get_tracemalloc_memory()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return {"rss_peak_bytes": rss, "traced_current_bytes": current,
            "traced_peak_bytes": peak, "tracer_metadata_bytes": tracer,
            "within_observed_gate": rss + tracer <= MEMORY_GATE and peak + tracer <= MEMORY_GATE}


def authenticate(expected_freeze_sha256):
    raw_freeze = bounded_read(HERE / "freeze.json", MAX_JSON)
    require(hashlib.sha256(raw_freeze).hexdigest() == expected_freeze_sha256, "freeze identity drift")
    frozen = json.loads(raw_freeze)
    require(frozen["schema"] == "d107_graph_metadata_v1", "wrong freeze schema")
    for item in frozen["files"]:
        require(digest(item["path"]) == item["sha256"], "frozen source drift: " + item["path"])
    lines = bounded_read(HERE / "DECODER_SHA256SUMS", MAX_JSON).decode().splitlines()
    for line in lines:
        expected, path = line.split("  ", 1)
        require(digest(path) == expected, "decoder dependency drift: " + path)
    for item in frozen["models"]:
        path = BENCHMARK / item["relative_path"]
        require(path.stat().st_size == item["size_bytes"], "model size drift")
        require(digest(path) == item["sha256"], "model identity drift")
    require(subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT).decode().strip() == frozen["branch"], "branch drift")
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip() == frozen["commit"], "HEAD drift")
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    require(hashlib.sha256(diff).hexdigest() == frozen["tracked_binary_diff_sha256"], "tracked worktree drift")
    require(not subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT), "staged changes appeared")
    return frozen


def tensor_record(tensor):
    require(not tensor.external_data and int(tensor.data_location) == 0, "external tensor forbidden")
    dims = [int(x) for x in tensor.dims]
    require(all(x >= 0 for x in dims), "negative tensor dimension")
    count = 1
    for dim in dims:
        count *= dim
    record = {"name": tensor.name, "dims": dims, "data_type": int(tensor.data_type),
              "raw_bytes": len(tensor.raw_data), "elements": count,
              "protobuf_sha256": hashlib.sha256(tensor.SerializeToString()).hexdigest()}
    if tensor.data_type in (6, 7) and count <= 64:
        if tensor.raw_data:
            code = "i" if tensor.data_type == 6 else "q"
            require(len(tensor.raw_data) == count * struct.calcsize(code), "integer layout payload mismatch")
            values = list(struct.unpack("<" + code * count, tensor.raw_data))
        else:
            values = list(tensor.int32_data if tensor.data_type == 6 else tensor.int64_data)
            require(len(values) == count, "integer layout data count mismatch")
        record["small_integer_values"] = values
    return record


def attribute_record(attr):
    record = {"name": attr.name, "type": int(attr.type)}
    if attr.type == 1:
        record["float_hex"] = float(attr.f).hex()
    elif attr.type == 2:
        record["integer"] = int(attr.i)
    elif attr.type == 3:
        record["string_hex"] = bytes(attr.s).hex()
    elif attr.type == 4:
        record["tensor"] = tensor_record(attr.t)
    elif attr.type == 6:
        record["float_hex_list"] = [float(x).hex() for x in attr.floats]
    elif attr.type == 7:
        record["integers"] = list(attr.ints)
    elif attr.type == 8:
        record["strings_hex"] = [bytes(x).hex() for x in attr.strings]
    elif attr.type == 9:
        record["tensors"] = [tensor_record(t) for t in attr.tensors]
    else:
        raise ValueError("unsupported or graph attribute: " + attr.name)
    return record


def value_record(value):
    require(value.type.HasField("tensor_type"), "non-tensor value annotation")
    tensor = value.type.tensor_type
    dims = None
    if tensor.HasField("shape"):
        dims = []
        for dim in tensor.shape.dim:
            kind = dim.WhichOneof("value")
            dims.append({"value": int(dim.dim_value)} if kind == "dim_value" else
                        {"symbol": dim.dim_param} if kind == "dim_param" else {"unknown": True})
    return {"name": value.name, "elem_type": int(tensor.elem_type), "annotated_dims": dims}


def extract(onnx, item):
    path = BENCHMARK / item["relative_path"]
    raw = bounded_read(path, MAX_RAW)
    require(len(raw) == item["size_bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"], "read bytes do not match frozen model")
    model = onnx.ModelProto()
    model.ParseFromString(raw)
    require(not model.functions, "local functions outside graph-only scope")
    require(not model.training_info, "training subgraphs outside graph-only scope")
    graph = model.graph
    require(len(graph.node) <= 10000 and len(graph.initializer) <= 10000, "graph count cap")
    require(not graph.sparse_initializer, "sparse initializers outside scope")
    nodes, producers, consumers, counts = [], {}, {}, {}
    for index, node in enumerate(graph.node):
        require(node.domain in ("", "ai.onnx"), "custom operator domain outside scope")
        attrs = [attribute_record(a) for a in node.attribute]
        require(len({a["name"] for a in attrs}) == len(attrs), "duplicate attributes")
        rec = {"index": index, "name": node.name, "op": node.op_type, "domain": node.domain,
               "inputs": list(node.input), "outputs": list(node.output), "attributes": attrs}
        nodes.append(rec)
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
        for name in node.output:
            if name:
                require(name not in producers, "duplicate graph producer")
                producers[name] = index
        for port, name in enumerate(node.input):
            if name:
                consumers.setdefault(name, []).append({"node": index, "port": port})
    tensors = [tensor_record(t) for t in graph.initializer]
    require(len({t["name"] for t in tensors}) == len(tensors), "duplicate initializer name")
    annotated = {"inputs": [value_record(v) for v in graph.input],
                 "outputs": [value_record(v) for v in graph.output],
                 "value_info": [value_record(v) for v in graph.value_info]}
    return {"schema": "d107_graph_inventory_v1", "source": item, "ir_version": int(model.ir_version),
            "opsets": [{"domain": o.domain, "version": int(o.version)} for o in model.opset_import],
            "graph_name": graph.name, "nodes": nodes, "operator_counts": counts,
            "initializers": tensors, "annotations": annotated,
            "producers": producers, "consumers": consumers,
            "softmax_nodes": [n["index"] for n in nodes if n["op"] == "Softmax"],
            "matmul_nodes": [n["index"] for n in nodes if n["op"] == "MatMul"],
            "shape_inference_executed": False, "network_executed": False,
            "native_hz_source_binding_verified": False}


def worker(expected_freeze_sha256):
    resource_setup()
    registered = json.loads(bounded_read(RUN / "preregistered.json", MAX_JSON))
    require(registered["freeze_sha256"] == expected_freeze_sha256, "worker freeze differs from registered run")
    require(registered["pid"] == os.getppid(), "worker must be started by its registered supervisor")
    frozen = authenticate(expected_freeze_sha256)
    import onnx
    import google.protobuf
    import numpy
    write_json("decoder_versions.json", {"python": sys.version, "onnx": onnx.__version__,
               "protobuf": google.protobuf.__version__, "numpy": numpy.__version__,
               "onnx_module": onnx.__file__, "cpu_affinity": sorted(os.sched_getaffinity(0))})
    summaries = []
    for item in frozen["models"]:
        inventory = extract(onnx, item)
        require(memory_report()["within_observed_gate"], "observed worker memory gate")
        name = Path(item["relative_path"]).stem + "_graph.json"
        write_json(name, inventory)
        summaries.append({"model": item["relative_path"], "nodes": len(inventory["nodes"]),
                          "softmax_nodes": inventory["softmax_nodes"], "matmul_nodes": inventory["matmul_nodes"],
                          "evidence": name})
        del inventory
    authenticate(expected_freeze_sha256)
    report = memory_report()
    require(report["within_observed_gate"], "final worker memory gate")
    write_json("worker_receipt.json", {"graph_metadata_complete": True, "models": summaries,
               "memory": report, "candidate_qualified": False, "formal_gain": 0})
    print(json.dumps({"graph_metadata_complete": True, "model_count": len(summaries)}), flush=True)


def supervisor():
    RUN.mkdir(parents=False, exist_ok=False)
    started = time.monotonic()
    result = {"schema": "d107_diagnostic_exit_v1", "complete": False, "formal_gain": 0,
              "new_mathematical_qualification": False, "new_network_execution": False}
    try:
        resource_setup()
        freeze_sha256 = digest(HERE / "freeze.json")
        frozen = authenticate(freeze_sha256)
        write_json("preregistered.json", {"freeze": frozen, "freeze_sha256": freeze_sha256,
                   "started_unix": time.time(), "pid": os.getpid(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
                   "worker_timeout_seconds": 60, "read_only_model_diagnostic": True})
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", freeze_sha256]
        with (RUN / "worker.log").open("xb") as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            result["worker_pid"] = proc.pid
            worker_start = time.monotonic()
            try:
                result["worker_returncode"] = proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                result["worker_returncode"] = proc.wait()
                result["timed_out"] = True
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                result["worker_returncode"] = proc.wait()
                raise
            finally:
                result["worker_wall_seconds"] = time.monotonic() - worker_start
        require(result["worker_returncode"] == 0, "worker failed or timed out")
        authenticate(freeze_sha256)
        receipt = json.loads(bounded_read(RUN / "worker_receipt.json", MAX_JSON))
        require(receipt["graph_metadata_complete"] and len(receipt["models"]) == len(frozen["models"]), "incomplete population")
        require(receipt["memory"]["within_observed_gate"], "worker memory receipt failed")
        require(memory_report()["within_observed_gate"], "supervisor memory gate")
        result["complete"] = True
    except BaseException as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        result["supervisor_wall_seconds"] = time.monotonic() - started
        result["supervisor_memory"] = memory_report()
        result["evidence_sha256"] = {p.name: digest(p) for p in sorted(RUN.iterdir()) if p.is_file()}
        write_json("exit.json", result)
    print(json.dumps({"complete": result["complete"], "run": str(RUN), "formal_gain": 0}), flush=True)
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--enabled"]:
        raise SystemExit(supervisor())
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        require(RUN.is_dir() and (RUN / "preregistered.json").is_file(), "worker requires registered run")
        worker(sys.argv[2])
    elif not sys.argv[1:]:
        print("disabled: graph diagnostic requires --enabled")
    else:
        raise SystemExit("unsupported arguments")
