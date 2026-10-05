"""Once-only metadata diagnostic. No candidate, tensor arithmetic or solver."""

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import tracemalloc

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
ROOT = EXP.parent.parent
RUN = EXP / "results/d179_preterminal_domain_20261004_v1"
PRIOR = EXP / "results/d158_joint_forward_support_20261004_v1/preregistered.json"
PRIOR_SHA = "64cd535e4ae10bfaeb8204d21841f991781aa981e2aeddb359a87ca8050103d9"
SCHEMA = "d179_preterminal_domain_v1"
FILES = ("PREREG.md", "DEFINITION_TEST.md", "inputs.json", "bind_structure.py", "run_structure.py", "launch_structure.py")
PYTHON = Path("/data1/Kane/miniconda3/bin/python")
COMMIT = "f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac"
DIFF_SHA = "29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5"
AS_CAP, MEMORY_CAP, RESERVE = 16 * 1024**3, 1024**3, 65536
WORK_CAP, BRANCH_CAP, EVIDENCE_CAP, ENTRY_CAP = 256_000_000, 200_000_000, 40_000_000, 64_000_000
MODEL_CAP, JSON_CAP = 64 * 1024**2, 8 * 1024**2
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Deadline(RuntimeError):
    pass


def stop(signum, frame):
    raise Deadline("registered deadline or termination: " + str(signum))


class Meter:
    def __init__(self):
        self.started = time.monotonic()
        self.rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        self.branch = self.entries = self.hash_files = self.hash_bytes = 0
        self.evidence = RESERVE

    def charge(self, amount, entries=0, evidence=False):
        require(type(amount) is int and amount >= 0 and type(entries) is int and entries >= 0,
                "invalid diagnostic charge")
        require(self.entries + entries <= ENTRY_CAP, "metadata entry cap")
        if evidence:
            require(self.evidence + amount <= EVIDENCE_CAP, "evidence work cap")
            self.evidence += amount
        else:
            require(self.branch + amount <= BRANCH_CAP
                    and EVIDENCE_CAP + self.branch + amount <= WORK_CAP, "branch/whole work cap")
            self.branch += amount
        self.entries += entries

    def snapshot(self):
        current, peak = tracemalloc.get_traced_memory()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        return dict(wall_s=time.monotonic() - self.started, branch_work=self.branch,
                    whole_work=EVIDENCE_CAP + self.branch, evidence_work=self.evidence,
                    evidence_prepaid_work=EVIDENCE_CAP, retained_metadata_entries=self.entries,
                    rss_initial_high_water_bytes=self.rss0, rss_high_water_bytes=rss,
                    rss_growth_bytes=max(0, rss - self.rss0), traced_current_bytes=current,
                    traced_peak_bytes=peak, tracer_metadata_bytes=tracemalloc.get_tracemalloc_memory(),
                    reserve_bytes=RESERVE, hashed_files=self.hash_files, hashed_bytes=self.hash_bytes,
                    complete_physical_qualification=False)

    def check(self):
        value = self.snapshot()
        require(value["rss_growth_bytes"] + RESERVE <= MEMORY_CAP
                and value["traced_peak_bytes"] + value["tracer_metadata_bytes"] + RESERVE <= MEMORY_CAP,
                "host memory observation cap")
        if value["wall_s"] >= 58:
            raise Deadline("58-second diagnostic/postcheck deadline")
        return value


def checked(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), "ordinary absolute file required")
    return path


def digest(path, meter=None):
    value = hashlib.sha256()
    with checked(path).open("rb") as stream:
        while True:
            if meter:
                meter.charge(16)
            chunk = stream.read(65536)
            if not chunk:
                break
            value.update(chunk)
            if meter:
                meter.hash_bytes += len(chunk)
                meter.check()
    if meter:
        meter.hash_files += 1
    return value.hexdigest()


def read_bytes(path, expected, cap, meter):
    path = checked(path)
    size = path.stat().st_size
    require(0 < size <= cap, "file size cap: " + str(path))
    meter.charge(4096 + size)
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    require(len(raw) == size and hashlib.sha256(raw).hexdigest() == expected,
            "authenticated byte mismatch: " + str(path))
    meter.hash_files += 1
    meter.hash_bytes += size
    meter.check()
    return raw


def json_read(path, expected, meter):
    raw = read_bytes(path, expected, JSON_CAP, meter)
    meter.charge(4 * len(raw), entries=len(raw))
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    def invalid(value):
        raise ValueError("nonstandard JSON number: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def identities(value, count):
    require(type(value) is dict and len(value) == count, "identity population differs")
    for path, sha in value.items():
        require(type(path) is str and Path(path).is_absolute() and type(sha) is str
                and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), "malformed identity")
    return value


def verify_all(mapping, meter):
    for path, sha in mapping.items():
        require(digest(path, meter) == sha, "identity drift: " + path)


def provenance(meter):
    def git(*args):
        meter.check()
        return subprocess.check_output(["git", *args], cwd=ROOT, timeout=3)
    value = dict(branch=git("branch", "--show-current").decode().strip(),
                 commit=git("rev-parse", "HEAD").decode().strip(),
                 tracked_diff_sha256=hashlib.sha256(git("diff", "--binary", "HEAD", "--")).hexdigest())
    require(value == dict(branch="redu-hz", commit=COMMIT, tracked_diff_sha256=DIFF_SHA), "production drift")
    return value


def serialization_bound(value, meter):
    meter.charge(12, evidence=True)
    if value is None or type(value) is bool:
        return 5
    if type(value) is int:
        return 3 + (abs(value).bit_length() + 2) // 3
    if type(value) is float:
        require(math.isfinite(value), "nonfinite JSON evidence")
        return 32
    if type(value) is str:
        return 2 + 12 * len(value)
    if type(value) in (list, tuple):
        return 2 + len(value) + sum(serialization_bound(x, meter) for x in value)
    require(type(value) is dict and all(type(k) is str for k in value), "unsupported evidence type")
    return 2 + 2 * len(value) + sum(serialization_bound(k, meter) + serialization_bound(v, meter)
                                   for k, v in value.items())


def save(name, value, meter=None):
    if meter:
        bound = serialization_bound(value, meter)
        meter.charge(1024 + 2 * bound, evidence=True)
    else:
        bound = RESERVE // 2
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    require(len(raw) <= bound, "evidence serialization bound")
    with (RUN / name).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return digest(RUN / name, meter)


def main():
    require(sys.argv[1:] == ["--enabled"], "explicit --enabled required")
    RUN.mkdir(exist_ok=False)
    meter = Meter()
    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.alarm(58)
    status = 1
    result = dict(schema=SCHEMA, diagnostic_complete=False, models=[], formal_gain=0,
                  source_precheck_complete=False, input_precheck_complete=False,
                  source_postcheck_complete=False, input_postcheck_complete=False,
                  candidate_executed=False, model_forward_executed=False, solver_executed=False,
                  mathematical_component_gate_passed=False, source_component_qualified=False,
                  actual_model_binding_qualified=False, native_HZ_admitted=False,
                  gpu_computation_completed=False, complete_physical_qualification=False)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))
        sys.dont_write_bytecode = True
        os.environ.update({name: "1" for name in THREADS})
        os.environ.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
                          CUDA_CACHE_PATH=str(RUN / "cuda_cache"), XDG_CACHE_HOME=str(RUN / "xdg_cache"),
                          TMPDIR=str(RUN / "tmp"))
        (RUN / "tmp").mkdir()
        tracemalloc.start()
        require(__debug__ and os.environ.get("PYTHONOPTIMIZE") in (None, "", "0"), "optimized run forbidden")
        prior = json_read(PRIOR, PRIOR_SHA, meter)
        require(prior.get("schema") == "d158_joint_forward_support_v1"
                and prior.get("cpu_affinity") == [0], "inherited registration differs")
        sources = identities(prior.get("source_sha256"), 7327)
        inputs = identities(prior.get("input_sha256"), 14)
        os.sched_setaffinity(0, {0})
        require(sorted(os.sched_getaffinity(0)) == [0], "CPU binding")
        freeze_sha = digest(HERE / "freeze.json", meter)
        freeze = json_read(HERE / "freeze.json", freeze_sha, meter)
        require(set(freeze) == {"schema", "source_sha256"} and freeze["schema"] == SCHEMA, "freeze schema")
        fresh = identities(freeze["source_sha256"], len(FILES))
        require(set(fresh) == {str(HERE / name) for name in FILES}, "freeze file population")
        verify_all(fresh, meter)
        verify_all(sources, meter)
        verify_all(inputs, meter)
        require(Path(sys.executable).resolve() == PYTHON.resolve()
                and str(PYTHON.resolve()) in sources, "authenticated interpreter required")
        selected = json_read(HERE / "inputs.json", fresh[str(HERE / "inputs.json")], meter)
        require(set(selected) == {"schema", "models"} and selected["schema"] == SCHEMA
                and type(selected["models"]) is list and len(selected["models"]) == 3, "three-model population")
        population = selected["models"]
        require(len({x["model_path"] for x in population}) == 3, "duplicate model")
        graph_ids = {}
        for item in population:
            require(set(item) == {"model_path", "model_sha256", "graph_path", "graph_sha256"}, "source descriptor")
            require(inputs.get(item["model_path"]) == item["model_sha256"], "model not inherited input")
            graph_ids[item["graph_path"]] = item["graph_sha256"]
        identities(graph_ids, 3)
        verify_all(graph_ids, meter)
        result.update(source_precheck_complete=True, input_precheck_complete=True,
                      provenance_before=provenance(meter), prior_manifest_sha256=PRIOR_SHA,
                      freeze_sha256=freeze_sha, source_count=len(sources), input_count=len(inputs),
                      cpu_affinity=[0], thread_environment={name: os.environ[name] for name in THREADS},
                      cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                      bytecode_disabled=sys.dont_write_bytecode)
        spec = importlib.util.find_spec("onnx")
        require(spec is not None and spec.origin and str(Path(spec.origin).resolve()) in sources, "ONNX provenance")
        import onnx
        require(str(Path(onnx.__file__).resolve()) in sources, "ONNX module identity")
        module_spec = importlib.util.spec_from_file_location("_d179_binding", HERE / "bind_structure.py")
        binder = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(binder)
        result["dependency_versions"] = {name: importlib.metadata.version(name) for name in ("onnx", "protobuf", "numpy")}
        result["python_version"] = sys.version
        for index, item in enumerate(population):
            record = dict(index=index, source=item, structure_complete=False)
            result["models"].append(record)
            try:
                raw = read_bytes(item["model_path"], item["model_sha256"], MODEL_CAP, meter)
                inventory = json_read(item["graph_path"], item["graph_sha256"], meter)
                require(inventory.get("source", {}).get("source", {}).get("model_sha256") == item["model_sha256"],
                        "saved graph/model identity mismatch")
                report = binder.inspect_model(onnx, raw, inventory, meter)
                require(report.get("structure_complete") is True, "incomplete structure binding")
                report.update(schema=SCHEMA, source=item, coefficient_values_read=False,
                              coefficient_validity_verified=False, actual_model_verification_qualified=False,
                              candidate_executed=False, formal_gain=0)
                name = "model_" + str(index) + ".json"
                sha = save(name, report, meter)
                record.update(structure_complete=True, artifact=name, sha256=sha,
                              match_count=len(report["matches"]), relu_count=len(report["relu_population"]))
                del report, inventory, raw
                meter.check()
            except (Deadline, MemoryError):
                raise
            except Exception as error:
                record["failure"] = type(error).__name__ + ": " + str(error)
                record["artifact"] = "model_" + str(index) + "_failure.json"
                record["sha256"] = save(record["artifact"], record, meter)
        verify_all(sources, meter)
        verify_all(inputs, meter)
        verify_all(graph_ids, meter)
        verify_all(fresh, meter)
        require(digest(PRIOR, meter) == PRIOR_SHA and digest(HERE / "freeze.json", meter) == freeze_sha,
                "registration drift")
        result.update(source_postcheck_complete=True, input_postcheck_complete=True,
                      provenance_after=provenance(meter))
        require(sorted(os.sched_getaffinity(0)) == [0], "postcheck CPU affinity drift")
        meter.check()
        require(len(result["models"]) == 3 and all(x["structure_complete"] for x in result["models"]),
                "incomplete three-model diagnostic")
        result["diagnostic_complete"] = True
        status = 0
    except BaseException as error:
        result["failure"] = type(error).__name__ + ": " + str(error)
    finally:
        signal.alarm(0)
        if not (result["source_postcheck_complete"] and result["input_postcheck_complete"]):
            result["postcheck_reason"] = "not completed after failure or deadline; evidence is unqualified"
        observations = meter.snapshot()
        result["host_observations"] = observations
        if (observations["rss_growth_bytes"] + RESERVE > MEMORY_CAP
                or observations["traced_peak_bytes"] + observations["tracer_metadata_bytes"] + RESERVE > MEMORY_CAP
                or observations["wall_s"] >= 58):
            result["diagnostic_complete"] = False
            result.setdefault("failure", "final time or memory gate")
            status = 1
        result["exit_status"] = status
        save("result.json", result)
        hashes = {path.name: digest(path) for path in sorted(RUN.glob("*.json"))}
        save("exit.json", dict(schema=SCHEMA, status=status, artifact_sha256=hashes,
                               diagnostic_complete=result["diagnostic_complete"], formal_gain=0,
                               native_HZ_admitted=False, candidate_executed=False))
        print(json.dumps(dict(status=status, diagnostic_complete=result["diagnostic_complete"],
                              result_path=str(RUN / "result.json"), failure=result.get("failure"))))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
