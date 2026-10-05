"""Single-use, metadata-only audit. No tensors are converted to numeric arrays.

The inherited source population is authenticated before importing ONNX. Model
bytes are parsed once, without external data, and all three registered models
are reported. A dimension bound is NOT a measured rank or a native shape bind.
"""

from collections import Counter
import hashlib
import importlib.util
import io
import json
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
RUN = EXP / "results/d152_live_amplitude_metadata_20261004_v1"
PRIOR = EXP / "results/d150_stable_phase_charts_20261004_v1/preregistered.json"
OLD = EXP / "results/d120_mixed_consumer_source_20261002_v1"
FREEZE = HERE / "freeze.json"
SCHEMA = "d152_live_amplitude_metadata_v1"
PRIOR_SHA = "7f13f7860a3490aa757a0fef853147d35620c2fa599a10a6336404df737a2813"
COMMIT = "f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac"
DIFF_SHA = "29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5"
PYTHON = Path("/data1/Kane/miniconda3/bin/python")
SOURCE_JSON_SHA = (
    "8bb8a5075e5575df40bda7ab629f9d2c0f904b73a5aac04b96fc8b39bf2b0f9d",
    "fb236df05faafd5f2403326fb5306a1d2cac5d563b7f6d011286e374e4143007",
    "a891450794e736132c5e357cec1d8fc2071fa4e322d72de3811cce7f3622b6b4",
)
AS_CAP, MEMORY_CAP, RESERVE = 16 * 1024**3, 1024**3, 65536
MODEL_CAP, JSON_CAP, NODE_CAP = 64 * 1024**2, 64 * 1024**2, 100000
FILES = ("PREREG.md", "audit_metadata.py")
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


class AuditFailure(Exception):
    pass


class Deadline(AuditFailure):
    pass


def stop(signum, _frame):
    raise Deadline("registered stage deadline or termination signal: " + str(signum))


class Meter:
    def __init__(self):
        self.started = time.monotonic()
        self.rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        self.hash_files = 0
        self.hash_bytes = 0

    def snapshot(self):
        current, peak = tracemalloc.get_traced_memory()
        metadata = tracemalloc.get_tracemalloc_memory()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        return dict(wall_s=time.monotonic() - self.started,
                    rss_high_water_bytes=rss, rss_initial_high_water_bytes=self.rss0,
                    rss_high_water_growth_bytes=max(0, rss - self.rss0),
                    traced_current_bytes=current, traced_peak_bytes=peak,
                    tracer_metadata_bytes=metadata, reserve_bytes=RESERVE,
                    authenticated_file_reads=self.hash_files,
                    authenticated_bytes_read=self.hash_bytes,
                    complete_physical_qualification=False)

    def check(self):
        view = self.snapshot()
        if (view["rss_high_water_growth_bytes"] + RESERVE > MEMORY_CAP
                or view["traced_peak_bytes"] + view["tracer_metadata_bytes"] + RESERVE > MEMORY_CAP):
            raise AuditFailure("registered host-observation cap exceeded")
        if view["wall_s"] >= 58:
            raise Deadline("58-second work/postcheck budget exhausted")
        return view


def checked_file(path):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise AuditFailure("missing, relative or linked file: " + str(path))
    return path


def digest(path, meter):
    path = checked_file(path)
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            meter.hash_bytes += len(chunk)
            value.update(chunk)
            meter.check()
    meter.hash_files += 1
    meter.check()
    return value.hexdigest()


def authenticated_bytes(path, expected, cap, meter):
    path = checked_file(path)
    if not 0 < path.stat().st_size <= cap:
        raise AuditFailure("registered file-size cap: " + str(path))
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    meter.hash_files += 1
    meter.hash_bytes += len(raw)
    meter.check()
    if not 0 < len(raw) <= cap or hashlib.sha256(raw).hexdigest() != expected:
        raise AuditFailure("authenticated bytes differ: " + str(path))
    return raw


def json_read(path, meter, expected=None):
    if expected is None:
        expected = digest(path, meter)
    return json.loads(authenticated_bytes(path, expected, JSON_CAP, meter))


def identities(mapping, count):
    if type(mapping) is not dict or len(mapping) != count:
        raise AuditFailure("inherited identity population differs")
    for path, value in mapping.items():
        if (not isinstance(path, str) or not Path(path).is_absolute()
                or not isinstance(value, str) or len(value) != 64
                or any(ch not in "0123456789abcdef" for ch in value)):
            raise AuditFailure("malformed inherited identity")
    return mapping


def verify_all(mapping, meter):
    for path, expected in mapping.items():
        if digest(path, meter) != expected:
            raise AuditFailure("identity drift: " + path)


def save(name, value):
    path = RUN / name
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def evidence_digest(path):
    """Bound the final archive buffer; the 64 KiB reserve covers this buffer."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            value.update(chunk)
    return value.hexdigest()


def provenance(meter):
    def git(*args):
        meter.check()
        return subprocess.check_output(["git", *args], cwd=ROOT, timeout=3)
    record = dict(branch=git("branch", "--show-current").decode().strip(),
                  commit=git("rev-parse", "HEAD").decode().strip(),
                  tracked_diff_sha256=hashlib.sha256(git("diff", "--binary", "HEAD", "--")).hexdigest())
    if record != dict(branch="redu-hz", commit=COMMIT, tracked_diff_sha256=DIFF_SHA):
        raise AuditFailure("tracked production provenance differs")
    return record


def declared_value(value):
    """Only stored protobuf type/shape metadata; never infer a shape."""
    record = dict(name=value.name, shape_binding_verified=False)
    if value.type.HasField("tensor_type"):
        tensor = value.type.tensor_type
        record["data_type"] = int(tensor.elem_type)
        record["declared_shape"] = [
            int(dim.dim_value) if dim.HasField("dim_value") else
            {"parameter": dim.dim_param} if dim.HasField("dim_param") else None
            for dim in tensor.shape.dim]
    else:
        record["unsupported_non_tensor_type"] = True
    return record


def tensor_metadata(tensor):
    if tensor.data_location != 0 or len(tensor.external_data):
        raise AuditFailure("external tensor data is forbidden: " + tensor.name)
    if tensor.HasField("segment"):
        raise AuditFailure("segmented tensor payload is unsupported")
    dims = [int(value) for value in tensor.dims]
    if any(value < 0 for value in dims):
        raise AuditFailure("negative initializer dimension")
    # Lengths do not access or convert individual floating-point coefficients.
    counts = {name: len(getattr(tensor, name)) for name in
              ("float_data", "int32_data", "string_data", "int64_data", "double_data", "uint64_data")}
    return dict(name=tensor.name, dims=dims, data_type=int(tensor.data_type),
                data_location="DEFAULT_INLINE", external_data_entries=0,
                byte_count=int(tensor.ByteSize()), byte_count_kind="whole_tensor_protobuf_message",
                raw_payload_bytes=len(tensor.raw_data), typed_payload_counts=counts,
                coefficient_values_read=False)


def gemm_match(onnx, model, relu, relu_index, consumer, consumer_index, initializers):
    if relu.domain not in ("", "ai.onnx") or relu.attribute:
        raise AuditFailure("matched Relu has unsupported domain/attributes")
    if len(relu.input) != 1 or not relu.input[0] or len(relu.output) != 1:
        raise AuditFailure("matched Relu is not unary")
    if (consumer.domain not in ("", "ai.onnx") or len(consumer.input) not in (2, 3)
            or len(consumer.output) != 1 or consumer.input[0] != relu.output[0]):
        raise AuditFailure("matched Gemm interface is unsupported")
    versions = [int(item.version) for item in model.opset_import if item.domain in ("", "ai.onnx")]
    if len(versions) != 1 or versions[0] < 7:
        raise AuditFailure("standard Gemm opset is missing, ambiguous or unsupported")
    schema = onnx.defs.get_schema("Gemm", versions[0], "")
    if set(schema.attributes) != {"alpha", "beta", "transA", "transB"}:
        raise AuditFailure("Gemm schema attribute population differs")
    attrs = {}
    for attr in consumer.attribute:
        if attr.name in attrs or attr.name not in schema.attributes:
            raise AuditFailure("duplicate or unknown Gemm attribute")
        expected_type = onnx.AttributeProto.INT if attr.name in ("transA", "transB") else onnx.AttributeProto.FLOAT
        if attr.type != expected_type or attr.ref_attr_name:
            raise AuditFailure("Gemm attribute has unsupported type/reference")
        item = dict(present=True, protobuf_type=int(attr.type), value_decoded=False)
        if attr.name in ("transA", "transB"):
            if attr.i not in (0, 1):
                raise AuditFailure("unsupported transpose flag")
            item.update(integer_value=int(attr.i), value_decoded=True)
        attrs[attr.name] = item
    for name in ("alpha", "beta", "transA", "transB"):
        if name not in attrs:
            attrs[name] = dict(present=False, schema_default="1.0" if name in ("alpha", "beta") else "0",
                               value_decoded=False)
    trans_a = attrs["transA"].get("integer_value", 0)
    trans_b = attrs["transB"].get("integer_value", 0)
    if trans_a != 0:
        raise AuditFailure("transA=1 has no registered per-sample interpretation")
    name = consumer.input[1]
    if not name or name not in initializers:
        raise AuditFailure("matched Gemm B is not an authenticated inline initializer")
    parameter = initializers[name]
    dims = parameter["dims"]
    widths = {1: 4, 10: 2, 11: 8, 16: 2}
    dtype = parameter["data_type"]
    if len(dims) != 2 or any(value <= 0 for value in dims) or dtype not in widths:
        raise AuditFailure("matched Gemm B has unsupported dimensions/type")
    count = dims[0] * dims[1]
    expected_bytes = count * widths[dtype]
    if expected_bytes > MODEL_CAP:
        raise AuditFailure("matched Gemm logical tensor-size cap")
    typed = parameter["typed_payload_counts"]
    expected_field = "float_data" if dtype == 1 else "double_data" if dtype == 11 else "int32_data"
    if parameter["raw_payload_bytes"]:
        valid_payload = parameter["raw_payload_bytes"] == expected_bytes and not any(typed.values())
    else:
        valid_payload = typed[expected_field] == count and not any(value for key, value in typed.items() if key != expected_field)
    if not valid_payload:
        raise AuditFailure("matched Gemm B payload metadata disagrees with dimensions")
    bias = None
    if len(consumer.input) == 3 and consumer.input[2]:
        if consumer.input[2] not in initializers:
            raise AuditFailure("matched Gemm C is not an inline initializer")
        bias = initializers[consumer.input[2]]
    hidden, output = (dims[1], dims[0]) if trans_b else (dims[0], dims[1])
    return dict(relu_index=relu_index, relu_name=relu.name, relu_output=relu.output[0],
                gemm_index=consumer_index, gemm_name=consumer.name, gemm_output=consumer.output[0],
                standard_domain=True, opset=versions[0], gemm_schema_since_version=int(schema.since_version),
                attributes=attrs, initializer_B=parameter, initializer_C=bias,
                B_logical_payload_bytes=expected_bytes, hidden_dimension_from_B=hidden,
                output_dimension_from_B=output, rank_upper_bound_from_dimensions=min(hidden, output),
                actual_rank_computed=False, coefficient_signs_computed=False,
                alpha_beta_numeric_values_read=False, native_shape_binding_verified=False,
                dimensions_scope="Gemm B after transB; transA=0; actual A/batch binding not verified")


def inspect_model(onnx, model, report, meter):
    graph = model.graph
    if len(graph.node) > NODE_CAP or len(graph.initializer) > NODE_CAP:
        raise AuditFailure("metadata population cap exceeded")
    report.update(ir_version=int(model.ir_version),
                  opsets=[dict(domain=item.domain, version=int(item.version)) for item in model.opset_import],
                  graph_name=graph.name, graph_inputs=[declared_value(item) for item in graph.input],
                  graph_outputs=[declared_value(item) for item in graph.output],
                  nodes=[], initializers=[], relu_population=[], matches=[])
    if model.functions or model.training_info or graph.sparse_initializer:
        raise AuditFailure("functions, training graphs or sparse initializers are not registered")
    initializers = {}
    graph_inputs = [item.name for item in graph.input]
    graph_outputs = [item.name for item in graph.output]
    if (len(set(graph_inputs)) != len(graph_inputs) or len(set(graph_outputs)) != len(graph_outputs)
            or any(not name for name in graph_inputs + graph_outputs)):
        raise AuditFailure("empty or duplicate graph input/output name")
    for tensor in graph.initializer:
        if not tensor.name or tensor.name in initializers or tensor.name in graph_inputs:
            raise AuditFailure("initializer name collision or overridable graph input")
        metadata = tensor_metadata(tensor)
        initializers[tensor.name] = metadata
        report["initializers"].append(metadata)
        meter.check()
    available = set(graph_inputs) | set(initializers)
    consumers = {}
    operations = Counter()
    for index, node in enumerate(graph.node):
        attrs = []
        for attr in node.attribute:
            attrs.append(dict(name=attr.name, protobuf_type=int(attr.type), value_decoded=False))
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS,
                             onnx.AttributeProto.SPARSE_TENSOR, onnx.AttributeProto.SPARSE_TENSORS):
                raise AuditFailure("nested graph/sparse tensor attributes are unsupported")
            if attr.type == onnx.AttributeProto.TENSOR:
                tensor_metadata(attr.t)
            elif attr.type == onnx.AttributeProto.TENSORS:
                for tensor in attr.tensors:
                    tensor_metadata(tensor)
        record = dict(index=index, name=node.name, domain=node.domain, op_type=node.op_type,
                      inputs=list(node.input), outputs=list(node.output), attributes=attrs)
        report["nodes"].append(record)
        operations[(node.domain, node.op_type)] += 1
        for port, name in enumerate(node.input):
            if name:
                if name not in available:
                    raise AuditFailure("missing/topologically unavailable node input: " + name)
                consumers.setdefault(name, []).append((index, port))
        for name in node.output:
            if name:
                if name in available:
                    raise AuditFailure("node output name collision: " + name)
                available.add(name)
        meter.check()
    if any(name not in available for name in graph_outputs):
        raise AuditFailure("graph output is not produced")
    report["op_counts"] = [dict(domain=domain, op_type=op, count=count)
                           for (domain, op), count in sorted(operations.items())]
    for index, node in enumerate(graph.node):
        if node.op_type != "Relu":
            continue
        ports = [dict(output=name, consumers=[dict(node_index=i, input_port=p) for i, p in consumers.get(name, ())],
                      is_graph_output=name in graph_outputs) for name in node.output]
        row = dict(index=index, name=node.name, domain=node.domain, inputs=list(node.input), outputs=ports,
                   matched=False, reason=None)
        report["relu_population"].append(row)
        if len(node.output) != 1 or not node.output[0]:
            row["reason"] = "Relu does not have exactly one nonempty output"
            continue
        if node.domain not in ("", "ai.onnx"):
            row["reason"] = "Relu is not in a standard ONNX domain"
            continue
        uses = consumers.get(node.output[0], [])
        if node.output[0] in graph_outputs or len(uses) != 1:
            row["reason"] = "Relu output has graph-output use or not exactly one consuming port"
            continue
        target_index, port = uses[0]
        target = graph.node[target_index]
        if target.op_type != "Gemm":
            row["reason"] = "sole consumer is not Gemm"
            continue
        if target.domain not in ("", "ai.onnx"):
            row["reason"] = "Gemm consumer is not in a standard ONNX domain"
            continue
        if (len(graph_outputs) != 1 or len(target.output) != 1
                or target.output[0] != graph_outputs[0] or consumers.get(target.output[0])):
            row["reason"] = "Gemm is not the unconsumed sole graph output"
            continue
        if port != 0:
            raise AuditFailure("Relu is used as a Gemm parameter rather than A")
        transpose_a = [attr for attr in target.attribute if attr.name == "transA"]
        if (len(transpose_a) == 1 and transpose_a[0].type == onnx.AttributeProto.INT
                and not transpose_a[0].ref_attr_name and transpose_a[0].i == 1):
            row["reason"] = "transA=1 has no registered per-sample interpretation"
            continue
        match = gemm_match(onnx, model, node, index, target, target_index, initializers)
        report["matches"].append(match)
        row.update(matched=True, reason="direct unary Relu to terminal standard Gemm", match_index=len(report["matches"]) - 1)
        meter.check()
    report.update(node_count=len(graph.node), initializer_count=len(graph.initializer),
                  relu_count=len(report["relu_population"]), match_count=len(report["matches"]),
                  all_top_level_node_ports_retained=True, metadata_complete=True)


def main():
    if sys.argv[1:] != ["--enabled"]:
        raise AuditFailure("explicit --enabled required")
    RUN.mkdir(exist_ok=False)
    meter = Meter()
    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.alarm(58)
    result = dict(schema=SCHEMA, metadata_evidence_only=True, model_population=3, models=[],
                  source_precheck_complete=False, input_precheck_complete=False,
                  source_postcheck_complete=False, input_postcheck_complete=False,
                  metadata_audit_passed=False, mathematical_component_gate_passed=False,
                  source_component_qualified=False, native_HZ_admitted=False,
                  actual_model_binding_qualified=False, actual_phase_column_binding_verified=False,
                  actual_rank_computed=False, model_forward_executed=False,
                  candidate_executed=False, solver_executed=False, gpu_computation_completed=False,
                  complete_physical_qualification=False, formal_gain=0, new_benchmark_solves=0)
    artifacts = []
    status = 1
    try:
        resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))
        sys.dont_write_bytecode = True
        os.environ.update({name: "1" for name in THREAD_ENV})
        os.environ.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
                          CUDA_CACHE_PATH=str(RUN / "cuda_cache"), XDG_CACHE_HOME=str(RUN / "xdg_cache"),
                          TMPDIR=str(RUN / "tmp"))
        (RUN / "tmp").mkdir()
        tracemalloc.start()
        if not __debug__ or os.environ.get("PYTHONOPTIMIZE") not in (None, "", "0"):
            raise AuditFailure("optimized execution is not registered")
        prior = json_read(PRIOR, meter, PRIOR_SHA)
        if prior.get("schema") != "d150_stable_phase_charts_v1":
            raise AuditFailure("inherited registration schema differs")
        source_ids = identities(prior.get("source_sha256"), 7285)
        input_ids = identities(prior.get("input_sha256"), 14)
        affinity = prior.get("cpu_affinity")
        if type(affinity) is not list or len(affinity) != 1 or type(affinity[0]) is not int:
            raise AuditFailure("inherited one-CPU binding differs")
        os.sched_setaffinity(0, set(affinity))
        if sorted(os.sched_getaffinity(0)) != affinity:
            raise AuditFailure("one-CPU binding failed")
        freeze_digest = digest(FREEZE, meter)
        frozen = json_read(FREEZE, meter, freeze_digest)
        if (set(frozen) != {"schema", "source_sha256"} or frozen.get("schema") != SCHEMA
                or type(frozen.get("source_sha256")) is not dict
                or set(frozen["source_sha256"]) != {str(HERE / name) for name in FILES}):
            raise AuditFailure("D152 exact two-file freeze differs")
        fresh_ids = identities(frozen["source_sha256"], 2)
        verify_all(fresh_ids, meter)
        verify_all(source_ids, meter)
        result["source_precheck_complete"] = True
        interpreter = str(Path(sys.executable).resolve())
        if (Path(interpreter) != PYTHON.resolve() or interpreter not in source_ids
                or digest(interpreter, meter) != source_ids[interpreter]):
            raise AuditFailure("interpreter is not inherited/authenticated")
        selected = []
        for index, expected in enumerate(SOURCE_JSON_SHA):
            path = OLD / ("complete_" + str(index) + ".json")
            if source_ids.get(str(path)) != expected:
                raise AuditFailure("D120 complete JSON identity differs")
            document = json_read(path, meter, expected)
            if document.get("schema") != "d120_mixed_consumer_source_v1" or type(document.get("source")) is not dict:
                raise AuditFailure("D120 complete source schema differs")
            selected.append(dict(source_json=str(path), source_json_sha256=expected, source=document["source"]))
        if [item["source"] for item in selected] != prior.get("selected_sources"):
            raise AuditFailure("fixed three-model source order/population differs")
        model_ids = {item["source"]["model_path"]: item["source"]["model_sha256"] for item in selected}
        if len(model_ids) != 3 or any(input_ids.get(path) != value for path, value in model_ids.items()):
            raise AuditFailure("model identity is not in the original input population")
        raw_models = {}
        for path, expected in input_ids.items():
            if path in model_ids:
                raw_models[path] = authenticated_bytes(path, expected, MODEL_CAP, meter)
            elif digest(path, meter) != expected:
                raise AuditFailure("inherited input drift: " + path)
        result["input_precheck_complete"] = True
        result["provenance_before"] = provenance(meter)
        manifest = dict(schema=SCHEMA, prior_manifest=str(PRIOR), prior_manifest_sha256=PRIOR_SHA,
                        source_identities=7285, original_input_identities=14, freeze_sha256=freeze_digest,
                        fresh_source_sha256=fresh_ids, selected_sources=selected, cpu_affinity=affinity,
                        address_space_bytes=AS_CAP, memory_observation_cap_bytes=MEMORY_CAP,
                        work_deadline_seconds=58, outer_shell_deadline_seconds=60,
                        metadata_only=True, coefficient_values_read=False,
                        inherited_sources_authenticated=True, inherited_inputs_authenticated=True)
        save("manifest_summary.json", manifest)
        artifacts.append("manifest_summary.json")
        # Only now may an authenticated non-stdlib dependency be imported.
        specification = importlib.util.find_spec("onnx")
        if (specification is None or not specification.origin
                or str(Path(specification.origin).resolve()) not in source_ids):
            raise AuditFailure("ONNX import discovery is outside the authenticated manifest")
        import onnx
        module_file = str(Path(onnx.__file__).resolve())
        if module_file not in source_ids or digest(module_file, meter) != source_ids[module_file]:
            raise AuditFailure("loaded ONNX module is outside the authenticated manifest")
        result["onnx_module_identity"] = dict(path=module_file, sha256=source_ids[module_file])
        for index, selected_source in enumerate(selected):
            report = dict(schema=SCHEMA, index=index, source=selected_source,
                          metadata_complete=False, coefficient_values_read=False,
                          actual_rank_computed=False, native_shape_binding_verified=False)
            fatal = None
            model = raw = None
            try:
                path = selected_source["source"]["model_path"]
                raw = raw_models.pop(path)
                report["model_file_bytes"] = len(raw)
                # BytesIO prevents a second path read or an external-data path lookup.
                model = onnx.load_model(io.BytesIO(raw), format="protobuf", load_external_data=False)
                meter.check()
                inspect_model(onnx, model, report, meter)
            except (Deadline, MemoryError) as error:
                report["metadata_complete"] = False
                report["failure"] = type(error).__name__ + ": " + str(error)
                fatal = error
            except Exception as error:
                report["failure"] = type(error).__name__ + ": " + str(error)
            finally:
                model = raw = None
            name = "model_" + str(index) + ".json"
            save(name, report)
            artifacts.append(name)
            result["models"].append(dict(index=index, artifact=name, metadata_complete=report["metadata_complete"],
                                         match_count=report.get("match_count"), failure=report.get("failure")))
            if fatal is not None:
                raise fatal
            meter.check()
        verify_all(source_ids, meter)
        result["source_postcheck_complete"] = True
        verify_all(input_ids, meter)
        result["input_postcheck_complete"] = True
        verify_all(fresh_ids, meter)
        if digest(FREEZE, meter) != freeze_digest or digest(PRIOR, meter) != PRIOR_SHA:
            raise AuditFailure("registration drift")
        result["provenance_after"] = provenance(meter)
        result["host_observations"] = meter.check()
        if len(result["models"]) != 3 or any(not item["metadata_complete"] for item in result["models"]):
            raise AuditFailure("not all three model metadata records completed")
        result["metadata_audit_passed"] = True
        status = 0
    except BaseException as error:
        result["failure"] = type(error).__name__ + ": " + str(error)
    finally:
        signal.alarm(0)
        result["host_observations"] = meter.snapshot()
        view = result["host_observations"]
        result["host_observations_within_caps"] = (
            view["rss_high_water_growth_bytes"] + RESERVE <= MEMORY_CAP
            and view["traced_peak_bytes"] + view["tracer_metadata_bytes"] + RESERVE <= MEMORY_CAP)
        if not result["host_observations_within_caps"] or view["wall_s"] >= 60:
            status = 1
            result["metadata_audit_passed"] = False
            result.setdefault("failure", "final resource observations exceed registered caps")
        result["exit_code"] = status
        save("result.json", result)
        artifacts.append("result.json")
        # No historical file is written; these hashes describe new evidence only.
        hashes = {name: evidence_digest(RUN / name) for name in artifacts}
        save("exit.json", dict(schema=SCHEMA, exit_code=status, artifacts=hashes,
                               metadata_audit_passed=result["metadata_audit_passed"],
                               retry_permitted=False, formal_gain=0,
                               wall_s=time.monotonic() - meter.started))
        exit_hash = evidence_digest(RUN / "exit.json")
        save("artifact_hashes.json", dict(artifacts=hashes, exit_sha256=exit_hash))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
