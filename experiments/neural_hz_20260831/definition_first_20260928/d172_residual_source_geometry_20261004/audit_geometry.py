"""Once-only coefficient geometry diagnostic; never execute a network or candidate.

All five MLP blocks are selected by ports, not labels or verifier outcomes.
FLOAT constants denote their exact real values; BN folding is outward enclosed.
The resulting sector constants are diagnostics, not model or domain qualification.
"""

from fractions import Fraction as F
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import resource
import signal
import struct
import subprocess
import sys
import time
import tracemalloc


HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
ROOT = EXP.parent.parent
RUN = EXP / "results/d172_residual_source_geometry_20261004_v1"
PRIOR = EXP / "results/d158_joint_forward_support_20261004_v1/preregistered.json"
PRIOR_SHA = "64cd535e4ae10bfaeb8204d21841f991781aa981e2aeddb359a87ca8050103d9"
SCHEMA = "d172_residual_source_geometry_v1"
FILES = ("PREREG.md", "inputs.json", "audit_geometry.py", "DEFINITION_AND_SCOPE.md")
PYTHON = Path("/data1/Kane/miniconda3/bin/python")
COMMIT = "f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac"
DIFF_SHA = "29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5"
AS_CAP, MEMORY_CAP, RESERVE = 16 * 1024**3, 1024**3, 65536
WORK_CAP, BRANCH_CAP, EVIDENCE_CAP, ENTRY_CAP = 256_000_000, 200_000_000, 40_000_000, 64_000_000
MODEL_CAP, GRAPH_CAP, JSON_CAP, BITS = 1_000_000, 1_000_000, 8_000_000, 512
GRID = 1 << 64
FINAL_BYTES = 0
FINAL_HASH_WORK = 0
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


class Failure(ValueError):
    pass


class Deadline(Failure):
    pass


def require(condition, message):
    if not condition:
        raise Failure(message)


def stop(signum, _frame):
    raise Deadline("registered deadline/termination: " + str(signum))


class Meter:
    def __init__(self):
        self.started = time.monotonic()
        self.rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        self.work = EVIDENCE_CAP
        self.branch = 0
        self.evidence = RESERVE
        self.entries = 0
        self.hash_files = 0
        self.hash_bytes = 0

    def charge(self, amount, *, evidence=False, entries=0):
        require(type(amount) is int and amount >= 0 and type(entries) is int and entries >= 0,
                "invalid prepaid accounting")
        require(evidence or amount <= WORK_CAP - self.work, "whole work cap")
        require(amount <= (EVIDENCE_CAP - self.evidence if evidence else BRANCH_CAP - self.branch),
                "evidence/model work cap")
        require(entries <= ENTRY_CAP - self.entries, "retained-entry conservative admission cap")
        if evidence:
            self.evidence += amount
        else:
            self.work += amount
            self.branch += amount
        self.entries += entries

    def snapshot(self):
        current, peak = tracemalloc.get_traced_memory()
        metadata = tracemalloc.get_tracemalloc_memory()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        return dict(wall_s=time.monotonic() - self.started, work_used=self.work,
                    branch_work_used=self.branch, evidence_work_used=self.evidence,
                    evidence_prepaid_work=EVIDENCE_CAP,
                    admitted_entries_total=self.entries, rss_initial_high_water_bytes=self.rss0,
                    rss_high_water_bytes=rss, rss_high_water_growth_bytes=max(0, rss - self.rss0),
                    traced_current_bytes=current, traced_peak_bytes=peak,
                    tracer_metadata_bytes=metadata, reserve_bytes=RESERVE,
                    authenticated_files=self.hash_files, authenticated_bytes=self.hash_bytes,
                    complete_physical_qualification=False)

    def check(self):
        view = self.snapshot()
        require(view["rss_high_water_growth_bytes"] + RESERVE <= MEMORY_CAP
                and view["traced_peak_bytes"] + view["tracer_metadata_bytes"] + RESERVE <= MEMORY_CAP,
                "registered host memory observation cap")
        if view["wall_s"] >= 58:
            raise Deadline("58-second work/postcheck budget")
        return view


def checked_file(path):
    path = Path(path)
    require(path.is_absolute() and not path.is_symlink() and path.is_file(),
            "ordinary absolute file required: " + str(path))
    return path


def digest(path, meter=None):
    global FINAL_HASH_WORK
    path = checked_file(path)
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            if meter is not None:
                meter.charge(16)  # One bounded 64-KiB authentication-buffer operation.
            else:
                require(FINAL_BYTES + FINAL_HASH_WORK + 16 <= RESERVE, "final authentication reserve")
                FINAL_HASH_WORK += 16
            chunk = stream.read(65536)
            if not chunk:
                break
            value.update(chunk)
            if meter is not None:
                meter.hash_bytes += len(chunk)
                meter.check()
    if meter is not None:
        meter.hash_files += 1
    return value.hexdigest()


def authenticated_bytes(path, expected, cap, meter):
    path = checked_file(path)
    size = path.stat().st_size
    require(0 < size <= cap, "file-size admission cap: " + str(path))
    meter.charge(4096 + size, entries=size)
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    require(len(raw) == size and hashlib.sha256(raw).hexdigest() == expected,
            "authenticated bytes mismatch: " + str(path))
    meter.hash_files += 1
    meter.hash_bytes += size
    meter.check()
    return raw


def json_document(raw, meter):
    meter.charge(4 * len(raw), entries=len(raw))
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise Failure("nonstandard JSON number: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def json_read(path, expected, meter, cap=JSON_CAP):
    return json_document(authenticated_bytes(path, expected, cap, meter), meter)


def identities(mapping, count):
    require(type(mapping) is dict and len(mapping) == count, "identity population differs")
    for path, value in mapping.items():
        require(type(path) is str and Path(path).is_absolute() and type(value) is str
                and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
                "malformed frozen identity")
    return mapping


def verify_all(mapping, meter):
    for path, expected in mapping.items():
        require(digest(path, meter) == expected, "identity drift: " + path)


def provenance(meter):
    def git(*args):
        meter.check()
        return subprocess.check_output(["git", *args], cwd=ROOT, timeout=3)
    result = dict(branch=git("branch", "--show-current").decode().strip(),
                  commit=git("rev-parse", "HEAD").decode().strip(),
                  tracked_diff_sha256=hashlib.sha256(git("diff", "--binary", "HEAD", "--")).hexdigest())
    require(result == dict(branch="redu-hz", commit=COMMIT, tracked_diff_sha256=DIFF_SHA),
            "production provenance drift")
    return result


def serialization_bill(value, meter):
    """Prepay traversal and an upper bound on compact JSON output characters."""
    meter.charge(12, evidence=True)
    if value is None or type(value) is bool:
        return 5
    if type(value) is int:
        return 3 + (abs(value).bit_length() + 2) // 3
    if type(value) is float:
        require(math.isfinite(value), "nonfinite evidence float")
        return 32
    if type(value) is str:
        return 2 + 12 * len(value)
    if type(value) in (list, tuple):
        return 2 + len(value) + sum(serialization_bill(x, meter) for x in value)
    require(type(value) is dict and all(type(k) is str for k in value), "unsupported evidence type")
    return 2 + 2 * len(value) + sum(serialization_bill(k, meter) + serialization_bill(v, meter)
                                   for k, v in value.items())


def save(name, value, meter=None):
    # Canonical compact JSON; full coefficient envelopes have their own artifact.
    global FINAL_BYTES
    if meter is not None:
        bound = serialization_bill(value, meter)
        meter.charge(1024 + 2 * bound, evidence=True)
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    path = RUN / name
    written = 0
    with path.open("x", encoding="utf-8") as stream:
        for chunk in encoder.iterencode(value):
            written += len(chunk)
            if meter is None:
                require(FINAL_BYTES + FINAL_HASH_WORK + written + 1 <= RESERVE,
                        "bounded final/partial summary reserve")
            stream.write(chunk)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if meter is None:
        FINAL_BYTES += written + 1
    else:
        require(written <= bound, "JSON billing bound mismatch")
    return digest(path, meter)


class Arithmetic:
    def __init__(self, meter):
        self.meter = meter

    def check(self, value):
        require(type(value) is F and max(value.numerator.bit_length(), value.denominator.bit_length()) <= BITS,
                "512-bit exact rational cap")
        return value

    def round(self, lo, hi=None):
        self.meter.charge(24)
        self.check(lo)
        hi = lo if hi is None else self.check(hi)
        require(lo <= hi, "reversed rational interval")
        low_scaled, high_scaled = lo.numerator * GRID, hi.numerator * GRID
        require(max(abs(low_scaled).bit_length(), abs(high_scaled).bit_length()) <= BITS,
                "512-bit rounding intermediate cap")
        lower = low_scaled // lo.denominator
        upper = -((-high_scaled) // hi.denominator)
        return self.check(F(lower, GRID)), self.check(F(upper, GRID))

    def add(self, a, b):
        self.meter.charge(8)
        return self.round(a[0] + b[0], a[1] + b[1])

    def mul(self, a, b):
        self.meter.charge(16)
        values = tuple(self.check(x * y) for x in a for y in b)
        return self.round(min(values), max(values))

    def divide(self, a, b):
        self.meter.charge(16)
        require(b[0] > 0 or b[1] < 0, "interval division crosses zero")
        values = tuple(self.check(x / y) for x in a for y in b)
        return self.round(min(values), max(values))

    def sqrt(self, value):
        self.meter.charge(48)
        self.check(value)
        require(value >= 0, "negative square-root input")
        scaled = value.numerator * GRID * GRID
        require(scaled.bit_length() <= BITS, "512-bit isqrt intermediate cap")
        lower = math.isqrt(scaled // value.denominator)
        square = lower * lower * value.denominator
        require(square.bit_length() <= BITS, "512-bit root endpoint-check intermediate")
        upper = lower if square == scaled else lower + 1
        return self.check(F(lower, GRID)), self.check(F(upper, GRID))

    def upper_add(self, x, y):
        return self.add((x, x), (y, y))[1]

    def upper_mul(self, x, y):
        require(x >= 0 and y >= 0, "nonnegative upper-bound arithmetic required")
        return self.mul((x, x), (y, y))[1]


def encode_fraction(value):
    return [value.numerator, value.denominator]


def encode_interval(value):
    return [encode_fraction(value[0]), encode_fraction(value[1])]


class Graph:
    def __init__(self, onnx, model, inventory, meter):
        self.onnx, self.model, self.meter = onnx, model, meter
        self.nodes = tuple(model.graph.node)
        meter.charge(4096 + 64 * len(self.nodes), entries=64 * len(self.nodes))
        require(not model.functions and not model.training_info and not model.graph.sparse_initializer,
                "functions, training graphs, sparse initializers unsupported")
        require(0 < len(self.nodes) <= 10000, "node cap")
        require([(x.domain, x.version) for x in model.opset_import] == [("", 9)],
                "this registered ordinary grammar requires standard opset 9")
        records = inventory.get("nodes")
        require(type(records) is list and len(records) == len(self.nodes), "full graph inventory mismatch")
        self.initializers, self.producers, self.consumers = {}, {}, {}
        self.cache = {}
        self.graph_outputs = {x.name for x in model.graph.output}
        graph_inputs = [x.name for x in model.graph.input]
        require(all(graph_inputs) and len(set(graph_inputs)) == len(graph_inputs)
                and len(self.graph_outputs) == len(model.graph.output) and all(self.graph_outputs),
                "duplicate/empty graph input/output identity")
        for tensor in model.graph.initializer:
            require(tensor.name and tensor.name not in self.initializers and not tensor.external_data
                    and tensor.data_location == 0 and not tensor.HasField("segment"),
                    "initializer identity/external data unsupported")
            self.initializers[tensor.name] = tensor
        require(not any(x.name in self.initializers for x in model.graph.input), "overridable initializer")
        for index, (node, record) in enumerate(zip(self.nodes, records)):
            require(record.get("index") == index and record.get("op") == node.op_type
                    and record.get("name") == node.name and record.get("domain") == node.domain
                    and tuple(record.get("inputs", ())) == tuple(node.input)
                    and tuple(record.get("outputs", ())) == tuple(node.output), "original saved port mismatch")
            for attribute in node.attribute:
                require(attribute.type not in (5, 10), "nested graph unsupported")
                require(attribute.type not in (11, 12), "sparse tensor attributes unsupported")
                tensors = (attribute.t,) if attribute.type == 4 else attribute.tensors if attribute.type == 9 else ()
                for tensor in tensors:
                    require(not tensor.external_data and tensor.data_location == 0
                            and not tensor.HasField("segment"), "external/segmented tensor attribute")
            for output in node.output:
                require(output and output not in self.producers and output not in self.initializers
                        and output not in graph_inputs,
                        "duplicate/empty output identity")
                self.producers[output] = index
            for port, name in enumerate(node.input):
                if name:
                    self.consumers.setdefault(name, []).append((index, port))

    def node(self, index, op, count=None):
        node = self.nodes[index]
        require(node.op_type == op and node.domain in ("", "ai.onnx")
                and len(node.output) == 1 and bool(node.output[0]), "node grammar: " + op)
        if count is not None:
            require(len(node.input) == count, "node arity: " + op)
        return node

    def previous(self, port):
        require(port in self.producers, "missing dynamic producer")
        return self.producers[port]

    def only_use(self, output, index, port):
        require(self.consumers.get(output) == [(index, port)] and output not in self.graph_outputs,
                "hidden live consumer/output: " + output)

    def next(self, output, op, port):
        uses = self.consumers.get(output, ())
        require(len(uses) == 1 and uses[0][1] == port and output not in self.graph_outputs,
                "hidden consumer grammar")
        return uses[0][0], self.node(uses[0][0], op)

    def attrs(self, node, allowed):
        result = {}
        for a in node.attribute:
            require(a.name in allowed and a.name not in result, "unknown/duplicate operator attribute")
            result[a.name] = a
        return result

    def transpose(self, index):
        node = self.node(index, "Transpose", 1)
        attrs = self.attrs(node, {"perm"})
        require(set(attrs) == {"perm"} and attrs["perm"].type == 7
                and tuple(attrs["perm"].ints) == (0, 2, 1), "MLP channel transpose differs")
        return node

    def bias(self, index):
        node = self.node(index, "Add", 2)
        require(not node.attribute, "Add attributes unsupported")
        constant_ports = [p for p, name in enumerate(node.input) if name in self.initializers]
        require(len(constant_ports) == 1, "one initializer bias required")
        cp = constant_ports[0]
        return node, node.input[1 - cp], node.input[cp], 1 - cp

    def tensor(self, name, shape, arithmetic):
        require(name in self.initializers, "inline initializer required")
        if name in self.cache:
            old_shape, value = self.cache[name]
            require(old_shape == shape, "shared initializer shape mismatch")
            return value
        tensor = self.initializers[name]
        require(tuple(tensor.dims) == shape and tensor.data_type == 1, "exact FLOAT initializer shape")
        count = math.prod(shape)
        require(0 < count <= 1_000_000, "per-tensor scalar cap")
        self.meter.charge(4096 + 48 * count, entries=8 * count)
        require(not tensor.double_data and not tensor.int32_data and not tensor.int64_data
                and not tensor.uint64_data and not tensor.string_data, "ambiguous FLOAT payload")
        if tensor.raw_data:
            require(not tensor.float_data and len(tensor.raw_data) == 4 * count, "malformed FLOAT raw payload")
            numbers = (item[0] for item in struct.iter_unpack("<f", tensor.raw_data))
        else:
            require(len(tensor.float_data) == count, "malformed FLOAT repeated payload")
            numbers = iter(tensor.float_data)
        values = []
        for number in numbers:
            require(math.isfinite(number), "nonfinite FLOAT initializer")
            values.append(arithmetic.check(F.from_float(number)))
        result = tuple(values)
        self.cache[name] = (shape, result)
        return result

    def match(self, relu_index):
        relu = self.node(relu_index, "Relu", 1)
        require(not relu.attribute, "Relu attributes unsupported")
        b1i = self.previous(relu.input[0])
        b1, mm1out, b1name, b1port = self.bias(b1i)
        mm1i = self.previous(mm1out)
        mm1 = self.node(mm1i, "MatMul", 2)
        require(not mm1.attribute and mm1.input[1] in self.initializers, "first MLP weight grammar")
        t2i = self.previous(mm1.input[0]); t2 = self.transpose(t2i)
        bni = self.previous(t2.input[0]); bn = self.node(bni, "BatchNormalization", 5)
        t1i = self.previous(bn.input[0]); t1 = self.transpose(t1i)
        q = t1.input[0]
        mm2i, mm2 = self.next(relu.output[0], "MatMul", 0)
        require(len(mm2.input) == 2 and not mm2.attribute and mm2.input[1] in self.initializers,
                "second MLP weight grammar")
        uses = self.consumers.get(mm2.output[0], ())
        require(len(uses) == 1, "second MLP output has multiple consumers")
        b2i = uses[0][0]; b2, previous, b2name, b2port = self.bias(b2i)
        require(previous == mm2.output[0], "second MLP bias linkage")
        uses = self.consumers.get(b2.output[0], ())
        require(len(uses) == 1, "MLP branch has external consumer")
        addi = uses[0][0]; add = self.node(addi, "Add", 2)
        require(not add.attribute and list(add.input).count(q) == 1
                and list(add.input).count(b2.output[0]) == 1, "same-parent identity residual required")
        qport = list(add.input).index(q)
        for output, target, port in ((t1.output[0], bni, 0), (bn.output[0], t2i, 0),
                                    (t2.output[0], mm1i, 0), (mm1.output[0], b1i, b1port),
                                    (b1.output[0], relu_index, 0), (relu.output[0], mm2i, 0),
                                    (mm2.output[0], b2i, b2port),
                                    (b2.output[0], addi, 1 - qport)):
            self.only_use(output, target, port)
        require(sorted(self.consumers.get(q, ())) == sorted([(t1i, 0), (addi, qport)]),
                "registered complete parent consumers differ")
        require(tuple(self.initializers[mm1.input[1]].dims) == (48, 96)
                and tuple(self.initializers[mm2.input[1]].dims) == (96, 48), "registered complete MLP dimensions")
        return dict(relu_index=relu_index, q=q, z=add.output[0], n=48, m=96,
                    nodes=dict(transpose_before_bn=t1i, bn=bni, transpose_after_bn=t2i,
                               matmul1=mm1i, bias1=b1i, relu=relu_index,
                               matmul2=mm2i, bias2=b2i, residual_add=addi),
                    tensors=dict(W1=mm1.input[1], b1=b1name, W2=mm2.input[1], b2=b2name),
                    q_consumers=[list(x) for x in self.consumers[q]],
                    z_consumers=[list(x) for x in self.consumers.get(add.output[0], ())],
                    z_is_graph_output=add.output[0] in self.graph_outputs,
                    hidden_consumers_complete=True, native_shape_binding_verified=False)


def fold(graph, match, arithmetic):
    n, m = match["n"], match["m"]
    bn = graph.nodes[match["nodes"]["bn"]]
    attrs = graph.attrs(bn, {"epsilon", "momentum"})
    require("momentum" not in attrs or (attrs["momentum"].type == 1
                                       and math.isfinite(attrs["momentum"].f)),
            "finite opset-9 BN momentum attribute")
    require("epsilon" in attrs and attrs["epsilon"].type == 1
            and math.isfinite(attrs["epsilon"].f), "explicit finite BN epsilon required")
    epsilon = arithmetic.check(F.from_float(attrs["epsilon"].f))
    require(epsilon >= 0, "negative BN epsilon")
    gamma, beta, mean, variance = [graph.tensor(name, (n,), arithmetic) for name in bn.input[1:]]
    scales, shifts = [], []
    for i in range(n):
        total = arithmetic.check(variance[i] + epsilon)
        require(total > 0, "nonpositive BN variance plus epsilon")
        denominator = arithmetic.sqrt(total)
        require(denominator[0] > 0, "BN outward sqrt includes zero")
        scale = arithmetic.divide((gamma[i], gamma[i]), denominator)
        shift = arithmetic.add((beta[i], beta[i]),
                               arithmetic.mul((-mean[i], -mean[i]), scale))
        scales.append(scale); shifts.append(shift)
    names = match["tensors"]
    W1 = graph.tensor(names["W1"], (n, m), arithmetic)
    b1 = graph.tensor(names["b1"], (m,), arithmetic)
    W2 = graph.tensor(names["W2"], (m, n), arithmetic)
    b2 = graph.tensor(names["b2"], (n,), arithmetic)
    graph.meter.charge(64 * (n * m + n + m), entries=16 * (n * m + n + m))
    V, c = [], []
    for j in range(m):
        row = []
        bias = (b1[j], b1[j])
        for i in range(n):
            point = (W1[i * m + j], W1[i * m + j])
            row.append(arithmetic.mul(point, scales[i]))
            bias = arithmetic.add(bias, arithmetic.mul(point, shifts[i]))
        V.append(tuple(row)); c.append(bias)
    U = tuple(tuple((W2[j * n + i], W2[j * n + i]) for j in range(m)) for i in range(n))
    d = tuple((x, x) for x in b2)
    return tuple(V), tuple(c), U, d, tuple(scales), tuple(shifts)


def norm_square(matrix, arithmetic):
    rows, columns = len(matrix), len(matrix[0])
    arithmetic.meter.charge(16 * rows * columns, entries=rows + columns)
    row_sums, column_sums, frobenius = [], [F(0)] * columns, F(0)
    for row in matrix:
        total = F(0)
        for j, value in enumerate(row):
            magnitude = max(abs(value[0]), abs(value[1]))
            total = arithmetic.upper_add(total, magnitude)
            column_sums[j] = arithmetic.upper_add(column_sums[j], magnitude)
            frobenius = arithmetic.upper_add(frobenius, arithmetic.upper_mul(magnitude, magnitude))
        row_sums.append(total)
    one, infinity = max(column_sums), max(row_sums)
    product = arithmetic.upper_mul(one, infinity)
    return min(frobenius, product), dict(frobenius_squared_upper=encode_fraction(frobenius),
                                       one_norm_upper=encode_fraction(one),
                                       infinity_norm_upper=encode_fraction(infinity),
                                       one_times_infinity_upper=encode_fraction(product))


def geometry(V, U, arithmetic):
    K, Vdetail = norm_square(V, arithmetic)
    # lambda is a fixed exact rational selected from the certified K, not searched.
    lam = arithmetic.check(F(1) / arithmetic.check(F(1) + K))
    alpha = arithmetic.check(lam * K / 2)
    E = tuple(tuple(arithmetic.add(U[i][j], arithmetic.mul((lam, lam), V[j][i]))
                    for j in range(len(V))) for i in range(len(U)))
    Esq, Edetail = norm_square(E, arithmetic)
    Usq, Udetail = norm_square(U, arithmetic)
    nu, eps, u = arithmetic.sqrt(K)[1], arithmetic.sqrt(Esq)[1], arithmetic.sqrt(Usq)[1]
    defect = arithmetic.upper_mul(eps, nu)
    rho = arithmetic.upper_add(alpha, defect)
    L = arithmetic.upper_add(F(1), defect)
    triangle = arithmetic.upper_add(F(1), arithmetic.upper_mul(u, nu))
    ratio = arithmetic.divide((L, L), (triangle, triangle))
    return dict(K=encode_fraction(K), lambda_exact=encode_fraction(lam), alpha=encode_fraction(alpha),
                nu_upper=encode_fraction(nu), epsilon_upper=encode_fraction(eps), u_upper=encode_fraction(u),
                rho_upper=encode_fraction(rho), L_upper=encode_fraction(L),
                old_triangle_upper=encode_fraction(triangle),
                ratio_of_certified_upper_constants_L_over_triangle=encode_interval(ratio),
                ratio_is_not_true_operator_norm_ratio=True,
                lambda_rule="1/(1+K); no search", norm_rule="min(Frobenius_squared,one_norm*infinity_norm)",
                V_norm_certificate=Vdetail, U_norm_certificate=Udetail, E_norm_certificate=Edetail,
                parameter_only_diagnostic=True, candidate_bound_computed=False)


def inspect_model(onnx, raw, inventory, selected, index, report, meter, artifacts):
    meter.charge(4096 + 8 * len(raw), entries=8 * len(raw))
    model = onnx.load_model(io.BytesIO(raw), format="protobuf", load_external_data=False)
    graph = Graph(onnx, model, inventory, meter)
    arithmetic = Arithmetic(meter)
    relus = [i for i, node in enumerate(graph.nodes) if node.op_type == "Relu"]
    require(len(relus) == selected["expected_mlp_blocks"], "complete original ReLU population differs")
    report.update(node_count=len(graph.nodes), relu_population=relus, blocks=[])
    for block_index, relu_index in enumerate(relus):
        item = dict(relu_index=relu_index, complete=False)
        report["blocks"].append(item)
        try:
            match = graph.match(relu_index)
            item["structure"] = match
            V, c, U, d, scale, shift = fold(graph, match, arithmetic)
            item["geometry"] = geometry(V, U, arithmetic)
            # All originals needed for this block are sealed with exact rational values.
            meter.charge(128 * (48 * 96 * 2 + 96 + 48), evidence=True,
                         entries=24 * (48 * 96 * 2 + 96 + 48))
            used_names = list(match["tensors"].values()) + list(graph.nodes[match["nodes"]["bn"]].input[1:])
            original = {name: dict(shape=list(graph.cache[name][0]),
                                   values=[encode_fraction(x) for x in graph.cache[name][1]]) for name in used_names}
            packet = dict(schema=SCHEMA, model_sha256=selected["model_sha256"], structure=match,
                          arithmetic="original FLOAT exact; folded intervals outward dyadic64",
                          original_constants=original,
                          bn_epsilon=encode_fraction(F.from_float(
                              graph.attrs(graph.nodes[match["nodes"]["bn"]],
                                          {"epsilon", "momentum"})["epsilon"].f)),
                          V=[[encode_interval(x) for x in row] for row in V],
                          U=[[encode_interval(x) for x in row] for row in U],
                          c=[encode_interval(x) for x in c], d=[encode_interval(x) for x in d],
                          bn_scale=[encode_interval(x) for x in scale],
                          bn_shift=[encode_interval(x) for x in shift], geometry=item["geometry"])
            name = "coefficients_" + str(index) + "_" + str(block_index) + ".json"
            item["coefficient_packet_sha256"] = save(name, packet, meter)
            item["coefficient_packet"] = name
            artifacts.append(name)
            item["complete"] = True
            del packet, original, V, c, U, d, scale, shift
            graph.cache.clear()
            meter.check()
        except (Deadline, MemoryError):
            raise
        except Exception as error:
            item["failure"] = type(error).__name__ + ": " + str(error)
    report["diagnostic_complete"] = all(item["complete"] for item in report["blocks"])


def main():
    require(sys.argv[1:] == ["--enabled"], "explicit --enabled required")
    RUN.mkdir(exist_ok=False)
    meter = Meter()
    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.alarm(58)
    result = dict(schema=SCHEMA, coefficient_diagnostic_only=True, diagnostic_complete=False,
                  models=[], mathematical_component_gate_passed=False, source_component_qualified=False,
                  source_census_qualified=False, actual_model_binding_qualified=False,
                  actual_phase_column_binding_verified=False, native_HZ_admitted=False,
                  model_forward_executed=False, candidate_executed=False, solver_executed=False,
                  gpu_computation_completed=False, complete_physical_qualification=False,
                  formal_gain=0, new_benchmark_solves=0)
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
        require(__debug__ and os.environ.get("PYTHONOPTIMIZE") in (None, "", "0"), "optimized execution forbidden")
        prior = json_read(PRIOR, PRIOR_SHA, meter)
        require(prior.get("schema") == "d158_joint_forward_support_v1", "prior registration schema")
        source_ids = identities(prior.get("source_sha256"), 7327)
        input_ids = identities(prior.get("input_sha256"), 14)
        require(prior.get("cpu_affinity") == [0], "inherited CPU binding")
        os.sched_setaffinity(0, {0})
        require(sorted(os.sched_getaffinity(0)) == [0], "CPU binding failed")
        freeze_digest = digest(HERE / "freeze.json", meter)
        frozen = json_read(HERE / "freeze.json", freeze_digest, meter)
        require(set(frozen) == {"schema", "source_sha256"} and frozen["schema"] == SCHEMA,
                "freeze schema mismatch")
        fresh = identities(frozen["source_sha256"], 4)
        require(set(fresh) == {str(HERE / name) for name in FILES}, "four-file freeze differs")
        verify_all(fresh, meter)
        verify_all(source_ids, meter)
        result["source_precheck_complete"] = True
        require(Path(sys.executable).resolve() == PYTHON.resolve()
                and str(PYTHON.resolve()) in source_ids, "authenticated interpreter required")
        inputs = json_read(HERE / "inputs.json", fresh[str(HERE / "inputs.json")], meter)
        require(set(inputs) == {"schema", "models"} and inputs["schema"] == SCHEMA
                and type(inputs["models"]) is list and len(inputs["models"]) == 2, "fixed input schema")
        selected = inputs["models"]
        require([x.get("expected_mlp_blocks") for x in selected] == [2, 3], "five-block full population")
        for item in selected:
            require(set(item) == {"model_path", "model_sha256", "graph_path", "graph_sha256", "expected_mlp_blocks"},
                    "fixed source descriptor schema")
            require(input_ids.get(item["model_path"]) == item["model_sha256"]
                    and source_ids.get(item["graph_path"]) == item["graph_sha256"], "selected source identity")
        require(len({x["model_path"] for x in selected}) == 2, "duplicate original model")
        verify_all(input_ids, meter)
        result["input_precheck_complete"] = True
        result["provenance_before"] = provenance(meter)
        result["registration"] = dict(prior_manifest=str(PRIOR), prior_manifest_sha256=PRIOR_SHA,
                                       freeze_sha256=freeze_digest, source_count=7327, input_count=14,
                                       selected_sources=selected, cpu_affinity=[0], address_space_bytes=AS_CAP,
                                       whole_work_cap=WORK_CAP, model_work_cap=BRANCH_CAP,
                                       evidence_work_cap=EVIDENCE_CAP, entry_cap=ENTRY_CAP,
                                       rational_bit_cap=BITS, dyadic_bits=64,
                                       inner_deadline_s=58, outer_deadline_s=60,
                                       arithmetic_work_includes_all_five_blocks=True,
                                       evidence_pool_prepaid=True, model_work_cap_is_cumulative=True,
                                       hash_unit="16 work units prepaid per bounded 64-KiB buffer operation",
                                       final_summary_reserve_units=RESERVE)
        spec = importlib.util.find_spec("onnx")
        require(spec is not None and spec.origin and str(Path(spec.origin).resolve()) in source_ids,
                "ONNX discovery outside authenticated source population")
        import onnx
        require(str(Path(onnx.__file__).resolve()) in source_ids, "ONNX module identity")
        result["dependency_versions"] = {name: importlib.metadata.version(name) for name in ("onnx", "protobuf", "numpy")}
        result["python_version"] = sys.version
        for index, item in enumerate(selected):
            model_work_start = meter.branch
            report = dict(index=index, source=item, diagnostic_complete=False)
            result["models"].append(report)
            try:
                raw = authenticated_bytes(item["model_path"], item["model_sha256"], MODEL_CAP, meter)
                inventory = json_read(item["graph_path"], item["graph_sha256"], meter, GRAPH_CAP)
                inspect_model(onnx, raw, inventory, item, index, report, meter, artifacts)
                del raw, inventory
            except (Deadline, MemoryError) as error:
                report["failure"] = type(error).__name__ + ": " + str(error)
                raise
            except Exception as error:
                report["failure"] = type(error).__name__ + ": " + str(error)
            finally:
                report["model_work_used"] = meter.branch - model_work_start
                name = "model_" + str(index) + ".json"
                if "failure" in report:
                    save(name, report)  # Explicitly bounded pre-reserved failure summary.
                else:
                    save(name, report, meter)
                artifacts.append(name)
        verify_all(source_ids, meter)
        verify_all(input_ids, meter)
        verify_all(fresh, meter)
        require(digest(PRIOR, meter) == PRIOR_SHA and digest(HERE / "freeze.json", meter) == freeze_digest,
                "registration drift")
        result["source_postcheck_complete"] = result["input_postcheck_complete"] = True
        result["provenance_after"] = provenance(meter)
        meter.check()
        require(len(result["models"]) == 2 and all(x["diagnostic_complete"] for x in result["models"])
                and sum(len(x["blocks"]) for x in result["models"]) == 5, "incomplete fixed five-block diagnostic")
        result["diagnostic_complete"] = True
        status = 0
    except BaseException as error:
        result["failure"] = type(error).__name__ + ": " + str(error)
    finally:
        signal.alarm(0)
        result["host_observations"] = meter.snapshot()
        view = result["host_observations"]
        if (view["rss_high_water_growth_bytes"] + RESERVE > MEMORY_CAP
                or view["traced_peak_bytes"] + view["tracer_metadata_bytes"] + RESERVE > MEMORY_CAP):
            result["diagnostic_complete"] = False
            result.setdefault("failure", "host-memory observation gate failed")
            status = 1
        result["exit_status"] = status
        save("result.json", result)
        artifacts.append("result.json")
        # Include a partially written evidence JSON if a deadline interrupted serialization.
        hashes = {path.name: digest(path) for path in sorted(RUN.glob("*.json"))}
        save("exit.json", dict(schema=SCHEMA, status=status, diagnostic_complete=result["diagnostic_complete"],
                               artifact_sha256=hashes, formal_gain=0, candidate_executed=False,
                               source_component_qualified=False, mathematical_component_gate_passed=False,
                               native_HZ_admitted=False, gpu_computation_completed=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
