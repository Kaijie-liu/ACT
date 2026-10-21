"""Owned native-solve diagnostics, not solver certificates (stdlib receiver).

The receiver checks identity and representation, not LP feasibility, optimality,
network lowering, or route coverage. The caller supplies the expected scope.
"""
import base64
import hashlib
import json
import math
import struct
from dataclasses import dataclass


SCHEMA = "NATIVE_F0_DIAGNOSTIC_CAPTURE_V1"
GRADE = "CAPTURED_NUMERICAL_PROBLEM_NOT_CHECKED_PROOF"
MAX_BYTES = 128 * 2**20


class CaptureError(ValueError):
    pass


class CaptureDeadline(CaptureError, TimeoutError):
    pass


class CaptureCapacity(CaptureError):
    pass


def require(ok, reason):
    if not ok:
        raise CaptureError(reason)


@dataclass(frozen=True)
class CapturedSolve:
    payload: bytes
    sha256: str


class CaptureLedger:
    """One request/stage-owned byte ledger, sharing its existing Work/deadline."""
    def __init__(self, work, byte_limit):
        require(type(byte_limit) is int and 0 < byte_limit <= MAX_BYTES, "capture byte limit")
        self.work, self.byte_limit, self.used = work, byte_limit, 0
        self.tick()

    def tick(self):
        try:
            self.work.tick()
        except TimeoutError as exc:
            raise CaptureDeadline(str(exc)) from exc

    def visit(self, amount):
        self.tick()
        try:
            self.work.visit("coefficient_visits", amount)
        except TimeoutError as exc:
            raise CaptureDeadline(str(exc)) from exc

    def reserve(self, upper):
        self.tick()
        if type(upper) is not int or upper < 0 or self.used + upper > self.byte_limit:
            raise CaptureCapacity("cumulative capture bytes including pending record")

    def accept(self, size):
        self.reserve(size)
        self.used += size


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def bounded_metadata(value):
    """Bound traversal and encoded allocation before JSON copies caller data."""
    remaining = 16384
    nodes = 0
    def visit(v, depth=0):
        nonlocal remaining, nodes
        nodes += 1
        require(depth <= 12 and nodes <= 512, "metadata depth/nodes")
        if type(v) is str:
            size = 12*len(v)+2  # supplementary characters use two Unicode escapes
        elif v is None or type(v) is bool:
            size = 5
        elif type(v) is int:
            require(v.bit_length() <= 256, "metadata integer")
            size = 80
        elif type(v) is float:
            require(math.isfinite(v), "metadata finite float")
            size = 32
        elif type(v) in (list, dict):
            require(len(v) <= 512, "metadata container")
            remaining -= 2 + 2*len(v)
            require(remaining >= 0, "metadata encoded upper bound")
            if type(v) is dict:
                for key, item in v.items():
                    require(type(key) is str, "metadata key")
                    visit(key, depth+1); visit(item, depth+1)
            else:
                for item in v:
                    visit(item, depth+1)
            return
        else:
            raise CaptureError("metadata type")
        remaining -= size
        require(remaining >= 0, "metadata encoded upper bound")
    require(type(value) is dict, "metadata object")
    visit(value)
    raw = canonical(value)
    require(len(raw) <= 16384, "metadata actual encoded bytes")
    return raw


def checked_scope(scope):
    require(type(scope) is dict and set(scope) == {
        "request", "config", "model", "domain", "roster", "path", "ordinal",
        "property_index", "pairs", "q", "constant",
    }, "complete diagnostic scope")
    for name in ("request", "config", "model", "domain", "roster"):
        value = scope[name]
        require(type(value) is str and len(value) == 64 and
                all(c in "0123456789abcdef" for c in value), "scope hash: " + name)
    require(scope["path"] in ("pair", "monolithic"), "capture path")
    require(all(type(scope[k]) is int and scope[k] >= 0 for k in ("ordinal", "property_index")),
            "capture ordinal")
    pairs = scope["pairs"]
    require(type(pairs) is list and 1 <= len(pairs) <= 28 and all(
        type(p) is list and len(p) == 2 and all(type(i) is int and i >= 0 for i in p)
        and p[0] < p[1] for p in pairs), "canonical pair roster")
    require(len({tuple(p) for p in pairs}) == len(pairs), "duplicate pair")
    require(scope["path"] != "pair" or len(pairs) == 1, "pair path roster")
    require(type(scope["q"]) is list and 1 <= len(scope["q"]) <= 256, "full property shape")
    require(all(type(v) in (float, int) and math.isfinite(v)
                for v in scope["q"] + [scope["constant"]]), "finite property")
    raw = canonical(scope)
    require(len(raw) <= 16384, "scope size")
    return raw


def read_array(value, work):
    require(type(value) is dict and set(value) == {"dtype", "count", "base64"}, "array fields")
    require(value["dtype"] in ("<f8", "<i8") and type(value["count"]) is int
            and 0 <= value["count"] <= MAX_BYTES // 8, "array dtype/count")
    count = value["count"]
    encoded = value["base64"]
    require(type(encoded) is str and len(encoded) == 4 * ((count*8+2)//3), "encoded array size")
    work.tick()
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise CaptureError("invalid array base64") from exc
    require(len(raw) == count*8 and base64.b64encode(raw).decode() == encoded, "canonical buffer")
    work.visit("coefficient_visits", count)
    fmt = "<d" if value["dtype"] == "<f8" else "<q"
    result = tuple(x[0] for x in struct.iter_unpack(fmt, raw))
    work.tick()
    return result


def receive(record, *, expected_sha256, expected_scope, work):
    """Structural identity reception only; no candidate is accepted as SAFE."""
    try:
        return _receive(record, expected_sha256=expected_sha256,
                        expected_scope=expected_scope, work=work)
    except CaptureError:
        raise
    except TimeoutError as exc:
        raise CaptureDeadline(str(exc)) from exc
    except (KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
        raise CaptureError("malformed capture: " + type(exc).__name__) from exc


def _receive(record, *, expected_sha256, expected_scope, work):
    work.tick()
    require(type(record) is bytes and len(record) <= MAX_BYTES, "owned capture payload")
    require(hashlib.sha256(record).hexdigest() == expected_sha256, "external capture hash")
    work.visit("coefficient_visits", (len(record)+4095)//4096)
    value = json.loads(record)
    require(type(value) is dict and set(value) == {
        "schema", "grade", "scope", "event", "native_called", "native", "arrays",
        "matrices", "sources", "source_metadata", "result", "candidate", "deadline",
    }, "capture fields")
    require(canonical(value) == record and value["schema"] == SCHEMA and value["grade"] == GRADE,
            "canonical capture schema/grade")
    require(checked_scope(value["scope"]) == checked_scope(expected_scope), "external scope binding")
    require(type(value["deadline"]) in (int, float) and math.isfinite(value["deadline"])
            and value["deadline"] == work.deadline, "same online deadline")
    require(type(value["native_called"]) is bool, "native call flag")
    event = value["event"]
    failure = (event == "SCIPY_UNAVAILABLE" or (type(event) is str and
               any(event.startswith(p) and event[len(p):].isidentifier()
                   for p in ("LOWERING_FAILED:", "SOLVER_FAILED:"))))
    require(event == "RETURNED" or failure, "capture event")
    if failure:
        require(value["native_called"] == event.startswith("SOLVER_FAILED:"), "failure call flag")
    require(type(value["arrays"]) is dict and len(value["arrays"]) <= 2048
            and type(value["matrices"]) is dict and len(value["matrices"]) <= 1024
            and type(value["sources"]) is dict and len(value["sources"]) <= 64, "capture catalogs")
    bounded_metadata(value["source_metadata"])
    arrays = {k: read_array(v, work) for k, v in value["arrays"].items()}
    used_arrays, used_matrices = set(), set()
    def vector(name, length, dtype="<f8", finite=True):
        require(name in arrays and value["arrays"][name]["dtype"] == dtype,
                "vector reference/dtype: " + name)
        result = arrays[name]
        require(len(result) == length, "vector length: " + name)
        used_arrays.add(name)
        work.visit("coefficient_visits", len(result))
        if dtype == "<f8" and finite:
            require(all(math.isfinite(v) for v in result), "finite vector: " + name)
        return result
    def shape(name, expected):
        require(name in value["matrices"] and value["matrices"][name]["shape"] == list(expected),
                "matrix reference/shape: " + name)
        used_matrices.add(name)
    def nonnegative_int(v):
        return type(v) is int and v >= 0
    for name, mat in value["matrices"].items():
        work.tick()
        require(type(name) is str and type(mat) is dict and
                set(mat) == {"shape", "data", "indices", "indptr"}, "CSR fields")
        require(type(mat["shape"]) is list and len(mat["shape"]) == 2, "CSR rank")
        nr, nc = mat["shape"]
        require(nonnegative_int(nr) and nonnegative_int(nc), "CSR shape")
        data, idx, ptr = (arrays[mat[k]] for k in ("data", "indices", "indptr"))
        require(value["arrays"][mat["data"]]["dtype"] == "<f8" and all(
            value["arrays"][mat[k]]["dtype"] == "<i8" for k in ("indices", "indptr")), "CSR dtypes")
        require(len(ptr) == nr+1 and ptr[0] == 0 and ptr[-1] == len(data) == len(idx), "CSR lengths")
        require(all(0 <= a <= b <= len(data) for a, b in zip(ptr, ptr[1:])), "CSR pointer order")
        require(all(0 <= j < nc for j in idx) and all(math.isfinite(v) for v in data), "CSR columns/data")
        used_arrays.update(mat[k] for k in ("data", "indices", "indptr"))
        work.visit("coefficient_visits", len(data)+len(idx)+len(ptr))
    for name, source in value["sources"].items():
        require(type(name) is str and type(source) is dict and set(source) == {
            "frame_id", "exact_flag", "n_cont", "n_bin", "n_out"}, "source fields")
        require(type(source["frame_id"]) is int and type(source["exact_flag"]) is bool
                and all(nonnegative_int(source[k]) for k in ("n_cont", "n_bin", "n_out")),
                "source dimensions/frame")
        nc, nb, no = (source[k] for k in ("n_cont", "n_bin", "n_out"))
        vector(name+".c", no)
        ne, nu = (len(arrays[name+"."+k]) for k in ("b", "ub"))
        vector(name+".b", ne); vector(name+".ub", nu)
        for key, dims in (("Gc", (no, nc)), ("Gb", (no, nb)), ("Ac", (ne, nc)),
                          ("Ab", (ne, nb)), ("Auc", (nu, nc)), ("Aub", (nu, nb))):
            shape(name+"."+key, dims)
    def lowered(name, n, no=None):
        rows = value["matrices"][name+".A"]["shape"][0]
        shape(name+".A", (rows, n))
        for lo, hi, length, finite in (("var_lb", "var_ub", n, True),
                                      ("row_lb", "row_ub", rows, False)):
            l = vector(name+"."+lo, length, finite=finite)
            u = vector(name+"."+hi, length, finite=finite)
            require(all(not math.isnan(a) and not math.isnan(b) and a <= b for a, b in zip(l, u)),
                    "ordered native bounds")
        flags = vector(name+".integrality", n, dtype="<i8")
        require(all(v in (0, 1) for v in flags), "binary/continuous flags")
        if no is not None:
            vector(name+".value_center", no)
            shape(name+".value_matrix", (no, n))
        return flags
    def source_map(name, nc, nb):
        source = value["sources"][name]
        require(source["n_cont"] == nc and source["n_bin"] == nb, "source factor dimensions")
        return source["n_out"]
    def stored_row(name, row, n):
        # Binary64 reconstruction of stored CSR duplicates, not exact lowering proof.
        mat = value["matrices"][name]
        ptr, idx, data = (arrays[mat[k]] for k in ("indptr", "indices", "data"))
        result = [0.0]*n
        for p in range(ptr[row], ptr[row+1]):
            work.visit("coefficient_visits")
            result[idx[p]] += data[p]
        return tuple(result)
    native = value["native"]
    require((native is not None) == (event == "RETURNED"), "returned problem required")
    if event == "RETURNED":
        n = native["variables"]
        require(nonnegative_int(n), "variable count")
        require(value["native_called"] == (value["scope"]["path"] == "monolithic" or n > 0),
                "native call count/path binding")
        objective = vector("native.objective", n)
        if value["scope"]["path"] == "pair":
            require(set(native) == {"variables", "n_cont", "n_bin", "objective_row",
                    "objective_offset", "binary_map"}, "pair native fields")
            nc, nb = native["n_cont"], native["n_bin"]
            require(nonnegative_int(nc) and nonnegative_int(nb) and nc+nb == n
                    and native["binary_map"] == "beta=2*z-1",
                    "pair factor map")
            no = source_map("f0", nc, nb)
            flags = lowered("native", n, no)
            require(flags == (0,)*nc+(1,)*nb, "pair integrality map")
            r = native["objective_row"]
            require(nonnegative_int(r) and r < no and
                    native["objective_offset"] == arrays["native.value_center"][r].hex()
                    and objective == stored_row("native.value_matrix", r, n), "pair objective row/offset")
            if "input" in value["sources"]:
                require(value["sources"]["input"]["n_cont"] <= nc and
                        value["sources"]["input"]["n_bin"] <= nb, "input prefix dimensions")
        else:
            require(set(native) == {"variables", "objective_offset", "selector_offset", "local_offsets",
                    "branches", "binary_map"}, "monolithic native fields")
            require(native["binary_map"] == "local variables are selector-homogenized",
                    "homogenized local variable map")
            offset = native["selector_offset"]
            count = len(expected_scope["pairs"])
            require(native["objective_offset"] == "0x0.0p+0" and
                    nonnegative_int(offset) and len(native["local_offsets"]) == count
                    and len(native["branches"]) == count and offset+count == n, "monolithic selector map")
            flags = lowered("native", n)
            require(flags[offset:] == (1,)*count and
                    arrays["native.var_lb"][offset:] == (0,)*count and
                    arrays["native.var_ub"][offset:] == (1,)*count, "selector bounds/flags")
            cursor = 0
            for i, branch in enumerate(native["branches"]):
                require(set(branch) == {"pair", "n_cont", "n_bin"}
                        and branch["pair"] == expected_scope["pairs"][i]
                        and native["local_offsets"][i] == cursor, "branch pair/offset")
                nc, nb = branch["n_cont"], branch["n_bin"]
                require(nonnegative_int(nc) and nonnegative_int(nb), "branch dimensions")
                width = nc+nb
                require(source_map("f0_%d" % i, nc, nb) == 1, "scalar F0 branch")
                inp = value["sources"]["input_%d" % i]
                require(inp["n_cont"] <= nc and inp["n_bin"] <= nb, "branch input prefix dimensions")
                label = "branch%d" % i
                local_flags = lowered(label, width, 1)
                require(local_flags == (0,)*nc+(1,)*nb and flags[cursor:cursor+width] == local_flags,
                        "branch integrality map")
                require(objective[cursor:cursor+width] == stored_row(label+".value_matrix", 0, width)
                        and objective[offset+i] == arrays[label+".value_center"][0], "branch objective map")
                cursor += width
            require(cursor == offset, "branch variables cover local prefix")
    else:
        require(not arrays and not value["matrices"] and not value["sources"], "failure has no problem")
    require(value["candidate"] in ("RAW_UNCHECKED", "MISSING"), "candidate grade")
    if value["candidate"] == "RAW_UNCHECKED":
        require(native is not None, "candidate needs native problem")
        vector("candidate", native["variables"], finite=False)
    else:
        require("candidate" not in arrays, "missing candidate array")
    result = value["result"]
    require(type(result) is dict and set(result) == {
        "status", "mip_node_count", "fun", "mip_dual_bound", "mip_gap"}, "native result fields")
    for k in ("status", "mip_node_count"):
        require(result[k] is None or type(result[k]) is int, "result integer")
    for k in ("fun", "mip_dual_bound", "mip_gap"):
        require(result[k] is None or (type(result[k]) is str and float.fromhex(result[k]).hex() == result[k]),
                "raw binary64 result scalar")
    require(used_arrays == set(arrays) and used_matrices == set(value["matrices"]), "no unbound buffers")
    work.tick()
    return {"schema": SCHEMA, "grade": GRADE, "sha256": expected_sha256,
            "scope": value["scope"], "native_called": value["native_called"],
            "event": value["event"], "arrays": len(arrays), "matrices": len(value["matrices"])}
