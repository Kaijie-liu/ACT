"""Device-parametric proposals for V1 given-HZ support; exact acceptance unchanged.

CUDA requires an externally anchored execution context and a fresh owned worker.
That context is NOT resource admission. Hardware execution still needs a separate
admitted, hard-budget protocol. No production dispatch or native fallback here.
"""
import math
import os
import re
import time

import torch

from act.back_end.moe.check_batched_support import validated_records
from scoped_source.rowwise_bound import clock, identity, rational
from scoped_source.rowwise_native import _evaluate

ALLOCATOR_LIMIT = 2**30
CONTEXT_SCHEMA = "HZ_DEVICE_EXECUTION_CONTEXT_V1"
COST_SCOPE = "proposal API only; caller must charge HZ preparation, independent checking, publication and cleanup"


def validate_device(device, context, expected_context_sha256, batch_sha256, deadline):
    """Reject before CUDA initialization; resource admission remains the caller's job."""
    tick = clock(deadline)
    if device == "cpu":
        if context is not None or expected_context_sha256 is not None:
            raise ValueError("CPU must not carry CUDA execution context")
        return None
    if device != "cuda:0": raise ValueError("explicit cpu or isolated cuda:0 required")
    required = {"schema", "batch_sha256", "gpu_uuid", "deadline", "allocator_limit_bytes", "invocation"}
    if (type(context) is not dict or set(context) != required
            or type(expected_context_sha256) is not str or len(expected_context_sha256) != 64
            or identity(context) != expected_context_sha256
            or context["schema"] != CONTEXT_SCHEMA or context["batch_sha256"] != batch_sha256
            or type(context["deadline"]) not in (int, float) or context["deadline"] != deadline
            or type(context["allocator_limit_bytes"]) is not int or context["allocator_limit_bytes"] != ALLOCATOR_LIMIT
            or type(context["invocation"]) is not str or not context["invocation"]):
        raise ValueError("bound CUDA execution context required")
    gpu_uuid = context["gpu_uuid"]
    uuid_pattern = r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    if (type(gpu_uuid) is not str or re.fullmatch(uuid_pattern, gpu_uuid) is None
            or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid):
        raise ValueError("single UUID-isolated CUDA worker required")
    tick()
    return gpu_uuid


def _initialize_cuda(gpu_uuid):
    if torch.cuda.is_initialized(): raise ValueError("fresh CUDA worker required; initialization is charged")
    torch.cuda.init()
    if torch.cuda.device_count() != 1: raise ValueError("one visible CUDA device required")
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    if str(props.uuid) != gpu_uuid: raise ValueError("actual CUDA UUID mismatch")
    torch.cuda.memory.set_per_process_memory_fraction(ALLOCATOR_LIMIT / props.total_memory, 0)
    torch.cuda.reset_peak_memory_stats(0)
    return {"gpu_uuid": str(props.uuid), "name": props.name, "total_memory": props.total_memory}


def _synchronize(device):
    if device == "cuda:0": torch.cuda.synchronize(0)


def _readback(y, t, device):
    # Explicit blocking D2H, followed by a device-wide completed observation.
    host_y, host_t = y.T.to(device="cpu", non_blocking=False), t.T.to(device="cpu", non_blocking=False)
    _synchronize(device)
    return host_y.tolist(), host_t.tolist()


def _candidate_columns(base, objectives, constants, *, device, gpu_uuid, deadline, costs):
    tick = clock(deadline)
    if torch.get_num_threads() != 1: raise ValueError("one CPU thread required")
    begin = time.monotonic()

    def tensor(values):
        value = torch.tensor(values, dtype=torch.float64, device="cpu")
        if not bool(torch.isfinite(value).all()): raise ValueError("nonfinite tensor conversion")
        tick()
        return value

    def matrix(name):
        value = base[name]
        rows = [r for r in range(value["shape"][0])
                for _ in range(value["indptr"][r], value["indptr"][r+1])]
        indices = torch.tensor([rows, value["indices"]], dtype=torch.int64, device="cpu")
        data = tensor([float(rational(v)) for v in value["data"]])
        return torch.sparse_coo_tensor(indices, data, tuple(value["shape"]),
                                      dtype=torch.float64, device="cpu").coalesce()

    a, e = matrix("A"), matrix("E")
    c, d = tensor(objectives).T.contiguous(), tensor(constants)
    b, h = tensor(base["b"])[:, None], tensor(base["h"])[:, None]
    low, high = tensor(base["lower"])[:, None], tensor(base["upper"])[:, None]
    costs["host_tensors"] = time.monotonic()-begin
    begin = time.monotonic()
    hardware = _initialize_cuda(gpu_uuid) if device == "cuda:0" else None
    _synchronize(device); tick()
    costs["device_initialization_sync"] = time.monotonic()-begin
    begin = time.monotonic()
    a, e, c, d, b, h, low, high = [v.to(device=device, non_blocking=False) for v in (a,e,c,d,b,h,low,high)]
    at, et = a.transpose(0,1).coalesce(), e.transpose(0,1).coalesce()
    k = c.shape[1]
    y = torch.zeros((a.shape[0],k), dtype=torch.float64, device=device)
    t = torch.zeros((e.shape[0],k), dtype=torch.float64, device=device)
    best_y, best_t = y.clone(), t.clone()
    best = torch.full((k,), -math.inf, dtype=torch.float64, device=device)
    _synchronize(device); tick()
    costs["transfer_and_setup_sync"] = time.monotonic()-begin
    begin = time.monotonic()
    with torch.no_grad():
        for iteration in range(129):
            tick()
            r = c - torch.sparse.mm(at,y) - torch.sparse.mm(et,t)
            value = d+(b*y).sum(0)+(h*t).sum(0)+torch.minimum(r*low,r*high).sum(0)
            if not bool(torch.isfinite(value).all() and torch.isfinite(r).all()
                        and torch.isfinite(y).all() and torch.isfinite(t).all()):
                raise ValueError("nonfinite candidate iteration")
            improved = value > best
            best = torch.where(improved,value,best)
            best_y[:,improved], best_t[:,improved] = y[:,improved], t[:,improved]
            if iteration == 128: break
            argmin = torch.where(r>0,low,torch.where(r<0,high,(low+high)/2))
            step = 0.125/math.sqrt(iteration+1)
            y = torch.minimum(y+step*(b-torch.sparse.mm(a,argmin)),torch.zeros_like(y,device=device))
            t = t+step*(h-torch.sparse.mm(e,argmin))
    _synchronize(device); tick()
    costs["optimization_sync"] = time.monotonic()-begin
    begin = time.monotonic()
    ys, ts = _readback(best_y,best_t,device)
    tick()
    memory = ({"peak_allocated":torch.cuda.max_memory_allocated(0),
               "peak_reserved":torch.cuda.max_memory_reserved(0)} if device=="cuda:0" else None)
    tick()
    costs["readback_sync"] = time.monotonic()-begin
    return ys, ts, hardware, memory


def propose_batch(batch, *, expected_batch_sha256, deadline, device="cpu",
                  cuda_context=None, expected_context_sha256=None):
    """Return untrusted proposals only; never turn an exception into a partial success."""
    start = time.monotonic(); tick = clock(deadline)
    gpu_uuid = validate_device(device,cuda_context,expected_context_sha256,expected_batch_sha256,deadline)
    records = validated_records(batch,expected_batch_sha256=expected_batch_sha256,deadline=deadline)
    objectives = [[float(rational(v)) for v in lp["c"]] for _,lp in records]
    constants = [float(rational(lp["offset"])) for _,lp in records]
    costs = {"validation":time.monotonic()-start}
    ys, ts, hardware, memory = _candidate_columns(batch["base"],objectives,constants,
        device=device,gpu_uuid=gpu_uuid,deadline=deadline,costs=costs)
    if len(ys)!=len(records) or len(ts)!=len(records): raise ValueError("partial candidate columns")
    begin = time.monotonic(); entries=[]
    for (query,lp),y,t in zip(records,ys,ts):
        tick()
        candidate={"lp_sha256":identity(lp),"inequality_dual":y,"equality_dual":t}
        proposed=_evaluate(lp,candidate,tick)
        zero={"lp_sha256":identity(lp),"inequality_dual":[0]*len(lp["b"]),"equality_dual":[0]*len(lp["h"])}
        zero_value=_evaluate(lp,zero,tick)
        if zero_value>proposed: candidate,proposed=zero,zero_value
        candidate["claimed_lower_bound"]=str(proposed)
        entries.append({"id":query["id"],"certificate":candidate,"zero_candidate_lower_bound":str(zero_value)})
    if identity(batch)!=expected_batch_sha256: raise ValueError("batch changed during proposal")
    if cuda_context is not None and identity(cuda_context)!=expected_context_sha256:
        raise ValueError("execution context changed during proposal")
    tick()
    costs["exact_candidate_evaluation"] = time.monotonic()-begin
    elapsed=time.monotonic()-start
    costs["other"] = elapsed-sum(costs.values())
    costs["total"] = elapsed
    tick()
    return {"batch_sha256":expected_batch_sha256,"entries":entries,
            "algorithm":"projected_dual_subgradient_multiobjective_v1","iterations":128,
            "dtype":"float64","device":device,"execution_context_sha256":expected_context_sha256,
            "hardware":hardware,"allocator_memory":memory,"cost_seconds":costs,
            "cost_scope":COST_SCOPE,
            "hard_budget_supervision":False}
