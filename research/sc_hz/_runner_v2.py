
import sys, json, time, resource
sys.path.insert(0, "/data1/Kane/ACT")
resource.setrlimit(resource.RLIMIT_AS, (80 * 1024**3, resource.RLIM_INFINITY))
import numpy as np
from research.canonical_provenance import load_instance, build_provenance
from research.sc_hz.onnx_walker_resnet import forward_resnet
from research.sc_hz.vnnlib_parse import parse_vnnlib
from research.sc_hz.ops import lp_ub_rival_margin
bench, iid, K_target = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
out_path = sys.argv[4]
rec = {"bench": bench, "iid": iid, "K_target": K_target}
t0 = time.perf_counter()
try:
    prov = build_provenance(bench, iid)
    rec.update({
        "canonical_root": str(prov.canonical_root),
        "onnx_sha256": prov.onnx_sha256, "vnnlib_sha256": prov.vnnlib_sha256,
    })
    onnx_p, vnn_p = load_instance(bench, iid)
    if bench.startswith("cifar100"): n_in, n_classes = 3072, 100
    else: n_in, n_classes = 3*56*56, 200
    lb_x, ub_x, unsafe = parse_vnnlib(str(vnn_p), n_in, n_classes)
    result = forward_resnet(str(onnx_p), lb_x, ub_x, K_per_layer=100000,
                              streaming_K_target=K_target, streaming_chunk_size=256)
    rec["wall_s"] = time.perf_counter() - t0
    rec["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024*1024)
    rec["n_processed"] = result.n_nodes_processed
    rec["n_skipped"] = len(result.nodes_skipped)
    rec["output_ng"] = int(result.output_state.G_kept.shape[1])
    rec["tail_norm"] = float(result.output_state.tail_radius.sum() if result.output_state.tail_radius is not None else 0)
    rec["n_unsafe_conditions"] = len(unsafe)
    max_excess = -float("inf"); n_cert_conds = 0
    for d_out, t_thr, _ in unsafe:
        ub = lp_ub_rival_margin(result.output_state, d_out)
        excess = float(ub) - float(t_thr)
        if excess < 0: n_cert_conds += 1
        if excess > max_excess: max_excess = excess
    rec["max_excess"] = float(max_excess)
    rec["n_cert_conditions"] = n_cert_conds
    rec["status"] = "OK"
except MemoryError:
    rec["status"] = "OOM"
    rec["wall_s"] = time.perf_counter() - t0
    rec["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024*1024)
except Exception as e:
    rec["status"] = "ERROR"; rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    rec["wall_s"] = time.perf_counter() - t0
with open(out_path, "w") as f: json.dump(rec, f, indent=2, default=float)
print(f"DONE iid={iid} status={rec['status']} rss={rec.get('peak_rss_gb',0):.1f}GB wall={rec.get('wall_s',0):.0f}s ng={rec.get('output_ng',0)} max_e={rec.get('max_excess','n/a')}")
