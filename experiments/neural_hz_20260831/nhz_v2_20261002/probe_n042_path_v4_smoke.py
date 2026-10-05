import sys, os, csv, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nhz_sound_v9 import SoundEngineV9
from nhz_engine import parse_vnnlib
from nhz_path_v4 import solve_box_v4
import onnxruntime as ort
torch.cuda.set_per_process_memory_fraction(0.12)
csv.field_size_limit(10**9)
ov = {(r['benchmark'], r['iid']): r for r in csv.DictReader(open('/data1/Kane/HyZor/DIST_SHIFT_K3_HEADLINE_UPDATE_20260822/_DETAIL_K3_OVERLAY.csv'))}
ROOT = "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
for item in sys.argv[1].split(","):
    fam, iid = item.rsplit(":", 1); r = ov[(fam, iid)]
    mp = os.path.normpath(os.path.join(ROOT, fam, r['onnx'])); sp = os.path.normpath(os.path.join(ROOT, fam, r['vnnlib']))
    e = SoundEngineV9(mp, 'cuda'); s = ort.InferenceSession(mp, providers=['CPUExecutionProvider'])
    n_in = int(np.prod(e.input_shape)); o0, *_ = e.propagate(np.zeros(n_in), np.zeros(n_in), 0)
    spec = parse_vnnlib(sp, n_in, int(o0.c.numel())); t0 = time.time(); oc = 'CERT'; wit = None; info = {}
    for lb, ub in spec.boxes:
        o, info, wit = solve_box_v4(e, s, s.get_inputs()[0].name, spec, lb, ub, t0 + float(r['csv_timeout']))
        if o != 'CERT': oc = o; break
    wall = time.time() - t0
    if oc in ('CERT', 'ADV') and wall > float(r['csv_timeout']): oc = 'TIMEOUT(over)'
    print(fam, iid, r['raw_verdict'], '->', oc, round(wall, 1), wit[0] if wit else '', {k: (v['status'][:40], round(v['wall_s'], 1)) for k, v in info.get('milp', {}).items()}, flush=True)
