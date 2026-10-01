"""Fixed CPU controls/archive for device candidates; never run physical CUDA."""
import argparse
import ast
import io
import math
import os
from pathlib import Path
import shutil
import sys
import time
import unittest

from scoped_proof.io import ROOT, load, save, sha

CONFIG="configs/hz_device_candidates_20261001.json"
PROTOCOL_SHA="90ff1dc2ce9c20cf6b3bae19c04c7743ce1783fdc1b5b628603362681fbaa386"
TEST="act/back_end/moe/test_batched_support_device.py"
FILES=(CONFIG,TEST,"scripts/run_hz_device_candidates.py","act/back_end/moe/batched_support_device.py",
       "act/back_end/moe/batched_support.py","act/back_end/moe/check_batched_support.py",
       "scripts/hz_batch_support_worker.py","scripts/hz_batch_support_supervised.py",
       "act/back_end/solver/solver_hz.py","act/back_end/solver/hz_lp_export.py",
       "act/back_end/solver/check_hz_lp_export.py","act/back_end/solver/lp_certificate.py",
       "scoped_source/rowwise_bound.py","scoped_source/rowwise_native.py")


def protocol():
    if sha(ROOT/CONFIG)!=PROTOCOL_SHA: raise ValueError("frozen control protocol changed")
    config=load(ROOT/CONFIG)
    if config["cuda_execution_authorized"] is not False or config["devices_executed"]!=["cpu"]:
        raise ValueError("this entry never admits CUDA")
    for name,digest in config["frozen_dependencies"].items():
        if sha(ROOT/name)!=digest: raise ValueError("frozen dependency changed: "+name)
    return config


def test_names():
    cls=next(n for n in ast.parse((ROOT/TEST).read_text()).body if isinstance(n,ast.ClassDef) and n.name=="DeviceCandidateControls")
    return sorted(n.name for n in cls.body if isinstance(n,ast.FunctionDef) and n.name.startswith("test_"))


def expected_batch(case):
    """Recreate only the fixed tiny HZ identity, never optimize a candidate."""
    from act.back_end.moe.test_batched_support_device import source
    from act.back_end.moe.batched_support import prepare_batch
    hz,queries,context=source(case)
    return prepare_batch(hz,queries,context=context,deadline=time.monotonic()+30)


def inventory(summary,bindings,observations):
    required=set(protocol()["normal_cases"])|{"device_failures_stub","sync_failure_stub","late_readback_stub","partial_rejected"}
    if (set(bindings)!=set(FILES) or set(observations)!=required or summary["test_names"]!=test_names()
            or summary["tests"]!=len(test_names()) or summary["failures"] or summary["errors"] or summary["skipped"]
            or summary["physical_cuda"] is not False or summary["native_solves"]!=0 or summary["real_requests"]!=0):
        raise ValueError("complete control/implementation/evidence inventory")


def proposal_metadata(item):
    from act.back_end.moe.batched_support_device import COST_SCOPE
    cfg=protocol()
    for key in ("v1","v2"):
        p=item[key]
        if any(p.get(k)!=v for k,v in {"device":"cpu","algorithm":cfg["algorithm"],"iterations":128,"dtype":"float64"}.items()):
            raise ValueError("frozen candidate execution metadata")
    p=item["v2"]
    if (p["hardware"] is not None or p["allocator_memory"] is not None or p["execution_context_sha256"] is not None
            or p["hard_budget_supervision"] is not False or p["cost_scope"]!=COST_SCOPE):
        raise ValueError("CPU/proposal-only scope changed")
    costs=p["cost_seconds"]
    if (set(costs)!={"validation","host_tensors","device_initialization_sync","transfer_and_setup_sync",
                     "optimization_sync","readback_sync","exact_candidate_evaluation","other","total"}
            or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in costs.values())
            or abs(sum(v for k,v in costs.items() if k!="total")-costs["total"])>1e-9):
        raise ValueError("proposal component accounting")


def fault_inventory(obs,guarded_sha):
    from scoped_source.rowwise_bound import identity
    records=obs["device_failures_stub"]["records"]
    if type(records) is not list or len(records)!=2 or {r["error"] for r in records}!={"RuntimeError","OutOfMemoryError"}:
        raise ValueError("both device fault simulations required")
    if obs["sync_failure_stub"]["boundaries"] != ["cpu"]*3: raise ValueError("synchronization fault not reached")
    for item in records+[obs["sync_failure_stub"],obs["late_readback_stub"]]:
        if item["injection_reached"] is not True or item["physical_cuda"] is not False: raise ValueError("fault simulation scope")
    item=obs["partial_rejected"]
    if (identity(item["batch"])!=guarded_sha or item["candidates"]["batch_sha256"]!=guarded_sha
            or (item["required"],item["received"],len(item["batch"]["queries"]),len(item["candidates"]["entries"]))!=(4,3,4,3)
            or [r["id"] for r in item["candidates"]["entries"]]!=[r["id"] for r in item["batch"]["queries"][:3]]):
        raise ValueError("fixed partial source/denominator")


def run(root):
    protocol(); root.mkdir(parents=True,exist_ok=False)
    bindings={}
    for name in FILES:
        dst=root/"implementation"/name; dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,dst); bindings[name]=sha(dst)
    save(root/"implementation.json",bindings)
    start=time.monotonic()
    from act.back_end.moe.test_batched_support_device import DeviceCandidateControls
    import torch
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DeviceCandidateControls))
    (root/"tests.log").write_text(stream.getvalue())
    save(root/"observations.json",DeviceCandidateControls.observations)
    summary={"status":"PASS" if result.wasSuccessful() else "FAIL","tests":result.testsRun,
             "test_names":test_names(),"failures":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),
             "protocol_sha256":sha(ROOT/CONFIG),"implementation_sha256":sha(root/"implementation.json"),
             "observations_sha256":sha(root/"observations.json"),"log_sha256":sha(root/"tests.log"),
             "physical_cuda":torch.cuda.is_initialized(),"native_solves":0,"real_requests":0,
             "software":{"python":sys.version,"torch":torch.__version__,"cuda_build":torch.version.cuda},
             "seconds_with_imports":time.monotonic()-start}
    save(root/"summary.json",summary)
    print(stream.getvalue()); print(summary)
    return 0 if result.wasSuccessful() else 1


def audit(root):
    summary=load(root/"summary.json"); bindings=load(root/"implementation.json"); obs=load(root/"observations.json")
    inventory(summary,bindings,obs)
    if (summary["status"]!="PASS" or summary["protocol_sha256"]!=sha(ROOT/CONFIG)
            or summary["implementation_sha256"]!=sha(root/"implementation.json")
            or summary["observations_sha256"]!=sha(root/"observations.json") or summary["log_sha256"]!=sha(root/"tests.log")):
        raise ValueError("execution anchors")
    if type(summary["seconds_with_imports"]) not in (int,float) or not math.isfinite(summary["seconds_with_imports"]) or summary["seconds_with_imports"]<0:
        raise ValueError("invalid control elapsed time")
    for name,digest in bindings.items():
        if sha(ROOT/name)!=digest or sha(root/"implementation"/name)!=digest: raise ValueError("source binding: "+name)
    from act.back_end.moe.check_batched_support import check_batch
    from scoped_source.rowwise_bound import identity
    import torch
    checked={}
    for case in protocol()["normal_cases"]:
        item=obs[case]; batch=item["batch"]; anchor=identity(batch)
        if anchor!=identity(expected_batch(case)): raise ValueError("wrong fixed case/source identity")
        a=check_batch(batch,item["v1"],expected_batch_sha256=anchor,deadline=time.monotonic()+30)
        b=check_batch(batch,item["v2"],expected_batch_sha256=anchor,deadline=time.monotonic()+30)
        if a!=item["checked_v1"] or b!=item["checked_v2"] or a["results"]!=b["results"]:
            raise ValueError("exact saved-evidence differential")
        proposal_metadata(item)
        checked[case]={"bounds":b["results"],"proposal_seconds":item["v2"]["cost_seconds"],
                       "scope":b["status"],"complete_moe_proof":False}
    fault_inventory(obs,identity(expected_batch("guarded_two_sides_positive")))
    item=obs["partial_rejected"]
    try: check_batch(item["batch"],item["candidates"],expected_batch_sha256=identity(item["batch"]),deadline=time.monotonic()+30)
    except ValueError: pass
    else: raise ValueError("partial candidate accepted")
    if torch.cuda.is_initialized(): raise ValueError("archive check initialized CUDA")
    return {"schema":"HZ_DEVICE_CANDIDATE_ARCHIVE_V1","status":"PASS","root":str(root),
            "protocol_sha256":summary["protocol_sha256"],"summary_sha256":sha(root/"summary.json"),
            "implementation_sha256":summary["implementation_sha256"],"tests":summary["tests"],"checked":checked,
            "gpu_executions":0,"native_solves":0,"real_requests":0,"complete_moe_proofs":0,
            "hard_supervision_complete":False,"gpu_compatibility_established":False,
            "seconds_with_imports":summary["seconds_with_imports"]}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("root",type=Path)
    p.add_argument("--check",action="store_true"); p.add_argument("--report",type=Path); a=p.parse_args()
    root=a.root.resolve()
    if not root.is_relative_to(ROOT.parent/"baseline_runs"): raise ValueError("new project archive required")
    if a.check:
        result=audit(root)
        if a.report:
            target=a.report.resolve()
            if target.parent!=ROOT/"docs": raise ValueError("compact repo report only")
            save(target,result)
        print({k:v for k,v in result.items() if k!="checked"})
    else:
        if a.report: raise ValueError("report requires check")
        os.environ["CUDA_VISIBLE_DEVICES"]=""
        raise SystemExit(run(root))
