"""Explicit versioned compatibility wrapper; original backend stays clean."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time
from metamoe_las_repair import install
from recent_moe_deployment import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',type=Path,required=True)
    p.add_argument('--events',type=Path,required=True)
    p.add_argument('--request-config',type=Path,required=True)
    p.add_argument('--config',required=True)
    a=p.parse_args()
    with a.events.open('x') as f:
        seq=0
        def emit(kind,data):
            nonlocal seq
            f.write(json.dumps({'event':kind,'seq':seq,'monotonic':time.monotonic(),
                    'request_config_sha256':sha256(a.request_config),**data},allow_nan=False)+'\n')
            f.flush();seq+=1
        sys.path.insert(0,str(a.backend/'complete_verifier'))
        from branching_domains import BatchedDomainList
        from beta_CROWN_solver import LiRPANet
        install(BatchedDomainList,LiRPANet,emit)
        emit('INSTALLED',{'repair':'disconnected_zero_branching_metadata_v1','bounds_unchanged':True})
        sys.argv=[str(a.backend/'complete_verifier/abcrown.py'),'--config',a.config]
        runpy.run_path(sys.argv[0],run_name='__main__')


if __name__=='__main__':main()
