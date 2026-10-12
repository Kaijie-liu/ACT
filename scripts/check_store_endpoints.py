"""Portable CLI. With kit in sibling scoped_source, requires only python -I -S."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from scoped_source.check_store_endpoints import EndpointWork, check_directory
from scoped_source.hz_request_store import compact

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('--contract',required=True)
    p.add_argument('--proof',required=True); args=p.parse_args()
    started=time.monotonic(); work=EndpointWork(started+300)
    result=check_directory(args.root,expected_contract=args.contract,expected_proof=args.proof,work=work)
    print(compact(dict(result=result,counts=work.counts,check_seconds=time.monotonic()-started)).decode())
