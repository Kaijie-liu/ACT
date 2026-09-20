"""Relocated composition: check old output proof AND original-parameter routes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
STDLIB=tuple(Path(p).resolve() for p in sys.path if p)


def guard(event,args):
    if event=='open' and not isinstance(args[0],int):
        path=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
        if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT):
            raise PermissionError('checker is read-only')
        if not path.is_relative_to(ROOT) and not any(path.is_relative_to(p) for p in STDLIB):
            raise PermissionError('outside relocated bundle/stdlib')
    if event.startswith(('subprocess.','socket.')) or event in ('os.system','os.exec','os.fork'):
        raise PermissionError('external execution/network prohibited')
    if event=='import' and args[0].split('.')[0] in ('torch','numpy','scipy','highspy','gurobipy'):
        raise ImportError('model/solver dependency prohibited')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--extension-hash',required=True)
    args=p.parse_args()
    if not sys.flags.isolated or not sys.flags.no_site:raise ValueError('requires python -I -S')
    start=time.monotonic();sys.addaudithook(guard)
    raw=(ROOT/'extension.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.extension_hash:raise ValueError('extension identity')
    extension=json.loads(raw)
    for name,digest in extension['files'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('extension code/data identity')
    sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT))
    from runtime import verify
    from router_check import check,compact,digest
    bundle=json.loads((ROOT/'bundle.json').read_bytes());statement=bundle['statement']
    doc=json.loads((ROOT/'router_source.json').read_bytes())
    route=check(doc,json.loads((ROOT/'router_proof.json').read_bytes()),
        expected_request=statement['request'],expected_source_sha256=extension['source_identity'])
    if (route['covered_pairs']!=statement['routes']['feasible'] or
            route['excluded_pairs']!=statement['routes']['infeasible'] or statement['routes']['unresolved']):
        raise ValueError('new independent route cover does not discharge old exclusions')
    output=verify(ROOT,extension['old_bundle_sha256'],extension['old_statement_sha256'])
    if (output['required_obligations']!=len(route['covered_pairs'])*(statement['request']['classes']-1) or
            output['positive_obligations']!=output['required_obligations']):
        raise ValueError('not a complete positive output inventory')
    trusted=[v for v in output['trusted_base'] if v!='router_infeasibility_exclusions']
    trusted.append('declared_router_graph_corresponds_to_registered_eval_program')
    print(json.dumps({'status':'CHECKED_OUTPUT_REQUEST_WITH_SOURCE_ROUTER_COVER',
        'route_check':route,'output_check':output,
        'required_obligations':output['required_obligations'],
        'minimum_lower_bound':output['minimum_lower_bound'],
        'remaining_trusted_base':trusted,'router_exclusions_independently_proved':True,
        'complete_strict_network_certificate':False,'production_verdict_changed':False,
        'scope':'Original-parameter real router, complete supplied-HZ output proof; expert lowering/guards and graph correspondence remain trusted. Not deployed floating proof or a route-changing claim.',
        'isolated':True,'site_disabled':True,'check_seconds':time.monotonic()-start},sort_keys=True))


if __name__=='__main__':main()
