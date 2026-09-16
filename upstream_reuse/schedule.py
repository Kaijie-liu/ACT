"""Original deterministic schedule; only dependency binding differs (AST controlled)."""
from pathlib import Path
import time
from moe_evidence.generate import read, save, reference, identity, rational, pair_key, gate_envelope
from act.back_end.solver.lp_certificate import propose
from act.back_end.solver.check_hz_lp_export import check_export
from act.back_end.solver.rational_mccormick import build
from act.back_end.solver.check_rational_mccormick import check_construction

def propose_all(root,budget,*,cap=60,reserve=80):




    root=Path(root);m=read(root/'manifest.json');calls=[];values={}
    if not m['generation_complete']:return
    def solve(key,record):
        grant=budget.grant(cap,reserve);start=time.monotonic()
        calls.append({'key':key,'entered_seconds':start-budget.started,'granted_seconds':grant,'status':'PENDING'})
        save(root/'query_log.json',calls)
        try:cert=propose(record['lp'],time_limit=budget.grant(grant,reserve))
        except ValueError as exc:
            if str(exc)!='proposal solver did not complete':raise
            cert=None
        budget.remaining(2)
        if cert:save(root/(key+'.certificate.json'),cert)
        calls[-1].update(status='PROPOSED' if cert else 'UNAVAILABLE',seconds=time.monotonic()-start)
        save(root/'query_log.json',calls)
        return cert,reference(root,key+'.certificate.json') if cert else None
    order=[]
    for pair in sorted(tuple(p) for p in m['routes']['feasible']):
        prefix=pair_key(pair)
        pending=sorted((v for v in m['obligations'] if v['pair']==list(pair) and v['kind']=='residual'),key=lambda v:v['property_index'])
        if not pending:continue
        order.extend((prefix+'_gate_lo',prefix+'_gate_hi'))
        for row in pending:order.extend((prefix+f"_p{row['property_index']}_lo",prefix+f"_p{row['property_index']}_hi"))
    if set(order)!=set(m['supports']):raise ValueError('support schedule mismatch')
    for key in order:
        item=m['supports'][key]
        # Preserve the check reserve and return a checkable incomplete record.
        if budget.deadline-budget.clock() <= reserve+.01:return
        budget.remaining(2);record=read(root/item['export']['file'])
        check_export(record,None,expected_source_sha256=item['source_sha256'])
        cert,ref=solve(key,record)
        checked=check_export(record,cert,expected_source_sha256=item['source_sha256'])
        values[key]=rational(checked['bound']['checked_lower_bound']) if cert else None
        item.update(status='PROPOSED' if cert else 'UNAVAILABLE',certificate=ref);save(root/'manifest.json',m)
    for row in m['obligations']:
        if row['kind']!='residual':continue
        if budget.deadline-budget.clock() <= reserve+.01:return
        budget.remaining(2);key=pair_key(row['pair']);i=row['property_index'];base=key+f'_p{i}'
        lo,neg=values[base+'_lo'],values[base+'_hi']
        if lo is None or neg is None:continue
        if lo>-neg:raise ValueError('inconsistent checked difference')
        gate=gate_envelope(values[key+'_gate_lo'],values[key+'_gate_hi']);difference=[str(lo),str(-neg)]
        src=read(root/m['contexts'][key]['joint_source']['file']);prop=row['property']
        record=build(src,prop['q'],prop['constant'],gate,difference)
        check_construction(record,source_hash=identity(src),q=prop['q'],offset=prop['constant'],gate=gate,difference=difference)
        name=base+'_weighted';save(root/(name+'.export.json'),record);cert,ref=solve(name,record)
        check_construction(record,cert,source_hash=identity(src),q=prop['q'],offset=prop['constant'],gate=gate,difference=difference)
        row.update(weighted_status='PROPOSED' if cert else 'UNAVAILABLE',weighted=reference(root,name+'.export.json'),
                   certificate=ref,gate_bounds=gate,difference_bounds=difference)
        save(root/'manifest.json',m)
