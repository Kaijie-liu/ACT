"""Static shared-source ledger. No source/worker/model/native/GPU execution."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONTROLS='docs/hz_shared_source_controls_20261015.json'
SHAPE='docs/hz_full_resource_ledger_20261014.json'
SECOND_SHAPE='docs/hz_pair_scoped_capacity_20261015.json'


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(name): return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
def load(name): return json.loads((ROOT/name).read_bytes())


def snapshot_work(parameters,rows):
    require(type(parameters) is int and type(rows) is int and 0<rows<=parameters,'scalar/row inventory')
    return dict(row_visits=rows,coefficient_visits=2*parameters,arithmetic=0,total=2*parameters+rows)


def io_schedule():
    phases=[('capture_owned_census',dict(T=1)),('worker_initial_census',dict(T=1)),
            ('transfer_before_census',dict(T=1)),('transfer_after_census',dict(T=1)),
            ('source_construct_snapshot',dict(T=1)),('source_check_snapshot',dict(T=1)),
            ('production_source_seal',dict(T=1,X=1)),
            ('portable_initial_inventory',dict(T=1,S=1,G=1,C=1)),
            ('portable_load_snapshot_tiles',dict(T=1,S=1,G=1,C=1)),
            ('portable_source_seal',dict(T=1,X=1))]
    phases += [(s,dict(T=1,S=1,G=1,C=1)) for s in ('portable_final_inventory','archive_audit',
        'inner_terminal_archive_audit','parent_package_inventory','parent_receive_census',
        'parent_prepublication_census','parent_final_census')]
    keys=('T','X','S','G','C')
    return dict(events=[dict(phase=p,reads=r) for p,r in phases],
        coefficients={k:sum(r.get(k,0) for _,r in phases) for k in keys},
        parent_coefficients={k:sum(r.get(k,0) for _,r in phases[-4:]) for k in keys},
        formula='17*T + 2*X + 9*(S+G) + 9*C',logical_not_physical_io=True,
        omitted=['metadata/code/imports','in-memory hashing and copies','offline audit/relocation'])


def audit_controls(report):
    require(report['status']=='PASS' and not report['errors'] and not report['failures'],'passing controls')
    for name,h in report['bindings'].items(): require(sha(name)==h,'bound implementation drift: '+name)
    expected={(c,p,a,s) for c in ('mixed','deep','half','guarded') for p in (0,1)
              for a in ('dense','pair_scoped') for s in (False,True)}
    seen={}; details=[]
    for r in report['differentials']:
        k=r['case'],r['profile'],r['arm'],r['shared']
        require(k in expected and k not in seen,'complete unique differential roster'); seen[k]=r
    require(set(seen)==expected,'complete differential roster')
    for c,p,a,_ in sorted(k for k in expected if k[-1]):
        old,new=seen[c,p,a,False],seen[c,p,a,True]
        require(old['result']==new['result'] and old['endpoints']==new['endpoints'],'same exact result')
        st=new['stats']
        require(st['memberships']==old['stats']['memberships'] and st['stored_memberships']<=st['memberships']
                and st['explicit_locals']+st['aliased_locals']==st['local_nnz']==old['stats']['local_nnz'],
                'logical versus physical accounting')
        details.append(dict(case=c,profile=p,arm=a,old_forest_bytes=old['stats']['forest_bytes'],
            new_forest_bytes=st['forest_bytes'],packed_source_bytes=st['packed_source_bytes'],
            memberships=st['memberships'],stored_memberships=st['stored_memberships'],
            old_work=old['work'],new_work=new['work'],old_arena=old['arena_peak'],new_arena=new['arena_peak']))
    require(all(r['audit']['status']=='PASS' for r in report['supervised']),'all terminal audits')
    return details


def analyze():
    a,b=load(SHAPE)['inventory'],load(SECOND_SHAPE)['inventory']
    for k in ('inputs','parameter_values','affine_outputs','raw_binary64_tensor_bytes','mandatory_roots'):
        require(a[k]==b[k],'frozen inventory disagreement: '+k)
    p,r,x=a['parameter_values'],a['affine_outputs'],8*a['inputs']; t=a['raw_binary64_tensor_bytes']
    require(t==8*p+x,'raw binary64 inventory')
    report=load(CONTROLS); details=audit_controls(report); counts=snapshot_work(p,r)
    # The claim is explicitly about the bound implementation, not all algorithms.
    anchors=dict(report['bindings'])
    for n in (SHAPE,SECOND_SHAPE,CONTROLS,'scripts/assess_shared_source_cost.py',
              'scripts/test_shared_source_cost.py'): anchors[n]=sha(n)
    return dict(schema='SHARED_SOURCE_COST_V1',protocol_commit=report['protocol_commit'],anchors=anchors,
        decision='CONTROLS_PASS_FULL_SHAPE_REFUSED',native_runs=0,model_runs=0,gpu_runs=0,
        inventory=dict(parameters=p,affine_rows=r,tensor_bytes=t,center_bytes=x),
        snapshot_only_work=counts,three_snapshot_only_work=3*counts['total'],
        barriers=dict(work_cap=8000000,one_snapshot_exceeds=counts['total']>8000000,
                      expanded_memberships=a['mandatory_roots'],membership_cap=65536),
        io=io_schedule(),
        retained_storage_formula='T+H+S+G+C+K+Q+M+B+R+Z+I+P+L+J+V',
        retained_member_formula='n_source+n_pages+n_tiles+n_code+11',
        memory=dict(bank_payload='8*P; <=128 MiB admitted, plus Python row metadata',
            row_transient='<=4096 weights + bias; one packed fill plus immutable copy',
            pages='S immutable bytes; extra fill <=65536; no complete forest join',
            header='bounded 8 MiB raw metadata; parsed objects additional',
            coexistence='bank + builder/checker graph + pages/header + row/Fraction temporaries; consumer arena later',
            replay='production references released first; no allocator/RSS release guarantee',
            full_work_upper_closed=False,full_rss_upper_closed=False),
        details=details,full_capacity_admitted=False)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--check-report'); args=ap.parse_args(); r=analyze()
    if args.check_report:
        require(json.loads(Path(args.check_report).read_bytes())==r,'ledger changed')
        print(json.dumps(dict(status='PASS',decision=r['decision'],snapshot_work=r['snapshot_only_work'])))
    else: print(json.dumps(r,sort_keys=True,separators=(',',':')))


if __name__=='__main__': main()
