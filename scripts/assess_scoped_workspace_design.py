"""Static design arithmetic only; no producer, source build or numerical import.

The payload envelope is not a CPython/native RSS or full Work bound. All current
runtime caps remain unchanged. This is not an execution admission mechanism.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANCHORS = {
    'scoped_source/check_source_dag.py': '58e5e2158843631c9e6ed324cfeeb2eb38e51df1205d2e14d1ed8ac8ce634458',
    'scoped_source/check_affine_references.py': 'a0d688ee82bf88c42945edfd3abc9591a46998c01cf236cbf752ad5b549e9c62',
    'scoped_source/check_scaled_references.py': '244dc7ee5cf7fe46ab7097b2494dfec64d818e926b4726eb19d59a53c5074f9e',
    'scoped_source/scaled_endpoints.py': 'e8bd15830187dfe376231fc44fc780e09dc632eade411b6327b34ffffd9276ff',
    'scoped_source/forest_endpoints.py': '86396e9c662ba8c91f6ac3921dc7f410d9b977937df38a9fb782f82991c50a4c',
    'scoped_source/bounded_runtime.py': '8f4867b885c890890b097cb87e0cf8f84fe60e3b7c6b366c929b3a5839c9de2b',
    'scoped_source/direct_candidates.py': '3cbebd0a894b4db304c6d71d353c5b9d23ce7dbebc7367475fdeeb5a84ae9150',
    'act/back_end/moe/scaled_forest_endpoints.py': '118e88e3484ff4f4977f3b2535fa3aba8bdf7b1b1b7598da8232c0fe0ea57af1',
    'docs/hz_owned_capacity_20261014.json': 'd1e2bb02a5945080881f4e8aa6f22c113ed8baf319adb541e6fc0155324e4486',
    'docs/hz_full_resource_ledger_20261014.json': '04c51e4765ada9445dcdd80f63f745459366154037443a68dd565c093acbe9d2',
}
SLOT = 2065
SCALAR = SLOT - 1
ARENA = 128 * 2**20


def verify_files():
    for path, expected in ANCHORS.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError('scope design anchor drift: ' + path)


def shape(v):
    """Topology-level upper, including all router coordinates and pair guards."""
    n, e = v['inputs'], v['experts']
    a = sum(v['router_widths'][1:]) + 2 * sum(v['expert_widths'][1:])
    u = sum(v['router_widths'][1:-1]) + 2 * sum(v['expert_widths'][1:-1])
    return dict(pair_affine_outputs=a, pair_relu_units=u, guards=2*(e-2),
                pair_nodes_upper=n+a+4*u+2*(e-2),
                pair_factors_upper=n+2*a+5*u,
                pair_factors_old_loose_upper=n+2*a+7*u,
                global_factors_tighter_upper=n+2*v['affine_outputs']+5*v['relu_units'])


def chunks(k, q):
    if type(k) is not int or type(q) is not int or not 1 <= q <= k <= 8:
        raise ValueError('candidate width / checking microblock')
    return [min(q, k-i) for i in range(0, k, q)]


def payload(v, scope, q, *, k=8):
    """Proposed packed payload, not actual allocation or full process memory.

    Retain old conservative global counts for index sizes; no old report changes.
    The group bank is deliberately also included at vector peak (safe overlap
    overestimate). Fraction temporaries/Python metadata are outside this term.
    """
    chunks(k, q)
    n, f, h = v['nodes_upper'], v['factors_upper'], v['roots_upper']
    pn, pf, m = scope['pair_nodes_upper'], scope['pair_factors_upper'], v['pair_dual_rows_upper']
    root_index = 16*n + 4*h
    maps = 4*(n+f+pn+pf) + v['views'] + 12*m
    # Each endpoint retains a count, exact offset and <=4 (global ID,value) roots.
    objectives = k*(4+SCALAR+4*(4+SLOT))
    frame = 128 + k*(2152+16*m)  # unchanged native direct frame
    # One global-order scale accumulator set, reused across endpoint norms.
    # <=n_input distinct input scales plus auxiliary unit scale.
    groups = (v['inputs']+1)*(4+SLOT)
    # q rhs, two checked bounds per tile endpoint, three reusable scalar totals.
    scalars = (q+2*k+3)*SCALAR
    vector_bank = q*(pn+2*pf)*SLOT
    prepare_vectors = (pn+q*pf)*SLOT  # one objective pullback at a time
    fixed = root_index+maps+objectives+frame+groups+scalars
    peak = fixed + max(vector_bank, prepare_vectors)
    return dict(candidate_width=k, checker_microblock=q, subdivisions=chunks(k,q),
                root_index=root_index, scope_maps_and_rows=maps,
                sparse_objective_records=objectives, frame=frame,
                norm_group_records=groups, scalar_records=scalars,
                consume_vectors=vector_bank, objective_prepare_vectors=prepare_vectors,
                conservative_packed_payload=peak, arena=ARENA,
                packed_upper_fits=peak<=ARENA, arena_headroom=ARENA-peak,
                actual_bytes=None, full_peak_rss_upper=None, full_work_upper=None,
                implemented=False, execution_admitted=False)


def scan_delta(v, q):
    """Additional full-node and dual-row scans, NOT complete Work or runtime."""
    widths = [8, 8, 2]  # 18 endpoints per registered pair
    blocks = sum((k+q-1)//q for k in widths)*v['pairs']
    old = len(widths)*v['pairs']
    extra = blocks-old
    return dict(old_candidate_tiles=old, new_checking_microblocks=blocks,
                extra_dual_node_row_visits_upper=extra*v['nodes_upper'],
                extra_dual_constraint_row_visits_upper=extra*v['pair_dual_rows_upper'],
                objective_pullbacks_upper=v['endpoints_upper'],
                full_factor_checks_retained=True,
                scope_metadata_entries_upper=v['pairs']*(v['nodes_upper']+v['factors_upper']+
                                                         v['views']+v['pair_dual_rows_upper']),
                byte_access_and_scope_work_not_included=True,
                total_work_upper=None, runtime_seconds=None)


def analyze():
    verify_files()
    v=json.loads((ROOT/'docs/hz_full_resource_ledger_20261014.json').read_bytes())['inventory']
    if (v['inputs'],v['experts'],v['classes']) != (3072,8,10):
        raise ValueError('registered shape only')
    s=shape(v)
    variants={str(q):payload(v,s,q) for q in (1,2,3,4,8)}
    # Choose from fixed byte arithmetic, not measured verification outcomes.
    fit=max(q for q in range(1,9) if payload(v,s,q)['packed_upper_fits'])
    return dict(schema='SCOPED_WORKSPACE_DESIGN_STATIC_V1',date='2026-10-14',
        based_on='8e4d9165732ee10bdee82facef9ab78c04687ed7',anchors=ANCHORS,
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope=s,variants=variants,largest_microblock_fitting_packed_envelope=fit,
        scope_only_eight_objective_vectors_upper=8*s['pair_factors_upper']*SLOT,
        selected_scan_subledger=scan_delta(v,fit),
        source_membership_cap_unchanged=65536,work_cap_unchanged=8000000,seconds_unchanged=300,
        obligations_unchanged=252,max_endpoints_unchanged=504,max_roles_unchanged=1008,
        production_changed=False,real_model_runs=0,gpu_runs=0,source_builds=0,
        complete_cost_closed=False,new_quota=None,execution_admitted=False,
        candidate_format_changed=False,
        still_blocked=['old full-source membership and Work admission',
                      'unchanged candidate full-stream cap and general-profile tile envelope',
                      'source capture/checking and Python/native complete lifetime',
                      'production correspondence and differential/fault controls'],
        decision='PAIR_SCOPE_ALONE_INSUFFICIENT_EXPLICIT_MICROBLOCK_DESIGN_ONLY')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-report',type=Path); args=p.parse_args(); result=analyze()
    if args.check_report:
        if json.loads(args.check_report.read_bytes())!=result:
            raise ValueError('scope design report drift')
        print('PASS: scoped microblock static design; full cost and execution not admitted')
    else:
        print(json.dumps(result,sort_keys=True,indent=2))
