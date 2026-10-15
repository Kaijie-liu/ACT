"""Static shared-path counting and storage envelopes; never executes that path.

These are conditional complete-traversal uppers; current limits may refuse earlier. They
are neither new quotas nor RSS/time predictions. Counts use the current Work
instrumentation, not all machine operations. Only standard-library file reads,
AST inspection and small integer arithmetic run in this analysis.
"""
import argparse
import json
from pathlib import Path

from scripts import assess_shared_source_decision as prior

ROOT = prior.ROOT
PARENT = 'docs/hz_shared_source_decision_20261015.json'
BASE = '82411e0b4045a7015b732bd549321ccffab3544d'
COUNTERS = ('row_visits', 'coefficient_visits', 'arithmetic')


def counts(rows, coefficients, exact):
    return dict(zip(COUNTERS, (rows, coefficients, exact), strict=True))


def total(phases):
    return sum(sum(p.values()) for p in phases.values())


def source_work(v):
    n, a, u = v['inputs'], v['affine_outputs'], v['relu_units']
    w1, w2 = v['first_weight_positions'], v['later_weight_positions']
    f, nodes, refs = v['factors_upper'], v['nodes_upper'], v['references_upper']
    h, views, guards = v['roots_upper'], v['views'], v['guards']
    explicit = n + 2*a + 12*u
    local = w1 + explicit  # Upper inventories, NOT actual nnz minus dense width.
    prior.require(local == v['local_entries_upper'], 'local envelope binding')
    first_rows = v['router_widths'][1] + v['experts']*v['expert_widths'][1]
    snapshot = counts(a, 2*(w1+w2+a), 0)
    replay = counts(a, w1+w2, 0)
    producer = dict(snapshot=snapshot, replay=replay,
        equations=counts(f+nodes+u, 2*w1+w2,
                         n+w1+2*w2+2*a+9*u+guards+3*local+refs))
    encode = counts(2*h+2*nodes, explicit+refs+f, 0)
    checker = dict(snapshot=snapshot, replay=replay,
        intake=counts(3*nodes+first_rows+2*h, explicit+refs+f, 0),
        initialize=counts(views+nodes, local+refs+2*w1, 3*local),
        equations=counts(f+views+u, w1+w2, n+w1+2*w2+a+9*u+guards+refs))
    return dict(producer=producer, encode=encode, one_checker=checker,
        producer_upper=total(producer), encode_upper=sum(encode.values()),
        one_checker_upper=total(checker),
        build_and_two_checks_upper=total(producer)+sum(encode.values())+2*total(checker),
        source_seal_counted_in_consumer_finish=True,
        assumptions=['all first-affine positions visited, including zero',
            'stored memberships <= expanded membership upper',
            'every local may have a different scale group',
            'complete finite/bit/encoding checks retained; violations refuse'])


def consumer_work(v, scoped, chunks):
    prior.require(type(scoped) is bool, 'explicit consumer arm')
    prior.positive(chunks)
    prior.require(chunks <= prior.policy()['members'], 'bounded source chunk count')
    n, f = v['nodes_upper'], v['factors_upper']
    j = v['local_entries_upper']+v['references_upper']
    a, h = v['first_weight_positions'], v['roots_upper']
    b, d, e = v['pairs'], v['property_duties'], v['endpoints_upper']
    t, m, c = v['pair_local_tiles_upper'], v['pair_dual_rows_upper'], v['classes']
    prior.require(e == b*18 and t == b*3 and c == 10,
                  'registered per-pair endpoint/tile upper roster')
    k = b*(4+4+1) if scoped else t  # widths 8/8/2, checking two at a time
    phases = dict(roots=4*h+3*v['views']+1,
        plan=12+b*(n+a+3*j+4*f+4)+d+e*(1+2*c),
        scopes=t*(4*(n+f)+4) if scoped else 0,
        objectives=e*(5*n+2*c+58+a+11*j) if scoped else e*(3*n+2*c+46+a+6*j),
        consume_without_dual_pullback=e*(36+13*m+32*f)+3*m*k if scoped
                                     else e*(36+8*m+21*f)+3*m*k,
        dual_pullback=(4*e+k)*n+e+k*a+(k+10*e)*j if scoped
                      else (6*e+k)*n+2*e+k*a+(k+5*e)*j,
        finish=e+d+chunks+9)
    proposer = e*(33+2*m)
    production_frame, replay_frame = 3*t+5*e+4*m*e, 3*t+e
    core = sum(phases.values())
    return dict(phases=phases, check_blocks_upper=k, source_chunks_upper=chunks,
        consumer_core_upper=core, fixed_proposer_upper=proposer,
        production_frame_upper=production_frame, replay_frame_upper=replay_frame,
        production_upper=core+proposer+production_frame,
        replay_upper=core+replay_frame,
        production_and_replay_upper=2*core+proposer+production_frame+replay_frame,
        actual_work=None, native_proposer_upper=None,
        assumptions=['complete candidate pairs present', 'all global closures used as upper',
            'full merge and both norms retained', 'fresh independent replay',
            'fixed standard-library proposer only; no native/GPU proposal'])


def dense_arena(v, frame):
    p = prior.policy(); prior.positive(frame)
    slot = 1+prior.structure('scoped_source/bounded_runtime.py', 'HEADER')+2*((p['work_bits']+7)//8)
    desc = 4*(4+slot-1)
    n, f, k = v['nodes_upper'], v['factors_upper'], 8
    root = 16*n+4*v['roots_upper']
    # Shared sparse descriptors replace old dense objective roots, not the
    # original dual alpha plus its pullback copy, which coexist during consume.
    phases = dict(descriptor_seal=root+(k+1)*desc+(k-1)*f*slot,
        objective_pullback=root+k*desc+slot*(n+k*f),
        frame_seal=root+k*desc+k*f*slot+2*frame,
        full_consume=root+k*desc+2*k*slot*(n+f)+frame)
    nmin = v['inputs']+v['affine_outputs']+v['guards']
    lower = 2*k*slot*(nmin+v['inputs'])
    return dict(slot_bytes=slot, one_descriptor_upper=desc, root_upper=root,
        phases=phases, owned_payload_upper=max(phases.values()),
        complete_width8_vector_lower=lower,
        lower_assessment=prior.classify(p['total_bytes'], lower=lower),
        conditional_on='complete all-present width-8 dense consumption',
        excludes=['upstream graph/bank','caller candidate buffers','Python overhead',
                  'Fraction temporaries','allocator retained memory'],
        actual_peak=None, full_rss_upper=None)


def metadata(bits, duties, tiles, work_digits):
    prior.positive(bits, duties, tiles, work_digits)
    prior.require(bits <= 8192 and duties <= 252 and tiles <= 84, 'registered field envelope')
    digits = len(str((1 << bits)-1))
    rational_text = 2*digits+2  # optional minus and slash, each integer <= bits
    result = 233+duties*(52+rational_text)+(duties-1)
    phases, progress = 18+2*tiles, 32+4*tiles
    return dict(integer_digits=digits, fraction_text_upper=rational_text,
        result_upper=result, result_assessment=prior.classify(2**20, upper=result),
        candidate_report_full_result_copies=2,
        candidate_result_fields_upper=2*result,
        actual_result_bytes=None, upper_excess_proves_actual_overflow=False,
        one_result_bytes_over=2**19,
        implication='actual result above half report cap implies report overflow; no such size observed',
        phase_rows=phases, progress_rows=progress, counter_digits=work_digits,
        phase_list_upper=1+phases*(121+3*work_digits+24),
        progress_bytes_upper=progress*(115+3*work_digits+24),
        total_report_upper=None,
        pending=['archive event list','manifest and source header exact layout',
                 'source/candidate/member hash maps','all retained snapshots and reports'])


def analyze():
    saved = prior.load(PARENT)
    prior.require(saved == prior.analyze(), 'bound parent decision/report drift')
    v = saved['inventory']; p = prior.policy()
    src = source_work(v)
    arms = {name:consumer_work(v, scoped, p['members'])
            for name, scoped in (('dense_reference', False), ('pair_scoped', True))}
    # Two seals already in the two consumer finish terms: never add again.
    for arm in arms.values():
        arm['postcapture_counted_work_upper'] = src['build_and_two_checks_upper']+arm['production_and_replay_upper']
    highest = max(a['postcapture_counted_work_upper'] for a in arms.values())
    native_frame = saved['candidate_stream']['binary64']['largest_tile_upper']
    anchors = dict(saved['anchors'])
    for name in (PARENT, 'scripts/assess_shared_execution_envelope.py',
                 'scripts/test_shared_execution_envelope.py'):
        anchors[name] = prior.sha(name)
    return dict(schema='SHARED_EXECUTION_ENVELOPE_STATIC_V1', based_on=BASE, anchors=anchors,
        inventory=v, source_counted_work=src, consumers=arms,
        dense_owned_arena=dense_arena(v, native_frame),
        metadata_under_current_counter_cap=metadata(p['work_bits'],v['property_duties'],v['pair_local_tiles_upper'],len(str(p['visits']))),
        metadata_under_conditional_loop_upper=metadata(p['work_bits'],v['property_duties'],v['pair_local_tiles_upper'],len(str(highest))),
        file_read_core=dict(kind='logical file bytes, not physical device IO',
            online='17T+2X+9(S+G)+9C',parent='4T+4(S+G)+4C',
            definitions=dict(T='source tensor bytes including center',X='center bytes',
                S='forest payload',G='forest page index',C='candidate stream'),
            excludes=['metadata/code/imports','in-memory hash passes','offline audit',
                      'future checkpoint/live-source capture input IO']),
        open_contract=dict(capture_work_upper=None, native_candidate_work_upper=None,
            full_archive_demand_upper=None, full_rss_upper=None, completion_seconds_upper=None,
            reasons=['native source capture/proposal not part of current fixed-case supervisor',
                'generated whole metadata/file layout not yet bounded',
                'owned payload is not CPython/OS RSS; object/transient lifetimes remain',
                'Work excludes much rational/string/hash/IO/cleanup time']),
        current_refusal=dict(work=saved['source_floor_assessment'],
            memberships=saved['membership']['assessment'],
            dense_arena=prior.classify(p['total_bytes'],lower=dense_arena(v,native_frame)['complete_width8_vector_lower']),
            candidate_stream=saved['candidate_stream']),
        unchanged=dict(work=p['visits'],memberships=p['rows'],arena=p['total_bytes'],
            seconds=300,stored_bits=p['stored_bits'],work_bits=p['work_bits'],
            report_bytes=2**20,acceptance='strict > 1/10000000'),
        decision='STATIC_SUBCONTRACTS_CLOSED_FULL_RESOURCE_CONTRACT_NOT_ADMITTED',
        no_upper_used_as_quota=True, new_quota=None, runtime_changed=False,
        model_runs=0,gpu_runs=0,solver_calls=0,source_builds=0,candidate_builds=0,
        full_capacity_admitted=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check-report',type=Path)
    args=ap.parse_args(); result=analyze()
    if args.check_report:
        prior.require(json.loads(args.check_report.read_bytes())==result, 'execution envelope drift')
        print('PASS: static component uppers; unchanged policy still refuses; no runtime execution')
    else:
        print(json.dumps(result,sort_keys=True,indent=2))


if __name__=='__main__': main()
