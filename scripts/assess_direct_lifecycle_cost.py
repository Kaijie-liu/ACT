"""Static V2 accounting; no source, candidate, worker or numerical execution.

Reconcile the existing code and archived controls, not a capacity benchmark.
Complete-path lower bounds are counterfactual where earlier gates refuse.
Local upper bounds must not become a new quota or a full RSS claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = '9d30c76f8fcf866302a304b3712e2f69aeeee0b0'
V1 = 'docs/hz_direct_join_controls_20261015.json'
V2 = 'docs/hz_direct_lifecycle_controls_20261015.json'
OLD = 'docs/hz_direct_join_cost_20261015.json'
CAPACITY = 'docs/hz_pair_scoped_capacity_20261015.json'
SHAPE = 'docs/hz_full_resource_ledger_20261014.json'
COUNTERS = ('row_visits', 'coefficient_visits', 'arithmetic')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    return json.loads((ROOT / name).read_bytes())


def verify_bindings(bindings):
    for name, expected in bindings.items():
        require(sha(ROOT / name) == expected, 'bound implementation drift: ' + name)


def natural(value):
    require(type(value) is int and value >= 0, 'nonnegative integer count')


def lease_work(nodes, roots, rows, arm):
    """Normal fixed proposer only, excluding prepare/encode/consume/replay.

    For CPython's tuple length hint, dense keys scan ExactMap twice. Sparse
    keys scan once; ordered lookups visit 1+...+k entries. Counts include the
    original checks and explicit proposed/zero candidate generation.
    """
    natural(nodes)
    require(nodes > 0 and arm in ('dense', 'pair_scoped'), 'nodes/consumer')
    require(type(roots) is list and type(rows) is list and len(roots) == len(rows), 'endpoint roster')
    result = dict.fromkeys(COUNTERS, 0)
    for k, m in zip(roots, rows, strict=True):
        natural(k); natural(m)
        require(k <= min(nodes, 4), 'at most four objective roots')
        result['row_visits'] += k + 2
        result['arithmetic'] += k + 1
        access = 2 * nodes + 4 * k if arm == 'dense' else 3 * k + k * (k + 1) // 2
        result['coefficient_visits'] += access + 2 * m
    return result


def archived_delta(old, new):
    """Read the same-candidate controls, not end-to-end timing experiments."""
    require(old['status'] == new['status'] == 'PASS', 'passing archived controls')
    previous = {(r['case'], r['profile']): r for r in old['differentials']}
    require(len(previous) == len(old['differentials']) == 8, 'old unique complete roster')
    expected = {(case, profile, arm) for case, profile in previous for arm in ('dense', 'pair_scoped')}
    result, seen = [], set()
    for row in new['differentials']:
        key = row['case'], row['profile'], row['arm']
        require(key in expected and key not in seen, 'new unique complete roster')
        seen.add(key)
        ref = previous[key[:2]]
        require(row['exact_result_sha256'] == ref['result']['exact_result_sha256']
                and row['checks'] == ref['candidate_checks'], 'same exact result and roles')
        e, k = ref['result']['required_endpoints'], row['lifecycle']['root_terms']
        previous_work = ref['dense_work' if row['arm'] == 'dense' else 'scoped_work']
        delta = {c: row['work'][c] - previous_work[c] for c in COUNTERS}
        require(delta['row_visits'] == 2*e+k and delta['arithmetic'] == e+k
                and delta['coefficient_visits'] >= 2*k, 'archived lifecycle Work reconciliation')
        result.append(dict(case=key[0], profile=key[1], arm=key[2], endpoints=e,
                           root_terms=k, delta=delta, exact_result_sha256=row['exact_result_sha256']))
    require(seen == expected, 'all archived consumer/profile combinations')
    return result


def io_schedule():
    """Successful online logical reads. Metadata/imports/hash CPU are extra.

    T includes center X. Census sees both source originals and packaged copies.
    In-memory hashes/seals are not disk reads. No offline audit is counted here.
    """
    events = [
        ('worker_initial_census', dict(T=1)),
        ('source_reader_and_copy', dict(T=1, X=1)),
        ('source_construct', dict(T=1)),
        ('production_source_check', dict(T=1)),
        ('production_source_reseal', dict(T=1, X=1)),
        ('portable_initial_inventory', dict(T=1, S=1, C=1)),
        ('portable_source_and_candidate_read', dict(T=1, S=1, C=1)),
        ('portable_source_reseal', dict(T=1, X=1)),
        ('portable_final_inventory', dict(T=1, S=1, C=1)),
        ('archive_audit', dict(T=1, S=1, C=1)),
        ('inner_terminal_archive_audit', dict(T=1, S=1, C=1)),
        ('parent_package_inventory', dict(T=1, S=1, C=1)),
        ('parent_receive_census', dict(T=2, S=1, C=1)),
        ('parent_prepublication_census', dict(T=2, S=1, C=1)),
        ('parent_final_census', dict(T=2, S=1, C=1)),
    ]
    total = {k: sum(counts.get(k, 0) for _, counts in events) for k in ('T', 'X', 'S', 'C')}
    return dict(events=[dict(phase=name, reads=counts) for name, counts in events], coefficients=total,
                formula_lower='18*T + 3*X + 9*S + 9*C', logical_not_physical_io=True,
                omitted=['metadata and code reads', 'Python imports', 'procfs supervision',
                         'in-memory hashing/copies', 'offline audit and relocation'])


STORAGE = dict(T=2, H=2, S=1, C=1, K=1, Q=1, M=1, B=1, R=1, Z=1,
               I=1, P=1, L=1, J=1, V=1)


def retained_bytes(sizes):
    require(set(sizes) == set(STORAGE), 'complete storage inventory')
    for n in sizes.values():
        natural(n)
    return sum(STORAGE[k] * n for k, n in sizes.items())


def capacity_migration(capacity):
    """Account for two intentional V1 direct-source adapters in older bounds.

    The old regression deliberately refuses current hashes. Do not rewrite it.
    Read its frozen Git blobs and require exactly the documented insertions;
    mathematical consumer loops and owned arena implementations stay unchanged.
    """
    before = "    require(type(forest) in (CheckedScaledForest,CheckedWireForest) and getattr(forest,'_sealed',False) is True\n"
    after = """    admitted=type(forest) in (CheckedScaledForest,CheckedWireForest)
    if not admitted and type(forest).__module__=='scoped_source.check_direct_source':
        from scoped_source.check_direct_source import CheckedDirectForest
        admitted=type(forest) is CheckedDirectForest
    require(admitted and getattr(forest,'_sealed',False) is True
"""
    seal = "        if type(self.__forest) is CheckedWireForest: self.__forest.source_access.seal()\n"
    addition = """        if type(self.__forest).__module__=='scoped_source.check_direct_source':
            from scoped_source.check_direct_source import CheckedDirectForest
            require(type(self.__forest) is CheckedDirectForest,'closed direct type')
            self.__forest.seal()
"""
    edits = {'scoped_source/check_scaled_references.py': (before, after),
             'scoped_source/scaled_endpoints.py': (seal, seal + addition)}
    base = 'd2c7a6f68184c7773ba3f2496db03398a9156ff4'
    require(capacity['based_on'] == base, 'historical capacity commit')
    observed = []
    for name, expected in capacity['anchors'].items():
        current = (ROOT/name).read_bytes()
        if name not in edits:
            require(hashlib.sha256(current).hexdigest() == expected, 'capacity dependency drift: ' + name)
            continue
        historical = subprocess.check_output(['git', 'show', base + ':' + name], cwd=ROOT)
        require(len(historical) < 128*2**10 and hashlib.sha256(historical).hexdigest() == expected,
                'frozen Git blob identity')
        old, new = (v.encode() for v in edits[name])
        require(historical.count(old) == 1 and historical.replace(old, new, 1) == current,
                'only reviewed direct-source adapters may differ: ' + name)
        observed.append(dict(file=name, frozen=expected, current=sha(ROOT/name)))
    require(len(observed) == 2, 'complete two-adapter migration')
    return dict(old_anchor_suite_passes_current=False, old_results_unchanged=True,
                changed_dependencies=observed, consumer_loops_and_arena_unchanged=True,
                added_seal_in_v2_io_ledger=True, new_execution=False)


def analyze():
    old, current, previous, capacity, shape = [load(n) for n in (V1, V2, OLD, CAPACITY, SHAPE)]
    # Source/consumer code remains exactly bound to V1. V2 bindings include the
    # actual worker, supervision, lifecycle, replay and frozen protocol.
    verify_bindings(previous['anchors'])
    verify_bindings(current['bindings'])
    migration = capacity_migration(capacity)
    anchors = dict(previous['anchors'], **current['bindings'])
    anchors.update({n: sha(ROOT/n) for n in (V1, V2, OLD, CAPACITY, SHAPE)})
    v = shape['inventory']
    require(v == capacity['inventory'] and (v['inputs'], v['experts'], v['classes']) == (3072, 8, 10),
            'frozen registered shape')
    endpoints, n, m = v['endpoints_upper'], v['nodes_upper'], v['pair_dual_rows_upper']
    source_floor = 3 * v['first_weight_positions']
    floors = {arm: dict(source_first_affine=source_floor,
                        two_consumers=2 * capacity['support_work_floor'][key]['work_lower'])
              for arm, key in (('dense', 'dense_reference'), ('pair_scoped', 'pair_scoped'))}
    for f in floors.values():
        f['total'] = sum(f.values())
    fixed = {arm: lease_work(n, [4]*endpoints, [m]*endpoints, arm)
             for arm in ('dense', 'pair_scoped')}
    tensor, center = v['raw_binary64_tensor_bytes'], 8*v['inputs']
    reads = io_schedule()
    reads['registered_tensor_terms_only'] = 18*tensor + 3*center
    reads['parent_tensor_forest_candidate_min'] = '7*T + 4*S + 4*C'
    return dict(
        schema='DIRECT_LIFECYCLE_COST_STATIC_V2', based_on=BASE,
        anchors=anchors, analysis_sha256=sha(Path(__file__)), inventory=v,
        historical_capacity_migration=migration,
        decision='NO_FULL_TOPOLOGY_ADMISSION_FIXED_STUB_ONLY',
        conditional_complete_work_floors=floors,
        source_core_upper=previous['source_equations_and_format_only'],
        fixed_proposer_only_upper=dict(endpoint_count=endpoints, roots_per_endpoint=4,
            dual_rows_per_endpoint=m, arms=fixed, totals={a: sum(c.values()) for a, c in fixed.items()},
            excludes=['Driver.prepare', 'encoding', 'both consumers', 'replay', 'source', 'hash/IO/GC']),
        archived_increment_checks=archived_delta(old, current),
        io=reads,
        storage=dict(coefficients=STORAGE, symbol_meaning=dict(T='tensor payload including center',
            H='source manifest', S='forest', C='all candidate frames', K='checker code',
            Q='property contract', M='package manifest', B='bundle', R='result', Z='inner terminal',
            I='invocation', P='progress', L='log', J='candidate report', V='outer receipt'),
            raw_tensor_copies=2, raw_tensor_bytes=tensor,
            dense_first_weights_subcase_lower=2*tensor + 12*v['first_weight_positions'],
            dense_subcase_is_not_checkpoint_measurement=True,
            outer_metadata_reserve=3*2**20+2*65536, inner_terminal_reserve=8192,
            worker_archive_limit='134217728 - 3276800 - initial_census_bytes',
            member_count='2*source_members + candidate_tiles + checker_code_files + 11',
            worker_member_limit='4096 - initial_census_members - 3',
            full_archive_upper=None),
        barriers=dict(memberships_lower=v['mandatory_roots'], memberships_cap=65536,
            work_cap=8000000, storage_cap=128*2**20, worker_single_file_cap=2**20,
            allowed_cases=['half', 'guarded'], real_or_native_dispatch=False,
            binary64_source_coefficients_only=True),
        lifetime=dict(production_graph_retained_during_replay=False,
            production_graph_raw_coexist_during_first_check=True,
            builder_graph_plus_mutable_and_immutable_encoding=True,
            production_owner_released_before_replay=True,
            python_allocator_rss_release_proven=False,
            replay_raw_forest_only_during_source_construction=True,
            replay_graph_and_current_candidate_live=True,
            worker_cleaned_before_parent_receive=True,
            parent_metadata_and_inventory_live=True, sampled_rss_limit=2*2**30,
            pair_arena_any_admitted_frame_upper=capacity['any_admitted_frame_arena']['owned_upper'],
            complete_rss_upper=None),
        clock=dict(budget=300, start='supervise API entry',
            run_deadline='D - min(1, budget/5)', cleanup_deadline='D - min(1, budget/5)/2',
            includes=['worker imports/capture', 'source/plan/roots/candidates', 'both exact consumers',
                      'portable source recheck', 'hash/serialization/IO', 'group cleanup',
                      'parent receive/publication/final census'],
            outside_online=['parent module import before API call', 'offline audit/relocation'],
            late_saved_success_rejected=True, guaranteed_return_within_300=False),
        work_accounting=dict(parent_restores_child_counters=True, source_semantic_passes=3,
            exact_consumer_passes=2, proposer_passes=1,
            parent_repeats_lp_proof=False,
            not_counted_as_visits=['fixed fixture capture', 'hash/JSON/struct/read/write/fsync',
                                  'garbage collection', 'process observation', 'parent structural reception'],
            these_costs_are_in_wall_budget=True, killed_final_counter_is_unknown=True),
        comparison=dict(fixed_candidates_matched=True, same_consumer_in_production_replay=True,
            dense_root_enumeration_scans_nodes=True, scoped_root_enumeration_max=4,
            root_access_is_a_comparison_factor=True, native_matched_implemented=False),
        complete_work_upper=None, complete_cost_closed=False, execution_admitted=False,
        new_quota=None, runtime_changes=False, source_builds=0, candidate_generations=0,
        model_runs=0, gpu_runs=0, solver_calls=0,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-report', type=Path)
    args = parser.parse_args()
    report = analyze()
    if args.check_report:
        require(json.loads(args.check_report.read_bytes()) == report, 'static V2 report drift')
        print('PASS: V2 code/cost reconciliation; no execution admission')
    else:
        print(json.dumps(report, sort_keys=True, indent=2))
