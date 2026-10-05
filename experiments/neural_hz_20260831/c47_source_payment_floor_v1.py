"""State-size lower bound: may reject an unpaid append, never authorize it."""


def metadata_work(map_entries,halves):
    if (type(map_entries) is not int or type(halves) is not int
            or not 0<=halves<=map_entries<=64_000_000):
        raise ValueError('literal complete source map and half counts required')
    return 16*map_entries+64*halves+256


def append_screen(used,map_entries,halves,*,cap=256_000_000):
    if type(used) is not int or type(cap) is not int or not 0<=used<=cap<=256_000_000:
        raise ValueError('unchanged whole work cap and actual used work required')
    floor=metadata_work(map_entries,halves)
    return dict(actual_baseline_work=used,remaining_work=cap-used,
        one_mandatory_metadata_check_work=floor,
        append_impossible_from_lower_bound=floor>cap-used,
        shortfall_from_this_check_alone=max(0,floor-(cap-used)),
        complete_runtime_payment_proved=False,actual_attempt_authorized=False,
        includes_source_geometry_hash_inverse_and_witness_cost=False,formal_gain=0)
