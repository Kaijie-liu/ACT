"""Four dictionary-only checks of the precise C128v2 budget diagnostic fix."""
from copy import deepcopy

import pytest

from experiments.neural_hz_20260831.c128_complete_source_v2 import normalize_metadata


def _metadata(*, whole=100, branch=200, reserve=1000):
    return dict(schema="unchanged_source_schema", lineage_schema="unchanged_inverse_schema",
        independent_qualification_pending=True, root=1,
        field_scalars=dict(old_n_cont=2, old_n_bin=1, old_n_eq=1, logical_n_cont=4),
        hz=dict(frame_id=17, exact=True, n_cont=4, n_bin=1, n_eq=3, n_ineq=1),
        origin_binding=(17, 2, "original_source_digest"),
        nodes=[dict(kind="source", width=2, parents=[], support_work=3),
               dict(kind="op", width=2, parents=[0], support_work=7)],
        report=dict(whole_base_work=whole, branch_base_work=branch,
            support_work=10, total_work_upper=whole+32, largest_branch_work_upper=branch+32,
            affine_work_upper=whole-1, binary_edges=2, ownership_words=2,
            node_counts=[dict(kind="source", width=2, continuous_edges=2, support_work=3),
                         dict(kind="op", width=2, continuous_edges=4, support_work=7)],
            alias_quotient=dict(coupled_extra_capacity=min(reserve-whole,reserve-branch),
                coupled_extra_work=32, exact_aliases=1, original_inverse_preserved=True),
            other_diagnostic=dict(coupled_extra_capacity=777)))


def test_exact_capacity_formula_allows_only_derived_capacity_difference_without_mutation():
    old = _metadata(whole=100, branch=200)
    new = _metadata(whole=50, branch=120)
    old_before, new_before = deepcopy(old), deepcopy(new)
    assert old["report"]["alias_quotient"]["coupled_extra_capacity"] == 800
    assert new["report"]["alias_quotient"]["coupled_extra_capacity"] == 880
    a = normalize_metadata(old, source_reserve=1000)
    b = normalize_metadata(new, source_reserve=1000)
    assert a == b
    assert a["report"]["alias_quotient"]["coupled_extra_capacity"] == "exact_original_pool_formula_checked"
    assert old == old_before and new == new_before
    assert a["report"]["alias_quotient"] is not old["report"]["alias_quotient"]


def test_wrong_capacity_and_excess_actual_work_reject_before_normalization():
    for bad in (799, 801):
        value = _metadata()
        value["report"]["alias_quotient"]["coupled_extra_capacity"] = bad
        before = deepcopy(value)
        with pytest.raises(ValueError, match="original pool formula"):
            normalize_metadata(value, source_reserve=1000)
        assert value == before
    value = _metadata()
    value["report"]["alias_quotient"]["coupled_extra_work"] = 801
    with pytest.raises(ValueError, match="original pool formula"):
        normalize_metadata(value, source_reserve=1000)


def test_whole_and_branch_bottlenecks_both_use_the_original_minimum():
    whole_limited = _metadata(whole=300, branch=100)
    branch_limited = _metadata(whole=100, branch=300)
    tied = _metadata(whole=300, branch=300)
    for value in (whole_limited, branch_limited, tied):
        assert value["report"]["alias_quotient"]["coupled_extra_capacity"] == 700
    expected = normalize_metadata(whole_limited, source_reserve=1000)
    assert normalize_metadata(branch_limited, source_reserve=1000) == expected
    assert normalize_metadata(tied, source_reserve=1000) == expected


def test_all_other_semantic_and_nested_metadata_remains_in_the_comparison():
    value = _metadata()
    expected = deepcopy(value)
    for field in ("whole_base_work", "branch_base_work", "support_work", "total_work_upper",
                  "largest_branch_work_upper", "affine_work_upper"):
        expected["report"].pop(field)
    for item in expected["nodes"] + expected["report"]["node_counts"]:
        item.pop("support_work")
    expected["report"]["alias_quotient"]["coupled_extra_capacity"] = "exact_original_pool_formula_checked"
    assert normalize_metadata(value, source_reserve=1000) == expected
    # A same-named field elsewhere is NOT recursively erased; actual work,
    # alias results, source frame and graph incidences remain semantic evidence.
    for path, replacement in (
        (("report", "other_diagnostic", "coupled_extra_capacity"), 778),
        (("report", "alias_quotient", "coupled_extra_work"), 33),
        (("report", "alias_quotient", "exact_aliases"), 2),
        (("hz", "frame_id"), 18),
        (("report", "node_counts", 1, "continuous_edges"), 5),
    ):
        changed = deepcopy(value)
        destination = changed
        for step in path[:-1]:
            destination = destination[step]
        destination[path[-1]] = replacement
        assert normalize_metadata(changed, source_reserve=1000) != expected
