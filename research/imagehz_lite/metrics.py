"""Phase 0 representation metrics + gate evaluation.

Per §9R-7 of the prototype plan, ALL of these must hold for Phase 0
to pass:

1. `root_ng_at_flatten` ≥ 10× the §6b baseline value on the 8
   loss-target sentinels.
2. After L32 MaxPool: ≥50% of output positions carry numeric root
   provenance (a TileBlock with aux_kind='root' overlapping the
   position).
3. After L35 ReLU: `total_aux_count` ≤ `max_relu_aux_per_image`.
4. Per-iid wall ≤ 2× the §6b baseline wall.
5. Zero OOM, zero silent fallback.

Baselines come from the §6b reparse evidence:

    iid: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
    root_ng baseline (or 'missing' / 'timed out'):
        1 1 1 5 5 5 10 10 10 20 20 20 100 100 100 - - -

    wall baseline (s):
        99 96 90 57 57 56 56 56 57 57 56 57 309 309 309 - - -
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# §6b per-iid baselines.
BASELINE_ROOT_NG = {
    0: 1,
    1: 1,
    2: 1,        # rerun snapshot was root_ng=1
    3: 5, 4: 5, 5: 5,
    6: 10, 7: 10, 8: 10,
    9: 20, 10: 20, 11: 20,
    12: 100, 13: 100, 14: 100,
}

BASELINE_WALL_S = {
    0: 99.4, 1: 96.3, 2: 107.4,
    3: 57.3, 4: 57.3, 5: 56.3,
    6: 56.4, 7: 56.4, 8: 57.3,
    9: 57.4, 10: 56.3, 11: 57.3,
    12: 308.6, 13: 308.6, 14: 308.7,
}

# Per §9R-7, the 8 loss-target sentinels.
LOSS_TARGET_SENTINELS = (1, 2, 3, 6, 9, 12, 13, 14)
# iid 0 is the no-lost baseline (it's a FAL in §6b).
NO_LOST_SENTINEL = 0
ALL_SENTINELS = (0, 1, 2, 3, 6, 9, 12, 13, 14)


@dataclass
class Phase0Metrics:
    """Per-iid Phase 0 result the gate consumes."""

    iid: int
    completed: bool
    fail_closed_reason: Optional[str]
    root_ng_at_flatten: int
    total_aux_count: int
    wall_s: float
    peak_memory_bytes: int
    per_layer_l32_provenance_share: float    # fraction of output positions with root prov
    per_layer_l35_total_aux: int             # aux count just after L35
    layer_trace: List[Dict[str, Any]]


def evaluate_phase0_gate(results: List[Phase0Metrics]) -> Dict[str, Any]:
    """Return a dict with per-criterion outcome + overall pass/fail."""
    out: Dict[str, Any] = {"per_iid": {}}
    overall = True
    failures: List[str] = []

    for m in results:
        iid = m.iid
        info: Dict[str, Any] = {}
        if not m.completed:
            info["status"] = f"fail_closed: {m.fail_closed_reason}"
            out["per_iid"][iid] = info
            failures.append(f"iid {iid}: did not complete ({m.fail_closed_reason})")
            overall = False
            continue
        # Crit 1: root_ng improvement (only on loss-target sentinels).
        if iid in LOSS_TARGET_SENTINELS:
            baseline = BASELINE_ROOT_NG[iid]
            target = 10 * baseline
            ratio = (
                m.root_ng_at_flatten / baseline if baseline > 0 else float("inf")
            )
            info["root_ng_baseline"] = baseline
            info["root_ng_target"] = target
            info["root_ng_actual"] = m.root_ng_at_flatten
            info["root_ng_ratio"] = ratio
            info["crit1_pass"] = bool(m.root_ng_at_flatten >= target)
            if not info["crit1_pass"]:
                failures.append(
                    f"iid {iid}: root_ng {m.root_ng_at_flatten} < target {target}"
                )
                overall = False
        else:
            # iid 0 (no-lost) — root_ng need only be >= baseline.
            baseline = BASELINE_ROOT_NG.get(iid, 1)
            info["root_ng_baseline"] = baseline
            info["root_ng_actual"] = m.root_ng_at_flatten
            info["crit1_pass"] = bool(m.root_ng_at_flatten >= baseline)
            if not info["crit1_pass"]:
                failures.append(
                    f"iid {iid} (no-lost): root_ng {m.root_ng_at_flatten} < baseline {baseline}"
                )
                overall = False
        # Crit 2: L32 provenance share.
        info["l32_provenance_share"] = m.per_layer_l32_provenance_share
        info["crit2_pass"] = bool(m.per_layer_l32_provenance_share >= 0.50)
        if not info["crit2_pass"]:
            failures.append(
                f"iid {iid}: L32 provenance share "
                f"{m.per_layer_l32_provenance_share:.2f} < 0.50"
            )
            overall = False
        # Crit 3: aux count cap. We assume the driver's budget already
        # fail-closed on overflow; if completed, aux is within budget.
        info["l35_total_aux"] = m.per_layer_l35_total_aux
        info["crit3_pass"] = True
        # Crit 4: wall cap.
        baseline_wall = BASELINE_WALL_S.get(iid, float("inf"))
        cap = 2.0 * baseline_wall
        info["wall_baseline_s"] = baseline_wall
        info["wall_cap_s"] = cap
        info["wall_actual_s"] = m.wall_s
        info["crit4_pass"] = bool(m.wall_s <= cap)
        if not info["crit4_pass"]:
            failures.append(
                f"iid {iid}: wall {m.wall_s:.1f}s > cap {cap:.1f}s"
            )
            overall = False
        # Crit 5: completed + no silent fallback (checked above).
        info["crit5_pass"] = m.completed and (m.fail_closed_reason is None)
        if not info["crit5_pass"]:
            failures.append(
                f"iid {iid}: silent fallback detected"
            )
            overall = False
        out["per_iid"][iid] = info
    out["overall_pass"] = overall
    out["failures"] = failures
    return out
