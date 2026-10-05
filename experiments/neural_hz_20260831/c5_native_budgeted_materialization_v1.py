"""Diagnostic sequence budget around the unchanged ordered compiler."""

from experiments.neural_hz_20260831 import c5_ordered_union_contraction_v3 as compiler


class BudgetedMaterializer:
    def __init__(self, branches, *, whole_cap=256_000_000, branch_cap=200_000_000):
        if type(branches) is not int or branches < 1:
            raise ValueError("invalid branch count")
        for cap in (whole_cap, branch_cap):
            if type(cap) is not int or cap < 0:
                raise ValueError("invalid sequence cap")
        self.remaining_whole = whole_cap
        self.remaining_branches = [branch_cap] * branches
        self.stage_stats = []
        self.failed = False

    def run(self, expr, rows, *, observe=None):
        if self.failed or len(expr.terms) != len(self.remaining_branches):
            raise ValueError("invalid/failed materialization sequence")
        original = compiler.contract
        index = 0

        def charged(*args, **kwargs):
            nonlocal index
            if index >= len(self.remaining_branches):
                raise ValueError("unexpected branch call")
            kwargs["max_products"] = min(kwargs.get("max_products", 200_000_000),
                                         self.remaining_whole, self.remaining_branches[index])
            result = original(*args, **kwargs)
            cost = result.stats["channel_product_upper_bound"]
            self.remaining_whole -= cost
            self.remaining_branches[index] -= cost
            index += 1
            if not result.stats["quarter_product_gate"]:
                raise MemoryError("branch quarter-product gate failed")
            return result

        # Single-thread diagnostic only; no persistent production modification.
        compiler.contract = charged
        try:
            result, stats = compiler.materialize(expr, rows, max_products=self.remaining_whole, observe=observe)
            if index != len(self.remaining_branches):
                raise ValueError("incomplete source-branch sequence")
            self.stage_stats.append(stats)
            return result, stats
        except BaseException:
            self.failed = True
            raise
        finally:
            compiler.contract = original
