"""Explicit request-local composition of the three checked shortcuts.

No default verifier change. No property deduplication, new relaxation, solver
acceptance, cached cross-request answers or extra budget. A hard outer process
deadline is still required by the caller.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckedExecutionOptions:
    expert_base: bool = False
    route_feasibility: bool = False
    score_nonzero: bool = False

    def __post_init__(self):
        if any(type(v) is not bool for v in asdict(self).values()):
            raise ValueError('shortcut flags must be explicit booleans')


@contextmanager
def checked_execution(directory, *, request_sha256, input_sha256, options):
    """Install only requested hooks, restoring all on every exit path."""
    if not isinstance(options, CheckedExecutionOptions):
        raise TypeError('CheckedExecutionOptions required')
    from act.back_end.solver.current_assignment import AssignmentScope
    AssignmentScope(request_sha256, input_sha256, 'composition').validate()
    directory = Path(directory)
    args = dict(request_sha256=request_sha256, input_sha256=input_sha256)
    with ExitStack() as stack:
        expert = None
        if options.expert_base:
            from act.back_end.solver.checked_base_session import checked_base_experts
            expert = stack.enter_context(checked_base_experts(directory/'protected', **args))
        if options.route_feasibility:
            from act.back_end.solver.checked_route_feasibility import checked_route_feasibility
            stack.enter_context(checked_route_feasibility(directory/'routing', **args))
        if options.score_nonzero:
            from act.back_end.solver.checked_nonzero_support import checked_nonzero_support
            stack.enter_context(checked_nonzero_support(directory/'nonzero', **args))
        yield expert
