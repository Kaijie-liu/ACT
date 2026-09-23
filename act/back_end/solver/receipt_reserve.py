"""Request-local opt-in native soft-limit reserve; no new acceptance rule.

Only output queries use the reserve. The protected row order, equal-share
deadlines, base budget, native fallback and parent receipt checks are unchanged.
Both experiment arms use this interface (1.0 vs 0.8) and its accounting.
"""
from contextlib import contextmanager
from pathlib import Path

from act.back_end.solver.checked_base_session import CheckedBaseSession
from act.back_end.solver.native_feasibility_worker import native_fraction


def command_with_fraction(command, fraction):
    native_fraction({}, fraction)
    worker = Path(__file__).with_name('native_feasibility_worker.py').resolve()
    if (len(command) != 6 or Path(command[1]).resolve() != worker or
            command[2] != '--model' or command[4] != '--sha256'):
        raise ValueError('unexpected native invocation')
    return [*command, '--output-budget-fraction', str(float(fraction))]


@contextmanager
def reserve_on_checked_experts(cls, fraction):
    """Use inside checked_execution; bind a fresh factory for each evaluation."""
    native_fraction({}, fraction)
    if cls is None:
        raise ValueError('checked expert execution required')
    original = cls.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        factory = self.session_factory

        def create(model, folder):
            session = factory(model, folder)
            if not isinstance(session, CheckedBaseSession) or session.command_factory is not None:
                raise ValueError('cannot replace an unknown session/worker')
            session.command_factory = lambda cmd: command_with_fraction(cmd, fraction)
            return session

        self.session_factory = create

    cls.__init__ = init
    try:
        yield
    finally:
        cls.__init__ = original
