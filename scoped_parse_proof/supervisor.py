"""Original hard supervisor/acceptance with opt-in construction dispatch only."""
from scoped_proof.io import PYTHON
from scoped_proof.supervisor import supervise as original_supervise
from scoped_parse_proof.contract import policy


def command(phase, folder, deadline):
    module = 'scoped_parse_proof.worker' if phase in ('construct','source_check') else 'scoped_proof.worker'
    return [PYTHON]+(['-S'] if phase in ('construct','source_check','aggregate') else [])+[
        '-m',module,phase,str(folder),'--deadline',str(deadline)]


def supervise(root, spec, *, budget=300., rss_limit=8*2**30, command_factory=None):
    def dispatch(phase, folder, deadline):
        policy(spec)  # validation occurs INSIDE the original timed request
        return (command_factory or command)(phase,folder,deadline)
    return original_supervise(root,spec,budget=budget,rss_limit=rss_limit,command_factory=dispatch)
