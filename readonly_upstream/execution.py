"""Same supervisor/deadline; explicit bound opt-in, no real request loader."""
from scoped_proof.io import PYTHON, save
from source_enclosure.format import identity
from source_cost_supervised.supervisor import supervise as watched


def validate_method(method):
    if (type(method) is not dict or set(method) != {'schema','readonly'}
            or method['schema'] != 'READONLY_UPSTREAM_R1' or type(method['readonly']) is not bool):
        raise ValueError('explicit upstream representation option')


def supervise(root, spec, method, *, budget=300., rss_limit=8*2**30, command_hook=None):
    validate_method(method)
    def command(phase, folder, deadline):
        if phase == 'profile': save(folder/'method.json', method)
        module = 'readonly_upstream.' + ('worker' if phase=='profile' else 'audit')
        args = [PYTHON,'-S','-m',module,str(folder),'--method-sha256',identity(method)]
        return command_hook(phase,folder,deadline,args) if command_hook else args
    return watched(root,spec,budget=budget,rss_limit=rss_limit,command_factory=command)
