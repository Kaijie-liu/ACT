"""Observation only: retain original backend assertions and numerical path."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time


def install(domains, solver, emit):
    old_init, old_add, old_get = domains.__init__, domains.add, solver.get_lA

    def tensors(values):
        return {k: {'shape': list(v.shape), 'nonzero': int(v.count_nonzero()),
                    'dtype': str(v.dtype)} for k, v in values.items()}

    def nodes(net):
        return [{'name': n.name, 'type': type(n).__name__,
                 'used': getattr(n, 'used', None), 'perturbed': n.perturbed,
                 'lA_none': getattr(n, 'lA', None) is None,
                 'output_shape': list(n.output_shape)}
                for n in net.get_splittable_activations()]

    def init(self, ret, lAs, *args, **kwargs):
        emit('DOMAIN_INIT', {'lAs': tensors(lAs)})
        return old_init(self, ret, lAs, *args, **kwargs)

    def add(self, bounds, *args, **kwargs):
        emit('DOMAIN_ADD', {'stored_keys': sorted(self.all_lAs),
             'returned_lAs': tensors(bounds['lAs']), 'nodes': nodes(self.net.net),
             'C': self.net.c.detach().cpu().tolist()})
        return old_add(self, bounds, *args, **kwargs)

    def get(self, *args, **kwargs):
        result = old_get(self, *args, **kwargs)
        emit('GET_LA', {'lAs': tensors(result), 'nodes': nodes(self.net)})
        return result

    domains.__init__, domains.add, solver.get_lA = init, add, get
    return old_init, old_add, old_get


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', type=Path, required=True)
    p.add_argument('--events', type=Path, required=True)
    p.add_argument('--config', required=True)
    a = p.parse_args()
    with a.events.open('x') as f:
        def emit(kind, data):
            f.write(json.dumps({'event': kind, 'monotonic': time.monotonic(), **data}, allow_nan=False)+'\n')
            f.flush()
        sys.path.insert(0, str(a.backend / 'complete_verifier'))
        from branching_domains import BatchedDomainList
        from beta_CROWN_solver import LiRPANet
        install(BatchedDomainList, LiRPANet, emit)
        sys.argv = [str(a.backend/'complete_verifier/abcrown.py'), '--config', a.config]
        runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
