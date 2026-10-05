"""Source-bound reconstruction equations for the full raw chain cohort."""
from types import SimpleNamespace
import numpy as np
from experiments.neural_hz_20260831.c57_scalar_consumer_probe_v2 import Probe as ChainProbe
from experiments.neural_hz_20260831.c58_reconstruction_equations_v1 import build, audit, encode, decode
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect


class Probe(ChainProbe):
    def finish(self, hz, *, observe=None):
        binding = self.chain_hash.hexdigest()
        packet = build(self.roots, self.weights, n_bin=hz.n_bin, source_binding=binding,
                       pool=self.pool, enabled=True)
        proof = audit(packet, self.roots, self.weights, n_bin=hz.n_bin,
                      source_binding=binding, pool=self.pool)
        self.pool.charge('c58_archive_and_packet_owner_preflight',
                         32 * len(self.weights) + 64 * len(packet['pairs']) + 8192)
        payload = encode(packet)
        restored = decode(payload)
        replay = audit(restored, self.roots, self.weights, n_bin=hz.n_bin,
                       source_binding=binding, pool=self.pool)
        owners = collect(SimpleNamespace(), dict(packet=packet, restored=restored,
                         archive=np.frombuffer(payload, np.uint8)))
        physical = owners.measure()
        if physical.resident_entries > 64_000_000:
            raise MemoryError('complete inverse packet/restore/archive entry cap')
        report = dict(schema='c58_actual_source_inverse_equation_realization_v1',
                      counts=dict(self.counts), proof=proof, restored_proof=replay,
                      chain_identity_sha256=binding, inverse_packet_sha256=packet['seal'],
                      inverse_archive_bytes=len(payload), packet_and_restore_and_archive_numeric_bytes=physical.resident_bytes,
                      packet_and_restore_and_archive_numeric_entries=physical.resident_entries,
                      packet_and_restore_and_archive_python_shallow_bytes=owners.python_shallow_bytes,
                      complete_consumer_analysis=False, complete_physical_HZ_reduction_proved=False,
                      native_solver_or_witness_executed=False, formal_gain=0)
        if observe:
            observe(dict(event='complete_actual_inverse_equations_proved', **report))
        # These explicitly returned owners are removed from JSON by the worker,
        # retained in its whole-state ledger, and written as an exclusive artifact.
        return dict(**report, packet=packet, restored_packet=restored, archive_bytes=payload)
