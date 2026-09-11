import copy
import unittest

import numpy as np
import scipy.sparse as sp

from act.back_end.solver.solver_hz import SparseHZono
from act.back_end.solver.hz_lp_export import export
from act.back_end.solver.check_hz_lp_export import check_export
from act.back_end.solver.lp_certificate import propose


class HZExportTests(unittest.TestCase):
    def record(self):
        hz = SparseHZono(c=np.array([.1, .3]), Gc=sp.csr_matrix([[2., .5], [1., -.25]]),
                        Gb=sp.csr_matrix([[.25], [0.]]), Ac=sp.csr_matrix([[1., 1.]]),
                        Ab=sp.csr_matrix([[0.]]), b=np.array([0.]),
                        Auc=sp.csr_matrix([[1., 0.]]), Aub=sp.csr_matrix([[1.]]),
                        ub=np.array([.5]), frame_id=41)
        return export(hz, [1, -1])

    def test_actual_sparse_hz_relaxation_and_rational_bound(self):
        record = self.record()
        certificate = propose(record["lp"])
        result = check_export(record, certificate, expected_source_sha256=record["source_sha256"])
        self.assertEqual(result["n_relaxed_binaries"], 1)
        self.assertEqual(result["bound"]["status"], "CHECKED")

    def test_omission_sign_offset_and_factor_box_mutations(self):
        original = self.record()
        for mutate in (
            lambda r: r["lp"]["A"].clear(),
            lambda r: r["lp"]["A"][0].__setitem__(0, -1),
            lambda r: r["lp"].__setitem__("offset", 100),
            lambda r: r["lp"]["lower"].__setitem__(-1, 0),
            lambda r: r["lp"]["c"].__setitem__(0, 5),
            lambda r: r["source"]["ub"].__setitem__(0, 1),
        ):
            record = copy.deepcopy(original)
            mutate(record)
            with self.assertRaises(ValueError):
                check_export(record, expected_source_sha256=original["source_sha256"])


if __name__ == "__main__":
    unittest.main()
