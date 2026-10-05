"""Full ordinary continuous/binary/EQ/INEQ assembly, including empty rows."""
import numpy as np
import pytest
from experiments.neural_hz_20260831.c67_birth_emission_v1 import lift
from experiments.neural_hz_20260831.c67_direct_csr_v1 import matrices
from experiments.neural_hz_20260831.c62_physical_quotient_v1 import same_matrix,equal
from experiments.neural_hz_20260831.test_c62_physical_boundary_v2 import complete


@pytest.mark.parametrize('kind',['chain','shared','conv_disjoint'])
def test_actual_owned_assembly_equals_original_csr_bits_and_rejects_foreign_rows(kind):
    _,saved=complete(kind);seen=[]
    def observe(encoder,nodes,root,er,es,lr,ls,tracker):
        for rows in (encoder.eq,encoder.ineq):
            expected=encoder.matrices(rows);got=matrices(encoder,rows)
            assert same_matrix(expected[0],got[0]) and same_matrix(expected[1],got[1]) and equal(expected[2],got[2])
            for a,b in zip(expected[:2],got[:2]):
                assert a.indices.dtype==b.indices.dtype and a.indptr.dtype==b.indptr.dtype
            with pytest.raises(ValueError):matrices(encoder,list(rows))
            seen.append(len(rows))
    result=lift(saved['expression'],saved['keep'],enabled=True,before_fold=observe)
    assert len(seen)==2 and result['fields']['report']['alias_quotient']['work_parts']['c67_direct_final_index_assembly']==1024
