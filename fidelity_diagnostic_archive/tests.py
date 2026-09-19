from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from single_check_portable.execution import ROOT, save_new
from lp_sandwich.check import identity
from fidelity_supervised.native import OPTIONS
from fidelity_diagnostic_archive.review import imported_evidence, aggregate
from sparse_diagnostic_archive.tests import Controls as OldArchiveControls, failed
from fidelity_diagnostic_archive.structure import input_bit_inventory, basis_inventory


class ImportControls(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='fidelity_archive_controls_',dir=ROOT/'data/moe/results'))
        (self.root/'native').mkdir()
        self.expected={'rows':[{'entries':[[0,3.6e-10]],'lower':'-inf','upper':1.}],
                       'cost':[1.],'lower':[0.],'upper':[1.],'offset':0.,'sense':'minimize'}
        self.values={'input.json':{'submitted':self.expected},
            'preflight.json':{'matrix_entries':1},
            'import_status.json':{'status':'HighsStatus.kOk','options':dict(OPTIONS),
                                   'submitted_sha256':identity(self.expected)},
            'import.json':{'status':'HighsStatus.kOk','options':dict(OPTIONS),
                           'submitted_sha256':identity(self.expected),'readback':deepcopy(self.expected),'seconds':.01},
            'capture.json':{k:deepcopy(self.expected) for k in ('submitted','readback_before','readback_after')},
            'raw_native.json':{'model_status':'untrusted','basis_valid':True,'value_valid':True,'native_objective':0.}}
        for name,v in self.values.items():save_new(self.root/'native'/name,v)

    def use(self,values):
        with patch('fidelity_diagnostic_archive.review.read',lambda p:values[p.name]):
            return imported_evidence(self.root)

    def test_complete_readback_and_missing_evidence(self):
        r=imported_evidence(self.root)
        self.assertTrue(r['full_readback_checked'] and r['before_after_match'])
        self.assertEqual(r['matrix_entries'],1)
        missing=imported_evidence(self.root/'nonexistent')
        self.assertIsNone(missing['native_model_status'])
        self.assertFalse(missing['full_readback_checked'])

    def test_tampered_options_identity_or_readback_rejected(self):
        for mutation in ('options','hash','before','after','missing_import'):
            v=deepcopy(self.values)
            if mutation=='options':v['import.json']['options']['small_matrix_value']=1e-9
            elif mutation=='hash':v['import_status.json']['submitted_sha256']='other'
            elif mutation=='before':v['import.json']['readback']['rows'][0]['entries']=[]
            elif mutation=='after':v['capture.json']['readback_after']['rows'][0]['entries']=[]
            else:v['import.json']=None
            with self.assertRaises(ValueError):self.use(v)

    def test_partial_warning_not_fidelity_success(self):
        v=deepcopy(self.values);v['capture.json']=None
        v['import.json']['status']=v['import_status.json']['status']='HighsStatus.kWarning'
        self.assertFalse(self.use(v)['full_readback_checked'])

    def test_nonpositive_native_objective_not_checked_upper(self):
        rows=failed();self.assertEqual(aggregate(rows)['checked_upper_bounds'],0)
        self.values['raw_native.json']['native_objective']=-1e20
        self.assertNotIn('upper_bound',self.use(self.values))

    def test_input_bits_and_basis_counts_are_not_solution_claims(self):
        lp={'c':['1/8'],'lower':[-1],'upper':[1],'h':[],'b':[],
            'offset':0,'E':{'data':[]},'A':{'data':['257/512']}}
        r=input_bit_inventory(lp)
        self.assertEqual((r['max_numerator_bits'],r['max_denominator_bits']),(9,10))
        self.assertEqual(r['denominator_location'],'A.data[0]')
        self.assertIsNone(basis_inventory(None))
        m={'status':'MAPPED_HINT_ONLY','hint':{'basic_columns':[{'kind':'x'},{'kind':'E_residual'}],
            'anchors':[{'column':{'kind':'A_slack'},'at':'zero'}],'rows':[1,2]}}
        self.assertEqual(basis_inventory(m)['basic_counts'],{'x':1,'E_residual':1})


if __name__=='__main__':unittest.main()
