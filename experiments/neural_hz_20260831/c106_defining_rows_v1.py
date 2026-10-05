# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bind actual original operator arrays; never cache inferred numeric facts."""
from collections import Counter
from functools import partial
import scipy.sparse as sp
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831._c106_defining_rows_v1 import csr_row,conv_row,sum_row


def bind_node(node,nodes):
    """One structural lowering; original C104 encoder remains the consumer."""
    if node['kind']=='source':return None
    if node['kind']=='sum':
        parents=tuple((nodes[p]['support'],nodes[p]['slots'],nodes[p]['exponents'],int(number))
            for p,number in sorted(Counter(node['parents']).items()))
        return partial(sum_row,parents)
    if node['kind']!='op':raise ValueError('unsupported defining node')
    parent=nodes[node['parents'][0]];op=node['op']
    arrays=(parent['needed'],parent['slots'],parent['exponents'])
    if type(op) is sp.csr_matrix:
        return partial(csr_row,op.indptr,op.indices,op.data,*arrays)
    if type(op) is not ImplicitConv2DOp:raise ValueError('unsupported defining operator')
    geometry=tuple(map(int,(*op.input_shape,*op.output_shape[1:],*op._stride,
        *op._padding,*op._dilation,op._groups)))
    return partial(conv_row,op._kernel,*arrays,op._row_mask,geometry)
