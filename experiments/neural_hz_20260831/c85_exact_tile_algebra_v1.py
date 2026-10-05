"""Published integer tile identity plus exact, box-redundant affine equations."""
from fractions import Fraction as F
import math
import numpy as np

T=np.array([[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]],np.int64)
H=np.array([[2,0,0],[1,1,1],[1,-1,1],[0,0,2]],np.int64)
R=np.array([[1,1,1,0],[0,1,-1,-1]],np.int64)


def direct(d,g):
    """Independent spatial cross-correlation, with no transform matrices."""
    return [[sum((d[i+a][j+b]*g[a][b] for a in range(3) for b in range(3)),F(0))
             for j in range(2)] for i in range(2)]


def prove_basis():
    checks=0
    for kernel in range(9):
        g=np.zeros((3,3),np.int64);g.flat[kernel]=1
        for coordinate in range(16):
            d=np.zeros((4,4),np.int64);d.flat[coordinate]=1
            actual=R@((H@g@H.T)*(T@d@T.T))@R.T
            expected=direct(d.tolist(),g.tolist())
            if any(F(int(actual[i,j]),4)!=expected[i][j] for i in range(2) for j in range(2)):
                raise ValueError('integer tile polynomial identity failed')
            checks+=4
    return dict(kernel_basis=9,input_basis=16,output_coefficients_proved=checks,
                full_bilinear_identity=True,sampled_point_proof=False)


def power_bound(value):
    """Smallest positive dyadic >= the exact nonnegative bound; zero uses1."""
    value=F(value)
    if value<0:raise ValueError('negative affine norm')
    if not value:return F(1)
    exponent=value.numerator.bit_length()-value.denominator.bit_length()
    p=F(2)**exponent
    return p if p>=value else 2*p


def equation(terms,constant,slot):
    """An actual sparse equation p*z - terms = constant, exact Fractions."""
    merged={}
    for col,value in terms:
        if not 0<=col<slot:raise ValueError('original shared parent/topological slot required')
        merged[col]=merged.get(col,F(0))+F(value)
    merged={k:v for k,v in sorted(merged.items()) if v}
    constant=F(constant);norm=abs(constant)+sum(map(abs,merged.values()),F(0))
    pivot=power_bound(norm)
    return dict(slot=slot,coefficients=tuple([*((k,-v) for k,v in merged.items()),(slot,pivot)]),
                rhs=constant,pivot=pivot,box_bound=norm)


def reconstruct_actual(rows,original):
    """Read actual emitted equations, not a parallel lift-formula registry."""
    point=list(map(F,original))
    for row in rows:
        slot=row['slot']
        if slot!=len(point):raise ValueError('complete topological equation sequence required')
        entries=dict(row['coefficients']);pivot=entries.pop(slot)
        value=(row['rhs']-sum((v*point[c] for c,v in entries.items()),F(0)))/pivot
        if abs(value)>1:raise ValueError('new factor box was not redundant for original input')
        point.append(value)
    return point


def tile_input_equations(global_ids,centers,scales,first_slot):
    """Share original IDs; pad cells have ID -1 and exact constant zero."""
    ids=np.asarray(global_ids);centers=np.asarray(centers,dtype=object);scales=np.asarray(scales,dtype=object)
    if ids.shape!=(4,4) or centers.shape!=(4,4) or scales.shape!=(4,4):
        raise ValueError('complete4x4 tile required')
    rows=[]
    for a in range(4):
        for b in range(4):
            terms=[];constant=F(0)
            for i in range(4):
                for j in range(4):
                    q=int(T[a,i])*int(T[b,j])
                    if not q:continue
                    col=int(ids[i,j])
                    if col==-1:
                        if centers[i,j] or scales[i,j]:raise ValueError('padding must remain exact zero')
                        continue
                    constant+=q*F(centers[i,j]);terms.append((col,q*F(scales[i,j])))
            rows.append(equation(terms,constant,first_slot+len(rows)))
    return rows


def recover_filter(numerator):
    """Independent inverse from selected actual H*g*H^T entries, times4.

    For a column transform H, entries0/3 give endpoints, entries1-2 give
    twice the middle. Apply that DIFFERENT inverse twice, avoiding division.
    The output equals4 times the original aligned integer kernel.
    """
    a=np.asarray(numerator)
    if a.shape[-2:]!=(4,4) or a.dtype!=np.int64:
        raise ValueError('actual signed integer transformed4x4 tensor required')
    first=np.stack((a[...,0,:],a[...,1,:]-a[...,2,:],a[...,3,:]),axis=-2)
    return np.stack((first[...,0],first[...,1]-first[...,2],first[...,3]),axis=-1)
