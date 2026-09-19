"""Small-prime sparse elimination, bounded CRT, exact-residual-gated reconstruction."""
from fractions import Fraction as F
from hashlib import sha256
from math import gcd, isqrt
import heapq
import math
import time
from lp_sandwich.check import rational, identity
from sparse_basis.engine import POLICY as OLD_POLICY, Limit, Singular

POLICY = {**OLD_POLICY, 'arithmetic': 'bounded_multimodular_rational_reconstruction_V1',
          'primes': 128, 'prime_candidates': 4096, 'prime_ceiling': 2**30,
          'reconstruction': 'symmetric_bound_floor_sqrt((M-1)/2)_exact_residual_required',
          'live_accounting': 'exact_source_plus_active_field_plus_retained_field_pivots'}

class Unresolved(Exception):
    pass

class BadPrime(Exception):
    pass

class Budget:
    def __init__(self,deadline):
        now=time.monotonic()
        if not math.isfinite(deadline) or deadline>now+300:raise ValueError('bounded original deadline required')
        self.deadline,self.operations=deadline,0
        self.context={'phase':'admission'}
        self.arithmetic={'max_integer_bits':0,'max_fraction_numerator_bits':0,
                         'max_fraction_denominator_bits':0,'first_limit':None}

    def where(self,phase,operation,**fields):
        self.context={'phase':phase,'operation':operation,**fields}

    def tick(self):
        if time.monotonic()>=self.deadline:raise TimeoutError('original deadline')

    def visit(self):
        self.tick();self.operations+=1
        if self.operations>POLICY['operations']:
            self.fail('operations',self.operations,POLICY['operations'])

    def fail(self,kind,observed,cap,**extra):
        if self.arithmetic['first_limit'] is None:
            self.arithmetic['first_limit']={**self.context,'kind':kind,'observed':observed,'cap':cap,**extra}
        raise Limit(kind+' budget')

    def integer(self,v):
        self.visit()
        if type(v) is not int:raise ValueError('integer arithmetic required')
        bits=abs(v).bit_length()
        self.arithmetic['max_integer_bits']=max(self.arithmetic['max_integer_bits'],bits)
        if bits>POLICY['max_bits']:
            self.fail('integer bits',bits,POLICY['max_bits'],value_sha256=sha256(format(v,'x').encode()).hexdigest())
        return v

    def value(self,v):
        self.visit();v=v if isinstance(v,F) else rational(v)
        n,d=abs(v.numerator).bit_length(),v.denominator.bit_length()
        self.arithmetic['max_fraction_numerator_bits']=max(self.arithmetic['max_fraction_numerator_bits'],n)
        self.arithmetic['max_fraction_denominator_bits']=max(self.arithmetic['max_fraction_denominator_bits'],d)
        if max(n,d)>POLICY['max_bits']:
            self.fail('rational bits',max(n,d),POLICY['max_bits'],numerator_bits=n,denominator_bits=d,
                      value_sha256=sha256((format(v.numerator,'x')+'/'+format(v.denominator,'x')).encode()).hexdigest())
        return v


def statistics():
    result = dict.fromkeys(('peak_active_nnz','peak_live_nnz','peak_heap_entries',
        'fill_insertions','heap_pops','heap_rebuilds','pivots','pivot_row_candidates',
        'singleton_column_pivots','row_updates','max_field_product_bits',
        'prime_candidates_tested','primes_tried','primes_used','bad_denominator_primes',
        'singular_primes','reconstruction_attempts','residual_rejections',
        'max_modulus_bits'), 0)
    result['rounds'] = []
    return result


def primes(b, stats):
    """Deterministic <=30-bit primes. All trial division consumes shared budget."""
    emitted = 0
    for offset in range(POLICY['prime_candidates']):
        b.where('prime_schedule','trial_division'); b.visit()
        p = POLICY['prime_ceiling'] - 1 - 2*offset
        if p < 3:
            return
        stats['prime_candidates_tested'] += 1
        prime = True
        for divisor in range(3, isqrt(p)+1, 2):
            b.visit()
            if p % divisor == 0:
                prime = False
                break
        if prime:
            yield p
            emitted += 1
            if emitted >= POLICY['primes']:
                return


def product(a, c, p, b, stats):
    b.visit()
    v = a*c
    stats['max_field_product_bits'] = max(stats['max_field_product_bits'], v.bit_length())
    if not (0 <= a < p and 0 <= c < p and p < 2**30):
        raise ValueError('field product domain')
    return v % p


def field_value(v, p, b, stats):
    b.visit()
    den = v.denominator % p
    if den == 0:
        raise BadPrime('input denominator divisible by prime')
    return product(v.numerator % p, pow(den, -1, p), p, b, stats)


def solve_field(system, p, b, stats):
    """Independent sparse factorization; no exact cross-products or row LCM."""
    m = len(system)
    source_nnz = sum(len(row) for row, _ in system)
    rows, right = {}, {}
    b.where('finite_field','map',prime=p)
    for i, (row, rhs) in enumerate(system):
        rows[i] = {}
        for j, v in row.items():
            value = field_value(v, p, b, stats)
            if value:
                rows[i][j] = value
        right[i] = field_value(rhs, p, b, stats)
    incidence = {j:set() for j in range(m)}
    for i,row in rows.items():
        for j in row:
            b.visit(); incidence[j].add(i)
    active = sum(map(len, rows.values()))
    retained = 0
    heap = [(len(ids),j) for j,ids in incidence.items()]
    heapq.heapify(heap)
    pivots = []

    def observe():
        b.tick()
        stats['peak_active_nnz'] = max(stats['peak_active_nnz'], active)
        stats['peak_live_nnz'] = max(stats['peak_live_nnz'], source_nnz+active+retained)
        stats['peak_heap_entries'] = max(stats['peak_heap_entries'], len(heap))
        for name,value in (('live_nnz',source_nnz+active+retained),
                           ('fill_insertions',stats['fill_insertions']),
                           ('heap_entries',len(heap))):
            if value > POLICY[name]:
                b.fail(name,value,POLICY[name])
    observe()
    b.where('finite_field','elimination',prime=p)
    while rows:
        b.visit()
        while heap:
            degree,col = heapq.heappop(heap)
            stats['heap_pops'] += 1; b.visit()
            if col in incidence and degree == len(incidence[col]):
                break
        else:
            raise Singular('no active pivot')
        if not degree:
            raise Singular('singular over this finite field only')
        stats['pivot_row_candidates'] += degree
        pivot_id = min(incidence[col], key=lambda i:(len(rows[i]),i))
        pivot = rows.pop(pivot_id)
        rhs = right.pop(pivot_id)
        active -= len(pivot)
        inv = pow(pivot[col],-1,p)
        normalized = {j:product(v,inv,p,b,stats) for j,v in pivot.items()}
        target = product(rhs,inv,p,b,stats)
        retained += len(normalized)
        pivots.append((col,normalized,target))
        stats['pivots'] += 1
        if degree == 1:
            stats['singleton_column_pivots'] += 1
        affected = sorted(incidence[col]-{pivot_id})
        changed = set(pivot)
        for j in pivot:
            incidence[j].remove(pivot_id)
        del incidence[col]
        for i in affected:
            b.visit(); stats['row_updates'] += 1
            row = rows[i]; factor = row.pop(col); active -= 1
            right[i] = (right[i]-product(factor,target,p,b,stats)) % p
            for j,v in normalized.items():
                if j == col:
                    continue
                existed = j in row
                value = (row.get(j,0)-product(factor,v,p,b,stats)) % p
                if value:
                    row[j] = value
                    if not existed:
                        active += 1
                        stats['fill_insertions'] += 1
                        incidence[j].add(i)
                elif existed:
                    del row[j]; active -= 1; incidence[j].remove(i)
                changed.add(j)
                observe()
            if not row:
                raise Singular('dependent selected basis modulo prime; no LP verdict')
        for j in changed:
            if j in incidence:
                heapq.heappush(heap,(len(incidence[j]),j))
        if len(heap) > max(1024,4*len(incidence)):
            heap = [(len(ids),j) for j,ids in incidence.items()]
            heapq.heapify(heap); stats['heap_rebuilds'] += 1
        observe()
    x = [0]*m
    b.where('finite_field','back_substitution',prime=p)
    for col,row,rhs in reversed(pivots):
        for j,v in row.items():
            if j != col:
                rhs = (rhs-product(v,x[j],p,b,stats)) % p
        x[col] = rhs
    return x


def reconstruct(residue, modulus, b):
    """Euclidean reconstruction. No residue*denominator congruence cross-product.

    Exact original equations, not reconstruction heuristics, gate output.
    """
    b.where('reconstruction','euclidean')
    bound = isqrt((modulus-1)//2)
    r0,r1,t0,t1 = modulus,residue,0,1
    while abs(r1) > bound:
        b.visit()
        q = r0//r1
        r0,r1 = r1,b.integer(r0-b.integer(q*r1))
        t0,t1 = t1,b.integer(t0-b.integer(q*t1))
    if t1 == 0 or abs(t1)>bound or gcd(r1,t1)!=1 or gcd(t1,modulus)!=1:
        return None
    return b.value(F(r1,t1))


def verify_residual(system, values, b):
    b.where('exact_residual','original_rational_equations')
    for row,rhs in system:
        total = F(0)
        for j,v in row.items():
            total = b.value(total+b.value(v*values[j]))
        if total != rhs:
            return False
    return True


def eliminate(system, b, stats):
    m = len(system)
    if m > POLICY['equations']:
        b.fail('equations',m,POLICY['equations'])
    nnz = sum(len(row) for row,_ in system)
    if nnz > POLICY['input_nnz']:
        b.fail('input_nnz',nnz,POLICY['input_nnz'])
    for i,(row,rhs) in enumerate(system):
        if any(type(j) is not int or not 0<=j<m for j in row):
            raise ValueError('basis column range')
        for j,v in row.items():
            row[j] = b.value(v)
        system[i] = (row,b.value(rhs))
    b.tick()
    if not m:
        return []
    residue,modulus = [0]*m,1
    for p in primes(b,stats):
        stats['primes_tried'] += 1
        record = {'prime':p}
        stats['rounds'].append(record)
        try:
            xmod = solve_field(system,p,b,stats)
        except BadPrime:
            record['status'] = 'SKIP_DENOMINATOR'
            stats['bad_denominator_primes'] += 1
            continue
        except Singular:
            record['status'] = 'SKIP_SINGULAR_FIELD'
            stats['singular_primes'] += 1
            continue
        b.where('CRT','merge',prime=p)
        if len(xmod)!=m or any(type(v) is not int or not 0<=v<p for v in xmod):
            raise ValueError('field solution dimensions/range')
        next_modulus = b.integer(modulus*p)
        inverse = pow(modulus % p,-1,p)
        for j,v in enumerate(xmod):
            delta = product((v-residue[j]) % p,inverse,p,b,stats)
            residue[j] = b.integer(residue[j]+b.integer(modulus*delta))
        modulus = next_modulus
        stats['primes_used'] += 1
        stats['max_modulus_bits'] = max(stats['max_modulus_bits'],modulus.bit_length())
        record['modulus_bits'] = modulus.bit_length()
        stats['reconstruction_attempts'] += 1
        values = []
        for value in residue:
            v = reconstruct(value,modulus,b)
            if v is None:
                break
            values.append(v)
        if len(values) != m:
            record['status'] = 'RECONSTRUCTION_INCOMPLETE'
        elif verify_residual(system,values,b):
            record['status'] = 'EXACT_SYSTEM_RESIDUAL_ZERO'
            b.tick()
            return values
        else:
            stats['residual_rejections'] += 1
            record['status'] = 'RECONSTRUCTION_REJECTED_BY_EXACT_RESIDUAL'
    raise Unresolved('fixed modular schedule exhausted; no LP or network verdict')
