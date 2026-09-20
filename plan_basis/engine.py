"""Separate plan-reuse research arithmetic; not a production/supervised interface."""
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
          'live_accounting': 'source_active_pivots_plus_plan_index_units',
          'plan_entries': 1000000, 'plan_reuse': 'checked_symbolic_schedule_V1'}

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
        self.phase_operations={}; self.phase_seconds={}
        self.clock=time.monotonic(); self.key='admission'
        self.plan_units=0
        self.arithmetic={'max_integer_bits':0,'max_fraction_numerator_bits':0,
                         'max_fraction_denominator_bits':0,'first_limit':None}

    def where(self,phase,operation,**fields):
        now=time.monotonic()
        self.phase_seconds[self.key]=self.phase_seconds.get(self.key,0.)+now-self.clock
        self.clock=now; self.key=phase+'/'+operation
        self.phase_operations.setdefault(self.key,0)
        self.context={'phase':phase,'operation':operation,**fields}

    def tick(self):
        if time.monotonic()>=self.deadline:raise TimeoutError('original deadline')

    def visit(self):
        self.tick();self.operations+=1
        self.phase_operations[self.key]=self.phase_operations.get(self.key,0)+1
        if self.operations>POLICY['operations']:
            self.fail('operations',self.operations,POLICY['operations'])

    def costs(self):
        now=time.monotonic()
        seconds=dict(self.phase_seconds)
        seconds[self.key]=seconds.get(self.key,0.)+now-self.clock
        return {'operations':dict(self.phase_operations),'seconds':seconds,
                'scope':'nested instrumented arithmetic including plan work; not request total'}

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


def solve_field(system,p,b,stats,*,plan=None,scope='analytic',capture=True):
    from .field import solve
    return solve(system,p,b,stats,plan=plan,scope=scope,capture=capture)


def reconstruct(residue, modulus, b, detail=None):
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
    reason = ('zero_denominator' if t1 == 0 else
              'denominator_bound' if abs(t1)>bound else
              'not_coprime' if gcd(r1,t1)!=1 else
              'denominator_not_invertible' if gcd(t1,modulus)!=1 else None)
    if detail is not None:
        detail.update({'bound_bits':bound.bit_length(),'failure_reason':reason})
    if reason is not None:
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


def eliminate(system, b, stats, *, reuse=True, scope='analytic'):
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
    plan = None
    stats.update({'plan_builds':0,'plan_replays':0,'plan_invalidations':[],
                  'peak_plan_units':0})
    for p in primes(b,stats):
        stats['primes_tried'] += 1
        record = {'prime':p}
        stats['rounds'].append(record)
        try:
            xmod, plan = solve_field(system,p,b,stats,plan=plan if reuse else None,
                                     scope=scope,capture=reuse)
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
        progress={'successful_prefix':0,'failed_coordinate':None,'failure_reason':None,
                  'complete_vector':False,'status':'IN_PROGRESS'}
        record['reconstruction_progress']=progress
        for coordinate,value in enumerate(residue):
            detail={}
            try:
                v = reconstruct(value,modulus,b,detail)
            except (Limit,TimeoutError):
                progress.update(failed_coordinate=coordinate,status='INTERRUPTED')
                raise
            if v is None:
                progress.update(detail,failed_coordinate=coordinate,status='INCOMPLETE')
                break
            values.append(v)
            progress['successful_prefix']=len(values)
        if len(values)==m:
            progress.update(complete_vector=True,status='AWAITING_EXACT_RESIDUAL')
        if len(values) != m:
            record['status'] = 'RECONSTRUCTION_INCOMPLETE'
        elif verify_residual(system,values,b):
            progress['status']='EXACT_SYSTEM_RESIDUAL_ZERO'
            record['status'] = 'EXACT_SYSTEM_RESIDUAL_ZERO'
            b.tick()
            return values
        else:
            progress['status']='REJECTED_BY_EXACT_RESIDUAL'
            stats['residual_rejections'] += 1
            record['status'] = 'RECONSTRUCTION_REJECTED_BY_EXACT_RESIDUAL'
    raise Unresolved('fixed modular schedule exhausted; no LP or network verdict')
