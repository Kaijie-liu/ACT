"""Bounded partial arithmetic records; never a certificate or resumed search."""
from pathlib import Path
import math
import time
from modular_basis.engine import Budget, POLICY, statistics
from single_check_portable.execution import read,save_new
from portable_proof.runtime import digest
from lp_sandwich.check import identity

JOURNAL_POLICY={'schema':'MODULAR_ARITHMETIC_JOURNAL_V1','operation_stride':100000,'max_events':2048,
                'phase_order':'bounded_modular_cycles_v1'}
PHASES=('setup','assembly','prime_schedule','finite_field','CRT','reconstruction','exact_residual','candidate')
NEXT={'setup':('assembly',), 'assembly':('prime_schedule','candidate'),
      'prime_schedule':('finite_field',), 'finite_field':('prime_schedule','CRT'),
      'CRT':('reconstruction',), 'reconstruction':('prime_schedule','exact_residual'),
      'exact_residual':('prime_schedule','candidate'), 'candidate':()}


def validate_progress(stats, previous=None):
    """Structural metadata check only, not a modular arithmetic certificate."""
    if not stats:
        if previous:raise ValueError('arithmetic progress disappeared')
        return
    if set(stats)!=set(statistics()):raise ValueError('modular progress fields')
    for key,value in stats.items():
        if key=='rounds':continue
        if type(value) is not int or value<0:raise ValueError('progress counter')
        if previous and value<previous[key]:raise ValueError('progress counter decreased')
    rounds=stats['rounds']
    if (not isinstance(rounds,list) or len(rounds)!=stats['primes_tried'] or
            stats['primes_tried']>POLICY['primes'] or stats['primes_used']>stats['primes_tried'] or
            stats['max_modulus_bits']>POLICY['max_bits'] or stats['max_field_product_bits']>60):
        raise ValueError('bounded modular round accounting')
    statuses={'SKIP_DENOMINATOR','SKIP_SINGULAR_FIELD','RECONSTRUCTION_INCOMPLETE',
              'EXACT_SYSTEM_RESIDUAL_ZERO','RECONSTRUCTION_REJECTED_BY_EXACT_RESIDUAL'}
    prev_prime=POLICY['prime_ceiling']
    for i,row in enumerate(rounds):
        if (set(row)-{'prime','status','modulus_bits'} or type(row['prime']) is not int or
                not 2<row['prime']<prev_prime or
                ('status' in row and row['status'] not in statuses) or
                ('status' not in row and i!=len(rounds)-1)):
            raise ValueError('round identity/order/status')
        if 'modulus_bits' in row and (type(row['modulus_bits']) is not int or
                                      not 0<row['modulus_bits']<=POLICY['max_bits']):
            raise ValueError('round modulus bounds')
        prev_prime=row['prime']
    if previous:
        old=previous['rounds']
        if rounds[:max(0,len(old)-1)]!=old[:-1]:raise ValueError('completed round prefix changed')
        if old and any(rounds[len(old)-1].get(k)!=v for k,v in old[-1].items()):
            raise ValueError('observed round metadata changed')


class TimedBudget(Budget):
    def __init__(self,deadline,root,*,begin,original_started,bindings):
        super().__init__(deadline)
        self.root=Path(root).resolve()
        if not self.root.is_relative_to(Path('/data1/Kane/MOE')):raise ValueError('outside workspace')
        self.begin,self.original_started=begin,original_started
        self.bindings={**bindings,'plan_sha256':digest((self.root/'plan.json').read_bytes()),
                       'original_started':original_started,'deadline_monotonic':deadline}
        self.phase,self.phase_started='setup',0.
        self.segments=[];self.sequence=0;self.closed=False;self.stats={}
        self.folder=self.root/'arithmetic_events';self.folder.mkdir(exist_ok=False)
        self.snapshot('start')

    def snapshot(self,reason):
        if self.sequence>=JOURNAL_POLICY['max_events']:raise ValueError('bounded journal exhausted')
        now=time.monotonic()
        save_new(self.folder/f'{self.sequence:04d}.json',{
            'schema':JOURNAL_POLICY['schema'],'policy':JOURNAL_POLICY,'sequence':self.sequence,
            'bindings':self.bindings,'reason':reason,'component_started':self.begin,
            'request_seconds':now-self.original_started,'component_seconds':now-self.begin,
            'phase':self.phase,'phase_started_seconds':self.phase_started,'closed':self.closed,
            'segments':self.segments,'context':self.context,'operations':self.operations,
            'arithmetic':self.arithmetic,'stats':self.stats,'evidence':'PARTIAL_DIAGNOSTIC_NOT_PROOF'})
        self.sequence+=1

    def where(self,phase,operation,**fields):
        super().where(phase,operation,**fields)
        if phase!=self.phase:
            if phase not in NEXT[self.phase]:
                raise ValueError('unregistered arithmetic phase transition')
            now=time.monotonic()-self.begin
            self.segments.append({'phase':self.phase,'start_seconds':self.phase_started,
                                  'end_seconds':now,'seconds':now-self.phase_started})
            self.phase,self.phase_started=phase,now
            self.snapshot('phase')

    def visit(self):
        super().visit()
        if self.operations%JOURNAL_POLICY['operation_stride']==0:self.snapshot('progress')

    def fail(self,*args,**kwargs):
        try:super().fail(*args,**kwargs)
        finally:self.snapshot('limit')

    def finish(self):
        now=time.monotonic()-self.begin
        self.segments.append({'phase':self.phase,'start_seconds':self.phase_started,
                              'end_seconds':now,'seconds':now-self.phase_started})
        self.phase_started=now;self.closed=True;self.snapshot('closed')

    def summary(self,seconds):
        observed=sum(s['seconds'] for s in self.segments)
        return {'policy':dict(JOURNAL_POLICY),'segments':self.segments,
                'observed_phase_seconds':observed,'component_residual_seconds':seconds-observed,
                'event_count':self.sequence,'component_started':self.begin,
                'scope':'nested constructor phases; progress serialization charged; not independent proofs'}


def journal_costs(root,*,allow_partial,construction=None):
    """Reconstruct recorded prefix. A malformed final write may be censored."""
    root=Path(root);paths=sorted((root/'arithmetic_events').glob('*.json'))
    if not paths:
        if construction is not None:raise ValueError('construction missing journal')
        return {'event_count':0,'complete':False,'segments':None,'current_phase':None,
                'current_phase_seconds':None,'censored':None}
    p=read(root/'plan.json');v=read(root/'prepared.json');m=read(root/'mapping.json')
    bindings={'lp_sha256':identity(v['lp']),'statement_sha256':identity(v['statement']),
              'hint_sha256':identity(m['hint']),'plan_sha256':digest((root/'plan.json').read_bytes()),
              'original_started':p['started'],'deadline_monotonic':p['started']+218}
    last=None;partial=None;previous_request=-1;component_start=None
    for i,path in enumerate(paths):
        if path.name!=f'{i:04d}.json' or i>=JOURNAL_POLICY['max_events']:raise ValueError('journal omission/order')
        try:r=read(path)
        except (ValueError,UnicodeError):
            if not allow_partial or i!=len(paths)-1:raise
            partial=str(path.relative_to(root));break
        if (r['schema']!=JOURNAL_POLICY['schema'] or r['policy']!=JOURNAL_POLICY or r['sequence']!=i or
                r['bindings']!=bindings or r['evidence']!='PARTIAL_DIAGNOSTIC_NOT_PROOF' or
                r['phase'] not in PHASES or not isinstance(r['closed'],bool)):
            raise ValueError('journal binding/schema')
        if last is not None and last['closed']:raise ValueError('journal continued after closure')
        validate_progress(r['stats'],None if last is None else last['stats'])
        if component_start is None:component_start=r['component_started']
        values=[r[k] for k in ('component_started','request_seconds','component_seconds','phase_started_seconds')]
        if (any(not math.isfinite(x) for x in values) or r['component_started']!=component_start or
                not p['started']<=component_start or not 0<=r['phase_started_seconds']<=r['component_seconds'] or
                r['request_seconds']<previous_request or
                abs(r['request_seconds']-r['component_seconds']-(component_start-p['started']))>1e-7):
            raise ValueError('journal clock')
        end=0.;prev_phase=None
        for s in r['segments']:
            if (s['phase'] not in PHASES or (prev_phase is None and s['phase']!='setup') or
                    (prev_phase is not None and s['phase'] not in NEXT[prev_phase]) or
                    not all(math.isfinite(s[k]) for k in ('start_seconds','end_seconds','seconds')) or
                    abs(s['start_seconds']-end)>1e-8 or not end<=s['end_seconds']<=r['component_seconds'] or
                    abs(s['seconds']-s['end_seconds']+s['start_seconds'])>1e-8):
                raise ValueError('arithmetic segment clock/order')
            end=s['end_seconds'];prev_phase=s['phase']
        if ((r['closed'] and (not r['segments'] or prev_phase!=r['phase'])) or
                (not r['closed'] and prev_phase is not None and r['phase'] not in NEXT[prev_phase]) or
                (not r['segments'] and r['phase']!='setup')):
            raise ValueError('cyclic phase/current segment mismatch')
        if abs(end-r['phase_started_seconds'])>1e-8:raise ValueError('arithmetic phase gap')
        if last and (r['segments'][:len(last['segments'])]!=last['segments'] or r['operations']<last['operations']):
            raise ValueError('arithmetic prefix changed')
        previous_request=r['request_seconds'];last=r
    if construction is not None:
        if last is None or partial or not last['closed']:raise ValueError('incomplete journal for complete construction')
        j=construction['journal'];observed=sum(s['seconds'] for s in last['segments'])
        if (j['policy']!=JOURNAL_POLICY or j['event_count']!=len(paths) or
                j['component_started']!=component_start or j['segments']!=last['segments'] or
                last['arithmetic']!=construction['arithmetic'] or last['stats']!=construction['stats'] or
                last['operations']!=construction['operations'] or
                last['component_seconds']>construction['seconds'] or
                abs(j['observed_phase_seconds']-observed)>1e-8 or j['component_residual_seconds']<0 or
                abs(observed+j['component_residual_seconds']-construction['seconds'])>1e-8):
            raise ValueError('construction phase accounting')
    elif not allow_partial:raise ValueError('missing construction result')
    if last is None:
        return {'event_count':0,'complete':False,'segments':None,'current_phase':None,
                'current_phase_seconds':None,'censored':True,'partial_record':partial}
    return {'event_count':last['sequence']+1,'complete':construction is not None,
            'component_started':component_start,'last_observed_request_seconds':last['request_seconds'],
            'segments':last['segments'],'current_phase':last['phase'],
            'current_phase_seconds':None if construction is None else last['segments'][-1]['seconds'],
            'current_phase_observed_window_seconds':max(0,last['component_seconds']-last['phase_started_seconds']),
            'censored':construction is None,'partial_record':partial,
            'arithmetic':last['arithmetic'],'operations':last['operations'],
            'modular_progress':last['stats'],
            'scope':'saved progress only; unobserved time is not zero and no point is certified'}
