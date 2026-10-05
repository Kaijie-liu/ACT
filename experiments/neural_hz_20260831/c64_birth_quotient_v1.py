"""Birth-integrated complete precision statistics and exact owned-row quotient."""
from collections import Counter
import hashlib
import json
import math
import numpy as np
from experiments.neural_hz_20260831.c57_scalar_consumer_probe_v2 import native_word
from experiments.neural_hz_20260831.c62_precision_plan_v1 import odd_significands
from experiments.neural_hz_20260831.c62_local_equations_v1 import encode,SCHEMA
from experiments.neural_hz_20260831.c64_compact_boundary_dp_v1 import choose
from experiments.neural_hz_20260831.c64_gauged_products_v1 import products,gauge_definition
from experiments.neural_hz_20260831.c18_sparse_rewrite_v1 import sort_cost


class BirthTracker:
    def __init__(self,encoder,old_nc,old_eq,output_slots):
        self.encoder=encoder;self.pool=encoder.pool;self.old_nc=old_nc;self.old_eq=old_eq
        nc=encoder.nc+16384
        # The unchanged C31 32*MAIN metadata allowance now pays classification,
        # raw/parent/ratio flags, complete row maps and per-birth registration.
        self.raw=np.zeros(nc,bool);self.protected=np.zeros(nc,bool);self.protected[output_slots]=True
        self.maxima=np.zeros(nc,np.uint64);self.parents={};self.local={};self.defining={}
        self.tags={};self.numerators={};self.equations={};self.hits=[]
        self.blocks=[];self.current_possible=False;self.current_dyadic=False
        self.external_rows=self.external_occurrences=self.inspected_rows=self.inspected_coefficients=0
        self.dyadic_maximum_occurrences=0

    def begin(self,node,coordinates):
        self.pool.charge('c64_birth_block_dispatch',32+8*len(node['parents']))
        self.current_possible=any(self.blocks[p] for p in node['parents'])
        counts=Counter(node['parents'])
        self.current_dyadic=node['kind']=='sum' and all(n>0 and n&(n-1)==0 for n in counts.values())
        self.node_slots=node['slots'][coordinates]

    def end(self):
        self.blocks.append(bool(self.raw[self.node_slots].any()));self.node_slots=None

    def collect(self,row,dyadic=False):
        cc,cv,bc,bv,rhs=self.encoder.eq[row]
        self.pool.charge('c63_routed_incidence_row',16+3*len(cc))
        self.inspected_rows+=1;self.inspected_coefficients+=len(cc)
        positions=np.flatnonzero(self.raw[cc])
        if not len(positions):return
        ids=cc[positions];self.external_rows+=1;self.external_occurrences+=len(ids)
        if dyadic:
            # Source sum multiplicities are powers of two and the exact encoder
            # only applies powers of two, including its pivot/radix links.
            self.pool.charge('c64_source_dyadic_maximum',32+8*len(cc)+2*len(ids))
            np.maximum.at(self.maxima,ids,1);self.dyadic_maximum_occurrences+=len(ids)
        else:
            self.pool.charge('c62_complete_external_maximum',32+8*len(cc)+16*len(ids))
            np.maximum.at(self.maxima,ids,odd_significands(cv[positions]))
        self.hits.append((row,positions))

    def born(self,slot,physical,first):
        cc,cv,bc,bv,rhs=self.encoder.eq[physical]
        candidate=(not self.protected[slot] and len(cc)==2 and not len(bc) and rhs==0. and cc[-1]==slot)
        if candidate:
            parent=int(cc[0]);pm,pe=math.frexp(float(cv[-1]))
            if not 0<=parent<slot or pm!=.5:raise ValueError('positive dyadic topological birth required')
            r=math.ldexp(-float(cv[0]),1-pe)
            if math.ldexp(r,pe-1)!=-float(cv[0]) or not 2.**-60<=abs(r)<=1.:
                raise ValueError('exact bounded contractive local birth required')
            word=native_word(r)
            if word not in self.equations:
                self.pool.charge('c62_distinct_local_equation_encoding',128)
                self.equations[word]=encode(0,word)
            tag,numerator=self.equations[word]
            self.raw[slot]=True;self.parents[slot]=parent;self.local[slot]=word;self.defining[slot]=physical
            self.tags[slot]=tag+parent;self.numerators[slot]=numerator
        for row in range(first,len(self.encoder.eq)):
            # Every extra physical row is conservatively inspected. Raw own
            # defining rows are covered by the exact recurrence, not maxima.
            if row!=physical:self.collect(row,self.current_dyadic)
            elif self.current_possible and not candidate:self.collect(row,self.current_dyadic)

    def fold(self,eq_roots,eq_scales,ineq_scales,*,observe=None):
        encoder=self.encoder;pool=self.pool;nc=encoder.nc+len(encoder.def_rows)
        selected,roots,weights,stats=choose(self.parents,self.local,self.maxima,nc,pool=pool)
        chosen=np.flatnonzero(selected);old_nnz=sum(len(r[0])+len(r[2]) for rows in (encoder.eq,encoder.ineq) for r in rows)
        if not len(chosen):raise ValueError('no strict original-boundary quotient')
        erased=np.zeros(len(encoder.eq),bool);shifts=np.zeros(len(encoder.eq),np.int64)
        rewritten=general=occurrences=0;shift_hist=Counter()
        for v0 in chosen:
            v=int(v0);row=self.defining[v];erased[row]=True
            encoder.ledger.change(encoder.eq[row][0],encoder.eq_uids[row],-1)
        pending=list(self.hits)
        # A retained raw defining row has one parent edge; this complete set
        # never depended on the skipped external-consumer scan.
        for v,p in self.parents.items():
            if not selected[v] and selected[p]:pending.append((self.defining[v],np.array([0],np.int64)))
        for row,positions in pending:
            if erased[row]:continue
            cc,cv,bc,bv,rhs=encoder.eq[row];active=positions[selected[cc[positions]]]
            if not len(active):continue
            pool.charge('product_operand_gathers',2*len(active))
            ratios=np.array([math.ldexp(float(weights[int(c)][0]),weights[int(c)][1]) for c in cc[active]])
            # These native ratios are derived only where a consumer exists.
            if any(native_word(x)!=weights[int(c)] for c,x in zip(cc[active],ratios)):raise ValueError('non-native retained-boundary multiplier')
            multiplied,proof=products(cv[active],ratios,pool=pool);general+=proof['general'];occurrences+=len(active)
            pool.charge('rewrite_sort',sort_cost(len(cc)))
            old_columns=cc[active].copy();new_columns=roots[old_columns]
            encoder.ledger.change(old_columns,encoder.eq_uids[row],-1)
            encoder.ledger.change(new_columns,encoder.eq_uids[row],1)
            cc[active]=new_columns;cv[active]=multiplied
            order=np.argsort(cc,kind='stable');cc,cv=cc[order],cv[order]
            if np.any(np.diff(cc)<=0):raise ValueError('coalescence outside complete precision-boundary class')
            if selected[cc[-1]]:raise ValueError('defining pivot was selected for removal')
            cv,bv,rhs,q=gauge_definition(cv,bv,rhs,proof['minimum_absolute_product'],pool=pool)
            encoder.eq[row]=(cc,cv,bc,bv,rhs);shifts[row]=q;shift_hist[q]+=1;rewritten+=1
        encoder.ledger.finish()
        if np.any(encoder.ledger.words[chosen-self.old_nc]):raise ValueError('erased MAIN still has an actual incidence')
        row_map=np.cumsum(~erased,dtype=np.int64)-1
        er=row_map[np.asarray(eq_roots,np.int64)].copy()
        es=np.asarray(eq_scales,np.int64)+shifts[np.asarray(eq_roots,np.int64)]
        for v0 in chosen:
            v=int(v0);i=self.old_eq+v-self.old_nc;er[i]=self.tags[v];es.view(np.float64)[i]=self.numerators[v]
        def_rows=np.asarray(encoder.def_rows,np.int64);radix_gauges=shifts[def_rows].copy()
        encoder.def_rows=[int(row_map[r]) for r in def_rows]
        encoder.eq=[value for i,value in enumerate(encoder.eq) if not erased[i]]
        encoder.eq_uids=[uid for i,uid in enumerate(encoder.eq_uids) if not erased[i]]
        new_nnz=sum(len(r[0])+len(r[2]) for rows in (encoder.eq,encoder.ineq) for r in rows)
        encoder.entries=new_nnz+len(encoder.eq)+len(encoder.ineq)
        if new_nnz!=old_nnz-2*len(chosen):raise ValueError('complete strict nnz identity differs')
        digest=hashlib.sha256()
        for v in sorted(self.parents):
            digest.update(json.dumps([v,self.parents[v],self.local[v],int(self.maxima[v]),bool(selected[v]),int(roots[v]),weights[v]]).encode())
        report=dict(schema='c64_owned_birth_precision_quotient_v1',lineage_schema=SCHEMA,**stats,
            external_rows=self.external_rows,external_occurrences=self.external_occurrences,
            inspected_rows=self.inspected_rows,inspected_coefficients=self.inspected_coefficients,
            dyadic_maximum_occurrences=self.dyadic_maximum_occurrences,identity_sha256=digest.hexdigest(),
            old_predicate_nnz=old_nnz,new_predicate_nnz=new_nnz,rewritten_rows=rewritten,
            exact_native_general_products=general,rewritten_occurrences=occurrences,row_gauges=dict(shift_hist),
            coupled_extra_work=pool.used,coupled_extra_capacity=pool.capacity,work_parts=dict(pool.parts),
            all_removed_actual_owner_words_zero=True,old_independent_frontier_retirement_not_used=True,
            source_first_native_or_LIVE_admission=False,formal_gain=0)
        if observe:observe('c64_complete_owned_quotient',report)
        return er,es,radix_gauges,report
