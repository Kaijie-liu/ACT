"""Saved-record diagnostics and two figures; zero model or solver calls."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS=('D-act','D-tot','M','M-reg')
COLORS=('#276484','#638858','#b64945','#c18b35')


def derive(source):
    a=source['arms'];rows=[]
    for name in ARMS:
        x=a[name];c=x['clean_correct']/x['denominator'];r=x['certified']/x['clean_correct']
        width=np.mean([d['mean_margin_width'] for d in source['details'] if d['arm']==name])
        assert np.isclose(c*r,x['certified_accuracy'])
        rows.append(dict(arm=name,clean=c,conditional=r,certified=c*r,width=float(width),
                         single_route=x.get('single_route_fraction')))
    M=rows[2];D=rows[0]
    clean=(M['clean']-D['clean'])*(M['conditional']+D['conditional'])/2
    conditional=(M['conditional']-D['conditional'])*(M['clean']+D['clean'])/2
    assert abs(clean+conditional-(M['certified']-D['certified']))<1e-15
    return dict(schema='S2A_DESCRIPTIVE_DECOMPOSITION_V1',rows=rows,
        m_minus_dact=dict(total_pp=100*(M['certified']-D['certified']),
            clean_component_pp=100*clean,conditional_component_pp=100*conditional,
            formula='Delta(CR)=DeltaC*(R_M+R_D)/2+DeltaR*(C_M+C_D)/2'),
        model_calls=0,causal_identification=False,
        scope='Each model has its own clean-correct set; learned-scale widths are descriptive.')


def render(source,data,output):
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(1,2,figsize=(8.4,3.1),constrained_layout=True)
    x=np.arange(4);clean=np.array([r['clean'] for r in data['rows']])*100
    cert=np.array([r['certified'] for r in data['rows']])*100
    ax[0].bar(x,cert,color=COLORS,label='Certified')
    ax[0].bar(x,clean-cert,bottom=cert,color=COLORS,alpha=.3,label='Clean correct, not certified')
    ax[0].set(xticks=x,xticklabels=ARMS,ylabel='All test inputs (%)',ylim=(0,59),title='(a) Accuracy decomposition')
    for i,(c,v) in enumerate(zip(clean,cert)):
        ax[0].text(i,v/2,f'{v:.2f}',ha='center',va='center',color='white')
        ax[0].text(i,c+1,f'{c:.2f}',ha='center',fontsize=8)
    ax[0].legend(fontsize=7,loc='upper right')
    cond=np.array([r['conditional'] for r in data['rows']])*100
    ax[1].bar(x,cond,color=COLORS,width=.65)
    for i,name in enumerate(ARMS):
        dots=[100*d['certified']/d['clean'] for d in source['details'] if d['arm']==name]
        ax[1].scatter(i+np.linspace(-.16,.16,len(dots)),dots,c='black',s=12,zorder=3)
        ax[1].text(i,cond[i]+2,f'{cond[i]:.2f}',ha='center',fontsize=8)
    ax[1].set(xticks=x,xticklabels=ARMS,ylabel='Certified / clean correct (%)',ylim=(0,75),title='(b) Different conditional cohorts')
    for a in ax:a.spines[['top','right']].set_visible(False)
    fig.savefig(output/'s2a_accuracy_decomposition.pdf');fig.savefig(output/'s2a_accuracy_decomposition.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.1,3.4),constrained_layout=True)
    for name,color,row in zip(ARMS,COLORS,data['rows']):
        ds=[d for d in source['details'] if d['arm']==name]
        ax.scatter([d['mean_margin_width'] for d in ds],[100*d['certified']/d['count'] for d in ds],color=color,s=34,label=name)
        ax.scatter([row['width']],[100*row['certified']],color=color,marker='D',s=80,edgecolors='black')
    ax.set(xlabel='Mean projected margin-interval width (learned scale)',ylabel='IBP certified accuracy (%)',title='Narrower intervals do not imply more certification')
    ax.legend(ncol=2,fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.savefig(output/'s2a_width_certification.pdf');fig.savefig(output/'s2a_width_certification.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('paper/results/s2a_primary.json'))
    p.add_argument('--output',type=Path,default=Path('paper/figures'))
    p.add_argument('--data',type=Path,default=Path('paper/results/s2a_decomposition.json'));args=p.parse_args()
    raw=args.source.read_bytes();source=json.loads(raw);data=derive(source)
    data['source_sha256']=hashlib.sha256(raw).hexdigest()
    args.data.write_text(json.dumps(data,indent=2)+'\n');render(source,data,args.output)
