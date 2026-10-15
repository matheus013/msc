"""PT-BR figures: saved outcomes in two distinct cohorts, not pooled costs."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
data=ROOT/'experiment/data/08_reporting/sca_article/dual'
out=ROOT/'docs/supply_chain_analytics_article'
payload=json.loads((data/'dual_results.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,cohort,label in zip(axes,payload['cohorts'],['Varejo brasileiro (BA)','M5: recorte FOODS']):
    rows=[r for r in cohort['summaries'] if r['alpha']==.7]
    baseline=next(r for r in rows if r['strategy']=='global_store_out')
    chosen=[next(r for r in rows if r['strategy']==name) for name in ['global_store_out','profile_store_out','legacy_EOQ']]
    for row,color,marker,title,offset in zip(chosen,['#24557a','#aa5429','#347d52'],['o','s','^'],['Global','Por perfil','EOQ legado'],[(4,-18),(4,8),(4,8)]):
        x=row['TIC_mean']/baseline['TIC_mean'];y=row['NS_mean']
        ax.scatter(x,y,color=color,marker=marker,s=65)
        ax.annotate(title,(x,y),xytext=offset,textcoords='offset points',fontsize=9)
    ax.axhline(.7,linestyle='--',color='#777777',linewidth=1)
    ax.set(xlabel='Custo relativo à global viável na própria base',ylabel='Serviço médio por série',
           title=label,ylim=(0,1.03))
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
fig.savefig(out/'Figure_dual_1.png',dpi=300);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
for ax,field,cifield,title in [(axes[0],'cost_reduction_percent','cost_reduction_ci','Redução de custo frente à global (%)'),(axes[1],'fill_difference_pp','fill_difference_ci_pp','Diferença de serviço (pontos percentuais)')]:
    for cohort,color,shift,label in zip(payload['cohorts'],['#24557a','#aa5429'],[-.003,.003],['BA','M5']):
        rows=cohort['contrasts'];x=np.array([r['alpha'] for r in rows])+shift;y=np.array([r[field] for r in rows]);lo=np.array([r[cifield][0] for r in rows]);hi=np.array([r[cifield][1] for r in rows])
        ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt='o-',capsize=3,color=color,label=label)
    ax.axhline(0,color='#777777',linewidth=1);ax.set(xlabel='Limiar de serviço na seleção',ylabel=title,xticks=[.7,.8,.85,.9,.95])
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);ax.legend(frameon=False)
fig.savefig(out/'Figure_dual_2.png',dpi=300);plt.close(fig)
print('Two PT-BR figures generated; costs normalized within cohort.')
