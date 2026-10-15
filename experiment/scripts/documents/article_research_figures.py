"""Build manuscript figures from the completed, balanced Bahia evaluation.

No training, policy selection or experiment output is modified.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT/'experiment/data/08_reporting/research_v2/bahia145_cpu16_2026-10-15'
DOC = ROOT/'docs/supply_chain_analytics_article'
OUT = DOC/'figures'
OUT.mkdir(exist_ok=True)
status = json.loads((RUN/'status.json').read_text())
assert status['state'] == 'completed'
raw = pd.read_csv(RUN/'kpis.csv', dtype={'store_id':str,'item_id':str,'warehouse':str})
keys = ['dataset','origin','warehouse','store_id','item_id']
test = raw[raw.block.eq('test')].groupby(keys+['policy'], observed=True)[['TIC','NS','Demand','Served']].mean().reset_index()
assert len(test) == 2030
# Selections were audited against the completed experiment; do not choose on test.
selection_path = ROOT/'docs/scientific_writing_review_2026-10-15/bahia_completed/selections.csv'
sel = pd.read_csv(selection_path)
sel = sel[np.isclose(sel.alpha,.7)].copy()
assert len(sel) == 580
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.facecolor':'white','axes.labelsize':11,'savefig.dpi':300,
                     'font.family':'serif','pgf.texsystem':'pdflatex','pgf.rcfonts':False,
                     'pgf.preamble':r'\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}\usepackage{mathptmx}'})
blue, orange, gray = '#176B87', '#BE5727', '#89939A'

def save(fig, name):
    fig.savefig(OUT/f'{name}.png', bbox_inches='tight', facecolor='white')
    # PGF uses the installed TeX engine and embeds fonts in a vector PDF.
    # This avoids dependence on the unavailable fontTools PDF embedding DLL.
    fig.savefig(OUT/f'{name}.pdf', backend='pgf', bbox_inches='tight')
    plt.close(fig)

# An explicit information path prevents confusion between fitting and selection.
fig, ax = plt.subplots(figsize=(6.5,3.8))
ax.set(xlim=(0,1), ylim=(0,1)); ax.axis('off')
boxes = [(.02,.60,'1. Séries\ne origens','145 séries\nTreino: 17/23 ciclos'),
         (.36,.60,'2. Ajuste\nno treino','Perfis e parâmetros\n7 políticas\n3 sementes*'),
         (.70,.60,'3. Seleção\nna validação','7 ciclos; outras lojas\nGlobal ou por perfil\nReferência: 0,70'),
         (.70,.12,'4. Teste\nreservado','7 ciclos de teste\nEstado inicial comum'),
         (.36,.12,'5. Comparação\npareada','Custo e atendimento\nDistribuição por série'),
         (.02,.12,'6. Decisão\nde pesquisa','Serviço adicional\njustifica o custo?')]
for x,y,title,body in boxes:
    ax.add_patch(FancyBboxPatch((x,y),.28,.27,boxstyle='round,pad=0.012',
                                facecolor='#F1F6F8',edgecolor=blue,linewidth=1.2))
    ax.text(x+.14,y+.205,title,ha='center',va='center',fontsize=10,weight='bold',color=blue)
    ax.text(x+.14,y+.08,body,ha='center',va='center',fontsize=9.5,linespacing=1.2)
for start,end in [((.31,.735),(.345,.735)),((.65,.735),(.685,.735)),
                  ((.84,.58),(.84,.41)),((.685,.255),(.65,.255)),((.345,.255),(.31,.255))]:
    ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':blue,'lw':1.8})
ax.text(.02,.005,'* Estocásticos: sementes 11, 29 e 47; determinísticos: um ajuste.\n'
        'Teste reservado; origens com um ciclo de teste compartilhado.',fontsize=9)
save(fig,'fluxo_experimental')

summary = test.groupby('policy')[['TIC','NS']].mean().sort_values('TIC')
efficient = []
for p,r in summary.iterrows():
    dominated = ((summary.TIC <= r.TIC) & (summary.NS >= r.NS) &
                 ((summary.TIC < r.TIC) | (summary.NS > r.NS))).any()
    if not dominated: efficient.append(p)
fig,ax = plt.subplots(figsize=(7,3.92))
front = summary.loc[efficient].sort_values('TIC')
ax.plot(front.TIC,front.NS,color=blue,lw=1.4,label='Fronteira descritiva')
offsets={'PPO':(8,-15),'GA':(-28,10),'GA-DQN':(10,-12),'DQN':(8,0),
         'EOQCycle':(10,-10),'sSCycle':(8,10),'CappedBaseStock':(-110,8)}
for p,r in summary.iterrows():
    ax.scatter(r.TIC,r.NS,s=65,color=blue if p in efficient else gray,zorder=3)
    ax.annotate(p,(r.TIC,r.NS),xytext=offsets[p],textcoords='offset points',fontsize=10.5)
ax.axhline(.7,color=orange,ls='--',lw=1,label='Referência de atendimento: 0,70')
ax.set(xlabel='Custo médio no teste (unidades do cenário)',ylabel='Atendimento médio por caso',ylim=(.49,.87))
ax.grid(alpha=.15);ax.legend(loc='lower right',fontsize=9.5)
save(fig,'custo_servico')

colors={'GA':orange,'sSCycle':blue,'EOQCycle':'#58A596'}
names={'Sparse_High_Impact':'Demanda esparsa','Unstable_Trend':'Tendência instável',
       'High_Volatility_Seasonal':'Volatilidade/sazonalidade',
       'High_Vol_Seasonal':'Volatilidade/\nsazonalidade'}
fig,axs=plt.subplots(1,2,figsize=(7.2,3.33),sharex=True)
for ax,origin in zip(axs,[17,23]):
    g=sel[(sel.origin.eq(origin)) & sel.strategy.eq('perfil')]
    counts=pd.crosstab(g.profile,g.policy)
    counts=counts.reindex(columns=['GA','sSCycle','EOQCycle'],fill_value=0)
    left=np.zeros(len(counts))
    for policy in counts:
        values=counts[policy].to_numpy()
        ax.barh(np.arange(len(counts)),values,left=left,color=colors[policy],label=policy,height=.58)
        for y,v,l in zip(range(len(counts)),values,left):
            if v: ax.text(l+v/2,y,str(v),ha='center',va='center',color='white',fontsize=11)
        left+=values
    ax.set_yticks(range(len(counts)),[names.get(p,p) for p in counts.index],fontsize=10.5)
    ax.set(title=f'Treino: {origin} ciclos',xlabel='Número de séries',xlim=(0,90))
    ax.invert_yaxis()
handles, labels=axs[1].get_legend_handles_labels()
fig.legend(handles, labels,loc='lower center',ncol=3,fontsize=10,frameon=False)
fig.tight_layout(w_pad=2,rect=(0,.12,1,1))
save(fig,'escolhas_perfil')

fig,axs=plt.subplots(1,2,figsize=(7,3.27),sharey=True)
for ax,origin in zip(axs,[17,23]):
    g=sel[sel.origin.eq(origin)]
    vals=[g.loc[g.strategy.eq(s),'NS'].dropna().to_numpy() for s in ['global','perfil']]
    b=ax.boxplot(vals,tick_labels=['Global','Por perfil'],patch_artist=True,widths=.5,
                 medianprops={'color':'black'},flierprops={'markersize':2,'alpha':.35})
    for patch,c in zip(b['boxes'],[blue,orange]): patch.set_facecolor(c);patch.set_alpha(.65)
    ax.axhline(.7,color=orange,ls='--',lw=1)
    ax.set(title=f'Treino: {origin} ciclos',ylim=(-.03,1.06))
    ax.text(.03,.05,f'n = {len(vals[0])} por estratégia',transform=ax.transAxes,fontsize=9.5)
    ax.grid(axis='y',alpha=.15)
axs[0].set_ylabel('Atendimento por série no teste')
fig.tight_layout()
save(fig,'distribuicao_atendimento')

(OUT/'proveniencia.json').write_text(json.dumps({
    'run':RUN.name,'completed_fits':status['completed_fits'],'case_count':290,
    'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [RUN/'kpis.csv',selection_path]},
    'selection':'Other-store validation; held-store test; alpha=0.70',
    'frontier':'Descriptive test means; not prospective selection or uncertainty inference'
},indent=2),encoding='utf-8')
print('Four manuscript figures generated from complete Bahia results (PNG and vector PDF).')
