"""Vector article figures from completed pilot diagnostics; no model fitting."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'experiment/data/08_reporting/research_v2/m5_pse_pilot30_v2_2026-10-15/diagnostics_v1'
OUT=ROOT/'docs/supply_chain_analytics_article/figures'
plt.rcParams.update({'font.size':10,'font.family':'serif','axes.spines.top':False,'axes.spines.right':False,
    'pgf.texsystem':'pdflatex','pgf.rcfonts':False,
    'pgf.preamble':r'\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}\usepackage{mathptmx}'})
data=pd.read_csv(SOURCE/'scored_strategies.csv')
strategies=['Global','Perfil','AIPE_PSE','PSE_com_recuo']
names=['Global','PSE por perfil','PSE preditivo','Com recuo']
colors=['#31688e','#35a182','#d77b32','#8362a6']


def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=240,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/(name+'.pdf'),backend='pgf',bbox_inches='tight',facecolor='white')
    plt.close(fig)
    shutil.copy2(OUT/(name+'.pdf'),ROOT/'docs/master_proposal/figures'/(name+'.pdf'))


fig,axes=plt.subplots(1,2,figsize=(7.2,3.35),gridspec_kw={'width_ratios':[1.15,1]})
costs=[data[data.strategy.eq(s)].TIC.mean() for s in strategies]
axes[0].barh(names,costs,color=colors)
axes[0].invert_yaxis();axes[0].set(xlabel='Custo médio por caso',title='Custo no teste',xlim=(0,1480))
for i,v in enumerate(costs):axes[0].text(v+22,i,f'{v:.0f}',va='center',fontsize=9)
axes[1].boxplot([100*data[data.strategy.eq(s)].NS.dropna() for s in strategies],tick_labels=['Global','Perfil','Pred.','Recuo'])
axes[1].axhline(70,color='#b65a31',ls='--',lw=1)
axes[1].set(ylabel='Atendimento por caso (%)',title='Distribuição do serviço',ylim=(0,105))
fig.tight_layout(w_pad=2)
save(fig,'aipe_m5_comparacao')

failure=pd.read_csv(SOURCE/'service_failure_decomposition.csv',index_col='strategy').loc[strategies]
columns=['atendeu','erro_selecao','lacuna_portfolio','limite_fisico','demanda_zero']
labels=['Atende a 0,70','Erro de seleção','Lacuna do portfólio','Limite físico inicial','Demanda zero']
fig,ax=plt.subplots(figsize=(7.2,3.05));left=np.zeros(4)
for column,label,color in zip(columns,labels,['#35a182','#dc9a32','#5276a1','#9d718e','#d9dfe4']):
    values=failure[column].to_numpy()
    ax.barh(names,values,left=left,color=color,label=label,height=.58)
    for y,(value,start) in enumerate(zip(values,left)):
        ax.text(start+value/2,y,str(value),ha='center',va='center',fontsize=9,color='white' if column in ['atendeu','lacuna_portfolio','limite_fisico'] else '#25384a')
    left+=values
assert np.all(left==90)
ax.invert_yaxis();ax.set(xlabel='Casos série--origem',xlim=(0,90))
ax.legend(loc='upper center',bbox_to_anchor=(.42,-.22),ncol=3,frameon=False,fontsize=8.5)
fig.tight_layout(rect=(0,.1,1,1));save(fig,'aipe_m5_falhas')
print('Two M5 vector figures generated and shared with dissertation.')
