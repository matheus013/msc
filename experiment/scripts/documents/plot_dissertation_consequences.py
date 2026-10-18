from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[3]
parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',type=Path,default=root/'docs/master_proposal/figures')
args=parser.parse_args()
p=root/'docs/supply_chain_analytics_article/reviews/pse_consequences_2026-10-15/summary.csv'
d=pd.read_csv(p).set_index('strategy')
keys=['Perfil','AIPE_PSE','Classe_floresta_controle','Consequencia','Consequencia_recuo']
labels=['Por perfil','Preditivo por classe','Classe: controle','Consequências','Consequências + recuo']
g=d.loc[keys]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(11.4,4.2),layout='constrained',sharey=True)
for ax,col,title,xlabel in zip(axes,['cost_change_percent','service_change_pp'],['Custo','Atendimento'],['Diferença de custo (%)','Diferença de fill rate (p.p.)']):
 values=g[col].to_numpy();ax.barh(labels,values,color=['#6287a2']*4+['#26836b']);ax.axvline(0,color='#333333',lw=1);ax.set(title=title,xlabel=xlabel)
 for i,v in enumerate(values):ax.annotate(f'{v:+.2f}'.replace('.',','),(v,i),xytext=(4 if v>=0 else -4,0),textcoords='offset points',ha='left' if v>=0 else 'right',va='center',fontsize=9)
 ax.margins(x=.25)
axes[0].invert_yaxis()
out=args.output_dir
for extension in ['pdf','png','svg']:fig.savefig(out/f'aipe_consequencias_comparacao.{extension}',dpi=170)
