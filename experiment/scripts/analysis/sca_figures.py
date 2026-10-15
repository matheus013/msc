"""Reproducible data plots for the SCA manuscript, without exposing row IDs."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
data = ROOT / 'experiment/data/08_reporting/sca_article'
out = ROOT / 'docs/supply_chain_analytics_article'
historical = pd.read_csv(data/'policy_summary.csv')
temporal = json.loads((data/'temporal/results.json').read_text())
plt.rcParams.update({'font.size':10, 'font.family':'DejaVu Sans'})
fig, axes = plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
ax=axes[0]
for _,row in historical.iterrows():
    ax.scatter(row.TIC,row.NS,s=27,color='#24557a')
    offset={'EOQ':(-30,10),'sS':(8,-10),'GA':(7,10),'SA':(7,-11),
            'BigDataNewsvendor':(-40,8),'GA-DQN':(7,7)}.get(row.policy,(3,5))
    ax.annotate(row.policy,(row.TIC,row.NS),xytext=offset,textcoords='offset points',fontsize=8)
ax.axhline(.7,color='#9a4c22',linestyle='--',linewidth=1)
ax.set(xscale='log',xlabel='Mean simulated cost (log scale)',ylabel='Mean series fill rate',
       title='(a) Preserved 12-policy matrix',ylim=(.15,1.04))
ax.xaxis.set_major_locator(FixedLocator([200,500,1000,3000,6000]))
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_minor_formatter(NullFormatter())
ax=axes[1]
for strategy,marker,color,label in [('global','o','#24557a','Global selector'),('profile','s','#aa5429','Profile selector'),('EOQ','^','#555555','Legacy EOQ'),('EOQPerCycle','D','#347d52','Cycle-unit EOQ')]:
    rows=[r for r in temporal['summary'] if r['strategy']==strategy and r['alpha']==.7]
    r=rows[0]
    ax.scatter(r['TIC_mean'],r['NS_mean'],marker=marker,color=color,s=60,label=label)
    ax.annotate(label,(r['TIC_mean'],r['NS_mean']),xytext=(8,-16 if strategy=='profile' else 8),textcoords='offset points',fontsize=9)
ax.axhline(.7,color='#9a4c22',linestyle='--',linewidth=1)
ax.set(xlabel='Mean simulated cost (11 test cycles)',ylabel='Mean series fill rate',
       title='(b) Temporal diagnostic, threshold 0.70',ylim=(.65,1),xlim=(140,460))
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=.15)
fig.savefig(out/'Figure_1.png',dpi=300)
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
contrasts=[r for r in temporal['contrasts'] if r['baseline']=='global']
x=[r['alpha'] for r in contrasts]
for ax,field,interval,title in [(axes[0],'cost_reduction_percent','cost_reduction_ci','Cost reduction vs global (%)'),(axes[1],'service_difference','service_difference_ci','Fill-rate difference vs global (pp)')]:
    factor=100 if field=='service_difference' else 1
    y=[r[field]*factor for r in contrasts]
    low=[(r[field]-r[interval][0])*factor for r in contrasts]
    high=[(r[interval][1]-r[field])*factor for r in contrasts]
    ax.errorbar(x,y,yerr=[low,high],fmt='o-',capsize=4,color='#24557a',linewidth=1.2)
    ax.axhline(0,color='#555555',linewidth=1)
    ax.set(xlabel='Validation service threshold',ylabel=title,xticks=x)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=.15)
fig.savefig(out/'Figure_2.png',dpi=300)
plt.close(fig)
print('Figures generated from frozen aggregate outputs.')
