"""Publish aggregate BA/M5 evidence from completed frozen runs, without fitting.

Creates new inputs/figures only; never edits canonical scientific run files or prose.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'experiment/data/08_reporting/research_v2'
ARTICLE=ROOT/'docs/supply_chain_analytics_article'
THESIS=ROOT/'docs/master_proposal'
OUT=ROOT/'docs/operations/m5_final_2026-10-17'
CASE=['origin','warehouse','store_id','item_id']
RUNS={'BA':'aipe_ba145_unified_2026-10-15','M5':'aipe_m5_2408_unified_2026-10-15'}
ORDER=['Global','Perfil','AIPE_PSE','Classe_floresta_controle','Consequencia','Consequencia_sem_perfil','Consequencia_recuo','PSE_sem_perfil','Modo_rotulos']
NAMES={'Global':'Referência global','Perfil':'PSE por perfil','AIPE_PSE':'PSE por classe (XGBoost)','Classe_floresta_controle':'Classe: floresta de controle','Consequencia':'Consequências','Consequencia_sem_perfil':'Consequências sem perfil','Consequencia_recuo':'Consequências com recuo','PSE_sem_perfil':'Classe sem perfil','Modo_rotulos':'Moda dos rótulos'}
DISPLAY=ORDER[:7]
SOURCES={}
def read(path):
 SOURCES[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
 return pd.read_csv(path,dtype={c:str for c in CASE[1:]},low_memory=False)
def js(path):
 SOURCES[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
 return json.loads(path.read_text(encoding='utf8'))
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def num(x,d=2):
 if x is None or not np.isfinite(x):return '--'
 return f'{x:,.{d}f}'.translate(str.maketrans(',.','.,'))
def interval(pair,column,draws=5000):
 groups=pair.groupby(['warehouse','store_id'])[column].agg(['sum','count'])
 rng=np.random.default_rng(20261017)
 indices=rng.integers(0,len(groups),(draws,len(groups)))
 n=groups['sum'].to_numpy()[indices].sum(axis=1);d=groups['count'].to_numpy()[indices].sum(axis=1)
 samples=n[d>0]/d[d>0]
 return {'mean':float(pair[column].mean()),'low':float(np.quantile(samples,.025)) if len(samples) else None,'high':float(np.quantile(samples,.975)) if len(samples) else None,'clusters':len(groups),'draws':draws,'conditional':True}
def summarize(scored,plans):
 baseline=scored[scored.strategy.eq('Global')][CASE+['TIC','NS','policy']]
 result={}
 for strategy,g in scored.groupby('strategy'):
  p=g.merge(baseline,on=CASE,suffixes=('','_reference'),validate='one_to_one')
  p['dc']=p.TIC-p.TIC_reference;p['ds']=p.NS-p.NS_reference
  savings=-p.dc
  enriched=g.merge(plans,on=CASE,validate='many_to_one')
  failures=np.select([enriched.NS.isna(),~enriched.physical_target_feasible,enriched.oracle_service.lt(.7-1e-9),~enriched.feasible],['demanda_zero','limite_fisico','lacuna_portfolio','erro_selecao'],default='atendeu')
  comp=g[g.comparable]
  result[strategy]={'cases':len(g),'defined_service':int(g.NS.notna().sum()),'mean_cost':g.TIC.mean(),'mean_service':g.NS.mean(),'feasible':int(g.feasible.sum()),'cost_change_percent':100*(g.TIC.mean()/p.TIC_reference.mean()-1),'service_change_pp':100*p.ds.mean(),'interventions_vs_global':int(p.policy.ne(p.policy_reference).sum()),'higher_cost_cases':int(p.dc.gt(1e-6).sum()),'lower_service_cases':int(p.ds.lt(-1e-9).sum()),'new_service_failures':int((p.NS_reference.ge(.7-1e-9)&p.NS.lt(.7-1e-9)).sum()),'recovered_service_cases':int((p.NS_reference.lt(.7-1e-9)&p.NS.ge(.7-1e-9)).sum()),'positive_savings_cases':int(savings.gt(1e-6).sum()),'top_positive_saving_share':float(savings.clip(lower=0).max()/savings.clip(lower=0).sum()) if savings.clip(lower=0).sum()>0 else None,'top_saving_net_share':float(savings.max()/savings.sum()) if savings.sum()>1e-9 else None,'cost_difference_ci95':interval(p,'dc'),'service_difference_ci95':interval(p,'ds'),'portfolio_comparable':len(comp),'portfolio_hits':int(g.oracle_hit.sum()),'portfolio_efficiency_mean_percent':comp.efficiency_percent.mean(),'portfolio_efficiency_median_percent':comp.efficiency_percent.median(),'portfolio_within_1_percent':int(g.within_1_percent.sum()),'portfolio_within_5_percent':int(g.within_5_percent.sum()),'failure_decomposition':dict(pd.Series(failures).value_counts())}
 return clean(result)
def dataset(ds,name):
 run=BASE/name;status=js(run/'status.json');plan=js(run/'plan.json')
 if status['state']!='completed' or status['completed_fits']!=plan['planned_fits']:raise ValueError(f'{ds}: evaluation not completed; current state={status["state"]}')
 k=read(run/'kpis.csv');s=read(run/'selection/case_metrics.csv');p=read(run/'selection/perfect_information.csv');real=read(run/'selection/realized_theory_comparisons.csv')
 assert len(p)==plan['series_origin_cases'] and not p.duplicated(CASE).any()
 # Report solver certificates honestly; unproven plans remain outside efficiency comparisons.
 expected={(policy,seed,block) for policy in plan['policies'] for seed in (plan['seeds'] if policy in {'GA','PPO','GA-DQN','DQN'} else [0]) for block in ['validation','test']}
 assert len(k)==2*plan['planned_fits'] and not k.duplicated(CASE+['policy','seed','block']).any()
 for _,g in k.groupby(CASE):assert set(g[['policy','seed','block']].itertuples(index=False,name=None))==expected
 assert set(s.strategy)==set(ORDER)
 assert not s.duplicated(CASE+['strategy']).any()
 assert s.groupby('strategy').size().eq(len(p)).all()
 assert len(k[CASE[1:]].drop_duplicates())==max(c['series'] for c in plan['cohorts'])
 means=k[k.block.eq('test')].groupby(CASE+['policy'],as_index=False)[['TIC','NS']].mean()
 assert means.groupby(CASE).NS.count().isin([0,len(plan['policies'])]).all()
 defined=int(means.groupby(CASE).NS.count().gt(0).sum())
 assert s.groupby('strategy').NS.count().eq(defined).all()
 looked=s.merge(means,on=CASE+['policy'],suffixes=('_reported','_matrix'),validate='many_to_one')
 assert len(looked)==len(s)
 np.testing.assert_allclose(looked.TIC_reported,looked.TIC_matrix,rtol=1e-12,atol=1e-10)
 np.testing.assert_allclose(looked.NS_reported,looked.NS_matrix,rtol=1e-12,atol=1e-12,equal_nan=True)
 assert np.array_equal(s.feasible.to_numpy(),s.NS.ge(.7-1e-9).to_numpy())
 # Reconstruct oracle policy using tuple ordering, independently of reporting
 # code: feasible minimum cost, otherwise maximum defined service then cost.
 winners=[]
 for key,g in means.groupby(CASE):
  records=list(g[['policy','TIC','NS']].itertuples(index=False,name=None))
  feasible=[v for v in records if np.isfinite(v[2]) and v[2]>=.7-1e-9]
  winner=min(feasible,key=lambda v:(v[1],v[0])) if feasible else min(records,key=lambda v:(-v[2] if np.isfinite(v[2]) else float('inf'),v[1],v[0]))
  winners.append(dict(zip(CASE,key))|{'oracle_policy_checked':winner[0],'oracle_cost_checked':winner[1],'oracle_service_checked':winner[2]})
 reconstructed=s.merge(pd.DataFrame(winners),on=CASE,validate='many_to_one')
 assert reconstructed.oracle_policy.eq(reconstructed.oracle_policy_checked).all()
 np.testing.assert_allclose(reconstructed.oracle_cost,reconstructed.oracle_cost_checked,rtol=1e-12,atol=1e-10)
 np.testing.assert_allclose(reconstructed.oracle_service,reconstructed.oracle_service_checked,rtol=1e-12,atol=1e-12,equal_nan=True)
 original=read(run/'selection/summary.csv').set_index('strategy')
 for st,g in s.groupby('strategy'):
  np.testing.assert_allclose([g.TIC.mean(),g.NS.mean()],[original.loc[st,'mean_cost'],original.loc[st,'mean_service']],rtol=1e-12)
 summary=summarize(s,p)
 theory={}
 # The portfolio oracle is over seed-averaged costs. The information-perfect
 # comparison below is at each realized selected policy/seed, not a case-mean ratio.
 for st,g in real.groupby('strategy'):
  valid=g[g.theory_comparable]
  assert valid.theory_efficiency_percent.le(100+1e-5).all()
  percase=g.groupby(CASE,as_index=False).agg(realizations=('seed','size'),comparable=('theory_comparable','sum'),hits=('theory_hit','sum'),physical_attainment_percent=('physical_attainment_percent','mean'))
  theory[st]={'selected_seed_realizations':len(g),'comparable_realizations':len(valid),'hits_realizations':int(g.theory_hit.sum()),'efficiency_mean_percent_comparable_realizations':valid.theory_efficiency_percent.mean(),'efficiency_median_percent_comparable_realizations':valid.theory_efficiency_percent.median(),'cases_all_selected_seeds_comparable':int(percase.comparable.eq(percase.realizations).sum()),'cases_all_selected_seeds_hit':int(percase.hits.eq(percase.realizations).sum()),'physical_attainment_mean_percent_cases':percase.physical_attainment_percent.mean()}
 byorigin={int(o):summarize(g,p[p.origin.eq(o)]) for o,g in s.groupby('origin')}
 policy=means.groupby('policy').agg(cases=('TIC','size'),mean_cost=('TIC','mean'),mean_service=('NS','mean'),defined_service=('NS','count')).reset_index()
 policy.to_csv(OUT/f'{ds}_policies.csv',index=False)
 s.groupby(['origin','strategy']).agg(cases=('TIC','size'),mean_cost=('TIC','mean'),mean_service=('NS','mean'),defined_service=('NS','count'),feasible=('feasible','sum')).to_csv(OUT/f'{ds}_by_origin.csv')
 timings=k.drop_duplicates(CASE+['policy','seed'])
 if 'reused_from' in timings:timings=timings[timings.reused_from.isna()|timings.reused_from.eq('')]
 # canonical reused checkpoints are additionally identified inside their payloads.
 import sqlite3
 with sqlite3.connect((run/'results.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
  q="SELECT policy,seconds FROM fits WHERE json_extract(payload,'$.canonical_reuse') IS NULL"
  cols=[x[1] for x in db.execute('PRAGMA table_info(fits)')]
  rawcol=cols[-1];timecol=cols[-2]
  qt=f'SELECT policy,{timecol} seconds FROM fits WHERE json_extract({rawcol},\'$.canonical_reuse\') IS NULL AND json_extract({rawcol},\'$.reused_from\') IS NULL'
  timings=pd.read_sql_query(qt,db)
 times=timings.groupby('policy').seconds.agg(['count','median','mean']).reset_index()
 times.to_csv(OUT/f'{ds}_fit_times_new.csv',index=False)
 folds=js(run/'selection/inner_validation.json')
 margins=dict(pd.Series([str(x['margin']) for x in folds]).value_counts())
 entry={'run':name,'status':'completed_and_audited','cohorts':plan['cohorts'],'cases':len(p),'series':max(c['series'] for c in plan['cohorts']),'stores':s.store_id.nunique(),'defined_service_cases':defined,'zero_demand_test_cases':len(p)-defined,'planned_fits':plan['planned_fits'],'kpi_rows':len(k),'physical_target_feasible_cases':int(p.physical_target_feasible.sum()),'physical_target_feasible_positive_demand_cases':int((p.physical_target_feasible & p.max_service.notna()).sum()),'physical_target_feasible_convention':'Raw flag includes zero-demand cases as unconstrained; a defined service target applies only to positive-demand cases.','proven_optimal_cases':int(p.proven_optimal.sum()),'summary':summary,'by_origin':byorigin,'perfect_information_by_realization':theory,'margin_frequency_outer_folds':margins,'feature_stability':'Not persisted by canonical selectors; no post-hoc importance inference performed','cost_unit':'scenario cost units; not BRL','bootstrap_unit':'warehouse/store cluster, preserving products and origins; fixed fitted selectors/controllers','m5_scope':'Historical fixed 2,408 FOODS identifiers from 14,370 parent candidates; eligibility determined at each training prefix' if ds=='M5' else '145 fixed historical series; 115 stores; nine products','policy_summary':policy.to_dict('records'),'fit_times_nonreused':times.to_dict('records')}
 profiles=k[CASE+['profile']].drop_duplicates()
 assert not profiles.duplicated(CASE).any()
 entry['prefix_profiles_by_origin']={int(o):g.profile.value_counts().to_dict() for o,g in profiles.groupby('origin')}
 xgb=s[s.strategy.eq('AIPE_PSE')]
 entry['validation_label_coverage_by_origin']={int(o):{'validation_label_counts':g.validation_label.value_counts().to_dict(),'held_validation_labels_unseen':int(g.label_unseen_in_training.sum())} for o,g in xgb.groupby('origin')}
 membership=run/'selection/pilot_vs_remaining_cohort.csv'
 if membership.exists():entry['development_membership_summary']=read(membership).to_dict('records')
 if ds=='M5':
  pilot=read(BASE/'m5_pse_pilot30_v2_2026-10-15/features.csv')[CASE[1:]].drop_duplicates()
  prior=pd.MultiIndex.from_frame(pilot)
  annotated=s.copy();annotated['development_pilot_series']=pd.MultiIndex.from_frame(annotated[CASE[1:]]).isin(prior)
  entry['development_vs_remaining']={str(flag):summarize(g,p) for flag,g in annotated.groupby('development_pilot_series')}
  entry['development_membership_limit']='Separates previously inspected 30 series; remaining series share stores/data and do not establish independent confirmation.'
 return clean(entry),s

def save(fig,name):
 for ext in ['pdf','png']:
  path=ARTICLE/'figures'/f'{name}.{ext}'
  kwargs={'backend':'pgf'} if ext=='pdf' else {}
  fig.savefig(path,dpi=200,bbox_inches='tight',facecolor='white',**kwargs);shutil.copy2(path,THESIS/'figures'/path.name)
 plt.close(fig)
def figures(evidence,frames):
 plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pgf.texsystem':'pdflatex','pgf.rcfonts':False,'pgf.preamble':r'\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}\usepackage{mathptmx}'})
 fig,axes=plt.subplots(1,2,figsize=(8.2,3.7),layout='constrained')
 colors=plt.get_cmap('tab10')
 for ax,(ds,d) in zip(axes,evidence.items()):
  for i,st in enumerate(DISPLAY):
   r=d['summary'][st];ax.scatter(r['mean_service']*100,r['mean_cost'],s=50,color=colors(i),marker='o' if st!='Consequencia_recuo' else 'D',label=NAMES[st])
  ax.axvline(70,color='#888888',ls='--',lw=.8);ax.set(title='Bahia · base empresarial' if ds=='BA' else 'M5 · coorte ampliada',xlabel='Atendimento médio definido (%)',ylabel='Custo médio de cenário')
 handles,labels=axes[0].get_legend_handles_labels()
 fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,-.03),ncol=2,fontsize=8.5,frameon=False)
 save(fig,'aipe_unificado_custo_servico_20261017')
 fig,axes=plt.subplots(2,2,figsize=(8.2,7.0),layout='constrained')
 styles=[st for st in DISPLAY if st!='Global']
 for col,(ds,d) in enumerate(evidence.items()):
  for row,(key,mult,label) in enumerate([('cost_difference_ci95',1,'Diferença de custo frente à global'),('service_difference_ci95',100,'Diferença de atendimento (p.p.)')]):
   ax=axes[row,col]
   for y,st in enumerate(styles):
    r=d['summary'][st][key];mu=r['mean']*mult;lo=r['low']*mult;hi=r['high']*mult
    ax.errorbar(mu,y,xerr=[[max(0,mu-lo)],[max(0,hi-mu)]],fmt='o',color=colors(DISPLAY.index(st)),capsize=3)
   ax.set_yticks(range(len(styles)),[NAMES[st] for st in styles] if col==0 else ['']*len(styles));ax.invert_yaxis();ax.axvline(0,color='#888888',lw=.8);ax.set(xlabel=label,title=ds if row==0 else None)
 save(fig,'aipe_unificado_diferencas_20261017')
 fig,axes=plt.subplots(1,2,figsize=(8.2,3.7),layout='constrained')
 causes=['atendeu','erro_selecao','lacuna_portfolio','limite_fisico','demanda_zero'];labels=['Atende a 0,70','Erro de seleção','Lacuna do portfólio','Limite físico','Demanda zero'];palette=['#269781','#e6a23c','#5276a1','#9d718e','#d9dfe4']
 for col,(ax,(ds,d)) in enumerate(zip(axes,evidence.items())):
  left=np.zeros(len(DISPLAY))
  for cause,label,color in zip(causes,labels,palette):
   values=np.array([d['summary'][st]['failure_decomposition'].get(cause,0) for st in DISPLAY]);ax.barh(range(len(DISPLAY)),100*values/d['cases'],left=left,color=color,label=label);left+=100*values/d['cases']
  ax.set_yticks(range(len(DISPLAY)),[NAMES[x] for x in DISPLAY] if col==0 else ['']*len(DISPLAY));ax.invert_yaxis();ax.set(title=ds,xlabel='Casos série--origem (%)',xlim=(0,100))
 handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,-.03),ncol=3,frameon=False,fontsize=8.5)
 save(fig,'aipe_unificado_falhas_20261017')
 fig,axes=plt.subplots(1,2,figsize=(8.2,3.7),layout='constrained')
 selected=['Global','Perfil','AIPE_PSE','Consequencia','Consequencia_recuo']
 for ax,(ds,frame) in zip(axes,frames.items()):
  ax.boxplot([100*frame[frame.strategy.eq(st)].NS.dropna() for st in selected],tick_labels=['Global','Perfil','Classe','Conseq.','Recuo'],showfliers=False)
  ax.axhline(70,color='#a56738',ls='--',lw=1);ax.set(title=f'{ds} · atendimento por caso',ylabel='Atendimento (%)',ylim=(-2,103))
 save(fig,'aipe_unificado_distribuicao_20261017')
def tex(evidence):
 lines=['% Generated from completed canonical matrices; no controller training.','\\section{Resultados da avaliação unificada}','\\label{sec:resultados_unificados}']
 details=[r'\section{Diagnósticos complementares da avaliação unificada}',r'\label{sec:diagnosticos_unificados}']
 for ds,d in evidence.items():
  title='Bahia' if ds=='BA' else 'M5'
  lines += [r'\begin{table}[htbp]\centering\small',f'\\caption{{{title}: resultados por caso série--origem. Custo: {d["cases"]} casos; atendimento: {d["defined_service_cases"]} casos definidos. Custo em unidades de cenário.}}',f'\\label{{tab:unificado_{ds.lower()}}}',r'\begin{tabular}{@{}p{4.4cm}rrrr@{}}\toprule',r'Implementação & Custo médio & Atendimento & Viáveis & Trocas\\\midrule']
  for st in DISPLAY:
   r=d['summary'][st];lines.append(f'{NAMES[st]} & {num(r["mean_cost"])} & {num(100*r["mean_service"])}\\% & {r["feasible"]} & {r["interventions_vs_global"]}'+r'\\')
  lines += [r'\bottomrule\end{tabular}',r'\par\footnotesize Trocas indicam política diferente da referência global; viáveis têm atendimento $\geq0{,}70$.',r'\end{table}']
  mainlines=lines;lines=[]
  lines += [r'\begin{table}[htbp]\centering\small',f'\\caption{{{title}: diferenças pareadas frente à referência global. Intervalos condicionais de 95\\%, com 5.000 reamostragens por loja.}}',f'\\label{{tab:unificado_delta_{ds.lower()}}}',r'\begin{tabular}{@{}p{4.4cm}rr@{}}\toprule',r'Implementação & Custo: diferença [IC] & Serviço: p.p. [IC]\\\midrule']
  for st in DISPLAY[1:]:
   r=d['summary'][st];c=r['cost_difference_ci95'];s=r['service_difference_ci95'];lines.append(f'{NAMES[st]} & {num(c["mean"])} [{num(c["low"])}; {num(c["high"])}] & {num(100*s["mean"])} [{num(100*s["low"])}; {num(100*s["high"])}]'+r'\\')
  lines += [r'\bottomrule\end{tabular}\end{table}']
  lines += [r'\begin{table}[htbp]\centering\small',f'\\caption{{{title}: aproximação do oráculo do portfólio, calculado após agregar sementes. Eficiência somente entre casos comparáveis.}}',f'\\label{{tab:unificado_oraculo_{ds.lower()}}}',r'\begin{tabular}{@{}p{4.4cm}rrr@{}}\toprule',r'Implementação & Acertos / comparáveis & Eficiência média & Até 5\%\\\midrule']
  for st in DISPLAY:
   r=d['summary'][st];lines.append(f'{NAMES[st]} & {r["portfolio_hits"]}/{r["portfolio_comparable"]} & {num(r["portfolio_efficiency_mean_percent"])}\\% & {r["portfolio_within_5_percent"]}'+r'\\')
  lines += [r'\bottomrule\end{tabular}',r'\par\footnotesize Comparabilidade exige atendimento de pelo menos 0,70 tanto da implementação quanto do oráculo. Denominadores diferentes não sustentam comparação direta das eficiências condicionais.',r'\end{table}']
  lines += [r'\begin{table}[htbp]\centering\small',f'\\caption{{{title}: planejamento com informação perfeita. Comparações por realização da política escolhida; sementes não são observações independentes.}}',f'\\label{{tab:unificado_teorico_{ds.lower()}}}',r'\begin{tabular}{@{}p{4.4cm}rrr@{}}\toprule',r'Implementação & Acertos / realizações viáveis & Eficiência média & Casos: todos acertos\\\midrule']
  for st in DISPLAY:
   r=d['perfect_information_by_realization'][st]
   lines.append(f'{NAMES[st]} & {r["hits_realizations"]}/{r["comparable_realizations"]} & {num(r["efficiency_mean_percent_comparable_realizations"])}\\% & {r["cases_all_selected_seeds_hit"]}'+r'\\')
  lines += [r'\bottomrule\end{tabular}',f'\\par\\footnotesize {d["proven_optimal_cases"]}/{d["cases"]} planos com ótimo certificado; {d["physical_target_feasible_positive_demand_cases"]} casos com demanda positiva fisicamente permitem atingir a meta; {d["zero_demand_test_cases"]} casos sem demanda mantêm atendimento indefinido. Eficiência restringe-se às realizações viáveis com ótimo certificado.',r'\end{table}']
  details.extend(lines);lines=mainlines
 for name,cap,label in [('custo_servico','Custo e atendimento nas duas aplicações. As médias não mostram a distribuição individual nem constituem fronteira de dominância por série.','fig:unificado_custo_servico'),('diferencas','Diferenças pareadas frente à referência global. Intervalos condicionais por loja, preservando produtos e origens; dez lojas limitam a precisão no M5.','fig:unificado_diferencas'),('falhas','Diagnóstico retrospectivo dos casos. A classificação distingue falha de seleção, cobertura do portfólio e restrição física inicial; demanda zero mantém atendimento indefinido.','fig:unificado_falhas'),('distribuicao','Distribuição do atendimento nos casos com demanda positiva; linha horizontal indica 70\\%. Extremos são omitidos da exibição dos boxplots, mas permanecem nos cálculos.','fig:unificado_distribuicao')]:
  lines += [r'\begin{figure}[htbp]\centering',f'\\includegraphics[width=\\linewidth]{{figures/aipe_unificado_{name}_20261017.pdf}}',f'\\caption{{{cap}}}\\label{{{label}}}',r'\end{figure}']
 text='\n'.join(lines)+'\n'
 for dest in [ARTICLE,THESIS]:(dest/'resultados_unificados_20261017.tex').write_text(text,encoding='utf8')
 (THESIS/'diagnosticos_unificados_20261017.tex').write_text('\n'.join(details)+'\n',encoding='utf8')
def main():
 OUT.mkdir(exist_ok=True,parents=True)
 evidence={};frames={}
 for ds,name in RUNS.items():evidence[ds],frames[ds]=dataset(ds,name)
 figures(evidence,frames);tex(evidence)
 report={'status':'completed_matrix_audit_passed','controller_fits_started':0,'selectors_fitted':0,'sources_sha256':SOURCES,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'datasets':evidence,'limits':['Development/retrospective evaluation; not independent confirmation','Seed averages are within case, origins repeat series, bootstrap clusters are stores','Scenario costs and synthetic states; not calibrated savings in BRL','Per-realization theory metrics and seed-averaged portfolio metrics have different denominators','No feature-stability evidence is created when not persisted']}
 (OUT/'evidence.json').write_text(json.dumps(clean(report),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
 for ds,d in evidence.items():print(ds,json.dumps(d['summary'],ensure_ascii=False))
 print('Artifacts generated; no manuscript prose edited.')
if __name__=='__main__':main()
