"""Article diagnostics from completed research fits; no training or test selection.

Seeds are averaged inside a series/origin. Only cases with every declared
policy/seed in validation and test enter comparisons. Origins remain repeated
measurements, never additional independent series.
"""
from pathlib import Path
import json
import sqlite3
import numpy as np
import pandas as pd

KEYS=['warehouse','store_id','item_id']
STOCHASTIC={'GA','SA','DQN','PPO','GA-DQN'}


def pareto_mask(cost,service):
    cost=np.asarray(cost,float);service=np.asarray(service,float)
    valid=np.isfinite(cost)&np.isfinite(service)
    answer=np.zeros(len(cost),bool)
    for i in np.flatnonzero(valid):
        dominates=valid&(cost<=cost[i])&(service>=service[i])&((cost<cost[i])|(service>service[i]))
        answer[i]=not dominates.any()
    return answer


def balanced_results(rows,policies,seeds):
    """Exclude unfinished cases and mismatched seed sets from all policy means."""
    unit=['dataset','origin']+KEYS
    required={(policy,int(seed),block) for policy in policies
              for seed in (seeds if policy in STOCHASTIC else [0])
              for block in ['validation','test']}
    if rows.duplicated(unit+['policy','seed','block']).any():
        raise ValueError('Duplicate policy/seed/block results')
    rows=rows[rows.policy.isin(policies)&rows.block.isin(['validation','test'])].copy()
    if not rows.TIC.map(np.isfinite).all():raise ValueError('Nonfinite inventory cost')
    counts=rows.groupby(unit,observed=True,dropna=False).size()
    # Labels as well as counts must agree: three wrong seeds are not complete.
    valid=[]
    for key,group in rows.groupby(unit,observed=True,sort=False):
        labels=set(zip(group.policy,group.seed.astype(int),group.block))
        if labels==required:valid.append(key)
    if not valid:return rows.iloc[:0].drop(columns=['seed']),len(counts)
    index=pd.MultiIndex.from_tuples(valid,names=unit)
    keep=pd.MultiIndex.from_frame(rows[unit]).isin(index)
    complete=rows.loc[keep]
    averaged=complete.groupby(unit+['profile','policy','block'],observed=True,sort=False)[['TIC','NS']].mean().reset_index()
    return averaged,len(counts)-len(valid)


def select_rule(means,alpha):
    means=means[np.isfinite(means.TIC)&np.isfinite(means.NS)].copy()
    if means.empty:return None,True
    feasible=means[means.NS>=alpha]
    chosen=(feasible.sort_values(['TIC','NS','policy'],ascending=[True,False,True]) if len(feasible)
            else means.sort_values(['NS','TIC','policy'],ascending=[False,True,True]))
    return chosen.iloc[0].policy,feasible.empty


def held_store_selections(averaged,thresholds,selection_block='validation',evaluation_block='test'):
    """Choose on other stores' validation; look up the held store's test KPIs."""
    results=[]
    for (dataset,origin),cohort in averaged.groupby(['dataset','origin'],observed=True):
        validation=cohort[cohort.block.eq(selection_block)]
        test=cohort[cohort.block.eq(evaluation_block)]
        # Subtract the held store's sufficient statistics instead of rescanning
        # every other store for every fold (quadratic in the number of stores).
        totals=validation.groupby('policy',observed=True)[['TIC','NS']].agg(['sum','count'])
        profiles=validation.groupby(['profile','policy'],observed=True)[['TIC','NS']].agg(['sum','count'])
        validation_stores=validation.groupby(['warehouse','store_id'],observed=True,sort=False)
        for (warehouse,store),held in test.groupby(['warehouse','store_id'],observed=True,sort=False):
            own=validation_stores.get_group((warehouse,store))
            own_totals=own.groupby('policy',observed=True)[['TIC','NS']].agg(['sum','count'])
            own_profiles=own.groupby(['profile','policy'],observed=True)[['TIC','NS']].agg(['sum','count'])
            def other_means(total,excluded):
                difference=total.subtract(excluded.reindex(total.index).fillna(0))
                return pd.DataFrame({metric:difference[(metric,'sum')]/difference[(metric,'count')].replace(0,np.nan)
                                     for metric in ['TIC','NS']}).reset_index()
            global_means=other_means(totals,own_totals)
            profile_means=other_means(profiles,own_profiles)
            for alpha in thresholds:
                global_policy,global_fallback=select_rule(global_means,alpha)
                if global_policy is None:continue
                for profile,profile_test in held.groupby('profile',observed=True):
                    local=profile_means[profile_means.profile.eq(profile)]
                    profile_policy,profile_fallback=select_rule(local,alpha)
                    missing=profile_policy is None
                    if missing:profile_policy,profile_fallback=global_policy,True
                    for strategy,policy,fallback in [('global',global_policy,global_fallback),('perfil',profile_policy,profile_fallback)]:
                        chosen=profile_test[profile_test.policy.eq(policy)].copy()
                        chosen['alpha']=alpha;chosen['strategy']=strategy;chosen['fallback']=fallback
                        chosen['profile_unseen']=missing if strategy=='perfil' else False
                        results.append(chosen)
    columns=list(averaged.columns)+['alpha','strategy','fallback','profile_unseen']
    return pd.concat(results,ignore_index=True) if results else pd.DataFrame(columns=columns)


def read_completed_cohort(db,dataset,origin,policies,seeds):
    """Average seeds in SQLite, avoiding millions of Python payload dictionaries."""
    pairs=[(policy,int(seed)) for policy in policies for seed in (seeds if policy in STOCHASTIC else [0])]
    values=','.join('(?,?)' for _ in pairs)
    query=f'''WITH expected(policy,seed) AS (VALUES {values}),
      entries AS (
        SELECT json_extract(j.value,'$.warehouse') warehouse,
               json_extract(j.value,'$.store_id') store_id,
               json_extract(j.value,'$.item_id') item_id,
               json_extract(j.value,'$.profile') profile,
               f.policy,f.seed,json_extract(j.value,'$.block') block,
               json_extract(j.value,'$.TIC') TIC,json_extract(j.value,'$.NS') NS
        FROM fits f JOIN expected x ON f.policy=x.policy AND f.seed=x.seed
        JOIN json_each(f.payload,'$.rows') j
        WHERE f.dataset=? AND f.origin=?),
      complete AS (
        SELECT warehouse,store_id,item_id FROM entries
        WHERE block IN ('validation','test') GROUP BY warehouse,store_id,item_id
        HAVING count(*)=? AND count(DISTINCT json_array(policy,seed,block))=?)
      SELECT e.warehouse,e.store_id,e.item_id,e.profile,e.policy,e.block,
             avg(e.TIC) TIC,avg(e.NS) NS
      FROM entries e JOIN complete USING(warehouse,store_id,item_id)
      WHERE block IN ('validation','test')
      GROUP BY e.warehouse,e.store_id,e.item_id,e.profile,e.policy,e.block'''
    size=len(pairs)*2
    rows=pd.read_sql_query(query,db,params=[value for pair in pairs for value in pair]+[dataset,int(origin),size,size])
    total=db.execute("SELECT count(*) FROM (SELECT json_extract(payload,'$.rows[0].warehouse'),json_extract(payload,'$.rows[0].store_id'),json_extract(payload,'$.rows[0].item_id') FROM fits WHERE dataset=? AND origin=? GROUP BY 1,2,3)",[dataset,int(origin)]).fetchone()[0]
    rows['dataset']=dataset;rows['origin']=origin
    if not rows.TIC.map(np.isfinite).all():raise ValueError('Nonfinite inventory cost')
    for col in ['dataset','warehouse','store_id','item_id','profile','policy','block']:rows[col]=rows[col].astype('category')
    return rows,total-len(rows[KEYS].drop_duplicates())


def prefix_features(wide,origin):
    """CV² of positive demands (standard diagnostic), distinct from legacy POD CV²."""
    values=wide.iloc[:,:origin].to_numpy(float)
    positive=values>0;n=positive.sum(axis=1)
    sums=values.sum(axis=1);squares=(values*values).sum(axis=1)
    mean=np.divide(sums,n,out=np.full(len(n),np.nan),where=n>0)
    variance=np.divide(squares-n*mean**2,n-1,out=np.full(len(n),np.nan),where=n>1)
    cv2=np.maximum(variance,0)/mean**2
    return pd.DataFrame({'adi':np.divide(origin,n,out=np.full(len(n),np.nan),where=n>0),
                         'cv2_positive':cv2,'zero_fraction':1-n/origin,'history_cycles':origin,
                         'available_cycles':wide.shape[1]})


def _save(fig,output,name):
    import matplotlib.pyplot as plt
    for suffix in ['png','svg']:
        fig.savefig(output/f'{name}.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def render_figures(frontier,selections,features,prefix_counts,output,alpha=.7,historical=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype']='none'
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from matplotlib.patches import Patch
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    datasets=sorted(set(frontier.dataset)|set(prefix_counts.dataset))
    policies=sorted(frontier.policy.unique())
    colors={p:plt.get_cmap('tab20')(i%20) for i,p in enumerate(policies)}
    figures=[]
    def panels(title,height=5):
        fig,axes=plt.subplots(1,len(datasets),figsize=(8*len(datasets),height),squeeze=False,layout='constrained')
        fig.suptitle(('Análise retrospectiva · ' if historical else '')+title);return fig,axes[0]
    fig,axes=panels('Custo × fill rate — médias no horizonte avaliado')
    for ax,dataset in zip(axes,datasets):
        data=frontier[frontier.dataset.eq(dataset)].copy()
        data['pareto']=pareto_mask(data.TIC,data.NS)
        ordered=data.sort_values(['NS','TIC','policy'],na_position='last')
        positions=[]
        for value in ordered.NS.fillna(0):positions.append(max(value*100,positions[-1]+6.5 if positions else 2))
        if positions and positions[-1]>100:
            positions[-1]=100
            for i in range(len(positions)-2,-1,-1):positions[i]=min(positions[i],positions[i+1]-6.5)
        span=max(float(data.TIC.max()-data.TIC.min()),1.)
        label_x=float(data.TIC.max())+.06*span
        for row,label_y in zip(ordered.itertuples(),positions):
            marker='*' if row.pareto else ('o' if row.NS>=alpha else 'x')
            ax.scatter(row.TIC,row.NS*100,c=[colors[row.policy]],marker=marker,s=180 if row.pareto else 65)
            ax.annotate(row.policy,(row.TIC,row.NS*100),xytext=(label_x,label_y),fontsize=8,
                        arrowprops={'arrowstyle':'-','color':'.6','lw':.6},va='center')
        ax.set_xlim(float(data.TIC.min())-.05*span,float(data.TIC.max())+.55*span)
        line=data[data.pareto].sort_values('TIC')
        ax.plot(line.TIC,line.NS*100,color='.4',ls='--',lw=1)
        ax.axhline(alpha*100,color='crimson',ls=':',label=f'Meta {alpha:.0%}')
        ax.set(title=dataset,xlabel='Custo médio no horizonte avaliado',ylabel='Fill rate médio (%)',ylim=(-3,105))
        ax.legend(handles=[Patch(facecolor='.4',label='★ Pareto; ○ demais viáveis; × abaixo da meta')],fontsize=8)
    _save(fig,output,'01_fronteira_custo_servico');figures.append('01_fronteira_custo_servico')
    selected=selections[np.isclose(selections.alpha,alpha)] if len(selections) else selections
    fig,axes=panels(f'Políticas selecionadas por perfil — meta {alpha:.0%}')
    for ax,dataset in zip(axes,datasets):
        data=selected[selected.dataset.eq(dataset)&selected.strategy.eq('perfil')]
        if data.empty:ax.text(.5,.5,'Sem dobras válidas de seleção',ha='center',transform=ax.transAxes);continue
        counts=pd.crosstab(data.profile,data.policy).reindex(columns=policies,fill_value=0)
        shares=counts.div(counts.sum(axis=1),axis=0)*100
        shares.plot.bar(stacked=True,ax=ax,color=[colors[p] for p in policies],legend=False)
        ax.set(title=dataset,ylabel='Séries–origem (%)',xlabel='Perfil operacional recebido' if historical else 'Perfil do prefixo de treino',ylim=(0,100))
        ax.tick_params(axis='x',labelrotation=30,labelsize=8)
    fig.legend(handles=[Patch(color=colors[p],label=p) for p in policies],loc='lower center',ncol=min(6,max(1,len(policies))),fontsize=8,bbox_to_anchor=(.5,-.13))
    _save(fig,output,'02_politicas_por_perfil');figures.append('02_politicas_por_perfil')
    fig,axes=panels(f'Distribuição do fill rate por série–origem — meta {alpha:.0%}')
    for ax,dataset in zip(axes,datasets):
        data=selected[selected.dataset.eq(dataset)]
        groups=[data[data.strategy.eq(s)].NS.dropna().to_numpy()*100 for s in ['global','perfil']]
        if any(len(g) for g in groups):ax.boxplot(groups,tick_labels=['Global','Por perfil'],showfliers=True)
        for i,strategy in enumerate(['global','perfil'],1):
            g=data[data.strategy.eq(strategy)]
            ax.text(i,1,f'n={len(g)}; indef.={g.NS.isna().sum()}\nabaixo={int((g.NS<alpha).sum())}',ha='center',fontsize=8)
        ax.axhline(alpha*100,color='crimson',ls='--')
        ax.set(title=dataset,ylabel='Fill rate no horizonte avaliado (%)',ylim=(-8,105))
    _save(fig,output,'03_distribuicao_fill_rate');figures.append('03_distribuicao_fill_rate')
    fig,axes=panels('Política modal selecionada por perfil e meta de serviço',height=6)
    thresholds=sorted(selections.alpha.unique()) if len(selections) else [alpha]
    cmap=ListedColormap([colors[p] for p in policies] or ['white']);cmap.set_bad('#eeeeee')
    for ax,dataset in zip(axes,datasets):
        data=selections[selections.dataset.eq(dataset)&selections.strategy.eq('perfil')]
        profiles=sorted(data.profile.unique());matrix=np.full((max(1,len(profiles)),len(thresholds)),np.nan)
        annotations={}
        for (profile,threshold),g in data.groupby(['profile','alpha'],observed=True):
            counts=g.policy.value_counts();winner=sorted(counts[counts.eq(counts.max())].index)[0]
            i=profiles.index(profile);j=thresholds.index(threshold);matrix[i,j]=policies.index(winner)
            annotations[i,j]=f'{winner}\n{counts.max()/len(g):.0%} | F:{g.fallback.mean():.0%}'
        ax.imshow(np.ma.masked_invalid(matrix),cmap=cmap,norm=BoundaryNorm(np.arange(len(policies)+1)-.5,cmap.N),aspect='auto')
        for (i,j),label in annotations.items():ax.text(j,i,label,ha='center',va='center',fontsize=7,bbox={'facecolor':'white','alpha':.8,'edgecolor':'none','pad':1})
        ax.set(title=dataset,xticks=range(len(thresholds)),xticklabels=[f'{a:.0%}' for a in thresholds],
               yticks=range(max(1,len(profiles))),yticklabels=profiles or ['Sem seleção'],xlabel='Meta; F = proporção de escolhas com fallback')
    _save(fig,output,'04_heatmap_perfil_limiar');figures.append('04_heatmap_perfil_limiar')
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained');fig.suptitle('Demanda no prefixo e extensão dos históricos — '+('coortes avaliadas' if historical else 'painéis elegíveis'))
    for ax,field,label in zip(axes.flat,['adi','cv2_positive','zero_fraction','available_cycles'],
                             ['ADI (log1p)','CV² positivo (log1p)','Proporção de zeros','Extensão total do histórico (ciclos)']):
        groups={}
        for dataset in datasets:
            values=features.loc[features.dataset.eq(dataset),field].dropna().to_numpy()
            if field in ['adi','cv2_positive']:values=np.log1p(values)
            if len(values):groups[dataset]=values
        bins=np.histogram_bin_edges(np.concatenate(list(groups.values())),bins=30) if groups else 30
        for dataset,values in groups.items():ax.hist(values,bins=bins,density=True,histtype='step',lw=1.8,label=f'{dataset} (n={len(values):,})')
        ax.set(xlabel=label,ylabel='Densidade');ax.legend()
    ax=axes[1,1];labels=[f'{r.dataset}/{r.origin}' for r in prefix_counts.itertuples()]
    ax.bar(range(len(labels)),prefix_counts.zero_positive_prefix,color='#d95f02')
    ax.set(xticks=range(len(labels)),xticklabels=labels,ylabel='Séries sem demanda no prefixo',title='Coorte recebida no relatório' if historical else 'Antes do filtro de elegibilidade')
    ax.tick_params(axis='x',labelrotation=30)
    for i,row in enumerate(prefix_counts.itertuples()):ax.annotate(f'{int(row.zero_positive_prefix):,}\n/{int(row.candidate_series):,}',(i,row.zero_positive_prefix),ha='center',va='bottom',fontsize=8)
    ax.margins(y=.18)
    scope=('Distribuições e barras: coorte recebida.\nPerfis recebidos do experimento.\nSem seleção temporal independente.' if historical else 'Distribuições: séries elegíveis nos painéis.\nBarras: todas as candidatas, antes do filtro.\nOrigens repetidas não são séries independentes.')
    axes[1,2].axis('off');axes[1,2].text(0,.9,'ADI = ciclos / ciclos positivos.\nCV² calculado apenas nas demandas positivas.\nZero no prefixo: ADI e CV² indefinidos.\n\n'+scope,va='top',fontsize=11)
    _save(fig,output,'05_caracterizacao_demanda');figures.append('05_caracterizacao_demanda')
    return figures


def generate_research_visualizations(run,thresholds=(.7,.8,.85,.9,.95),datasets=('BA','M5')):
    run=Path(run);manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    output=run/'visualizations';output.mkdir(exist_ok=True)
    policies=manifest['policies'];seeds=manifest.get('independent_training_seeds',manifest.get('training_seeds',[11,29,47]))
    averaged_parts=[];selection_parts=[];feature_parts=[];counts=[];coverage=[]
    db=sqlite3.connect(f'{(run/"results.sqlite").as_uri()}?mode=ro',uri=True) if (run/'results.sqlite').exists() else None
    try:
        for dataset in datasets:
            cache=run/'panels'/f'{dataset}_{dataset}.json'
            if not cache.exists():continue
            for meta in json.loads(cache.read_text(encoding='utf-8')):
                origin=meta['origin']
                if db is not None:
                    averaged,incomplete=read_completed_cohort(db,dataset,origin,policies,seeds)
                else:
                    rows=pd.read_csv(run/'kpis.csv',dtype={k:str for k in KEYS})
                    rows=rows[rows.dataset.eq(dataset)&rows.origin.eq(origin)]
                    for col in ['dataset','warehouse','store_id','item_id','profile','policy','block']:rows[col]=rows[col].astype('category')
                    averaged,incomplete=balanced_results(rows,policies,seeds)
                if len(averaged) or incomplete:
                    coverage.append({'dataset':dataset,'origin':origin,'complete_cases':len(averaged[KEYS].drop_duplicates()),'incomplete_cases':incomplete})
                    if len(averaged):
                        averaged_parts.append(averaged[averaged.block.eq('test')].groupby(['dataset','policy'],observed=True)[['TIC','NS']].agg(['sum','count']))
                        selections=held_store_selections(averaged,thresholds)
                        if not selections.empty:selection_parts.append(selections)
                wide=pd.read_parquet(run/meta['panel'])
                features=prefix_features(wide,origin);features['dataset']=dataset;features['origin']=origin
                feature_parts.append(features)
                counts.append({field:meta[field] for field in ['dataset','origin','candidate_series','zero_positive_prefix','eligible_series']})
    finally:
        if db is not None:db.close()
    if not averaged_parts:raise ValueError('No fully completed balanced BA/M5 cases; no comparative figures generated')
    totals=pd.concat(averaged_parts).groupby(level=['dataset','policy'],observed=True).sum()
    frontier=pd.DataFrame({field:totals[(field,'sum')]/totals[(field,'count')].replace(0,np.nan) for field in ['TIC','NS']}).reset_index()
    frontier['pareto']=False
    for _,group in frontier.groupby('dataset',observed=True):frontier.loc[group.index,'pareto']=pareto_mask(group.TIC,group.NS)
    selections=pd.concat(selection_parts,ignore_index=True) if selection_parts else pd.DataFrame(columns=['dataset','profile','strategy','policy','alpha','NS','fallback'])
    features=pd.concat(feature_parts,ignore_index=True);prefix_counts=pd.DataFrame(counts)
    figures=render_figures(frontier,selections,features,prefix_counts,output)
    frontier.to_csv(output/'frontier.csv',index=False);selections.to_csv(output/'selections_validation_store_out.csv',index=False)
    prefix_counts.to_csv(output/'prefix_counts.csv',index=False)
    features.to_parquet(output/'prefix_features_eligible.parquet',index=False)
    status_path=run/'status.json'
    source_status=json.loads(status_path.read_text(encoding='utf-8')).get('state') if status_path.exists() else manifest.get('status')
    report={'status':'exploratory diagnostics, not final policy efficacy evidence','source_run':run.name,'source_status':source_status,
        'policies':policies,'thresholds':list(thresholds),'coverage':coverage,'figures':figures,'selection':'Other stores validation; held-store test. Seeds averaged within series/origin.',
        'limitations':['Only cases with every declared policy/seed and both blocks enter comparisons.',
                       'Policy frontier is mean cost versus mean defined fill rate, not per-series dominance or simultaneous uncertainty inference.',
                       'Profile/threshold heatmap shows modal choices across folds; percentage is modal share, F is fallback rate; ties resolved alphabetically.',
                       'Zero-demand test fill rate remains undefined and is counted separately.',
                       'Demand distributions describe eligible cached panels; zero-prefix counts use all candidates from cohort metadata.',
                       'Repeated origins are repeated measurements. Partial runs and pilots must not be presented as final evidence.']}
    (output/'manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    return report
