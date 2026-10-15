"""Post-pilot diagnosis; no inventory fitting and no changes to frozen results.

The guarded classifier is a development candidate, not independent evidence.
Its fixed rule uses only other stores' validation labels and model scores.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
for variable in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[variable]='1'
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiment/src'))
import numpy as np
import pandas as pd
from simulation.research.pse import CASE,best_policy,fit_predict_fold,predict_loso,score_predictions,paired_store_intervals
from simulation.research.production import STOCHASTIC,SEEDS
from simulation.pipelines.demand_profiling.nodes import _series_features,classify_operational_profiles

NAMES={'AIPE_PSE':'PSE preditivo com perfil','PSE_sem_perfil':'PSE preditivo sem perfil',
       'Perfil':'PSE por perfil','Global':'Seleção global','Modo_rotulos':'Moda dos rótulos',
       'PSE_com_recuo':'PSE preditivo com recuo ao perfil','Escolha_validacao_individual':'Escolha individual na validação'}


def guarded_predictions(base,fold_diagnostics,min_score=.6,min_support=3):
    """Reject weak individual recommendations; fallback decided without test."""
    pred=base[base.strategy.eq('AIPE_PSE')].merge(fold_diagnostics,on=CASE,validate='one_to_one')
    fallback=base[base.strategy.eq('Perfil')][CASE+['policy']].rename(columns={'policy':'fallback_policy'})
    pred=pred.merge(fallback,on=CASE,validate='one_to_one')
    pred['used_fallback']=pred.max_class_score.lt(min_score)|pred.class_support.lt(min_support)
    pred['original_policy']=pred.policy
    pred.loc[pred.used_fallback,'policy']=pred.loc[pred.used_fallback,'fallback_policy']
    pred['strategy']='PSE_com_recuo'
    return pred


def labels_and_stability(kpis):
    validation=kpis[kpis.block.eq('validation')]
    test=kpis[kpis.block.eq('test')]
    records=[]
    for key,group in validation.groupby(CASE):
        identity=dict(zip(CASE,key));mask=np.logical_and.reduce([test[k].eq(v) for k,v in identity.items()]);tg=test[mask]
        vm=group.groupby('policy')[['TIC','NS']].mean();tm=tg.groupby('policy')[['TIC','NS']].mean()
        vl=best_policy(group);tl=best_policy(tg)
        seed_labels=[]
        for seed in SEEDS:
            matched=group[(group.policy.isin(STOCHASTIC)&group.seed.eq(seed))|(~group.policy.isin(STOCHASTIC)&group.seed.eq(0))]
            seed_labels.append(best_policy(matched))
        feasible=vm[vm.NS.ge(.7-1e-9)].sort_values('TIC')
        margin=(feasible.TIC.iloc[1]-feasible.TIC.iloc[0])/feasible.TIC.iloc[0] if len(feasible)>1 and feasible.TIC.iloc[0]>0 else None
        selected=tm.loc[vl];winner=tm.loc[tl]
        if tm.NS.ge(.7-1e-9).any():
            objective_stable=selected.NS>=.7-1e-9 and np.isclose(selected.TIC,winner.TIC,rtol=1e-6,atol=1e-8)
        else:
            equal_service=(pd.isna(selected.NS) and pd.isna(winner.NS)) or np.isclose(selected.NS,winner.NS,rtol=1e-6,atol=1e-8)
            objective_stable=equal_service and np.isclose(selected.TIC,winner.TIC,rtol=1e-6,atol=1e-8)
        records.append(identity|{'validation_label':vl,'test_label':tl,'same_label':vl==tl,
            'same_objective_allowing_ties':bool(objective_stable),
            'validation_demand':float(group.groupby('policy').Demand.first().iloc[0]),
            'test_demand':float(tg.groupby('policy').Demand.first().iloc[0]),
            'validation_feasible_policies':int(vm.NS.ge(.7-1e-9).sum()),
            'test_feasible_policies':int(tm.NS.ge(.7-1e-9).sum()),
            'validation_label_test_cost':float(tm.loc[vl,'TIC']),
            'validation_label_test_service':None if pd.isna(tm.loc[vl,'NS']) else float(tm.loc[vl,'NS']),
            'seed_labels_disagree':len(set(seed_labels))>1,'seed_labels':'|'.join(seed_labels),
            'validation_runner_up_relative_gap':margin})
    return pd.DataFrame(records)


def strategy_summary(scored):
    rows=[]
    for strategy,g in scored.groupby('strategy'):
        rows.append({'strategy':strategy,'name':NAMES[strategy],'cases':len(g),
            'mean_cost':g.TIC.mean(),'mean_service':g.NS.mean(),'defined_service_cases':int(g.NS.notna().sum()),
            'feasible_cases':int(g.feasible.sum()),'oracle_hits':int(g.oracle_hit.sum()),
            'comparable_cases':int(g.comparable.sum()),'mean_efficiency_percent':g.efficiency_percent.mean()})
    return pd.DataFrame(rows)


def service_failure_decomposition(scored,plans):
    """Separate initial-state impossibility, portfolio coverage and selection."""
    physical=pd.DataFrame([{k:p[k] for k in CASE}|{'physical_target_feasible':p['target_feasible']} for p in plans])
    data=scored.merge(physical,on=CASE,validate='many_to_one')
    data['service_outcome']=np.select([
        data.NS.isna(),~data.physical_target_feasible,data.oracle_service.lt(.7-1e-9),~data.feasible],
        ['demanda_zero','limite_fisico','lacuna_portfolio','erro_selecao'],default='atendeu')
    return data.groupby(['strategy','service_outcome']).size().unstack(fill_value=0)


def eligible_profile_counts(source):
    """Describe the frozen eligible cohort using each origin's prefix only."""
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    parent=source.parent/manifest['reuse_audit']['source'];records=[]
    for meta in manifest['source_panels']:
        path=parent/meta['panel']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['panel_sha256']:
            raise ValueError('Frozen panel changed')
        panel=pd.read_parquet(path);origin=meta['origin']
        features=pd.DataFrame([_series_features(d[:origin],17) for d in panel.to_numpy(float)])
        profiles=classify_operational_profiles(features,{})
        sparse_rule=features.adi.ge(1.32)&features.cv2.ge(.49)&features.zero_streak_max.ge(5)
        for name,count in profiles.operational_profile.value_counts().items():
            records.append({'origin':origin,'operational_profile':name,'count':int(count),'eligible_total':len(panel),
                'sparse_default_cases':int((profiles.operational_profile.eq('Sparse_High_Impact')&~sparse_rule).sum())})
    return pd.DataFrame(records)


def figures(output,scored,stability,features,importances):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    names=['Global','Perfil','AIPE_PSE','PSE_com_recuo']
    colors=['#4c78a8','#59a14f','#e15759','#b279a2']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for strategy,color in zip(names,colors):
        g=scored[scored.strategy.eq(strategy)]
        axes[0].scatter(g.NS.mean()*100,g.TIC.mean(),color=color,s=65,label=NAMES[strategy])
    axes[0].axvline(70,color='gray',ls='--');axes[0].set(xlabel='Fill rate médio (%)',ylabel='Custo médio',title='Custo e serviço no teste')
    axes[0].legend(fontsize=8,loc='best')
    axes[1].boxplot([scored[scored.strategy.eq(s)].NS.dropna()*100 for s in names],tick_labels=['Global','Por perfil','Preditivo','Com recuo'])
    axes[1].axhline(70,color='gray',ls='--');axes[1].set(ylabel='Fill rate por caso (%)',title='A média não garante serviço individual')
    save(fig,output,'01_custo_servico');plt.close(fig)
    matrix=pd.crosstab(stability.validation_label,stability.test_label)
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained');im=ax.imshow(matrix.to_numpy(),cmap='Blues')
    ax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=45,ha='right');ax.set_yticks(range(len(matrix.index)),matrix.index)
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):ax.text(j,i,str(matrix.iloc[i,j]),ha='center',va='center')
    ax.set(xlabel='Vencedora retrospectiva no teste',ylabel='Vencedora na validação',title='Estabilidade dos rótulos — 90 casos (inclui demanda zero)')
    fig.colorbar(im,ax=ax,label='Casos');save(fig,output,'02_estabilidade_rotulos');plt.close(fig)
    joined=scored.merge(features[CASE+['operational_profile']],on=CASE,validate='many_to_one')
    counts=pd.crosstab([joined[joined.strategy.isin(names)].strategy,joined[joined.strategy.isin(names)].operational_profile],joined[joined.strategy.isin(names)].policy)
    counts.index=[NAMES[a]+' / '+b.replace('Sparse_High_Impact','Esparso').replace('Unstable_Trend','Tendência') for a,b in counts.index]
    ax=counts.plot.barh(stacked=True,figsize=(11,6),colormap='tab20');ax.set(xlabel='Casos série–origem',ylabel='',title='Políticas recomendadas por abordagem e perfil')
    ax.legend(bbox_to_anchor=(1.01,1),loc='upper left');ax.figure.tight_layout();save(ax.figure,output,'03_escolhas_por_perfil');plt.close(ax.figure)
    baseline=scored[scored.strategy.eq('Global')];main=scored[scored.strategy.eq('AIPE_PSE')]
    pair=main.merge(baseline,on=CASE,suffixes=('_pse','_global'));pair['delta']=pair.TIC_pse-pair.TIC_global
    by_store=pair.groupby('store_id').delta.sum().sort_values()
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained');ax.barh(by_store.index,by_store,color=['#e15759' if x>0 else '#59a14f' for x in by_store])
    ax.axvline(0,color='gray',lw=.8);ax.set(xlabel='Diferença de custo total: PSE preditivo − global',title='Contribuição de cada loja para a diferença de custo')
    save(fig,output,'04_diferenca_por_loja');plt.close(fig)
    importance=importances.groupby('feature').importance.mean().sort_values().tail(12)
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained');ax.barh(importance.index,importance);ax.set(xlabel='Importância média por ganho (30 folds)',title='Características usadas pelo classificador com perfil')
    save(fig,output,'05_importancia_features');plt.close(fig)


def save(fig,output,name):
    fig.savefig(output/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(output/(name+'.svg'),bbox_inches='tight')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='m5_pse_pilot30_v2_2026-10-15');args=parser.parse_args()
    if not args.run.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run')
    source=ROOT/'experiment/data/08_reporting/research_v2'/args.run
    output=source/'diagnostics_v1';output.mkdir(exist_ok=True)
    paths=[source/n for n in ['manifest.json','features.csv','kpis.csv','predictions_frozen_before_test.csv','perfect_information.json']]
    paths.append(Path(__file__))
    paths.append(ROOT/'experiment/src/simulation/research/pse.py')
    paths.append(ROOT/'experiment/src/simulation/research/production.py')
    paths.append(ROOT/'experiment/src/simulation/pipelines/demand_profiling/nodes.py')
    fingerprint={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    protocol={'purpose':'Post-pilot exploratory diagnosis; no independent validation claim',
        'guard_rule':{'minimum_class_score':.6,'minimum_predicted_class_examples':3,'fallback':'PSE por perfil'},
        'rule_selection':'One fixed conservative candidate; thresholds not searched on test',
        'inputs_sha256':fingerprint,'simulations_trained':0,
        'versions':{package:importlib.metadata.version(package) for package in ['numpy','pandas','scikit-learn','xgboost','matplotlib']}}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    dtype={k:str for k in CASE[1:]}
    features=pd.read_csv(source/'features.csv',dtype=dtype)
    eligible=eligible_profile_counts(source)
    eligible.to_csv(output/'eligible_profile_counts.csv',index=False)
    kpis=pd.read_csv(source/'kpis.csv',dtype=dtype)
    base=pd.read_csv(source/'predictions_frozen_before_test.csv',dtype=dtype)
    validation=kpis[kpis.block.eq('validation')];test=kpis[kpis.block.eq('test')]
    reproduced=predict_loso(features,validation)
    compare=CASE+['strategy','policy','validation_label','label_unseen_in_training']
    pd.testing.assert_frame_equal(base[compare].sort_values(CASE+['strategy']).reset_index(drop=True),
                                  reproduced[compare].sort_values(CASE+['strategy']).reset_index(drop=True))
    labels=validation.groupby(CASE).apply(lambda g:best_policy(g),include_groups=False).rename('label')
    data=features.set_index(CASE).join(labels).reset_index();diagnostics=[];importances=[]
    for origin,cohort in data.groupby('origin'):
        for store in sorted(cohort.store_id.unique()):
            train=cohort[cohort.store_id.ne(store)];held=cohort[cohort.store_id.eq(store)]
            fitted=fit_predict_fold(train,held,True)
            training_accuracy=float(np.mean(fitted['classes'][fitted['model'].predict(train).astype(int)]==train.label.to_numpy()))
            for position,(_,case) in enumerate(held.iterrows()):
                diagnostics.append({k:case[k] for k in CASE}|{'max_class_score':float(fitted['max_class_score'][position]),
                    'class_support':int(fitted['class_support'][position]),'training_examples':len(train),
                    'training_classes':train.label.nunique(),'training_accuracy':training_accuracy,
                    'profile_training_support':int(train.operational_profile.eq(case.operational_profile).sum())})
            for name,value in fitted['feature_importance'].items():
                importances.append({'origin':origin,'held_store':store,'feature':name,'importance':value})
    fold=pd.DataFrame(diagnostics);importance=pd.DataFrame(importances)
    guarded=guarded_predictions(base,fold)
    # This rule and its predictions are frozen before the TEST KPIs are scored.
    guarded.to_csv(output/'guarded_predictions_before_scoring.csv',index=False)
    stability=labels_and_stability(kpis)
    carry=base[base.strategy.eq('AIPE_PSE')].copy();carry.policy=carry.validation_label;carry.strategy='Escolha_validacao_individual'
    all_predictions=pd.concat([base,guarded[base.columns],carry],ignore_index=True)
    scored=score_predictions(all_predictions,test)
    summary=strategy_summary(scored)
    enriched=scored.merge(features[CASE+['operational_profile']],on=CASE,validate='many_to_one')
    for columns,name in [(['strategy','origin'],'by_origin'),(['strategy','store_id'],'by_store'),(['strategy','operational_profile'],'by_profile')]:
        grouped=enriched.groupby(columns).agg(cases=('TIC','size'),mean_cost=('TIC','mean'),mean_service=('NS','mean'),defined_service=('NS','count'),feasible_cases=('feasible','sum'),oracle_hits=('oracle_hit','sum'))
        grouped.to_csv(output/(name+'.csv'))
    stability.to_csv(output/'label_stability.csv',index=False)
    fold.to_csv(output/'fold_diagnostics.csv',index=False);importance.to_csv(output/'feature_importances.csv',index=False)
    scored.to_csv(output/'scored_strategies.csv',index=False);summary.to_csv(output/'strategy_summary.csv',index=False)
    pd.crosstab(stability.validation_label,stability.test_label).to_csv(output/'label_transitions.csv')
    pd.crosstab(base.strategy,base.policy).to_csv(output/'policy_choices.csv')
    intervals=paired_store_intervals(scored)
    guarded_base=scored[scored.strategy.isin(['PSE_com_recuo','Global','Perfil','AIPE_PSE'])].copy()
    guarded_base.strategy=guarded_base.strategy.replace({'AIPE_PSE':'PSE_original','PSE_com_recuo':'AIPE_PSE'})
    guarded_intervals=paired_store_intervals(guarded_base)
    (output/'intervals_original.json').write_text(json.dumps(intervals,indent=2),encoding='utf-8')
    (output/'intervals_guarded.json').write_text(json.dumps(guarded_intervals,indent=2),encoding='utf-8')
    usable=stability[stability.validation_demand.gt(0)&stability.test_demand.gt(0)]
    individual=base[base.strategy.eq('AIPE_PSE')].merge(fold,on=CASE)
    profiles=features.groupby(['origin','operational_profile']).size()
    profile_importance=importance[importance.feature.str.startswith('profile__')].importance.sum()
    plan=json.loads((source/'perfect_information.json').read_text(encoding='utf-8'))
    failure=service_failure_decomposition(scored,plan)
    failure.to_csv(output/'service_failure_decomposition.csv')
    findings={'cases':len(stability),'original_predictions_reproduced':True,
        'validation_test_same_label_cases':int(stability.same_label.sum()),
        'same_objective_allowing_ties_cases':int(stability.same_objective_allowing_ties.sum()),
        'both_blocks_positive_demand_cases':len(usable),'same_label_both_positive_cases':int(usable.same_label.sum()),
        'seed_sensitive_label_cases':int(stability.seed_labels_disagree.sum()),
        'no_validation_feasible_policy_cases':int(stability.validation_feasible_policies.eq(0).sum()),
        'no_test_feasible_policy_cases':int(stability.test_feasible_policies.eq(0).sum()),
        'physically_impossible_test_cases':sum(not p['target_feasible'] for p in plan),
        'classifier_validation_label_hits':int(individual.policy.eq(individual.validation_label).sum()),
        'classifier_mean_training_accuracy':float(fold.training_accuracy.mean()),
        'with_without_profile_same_predictions':int((base[base.strategy.eq('AIPE_PSE')].set_index(CASE).policy==base[base.strategy.eq('PSE_sem_perfil')].set_index(CASE).policy).sum()),
        'profile_total_split_importance':float(profile_importance),
        'fallback_cases':int(guarded.used_fallback.sum()),'changed_policy_cases':int(guarded.policy.ne(guarded.original_policy).sum()),
        'mean_max_class_score':float(fold.max_class_score.mean()),'median_predicted_class_training_support':float(fold.class_support.median()),
        'classification_unseen_labels':int(individual.label_unseen_in_training.sum())}
    (output/'findings.json').write_text(json.dumps(findings,indent=2),encoding='utf-8')
    figures(output,scored,stability,features,importance)
    summary_index=summary.set_index('strategy')
    original=summary_index.loc['AIPE_PSE'];guard=summary_index.loc['PSE_com_recuo'];global_result=summary_index.loc['Global']
    pair=scored[scored.strategy.eq('AIPE_PSE')].merge(scored[scored.strategy.eq('Global')],on=CASE,suffixes=('_pse','_global'))
    pair['cost_delta']=pair.TIC_pse-pair.TIC_global
    store_contribution=pair.groupby('store_id').cost_delta.sum().sort_values(ascending=False)
    store_contribution.to_csv(output/'cost_difference_by_store.csv')
    report=['# Diagnóstico do piloto M5 — PSE\n',
        'Análise exploratória posterior ao piloto; não foram executadas novas simulações. As 450 recomendações originais foram reproduzidas exatamente. Resultados originais e snapshots permanecem preservados.\n',
        '## Abordagens\n',
        'PSE por perfil associa um perfil informado a uma política com resultados de validação de outras lojas. PSE preditivo estima uma política por série a partir de características e, opcionalmente, perfil. Seleção global é a referência comum.\n',
        '## Resultados no teste\n',
        '| Estratégia | Custo médio | Fill rate médio | Casos ≥70% | Acertos do oráculo |',
        '|---|---:|---:|---:|---:|']
    for r in summary.itertuples():report.append(f'| {r.name} | {r.mean_cost:.2f} | {100*r.mean_service:.2f}% | {r.feasible_cases}/{r.defined_service_cases} | {r.oracle_hits}/{r.comparable_cases} |')
    report+=['\nServiço médio exclui quatro casos com demanda de teste zero. Custo inclui todos os 90 casos. Acerto de oráculo exige serviço mínimo e empate no menor custo entre políticas viáveis; os denominadores variam por estratégia. Não comparar eficiências condicionais como se fossem calculadas nas mesmas séries.\n',
        '## Instabilidade dos rótulos e tamanho do treinamento\n',
        f'A vencedora da validação coincide com a vencedora do teste em {findings["validation_test_same_label_cases"]}/90 casos; restringindo a demanda positiva nos dois blocos, em {findings["same_label_both_positive_cases"]}/{len(usable)}. Permitindo empates numéricos no objetivo completo da regra de seleção, a escolha permanece ótima em {findings["same_objective_allowing_ties_cases"]}/90. Mudança de vencedora não implica, por si só, grande perda: a escolha individual de validação e os acertos com empate também são avaliados.\n',
        f'As três sementes produzem vencedoras distintas em {findings["seed_sensitive_label_cases"]}/90 casos. Isso é uma análise de sensibilidade, não uma estimativa de probabilidade com apenas três sementes. Não há política viável em {findings["no_validation_feasible_policy_cases"]}/90 casos de validação e {findings["no_test_feasible_policy_cases"]}/90 de teste, incluindo demanda zero; {findings["physically_impossible_test_cases"]} casos de teste têm alvo fisicamente impossível nas condições iniciais.\n',
        f'Cada fold tem 27 exemplos para até sete classes. O classificador reproduz {100*findings["classifier_mean_training_accuracy"]:.1f}% dos rótulos de treinamento em média, mas acerta apenas {findings["classifier_validation_label_hits"]}/90 rótulos da loja excluída. Isso é consistente com sobreajuste, além da instabilidade temporal. Dois rótulos pertencem a classes ausentes no treinamento do respectivo fold.\n',
        '## Perfil como entrada\n',
        'Cada origem tem 26 séries Sparse_High_Impact e quatro Unstable_Trend. Não houve transição de perfil nas séries amostradas. Esse recorte oferece pouca diversidade para avaliar o benefício geral dos perfis.\n',
        'A concentração também existe no universo elegível, não apenas na amostra: na origem 50, são 826 séries esparsas, 57 de tendência e duas High_Vol_Seasonal, em 885. Nas origens 64 e 78, são respectivamente 1.977/2.033 e 2.360/2.408 séries esparsas. Todos os rótulos esparsos satisfazem a regra explícita: não resultam do perfil padrão de fallback. Portanto, simplesmente aumentar a amostra não garante diversidade suficiente para estudar todos os perfis. Perfis raros exigem suporte reportado e recuo explícito. Esses números usam apenas o treino em cada origem.\n',
        f'Com e sem perfil, as {findings["with_without_profile_same_predictions"]}/90 recomendações são idênticas. A importância total das colunas de perfil por ganho é {profile_importance:.6f}: o modelo não fez splits nessas colunas. Os perfis são derivados das próprias características numéricas; essa redundância e o desbalanceamento ajudam a explicar a ausência de efeito, sem provar que perfis sejam inúteis em outras amostras.\n',
        '## Extensão conservadora exploratória\n',
        f'Foi testada uma única regra fixa: usar o PSE por perfil se o maior escore do classificador for menor que 0,60 ou se a classe prevista tiver menos de três exemplos no treinamento. O escore não é probabilidade calibrada de sucesso. Houve recuo em {findings["fallback_cases"]}/90 casos e mudança efetiva de política em {findings["changed_policy_cases"]}/90. A regra não usa o teste para decidir; sua avaliação aqui ocorre depois de observar o desempenho do piloto e exige validação independente futura. Não foi feita busca de limiares no teste.\n',
        f'O recuo reduz custo em {100*(1-guard.mean_cost/original.mean_cost):.2f}% e aumenta serviço em {100*(guard.mean_service-original.mean_service):.2f} pontos percentuais frente ao preditivo original. Frente à global, custa {100*(guard.mean_cost/global_result.mean_cost-1):.3f}% a mais e perde {100*(global_result.mean_service-guard.mean_service):.2f} pontos de serviço. Continua abaixo de 70% em média ({100*guard.mean_service:.3f}%). Não demonstrou superioridade sobre a global ou o PSE por perfil.\n',
        '## Onde ocorrem as perdas\n',
        f'As lojas {store_contribution.index[0]} e {store_contribution.index[1]} respondem por {100*store_contribution.iloc[:2].sum()/store_contribution.sum():.1f}% da diferença líquida de custo entre o preditivo original e a global. As tabelas por origem, loja e perfil estão nos CSVs; o original tem custo maior e serviço menor em ambos os perfis e nas três origens.\n',
        f'Para o PSE preditivo original: {int(failure.loc["AIPE_PSE"].get("atendeu",0))} casos atendem ao alvo, {int(failure.loc["AIPE_PSE"].get("erro_selecao",0))} falham apesar de existir uma política viável no portfólio, {int(failure.loc["AIPE_PSE"].get("lacuna_portfolio",0))} falham sem política viável no portfólio embora o planejamento perfeito consiga atender, {int(failure.loc["AIPE_PSE"].get("limite_fisico",0))} têm restrição física inicial e quatro têm demanda zero. Essas causas exigem intervenções diferentes; não atribuir toda falta ao seletor.\n',
        'Os rótulos do classificador impõem viabilidade por série; as associações global e por perfil usam médias por grupo. Essa diferença de objetivo faz parte das abordagens atuais e limita atribuir toda diferença ao aprendizado. A moda dos rótulos, que compartilha a regra de rotulagem, também é reportada como controle.\n',
        '## Decisão para o próximo protocolo\n',
        'Manter PSE por perfil e preditivo como abordagens distintas. Não ampliar a execução ainda. O classificador atual tem poucos exemplos por classe e rótulos sensíveis ao horizonte e à semente. Comparar a extensão conservadora com os mesmos controles e preservar os resultados negativos. Para a próxima etapa, aumentar diversidade de perfis por critérios do treino, registrar suporte mínimo por classe/perfil e reservar séries que não participaram deste desenvolvimento. Seleção de hiperparâmetros ou limiares deve ocorrer em validação interna por loja; teste final fica isolado.\n',
        'Custos e serviços são médias entre sementes antes da avaliação por caso. As três origens da mesma série são dependentes. Intervalos são bootstrap exploratório de dez lojas, sem correção para múltiplas comparações. A coorte piloto é condicional à elegibilidade na primeira origem e não representa todas as FOODS.\n',
        'A separação entre desenvolvimento e avaliação segue a discussão de [Cawley e Talbot (2010)](https://www.jmlr.org/beta/papers/v11/cawley10a.html). O cuidado com escores e probabilidades está alinhado à [documentação de calibração do scikit-learn](https://scikit-learn.org/stable/api/sklearn.calibration.html).\n',
        '## Figuras\n']
    for name,title in [('01_custo_servico','Custo e atendimento'),('02_estabilidade_rotulos','Transição dos rótulos'),('03_escolhas_por_perfil','Escolhas por perfil'),('04_diferenca_por_loja','Diferença por loja'),('05_importancia_features','Importância das características')]:report.append(f'![{title}]({name}.png)\n')
    (output/'report.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps(findings,indent=2));print(summary.to_string(index=False));print('REPORT',output/'report.md')


if __name__=='__main__':main()
