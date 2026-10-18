"""Export a completed AIPE run into an auditable decision-support preview.

Reads existing files only; does not import simulation code, refit or score policies.
Decision records never read test metrics. Retrospective outcomes are a separate file.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
KEYS = ['origin', 'warehouse', 'store_id', 'item_id']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(x) for x in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, 'item'):
        return clean(value.item())
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def identity(row):
    return (int(row['origin']), *(str(row[k]) for k in KEYS[1:]))


def decision_id(key, strategy):
    return hashlib.sha256(json.dumps([list(key), strategy], ensure_ascii=False).encode()).hexdigest()[:24]


def final_prediction(row):
    """Stored predicted_cost/service describe the candidate, even after fallback."""
    chosen, candidate, reference = row['policy'], row.get('candidate_policy'), row.get('reference_policy')
    if chosen == candidate:
        return row.get('predicted_cost'), row.get('predicted_service'), 'candidate_prediction'
    if chosen == reference:
        return row.get('reference_predicted_cost'), row.get('reference_predicted_service'), 'reference_prediction'
    return None, None, 'not_persisted_for_this_strategy'


def make_decisions(predictions, panels, profile_lookup, run_name):
    """Inputs deliberately exclude test outcomes and validation winner diagnostics."""
    global_rows = predictions[predictions.strategy.eq('Global')]
    references = {identity(r): r['policy'] for r in global_rows.to_dict('records')}
    records = []
    for row in predictions.to_dict('records'):
        key = identity(row)
        reference = references[key]
        if pd.notna(row.get('reference_policy')) and row['reference_policy'] != reference:
            raise ValueError('Reference differs from frozen Global recommendation')
        panel = panels[key[0]]
        prefix = panel.loc[key[1:]].to_numpy(float)[:key[0]]
        cost, service, prediction_source = final_prediction(row)
        candidate = row.get('candidate_policy')
        candidate = candidate if pd.notna(candidate) else None
        margin = row.get('margin')
        margin_mode = 'always_reference' if isinstance(margin, float) and math.isinf(margin) else 'finite' if pd.notna(margin) else 'not_applicable'
        guarded = row['strategy'] == 'Consequencia_recuo'
        reason = ('regra_de_recuo_manteve_referencia' if guarded and row['policy'] == reference and candidate != reference
                  else 'seletor_escolheu_mesma_familia_da_referencia' if row['policy'] == reference
                  else 'recomendacao_contextual_difere_da_referencia')
        records.append({'decision_id': decision_id(key, row['strategy']),
            'run': run_name, **dict(zip(KEYS, key)), 'strategy': row['strategy'],
            'recommended_policy': row['policy'], 'reference_policy': reference,
            'candidate_policy': candidate, 'changed_reference': row['policy'] != reference,
            'reason_code': reason, 'profile': profile_lookup[key],
            'profile_origin': 'derived_from_training_prefix',
            'calibration_cutoff_cycle': key[0], 'assessment_start_cycle': key[0]+8,
            'horizon_cycles': 7, 'training_periods': [str(x) for x in panel.columns[:key[0]]],
            'training_sales': prefix.tolist(),
            'validation_periods': [str(x) for x in panel.columns[key[0]:key[0]+7]],
            'assessment_periods': [str(x) for x in panel.columns[key[0]+7:key[0]+14]],
            'expected_cost': cost, 'expected_fill_rate': service,
            'expected_value_source': prediction_source,
            'candidate_expected_cost': row.get('predicted_cost'),
            'candidate_expected_fill_rate': row.get('predicted_service'),
            'reference_expected_cost': row.get('reference_predicted_cost'),
            'reference_expected_fill_rate': row.get('reference_predicted_service'),
            'margin': margin, 'margin_mode': margin_mode,
            'cost_unit': 'scenario_cost_units', 'service_measure': 'quantity_fill_rate',
            'service_target': .7, 'lead_time_cycles': 2,
            'costs': {'holding_per_unit_cycle': 1, 'shortage_per_unit': 5,
                      'fixed_per_order': 50, 'variable_per_order_unit': .5},
            'input_status': {'sales': 'observed_sales_treated_as_demand',
                'costs': 'scenario_assumption', 'lead_time': 'scenario_assumption',
                'initial_inventory_and_pipeline': 'synthetic_common_state',
                'operational_constraints': 'not_integrated'},
            'policy_parameters': None, 'policy_parameters_status': 'not_persisted_in_canonical_run',
            'ready_for_purchase_order': False,
            'human_review': {'status': 'not_collected', 'decision': None, 'comment': None}})
    return clean(records)


def build_html(records, outcomes):
    # JSON is escaped for HTML script context; all dynamic text uses textContent.
    data = json.dumps(clean({'decisions': records, 'outcomes': outcomes}), ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    return '''<!doctype html><html lang="pt-BR"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AIPE — relatório de recomendações</title>
<style>body{font:16px system-ui;background:#f3f6fa;color:#17293d;margin:auto;padding:28px;max-width:1100px}h1{margin-bottom:8px}p{line-height:1.5}.notice{background:#fff2cf;padding:16px;border-left:5px solid #b57a12}.filters{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}select,input{padding:10px;max-width:100%;font:inherit}article{background:white;padding:24px;border:1px solid #d8e0eb;border-radius:10px}dl{display:grid;grid-template-columns:minmax(150px,1fr) 2fr;gap:12px}dt{font-weight:600}dd{margin:0;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}summary{cursor:pointer;padding:12px 0}a{color:#135b9a}small{color:#52677e}</style>
<h1>AIPE · apoio à decisão de estoques</h1><p>Prévia de pesquisa baseada em recomendações congeladas. Escolha a estratégia e o contexto; não há ranking de lojas.</p>
<div class="notice">Custos em unidades de cenário, sem conversão para reais. Estoques e prazos simulados. Parâmetros finais não foram persistidos: esta recomendação não é uma ordem de compra executável.</div>
<div class="filters"><label>Estratégia <select id="strategy"></select></label><label>Produto ou contexto <input id="search" placeholder="Filtrar identificadores"></label><label>Registro <select id="case"></select></label></div>
<p id="count"></p><article><h2 id="title"></h2><dl id="fields"></dl>
<details><summary>Informações e hipóteses disponíveis na decisão</summary><pre id="record"></pre></details>
<details><summary>Avaliação posterior · resultado observado na simulação</summary><p>Este resultado não é previsão nem justificativa disponível antes do teste. A interpretação exige considerar incerteza e o conjunto de casos.</p><pre id="observed"></pre></details></article>
<p>Arquivos abertos: <a href="recommendations.csv">recomendações CSV</a> · <a href="decisions.json">decisões JSON</a> · <a href="alternatives_validation.csv">alternativas na validação</a> · <a href="outcomes_retrospective.csv">avaliação posterior</a> · <a href="candidate_metrics_by_seed.csv">indicadores por semente</a> · <a href="manifest.json">manifesto</a> · <a href="README.md">limitações e dicionário</a></p>
<small>O painel não grava aprovação humana. IDs de contexto identificam registros experimentais.</small>
<script id="data" type="application/json">'''+data+'''</script><script>
const data=JSON.parse(document.getElementById('data').textContent),byId=new Map(data.outcomes.map(r=>[r.decision_id,r]));
const strategy=document.getElementById('strategy'),cases=document.getElementById('case'),search=document.getElementById('search');
for(const s of [...new Set(data.decisions.map(r=>r.strategy))])strategy.add(new Option(s,s));
function fmt(v,percent=false){return v===null||v===undefined?'Não disponível':(percent?(100*v).toFixed(2)+'%':Number(v).toLocaleString('pt-BR',{maximumFractionDigits:2}));}
function show(){const r=data.decisions.find(r=>r.decision_id===cases.value);document.getElementById('fields').replaceChildren();if(!r){document.getElementById('title').textContent='Nenhum registro';document.getElementById('record').textContent='';document.getElementById('observed').textContent='';return;}
document.getElementById('title').textContent='Família recomendada: '+r.recommended_policy;
const fields={'Contexto':r.store_id+' · produto '+r.item_id+' · origem '+r.origin,'Estratégia':r.strategy,'Referência':r.reference_policy,'Perfil derivado':r.profile,'Motivo registrado':r.reason_code,'Custo previsto da recomendação':fmt(r.expected_cost)+' · unidades de cenário','Atendimento previsto':fmt(r.expected_fill_rate,true),'Parâmetros da política':'Não persistidos nesta execução','Uso':'Revisão humana; não executar compra a partir desta prévia'};
for(const [k,v] of Object.entries(fields)){let dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=k;dd.textContent=v;document.getElementById('fields').append(dt,dd);}
document.getElementById('record').textContent=JSON.stringify(r,null,2);document.getElementById('observed').textContent=JSON.stringify(byId.get(r.decision_id)||{status:'não disponível'},null,2);}
function filter(){cases.replaceChildren();const q=search.value.trim().toLowerCase();for(const r of data.decisions.filter(r=>r.strategy===strategy.value&&[r.store_id,r.item_id,r.origin].join(' ').toLowerCase().includes(q))){cases.add(new Option(r.store_id+' · '+r.item_id+' · origem '+r.origin,r.decision_id));}document.getElementById('count').textContent=cases.length+' decisões neste filtro';show();}
strategy.onchange=filter;search.oninput=filter;cases.onchange=show;filter();</script></html>'''


def export(run, output):
    run, output = run.resolve(), output.resolve()
    allowed = (ROOT/'experiment/data/decision_reports').resolve()
    if not output.is_relative_to(allowed) or output == allowed or output.exists():
        raise ValueError('Use a new directory inside experiment/data/decision_reports')
    status = json.loads((run/'status.json').read_text(encoding='utf-8'))
    if status['state'] != 'completed':
        raise ValueError('Only completed runs can be exported; the active M5 queue is not touched')
    plan = json.loads((run/'plan.json').read_text(encoding='utf-8'))
    paths = [run/'plan.json', run/'manifest.json', run/'kpis.csv',
             run/'selection/predictions_frozen_before_scoring.csv', run/'selection/paired_reference.csv']
    frozen_protocol = run/'code_snapshot/experiment/src/simulation/research/protocol.py'
    frozen_key = str(frozen_protocol.relative_to(run/'code_snapshot'))
    if sha(frozen_protocol) != plan['code_sha256'][frozen_key]:
        raise ValueError('Frozen protocol does not match plan')
    paths.append(frozen_protocol)
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    # Fail closed if scenario constants used in this format differ from audited protocol.
    source = frozen_protocol.read_text(encoding='utf-8')
    for literal in ["'lead_time':2", "'alpha_min':.7", "'holding_cost_per_unit':1.",
                    "'stockout_cost_per_unit':5.", "'ordering_cost_per_order':50.", "'ordering_cost_per_unit':.5"]:
        if literal not in source:
            raise ValueError('Scenario differs from exporter version; audit mapping first')
    dtypes = {k: str for k in KEYS[1:]}
    predictions = pd.read_csv(paths[3], dtype=dtypes)
    kpis = pd.read_csv(run/'kpis.csv', dtype=dtypes)
    if predictions.duplicated(KEYS+['strategy']).any() or kpis.duplicated(KEYS+['policy','seed','block']).any():
        raise ValueError('Duplicate decision or candidate metrics')
    panels = {}
    for meta in plan['panel_metadata']:
        path = run/meta['panel']
        if sha(path) != meta['panel_sha256']:
            raise ValueError('Frozen panel changed')
        panels[meta['origin']] = pd.read_parquet(path)
        hashes[str(path.relative_to(ROOT))] = sha(path)
    profiles = {}
    for key,g in kpis.groupby(KEYS):
        if g.profile.nunique() != 1:
            raise ValueError('Profile differs across fits')
        profiles[tuple(key)] = g.profile.iloc[0]
    records = make_decisions(predictions, panels, profiles, run.name)
    expected = plan['series_origin_cases']
    if not predictions.groupby('strategy').size().eq(expected).all():
        raise ValueError('Missing recommendation cases')
    # This file is accessed only after decision records are formed.
    paired = pd.read_csv(run/'selection/paired_reference.csv', dtype=dtypes)
    if paired.duplicated(KEYS+['strategy']).any():
        raise ValueError('Duplicate retrospective outcomes')
    selected = {r['decision_id']:r['recommended_policy'] for r in records}
    outcomes = []
    for r in paired.to_dict('records'):
        if selected.get(decision_id(identity(r),r['strategy'])) != r['policy']:
            raise ValueError('Scored family differs from frozen recommendation')
        outcomes.append({'decision_id': decision_id(identity(r),r['strategy']),
            **{k:r[k] for k in KEYS}, 'strategy':r['strategy'],
            'observed_simulated_cost':r['TIC'], 'observed_fill_rate':r['NS'],
            'reference_observed_simulated_cost':r['TIC_reference'],
            'reference_observed_fill_rate':r['NS_reference'],
            'cost_difference':r['cost_difference'], 'fill_rate_difference':r['service_difference'],
            'new_service_failure':r['new_service_failure'], 'role':'retrospective_test_only'})
    if {r['decision_id'] for r in records} != {r['decision_id'] for r in outcomes}:
        raise ValueError('Outcome and decision IDs differ')
    alternatives = kpis[kpis.block.eq('validation')].groupby(KEYS+['policy'],as_index=False).agg(
        validation_cost_mean=('TIC','mean'),validation_fill_rate_mean=('NS','mean'),
        seeds=('seed','size'),defined_fill_rate_seeds=('NS','count'))
    alternatives['role'] = 'historical_validation_not_forecast_for_target'
    if any(sha(ROOT/p) != digest for p,digest in hashes.items()):
        raise ValueError('Inputs changed while exporting')
    output.mkdir(parents=True)
    write_json(output/'decisions.json',records)
    flat = [{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in records]
    pd.DataFrame(flat).to_csv(output/'recommendations.csv',index=False)
    pd.DataFrame(clean(outcomes)).to_csv(output/'outcomes_retrospective.csv',index=False)
    alternatives.to_csv(output/'alternatives_validation.csv',index=False)
    kpis.to_csv(output/'candidate_metrics_by_seed.csv',index=False)
    (output/'index.html').write_text(build_html(records,outcomes),encoding='utf-8')
    readme = '''# Relatório de recomendação AIPE — prévia de pesquisa

Abra index.html no navegador. Não requer servidor ou internet.

decisions.json contém o histórico de calibração, a família recomendada, referência,
perfil e estimativas persistidas disponíveis antes da pontuação do teste.
recommendations.csv é sua versão tabular, sem os campos aninhados.
expected_cost e expected_fill_rate correspondem à família efetivamente recomendada:
quando há recuo, são usadas as previsões da referência, não da candidata rejeitada.
Ausência de previsão permanece vazia/nula. margin_mode=always_reference representa
uma margem infinita, cuja versão numérica JSON é nula. Perfis são derivados.

alternatives_validation.csv resume médias de sementes na validação por família.
São resultados históricos simulados; não são previsões de desempenho do alvo.
candidate_metrics_by_seed.csv conserva todos os indicadores de validação e teste
por política/semente. NS é fill rate [0,1]; Demand e Served são quantidades;
TIC é custo total de cenário; TR é proporção de períodos com ruptura; BE é razão
de variâncias dos pedidos e demanda; FP é frequência de pedidos por período.
TerminalCommittedUnits é saldo final mais pipeline. Custos desagregados, ações,
pesos dos modelos e parâmetros finais não foram persistidos neste caminho.

outcomes_retrospective.csv contém somente avaliação posterior de teste;
cost_difference é AIPE menos referência, portanto negativo indica economia.
fill_rate_difference é diferença em fração: 0,01 corresponde a 1 ponto percentual.
Os resultados posteriores não foram usados para gerar ou justificar as decisões
deste exportador. A prévia contém múltiplas instanciações, sem escolher a vencedora
do teste como recomendação principal para todas as situações.

Não há custo calibrado em reais, intervalo de previsão por decisão, garantia de
atendimento ou validação de uso por gerentes. Estoque/pipeline inicial são
sintéticos, e não saldos observados. Não executar compras com esta prévia.
assessment_start_cycle usa posição 1-based; origin é comprimento do treino.
A seleção usa validações de outras lojas antes do teste. Rótulos M5 YYYYCC são
sintéticos. Human_review=not_collected significa ausência de decisão humana;
o HTML não coleta nem grava parecer. IDs operacionais ficam em pasta local de dados.

manifest.json liga saídas a hashes dos arquivos de entrada e do exportador.
O arquivo não contém dados de estoque reais nem completa ausências silenciosamente.
'''
    (output/'README.md').write_text(readme,encoding='utf-8')
    write_json(output/'manifest.json',{'format_version':'1.0','created_utc':datetime.now(timezone.utc).isoformat(),
        'run':run.name,'dataset':plan['dataset'],'cases':expected,'decision_records':len(records),
        'strategies':sorted(predictions.strategy.unique().tolist()),
        'status':'research_preview_not_operational_purchase_order',
        'source_hashes':hashes,'exporter_sha256':sha(Path(__file__)),
        'outputs':{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()},
        'limitations':['scenario_costs','synthetic_initial_state','missing_fitted_policy_parameters',
                       'no_complete_decision_uncertainty','no_human_evaluation']})
    print(json.dumps({'output':str(output),'cases':expected,'decisions':len(records),'training_started':False}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    export(args.run,args.output)
