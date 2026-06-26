# Validação dos novos artefatos de apresentação

Gerado em: 2026-06-26 14:18

## Fontes usadas
- `data\04_feature\demand_profiles.parquet`
- `data\08_reporting\profiles\profile_policy_metrics.csv`
- `data\08_reporting\profiles\dominant_policy_by_profile.csv`
- `data\08_reporting\strategy\strategy_cost_comparison.csv`
- `data\03_primary\scenarios.parquet`

## Checagens

- Total de séries (demand_profiles.parquet): **145** ✅ (== 145)
- Distribuição por POD: Sparse High Impact=116, Unstable Trend=18, High Vol. Seasonal=11
- Séries no quadrante Lumpy (recomputado de ADI/CV² em demand_profiles.parquet): **100.0%** (145/145)
- Redução de CTI da estratégia B vs. política única: **+6.17%** ✅ (≈6,2%)
- Estratégia C (oráculo por série) marcada como exploratória na figura e descrita como 'Oráculo por série (exploratório)' no artefato fonte. ✅
- Jornaleiro não aparece como política dominante em nenhum perfil com NS médio < 0,70. ✅
- Soma de séries nos perfis com representação no Experimento 2: **145** ✅ (== 145)

## Alertas / divergências

- ALERTA: a Tabela 'Características do dataset por experimento' (docs/master_proposal/capitulos/resultados.tex, Seção 'Visão Geral') reporta 71% de séries Lumpy para o Experimento 2 (BA). Os artefatos atuais (demand_profiles.parquet e scenarios_meta.parquet, ambos com 145 séries) mostram 100.0% (coluna 'group' e quadrante recomputado de ADI/CV² concordam). Não foi encontrada nenhuma fonte de dados atual que sustente 71%. A própria Seção 'Experimento 2' do Capítulo de Resultados afirma que as 145 séries foram 'classificadas no quadrante Lumpy' (ou seja, 100%), o que é consistente com os dados, mas contradiz a tabela-resumo da Seção 'Visão Geral'. Recomenda-se revisar/corrigir essa tabela na dissertação; nenhuma alteração foi feita na dissertação ou na apresentação nesta tarefa.
- ALERTA (uso indevido evitado): a coluna 'dominant_policy' em demand_profiles.parquet é idêntica, por POD, à 'política de referência inicial' heurística da Tabela 4.1 da dissertação (GA-DQN/GA-PPO/PPO), e NÃO ao resultado empírico do benchmark (SA/EOQ/EOQ, ver profiles/dominant_policy_by_profile.csv). Esta coluna NÃO foi usada em nenhuma das novas visualizações; o 'oráculo por série' usado no gráfico de trade-off vem de strategy/strategy_cost_comparison.csv (estratégia C), que aplica corretamente a restrição NS≥0,70 por série.

## Resultado científico

Nenhum número de CTI, NS, TR, BE, FP ou estratégia foi recalculado ou alterado nesta tarefa. Todas as figuras leem diretamente os artefatos já existentes em `data/08_reporting/profiles/` e `data/08_reporting/strategy/`, ou recomputam apenas características já presentes em `demand_profiles.parquet` (ADI, CV², quadrante Syntetos-Boylan), sem reexecutar a simulação.