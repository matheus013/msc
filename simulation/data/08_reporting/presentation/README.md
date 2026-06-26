# Artefatos visuais para a apresentação de qualificação

Gerados por `simulation/src/reporting/presentation_visuals.py`, a partir dos artefatos já existentes em `data/08_reporting/profiles/` e `data/08_reporting/strategy/`, e de `data/04_feature/demand_profiles.parquet` / `data/03_primary/scenarios.parquet`. Nenhum resultado de simulação foi recalculado ou alterado.

## Como reproduzir

```bash
cd simulation
python src/reporting/presentation_visuals.py
# ou, via Kedro (node registrado no pipeline de reporting):
kedro run --pipeline reporting --nodes generate_presentation_visuals
```

## Arquivos

| Arquivo | Descrição |
|---|---|
| `syntetos_boylan_scatter.png/.pdf/.csv` | Plano ADI×CV² com os 4 quadrantes, destacando Lumpy |
| `strategy_tradeoff_cti_ns.png/.pdf/.csv` | CTI médio e NS médio por estratégia (A1/A2/B/C) |
| `profile_dominance_bars.png/.pdf/.csv` | Política dominante por perfil operacional |
| `profile_policy_heatmap_simplified.png/.pdf/.csv` | Heatmap CTI restrito a políticas prioritárias + viabilidade NS≥0,70 |
| `aipe_evidence_pipeline.png/.pdf` | Diagrama dados → características → POD → simulação → rótulo → PSE → recomendação |
| `lumpy_series_examples.png/.pdf/.csv` | 3 séries Lumpy próximas da mediana ADI/CV² (opcional) |
| `manifest.json` | Metadados de proveniência de cada artefato |
| `figures_validation.md` | Checagens numéricas e alertas de divergência |

## Pendências / observações

- ALERTA: a Tabela 'Características do dataset por experimento' (docs/master_proposal/capitulos/resultados.tex, Seção 'Visão Geral') reporta 71% de séries Lumpy para o Experimento 2 (BA). Os artefatos atuais (demand_profiles.parquet e scenarios_meta.parquet, ambos com 145 séries) mostram 100.0% (coluna 'group' e quadrante recomputado de ADI/CV² concordam). Não foi encontrada nenhuma fonte de dados atual que sustente 71%. A própria Seção 'Experimento 2' do Capítulo de Resultados afirma que as 145 séries foram 'classificadas no quadrante Lumpy' (ou seja, 100%), o que é consistente com os dados, mas contradiz a tabela-resumo da Seção 'Visão Geral'. Recomenda-se revisar/corrigir essa tabela na dissertação; nenhuma alteração foi feita na dissertação ou na apresentação nesta tarefa.
- ALERTA (uso indevido evitado): a coluna 'dominant_policy' em demand_profiles.parquet é idêntica, por POD, à 'política de referência inicial' heurística da Tabela 4.1 da dissertação (GA-DQN/GA-PPO/PPO), e NÃO ao resultado empírico do benchmark (SA/EOQ/EOQ, ver profiles/dominant_policy_by_profile.csv). Esta coluna NÃO foi usada em nenhuma das novas visualizações; o 'oráculo por série' usado no gráfico de trade-off vem de strategy/strategy_cost_comparison.csv (estratégia C), que aplica corretamente a restrição NS≥0,70 por série.

## Integração com a apresentação (etapa futura)

Esta tarefa **não** alterou `docs/qualification_presentation/`. Sugestões de substituição/complemento de slides estão no relatório da tarefa que gerou estes artefatos (não persistido neste diretório).