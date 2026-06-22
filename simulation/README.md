# simulation - Pipeline Kedro do AIPE

Pipeline de simulação, validação estatística e geração de relatórios para a dissertação de mestrado sobre seleção contextual de políticas de reposição.

O projeto suporta recortes por estado e produto, com parâmetros configuráveis em YAML.

## Instalação

```bash
cd simulation
pip install -e . -r requirements.txt
```

## Execução

```bash
# Pipeline completo com configuração padrão
kedro run

# Fase 1: Paraíba
kedro run --params "data_ingestion.states=['PB']"

# Fase 2: Bahia
kedro run --params "data_ingestion.states=['BA']"

# Validação estatística
kedro run --pipeline statistical_validation

# Figuras, tabelas e relatórios
kedro run --pipeline reporting

# Visualização do DAG
kedro viz
```

## Pipelines

| Pipeline | Descrição |
|---|---|
| `data_ingestion` | Carrega Parquet por UF, filtra dados e gera cenários por estado, produto e loja. |
| `demand_profiling` | Classifica séries em Perfis Operacionais de Demanda usando características como ADI e CV². |
| `demand_forecasting` | Treina modelos auxiliares de previsão e calcula métricas de erro. |
| `inventory_simulation` | Simula as políticas candidatas em ambiente comum de inventário. |
| `policy_selection` | Gera rótulos de política dominante e artefatos para o Policy Selection Engine. |
| `statistical_validation` | Executa Wilcoxon, Friedman, Nemenyi e tamanhos de efeito. |
| `reporting` | Gera figuras PDF, tabelas LaTeX e relatórios de validação. |

## Configuração

Os parâmetros ficam em:

```text
conf/base/parameters/
```

| Arquivo | Controla |
|---|---|
| `data_ingestion.yml` | Estados, produtos e intervalo de datas. |
| `data_cleaning.yml` | Regras de limpeza, dados ausentes e outliers. |
| `demand_profiling.yml` | Limiares de ADI, CV², burstiness e sazonalidade. |
| `forecasting.yml` | Modelos, janelas e hiperparâmetros de previsão. |
| `simulation.yml` | Custos, lead time, réplicas e políticas avaliadas. |
| `policy_selection.yml` | Critério de rótulo, NS mínimo e sementes aleatórias. |
| `reporting.yml` | Formato de figuras, tabelas e relatórios. |

## Políticas avaliadas

| Categoria | Políticas |
|---|---|
| Clássicas | EOQ, `(s,S)`, Jornaleiro |
| Meta-heurísticas | GA, SA, PSO, DE |
| Aprendizado por reforço | DQN, PPO, SARSA |
| Híbridas | GA-DQN, GA-PPO |

## Métricas

| Sigla | Descrição |
|---|---|
| CTI | Custo Total de Inventário |
| NS | Nível de Serviço |
| TR | Taxa de Ruptura |
| BE | Efeito Bullwhip |
| FP | Frequência de Pedidos |

## Estrutura do código

```text
src/simulation/
|-- core/
|   |-- inventory_env.py
|   |-- policies.py
|   |-- forecasting.py
|   `-- visualizations.py
`-- pipelines/
    |-- data_ingestion/
    |-- demand_profiling/
    |-- demand_forecasting/
    |-- inventory_simulation/
    |-- policy_selection/
    |-- statistical_validation/
    `-- reporting/
```

## Saídas principais

```text
data/08_reporting/
|-- comparison/
|-- demand/
|-- forecast/
|-- maps/
|-- profiles/
|-- statistical/
`-- strategy/
```

Essas saídas alimentam diretamente as tabelas e figuras da proposta em `docs/master_proposal`.
