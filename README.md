# AIPE — Adaptive Inventory Policy Engine

Repositório da dissertação de mestrado e artigo SBPO 2026 sobre seleção adaptativa de políticas de reposição em redes warehouse→loja com demanda intermitente, usando dados reais de varejo regional brasileiro.

---

## Estrutura

```
sbpo/
├── docs/
│   ├── sbpo/                  ← Artigo SBPO 2026 (LaTeX)
│   │   ├── main.tex
│   │   └── figuras/
│   ├── master_proposal/       ← Proposta de dissertação (LaTeX/XeLaTeX)
│   │   ├── tcc.tex
│   │   ├── capitulos/
│   │   ├── figures/
│   │   └── build/             ← Artefatos de compilação (gitignored)
│   └── references/            ← PDFs de referência
│
└── simulation/                ← Pipeline Kedro
    ├── conf/base/parameters/  ← YAMLs configuráveis
    ├── src/simulation/
    │   ├── core/              ← InventoryEnv, políticas, forecasters
    │   └── pipelines/         ← 7 pipelines Kedro
    └── data/                  ← Dados (gitignored)
```

---

## Dissertação de Mestrado

**"Um Framework Adaptativo para Seleção de Políticas de Reposição em Regimes Operacionais Heterogêneos"**
UFAL — Instituto de Computação

Propõe o AIPE (*Adaptive Inventory Policy Engine*), framework que seleciona automaticamente a política de reposição mais adequada para cada série loja-produto com base no perfil operacional de demanda (taxonomia ADI × CV²).

### Capítulos

| # | Título |
|---|---|
| 1 | Introdução |
| 2 | Referencial Teórico |
| 3 | Trabalhos Relacionados |
| 4 | Metodologia |
| 5 | Resultados Preliminares |
| 6 | Considerações Finais e Próximas Etapas |

### Compilar

```bash
cd docs/master_proposal
latexmk -pdf -xelatex -outdir=build -interaction=nonstopmode tcc.tex
# PDF gerado em build/tcc.pdf
```

---

## Artigo SBPO 2026

**"Otimização Multi-Echelon de Inventário com Demanda Intermitente: Uma Arquitetura Híbrida GA-RL"**

Avalia 12 políticas sobre dados reais (Fase 1: Paraíba, Fase 2: Bahia).
Fonte: [`docs/sbpo/main.tex`](docs/sbpo/main.tex)

---

## Pipeline de Simulação (Kedro)

```bash
cd simulation/
pip install -e . -r requirements.txt

kedro run                                      # pipeline completo (padrão: PB)
kedro run --params "data_ingestion.states=['BA']"  # Fase 2 — Bahia
kedro run --pipeline statistical_validation    # só validação estatística
kedro viz                                      # DAG visual
```

### Pipelines (7)

| Pipeline | Entrada | Saída |
|---|---|---|
| `data_ingestion` | Parquet por UF | `scenarios.parquet` |
| `demand_profiling` | Cenários | Perfis POD (ADI × CV²) |
| `demand_forecasting` | Cenários | `forecasters.pkl`, métricas |
| `inventory_simulation` | Cenários + params | `kpis.parquet` |
| `policy_selection` | KPIs + perfis | Rótulos PSE, features |
| `statistical_validation` | KPIs | Wilcoxon, Friedman-Nemenyi |
| `reporting` | KPIs + testes | Figuras PDF, tabelas LaTeX |

### Parâmetros configuráveis

| Arquivo | Controla |
|---|---|
| `data_ingestion.yml` | Estados, produtos, período |
| `data_cleaning.yml` | Filtros, thresholds de outlier |
| `demand_profiling.yml` | Limiares ADI, CV², burstiness |
| `forecasting.yml` | Modelos de previsão |
| `simulation.yml` | Custos, lead time, hiperparâmetros |
| `policy_selection.yml` | Critério de rótulo, random seed |
| `reporting.yml` | Formato das figuras e tabelas |

---

## 12 Políticas Avaliadas

| Categoria | Políticas |
|---|---|
| Clássicas | EOQ, (s,S), Jornaleiro |
| Meta-heurísticas | GA, SA, PSO, DE |
| Aprendizado por Reforço | DQN, PPO, SARSA |
| Híbridas (AIPE) | GA-DQN, GA-PPO |

## 5 KPIs

| KPI | Descrição |
|---|---|
| **TIC** | Custo total de inventário (holding + ruptura + pedido) |
| **NS** | Nível de serviço — proporção da demanda atendida |
| **TR** | Taxa de ruptura — ciclos com falta / total |
| **BE** | Efeito Bullwhip — Var(pedidos) / Var(demanda) |
| **FP** | Frequência de pedidos — pedidos / ciclos |

---

## Requisitos

```
python >= 3.9
kedro >= 0.19
pandas, numpy, scikit-learn, xgboost
deap, scipy, scikit-posthocs
matplotlib
```

```bash
pip install -r simulation/requirements.txt
```
