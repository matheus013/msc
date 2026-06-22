# AIPE - Adaptive Inventory Policy Engine

Repositório da dissertação de mestrado e do artigo SBPO 2026 sobre seleção contextual de políticas de reposição em redes varejistas com demanda intermitente.

O projeto combina uma proposta em LaTeX, um artigo científico e um pipeline Kedro para simulação, validação estatística e geração de artefatos experimentais.

## Visão geral

O AIPE, Adaptive Inventory Policy Engine, é um framework metodológico para recomendar políticas de reposição de estoque de acordo com o perfil operacional de cada série loja-produto.

A dissertação avalia políticas clássicas, meta-heurísticas, agentes de aprendizado por reforço e arquiteturas híbridas GA-RL em dados reais de uma rede varejista brasileira.

## Estrutura do repositório

```text
sbpo/
|-- docs/
|   |-- master_proposal/       Proposta de dissertação em LaTeX
|   |   |-- tcc.tex            Documento raiz
|   |   |-- capitulos/         Capítulos da proposta
|   |   |-- figures/           Figuras em LaTeX/TikZ
|   |   |-- img/               Figuras em PDF
|   |   |-- build/             PDF e artefatos de compilação
|   |   `-- README.md
|   |
|   |-- sbpo/                  Artigo SBPO 2026
|   `-- references/            PDFs e materiais de referência
|
`-- simulation/                Projeto Kedro
    |-- conf/base/parameters/  Parâmetros em YAML
    |-- src/simulation/        Código-fonte
    |-- data/                  Dados e artefatos experimentais
    `-- README.md
```

## Proposta de dissertação

Título:

**Um Framework Adaptativo para Seleção de Políticas de Reposição em Regimes Operacionais Heterogêneos**

Capítulos atuais:

| Capítulo | Título |
|---|---|
| 1 | Introdução |
| 2 | Referencial Teórico |
| 3 | Trabalhos Relacionados |
| 4 | Metodologia |
| 5 | Resultados Preliminares |
| 6 | Considerações Finais e Próximas Etapas |

Compilação:

```bash
cd docs/master_proposal
latexmk tcc.tex
```

O PDF final é gerado em:

```text
docs/master_proposal/build/tcc.pdf
```

A configuração de compilação fica em `docs/master_proposal/.latexmkrc` e direciona os artefatos para `build/`.

## Resultados consolidados na proposta

| Fase | Escopo | Leitura principal |
|---|---|---|
| Fase 1 | 15 lojas da Paraíba, produto 48130 | GA-DQN apresentou redução de CTI de até 48% em relação a políticas de referência no recorte preliminar. |
| Fase 2 | 145 séries da Bahia | A seleção por perfil reduziu o CTI médio em 6,2% frente à política única global, mantendo NS acima do limiar mínimo adotado. |

Os resultados são interpretados como evidência de contextualidade na escolha de políticas, não como dominância universal de uma única política.

## Pipeline de simulação

Instalação:

```bash
cd simulation
pip install -e . -r requirements.txt
```

Execução:

```bash
kedro run
kedro run --params "data_ingestion.states=['PB']"
kedro run --params "data_ingestion.states=['BA']"
kedro run --pipeline statistical_validation
kedro run --pipeline reporting
kedro viz
```

Pipelines principais:

| Pipeline | Função |
|---|---|
| `data_ingestion` | Carrega, filtra e prepara séries por estado, produto e loja. |
| `demand_profiling` | Calcula características operacionais e classifica perfis de demanda. |
| `demand_forecasting` | Treina modelos auxiliares de previsão. |
| `inventory_simulation` | Simula políticas de reposição sob um ambiente comum. |
| `policy_selection` | Gera rótulos de política dominante para o PSE. |
| `statistical_validation` | Aplica testes estatísticos e tamanhos de efeito. |
| `reporting` | Gera tabelas LaTeX, figuras e relatórios auxiliares. |

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

## Observações

- Os dados transacionais são proprietários e não devem ser versionados.
- Artefatos de build da dissertação devem permanecer em `docs/master_proposal/build/`.
- O PDF de entrega principal fica em `docs/master_proposal/build/tcc.pdf`.
- Use `git status` antes de alterar resultados, tabelas ou artefatos experimentais.
