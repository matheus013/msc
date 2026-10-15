# AIPE — projeto Python integrado

Este projeto concentra todos os experimentos, scripts, configurações e geração de resultados. O pacote interno `simulation` foi mantido para preservar imports e modelos serializados; não há um segundo projeto Python.

## Ambiente

O ambiente local já está em `.venv/`, com PyTorch CUDA instalado e dependências declaradas em `pyproject.toml`. No PowerShell:

```powershell
cd experiment
.\.venv\Scripts\Activate.ps1
python run.py device
```

Para criar um ambiente novo:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[test,documents]"
```

Mapas geográficos completos requerem o extra `maps`: `pip install -e ".[maps]"`. `requirements.txt` aponta para o mesmo projeto; as dependências não são duplicadas.

## Entrada única

Os exemplos partem de `experiment/`. Da raiz, usar `python experiment/run.py`.

```bash
python run.py --help
python run.py device
python run.py run --pipeline data_ingestion
python run.py run --params "data_ingestion.states=['BA']"
python run.py run --pipeline reporting
python run.py run --pipeline benchmark_m5 --env m5
python run.py run --pipeline benchmark_bot --env bot
```

Preparação de vendas usa um único conversor DuckDB, com filtro por UF, agregação entre arquivos e escrita sem materializar a base inteira em pandas:

```bash
python run.py prepare-sales --states PB
python run.py prepare-sales --states BA PB --output data/legacy/vendas_ba_pb.csv
python run.py prepare-pb
python run.py check-pb
python run.py pb --config conf/pb/config.yaml --out data/legacy/outputs/pb_new
python run.py plot-pb
```

O conversor não sobrescreve uma saída existente sem `--overwrite`. `pb` usa os mesmos ambientes, previsores e políticas do Kedro. Sua configuração de custos e recorte permanece em `conf/pb/config.yaml`; novas execuções usam os algoritmos corrigidos e não são reprodução literal da implementação antiga.

Outros comandos: `sample-data`, `prepare-json`, `analyze-pdf`, `maps`, `maps-full`, `maps-national` e `segment-plot`. `analyze-pdf` usa por padrão o PDF da qualificação. Os scripts são acessados pelo comando correspondente para evitar diferenças de diretório de trabalho.

## Dados e resultados

- `../data/source/`: fontes brutas internas, incluindo vendas por UF.
- `../data/m5_walmart/`: fonte bruta externa M5.
- `data/02_intermediate/` até `data/08_reporting/`: artefatos do Kedro.
- `data/legacy/`: dados preparados e resultados preservados dos experimentos anteriores.
- `../docs/qualification_presentation/`: versão atual dos resultados apresentados.

Os catálogos `base`, `bot` e `m5` continuam isolando seus resultados. As fontes originais são compartilhadas e permanecem fora dos diretórios de saída.

## GPU e mudanças de implementação

`auto` prioriza CUDA, depois MPS e CPU. Meta-heurísticas vetorizadas, DQN, PPO e LSTM PyTorch usam a GPU disponível. XGBoost usa `hist` com `device=cuda` quando CUDA está disponível; MPS usa CPU nesse componente. SARSA tabular, scikit-learn, tratamento de dados e figuras continuam em CPU.

Nos pipelines Kedro, um processo GPU é usado por padrão para evitar que vários processos dupliquem o uso de VRAM. `simulation.gpu_workers` permite ajuste explícito; em CPU vale `simulation.n_workers`. O executor `research-production` tem configuração própria, descrita abaixo.

A LSTM atual usa PyTorch/autograd, com treinamento completo da rede. Para o backend histórico, configurar `forecasting.lstm.backend: numpy` (ou `FORECASTING.LSTM.backend` no experimento PB). Não comparar novas execuções como se fossem idênticas aos resultados da qualificação. GPU não garante ganho em todos os tamanhos de lote; para medir em CPU:

```powershell
$env:AIPE_DEVICE = "cpu"
python run.py device
Remove-Item Env:AIPE_DEVICE
```

A seleção segue as interfaces oficiais de [PyTorch](https://pytorch.org/get-started/locally/) e [XGBoost](https://xgboost.readthedocs.io/en/stable/gpu/index.html). A variável `AIPE_DEVICE` é compartilhada por todos os componentes compatíveis.

## Verificação

### Executor de pesquisa com retomada

`research-production` usa CPU, 8 processos independentes, uma thread de cálculo por processo e prioridade reduzida por padrão, para coexistir com o uso normal da máquina. O processo principal confirma cada treino em SQLite; a fila mantém no máximo duas tarefas pendentes por processo. Dispositivo, número de processos e prioridade ficam registrados no manifesto; uma mudança exige outro nome de rodada.

```powershell
python run.py research-production --run NOVA_RODADA --device cpu --workers 8 --priority below-normal
```

Esse comando inicia a rodada completa. A rodada `census_2026-10-14` permanece interrompida por solicitação do usuário. Os testes de desempenho ficam separados dos resultados de pesquisa; a comparação de CPU/GPU está em `../docs/supply_chain_analytics_article/DESEMPENHO_CPU_GPU.md`.

### Filas independentes após conferência de escopo

`scripts/analysis/run_scoped_production.py` mantém Bahia145, M5 de referência e Brasil nacional em rodadas independentes. Por solicitação do usuário em 15/10/2026, o Brasil foi cancelado e removido do launcher `scripts/operations/start_scoped_experiments.ps1`. A Bahia já concluída é pulada, e apenas o M5 continua, por padrão com 8 processos CPU e prioridade reduzida. As duas coortes de referência incluem DQN como controle além das seis políticas principais.

`--resume-execution-profile` permite uma migração explícita de processos/prioridade mantendo os checkpoints, desde que os hashes científicos, políticas, dados e regras de importação não mudem. O histórico operacional fica no manifesto; uma rodada com `cancelled_by_user.json` recusa inicialização. O identificador do M5 ainda contém `cpu16` por continuidade dos arquivos, mas o número de processos atual vem de `status.json` e do manifesto.

Os manifestos congelam os IDs, as origens, o critério de informação passada e a versão `ga_gated_dqn_consistent_v2`. Essa variante treina e avalia com o mesmo gatilho/piso do GA e calcula a mesma penalidade terminal nas demonstrações e no treino. Os resultados antigos do híbrido são preservados, mas não reutilizados na variante corrigida. Os demais checkpoints só entram após auditoria do código científico, da configuração e dos dados. Consulte `../docs/supply_chain_analytics_article/CONFERENCIA_ESCOPO_E_TEMPO_2026-10-15.md` para quantidades e estimativas.

### Cinco análises geradas

Os pipelines `reporting`, `profile_analysis`, `final_report`, `benchmark_final`, `benchmark_m5` e `benchmark_bot` incluem o nó `generate_experiment_diagnostics`: fronteira custo–serviço/Pareto, políticas por perfil, distribuição do fill rate, heatmap perfil–limiar e caracterização da demanda. Figuras PNG/SVG, tabelas e manifesto são gravados em `data/08_reporting/experiment_diagnostics/`, ou no diretório equivalente do ambiente `m5`/`bot`. O portfólio vem dos resultados, sem fixar 12 políticas.

Para executar somente essas análises sobre os artefatos existentes, sem treinar políticas:

```powershell
python run.py run --pipelines experiment_diagnostics
python run.py run --pipelines experiment_diagnostics --env m5
```

A configuração `reporting.diagnostics` controla ativação, base, meta principal (70%), limiares (70%, 80%, 85%, 90%, 95%) e pasta de saída. Os KPIs planos do Kedro permitem uma análise retrospectiva com exclusão de loja, identificada como tal. No executor `research-production`, as mesmas figuras são geradas após a etapa Bahia/M5 com seleção na validação das outras lojas e avaliação no teste da loja excluída; entram somente casos completos de todas as políticas e sementes. Séries sem demanda no teste conservam fill rate indefinido.

Também é possível gerar as figuras a partir de uma rodada interrompida, sem retomá-la, ou da reanálise histórica preservada:

```powershell
python run.py research-visualizations --run census_2026-10-14
python run.py historical-visualizations
```

Nas rodadas novas, as distribuições de demanda usam os painéis elegíveis, enquanto a contagem de prefixos sem demanda vem de todas as candidatas no manifesto. Na reanálise histórica, ambas descrevem apenas a coorte preservada. CV² usa demandas positivas, distinguindo-se do CV² de todos os ciclos usado pelos perfis legados.

### Testes automatizados

```bash
python -m pytest tests -q
```

Os testes usam dados sintéticos e diretórios temporários, verificam a equivalência numérica entre simuladores CPU/GPU e exercitam ingestão, otimização, agentes RL e previsão.
