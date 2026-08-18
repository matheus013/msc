# Ajustes de infraestrutura — 2026-08-18

Registro dos ajustes feitos na sessão de 2026-08-18: pull, download da base
Walmart M5, reexecução dos experimentos e correção da infraestrutura de
dados que apareceu pelo caminho. Cada entrada segue descrição, motivo,
objetivo e impacto. Ordem cronológica.

---

## 1. Ambiente Python nesta máquina Windows

**Descrição:** Instalado Python 3.12.10 via winget; venv criado em
`C:\aipe-venv\py312`, **fora** da pasta do projeto (que fica dentro do
OneDrive).

**Motivo:** A máquina não tinha nenhum Python utilizável (só o stub da
Microsoft Store). Um venv dentro de `simulation/.venv312/` quebrou a
instalação do torch: o caminho completo (`.../OneDrive/Área de
Trabalho/Documentos/sbpo/simulation/.venv312/Lib/site-packages/torch-.../
licenses/third_party/kineto/.../duktape-1.5.2`) passa do limite de 260
caracteres do Windows.

**Objetivo:** Ter um ambiente funcional para rodar o pipeline Kedro.

**Impacto:** Nenhum no código do projeto. Efeito colateral: qualquer
comando `kedro` precisa ser invocado com o Python desse venv externo, não
com um `.venv` dentro do repositório.

---

## 2. Download da base Walmart M5

**Descrição:** Baixado `m5.zip` (47,9MB) de
`github.com/Nixtla/m5-forecasts/datasets/m5.zip` e extraído para
`simulation/data/01_raw/m5_walmart/`.

**Motivo:** Pedido explícito do usuário — rodar o benchmark externo contra
a base pública usada por Zabraoui et al. (2025).

**Objetivo:** Disponibilizar `sales_train_evaluation.csv`, `calendar.csv`
e `sell_prices.csv` para `m5_loader.py`.

**Impacto:** Nenhum no código. Dados ficam em `data/01_raw/` (gitignored).

---

## 3. Incidente: sobrescrita de `sales_raw.parquet` da Bahia

**Descrição:** Um `kedro run --env m5` (smoke test) sobrescreveu
`data/02_intermediate/sales_raw.parquet` — o único arquivo em disco dos
dados transacionais proprietários da Bahia (62MB) — com dados do M5 (5
séries, 7,9KB).

**Motivo:** `load_raw_sales()` sempre grava em `sales_raw`. O ambiente `m5`
não redefinia esse dataset no catálogo, então `--env m5` caía de volta no
caminho **base**, compartilhado com a Bahia. Causa raiz adicional: Kedro
não carrega `conf/local/` quando `--env <nome>` é passado explicitamente
(`default_run_env` é sobrescrito), então nem o catálogo local salvava.

**Objetivo (da correção):** Nunca mais um ambiente novo compartilhar
dataset com a base oficial da Bahia.

**Impacto:** Recuperado regenerando `sales_raw` do zero a partir da fonte
particionada original (`data/source/vendas/uf=*/`, 1,3GB, 27 estados, que
por sorte estava intacta fora de `simulation/` nesta máquina) — copiada
para `simulation/data/01_raw/vendas/` e reprocessada via
`kedro run --pipeline data_ingestion`. Resultado bateu exatamente com o
documentado (145 séries, 38 ciclos), confirmando integridade.
`conf/m5/catalog.yml` e (depois) `conf/bot/catalog.yml` passaram a
redefinir `vendas_partitioned`, `sales_raw`, `sales_filtered`,
`sales_cleaned`, `scenarios`, `scenarios_meta` isolados em `data/*/<env>/`.

---

## 4. Logging: `conf/logging.yml` na raiz de `conf/`

**Descrição:** Criado `simulation/conf/logging.yml` (nível DEBUG em
console + arquivo `simulation/logs/simulation.log`).

**Motivo:** Pedido do usuário ("máximo de logs"). Kedro 1.x só lê logging
de `conf/logging.yml` — não do padrão de ambiente (`conf/base/logging.yml`
existia e foi editado primeiro, mas nunca era carregado; a busca é
hardcoded via `find_config_file("conf/logging")`, não passa pelo
OmegaConfigLoader com padrões de ambiente).

**Objetivo:** Rastreabilidade completa das execuções (nível de política,
série, erro).

**Impacto:** Logs muito mais verbosos (inclui dump de configs/DataFrames
em cada nó). Trade-off aceito explicitamente pelo usuário. Arquivo
`conf/base/logging.yml` antigo ficou órfão (inofensivo, não é mais lido).

---

## 5. Interface local de acompanhamento de logs

**Descrição:** `log_viewer.py` — servidor HTTP local (stdlib apenas, sem
dependência nova), `http://127.0.0.1:8765/`, lista e exibe com auto-refresh
os `.log` de `simulation/logs/` e do scratchpad da sessão. Cores por
nível/logger/política/valores; auto-reload da aba quando o servidor
reinicia (poll de build id); clique manual desliga "seguir mais recente"
(bug corrigido — antes o auto-refresh sempre puxava de volta pro topo da
lista).

**Motivo:** Pedido do usuário para acompanhar as execuções sem ficar
pedindo trechos de log manualmente.

**Objetivo:** Visibilidade em tempo real das runs longas.

**Impacto:** Só arquivo de sessão (`scratchpad/log_viewer.py`), não faz
parte do repositório do projeto.

---

## 6. DuckDB para carga do M5 (`m5_loader.py`)

**Descrição:** Reescrita a leitura do `sales_train_evaluation.csv`
(30.490 séries × ~1.941 dias) para usar DuckDB: agrega os dias em ciclos
via soma aritmética entre colunas da mesma linha (não é um agregado entre
linhas, então não precisa de `GROUP BY`) e só materializa em pandas o
resultado já agregado (≤30.490 linhas), em vez de ler o CSV largo inteiro
com pandas.

**Motivo:** Pedido do usuário ("quero duckdb no m5 também"), depois de
identificar que `pd.read_csv` inteiro + filtro depois é o mesmo padrão que
quase causou OOM na base interna completa.

**Objetivo:** Reduzir custo de memória/tempo da ingestão do M5,
consistente com a mesma estratégia usada em "bot".

**Impacto:** `load_m5_as_internal()` ~4s para 200 séries/38 ciclos
(medido). Adiciona `duckdb` a `requirements.txt`. Nenhuma mudança de
schema de saída.

---

## 7. Bug de merge de parâmetros entre ambientes (`settings.py`)

**Descrição:** `CONFIG_LOADER_ARGS["merge_strategy"] = {"parameters": "soft"}`
em `simulation/settings.py`.

**Motivo:** Kedro 1.5 usa `merge_strategy="destructive"` por padrão — um
`conf/<env>/parameters.yml` que redefine parcialmente uma chave de topo
(ex.: só `data_ingestion.states`) **substitui** a chave inteira do
`conf/base`, não mescla. Descoberto porque `cv_threshold` virava `0.0`
(default do código, não o `1.5` do base) no ambiente `m5` — o filtro de
intermitência documentado como importante em `REIMPLEMENTACAO_SOTA.md`
vinha sendo **desativado silenciosamente** em toda execução do M5 até
aqui.

**Objetivo:** Ambientes (`m5`, `bot`) devem herdar tudo do `conf/base` e só
sobrescrever o que declaram explicitamente.

**Impacto:** Corrige o comportamento de TODOS os ambientes custom
(`m5`, `bot`), não só o caso que motivou a descoberta. Nenhum efeito no
ambiente `base`/`local` (não há merge entre camadas ali).

---

## 8. Isolamento completo de catálogo por ambiente

**Descrição:** `conf/m5/catalog.yml` e `conf/bot/catalog.yml` passaram a
redefinir também `scaled_params`, `demand_features`, `demand_profiles`,
`kpis_classical`, `kpis_sota_classical`, `kpis_zabraoui`,
`kpis_metaheuristic`, `kpis_rl`, `kpis_proposed` (além dos já isolados
`sales_raw/filtered/cleaned`, `scenarios`, `scenarios_meta`, `kpis`,
saídas de `08_reporting/`).

**Motivo — Incidente #2:** Mesmo com o incidente #1 corrigido, um smoke
test do `m5` rodando **em paralelo** com uma produção da Bahia sobrescreveu
`scaled_params.pkl` e três `kpis_*.parquet` sem prefixo no meio da
execução. A run de produção da Bahia (5322s, "concluída com sucesso",
exit 0) saiu com `sota_classical`/`zabraoui` **vazios** e
`classical`/`metaheuristic`/`proposed` com só 5 séries em vez de 145 —
silencioso, só descoberto conferindo os parquets individualmente (o `kpis`
agregado tinha 480 linhas/12 políticas em vez de ~2610/18).

**Objetivo:** Nenhum dataset intermediário tocado pelas pipelines
`benchmark_m5`/`benchmark_bot` (`di+dp+inv+fr`) deve compartilhar caminho
com a base oficial da Bahia — nem entrada/saída final, nem intermediários.

**Impacto:** A run de produção da Bahia corrompida por este incidente foi
descartada e reexecutada do zero (`prod_benchmark_final_v2.log`). Runs de
`m5`/`bot` agora podem, em princípio, rodar em paralelo com a Bahia sem
risco de contaminação cruzada — mas a prática desta sessão foi manter
execuções sequenciais por precaução.

---

## 9. DuckDB para a base interna completa ("bot")

**Descrição:** Novo módulo `duckdb_loader.py` — pré-passo DuckDB streaming
que calcula, sem nunca materializar as 55,3M linhas brutas em pandas, a
lista de produtos que sobrevivem aos filtros **globais** (entre todos os
27 estados) de produto ativo + CV mínimo. Essa lista filtra cada partição
de estado individualmente em `load_raw_sales()` (ainda em pandas, um
estado por vez) antes do `concat` — pico de memória limitado a uma
partição, não à soma das 27.

**Motivo:** `states=["all"]` sem nenhum recorte tentava concatenar 55,3M
linhas × 19 colunas com pandas antes de filtrar — mediu memória na casa de
dezenas de GB (quase esgotou os 32GB da máquina numa tentativa: 1,4GB
livres). Os filtros de produto ativo/CV são calculados globalmente por
`item_id`, então não dá pra decidir isso partição por partição sem mudar o
resultado — daí o pré-passo global antes do recorte por partição.

**Objetivo:** Tornar viável rodar a base interna completa (27 estados)
nesta máquina.

**Impacto:** Ingestão completa (55,3M → 27,1M linhas filtradas → 16,6M
após dedup) em ~70s de pico de memória (~8GB), contra quase-OOM antes.
`filter_by_parameters`/`clean_sales_data` continuam rodando sem nenhuma
mudança de código sobre os dados já reduzidos (idempotentes — confirmado
na prática: 0 linhas a mais removidas nas re-checagens).

---

## 10. Ambiente Kedro `bot` (base interna completa)

**Descrição:** Novos `conf/bot/parameters.yml`, `conf/bot/catalog.yml`,
`conf/bot/parameters_final_report.yml` e pipeline `benchmark_bot` em
`pipeline_registry.py` (`kedro run --pipeline benchmark_bot --env bot`).

**Motivo:** Rodar o portfólio de políticas sobre a base interna completa
(todos os 27 estados), em complemento ao recorte oficial da Bahia
(Experimento 2) e ao benchmark externo M5 — apelidada **"bot"** nesta
sessão para ter nome fácil ao lado de "m5"/Walmart.

**Objetivo:** Escala maior de validação, sem substituir o experimento
oficial da Bahia.

**Impacto:** Terceira base de comparação disponível. Não altera
`benchmark_final` (Bahia) nem `benchmark_m5`.

---

## 11. `max_stores` estourava memória mesmo configurado

**Descrição:** Movido o corte de `max_stores` (top-N séries por atividade,
por estado) para **antes** do zero-fill e da extração de perfil da
revendedora em `build_demand_scenarios()`, em vez de depois (onde já
existia, tarde demais).

**Motivo:** Com `states=["all"]`, o zero-fill expandia TODAS as ~10,4
milhões de séries candidatas (271.460 revendedoras × ~880 produtos) × 38
ciclos = ~400 milhões de linhas **antes** de `max_stores` cortar qualquer
coisa — `numpy._core._exceptions._ArrayMemoryError` alocando 2,96GB num
array de agrupamento. Mesmo corrigindo só o zero-fill, a extração de
perfil (`_mode_safe`, função Python por grupo) ainda rodava sobre as 271
mil revendedoras inteiras: ~10 minutos só nessa etapa.

**Objetivo:** `max_stores` (e qualquer filtro de seleção) precisa reduzir
o volume de dados **antes** das etapas caras, não depois.

**Impacto:** Pipeline de ingestão do "bot" caiu de "trava depois de 10+
min" para completar em ~4 min, com `max_stores=5000` dando 72.478 séries
pré-filtro → 4.787 finais.

---

## 12. Filtro de qualidade (`min_positive_cycles`) adiantado — o teto real

**Descrição:** O mesmo filtro `min_positive_cycles` que já existia (rodando
depois do zero-fill) passou a rodar também **antes**, sobre `df_agg`
pré-zero-fill — resultado matematicamente idêntico (zero-fill só acrescenta
linhas com `demand=0`, que nunca contam como positivas; não cria série
nova), só que sobre uma tabela ~15-20x menor. Roda sempre, não só quando
`max_stores` está setado. `max_stores` virou um teto **opcional** aplicado
sobre o pool já filtrado por qualidade, ainda estratificado por segmento.

**Motivo:** Achado empírico comparando `max_stores=5000` (72.478
candidatos pré-filtro) contra `max_stores=50000` (614.848 candidatos): os
dois convergiram para praticamente o mesmo total final (4.787 vs 4.869
séries). Ou seja, o teto real de cobertura desta base é o filtro de
qualidade (quantas revendedoras têm demanda persistente o bastante em
produtos válidos), não memória nem o valor de `max_stores` — rodar com
mais candidatos só gastava mais tempo processando série que ia ser
descartada de qualquer forma.

**Objetivo:** Pedido do usuário de cobertura **máxima** que a máquina
aguenta, sem desperdiçar tempo em candidatos fadados a serem descartados
pelo filtro de qualidade de qualquer forma. `max_stores` pode voltar a
`null` (sem teto artificial) com segurança, porque agora o volume que
chega ao zero-fill já é só o pool genuinamente válido.

**Impacto:** Validado sem regressão contra a Bahia (145 séries, mesma
distribuição de segmentos, idêntico ao original). Para "bot", elimina a
necessidade de adivinhar um `max_stores` — o resultado com `max_stores:
null` é a cobertura máxima real que a base suporta, calculada de forma
segura (nunca materializa o zero-fill dos candidatos descartados).

**Resultado final validado** (`max_stores: null`, cobertura máxima real):
**4.869 séries**, 38 ciclos, pipeline de ingestão completo em ~3,5 min,
memória totalmente recuperada ao final (23,8GB livres de 32GB). Grupos:
Lumpy 4.807, Erratic 46, Smooth 16. Segmentos: Platina 2.131, Ouro 916,
Rubi 762, Diamante GB 464, Esmeralda GB 463, Prata 131, Bronze 2 — todos
os 7 representados. ~34x mais séries que o recorte oficial da Bahia (145).

---

## 13. Amostragem estratificada por segmento

**Descrição:** A seleção de `max_stores` (quando setado) passou a
distribuir a cota igualmente entre os segmentos presentes em cada estado
(`sorted(segmentos_w)`, cota = `max_stores // n_segmentos`), em vez de
pegar top-N global por atividade.

**Motivo:** Pedido explícito do usuário — top-N por atividade tende a
devolver só as lojas de maior giro, tipicamente concentradas num único
segmento (ex. Platina), perdendo representatividade dos demais
(Bronze/Prata/Ouro/Rubi/Esmeralda GB/Diamante GB).

**Objetivo:** Garantir amostra de todos os segmentos disponíveis por
estado, não só os mais ativos.

**Impacto:** Confirmado na prática: com `max_stores=5000`, os cenários
finais cobriram os 7 segmentos (Platina, Ouro, Rubi, Diamante GB, Esmeralda
GB, Prata, Bronze), não só o topo. Com o filtro de qualidade adiantado
(#12) e `max_stores: null`, a estratificação por segmento deixa de ser
necessária para a cobertura máxima (não há mais corte artificial a
estratificar) — o código permanece disponível para quando `max_stores` for
setado explicitamente como teto.

---

## 14. `.gitignore`: `simulation/.tmp/`

**Descrição:** Adicionada entrada `simulation/.tmp/` ao `.gitignore`.

**Motivo:** DuckDB usa esse diretório para *spill* em disco quando a
memória aperta (visto no teste que quase deu OOM: acumulou 14GB em
arquivos `duckdb_temp_storage_*.tmp`).

**Objetivo:** Não versionar artefatos temporários de dezenas de GB.

**Impacto:** Diretório removido (14GB liberados) e agora ignorado pelo
git.

---

## 15. M5 também em cobertura máxima (`max_series: null`)

**Descrição:** `conf/m5/parameters.yml`: `data_ingestion.m5.max_series`
de `6000` para `null`.

**Motivo:** Mesma diretriz do usuário aplicada ao "bot" (#12) — cobertura
máxima que a base oferece. Diferente de "bot", a agregação do M5 já é
feita via DuckDB (#6) sem o padrão de zero-fill que causou o
`ArrayMemoryError` na base interna, então não há risco de crash
equivalente em usar todas as 30.490 séries.

**Objetivo:** Maximizar cobertura do benchmark externo M5, mesma filosofia
das outras duas bases.

**Impacto:** Testado e seguro: 30.490 → 6.165 séries (após
`active_product_window`/`cv_threshold` em `filter_by_parameters`) → 1.381
finais (após `min_positive_cycles`, filtro adiantado #12). Pipeline de
ingestão completo em ~8 segundos. Todas em FOODS (única categoria que
passa no `cv_threshold >= 1.5` nesta base, consistente com o já registrado
em `REIMPLEMENTACAO_SOTA.md`).

---

## 16. M5 sem nenhum filtro de qualidade (pedido explícito)

**Descrição:** `conf/m5/parameters.yml`: `min_positive_cycles: 0`,
`cv_threshold: 0`, `active_product_window: 0` (zeram os três filtros
herdados do `conf/base`), além de `max_cycles: null` (era 38 — usa todo o
horizonte do M5, ~93 ciclos em vez de truncar pra bater com o horizonte
interno).

**Motivo:** Pedido explícito do usuário ("se possível m5 sem cortar
nada").

**Objetivo:** M5 sem nenhum corte artificial — as 30.490 séries, todos os
~93 ciclos disponíveis.

**Impacto — ressalva metodológica registrada, não decisão minha:**
`cv_threshold >= 1.5` existia especificamente para tornar o M5 comparável
em regime de demanda ao recorte interno (`REIMPLEMENTACAO_SOTA.md`,
seções 4–5: M5 diário tem CV² tipicamente < 1, o recorte interno por ciclo
comercial está em 3,8–4,1). Sem esse filtro, os dois deixam de estar no
mesmo regime — a comparação externa deixa de ser "mesma classe de
demanda", vira "toda a base pública, sem qualificação de regime". Testado
e seguro: 30.490 → 30.490 séries (nenhuma cortada), 93 ciclos, pipeline de
ingestão em ~15s.

---

## 17. Bug: `segmento` colapsado por loja quebrava a categoria no M5

**Descrição:** `build_demand_scenarios()` extraía `segmento` (e os demais
campos de perfil) com `groupby(["warehouse","store_id"])` + moda,
assumindo que é constante por loja. Corrigido: `segmento` agora extraído
separadamente dos demais campos, com `groupby(["warehouse","store_id",
"item_id"])` (por série, não por loja) — mesma correção aplicada no bloco
de estratificação por `max_stores`.

**Motivo:** Na base interna, `segmento` é constante por `revendedor_cod`
(faixa de faturamento da revendedora — ver #`segmento` é ordinal acima).
No M5 (`m5_loader.py`), o campo equivalente vem de `cat_id`
(FOODS/HOUSEHOLD/HOBBIES) — categoria do PRODUTO, não da loja; uma mesma
loja M5 vende as três categorias ao mesmo tempo. Agregar por moda a nível
de loja fazia TODAS as séries de uma loja herdarem só a categoria mais
frequente. Descoberto ao rodar M5 sem cortes (#16): as 30.490 séries
saíram rotuladas 100% "FOODS", quando a base tem as três categorias.

**Objetivo:** Rótulo de categoria/segmento correto por série, nas duas
bases.

**Impacto:** Não afeta os KPIs numéricos da simulação (TIC/NS/TR/BE/FP) —
só o rótulo de categoria usado em relatórios/estratificação. Validado sem
regressão contra a Bahia (145 séries, mesma distribuição de segmentos
Platina/Ouro/Rubi/Diamante GB/Esmeralda GB/Prata, idêntica à anterior).
Para M5, corrige a distribuição de `FOODS: 30490` (100%, errado) para
`FOODS: 14370, HOUSEHOLD: 10470, HOBBIES: 5650` (correto).

---

## 18. Orçamento de RL reduzido só no M5 (dados intactos)

**Descrição:** `conf/m5/parameters.yml`: `simulation.dqn.episodes`,
`ppo.episodes`, `sarsa.episodes` de 500 → 50; `hybrid.rl_episodes` de 200
→ 30. `ga_generations` do híbrido e os demais parâmetros de
metaheurísticas (GA/SA/PSO/DE) ficam no padrão (já rápidos, vetorizados —
não são o gargalo).

**Motivo:** Com as 30.490 séries do M5 intactas (#16) em orçamento
completo de RL, a extrapolação do tempo medido na Bahia (145 séries →
88,7 min) dava ~13 dias corridos só para o M5 — perguntado ao usuário, que
optou por manter a cobertura de dados (nenhuma série cortada) e reduzir
o orçamento de treino em vez disso.

**Objetivo:** Tempo de execução viável sem sacrificar nenhuma série do M5.

**Impacto:** Reduz o custo do treino de RL/híbridas em 10x (RL) e ~6,7x
(híbridas) por série — tempo total esperado bem menor que 13 dias, embora
ainda não medido nesta sessão (run em andamento no momento deste
registro). Políticas RL/híbridas do M5 ficam com orçamento de treino menor
que Bahia/"bot" — relevante para qualquer comparação futura de qualidade
de solução entre as três bases (não é mais "mesmo orçamento, bases
diferentes", é "orçamento ajustado por base").

---

## 19. Paralelização entre séries (ProcessPoolExecutor)

**Descrição:** Os 6 nós de política (`run_classical_policies`,
`run_sota_classical_policies`, `run_zabraoui_policies`,
`run_metaheuristic_policies`, `run_rl_policies`,
`run_proposed_architecture`) reescritos para despachar cada série a um
`ProcessPoolExecutor` via uma infraestrutura genérica compartilhada
(`_run_parallel_policies` + workers de módulo `_worker_generic`/
`_worker_proposed`). Novo parâmetro `simulation.n_workers` (default:
núcleos disponíveis − 2).

**Motivo:** Cada série é 100% independente das demais (mesmo cfg, mesma
demanda, sem estado compartilhado) — mas o loop era sequencial, 1 núcleo
de 16 usados. "bot" (4.869 séries) e M5 (30.490 séries) em orçamento
completo davam ~48h e ~65h respectivamente — muito acima do teto de 12h
combinado com o usuário.

**Objetivo:** Usar os núcleos ociosos sem sacrificar orçamento de busca
nem cobertura de dados.

**Impacto:** Processos (não threads) — evita o conflito de libomp entre
torch/xgboost/sklearn documentado em REIMPLEMENTACAO_SOTA.md, que é
específico de threads dividindo o mesmo processo. Sementes são locais por
série (`cfg[...]["seed"]`, `params["random_seed"]`), sem RNG global
acumulado entre séries mesmo no código sequencial original — validado
empiricamente 2x (8 e 40 séries): resultado numérico **idêntico**
(0 diferenças) entre sequencial e paralelo. Ganho de velocidade no teste
pequeno foi modesto (1,6x com 10 workers, 40 séries, orçamento de teste
bem reduzido) porque o overhead fixo por processo (import de
torch/xgboost/sklearn no spawn) domina tarefas curtas; com o orçamento de
produção real (muito mais pesado por série) o ganho esperado fica bem
mais perto do número de workers. `n_workers=7` para bot e M5 (não os 14
disponíveis) porque os dois rodam ao mesmo tempo nesta máquina de 16
núcleos, com a Bahia (ainda sequencial) também ativa.

---

## 20. Meta-heurísticas reduzidas no M5 (viraram o novo gargalo)

**Descrição:** `conf/m5/parameters.yml`: `ga.generations` 50→10,
`sa.max_iter` 500→100, `pso.iterations` 80→16, `de.max_iter` 100→20 (~5x).

**Motivo:** Depois de medir os tempos reais por família na Bahia (v2:
classical 6,4s, metaheurística 8,12min, híbrida 23,34min, RL ~53,5min/145
séries), a projeção para M5 mostrou que meta-heurísticas — sem nenhum
corte, ao contrário do RL (#18) — passariam a dominar o tempo total
(28,5h das 65,5h estimadas), porque não são vetorizadas o bastante para
serem irrelevantes numa base de 30.490 séries.

**Objetivo:** Combinado com a paralelização (#19), caber no teto de 12h
sem cortar nenhuma série do M5.

**Impacto:** Estimativa combinada (cortes + 7 workers): M5 de ~65,5h
sequencial-sem-corte para ~6h; "bot" (sem corte de meta-heurística, só
paralelização) de ~48,3h para ~6,9h. Ambas dentro do teto de 12h. Medição
real pendente (runs em andamento no momento deste registro).

---

## 21. Fitness com termos de risco (TR, BE) — extensão sobre Zabraoui/proposta

**Descrição:** `core/inventory_env_torch.py`: nova função `_risk_terms`,
somada a `constrained_cost` (Eq. 4.2) E `zabraoui_fitness_cost` (Eq. 3),
usada por GA/SA/PSO/DE e por extensão GA-DQN/GA-PPO (todos herdam
`_ThresholdOptimizer.evaluate()`):

```
fitness += tr_weight * TR * (1 - NS) * clamp(TIC, min=1)   # risco de ruptura
fitness += be_weight * max(BE - 1, 0) * clamp(TIC, min=1)  # compra excessiva
```

Novos parâmetros `simulation.ga.tr_weight`/`be_weight` (default 1.0 cada).

**Motivo:** Pedido do usuário — TIC só registra o custo de ruptura JÁ
REALIZADO na trajetória simulada; TR (fração de ciclos sem produto pro
cliente — "TR é quando falta produto pro cliente") e BE (amplificação do
pedido sobre a demanda) capturam fragilidade que TIC sozinho não pune, e
não influenciavam a busca de nenhuma política antes desta mudança (achado
do item anterior desta conversa: TR/BE eram calculadas e testadas
estatisticamente, mas nunca otimizadas).

**Objetivo:** TR entra como probabilidade de gerar custo (ponderada por
NS — risco composto quando serviço já ruim E ruptura frequente
coincidem); BE entra como punição especificamente pela AMPLIFICAÇÃO acima
da variabilidade da própria demanda (BE≤1 não é punido — validado contra
a literatura: BE=1,0 é o zero natural da métrica, "sem amplificação",
ver fontes abaixo).

**Impacto:** Validado sem regressão de execução (0 erros, mesma
consistência sequencial=paralelo) mas **muda os ótimos encontrados** —
efeito esperado e desejado, não uma regressão a corrigir. Os resultados
de "bot"/M5 em andamento no momento deste registro já usam a fitness
nova; a Bahia (v2, já em RL quando a mudança foi aplicada) usa a fitness
ANTIGA só nas etapas que já haviam rodado (classical/metaheurística/
híbrida) — silenciosamente inconsistente entre etapas dessa run
especificamente, considerar re-executar a Bahia depois com a fitness nova
para manter as 3 bases comparáveis com o mesmo objetivo de otimização.
Validado contra a literatura (não é fórmula ad-hoc):
- BE = var(pedidos)/var(demanda), 1,0 = sem amplificação: [Bullwhip Effect — Finale Inventory](https://www.finaleinventory.com/inventory-planning-software/bullwhip-effect)
- Penalidade por probabilidade de ruptura em otimização com restrição de nível de serviço: [Optimization of a Stochastic Joint Replenishment Inventory System with Service Level Constraints](https://www.sciencedirect.com/science/article/abs/pii/S0305054822002349)
- GA + fitness penalizada para mitigar bullwhip: [Minimizing the bullwhip effect in a supply chain using genetic algorithms](https://www.tandfonline.com/doi/abs/10.1080/00207540500431347)

---

## 22. Bahia relançada como v3 (fitness nova, para consistência entre bases)

**Descrição:** `kedro run --pipeline benchmark_final --params
"simulation.n_workers=1"` → `prod_benchmark_final_v3.log`.

**Motivo:** A v2 da Bahia (item #21) tinha terminado com a fitness ANTIGA
nas etapas classical/metaheurística/híbrida (já haviam rodado quando a
mudança de #21 foi aplicada) — inconsistente com "bot"/M5, que já usavam a
fitness nova (TR+BE) do início ao fim.

**Objetivo:** As três bases otimizando pelo mesmo critério, para qualquer
comparação entre elas fazer sentido.

**Impacto:** Concluído com sucesso — `EXIT=0`,
`Pipeline execution completed successfully in 7076.1 sec` (~1h58min,
`n_workers=1` deliberado para não disputar núcleo com bot/M5 rodando em
paralelo). Conferido política por política (lição do incidente #2): `KPIs
agregados: 2610 linhas, 18 políticas, 145 séries | famílias:
{'metaheuristic': 580, 'classical': 435, 'sota_classical': 435, 'zabraoui':
435, 'rl': 435, 'hybrid': 290}` — idêntico em contagem à v2, agora com a
fitness nova em TODAS as etapas. Esta é a versão final/válida da Bahia
para a dissertação.

---

## 23. Correção de fidelidade ao Zabraoui: DQN, episódios de RL e operadores do GA

**Descrição:** O usuário apontou (e a auditoria confirmou, lendo o artigo
completo) que quatro coisas citadas como "adotadas de Zabraoui et al. (2025)"
nunca chegaram ao código:
1. `dqn.epsilon_start/end/decay`: citava Tab.4 (ε=0,2 fixo), mas o config
   tinha epsilon DECRESCENTE (1,0→0,01) — nunca fixo em 0,2.
2. `dqn.episodes`/`ppo.episodes`: citava Sec.3.8 ("no fewer than 1000
   episodes"), mas o config tinha 500 (e 50 no M5).
3. GA `crossover`: o artigo (Sec.3.4/4.4) diz "uniform crossover"; o código
   usava *blend crossover* (BLX-α), herdado sem mudança da versão DEAP de
   julho/2026.
4. GA `mutation`: o artigo diz "adaptive mutation"; o código usava taxa
   FIXA (0,05), não adaptativa.
Só os *valores numéricos* de população, probabilidade de crossover, taxa
inicial de mutação (GA) e coeficiente de entropia (PPO) de fato batiam com
as Tabelas 4/5 do artigo — a citação nos comentários dava a entender
alinhamento completo, quando só esses números estavam alinhados.

**Motivo:** Correção pedida explicitamente pelo usuário, que optou por
corrigir o **código** (não só a citação/documentação), mesmo sabendo que
isso invalida as execuções de "bot" e M5 em andamento no momento do
pedido (ambas rodando com os valores antigos).

**Objetivo:** Fazer o código bater de fato com o que os comentários e o
`REIMPLEMENTACAO_SOTA.md` afirmavam ter sido adotado do artigo-base, para
que qualquer citação de "hiperparâmetros alinhados a Zabraoui" na
dissertação seja verificável no código, não apenas no comentário.

**Impacto:**
- `conf/base/parameters/simulation.yml`: `dqn.epsilon_start=epsilon_end=0.2`,
  `epsilon_decay=1.0` (exploração constante); `dqn.episodes` e
  `ppo.episodes` de 500 → 1000; novo `ga.mutation_final_ratio: 0.2`.
- `conf/m5/parameters.yml`: `dqn.episodes`/`ppo.episodes` de 50 → 100
  (preserva o corte de 10x sobre o novo piso de 1000, mesma lógica de
  orçamento de tempo já documentada no item 18 — **não** uma correção de
  fidelidade, o corte em si continua sendo decisão nossa de tempo de
  execução, não do artigo).
- `core/metaheuristics_torch.py` (`TorchGA`): crossover trocado de blend
  para uniforme (cada gene herda do pai A ou B com p=0,5); mutação trocada
  de taxa fixa para adaptativa (decaimento linear de `mutation_prob` até
  `mutation_prob × mutation_final_ratio` na última geração — o artigo não
  dá fórmula para "adaptive mutation", esta é a leitura adotada, documentada
  na classe com a mesma transparência já usada em
  `VendorResponsivePolicy`). Testado isoladamente (20 indivíduos × 5
  gerações, série sintética): roda sem erro.
- `pipelines/inventory_simulation/nodes.py` (`_build_cfg`): novo campo
  `mutation_final_ratio` propagado ao `GENETIC_ALGORITHM` cfg.
- `simulation/REIMPLEMENTACAO_SOTA.md`: tabela "Hiperparâmetros adotados"
  reescrita para refletir o estado real (antes/depois desta correção).
- **NÃO corrigido, deliberadamente:** o horizonte de simulação. O artigo
  cita 365 dias (§4.7.1, só na seção híbrida) OU >40.000 *time steps*
  (§4.2, resultados principais) — as duas granularidades já divergem
  dentro do próprio artigo, e nenhuma bate com "ciclo comercial" (nossa
  unidade). Mudar a granularidade para dias quebraria o Experimento 2
  oficial da Bahia (145 séries, já validado) e o mapeamento
  `days_per_cycle=21` que torna Bahia/M5 comparáveis entre si — não
  alterado sem confirmação explícita adicional do usuário.
- **Consequência operacional:** as execuções de "bot" (`prod_bot_v2.log`,
  já na última etapa, `run_proposed_architecture`, ~1h56min de trabalho) e
  M5 (`prod_m5_v4.log`, em meta-heurística, ~2h12min) em andamento no
  momento desta correção usavam os valores ANTIGOS — descartadas e
  relançadas: `prod_bot_v3.log` (task `b3xkx0pcm`) e `prod_m5_v5.log`
  (task `b8isyiszt`). A Bahia v3 (`prod_benchmark_final_v3.log`, já
  CONCLUÍDA com 2610 linhas, ver item 22) também tinha rodado com o
  GA/DQN antigo — mesmo problema, pedido explicitamente ao usuário e
  confirmado: relançada como v4 (`prod_benchmark_final_v4.log`, task
  `b9puskok3`, `n_workers=1`). Dashboard (`scratchpad/dashboard.py`)
  atualizado para as três novas versões e reiniciado.

**Nota sobre por que este item existe:** esta correção não veio de uma
auditoria autônoma minha — o usuário afirmou ter certeza de que os
algoritmos do PDF do artigo-base não estavam implementados como descrito,
e pediu verificação. A leitura completa do artigo (22 páginas) confirmou a
suspeita. Registrado aqui integralmente porque é exatamente o tipo de
achado que este arquivo existe para não deixar perder: comentário de
código que cita uma fonte não é a mesma coisa que o código implementar o
que a fonte diz.

---

## 24. Correção de fidelidade ao Zabraoui: tratamento de dados do M5

**Descrição:** O usuário pediu para validar se o Zabraoui usa a base M5
inteira ou aplica algum corte/tratamento, e reproduzir o que ele de fato
faz. A Seção 3.7 do artigo (texto completo) é explícita:

> "Our experiments centered on a **curated selection of high-volatility
> food items**... The selection criteria included demand variability,
> **coefficient of variation**, and historical incidence of stockouts."
>
> "Outliers... were smoothed using a **robust 1.5 × IQR filtering
> method**... Missing values... forward-fill... Each time series was
> standardized using **z-score normalization**."

O artigo NUNCA usa o M5 inteiro (30.490 séries, 3 categorias) — usa só
**FOODS**, filtrado por CV/volatilidade. O `conf/m5/parameters.yml` desta
sessão (item #16, "M5 sem cortar nada") usava as 30.490 séries × 3
categorias, `cv_threshold=0` — o oposto do que o artigo-base faz com a
mesma base.

**Motivo:** Pedido explícito do usuário, no mesmo padrão de auditoria já
aplicado aos hiperparâmetros (item #23): "deve reproduzir qualquer
tratamento e seleção usada por ele... assim como os parâmetros" — ou seja,
corrigir o código, não só registrar a divergência.

**Objetivo:** Fazer a ingestão do M5 reproduzir a seleção e o tratamento
de dados que o artigo-base efetivamente usa, onde isso é possível
verificar precisamente a partir do texto.

**Impacto:**
- `conf/m5/parameters.yml`: `categories: ["FOODS"]` (era `null`, todas as
  3), `cv_threshold: 1.5` (era `0`) — mesmo valor já usado e testado no
  recorte interno e já validado para o M5-CA no item 4 de
  `REIMPLEMENTACAO_SOTA.md` (22 séries Lumpy de 23, regime comparável).
  `min_positive_cycles`/`active_product_window` permanecem OFF (0) — não
  são os critérios que o artigo cita, reativá-los seria inventar um corte
  que Zabraoui não descreve.
- `m5_loader.py`: novo passo de suavização de outliers (1,5×IQR) aplicado
  por série sobre a demanda agregada por ciclo — trunca a cauda superior
  em `Q3 + 1,5×(Q3−Q1)`, reproduzindo a Seção 3.7. Testado isoladamente
  (200 séries, 5 ciclos, categoria FOODS): roda sem erro, filtro de
  categoria confirmado (`df["segmento"].unique() == ["FOODS"]`).
- **NÃO reproduzido, com justificativa:**
  - "historical incidence of stockouts" como critério de seleção — o M5
    bruto só tem vendas realizadas, sem marcação de ruptura; não há proxy
    óbvio sem o artigo especificar um.
  - z-score normalization — normalizaria a escala para o treino do LSTM
    deles, mas corromperia as unidades reais de demanda que o nosso
    simulador usa para calcular custo de estoque. Mesma lógica da
    ressalva já registrada para o horizonte de 365 dias (item #23): não
    faz sentido no nosso pipeline, mesmo sendo citado no artigo.
  - forward-fill de valores ausentes — o `sales_train_evaluation.csv` do
    M5 não tem células ausentes (zero = sem venda, não é dado faltante);
    a nuance do artigo provavelmente se refere ao arquivo de preços
    (`sell_prices.csv`), que só é lido quando `with_revenue=true`
    (desligado por padrão) e já tem preenchimento por mediana em
    `_attach_revenue()`.
- **Consequência operacional:** M5 caiu de 30.490 para **2.408 séries**
  (confirmado no dashboard) — muito mais próximo da escala real que o
  artigo-base usa (não divulgada exatamente, mas descrita como "curated
  selection", não a base inteira). M5 relançado como `prod_m5_v6.log`
  (task `bbbdpdg77`), descartando o `prod_m5_v5.log` que tinha acabado de
  começar com a seleção antiga (baixo custo afundado — ainda em
  `run_classical_policies` no momento da correção).

---

## 25. Orçamento de busca do M5 alinhado ao Zabraoui, onde o artigo cobre

**Descrição:** Pedido do usuário: ajustar os parâmetros de cada algoritmo
do M5 para fidelidade ao Zabraoui — mesmo padrão dos itens #23/#24, agora
sobre os cortes de orçamento (`conf/m5/parameters.yml` bloco
`simulation`), que tinham sido feitos por teto de tempo (itens #18/#20)
quando o M5 tinha 30.490 séries, não por fidelidade.

**Motivo:** Com o M5 agora em 2.408 séries (item #24, 12,7x menos), o
orçamento completo ficou tempo-viável para os dois algoritmos que o
artigo de fato especifica: GA (Seç.4.4, "the best parameter set emerged
after 500 generations") e DQN/PPO (Seç.3.8, "no fewer than 1000
episodes" — já corrigido na base, item #23). Para SA, PSO, DE e SARSA,
**não há valor de fidelidade possível**: confirmado na auditoria do item
#23 que o artigo tem **zero ocorrências** desses quatro termos — não
existe parâmetro do Zabraoui para alinhar.

**Objetivo:** Usar o orçamento pleno do artigo onde ele é verificável;
deixar claro, em vez de silenciar, onde "fidelidade" simplesmente não se
aplica.

**Impacto:**
- `conf/m5/parameters.yml`: `ga.generations` 10 → **500**; `dqn.episodes`
  e `ppo.episodes` 100 → **1000** (remove o corte, iguala à Bahia/"bot").
- **Mantidos sem alteração, por ausência de referência no artigo:**
  `sa.max_iter=100`, `pso.iterations=16`, `de.max_iter=20`,
  `sarsa.episodes=50`, `hybrid.rl_episodes=30` — cortes de tempo puros,
  não uma escolha de fidelidade.
- **Estimativa de tempo, não medição:** com base no tempo real da Bahia
  v4 (metaheurística completa = 8m21s/145 séries sequencial;
  extrapolação de RL para 1000 episódios ≈ 107 min/145 séries), GA a 500
  gerações + DQN/PPO a 1000 episódios sobre 2.408 séries/7 workers deve
  ficar bem abaixo do teto de 12h — a redução de 12,7x em séries
  compensa o aumento de 10x/2x em gerações/episódios. **Não medido
  ainda**; acompanhar no dashboard e cortar de volta se necessário.
- M5 relançado como `prod_m5_v7.log` (task `bz42p2ykf`), descartando
  `prod_m5_v6.log` (só 3min em `run_metaheuristic_policies`, custo
  afundado desprezível).

---

## 26. Saída do AIPE: tabela de score por perfil, novo pipeline `profile_analysis`

**Descrição:** O usuário perguntou como o AIPE seleciona política por
perfil de loja e apontou que o dashboard só mostra política × score
geral. Investigação confirmou: **nenhuma das três pipelines em produção
(`benchmark_final`/`benchmark_bot`/`benchmark_m5`) roda a análise por
perfil nem o PSE real** — `pipeline_registry.py` só as compõe como
`di/dir_ + dp + inv (+ sv) + fr`, sem `rep` (onde vive
`profile_policy_analysis.py`) nem `ps` (`policy_selection`, o
classificador XGBoost). O usuário pediu a saída como "tabela com score de
cada política para cada perfil" e que "criação dos perfis deve fazer
parte do processo também".

**Motivo:** Fechar a lacuna entre o que o dashboard mostra (agregado
único) e o que a proposta (AIPE) precisa demonstrar (comportamento por
perfil operacional).

**Objetivo:** Preparar (não rodar ainda — as 3 simulações de produção
seguem intocadas) um pipeline que gera essa tabela assim que Bahia/bot/M5
terminarem, sem reexecutar a simulação de 18 políticas (a parte cara).

**Impacto:**
- `reporting/profile_policy_analysis.py`: `POLICY_ORDER` tinha só as 12
  políticas antigas — as 6 novas (PIL, CappedBaseStock, BigDataNewsvendor,
  MinMax, FixedInterval, VendorResponsive) ficavam **de fora do heatmap**
  (`_heatmap` filtra por `p in POLICY_ORDER`), embora já entrassem
  corretamente na tabela de dominância (que não depende dessa lista).
  Corrigido: as 18 agora aparecem em ambos.
  Nova coluna `score` na tabela `profile_policy_metrics.csv/.parquet`
  (perfil × política, todas as 18): reaproveita a MESMA formulação restrita
  da Eq. 4.2 já usada em `constrained_cost` (`score = -(TIC_mean + déficit
  de NS × peso × TIC_mean)`, maior = melhor) — não uma métrica nova,
  consistente com o resto do projeto. Testado com dados sintéticos: 18
  políticas × 2 perfis, `score` calculado corretamente, todos os artefatos
  gerados (incluindo novo `profile_policy_heatmap_score.pdf`).
  `run()` passou a aceitar `kpis`/`profiles`/`out_dir` explícitos (antes só
  lia caminhos hardcoded — não tinha noção de ambiente `m5`/`bot`).
- `reporting/nodes.py` (`generate_profile_policy_analysis`): agora passa os
  DataFrames já carregados pelo catálogo Kedro (respeitando isolamento por
  ambiente) e um `out_dir` isolado (`params:reporting.out_dir`), em vez de
  reler do disco em caminho fixo.
- Novos `conf/m5/parameters_reporting.yml` e `conf/bot/parameters_reporting.yml`:
  `reporting.out_dir` apontando pra `data/08_reporting/{m5,bot}/profiles`
  (mesmo padrão de isolamento do item #8).
- `reporting/pipeline.py`: novo `create_profile_analysis_pipeline()` —
  só o nó de análise por perfil, sem os nós de `demand_forecasting`/
  `statistical_validation` (não disponíveis em `benchmark_bot`/
  `benchmark_m5`, que quebrariam o resto do pipeline `reporting` completo).
- `pipeline_registry.py`: novo pipeline `"profile_analysis"` =
  `demand_profiling` (recalcula os perfis — "criação dos perfis faz parte
  do processo", pedido do usuário) + o nó de análise. Roda com
  `kedro run --pipeline profile_analysis [--env bot|m5]` depois que a
  simulação de cada base terminar. Validado via dry-run (`kedro registry
  list` + inspeção do catálogo resolvido): DAG monta corretamente, todos
  os inputs (`kpis`, `scenarios`, `scenarios_meta`) já existem no catálogo
  de cada ambiente. **Não executado ainda** — só preparado, a pedido
  explícito do usuário ("preparar agora, rodar depois").

---

## 27. Comparação "seleção por perfil vs. política única" + visibilidade no dash

**Descrição:** Dois pedidos do usuário nesta continuação: (1) "a ideia é
que a seleção por perfil seja melhor do que a geral" — precisa de uma
comparação explícita, não só a tabela de score; (2) "no dash deve ser
possível vê os perfis de cada experimento e quais variáveis de decisão
ele tem".

**Motivo:** (1) A tabela de score por perfil (item #26) mostra o score de
cada política em cada perfil, mas não responde diretamente "a seleção
contextual vale a pena?" — precisa comparar contra a alternativa de usar
uma única política pra tudo. Achado: `reporting/strategy_cost_comparison.py`
**já implementa exatamente essa comparação** (estratégias A1=política única
global, A2=baseline EOQ, B=seleção por perfil, C=oráculo por série, com
teste de Wilcoxon pareado H1: CTI_B < CTI_A1) — só tinha o mesmo problema
de I/O hardcoded (não isolado por ambiente) do item #26, e não estava
sendo chamado por nenhuma pipeline em produção.
(2) Perfis e variáveis de decisão eram invisíveis no dash — só política e
score geral apareciam.

**Objetivo:** Fechar a resposta metodológica completa (não só o score por
política×perfil, mas se a seleção por perfil de fato reduz custo vs.
política única) e dar visibilidade real-time no dash.

**Impacto:**
- `reporting/strategy_cost_comparison.py`: mesmo padrão de correção do
  item #26 — `run()`/`_load()` aceitam `kpis`/`profiles`/`out_dir`
  explícitos. Corrigidas também as "Checagem 1/2" do relatório de
  validação, que comparavam contra valores HARDCODED da Bahia (145
  séries, lista fixa de 12 políticas antigas) — sempre reportariam
  "DIVERGE" pra bot/M5/portfólio de 18. Agora comparam contra os valores
  reais dos próprios dados carregados. Testado com dados sintéticos (60
  séries × 18 políticas × 3 perfis): estratégia B mostrou 9,65% de
  redução de CTI vs. A1, todos os 9 artefatos gerados sem erro.
- `reporting/nodes.py` (`generate_strategy_cost_comparison`): mesmo padrão
  de correção — passa DataFrames do catálogo + `out_dir` isolado
  (subpasta `strategy/`, irmã de `profiles/`).
- `reporting/pipeline.py` (`create_profile_analysis_pipeline`): adicionado
  o nó `generate_strategy_cost_comparison`. Pipeline `profile_analysis`
  (item #26) agora tem 7 nós: `compute_demand_features`,
  `classify_operational_profiles`, `generate_policy_labels`,
  `generate_profile_policy_analysis`, `generate_strategy_cost_comparison`,
  `train_policy_selector`, `apply_policy_selector`. Validado via dry-run
  (`kedro registry list` + inspeção de catálogo): DAG resolve sem inputs/
  outputs faltando.
- `conf/bot/catalog.yml` e `conf/m5/catalog.yml`: novas entradas isoladas
  para `policy_labels`, `policy_selector_model`, `policy_selector_metrics`,
  `policy_recommendations` (datasets do PSE/`policy_selection`) — sem
  isso, rodar `profile_analysis --env bot|m5` reproduziria o padrão do
  incidente #2 (escrita no caminho base compartilhado com a Bahia).
- `dashboard.py`: nova seção "perfis operacionais (POD)" por execução —
  lê `demand_profiles.parquet` (já produzido cedo no pipeline, antes da
  simulação de 18 políticas) e mostra a distribuição real de séries por
  perfil. Confirmado funcionando: Bahia 80% Sparse_High_Impact/12,4%
  Unstable_Trend/7,6% High_Vol_Seasonal (145 séries); bot com 5 perfis
  representados (4.869 séries); M5 97,6% Sparse_High_Impact (2.408
  séries). Nova seção estática "variáveis de decisão por família de
  política" (referência fixa, 7 famílias: clássicas/SOTA/Zabraoui,
  Jornaleiro, PIL/Capped, Vendor-Responsive, meta-heurísticas, RL puro,
  híbridas — com as variáveis reais de cada uma, ex. RL: espaço de estado
  de 6 componentes + grade de ações discretas).
- Treino continua **por série** (decisão já confirmada pelo usuário nesta
  sessão) — nada disso muda a arquitetura de `inventory_simulation`; é
  visibilidade/relatório organizado por perfil, não retreino por perfil.
- **Não executado ainda** — só preparado. Roda com
  `kedro run --pipeline profile_analysis [--env bot|m5]` depois que
  Bahia/bot/M5 terminarem.

---

## 28. Mecanismo de pooling por perfil ("com perfil" vs. "sem perfil")

**Descrição:** Pedido do usuário: nova arquitetura de comparação -- pra
cada uma das 18 políticas, treinar UMA instância POR PERFIL (5 no total,
pooling: fitness/treino usa a demanda de TODAS as séries daquele perfil
agregada), versus UMA instância GLOBAL única (pooling sobre todas as
séries, ignorando perfil). Cada instância é depois avaliada
individualmente em cada série (KPI por série, mas a política por trás é
compartilhada dentro do pool). Confirmado explicitamente: mantém o
experimento atual (per-série independente, Bahia v4/bot/M5) como
referência à parte -- este é um experimento NOVO de 2 braços, não uma
substituição.

**Motivo:** Validar quantitativamente se conhecer o perfil operacional
antes do treino melhora o resultado, não só a seleção pós-hoc entre
políticas já treinadas (que é o que os itens #26/#27 já respondem).

**Objetivo:** Mecanismo reutilizável de "pool de séries" que qualquer uma
das 18 políticas aceita, sem duplicar código de treino.

**Impacto -- todas as 18 políticas agora aceitam `demand_pool`:**
- `core/metaheuristics_torch.py` (`_ThresholdOptimizer`, base de
  `TorchGA/SA/PSO/DE`): `evaluate()` avalia a mesma população contra CADA
  série do pool e usa o custo MÉDIO -- um único θ bom em média pro pool,
  em vez de um θ por série. Testado (4 séries sintéticas, sem erro,
  resultado difere do modo single-série como esperado).
- `core/rl_torch.py` (`DoubleDQNAgent`/`PPOAgent`/`ExpectedSARSAAgent`.
  `train()`): cada episódio sorteia (round-robin) uma série do pool em vez
  de sempre usar a mesma -- agente generaliza sobre o perfil. Testado
  sintaticamente (compila); teste funcional fica pro pipeline de
  orquestração (treino de RL é mais caro pra smoke-test isolado).
- `core/policies_sota.py` (`PILPolicy`, `CappedBaseStockPolicy`,
  `BigDataNewsvendorPolicy`): `_calibrate_batch` ganhou `demand_pool`
  (mesmo padrão de média sobre o pool do GA); BigDataNewsvendor concatena
  os pares (X,y) de TODAS as séries do pool antes de ajustar a regressão
  quantílica (um β compartilhado, mais linhas de treino). Testado, sem
  erro, `beta` ajustado com sucesso nos dois modos.
- `core/policies.py` (EOQ, (s,S), Newsvendor) e `core/policies_zabraoui.py`
  (MinMax, FixedInterval, VendorResponsive): **nenhuma mudança de código
  necessária** -- já calculam μ/σ a partir do array `demand` recebido;
  passar a demanda CONCATENADA do pool no lugar da série única já produz a
  versão pooled "de graça". Confirmado com teste de instanciação (6/6 OK).
- Híbridas (GA-DQN, GA-PPO): não têm código próprio de pooling -- reusam
  GA pooled + RL pooled por composição (herdam automaticamente).

**Pendente (próximo passo, não feito ainda):** a ORQUESTRAÇÃO -- um
pipeline novo que monta os 5 pools por perfil + 1 pool global a partir de
`scenarios`/`demand_profiles`, chama cada política com `demand_pool=...`,
avalia contra cada série, e produz a tabela de comparação com-perfil vs.
sem-perfil. Ainda não decidido em qual base rodar primeiro (Bahia é a
candidata natural: menor, já concluída, não interfere com bot/M5 em
andamento).

---

## 29. Bug: `cv_folds` do PSE quebrava com classes raras (NaN silencioso)

**Descrição:** Rodando `profile_analysis` de verdade na Bahia (145 séries),
`train_policy_selector` devolveu `CV accuracy=nan±nan`. Causa: com 11
políticas candidatas como rótulo (`best_policy`) e só 145 séries, algumas
políticas vencem em pouquíssimas séries (`policy_labels.parquet`:
`CappedBaseStock`: 1, `DE`/`DQN`: 2, `sS`/`Newsvendor`: 3).
`StratifiedKFold(n_splits=5)` exige >= 5 exemplos na classe menos
frequente; com só 1-2, `cross_validate` devolvia NaN por fold em vez de
erro -- falha silenciosa, sem aviso.

**Motivo:** Sem isso, tanto o modelo de produção (`train_policy_selector`)
quanto a comparação com/sem perfil (`evaluate_profile_feature_gain`, item
#28) ficam com métricas inúteis (NaN), sem nenhum sinal de que algo deu
errado.

**Objetivo:** `cv_folds` deve se adaptar ao tamanho da menor classe, com
aviso explícito quando isso acontece.

**Impacto:**
- `policy_selection/nodes.py`: nova `_safe_cv_folds(y, requested)` --
  `cv_folds` efetivo = `max(2, min(requested, tamanho_da_menor_classe))`,
  com log de aviso quando reduz. Usado em `train_policy_selector` e
  `evaluate_profile_feature_gain`.
- Testado contra os dados REAIS da Bahia (`policy_labels.parquet`, 145
  séries): reduziu corretamente de 5 para 2 (classe `CappedBaseStock` tem
  1 exemplo só), sem mais NaN.
- **Achado real da Bahia** (via `strategy_cost_comparison`, item #27, rodado
  com os dados de produção): seleção por perfil (B) reduz CTI só **0,2%**
  vs. política única global (A1) -- mas o oráculo por série (C, limite
  teórico) mostra **16,52%** de espaço. A maior parte do ganho possível
  NÃO é capturada pelos 3 perfis atuais (Sparse_High_Impact/
  Unstable_Trend/High_Vol_Seasonal presentes na Bahia) -- sinal de que os
  5 PODs (categorias fixas) são grossos demais pra explicar a
  heterogeneidade real entre séries; a informação mais fina que o PSE usa
  (features contínuas por série) tem mais chance de capturar esse gap do
  que agrupamento por perfil.

---

## 30. bot/M5 derrubados (evento de sessão, não bug de código) — foco realocado pra Bahia

**Descrição:** Ao retomar a sessão, `dashboard.py` (porta 8767) não
respondia e nenhum processo `python.exe` estava rodando. Os logs de "bot"
(`prod_bot_v3.log`) e M5 (`prod_m5_v7.log`) pararam de ser escritos às
14:25 e 14:28 respectivamente — a sessão foi retomada por volta das
18:54, ou seja, ~4h30min de silêncio. Não há erro nos logs (sem
traceback, sem "EXIT="): os processos simplesmente pararam de existir,
consistente com o aviso de tarefas "órfãs" (sessão anterior encerrada
derrubou os processos em background junto). Não é um bug de código desta
sessão.

**Estado salvo até a queda:**
- "bot": `kpis_classical`/`kpis_metaheuristic`/`kpis_proposed` completos;
  `kpis_rl` (RL puro) não chegou a ser salvo (Kedro só grava um dataset
  quando o nó inteiro termina) -- estava no meio dessa etapa.
- M5: mesma situação.

**Motivo/decisão:** O usuário pediu explicitamente pra focar na Bahia
(mais simples e rápida) até fechar a modelagem (pooling por perfil, PSE
com perfil como feature, etc. -- itens #26-#28) antes de reinvestir horas
de computação em bot/M5, que ficariam desatualizados de qualquer forma
com cada mudança de metodologia.

**Impacto:** bot e M5 **não foram relançados**. Próximos experimentos
desta sessão usam a Bahia (145 séries, já concluída e íntegra -- v4,
item #23) ou subconjuntos menores dela (como o `pooling_quick_experiment`
do item #28) como bancada de iteração rápida. Relançar bot/M5 fica para
depois que a modelagem estiver fechada.

---

## 31. `cv_folds` do PSE: correção real (LOO, a pedido do usuário) + resultado

**Descrição:** O item #29 (redução de `cv_folds`) não bastou -- ainda dava
NaN. Causa raiz real: o XGBoost rejeita qualquer fold de treino que não
contenha TODAS as classes vistas globalmente (`ValueError: Invalid
classes inferred`), não só por causa de estratificação -- isso quebra
até com `LeaveOneOut` (que o usuário pediu como alternativa ao k-fold),
sempre que a classe raríssima é a amostra deixada de fora.

**Motivo:** Pedido do usuário: "não tem uma outra estratégia sem ser
kfold?" -- trocado o padrão pra Leave-One-Out (`cv_strategy: "loo"`,
configurável, `"stratified_kfold"` disponível como alternativa). E:
classes com < 2 exemplos são excluídas da CV em QUALQUER estratégia
(inevitável matematicamente com o XGBoost) -- o modelo final continua
treinando com todos os dados, só a avaliação quantitativa exclui essas
classes.

**Objetivo:** Métrica de CV utilizável (não mais NaN), com o método mais
adequado ao tamanho pequeno da base (LOO usa o máximo de dado possível
pra treino em cada fold, importante com só 145 séries).

**Impacto:**
- `policy_selection/nodes.py`: `_build_cv(y, params)` substitui a função
  anterior -- mascara classes raras (sempre) e escolhe `LeaveOneOut` ou
  `StratifiedKFold` conforme `params["cv_strategy"]` (padrão `"loo"`).
- **Resultado real, agora sem NaN** (Bahia, 145 séries, LOO, 144 folds
  válidos após excluir `CappedBaseStock`):

  | | sem perfil | com perfil (feature) |
  |---|---|---|
  | Acurácia CV | 45,14% | 44,44% |
  | Ganho | -- | **-1,5%** (dentro do ruído, std±49,7%) |

  **Confirma e reforça o achado do item #27**: adicionar o perfil
  operacional como feature explícita ao PSE **não ajuda** -- o
  classificador já extrai das features contínuas (ADI, CV²...) tudo que
  a categoria de perfil ofereceria. Duas análises independentes (seleção
  por perfil vs. única global: +0,2%; perfil como feature do PSE: -1,5%)
  convergem pra mesma conclusão: os 5 PODs atuais carregam pouco sinal
  incremental sobre as features contínuas já disponíveis.

---

## 32. GPU real (RTX 4070 Ti): benchmark confirma o roteamento automático

**Descrição:** Pedido do usuário: testar GPU nesta máquina (desktop com
RTX 4070 Ti), já que a medição anterior (item 2c do
REIMPLEMENTACAO_SOTA.md, "35× mais lento na GPU") foi feita num notebook
(MPS da Apple). `torch` instalado era build **CPU-only**
(`2.13.0+cpu`) -- `cuda.is_available()` dava False mesmo com GPU real
presente (confirmado via `nvidia-smi`).

**Motivo/Objetivo:** Confirmar se a conclusão "CPU vence" se sustenta
numa GPU discreta de verdade, não só num notebook.

**Impacto:**
- Reinstalado `torch` com suporte CUDA:
  `pip install --index-url https://download.pytorch.org/whl/cu124 torch
  --force-reinstall --no-deps` -- trocou `2.13.0+cpu` por `2.6.0+cu124`
  (só versão com wheel CUDA disponível nesse índice). `cuda.is_available()`
  agora `True`, detecta "NVIDIA GeForce RTX 4070 Ti".
- Benchmark real (GA completo e simulação em lote bruto):

  | Lote | CPU | CUDA | Vencedor |
  |---|---|---|---|
  | GA pop=100 ger=50 (config de produção) | 0,441s | 2,045s | CPU, 4,6× |
  | 1.000 trajetórias | 0,0105s | 0,0361s | CPU, 3,4× |
  | 10.000 | 0,0319s | 0,0382s | CPU, levemente |
  | 100.000 | 0,2528s | 0,0372s | **CUDA, 6,8×** |

  Ponto de virada entre 10k-100k trajetórias -- mais baixo que os ~150k do
  notebook, mas ainda muito acima do lote real usado (população=100).
- `core/device.py`: `GPU_MIN_BATCH = {"cuda": 20_000, "mps": 150_000}` já
  existia com esse valor pra CUDA (calibrado sem medição real numa GPU
  CUDA de verdade) -- a medição confirma que já estava correto. **Nenhuma
  mudança de código necessária**: com `device: "auto"`, o roteamento já
  escolhe CPU pro GA/SA/PSO/DE de produção (lote pequeno) e só usaria GPU
  se o lote crescesse além de ~20k (cenário não usado hoje).
- **Ressalva:** downgrade de `torch` de `2.13.0` pra `2.6.0` (única versão
  com build CUDA disponível no índice usado) -- APIs usadas no projeto
  (nn.Module, optim.Adam, distributions.Categorical, operações de tensor
  padrão) são estáveis nessa faixa de versão; nenhuma quebra observada nos
  testes rodados após a troca.

---

## Estado no fim desta sessão

Todas as três relançadas mais uma vez após #23 (correção de fidelidade ao
Zabraoui: epsilon fixo, 1000/100 episódios, GA uniform crossover +
mutação adaptativa) — Bahia v3 e as primeiras tentativas de bot/M5 desta
rodada foram descartadas por terem rodado com o GA/DQN não corrigido.

| Base | Ambiente | Séries | Workers | Orçamento RL (DQN/PPO) | Fitness | GA (após #23) | Log | Status |
|---|---|---|---|---|---|---|---|---|
| Bahia (oficial, Experimento 2) | base | 145 | 1 (sequencial) | 1000 ep. | nova (TR+BE) | uniform×adaptativa | `prod_benchmark_final_v4.log` | **concluída** (exit 0, 3h51min, 2610 linhas/18 pol./145 séries) |
| "bot" (interna completa, 27 estados) | `bot` | 4.869 | 7 | 1000 ep. | nova (TR+BE) | uniform×adaptativa | `prod_bot_v3.log` | em andamento (~2h38min) |
| M5 (Walmart, FOODS+CV≥1,5 — #24) | `m5` | 2.408 | 7 | 1000 ep. (#25, sem corte) | nova (TR+BE) | uniform×adaptativa, 500 ger. (#25) | `prod_m5_v7.log` | em andamento (~2h37min) |

**Ferramentas de acompanhamento desta sessão** (scratchpad, fora do
repositório): log viewer (`http://127.0.0.1:8765/`) e dashboard
(`http://127.0.0.1:8767/`) — etapa atual, duração por etapa, progresso
série-a-série, parametrização real (via `KedroSession`) e resultados
parciais lidos dos `kpis_*.parquet`.

## Pendências / próximos passos

- **Bahia v4 CONCLUÍDA e verificada** (11:16, exit 0, 13892,8s ≈ 3h51min —
  quase 2x a v3, consistente com dqn/ppo episodes 500→1000): `KPIs
  agregados: 2610 linhas, 18 políticas, 145 séries | famílias:
  {'metaheuristic': 580, 'classical': 435, 'sota_classical': 435,
  'zabraoui': 435, 'rl': 435, 'hybrid': 290}` — contagem idêntica às
  versões anteriores, confirma integridade. **v4 é a versão final e
  válida da Bahia** (v3 SUPERSEDIDA pelo item #23 — GA/DQN não
  corrigidos). "bot" e M5 ainda em andamento (~2h38min).
- Bot (v3) e M5 (v7) estão rodando agora com o mesmo
  código corrigido (item #23) — precisam terminar e ser conferidas
  política por política (não só o `kpis` agregado, e não só o exit code)
  antes de dar qualquer uma como concluída, mesma lição do incidente #2.
- Confirmar que rodar as três em paralelo não reproduz o incidente #2,
  agora que o isolamento de catálogo (#8) cobre todos os datasets
  intermediários usados por `benchmark_m5`/`benchmark_bot`. A Bahia v3
  (descartada) tinha terminado limpa com bot/M5 ainda rodando ao lado —
  primeira evidência prática de que o isolamento se sustenta em paralelo;
  esta rodada (v4/v3/v5, lançadas juntas) é o segundo teste.
- Horizonte de simulação (365 dias vs. ciclos comerciais) permanece
  DIVERGENTE do artigo-base, deliberadamente não corrigido (ver item #23,
  ressalva) — decisão pendente do usuário se algum dia isso precisar ser
  reconciliado.
