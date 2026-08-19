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

## 33. Bug "nunca pedir" (PIL/CappedBaseStock/DQN/PPO) + métrica de custo total ajustado

**Descrição:** Pedido do usuário ("esse calculo de score não ta legal") levou
a investigar o `score` de `profile_policy_analysis.py` e descobrir algo mais
sério: PIL, CappedBaseStock, DQN e PPO colapsam para "nunca pedir" (S=0 ou
política sempre-ação-0) numa fração grande das séries da Bahia, com NS na
casa de 0,13-0,25 -- muito abaixo do resto do portfólio (GA/SA/PSO/DE:
NS≈0,87-0,90). Usuário autorizou ("Investigar e corrigir os 4 agora") via
AskUserQuestion. Em seguida, pediu uma métrica de escolha final ("custo total
ajustado") considerando estoque excessivo + indisponibilidade, e que o
comparativo final passasse a usar essa métrica.

**Causa-raiz #1 (confirmada numericamente, série BA/3775034/48062, grade S=0
a 40):** `constrained_cost` (Eq. 4.2) escala a penalidade de déficit de NS por
`tic_ref = clamp(TIC DA PRÓPRIA CANDIDATA, min=1.0)`. Um candidato "nunca
pedir" tem TIC quase zero, então penaliza a si mesmo com sua própria régua
minúscula -- mesmo com déficit de NS de 45+ pontos percentuais, a penalidade
em valor absoluto fica pequena. Confirmado: em uma série real, S=0 (NS=0,247,
custo=474,25) batia S=8 (NS=1,0, custo=587,06) na função objetivo ORIGINAL.

**Causa-raiz #2 (descoberta ao corrigir #1 e o problema persistir):**
mesmo com uma referência FIXA (não auto-referencial), `penalty_weight=10.0`
não bastava para séries de baixo volume onde o custo fixo de pedido (K=50)
domina -- o custo de de fato atingir NS≥0,70 (mais pedidos/estoque) cresce
mais rápido que a penalidade de aceitar o déficit. Empiricamente, séries que
precisavam de `penalty_weight≈20` (CappedBaseStock) a `≈30` (PIL, cuja
projeção de estoque tende a pedir mais agressivamente que um limiar simples
no mesmo S, ficando ainda mais caro).

**Causa-raiz #3 (DQN especificamente, não resolvida totalmente -- ver
Limitação):** DQN/PPO usam a recompensa CRUA do ambiente por passo (-custo),
sem NENHUM termo de NS -- diferente de `constrained_cost`. Adicionar uma
penalidade terminal (só computável no fim do episódio) resolveu PPO (porque
GAE calcula retorno multi-passo, propagando a correção pra toda a
trajetória numa única atualização), mas NÃO resolveu DQN sozinho: seu
bootstrap de 1 passo (TD) cria um ponto fixo autoconsistente ("nunca pedir"
já é a política gulosa em s', então o lookahead do Bellman nunca descobre o
benefício de pedir) -- testado com penalidade só no último passo e
distribuída por todos os passos, ambas sem efeito na série de teste, mesmo
com 500-1000 episódios.

**Motivo:** o objetivo é o mesmo em todo o projeto (Eq. 4.2, `constrained_cost`)
mas a busca em grade de PIL/CappedBaseStock e a otimização de DQN/PPO
exploravam essa brecha de um jeito que GA/SA/PSO/DE (busca populacional)
raramente encontravam sozinhos.

**Objetivo:** corrigir os 4 (autorizado pelo usuário) e criar uma métrica de
comparação final que não seja enganada pelo mesmo tipo de brecha.

**Impacto:**

1. **`inventory_env_torch.py`/`inventory_env.py`:** `.kpis()` ganhou
   decomposição de custo -- `HoldingCost`, `StockoutCost`, `OrderCost`,
   `AvgInventory` -- antes só existiam somados dentro de `TIC`. Propagado até
   `_kpi_row()` (nodes.py) para rodadas futuras gravarem esses campos em
   `kpis.parquet`.
2. **`constrained_cost`/`zabraoui_fitness_cost`/`fitness_cost`
   (inventory_env_torch.py):** novo parâmetro `tic_ref_fixed` -- quando dado,
   substitui a régua auto-referencial tanto no termo de déficit quanto em
   `_risk_terms`. Nova função `series_tic_ref(demand, cs) = max(cs *
   soma(demanda), 1.0)` -- custo do pior caso (nunca servir nada), intrínseco
   à série, não ao candidato. `None` preserva o comportamento antigo
   (retrocompatível).
3. **`penalty_weight`: 10.0 → 30.0** (`conf/base/parameters/simulation.yml`,
   `simulation.ga.penalty_weight` -- mesmo campo reaproveitado por
   GA/SA/PSO/DE e agora também por PIL/CappedBaseStock). Empiricamente
   necessário mesmo com `tic_ref_fixed`; não piora GA/SA/PSO/DE (o termo de
   deficit só entra em jogo quando NS<alpha_min, e eles já operam acima).
4. **`metaheuristics_torch.py` (`_ThresholdOptimizer`):** computa
   `self.tic_ref_fixed = series_tic_ref(...)` (média sobre o pool quando
   `demand_pool` é usado) e passa pra `_fitness()` -- GA/SA/PSO/DE agora
   usam a régua fixa também, por consistência (não tinham o bug de forma
   grave, mas ficavam sujeitos à mesma brecha em tese).
5. **`policies_sota.py` (`_calibrate_batch`, usado por PIL e
   CappedBaseStock):** passa `tic_ref_fixed=series_tic_ref(d, cs)` a cada
   candidato avaliado; `penalty_weight` deixou de ser um valor fixo
   hardcoded (10.0, nem lia config) e passou a ler
   `cfg["GENETIC_ALGORITHM"]["penalty_weight"]` (default 30.0).
   **Verificado end-to-end** em 3 séries reais degenerandas: série
   BA/9751111/75792 foi de NS=0,144 (S=0) pra NS=0,902 (S=5,71, viável) na
   janela de avaliação; série BA/3775034/48062 de NS=0,131 pra NS=0,402
   (melhora real, ainda não viável). Duas séries com janela de TREINO quase
   vazia (soma de demanda ≤5 em 17 ciclos) permaneceram em S=0 -- ver
   Limitação.
6. **`rl_torch.py`:** nova `_terminal_deficit_penalty(kpis, demand, cfg)` --
   aplicada ao ÚLTIMO passo do episódio em `PPOAgent._rollout` (única
   mudança) e ao buffer local de transições de `DoubleDQNAgent.train` (que
   passou a acumular a trajetória inteira antes de gravar no replay memory +
   chamar `replay()`, em vez de gravar/aprender passo a passo -- mesmo
   número de chamadas de `replay()` por episódio, só reordenadas).
   **PPO verificado**: mesma série 75792, NS 0,20→0,902. **DQN não
   resolvido** apesar da mudança estar no código -- ver Limitação.
7. **Métrica "custo total ajustado" (CTI_ajustado)**, pedido explícito do
   usuário -- implementada em `profile_policy_analysis.py`
   (`_add_adjusted_cost`) e `strategy_cost_comparison.py` (mesma fórmula
   reimplementada sobre a coluna já renomeada `CTI`, sem dependência
   cruzada entre os dois scripts):

   ```
   CTI_ajustado = CTI
                + deficit_NS * penalty_weight * CTI_ref_serie(FIXO = max
                  TIC observado entre as 18 políticas da mesma série -- não
                  o TIC da própria candidata)
                + excess_weight * max(0, HoldingCost - mediana(HoldingCost
                  na série))          [estoque excessivo; 0 se a rodada não
                                       tiver a decomposição de custo do item 1]
   ```

   `penalty_weight=10.0`, `excess_weight=0.5` (constantes do módulo de
   relatório, independentes do `penalty_weight=30.0` de treino do item 3 --
   aqui a régua é o MÁXIMO observado entre políticas da mesma série, não
   `series_tic_ref`, porque no relatório já se sabe o resultado de todas as
   18). Passa a ser o critério de seleção em `_dominant_policy_per_profile`
   (perfil B), `_pick_global_best`/`_dominant_by_profile` (estratégias A1/B
   de `strategy_cost_comparison.py`) e no oráculo por série (C) -- CTI bruto
   mantido só como referência lado a lado. Novas colunas/artefatos:
   `CTI_ajustado_mean`, `score_ajustado`,
   `profile_policy_heatmap_cti_ajustado.pdf`, colunas `CTI_ajustado_*` em
   `strategy_cost_comparison.csv`/`table_strategy_comparison.tex`.
   **Verificado** em cima do `kpis.parquet` atual da Bahia (sem a
   decomposição de custo do item 1 ainda -- excess_weight fica em 0):
   PIL/DQN/PPO (TIC_mean≈95, NS≈0,20) passam de `score`≈-571 pra
   `CTI_ajustado_mean`≈14.146 -- corretamente rebaixados ao fundo do
   ranking em vez de aparecerem artificialmente competitivos.

**Limitação (não resolvida nesta sessão, documentada em vez de ocultada):**
1. **DQN standalone continua colapsando** em pelo menos uma série testada,
   mesmo com a penalidade terminal implementada e testado até 1000 episódios
   -- é um problema de atribuição de crédito de 1 passo (bootstrap TD) que
   penalidade de recompensa sozinha não resolve. A arquitetura híbrida
   GA-DQN (`_worker_proposed`, `prepopulate_from_ga`) já usa exatamente o
   mecanismo que provavelmente resolveria isso (warm-start do replay buffer
   com a política do GA) -- o DQN standalone (`_worker_generic`, família
   "rl") não tem acesso aos parâmetros do GA nessa etapa do pipeline. Aplicar
   o mesmo warm-start ao DQN standalone é o próximo passo natural, não feito
   aqui por escopo.
2. **Duas das quatro séries de teste têm janela de TREINO quase vazia**
   (soma de demanda ≤5 unidades em 17 ciclos, vs. dezenas na janela de
   avaliação) -- nenhuma correção de penalidade ajuda aqui, porque o
   déficit de NS nem aparece DENTRO da janela de treino (a política parece
   ótima ali). É um problema de não-estacionariedade treino/avaliação do
   split walk-forward pra séries muito intermitentes, não um bug de fórmula
   -- possivelmente candidato a um ajuste futuro (ex.: janela de treino
   maior, ou prior de encolhimento pro nível populacional do perfil).
3. Nenhuma re-execução completa da Bahia foi feita ainda -- os números
   verificados são de recalibração isolada de séries específicas. O
   `kpis.parquet`/`demand_profiles.parquet` atuais ainda refletem a versão
   ANTERIOR ao fix (exceto onde a análise de relatório reage via
   `CTI_ajustado`, que já corrige a SELEÇÃO/COMPARAÇÃO mesmo sem re-treinar).
   Uma nova rodada de `benchmark_final` da Bahia é necessária pra que
   PIL/CappedBaseStock/PPO reflitam o treino corrigido nos KPIs brutos.

---

## 34. Experimento completo "com perfil" vs "sem perfil" (pooling, Bahia — 145 séries)

**Descrição:** Pedido do usuário ("mate todas as simulações e só deixe a
bahia com e sem perfil"): encerrada a rodada per-série padrão
(`benchmark_final`, que já refletiria o fix do item #33 mas não chegou a
terminar) e lançado em seu lugar o experimento completo de pooling —
`scratchpad/pooling_full_bahia.py` (elevação do protótipo pequeno
`pooling_quick_experiment.py`, já validado antes, para as 145 séries
reais/3 perfis reais/orçamentos de produção completos, lidos via
`KedroSession` — não hardcoded).

**Motivo:** decisão explícita do usuário de focar exclusivamente na
comparação com/sem perfil agora, adiando a rodada per-série padrão.

**Objetivo:** responder definitivamente se treinar 1 instância POR PERFIL
operacional (pooling) supera treinar 1 instância GLOBAL, usando a métrica
de escolha final corrigida (`CTI_ajustado`, item #33).

**Desenho:**
- `com_perfil`: 1 instância treinada por perfil (`Sparse_High_Impact` n=116,
  `Unstable_Trend` n=18, `High_Vol_Seasonal` n=11), pool = janela de TREINO
  (`demand_train`, walk-forward) de todas as séries do perfil.
- `sem_perfil`: 1 instância GLOBAL, pool = janela de treino das 145 séries.
- Ambos avaliados contra `demand_eval` de CADA série, com o **cfg REAL
  daquela série** (`scaled_params.pkl`, mesmo usado pelo benchmark padrão)
  -- só o TREINO usa um cfg representativo do grupo (média de mu/sigma via
  `rop_ref`), porque os otimizadores/agentes não aceitam cfg distinto por
  membro do pool (mesma limitação do protótipo pequeno).
- 18 políticas, orçamentos de produção (GA pop=100/gen=50, DQN/PPO 1000
  episódios cada, penalty_weight=30 -- fix do item #33 já embutido nas
  bibliotecas centrais, portanto ativo aqui sem mudança adicional).

**Impacto:**
- Rodou em 1464,7s (~24,4min) -- muito mais rápido que os per-série (~4h),
  porque treina só 4 instâncias por política (3 perfis + 1 global) em vez
  de 145. Concluído com exit 0, **5220/5220 linhas íntegras** (18×145×2),
  **zero falhas de treino/avaliação** no log.
- **Resultado**: pooling por perfil NÃO supera o treino global para as
  políticas boas -- pelo contrário, piora:

  | Política | CTI ajustado (sem perfil) | CTI ajustado (com perfil) | Vencedor |
  |---|---|---|---|
  | VendorResponsive | **684,30** | 852,35 | sem perfil (-24,6%) |
  | FixedInterval | **702,79** | 804,74 | sem perfil (-14,5%) |
  | PPO | **4.768** | 7.516 | sem perfil (-57,6%) |
  | GA-DQN | **1.739** | 2.204 | sem perfil (-26,7%) |

  Nas políticas mais fracas (CappedBaseStock, SARSA, PIL, DQN), "com
  perfil" ganha 17-34% -- mas nenhuma fica perto de ser competitiva
  (CTI_ajustado ainda 5x a 56x pior que a vencedora, VendorResponsive).
- **Conclusão metodológica**: reforça o achado do item #27 (PSE: perfil
  como feature só dava +0,2% de acurácia) -- o perfil operacional (POD) não
  parece uma unidade de agrupamento útil para TREINO neste dataset,
  provavelmente porque a heterogeneidade DENTRO de cada perfil (sobretudo
  Sparse_High_Impact, 116 das 145 séries) é grande demais para um modelo
  compartilhado por perfil ganhar sobre um modelo global ou por série.
  Decisão sugerida: não adotar pooling por perfil como estratégia de
  treino padrão; manter o treino por série (arquitetura principal) como
  referência.
- `simulation/src/simulation/pipelines/inventory_simulation/nodes.py`
  (`_kpi_row`) e `core/inventory_env*.py` (decomposição de custo, item #33)
  reaproveitados sem alteração -- resultado já sai com
  `HoldingCost`/`StockoutCost`/`OrderCost`/`AvgInventory` por linha.
- **Dashboard** (`scratchpad/dashboard.py`, porta 8767): `RUNS` esvaziado
  (per-série/bot/M5 removidos -- pedido do usuário "no dash só deve
  mostrar simulações rodando"); card dedicado `POOLING_RUN` com parser de
  log próprio (`get_pooling_status`/`get_pooling_results`, formato de log
  diferente do kedro) e tabela por (modo × perfil × política) com TODAS as
  métricas (TIC/NS/TR/BE/FP/HoldingCost/StockoutCost/OrderCost/
  AvgInventory/CTI_ajustado -- pedido do usuário "mesmo com CTI ajustado
  quero que mostre todas as métricas"), não só CTI_ajustado.
  `profile_policy_analysis._aggregate_by_profile` também passou a incluir
  a decomposição de custo nas métricas agregadas (antes só TIC/NS/TR/BE/
  FP/CTI_ajustado).

**Limitação:** cfg de TREINO único por grupo (não por série -- ver
Desenho); resultado válido como resposta à pergunta "pooling por perfil
ajuda?", mas não substitui uma rodada per-série completa com o fix do item
#33 (`benchmark_final`, ainda pendente -- ver Pendências).

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

---

## 35. Bahia v6 (`benchmark_final` pós-fix #33): resultado verificado em escala completa

**Descrição:** Pedido do usuário ("quero" -- relançar a Bahia per-série
após o experimento de pooling terminar). Rodou com 14 processos paralelos
(diferente da v4, que usou 1 worker sequencial) -- 1059,2s (~17,6min),
exit 0, **2610/2610 linhas íntegras** (18 políticas × 145 séries, sem
falhas).

**Motivo/Objetivo:** confirmar em escala completa (não só nas poucas
séries testadas manualmente no item #33) que o fix `tic_ref_fixed` +
`penalty_weight=30` (PIL/CappedBaseStock) e a penalidade terminal de
déficit de NS (PPO/DQN) realmente resolvem o colapso "nunca pedir".

**Impacto -- NS médio das 4 políticas antes degeneradas, ANTES vs DEPOIS do fix:**

| Política | NS antes (~) | NS agora (v6) | TIC agora | Situação |
|---|---|---|---|---|
| PIL | 0,20 | **0,752** | 853,95 | corrigido -- viável em média |
| CappedBaseStock | 0,25 | **0,642** | 561,87 | melhora grande, ainda abaixo de alpha_min=0,70 em média |
| PPO | 0,20 | **0,963** | 5.112,75 | corrigido -- TIC subiu (esperado: agora realmente pede) |
| DQN | 0,20 | 0,225 | 126,73 | **praticamente inalterado** -- confirma em escala completa a limitação já documentada no item #33 (bootstrap de 1 passo não credita a penalidade terminal às decisões iniciais do episódio) |

Confirma numericamente, na base inteira, o que o item #33 tinha verificado
só em 3-4 séries manualmente: PIL/CappedBaseStock/PPO respondem bem ao
fix; DQN precisa de uma correção estrutural diferente (warm-start do
replay buffer a partir do GA, como já feito na arquitetura híbrida GA-DQN
-- ver item #33, Limitação #1 -- ainda não implementado para o DQN
standalone).

**Este é agora o `kpis.parquet` vigente da Bahia** -- reflete o treino
corrigido; os relatórios (`profile_policy_analysis`/
`strategy_cost_comparison`, item #33) devem ser regenerados sobre ele pra
refletir os números atualizados (ainda não refeito nesta sessão).

---

## 36. Experimento completo "com perfil" vs "sem perfil" no M5 (2.408 séries)

**Descrição:** Pedido do usuário ("quando termina esse rode m5 com perfil x
sem perfil") -- mesma metodologia do item #34 (`pooling_full_bahia.py`),
adaptada pra M5 (`scratchpad/pooling_full_m5.py`): caminhos isolados
(`data/*/m5/*`), `KedroSession.create(env="m5")` pros orçamentos reais
(GA gerações=500 -- item #25 -- em vez de 50; DQN/PPO 1000 episódios;
penalty_weight=30 herdado do base). `scenarios`/`scenarios_meta`/
`scaled_params`/`demand_profiles` do M5 já existiam no disco (gerados
antes da run de produção M5 ter sido encerrada, item #30) -- reaproveitados
sem re-rodar a preparação de dados.

**Motivo/Objetivo:** mesma pergunta do item #34 (pooling por perfil ajuda?),
agora no M5 -- dataset de escala e distribuição de perfil muito diferentes
da Bahia (2.408 séries, 97,6% no perfil Sparse_High_Impact, só 58 nos
outros dois perfis somados).

**Impacto:**
- Rodou em 21.040,6s (~5h50min) -- muito mais lento que a Bahia (~24min)
  por causa do GA com 500 gerações (10x) por grupo, refletido sobretudo em
  GA-DQN (1461,7s) e GA-PPO (2342,0s), as duas etapas mais lentas.
  Concluído com exit 0, **86.688/86.688 linhas íntegras**
  (18 políticas × 2.408 séries × 2 modos), zero falhas.
- **Resultado, MUITO diferente do padrão da Bahia**:
  - Vencedora isolada em QUALQUER modo: **BigDataNewsvendor**
    (NS≈0,82, CTI_ajustado≈5.770-5.928) -- ordens de magnitude melhor que
    qualquer outra política (as demais ficam na casa de dezenas/centenas
    de milhares de CTI_ajustado). `com_perfil` levemente melhor aqui
    (5.770 vs 5.928, +2,7%).
  - Ao contrário da Bahia (onde só as políticas fracas ganhavam com
    pooling), no M5 **17 das 18 políticas** melhoram com `com_perfil` --
    mas a maior parte desse "ganho" é dominada por políticas que colapsam
    de formas DIFERENTES em cada modo (ex.: PSO NS cai pra 0,28 em
    `sem_perfil` vs 0,96 em `com_perfil`; MinMax/Newsvendor com NS
    0,24-0,42 nos dois modos) -- não é um sinal limpo de "perfil ajuda",
    é mais reflexo de optimização instável em séries muito esparsas.
  - As duas exceções (pioram com `com_perfil`): FixedInterval (-11,3%) e
    VendorResponsive (-13,5%) -- coincidentemente as vencedoras da Bahia.
- **Conclusão metodológica**: a ÚNICA leitura robusta é que
  BigDataNewsvendor domina o M5 por larga margem, com ou sem perfil --
  resultado qualitativamente diferente da Bahia (onde VendorResponsive/
  FixedInterval venciam). Reforça que a política dominante depende
  fortemente da escala/regime de custo do dataset (M5: alto volume,
  Bahia: majoritariamente Lumpy/baixo volume) -- não há uma política
  universal, e o sinal de "pooling por perfil ajuda ou atrapalha" também
  parece dependente do dataset (Bahia: atrapalha as boas políticas; M5:
  ambíguo, dominado por instabilidade das políticas fracas). **Não dá pra
  extrapolar uma recomendação única de "usar ou não pooling por perfil"
  a partir dos dois experimentos** -- eles discordam.
- Dashboard (`scratchpad/dashboard.py`): `POOLING_RUN` (singular) virou
  `POOLING_RUNS` (lista) -- 2 cards agora, Bahia e M5, cada um com seu
  parser de log e tabela de resultados por (modo × perfil × política) com
  todas as métricas.

**Limitação:** mesma do item #34 (cfg de treino único por grupo, não por
série); aqui ainda mais relevante dado o desbalanceamento extremo dos
perfis do M5 (perfil com 2.350 séries treinado com o mesmo cfg
representativo de mu/sigma médios, que pode não representar bem a
heterogeneidade interna desse grupo).

---

## 37. Validação de robustez: com/sem perfil em múltiplos splits temporais (EM ANDAMENTO)

**Descrição:** Usuário perguntou se os itens #34/#36 (com/sem perfil) foram
validados com leave-one-out -- não foram (LOO só é usado no PSE, item #31,
pipeline diferente). Os dois experimentos de pooling usaram um ÚNICO split
walk-forward (Bahia: ciclo 17/38; M5: ciclo 19/93), sem nenhuma checagem de
robustez sobre esse corte. Pedido do usuário: testar múltiplos splits
temporais (não LOO -- LOO não se aplica a séries temporais aqui, mudaria a
ordem causal treino→avaliação).

**Motivo/Objetivo:** verificar se as conclusões dos itens #34 (Bahia:
sem_perfil vence nas políticas boas) e #36 (M5: sinal ambíguo, dominado
por BigDataNewsvendor) dependem de qual parte da série foi usada como
treino, ou se são estáveis.

**Desenho:** `pooling_full_bahia.py`/`pooling_full_m5.py` ganharam
`--train_split_cycles` (override do corte) e `--out_suffix` (evita
sobrescrever os resultados principais dos itens #34/#36).
`pooling_full_m5.py` ganhou também `--reduced_budget` (GA 500→100
gerações, DQN/PPO 1000→300 episódios, híbridas 50→30 ger./200→60 ep.) --
decisão do usuário (AskUserQuestion): orçamento de produção completo pro
M5 custaria ~5h50min × 2 splits ≈ 12h; reduzido cabe em ~1-1,5h por split,
suficiente pra checar se a DIREÇÃO da conclusão muda (não é o resultado
"oficial", só validação de robustez). Bahia mantém orçamento de produção
completo (rápido, ~24min/split, sem necessidade de reduzir).

Splits testados: Bahia 10 e 25 (vs. original 17, de 38 ciclos); M5 10 e 40
(vs. original 19, de 93 ciclos). Rodados SEQUENCIALMENTE (nunca em
paralelo -- mesma regra do resto da sessão), via
`scratchpad/run_robustness_splits.sh`.

**Status:** lançado às 2026-08-19 03:30, ainda em andamento. Resultado
final e conclusão a registrar aqui quando terminar (Bahia: rápido,
~1h total pros 2 splits; M5: ~2-3h total pros 2 splits reduzidos).

---

## 38. Análise estatística pareada (com/sem perfil) + extensão pra M5/bot per-série (EM ANDAMENTO)

**Descrição:** Pedido do usuário: (a) análise estatística completa
com/sem perfil (não só ganho % agregado); (b) revisão de literatura sobre
pooling vs. treino individual (ver `docs/references/
pooling_vs_treino_individual.md`); (c) estender o fix #33 + validações às
rodadas per-série de M5 e "bot" (a Bahia per-série já rodou com o fix,
item #35).

**Impacto -- (a) `reporting/pooling_statistical_analysis.py` (novo):**
testes pareados de Wilcoxon (signed-rank, bilateral) + correção de Holm,
mesmo padrão de `strategy_cost_comparison._strategy_hypothesis_tests`,
aplicados a com_perfil vs sem_perfil, por política e agregado. Corrigido
um bug de direção do "vencedor" (a lógica original tratava delta>0 como
sempre favorecendo sem_perfil, o que inverte a resposta certa pra NS,
onde maior é melhor -- só CTI_ajustado/TIC têm "menor é melhor").

Resultado (dados já existentes dos itens #34/#36, sem re-treinar):
- **Bahia** (2610 pares): agregado, CTI_ajustado e TIC favorecem
  `com_perfil` (p<0,003); NS não difere (p=0,073). Por política: 17/18
  significativas (Holm) -- `sem_perfil` vence em PPO, GA-DQN,
  VendorResponsive, FixedInterval; todo o resto favorece `com_perfil`.
- **M5** (43.344 pares): TODAS as 18 políticas significativas (amostra
  grande). `sem_perfil` vence só em FixedInterval e VendorResponsive --
  **as mesmas duas políticas da Bahia**, replicado nos dois datasets.
- **Achado mais robusto**: FixedInterval/VendorResponsive (limiar
  Zabraoui ADAPTATIVO, recalculado por ciclo) preferem `sem_perfil` de
  forma consistente e estatisticamente significativa nos dois datasets --
  não é ruído de execução única, é um padrão real.

**Revisão de literatura** (`docs/references/pooling_vs_treino_individual.md`):
principal achado -- a literatura de *global vs. local forecasting models*
e *validation-driven clustering* explica o padrão observado: pooling só
ajuda quando o grupo tem dinâmica genuinamente similar; o Perfil
Operacional (POD) do AIPE é classificação por REGRAS FIXAS (ADI/CV²/
burstiness), nunca validada contra o alvo de custo (CTI_ajustado) --
consistente com por que "com_perfil" não vence de forma uniforme.
Direções de trabalho futuro citáveis: clustering validado por desempenho
de política (não regras fixas), fallback automático perfil→global por
política, "meta-modelo" por série (Meisheri et al. 2022, Neural Computing
and Applications) como terceiro ponto entre perfil e global.

**Extensão pra M5/bot per-série -- QUEUED, ainda não iniciada:**
`scratchpad/run_m5_bot_perserie.sh` (`kedro run --pipeline benchmark_m5
--env m5` seguido de `--pipeline benchmark_bot --env bot`, sequencial)
pronta pra disparar assim que a cadeia de robustez do item #37 terminar
(nunca em paralelo). M5 per-série (2408 séries, orçamento de produção
completo) e "bot" (4869 séries, pipeline completo incluindo
data_ingestion) são MUITO mais lentas que a Bahia per-série (17,6min/145
séries) -- estimativa grosseira por escala: M5 ~5h, bot ~10h+, total
~15h+ combinados. "bot" nunca tinha sido concluído nesta sessão (item #30:
encerrado por evento externo antes de terminar) -- será a primeira vez
com o fix #33 completo.

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
