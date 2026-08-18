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

## Estado no fim desta sessão

Três execuções de produção rodando:

| Base | Ambiente | Séries | Orçamento RL | Log |
|---|---|---|---|---|
| Bahia (oficial, Experimento 2) | base | 145 | 500 ep. | `prod_benchmark_final_v2.log` |
| "bot" (interna completa, 27 estados) | `bot` | 4.869 | 500 ep. | `prod_bot.log` |
| M5 (Walmart, externa, sem cortes — #16) | `m5` | 30.490 | 50 ep. (#18) | `prod_m5_v3.log` |

## Pendências / próximos passos

- As três runs de produção precisam terminar e ser conferidas política
  por política (não só o `kpis` agregado, e não só o exit code) antes de
  dar qualquer uma como concluída — foi exatamente essa checagem que
  revelou o incidente #2 na primeira tentativa da Bahia.
- Confirmar que rodar as três em paralelo não reproduz o incidente #2,
  agora que o isolamento de catálogo (#8) cobre todos os datasets
  intermediários usados por `benchmark_m5`/`benchmark_bot`. Sequência desta
  sessão: lançadas em paralelo deliberadamente como teste do isolamento
  (Bahia sozinha primeiro, depois bot e M5 somados sem parar a Bahia).
