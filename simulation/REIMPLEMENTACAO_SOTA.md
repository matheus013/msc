# Reimplementação das políticas para versões estado da arte

Data: 2026-08-17 · Commit base: `0689e9b`

Referências e critério de escolha: [`docs/references/estado_da_arte_politicas.md`](../docs/references/estado_da_arte_politicas.md).
Regra aplicada: quando havia mais de um candidato por política, ficou **a publicada mais recente em veículo relevante**.

---

## 1. Backup dos resultados anteriores

`/Users/matheus/Documents/mestrado_resultados_backup_2026-08-17/` — **fora do repositório**, 248 MB, 274 arquivos.

Contém `simulation/data/` completo (exceto o M5, re-baixável), `conf/` e o código original (`policies_ORIGINAL.py`, `inventory_env_ORIGINAL.py`).

**Não apague.** `simulation/data/01_raw/vendas/` está vazio no repositório: `02_intermediate/sales_raw.parquet` (62 MB, md5 `0fb7fea45334a7dd99dc7dd6256e767c`) é a única cópia em disco dos dados transacionais proprietários.

---

## 2. Portfólio: 12 → 15 políticas

O baseline foi **preservado intacto**, para que a redução de 48% e as demais comparações do Capítulo 5 continuem tendo denominador.

| Família | Políticas | O que mudou |
|---|---|---|
| Clássicas baseline | EOQ, (s,S), Jornaleiro | **Intactas** |
| Clássicas SOTA *(novas)* | PIL, CappedBaseStock, BigDataNewsvendor | Adicionadas |
| Meta-heurísticas | GA, SA, PSO, DE | Aptidão reformulada |
| RL | DQN, PPO, SARSA | Reimplementadas em PyTorch |
| Híbridas | GA-DQN, GA-PPO | Sobre o novo núcleo de RL |

### Clássicas estado da arte — `core/policies_sota.py`

- **`PILPolicy`** — van Jaarsveld & Arts, *Operations Research* 72(5):1790–1805, **2024**. Mais recente do slot (s,S). Pede de forma que o estoque **esperado no instante da chegada** atinja um alvo S. Ponto central: sob *lost sales* a projeção aplica `max(0,·)` a cada ciclo intermediário — projetar linearmente enfardaria o resultado justamente em séries Lumpy, onde o estoque zera com frequência. A projeção usa reamostragem da demanda empírica, não a média, porque em regime intermitente a média é péssimo resumo da distribuição.
- **`CappedBaseStockPolicy`** — Xin, *Operations Research* 69(1):61–70, **2021**. Incluída mesmo não sendo a mais recente porque é o benchmark que a literatura de DRL em *lost sales* **não supera de forma consistente**. Sem ela, os agentes de RL do portfólio seriam comparados só contra heurísticas subdimensionadas, inflando qualquer ganho reportado.
- **`BigDataNewsvendorPolicy`** — Ban & Rudin, *Operations Research* 67(1):90–108, **2019**. Mais recente do slot Jornaleiro. Regressão quantílica no quantil crítico τ = c_s/(c_s+h), resolvida como programa linear com regularização L1. Características escolhidas para descrever intermitência (fração de zeros recentes, ciclos desde a última venda), não apenas nível.

### RL em PyTorch — `core/rl_torch.py`

- **`DoubleDQNAgent`** — Double DQN (van Hasselt et al., 2016) + dueling (Wang et al., 2016), perda de Huber, rede alvo.
- **`PPOAgent`** — PPO (Schulman et al., 2017) com GAE (Schulman et al., 2016), bônus de entropia, recorte da função valor, minilotes.
- **`ExpectedSARSAAgent`** — Expected SARSA (van Seijen et al., 2009), discretização 2D (estoque × pipeline). O slot SARSA **não tem estado da arte publicado** em inventário; permanece como baseline tabular declarado.

Nomes antigos (`DQNPolicy`, `PPOPolicy`, `SARSAPolicy`) seguem exportados como aliases — nada quebra a jusante.

---

## 2a. Adoção da versão de Zabraoui et al. (2025)

Decisão: onde não há estado da arte publicado, usar a versão do artigo-base. PDF em `docs/references/1-s2.0-S2949863525000548-main.pdf`.

A leitura do texto completo mudou três coisas.

### O artigo não cobre a maioria do portfólio

A Seção 3.5 ("Heuristic models for baseline comparison") define **quatro** heurísticas: `(s,S)`, `Min-Max`, `Fixed Interval Replenishment`, `Vendor-Responsive`. Contagem de ocorrências no texto integral:

| Termo | Ocorrências | Onde |
|---|---|---|
| DQN | 41 | metodologia e resultados |
| Genetic | 36 | metodologia e resultados |
| PPO | 33 | metodologia e resultados |
| EOQ | 5 | **somente revisão de literatura** |
| (s,S) | 1 | Seção 3.5 |
| Min–Max / Fixed Interval / Vendor-Responsive | 2 / 1 / 1 | Seção 3.5 |
| **SA, PSO, DE, SARSA, Newsvendor** | **0** | — |

Consequência: a versão do Zabraoui existe para **(s,S), GA, DQN, PPO, GA-DQN e GA-PPO**, e acrescenta três heurísticas que o portfólio não tinha. Para **EOQ, Jornaleiro, SA, PSO, DE e SARSA o artigo não oferece nada** — a lacuna documentada na Seção 3.2 de `estado_da_arte_politicas.md` permanece.

Implementadas em `core/policies_zabraoui.py`: `MinMaxPolicy`, `FixedIntervalPolicy`, `VendorResponsivePolicy`. **Portfólio: 15 → 18 políticas.**

Ressalva de fidelidade registrada no próprio módulo: o artigo descreve cada uma dessas três em uma frase, sem equação. Min-Max e Fixed Interval seguem a definição clássica; **Vendor-Responsive é a mais subespecificada** ("adapts order timing based on supplier lead times and historical service levels") e a formulação adotada é uma leitura razoável, não uma transcrição — não a apresente na dissertação como sendo exatamente a do artigo.

### CORREÇÃO: GA-PPO não é contribuição original

Registro anterior deste documento afirmava que nenhum trabalho publicado combinava GA + PPO. **Errado.** O artigo cobre as duas variantes:

- §4.7.3, Algoritmo 1, linha 8: *"Initialize PPO or DQN agent π using θ\* as prior policy anchor"*
- §4.7.4 compara as duas: **GA-DQN 94% contra GA-PPO 91%** de nível de serviço; GA-PPO com *"more reactive ordering patterns and higher volatility in stock levels"*

Toda afirmação de originalidade sobre GA-PPO deve sair do texto da dissertação. O achado do artigo (GA-PPO mais reativo e volátil) converge com o FP alto observado no Experimento 1 — usar como convergência de evidência, não como novidade.

### CORREÇÃO: a aptidão do GA que eu havia "corrigido" era a do artigo

Equação (3) do artigo: `Fitness = λ1 · ServiceLevel − λ2 · TotalInventoryCost`. É a soma ponderada — exatamente o que o código original tinha em `fitness_weights: [1.0, 0.0001]`. A troca pela forma restrita **afastava** o código do artigo-base.

Default agora é `fitness_mode: "zabraoui"`. A variante `"constrained"` (Eq. 4.2 da proposta) segue disponível, e a ressalva sobre sensibilidade de escala está registrada em `zabraoui_fitness_cost()` — no M5, com itens de alto giro e escala homogênea, o efeito é pequeno; no recorte brasileiro, com razão de 124× entre lojas, não é.

### Modelo de custo: o artigo não tem custo variável de pedido

Equação (1): `J = Σ (c_h·I_t + c_p·S_t + c_o·R_t)`, com **R_t = número de pedidos emitidos**. O texto é explícito: *"c_o × R_t correctly reflects the total ordering cost ... by distinguishing between the number of ordering actions and the quantity purchased"*.

O ambiente da dissertação inclui `c_o^var · Q_t = 0,5·Q`. `policies_zabraoui.zabraoui_cost(cfg)` zera esse termo. A diferença desloca o ótimo: com custo variável, lotes grandes são penalizados e políticas de reposição frequente ganham vantagem relativa.

### Hiperparâmetros adotados

| Parâmetro | Valor | Fonte |
|---|---|---|
| GA população | 100 | Tabela 5 (faixa 50–200) |
| GA crossover | **0,8** (era 0,9) | Tabela 5 (faixa 0,6–0,9) |
| GA mutação | 0,05 | Tabela 5 (faixa 0,01–0,1) |
| GA elitismo | sim | §3.4 |
| ε (DQN) | 0,2 | Tabela 4 |
| Entropia (PPO) | **0,005** (era 0,01) | Tabela 4 |
| Episódios de RL | **≥1000** | §3.8 |
| Horizonte | 365 dias por item | §3.8 |

---

## 2b. Quais políticas NÃO receberam upgrade para estado da arte

Resposta direta: **8 das 12 não receberam upgrade algorítmico**, e em 7 dos casos o motivo é que **não existe estado da arte publicado em veículo relevante** para aquela política em inventário.

| Política | Upgrade SOTA? | O que de fato aconteceu |
|---|---|---|
| **EOQ** | **Não** | Sem frente de pesquisa ativa há décadas. Permanece baseline por design. |
| **(s,S)** | **Não** — ganhou par SOTA | Preservado como baseline do varejo. **PIL** (Oper. Res. 2024) entrou ao lado. |
| **Jornaleiro** | **Não** — ganhou par SOTA | Preservado. **BigDataNewsvendor** (Oper. Res. 2019) entrou ao lado. |
| **GA** | **Não** | Sem SOTA publicado. Ganhou elitismo, objetivo corrigido e vetorização. |
| **SA** | **Não** | Sem SOTA publicado. Cronograma de Metropolis agora explícito, em vez de `scipy.dual_annealing` (caixa preta). |
| **PSO** | **Não** | Sem SOTA publicado. Ganhou fator de constrição de Clerc & Kennedy (2002), que corrige divergência de velocidade. |
| **DE** | **Não** | Sem SOTA publicado. Formulação canônica `rand/1/bin` explícita. |
| **DQN** | **Sim** | Double DQN (van Hasselt et al., 2016) + dueling (Wang et al., 2016). |
| **PPO** | **Sim** | PPO com GAE (Schulman et al., 2016/2017). O gradiente anterior estava incorreto. |
| **SARSA** | **Parcial** | Expected SARSA (van Seijen et al., 2009) reduz variância, mas **não é SOTA de inventário** — não existe um, e o Zabraoui também não traz SARSA. |
| **GA-DQN** | **Sim, indireto** | A arquitetura já é Zabraoui 2025; herdou o núcleo Double DQN. |
| **GA-PPO** | **Sim, indireto** | Também está em Zabraoui 2025 (§4.7.3–4.7.4) — **não é original**. Herdou o núcleo PPO-GAE. |

Leitura para o Capítulo 3: só **DQN e PPO** tinham estado da arte identificável para onde subir. Nas meta-heurísticas a ausência é um resultado em si — a frente teórica migrou integralmente para DRL, e isso sustenta seu enquadramento de tratá-las como componentes conhecidos e não como contribuição.

Os slots clássicos ficaram cobertos por duas vias: a decisão de **manter + adicionar** trouxe PIL, CappedBaseStock e BigDataNewsvendor; a adoção da versão do artigo-base trouxe MinMax, FixedInterval e VendorResponsive. EOQ, (s,S) e Jornaleiro seguem intactos como baseline.

**Cobertura final por política** — 6 das 12 originais têm hoje versão SOTA ou do artigo-base no portfólio; 6 permanecem sem (EOQ, Jornaleiro, SA, PSO, DE, SARSA), e em todos esses casos porque nem a literatura de primeira linha nem o Zabraoui oferecem uma.

---

## 2c. Conversão para PyTorch e uso de GPU

Toda a pilha numérica passou a ser torch. Módulos novos:

| Módulo | Conteúdo |
|---|---|
| `core/device.py` | Detecção e roteamento de dispositivo (CUDA > MPS > CPU) |
| `core/inventory_env_torch.py` | `BatchInventoryEnv` — simulador vetorizado, B trajetórias em paralelo |
| `core/metaheuristics_torch.py` | GA, SA, PSO, DE vetorizados. **DEAP e scipy deixaram de ser necessários** |
| `core/rl_torch.py` | Double DQN, PPO-GAE, Expected SARSA |
| `core/policies_sota.py` | Calibração de PIL e CappedBaseStock por busca em grade vetorizada |

**Equivalência numérica verificada** contra o `InventoryEnv` original: 30 casos (6 séries × 5 parametrizações), erro relativo máximo de **3,7e-7** em TIC, NS, TR, BE e FP — precisão de float32. Sem isso, nenhum resultado novo seria comparável aos antigos.

### Sobre a GPU: ela existe, e usá-la aqui deixa mais lento

MPS está disponível (Apple Silicon) e a detecção foi implementada. Mas a medição contradiz o uso ingênuo:

| Lote | CPU | MPS | Ganho |
|---|---|---|---|
| 100 | 0,0028 s | 0,0694 s | **0,04×** |
| 1.000 | 0,0042 s | 0,0685 s | 0,06× |
| 10.000 | 0,0127 s | 0,0636 s | 0,20× |
| 100.000 | 0,0889 s | 0,1009 s | 0,88× |
| 400.000 | 0,3567 s | 0,2620 s | **1,36×** |

O ponto de virada fica em ~150.000 trajetórias simultâneas. A população do GA tem shape 100×3 — **quatro ordens de grandeza abaixo**. Nesse tamanho, lançar kernels e sincronizar custa mais que a conta.

Medido nas meta-heurísticas completas (GA 100×50, SA 500 iter, PSO 40×80, DE 100 iter): **CPU 1,14 s contra MPS 40,53 s — 35× mais lento na GPU.**

Por isso `device.py` roteia **consciente do tamanho do lote**: com `device: "auto"`, usa acelerador só acima do limiar medido (`GPU_MIN_BATCH`: MPS 150k, CUDA 20k — CUDA tem overhead menor). Preferência explícita (`AIPE_DEVICE=mps`) é sempre respeitada, para reproduzir medições.

Se este projeto rodar em máquina com CUDA e o benchmark for ampliado para avaliar muitas séries em um único lote, o roteamento passa a usar GPU automaticamente. Na configuração atual, CPU é a escolha correta e é o que o código faz.

### Ganho real da vetorização

O ganho não veio da GPU, veio de trocar o laço Python por operações tensoriais: GA de 100 indivíduos × 50 gerações executa em **0,11 s** (antes, 5.000 simulações sequenciais via DEAP). `CappedBaseStock` caiu de 0,06 s para 0,003 s.

Efeito colateral relevante: com a busca funcionando bem, **GA, SA, PSO e DE agora convergem para praticamente a mesma solução** (custo de treino ~2.435, Q≈166 na série de teste). Antes, o GA parava em soluções degeneradas do tipo ROP=0, Q=1. Isso é evidência de que o objetivo corrigido e a busca vetorizada estão fazendo o trabalho — e também sugere que, neste espaço de 3 parâmetros, a escolha da meta-heurística importa pouco: todas encontram o mesmo ótimo. Vale registrar isso no Capítulo 5, porque enfraquece qualquer afirmação de superioridade entre elas.

---

## 2d. Integração no Kedro

Nenhum script avulso. Tudo entrou no grafo, com entradas vindo do catálogo e saídas versionadas como datasets.

| Pipeline | Comando | O que faz |
|---|---|---|
| `data_resume` | `kedro run --pipeline data_resume` | Ingestão a partir de `sales_raw` já materializado |
| `final_report` | `kedro run --pipeline final_report` | Redundância, estratos de volume, confronto com a proposta, tabelas LaTeX |
| `benchmark_final` | `kedro run --pipeline benchmark_final` | Experimento 2 (BA) reexecutado ponta a ponta + análises |
| `benchmark_m5` | `kedro run --pipeline benchmark_m5 --env m5` | Comparação externa na base pública Walmart M5 |

**`data_resume` existe por necessidade, não por conveniência.** A fonte particionada original (`data/source/vendas`) não existe mais nesta máquina e os diretórios `01_raw/vendas` estão vazios. `data/02_intermediate/sales_raw.parquet` é a **única cópia em disco** dos dados transacionais proprietários — daí um ponto de entrada que parte dele. O `conf/local/catalog.yml` foi ajustado para apontar para um diretório existente, de modo que o pipeline não quebre ao resolver o catálogo.

**Ambiente `m5`** (`conf/m5/`) redireciona as saídas para diretórios próprios, para que o benchmark externo não sobrescreva os artefatos da base interna, e desliga o confronto com a proposta (que só faz sentido sobre os dados brasileiros).

Novos datasets no catálogo: `policy_redundancy`, `kpis_by_volume`, `proposal_comparison` (CSV) e `final_latex_tables` (PartitionedDataset de `.tex`, prontos para `\input{}`).

Parâmetros em `conf/base/parameters/final_report.yml`, incluindo o critério de redundância (`rho_min`, `dif_max`) e o caminho do baseline da proposta.

### Ambiente de execução

O projeto declara `kedro_init_version = "1.4.0"`, que exige **Python ≥ 3.10**. O `.venv` existente é Python 3.9.6, onde só é instalável o Kedro 1.0 — e o Kedro recusa executar com incompatibilidade de *minor*. Foi criado `.venv312/` com Python 3.12 e Kedro 1.4, que é o ambiente correto para `kedro run`.

---

## 3. Dois defeitos corrigidos que afetam resultados já publicados

Ambos merecem menção no texto da dissertação, porque mudam a interpretação do Capítulo 5.

**O "PPO" anterior não otimizava o objetivo do PPO.** A atualização do ator somava a vantagem diretamente aos logits alvo e treinava por erro quadrático:

```python
tgt_a[i, A[i]] += adv[i] * min(rat[i], cl[i])
self.actor.update(S, tgt_a)          # MSE contra logits modificados
```

Isso não é o gradiente da *surrogate function* recortada. A degeneração relatada no Experimento 1 (FP = 0,98, PPO pedindo em 98% dos ciclos) é **consistente com esse defeito**, e não necessariamente com uma limitação do PPO no regime Lumpy. A conclusão da Seção 5.x sobre "degeneração do PPO isolado" precisa ser reavaliada com a implementação correta.

**A aptidão das meta-heurísticas não era a da Equação (4.2).** Usava soma ponderada `w0·NS − w1·CTI` com pesos fixos, em vez de `min CTI s.a. NS ≥ α_min`. Como o CTI varia por ordens de grandeza entre séries (razão de 124× entre a maior e a menor loja no piloto), os mesmos pesos privilegiavam serviço em séries de baixo volume e custo em séries de alto volume — contaminando a comparação entre políticas. Agora a restrição entra por penalidade proporcional ao déficit e relativa ao próprio custo.

Verificação da nova aptidão: `não pedir` → −18.574 · `pedir bem` → −2.825. A restrição morde.

`fitness_mode: "weighted"` reproduz o comportamento antigo, para quem precisar regenerar os números atuais.

---

## 4. Seletor de base de dados

```bash
kedro run                                          # base interna (padrão)
kedro run --params "data_ingestion.source=m5"      # Walmart M5
```

`m5_loader.py` entra pelo **mesmo ponto** da ingestão e emite o schema interno, então filtragem, limpeza, construção de cenários e todo o resto do pipeline seguem inalterados.

Mapeamento: `warehouse ← state_id`, `store_id ← store_id`, `item_id ← item_id`, `segmento ← cat_id`.

Parâmetros em `conf/base/parameters/data_ingestion.yml`, bloco `m5`: `days_per_cycle` (21, casa com o ciclo comercial interno), `states`, `categories`, `max_series`, `max_cycles`, `with_revenue`, `seed`.

**A agregação em ciclos de 21 dias é o que torna as bases comparáveis.** O M5 diário tem CV² tipicamente < 1; o recorte brasileiro está em 3,8–4,1. Verificado: após agregar e aplicar `cv_threshold ≥ 1.5`, o recorte CA do M5 dá **22 séries Lumpy de 23**, ADI 2,81 e CV² 2,88 — contra ADI 2,4 e CV² 4,1 do Experimento 2. Regime comparável.

---

## 5. Ponto de atenção antes de rodar para valer

**Os parâmetros de custo não transferem entre as bases.** Com `h=1, c_s=5, c_o^fix=50` e séries M5 de baixo volume (μ ≈ 7 unidades/ciclo), romper sai mais barato que pedir: uma ruptura de 7 unidades custa 35, um pedido custa 50 + 0,5·Q. Todas as políticas convergem para NS baixo — comportamento economicamente correto dada essa parametrização, mas que inviabiliza a comparação com o recorte brasileiro.

Antes de gerar resultados de comparação externa, **recalibre os custos por base** (por exemplo, escalando `order_fixed` pelo volume médio da série) ou restrinja o recorte M5 a séries de volume compatível via `categories: ["FOODS"]` e filtro de μ mínimo.

---

## 6. Ambiente

`requirements.txt` precisa de `torch>=2.0` (instalado: 2.8.0). O venv em `.venv/` estava incompleto e recebeu `torch, deap, scikit-learn, xgboost, statsmodels, scikit-posthocs, pyyaml, kedro, kedro-datasets`.

`rl_torch.py` fixa `torch.set_num_threads(1)`. Não é preferência: `simulation.core.__init__` carrega xgboost e scikit-learn, cada um com sua libomp; o backward do torch em múltiplas threads sobre uma libomp já inicializada **causa segmentation fault** no macOS, na thread do autograd e sem frame Python. Ajustável por `AIPE_TORCH_THREADS`. Sem custo prático — as redes têm ~10⁴ parâmetros.

---

## 7. Estado de validação

Testado ponta a ponta sobre recorte M5 (4 séries, orçamento reduzido): **15 políticas, 60 linhas de KPI, 11,4 s**, todas as famílias produzindo resultado.

Ainda **não executado**: benchmark completo sobre as 145 séries da Bahia com orçamento de produção (GA 100×50, RL 500 episódios). Os resultados em `simulation/data/07_model_output/` continuam sendo os **antigos**, gerados pelo código anterior — foi para isso que o backup foi feito.
