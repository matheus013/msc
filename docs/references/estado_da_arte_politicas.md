# Estado da Arte por Política — Revisão Bibliográfica AIPE

Levantamento de 17/08/2026 para as 12 políticas do portfólio + a camada de seleção contextual (PSE).

**Filtro de veículo aplicado:** apenas periódicos indexados em Scopus/WoS com tradição em Pesquisa Operacional, Gestão de Operações ou Engenharia de Produção. Foram descartados anais de conferência de baixa seletividade e periódicos de escopo genérico — a lista do que caiu e o porquê está na Seção 4.

**Aderência ao cenário AIPE** (varejo real, multi-loja, *lost sales*, Lumpy com CV² 3.8–4.1, T=38, L=2): `★★★★★` muito próximo · `★☆☆☆☆` distante.

---

## 1. Tabela consolidada — política × artigo × ano × veículo

| # | Política | Artigo | Ano | Veículo | Ader. | PDF local |
|---|---|---|---|---|---|---|
| 1 | **EOQ** | *sem frente de pesquisa própria* — entra como baseline em Zabraoui et al. (linha 11) | 2025 | Supply Chain Analytics | ★★★★☆ | ⚠ pendente |
| 2 | **(s,S)** | van Jaarsveld & Arts, *Projected Inventory-Level Policies for Lost Sales Inventory Systems: Asymptotic Optimality in Two Regimes* | **2024** | **Operations Research** 72(5):1790–1805 | ★★★☆☆ | `VanJaarsveldArts2024_PIL_LostSales_OperationsResearch.pdf` |
| 2b | *(s,S) — fronteira algorítmica* | Temizöz, Imdahl, Dijkman, Lamghari-Idrissi, van Jaarsveld, *Deep Controlled Learning for Inventory Control* | 2025 | **EJOR** 324(1):104–117 | ★★★☆☆ | `Temizoz2025_DeepControlledLearning_EJOR.pdf` |
| 3 | **Jornaleiro** | Ban & Rudin, *The Big Data Newsvendor: Practical Insights from Machine Learning* | 2019 | **Operations Research** 67(1):90–108 | ★★★☆☆ | `BanRudin2019_BigDataNewsvendor_OR.pdf` |
| 4 | **GA** | Zabraoui et al. (linha 11) — GA otimizando ROP e estoque de segurança em dados reais | 2025 | Supply Chain Analytics | ★★★★☆ | ⚠ pendente |
| 5 | **SA** | — **sem SOTA em veículo relevante** (ver Seção 3.2) | — | — | ★☆☆☆☆ | — |
| 6 | **PSO** | — **sem SOTA em veículo relevante** | — | — | ★☆☆☆☆ | — |
| 7 | **DE** | — **sem SOTA em veículo relevante** | — | — | ★☆☆☆☆ | — |
| 8 | **DQN** | Oroojlooyjadid, Nazari, Snyder, Takáč, *A Deep Q-Network for the Beer Game* | 2022 | **M&SOM** | ★★☆☆☆ | `Oroojlooyjadid2022_BeerGameDQN_MSOM.pdf` |
| 8b | *DQN — referência de viabilidade* | Gijsbrechts, Boute, Van Mieghem, Zhang, *Can DRL Improve Inventory Management? Lost Sales, Dual-Sourcing, Multi-Echelon* | 2022 | **M&SOM** 24(3) | ★★★☆☆ | `Gijsbrechts2022_CanDRLImproveInventory_MSOM.pdf` |
| 9 | **PPO** | Vanvuchelen, Gijsbrechts, Boute, *Use of Proximal Policy Optimization for the Joint Replenishment Problem* | 2020 | **Computers in Industry** 119:103239 | ★★☆☆☆ | `Vanvuchelen2020_PPO_JointReplenishment_CompInd.pdf` |
| 9b | *PPO — fronteira multi-escalão* | Liu, Hu, Peng, Yang, *Multi-Agent Deep Reinforcement Learning for Multi-Echelon Inventory Management* | 2025 | **Production and Operations Management** | ★★★☆☆ | ⚠ pendente |
| 10 | **SARSA** | — **lacuna real**: nenhum trabalho em veículo relevante aplica SARSA a inventário | — | — | ★☆☆☆☆ | — |
| 11 | **GA-DQN** | Zabraoui, Hmamou, Chafi, Kammouri Alami, *A comparative study of multi-algorithm optimization for inventory analytics in supply chains* | 2025 | **Supply Chain Analytics** 12:100154 | ★★★★☆ | ⚠ pendente |
| 12 | **GA-PPO** | — **nenhum trabalho publicado encontrado** | — | — | — | — |
| — | **PSE** | Li, Kang, Petropoulos, Li, *Feature-based intermittent demand forecast combinations: accuracy and inventory implications* | 2023 | **IJPR** 61(22):7557–7572 | ★★★★☆ | `Li2023_FeatureBasedIntermittentDemand_IJPR.pdf` |

Sobre o veículo de Zabraoui et al.: *Supply Chain Analytics* (Elsevier, ISSN 2949-8635) é indexado em Scopus e Web of Science, SJR 0,896. É periódico novo (2023), mas legítimo — mantido na lista.

---

## 2. Situação dos downloads

**7 de 9 baixados** em `docs/references/`. Todos verificados pelo conteúdo, não apenas pelo nome do arquivo.

Nas linhas 2, 2b, 3 e PSE, o arquivo obtido é a versão de repositório (arXiv / LBS Research Online), idêntica em conteúdo à publicada, mas com paginação diferente da versão final do periódico. Para citação com número de página exato, use os metadados da tabela acima, não a paginação do PDF.

**2 pendentes** — ambos são *open access*, mas o editor bloqueia download automatizado (proteção anti-bot). Abrindo no navegador, o PDF baixa normalmente:

| Artigo | Situação | Link |
|---|---|---|
| Zabraoui et al. 2025 (Supply Chain Analytics) | **Gold OA, CC-BY 4.0** | <https://doi.org/10.1016/j.sca.2025.100154> |
| Liu et al. 2025 (Production and Operations Management) | **Hybrid OA** | <https://doi.org/10.1177/10591478241305863> |

O de Zabraoui é o mais importante dos dois — é o artigo-base da arquitetura GA-DQN e o cenário mais próximo do seu.

---

## 3. Leitura por família

### 3.1 Clássicas

**EOQ não tem fronteira de pesquisa própria em veículo relevante.** O que existe são extensões setoriais em periódicos de aplicação. Para a dissertação isso é suficiente e até conveniente: o EOQ entra como baseline, e a literatura de OM não o trata como objeto de pesquisa ativo há décadas.

**(s,S) é a família com a fronteira mais sofisticada — e a mais desconfortável para o seu argumento.** Dois achados a incorporar:

1. **Nenhuma abordagem DRL superou consistentemente a política `capped base-stock` em *lost sales*.** Isso importa porque seu ambiente é *lost sales* ([`inventory_env.py:131`](../../simulation/src/simulation/core/inventory_env.py#L131)).
2. **A política PIL de van Jaarsveld & Arts (Operations Research, 2024) é assintoticamente ótima em dois regimes** — custo de ruptura → ∞ e *lead time* longo — e domina a *constant order policy* para qualquer *lead time* finito.

Seu (s,S) atual é uma parametrização heurística ([`policies.py:94-109`](../../simulation/src/simulation/core/policies.py#L94-L109): `s = μL + zσ√L`, `S = s + Q_EOQ`), não a melhor política clássica conhecida para *lost sales*. **A banca pode apontar baseline subdimensionado.** Não é preciso reimplementar PIL — mas declare no Cap. 3 que o (s,S) do portfólio é a política *praticada no varejo regional brasileiro*, não o ótimo teórico da literatura de *lost sales*, citando van Jaarsveld & Arts (2024) e Temizöz et al. (2025) para delimitar o recorte.

**Jornaleiro:** a fronteira migrou de "estimar o quantil" para decisão orientada a *features*. Ban & Rudin (2019) é o marco que dispensa a estimação da distribuição em favor de aprender a decisão direto das características — conceitualmente é o PSE um nível abaixo.

### 3.2 Meta-heurísticas (GA, SA, PSO, DE)

**Achado central, e é um resultado negativo forte: não há estado da arte de meta-heurísticas para inventário em veículo relevante.** Aplicando o filtro de veículo, as âncoras que eu havia levantado para SA (PLOS ONE), PSO (periódicos de estudo de caso setorial) e DE (MDPI *Sustainability*) caem todas. A frente teórica migrou integralmente para DRL e métodos data-driven; o que resta de meta-heurística em inventário são aplicações setoriais em periódicos de baixa seletividade.

Isso **fortalece** o seu enquadramento em vez de enfraquecê-lo: você já trata GA/SA/PSO/DE como métodos conhecidos de parametrização, não como contribuição — e a literatura confirma que essa é a leitura correta. O GA sobrevive na lista apenas porque aparece dentro de Zabraoui et al. (2025).

Recomendação de escrita: no Cap. 3, afirme explicitamente que meta-heurísticas aplicadas a inventário não constituem frente de pesquisa ativa em veículos de primeira linha, e que sua inclusão no portfólio se justifica pela cobertura de famílias de busca — não por haver um resultado de referência a superar.

### 3.3 Aprendizado por reforço

**DQN.** Oroojlooyjadid et al. (M&SOM, 2022) e Gijsbrechts et al. (M&SOM, 2022) permanecem as duas referências centrais e ambas já estão no seu `.bib`. Gijsbrechts é a mais útil das duas para você: mostra que DRL é viável em *lost sales* e multi-escalão, mas **sem dominar heurísticas bem calibradas** — exatamente o padrão que seus experimentos reproduzem.

**PPO.** Vanvuchelen et al. (2020) é a origem da aplicação a reposição. A fronteira atual é multiagente: Liu et al. (POM, 2025) aplica HAPPO a multi-escalão descentralizado e obtém custo menor que DRL de agente único e que heurísticas. Achado colateral relevante: **compartilhar informação apenas na fase de treino atenua o efeito bullwhip** — dialoga direto com sua métrica BE, ainda que MARL esteja fora do seu escopo declarado ([`introducao.tex:102`](../master_proposal/capitulos/introducao.tex#L102)).

**SARSA é lacuna genuína.** Nenhum trabalho em veículo relevante aplica SARSA a controle de inventário; ele sobrevive como baseline tabular didático. Recomendo posicioná-lo explicitamente como baseline de TD-learning *on-policy*, deliberadamente simples, para contrastar com o DQN *off-policy* — e observar que o resultado degenerado do Experimento 1 (FP = 0) é consistente com a ausência do método na literatura aplicada.

### 3.4 Híbridas GA-RL

**GA-DQN:** Zabraoui et al. (2025) é a âncora, e já é seu artigo-base. Confirmado via Crossref e pela leitura do texto completo: *Supply Chain Analytics*, vol. 12, art. 100154, 2025, CC-BY 4.0; compara RL, GA, DL, ML e heurísticas sob framework unificado no Walmart M5; GA-DQN eleva NS de 61% para 94% sobre DQN isolado.

**GA-PPO — CORREÇÃO.** Em versão anterior deste documento eu afirmei que nenhum trabalho publicado combinava GA + PPO, e que a variante seria original desta dissertação. **Isso está errado.** A leitura do texto completo de Zabraoui et al. (2025) mostra que o artigo cobre as duas variantes:

- §4.7.3, Algoritmo 1, linha 8: *"Initialize PPO or DQN agent π using θ\* as prior policy anchor"*
- §4.7.3: *"a DRL agent (PPO or DQN)"*
- §4.7.4 "Hybrid DRL performance comparison" compara GA-PPO e GA-DQN diretamente, reportando **GA-DQN 94% contra GA-PPO 91% de nível de serviço**, com GA-PPO exibindo "more reactive ordering patterns and higher volatility in stock levels"

**Consequência para a dissertação:** GA-PPO **não é contribuição original**. Qualquer afirmação de originalidade sobre essa variante precisa sair do texto. O achado de Zabraoui — GA-PPO mais reativo e mais volátil que GA-DQN — é, aliás, consistente com o que o Experimento 1 observou (FP alta no PPO), e vale citar como convergência de evidência em vez de novidade.

### 3.5 Camada de seleção (PSE)

Após o filtro de veículo, resta **uma única referência**: Li, Kang, Petropoulos & Li (IJPR, 2023). Framework *feature-based* para demanda intermitente que extrai características das séries, combina previsões e avalia **implicações de inventário** — não só acurácia — sobre duas bases reais. É a referência mais próxima da sua camada de entrada, no mesmo regime de demanda. **Diferença decisiva: seleciona/combina *previsores*, não *políticas de reposição*.**

O trabalho conceitualmente mais próximo do PSE que localizei — Varghese et al., seleção dinâmica de política de reposição por ML supervisionado em cadeia de 3 escalões — está em anais IEOM, conferência de baixa seletividade, e por isso **não entra na lista**. Registro aqui mesmo assim, porque a banca pode levantá-lo: dados sintéticos, cadeia serial, 4 políticas candidatas, sem regime intermitente. Se for citado, que seja como evidência de interesse na direção, jamais como estado da arte.

**Consequência para o posicionamento:** com o filtro de veículo aplicado, a seleção contextual de *políticas de reposição* por características operacionais da série **não tem estado da arte estabelecido em periódico de primeira linha**. Li et al. (2023) ocupa o análogo no nível de previsão. Essa é a formulação mais forte e mais defensável da sua lacuna — e é mais forte do que a redação atual da Seção 3.5.

---

## 4. O que foi descartado pelo filtro de veículo

| Referência | Veículo | Motivo |
|---|---|---|
| Varghese et al. 2022, *Dynamic Selection of Inventory Replenishment Policies* | anais IEOM | conferência de baixa seletividade; sem revisão por pares robusta |
| Erkayman et al. 2025, *Markov approach... intermittent demand* | PLOS ONE | multidisciplinar de escopo genérico, sem tradição em PO — **atenção: já citado em `correlatos.tex`** |
| *Optimizing Inventory in Convenience Stores... Random Forest and GA* | Logistics (MDPI) | periódico MDPI, baixa seletividade |
| *Data-Driven Simulation–Optimization for Sustainable (s,S)...* | Sustainability (MDPI) | fora de escopo de PO |
| *A practical approach to replenishment... extended (R,s,Q)* | Scientific Reports | escopo genérico |
| Estudos de caso PSO (indústria elétrica, perecíveis, agrícolas) | periódicos regionais | baixa circulação, sem indexação relevante |
| *Deep Neural Newsvendor*, *Deep Generative Demand Learning* | arXiv | preprints, sem publicação confirmada |

Dois preprints foram mantidos, **separados em `docs/references/preprints/`**, porque vêm de grupos industriais de peso e representam a fronteira real, ainda que sem publicação. Não são citáveis como estado da arte consolidado — decida se entram:

- `Maggiar2025_StructureInformedDRL_arXiv.pdf` — Amazon; embute estrutura da política ótima na arquitetura da rede
- `Xie2026_DeepStock_Alibaba_arXiv.pdf` — Alibaba/Chicago Booth; DRL em produção em >1M pares SKU-armazém na Tmall, −20% no tempo de giro

---

## 5. Entradas a adicionar no `referencias.bib`

Ausentes hoje e necessárias após esta revisão:

- `VanJaarsveldArts2024PIL` — Operations Research 72(5):1790–1805, 2024 — **prioritário**, delimita o baseline de *lost sales*
- `Temizoz2025DeepControlledLearning` — EJOR 324(1):104–117, 2025 — SOTA algorítmico em *lost sales*
- `Li2023FeatureBasedIntermittent` — IJPR 61(22):7557–7572, 2023 — **prioritário**, único análogo do PSE em veículo relevante
- `Liu2025MADRLMultiEchelon` — Production and Operations Management, 2025 — fronteira PPO multi-escalão e bullwhip
- `BanRudin2019BigDataNewsvendor` — Operations Research 67(1):90–108, 2019 — fundamenta a lógica *feature-based* do PSE

Já presentes no `.bib` e confirmados: `Oroojlooyjadid2022BeerGameDQN`, `Gijsbrechts2022DRLInventoryMSOM`, `Vanvuchelen2020PPOJointReplenishment`, `Zabraoui2025GaDQN`.

Revisar: `Erkayman2025MarkovInventory` (PLOS ONE) — citado em `correlatos.tex` como âncora de meta-heurística em demanda intermitente. Não reprova por si só, mas é o elo mais fraco da seção.

---

## 6. Base de dados Walmart (M5)

Base usada por Zabraoui et al. (2025) baixada e disponível em `simulation/data/01_raw/m5_walmart/` — 488 MB, coberta pelo `.gitignore`.

| Arquivo | Tamanho | Conteúdo |
|---|---|---|
| `sales_train_evaluation.csv` | 116 MB | 30.490 séries loja-produto × 1.941 dias |
| `sales_train_validation.csv` | 114 MB | mesma base até d_1913 |
| `sell_prices.csv` | 226 MB | preço semanal por loja-item |
| `calendar.csv` | 110 KB | datas, eventos, SNAP por estado |
| `sales_test_evaluation.csv` / `sales_test_validation.csv` | 3,1 / 2,8 MB | janelas de teste (28 dias) |
| `weights_evaluation.csv` / `weights_validation.csv` | 2,2 / 2,1 MB | pesos WRMSSE da competição |

Estrutura de `sales_train_*`: `item_id, dept_id, cat_id, store_id, state_id, d_1 … d_1941`. Granularidade diária, 10 lojas (CA/TX/WI), 3.049 produtos, 3 categorias.

**Fonte:** mirror público do repositório Nixtla (`github.com/Nixtla/m5-forecasts`), idêntico ao pacote da competição M5 no Kaggle. Não exigiu credenciais.

**Ressalva de comparabilidade:** o M5 é diário e majoritariamente CV < 1; seu recorte é por ciclo comercial de ~21 dias e Lumpy com CV² 3,8–4,1. Para usar o M5 como base de comparação externa, será preciso agregar as vendas em ciclos equivalentes e aplicar seu filtro de intermitência (CV ≥ 1,5) — sem isso, a comparação com o Experimento 2 não é justa, e é exatamente o ponto que a Tabela `tab:relwork_intro` já registra ao marcar Zabraoui como CV < 1.
