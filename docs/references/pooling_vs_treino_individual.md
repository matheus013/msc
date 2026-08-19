# Pooling vs. Treino Individual — Revisão Bibliográfica AIPE

Levantamento de 19/08/2026, motivado pelos experimentos "com perfil vs sem
perfil" (`simulation/AJUSTES_INFRA_2026-08-18.md`, itens #34/#36/#37):
treinar UMA instância de política POR PERFIL operacional (pooling
por perfil) supera treinar uma instância GLOBAL (pooling sem perfil), ou
treinar uma POR SÉRIE (arquitetura principal do AIPE)?

**Filtro de veículo** (mesmo critério de `estado_da_arte_politicas.md`):
periódicos indexados Scopus/WoS com tradição em PO/Gestão de Operações/
Aprendizado de Máquina aplicado. Pré-prints (arXiv sem venue) são citados
como leitura de apoio, marcados `⚠ preprint`, não como SOTA equivalente a
artigo revisado por pares.

**Aderência ao cenário AIPE** (séries heterogêneas de volume/perfil, regime
Lumpy, decisão de POLÍTICA de reposição — não só previsão de demanda):
`★★★★★` muito próximo · `★☆☆☆☆` distante.

---

## 1. Síntese direta (para quem vai só ler isto)

A pergunta "agrupar séries pra treinar um modelo compartilhado" tem
literatura extensa do lado de **previsão** (global vs. local forecasting
models) e mais escassa do lado de **política de decisão** (RL/otimização
compartilhada entre produtos). As duas convergem na mesma condição:

> Pooling só ajuda quando as séries do grupo compartilham dinâmica
> genuinamente similar; quando o grupo retém heterogeneidade substancial,
> o modelo compartilhado sofre "transferência negativa" e um modelo
> global (ou individual) tende a vencer.

Isso bate exatamente com o que os itens #34/#36/#37 encontraram
empiricamente: o Perfil Operacional de Demanda (POD) do AIPE é
**baseado em regras** (limiares de ADI/CV²/burstiness — não é aprendido
nem validado contra o desempenho de política que ele deveria agrupar).
A literatura de *validation-driven clustering* (Seção 2.3) é explícita
sobre esse ponto: o critério de agrupamento precisa ser validado contra a
métrica-alvo, senão o "pool" mistura séries que não deveriam estar juntas
— exatamente o padrão observado (perfil `Sparse_High_Impact` concentra
80% da Bahia e 97,6% do M5, quase certamente heterogêneo por dentro).

---

## 2. Previsão de séries temporais: modelos globais vs. locais

| # | Achado | Fonte | Ano | Veículo | Ader. |
|---|---|---|---|---|---|
| 1 | Modelos globais (pooled) superam modelos locais mesmo quando as séries do dataset são não-relacionadas; TiDE (rede neural simples) consistentemente supera modelos locais com menor custo computacional | Damato, Rubattu, Azzimonti, Corani — *Intermittent time series forecasting: local vs global models* | 2026 | arXiv 2601.14031 | ★★★★☆ `⚠ preprint` |
| 2 | Estudo de simulação sobre quando modelos globais funcionam/falham em séries heterogêneas | *Global Models for Time Series Forecasting: A Simulation Study* | 2020 | arXiv 2012.12485 | ★★★☆☆ `⚠ preprint` |
| 3 | Modelos globais grandes/complexos são MAIS caros computacionalmente E menos precisos que versões simples — tamanho/complexidade do modelo importa, não só "pooling é sempre melhor" | mesmo grupo (item 1) | 2026 | idem | ★★★★☆ `⚠ preprint` |

**Nota de aderência**: item 1 é o mais próximo do cenário AIPE — testa
justamente **demanda intermitente** (>40 mil séries reais, 5 datasets),
o mesmo regime da Bahia (Lumpy) e da maior parte do M5 pós-filtro (item
#24). Mas mede erro de PREVISÃO probabilística, não custo de política de
estoque — a métrica-alvo é diferente da nossa (CTI/NS/CTI_ajustado).

---

## 3. Quando pooling AJUDA vs. quando ATRAPALHA — o mecanismo

Este é o bloco mais relevante para explicar o resultado do item #34/#36.

### 3.1 Regularização e "cross-learning" (a favor do pooling)

Pooling reduz variância de estimação ao compartilhar padrões (sazonalidade,
promoções, ciclos regionais) entre séries — útil sobretudo pra séries
novas ou com poucos dados (*data-scarce*). Isso favorece pooling
justamente nas séries mais esparsas/intermitentes — o que **poderia**
explicar por que PIL/CappedBaseStock/DQN/SARSA (as políticas mais fracas,
mais sensíveis a pouco dado de treino) ganharam com `com_perfil` nos
itens #34/#36: são exatamente as políticas cuja calibração por série
sofre mais com janelas de treino curtas (17-19 ciclos).

### 3.2 Heterogeneidade e transferência negativa (contra o pooling)

> "Cluster-level pooling only reduces estimation variance when series
> within each group share truly similar dynamics. In more heterogeneous
> contexts, if clusters retain substantial heterogeneity, the model must
> accommodate divergent patterns" [...] "naive specialization can induce
> negative transfer."

— *Forecasting Multivariate Time Series under Predictive Heterogeneity:
A Validation-Driven Clustering Framework*, arXiv 2604.13748 (2026,
`⚠ preprint`, ★★★★☆ aderência — é sobre clustering pra pooling, aplicável
a qualquer alvo de agrupamento, não só previsão).

**Mecanismo proposto por esse grupo**: um framework *validation-driven*
decide QUANDO especializar (agrupar/pooling) vs. manter global, com
base em desempenho fora da amostra — e inclui um **mecanismo de fallback
sem vazamento** que reverte pro modelo global sempre que a especialização
falha em melhorar a validação. Isso é conceitualmente o que os itens
#34/#36/#37 fizeram na prática (comparar com/sem perfil e medir qual
vence, por política) — só que de forma A POSTERIORI (depois de rodar as
duas opções completas), não como parte de um pipeline de decisão
automática. Fica registrado aqui como direção de trabalho futuro: um
critério de fallback automático (por política, ou por política×perfil)
seria a extensão natural do achado atual.

### 3.3 Clustering baseado em desempenho preditivo, não em regras fixas

> Time series clustering based on **prediction accuracy** of global
> forecasting models — Knowledge-Based Systems, 2025 (Elsevier, revisado
> por pares, ★★★★☆ aderência).

Proposta central: agrupar séries pelo ERRO que um modelo global comete
nelas (não por características estatísticas fixas tipo ADI/CV²), e só
então decidir se compensa especializar. Isso é uma crítica direta,
independente, ao tipo de classificação usada pelo POD do AIPE
(`classify_operational_profiles`, regras fixas de ADI/CV²/burstiness/
zero_streak — ver `AJUSTES_INFRA` item #26): a literatura recomenda
validar o critério de agrupamento contra o alvo (aqui, CTI_ajustado),
não assumir que um critério estatístico da demanda por si só produz
grupos úteis pra treino.

### 3.4 Segmentação prática (ABC-XYZ) — mesma limitação, do lado da indústria

Literatura prática de segmentação (ABC-XYZ e variantes) parte do mesmo
princípio de agrupar por características de demanda pra aplicar
políticas/parâmetros diferenciados por segmento — mas mesmo aí a crítica
recorrente é que tratar todos os itens do mesmo segmento uniformemente
ainda esconde heterogeneidade real ("*a 5.000-SKU portfolio is not one
forecasting problem; it's several problems mixed together*" — fonte de
prática de mercado, não periódico revisado, citada aqui só como contexto,
`★★☆☆☆`). Reforça o mesmo ponto por um caminho diferente (prática de
segmentação de estoque, não ML): classificação fixa por características
observáveis da demanda tende a deixar heterogeneidade residual dentro do
grupo.

---

## 4. Política compartilhada (RL/otimização) entre múltiplos produtos

| # | Achado | Fonte | Ano | Veículo | Ader. |
|---|---|---|---|---|---|
| 4 | Retailers/produtos com mesmos parâmetros de custo e dinâmica de demanda têm problemas de decisão "estruturalmente simétricos" — justifica política RL compartilhada, mantendo o modelo compacto e escalável | Sultana, Meisheri, Baniwal, Nath, Ravindran, Khadilkar — *Reinforcement Learning for Multi-Product Multi-Node Inventory Management in Supply Chains* | 2020 | arXiv 2006.04037 | ★★★☆☆ `⚠ preprint` |
| 5 | Extensão publicada do item 4: RL escalável pra centenas de produtos via "meta-modelo" por produto — minimiza retreino quando o sistema muda, tratando explicitamente o trade-off entre 1 política por produto e 1 política compartilhada | Meisheri, Sultana, Baranwal, Baniwal, Nath, Verma, Ravindran, Khadilkar — *Scalable multi-product inventory control with lead time constraints using reinforcement learning* | 2022 | **Neural Computing and Applications** 34(3):1735–1757 | ★★★★☆ |

**Nota de aderência**: item 5 é o mais próximo, do lado de política (não
previsão): é peer-reviewed, aborda explicitamente a mesma pergunta do AIPE
("compartilhar ou não compartilhar o modelo entre produtos?"), com um
"meta-modelo" por produto como caminho intermediário entre os dois
extremos testados nos itens #34/#36 (perfil vs. global) — não testamos
essa terceira opção (meta-modelo por SÉRIE, não por perfil nem global),
fica como direção de trabalho futuro citável.

---

## 5. Síntese para a dissertação

1. **A pergunta "pooling ajuda?" não tem resposta universal na
   literatura** — depende de quão homogêneo é o grupo, do tamanho de
   amostra por série, e de qual sinal (erro de previsão vs. custo de
   política) está sendo otimizado. Os itens #34/#36/#37 do AIPE são
   consistentes com essa literatura: a resposta MUDA por política E por
   dataset (Bahia vs. M5), não é uma constante.
2. **O achado mais forte e replicado** (FixedInterval/VendorResponsive
   preferem `sem_perfil` nos dois datasets, estatisticamente significativo
   nos dois — ver `pooling_statistical_analysis.py`) é consistente com a
   literatura de heterogeneidade: essas são as políticas de limiar
   ADAPTATIVO (recalculam o limiar a cada ciclo com base no NS realizado)
   — MENOS dependentes de um treino específico de grupo, e por isso menos
   beneficiadas por especialização e mais beneficiadas pelo volume/
   diversidade extra do pool global (Seção 3.1, cross-learning).
3. **O POD do AIPE (perfil operacional) é uma classificação fixa por
   regras, não validada contra o alvo de custo** — a Seção 3.2/3.3 dá
   base bibliográfica direta pra essa ser a explicação central de por que
   pooling por perfil não bate pooling global de forma consistente:
   critério de agrupamento não teve seu propósito (reduzir heterogeneidade
   relevante pro CTI_ajustado) validado, só assumido.
4. **Direção de trabalho futuro, citável**: (a) clustering validado por
   desempenho de política (Seção 3.3) em vez de regras fixas de ADI/CV²;
   (b) mecanismo de fallback automático perfil→global por política (Seção
   3.2); (c) meta-modelo por série como terceiro ponto do espectro
   perfil/global (Seção 4, item 5).

---

## 6. Termos de busca usados (rastreabilidade)

- "global forecasting models vs local models pooling time series inventory demand"
- "pooled reinforcement learning inventory replenishment policy multiple products shared model"
- "demand classification clustering ABC-XYZ segmentation inventory policy heterogeneity within cluster"
- "when pooling hurts heterogeneous time series global model clustering before training benefit"
- "negative transfer multi-task learning inventory newsvendor cross-series learning intermittent demand"
- "Scalable multi-product inventory control with lead time constraints using reinforcement learning" (autores/venue)

Buscas feitas via WebSearch (motor não identificado nos resultados, sem
filtro de venue embutido — todo o filtro de qualidade/aderência acima foi
aplicado manualmente, mesmo critério de `estado_da_arte_politicas.md`).
