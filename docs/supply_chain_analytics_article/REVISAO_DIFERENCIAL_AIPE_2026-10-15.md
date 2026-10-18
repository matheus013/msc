# Oportunidades de contribuição para o AIPE

Revisão dirigida em 15/10/2026. Resultado: **a seleção contextual, isoladamente, não sustenta uma alegação forte de originalidade. A hipótese mais promissora é selecionar políticas por benefício operacional esperado, com decisão explícita de manter uma referência quando a evidência para substituí-la for insuficiente.** Essa hipótese ainda precisa ser implementada e validada; não é resultado demonstrado pelo piloto.

## Escopo e rastreabilidade

Foram executadas 12 consultas bibliográficas no Crossref: oito sobre classificação/seleção de políticas, seleção de algoritmos, decisão orientada ao objetivo e controle de risco; quatro complementares sobre recomendação contextual, incerteza, melhoria conservadora e regret em reposição. As primeiras oito produziram 113 DOIs distintos, incluindo muitos resultados sem aderência. Esse número mede descoberta bibliográfica, não artigos incluídos nem textos integralmente revisados. Consultas, respostas, datas e erros estão em `experiment/data/literature/aipe_opportunities_2026-10-15/`. O coletor preserva respostas por consulta e interrompe a coleta em caso de limitação de acesso.

A seleção priorizou proximidade com a decisão do AIPE e capacidade de contradizer uma alegação de novidade. Houve leitura dirigida de resumos, formulações e seções pertinentes: dois PDFs da revista já disponíveis, cinco novos PDFs abertos e o HTML integral do estudo de Francuz e Bányai. Outro trabalho foi consultado pelo resumo dos autores. Dois antecedentes de estoque tiveram apenas metadados reconfirmados nesta rodada; suas descrições metodológicas detalhadas exigem texto completo. Não se apresenta revisão sistemática, busca exaustiva ou comprovação de inexistência de trabalhos equivalentes. O Crossref tem cobertura e ordenação imperfeitas; resultados negativos de busca não demonstram lacuna.

Cinco novos PDFs foram preservados em `docs/references/aipe_opportunities/2026-10-15/`, com manifesto de origem. O PDF MDPI retornou HTTP 403, mas o texto completo foi lido no navegador. ScienceDirect e SSRN apresentaram bloqueios; não foram contornados. Nenhum PDF anterior foi substituído. Os manuscritos e os experimentos permanecem inalterados por esta revisão.

## Matriz dos trabalhos decisivos

| Trabalho e evidência nesta rodada | O que já cobre | Implicação para o AIPE |
|---|---|---|
| Mohammaditabar, Ghodsypour e O'Brien (2012), [Inventory control system design by integrating inventory classification and policy selection](https://doi.org/10.1016/j.ijpe.2011.03.012). Metadados. | O próprio objeto publicado integra classificação e seleção de políticas. | Não reivindicar como inédita a integração genérica. Consultar o texto completo antes de afirmar diferenças específicas do algoritmo ou protocolo. |
| Sheikh-Zadeh, Rossetti e Scott (2021), [Performance-based inventory classification methods for large-scale multi-echelon replenishment systems](https://doi.org/10.1016/j.omega.2020.102276). Metadados. | Antecedente explícito de classificação baseada em desempenho. | Evitar apresentar classificação pelo desempenho como nova. O contexto multi-echelon também é mais amplo que o ambiente independente atual do AIPE. |
| Kerschke et al., [Automated Algorithm Selection: Survey and Perspectives](https://arxiv.org/abs/1811.11597), versão dos autores aceita para publicação. PDF, leitura dirigida. | Seleção por instância, atributos, seletor global, oráculo por instância, fechamento da lacuna entre ambos, seleção sensível a custos e algoritmos de reserva. | XGBoost para escolher uma família e comparar com o oráculo são mecanismos consolidados. O percentual de fechamento da lacuna não deve ser anunciado como nova métrica. |
| Zabraoui et al. (2025), [A comparative study of multi-algorithm optimization for inventory analytics in supply chains](https://doi.org/10.1016/j.sca.2025.100154). PDF local, leitura dirigida. | Comparação no M5 e combinação GA–DQN com previsão e controle de estoques. | M5, portfólio heterogêneo e GA–DQN não distinguem o AIPE por si. A diferença a estudar está na camada de decisão entre famílias, não na invenção do híbrido. |
| Francuz e Bányai (2026), [Metaheuristic-Based Model Selection Framework for EOQ and Inventory Policies Using Machine Learning and Multi-Objective Optimization](https://doi.org/10.3390/a19050415). HTML integral, especialmente §§2.2, 3.1, 3.3–3.4 e 5. | Otimização conjunta da família e parâmetros de (s,Q), (s,S), (R,S), custo/serviço/estoque, NSGA-II e random forest como aproximação do simulador. Experimentos com demanda sintética sazonal e backlog. | É um comparador conceitual muito próximo. O ML aproxima resultados das simulações; não é um classificador direto de famílias para novas séries. AIPE pode investigar transferência entre séries/lojas, vendas perdidas e evidência para substituir a referência; essas diferenças precisam aparecer na formulação e no teste. Não alegar novidade apenas por integrar seleção, múltiplos objetivos ou aceleração. |
| Theunissen, Alam e Sajjad (2026), [A topological analytics framework for preserving decision structure in supply chain inventory classification](https://doi.org/10.1016/j.sca.2026.100229). PDF local, especialmente resumo e introdução. | Validação estrutural de critérios de classificação, distinguindo destaque estatístico de utilidade decisória. | Não atribuir significado operacional a um perfil apenas porque foi calculado ou possui importância preditiva. Seu alvo é validação de critérios, diferente da recomendação de família de reposição. Não reproduzir a abordagem topológica só para acrescentar complexidade. |
| Elmachtoub e Grigas, [Smart “Predict, then Optimize”](https://arxiv.org/abs/1710.08005), DOI publicado [10.1287/mnsc.2020.3922](https://doi.org/10.1287/mnsc.2020.3922). PDF, formulação e introdução. | Aprendizado orientado ao erro de decisão e à estrutura do objetivo, em vez de apenas erro preditivo. | Penalizar escolhas segundo sua perda operacional tem fundamento, mas não constitui invenção geral do AIPE. A formulação SPO+ não se transfere automaticamente ao simulador de estoque e à restrição de serviço. |
| Geifman e El-Yaniv (2019), [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html). PDF e resumo dos autores. | Predição seletiva com rejeição e compromisso entre risco e cobertura. | Recusar uma recomendação de baixa confiança já é estabelecido. O AIPE precisa definir risco em custo e atendimento e medir a fração de decisões em que intervém. |
| Mozannar e Sontag (2020), [Consistent Estimators for Learning to Defer to an Expert](https://proceedings.mlr.press/v119/mozannar20b.html). Resumo dos autores. | Aprendizado de quando decidir e quando delegar, com redução a aprendizado sensível a custos. | Um mecanismo de recuo é parte da decisão aprendida. Uma política de referência não equivale automaticamente ao especialista modelado nesse trabalho; não importar suas garantias. |
| Laroche, Trichelair e Tachet des Combes (2019), [Safe Policy Improvement with Baseline Bootstrapping](https://proceedings.mlr.press/v97/laroche19a.html). PDF, leitura dirigida. | Recuo à política de referência em regiões com incerteza, no contexto de RL em lote, com garantias sob hipóteses próprias. | Manter a referência quando faltam dados não é novo em geral. AIPE escolhe famílias por série, enquanto esse trabalho restringe melhoria de uma política em MDP. O limiar atual do classificador não possui essas garantias. |
| Angelopoulos et al., [Conformal Risk Control](https://arxiv.org/abs/2208.02814), versão consultada v4/2025. PDF, formulação e resultado principal. | Controle de risco esperado com perdas limitadas e estrutura monotônica, sob permutabilidade apropriada. | Pode fundamentar uma extensão futura, mas custo não limitado, dependência temporal e seleção entre políticas exigem tratamento específico. Não garante fill rate mínimo em toda série nem robustez automática a mudança de distribuição. |

As interpretações de oportunidade abaixo são inferências da comparação entre essas fontes e os resultados locais, não conclusões atribuídas a um único artigo.

## Onde existe espaço para uma contribuição mais forte

### 1. Prioridade principal: decidir se vale substituir a referência

Pergunta: **em séries heterogêneas e com pouco histórico, quando a evidência disponível justifica trocar uma política de referência por uma recomendação contextual?**

Manter as duas abordagens de PSE: por perfil informado e preditiva por características. Acrescentar uma regra de decisão comum: a recomendação só substitui a referência quando apresenta benefício operacional validado e risco de deterioração aceitável. A referência pode ser global ou por perfil, mas a escolha entre elas deve ocorrer na validação interna, sem escolher retrospectivamente a mais favorável no teste.

O componente a desenvolver deve estimar diferenças de custo e de serviço em relação à referência, não interpretar o maior escore de classe como probabilidade de sucesso. Um critério candidato é exigir evidência de ganho de custo acima de uma margem e ausência de deterioração de serviço acima de uma tolerância. Margens, tolerâncias, unidade de avaliação e tratamento dos casos inviáveis precisam ser definidos antes do teste. Intervalos calculados em poucos grupos não fornecem garantia universal; chamar o procedimento inicialmente de conservador em termos empíricos.

O diferencial potencial é uma formulação e validação de seleção de famílias para reposição com vendas perdidas, baixo suporte e restrições de atendimento, explicitando o valor e o risco de intervir. Seleção, rejeição e recuo são os fundamentos herdados. Apenas combiná-los com XGBoost não basta: precisamos demonstrar que o mecanismo responde a um problema operacional relevante e produz ganho ou reduz deteriorações em dados independentes.

### 2. Substituir o rótulo rígido pela consequência da escolha

Hoje, trocar uma vencedora por outra quase empatada conta como erro, assim como escolher uma política muito pior. Além disso, a identidade da vencedora muda com janela e semente. Proponho comparar três representações, mantendo o mesmo conjunto de características e orçamento:

1. Classificação da família vencedora, como controle atual.
2. Estimativa de custo e serviço por família, seguida de decisão com restrição de atendimento.
3. Aprendizado de diferenças entre famílias ou de preferência sensível à perda operacional.

Políticas praticamente equivalentes podem receber tratamento de empate, com tolerância definida na validação. Restrições de serviço não devem desaparecer dentro de uma penalidade escolhida para favorecer um método. Comparações devem avaliar tanto perda monetária quanto violação de atendimento. O fundamento de aprendizado orientado à decisão já existe; a contribuição deve estar na formulação para o problema e nos resultados, não no nome do classificador.

### 3. Contribuição de diagnóstico: localizar a perda recuperável

Separar três limitações ajuda a decidir onde investir: erro do seletor quando havia família viável; insuficiência do portfólio quando um planejamento com informação perfeita seria viável; e inviabilidade física nas condições iniciais. Demanda de teste zero fica separada.

Essa decomposição pode sustentar uma análise útil para gestores, mas não deve ser anunciada como nova teoria sem pesquisa adicional. Um ótimo com informação perfeita é um limite retrospectivo, não uma política executável. A decomposição do serviço também não é automaticamente uma decomposição aditiva do custo.

Para custo, o fechamento da lacuna entre referência e oráculo pode ser apresentado como `(C_ref − C_AIPE)/(C_ref − C_oráculo)`, quando o denominador for positivo, com a mesma população, objetivo e regra de viabilidade. Resultados negativos e acima de um devem ser investigados e explicados, sem truncamento oportunista. Em avaliação finita, uma escolha global estimada na validação pode perder para o AIPE ou mesmo diferir da melhor família global retrospectiva; esses conceitos precisam de nomes distintos. Não misturar médias condicionadas a conjuntos de séries diferentes.

### 4. Eficiência computacional como resultado secundário

Medir custo total de produzir o seletor, calibrar as políticas candidatas e atender novas séries. Inferência rápida não significa pipeline barato se todos os controladores forem treinados para toda série antes da escolha. A economia só será demonstrada se o modo de uso realmente evitar ajustes desnecessários ou amortizar seu custo. O artigo de Francuz e Bányai já cobre aproximação do simulador e redução de avaliações; portanto, aceleração genérica não é argumento suficiente de novidade.

## Confronto com o piloto atual

Fonte local: `experiment/data/08_reporting/research_v2/m5_pse_pilot30_v2_2026-10-15/diagnostics_v1/report.md`.

- Há 27 exemplos de treinamento por fold para até sete classes; o seletor reproduz os rótulos do treino, mas acerta 19/90 rótulos nas lojas excluídas. Não é base suficiente para defender ganho generalizável do preditivo.
- A vencedora nominal da validação persiste no teste em 29/90 casos; sementes alteram a vencedora em 39/90. São sinais de instabilidade, não estimativas precisas de probabilidade.
- O preditivo apresenta custo médio 1.238,30 e serviço 67,35%, contra 1.086,97 e 71,46% da seleção global. O recuo exploratório chega a 1.087,26 e 69,96%: melhora o preditivo, mas não supera a referência global.
- Com e sem perfil, as 90 decisões coincidem. Como os perfis derivam das próprias características e a amostra é concentrada, isso não prova que informação contextual independente seja inútil. Um perfil determinístico das mesmas entradas não adiciona informação nova, embora possa alterar a facilidade de aprendizagem por um modelo finito.
- Entre os 90 casos, há 21 erros de seleção com alternativa viável, 15 limitações do portfólio, 11 inviabilidades físicas e quatro demandas zero; 39 atendem ao alvo. Portanto, ampliar o classificador não resolve toda falha de atendimento.

Esses resultados motivam a hipótese proposta; não a comprovam. O piloto já observado é conjunto de desenvolvimento. A regra exploratória de recuo não pode ser avaliada como hipótese confirmatória sobre o mesmo teste.

## Protocolo mínimo para testar o diferencial

1. **Harmonizar o problema.** Mesmas convenções de custo, estoque inicial, lead time, restrições e calibração entre bases. Resolver a diferença de calibração de CappedBaseStock entre a rodada Bahia e o piloto M5 antes de atribuir diferenças às bases. Uniformizar também o objetivo: hoje a rotulagem individual e a associação global/por perfil usam restrições de serviço em níveis distintos.
2. **Separar desenvolvimento e confirmação.** Reservar séries e períodos ainda não usados para escolher o método. Validação interna por loja e tempo para hiperparâmetros, margens, referência e recuo; teste externo temporal preservado. As origens de uma série não são amostras independentes.
3. **Comparar o necessário.** Global, PSE por perfil, vencedora individual da validação, preditivo rígido, preditivo orientado à consequência e versão conservadora. Os oráculos ficam como limites retrospectivos. Aumentar candidatos só se a análise mostrar insuficiência relevante do portfólio.
4. **Medir benefício e deterioração.** Diferença pareada de custo, serviço médio, fração de séries abaixo do alvo, frequência e magnitude de piora em relação à referência, fração de intervenções e desempenho nas intervenções. Avaliar em população comum, com demandas zero identificadas.
5. **Testar mecanismos.** Com/sem perfil, com/sem recuo, rótulo rígido versus consequência, curvas de aprendizagem e estabilidade entre janelas/sementes. Reportar suporte por perfil/classe. Perfis raros podem ser enriquecidos em análise específica; a avaliação representativa deve manter pesos/população declarados.
6. **Quantificar incerteza e esforço.** Intervalos pareados por unidades compatíveis com a dependência, efeitos por loja/perfil e tempo completo do pipeline. Dez lojas limitam a precisão. Não transformar três sementes em três vezes mais séries.

Antes de uma execução grande, o principal teste de viabilidade é verificar se existe heterogeneidade útil e ganho recuperável estável: se referência e oráculo forem próximos, ou a família vencedora não generalizar entre períodos, pouca vantagem estará disponível para qualquer seletor. Aumentar amostra sem mudar esse diagnóstico pode apenas confirmar a ausência de ganho.

## Alegações recomendadas e alegações a evitar

**Hipótese de contribuição:** “O AIPE investiga seleção contextual de famílias de reposição com decisão explícita de intervenção, orientada a custo e atendimento, avaliando quando a recomendação deve substituir uma referência e distinguindo erros de escolha de limitações do portfólio e do sistema.”

Essa frase descreve a pesquisa proposta. O resumo final do artigo deverá dizer o que foi efetivamente implementado e demonstrado, incluindo eventual resultado negativo.

Evitar: primeiro sistema de seleção contextual; primeira integração de classificação e política; novidade por usar M5 ou GA–DQN; garantia de serviço por confiança do XGBoost; superioridade já estabelecida; ótimo teórico como desempenho executável; redução computacional baseada apenas em tempo de inferência.

## Pendências bibliográficas que limitam a novidade

- Obter os textos completos dos antecedentes de 2012 e 2021 para comparação de objetivos, seleção e avaliação, sem inferir detalhes pelo título.
- Rastrear citações dos antecedentes e do trabalho de 2026 em uma base de citações. Esta rodada não cobre todas as citações posteriores nem literatura não indexada.
- A busca encontrou o registro SSRN **“Forecasting for Inventory Decisions: A Decision-regret Benchmark for Perishability-aware Multi-Echelon Retail Replenishment using the M5/Walmart Data”**, DOI [10.2139/ssrn.7051299](https://doi.org/10.2139/ssrn.7051299), e **“Projection-Based Safe Reinforcement Learning for Capacity-Constrained Multi-Echelon Inventory Control under Non-Stationary Demand and Stochastic Lead Times”**, DOI [10.2139/ssrn.7175419](https://doi.org/10.2139/ssrn.7175419). Os metadados consultados indicam apenas 2026 e o acesso ao SSRN foi bloqueado. São alertas de proximidade, não evidência de publicação anterior à data de corte, qualidade revisada por pares ou equivalência metodológica. Verificar versão, data e texto antes de incluí-los como antecedentes confirmados. Não afirmar ausência de trabalhos sobre regret no M5 ou RL conservador em estoques.
- O resultado “Safe Reinforcement Learning with Contextual Information”, DOI 10.2139/ssrn.4583667, é sobre gestão de comorbidades segundo os metadados; não foi tratado como estudo de estoques.

**Decisão recomendada:** concentrar o próximo desenvolvimento no valor da intervenção contextual, mantendo as duas abordagens de PSE. Usar diagnóstico de limites e eficiência como sustentação. Se a formulação conservadora não melhorar a relação custo/serviço ou reduzir deteriorações em avaliação independente, não alegar superioridade do AIPE; relatar as condições em que contextualizar compensa e em que a referência permanece preferível.
