# Revisão inicial de contribuição e novidade

Data: 2026-10-14. Base: três artigos publicados do acervo local e documentos da qualificação/proposta. Trata-se de uma comparação dirigida, não uma revisão sistemática exaustiva. O catálogo contém registros alheios à revista, incluindo capítulos de livro com DOI `10.4324/...`; o total de registros não mede sozinho a cobertura editorial.

**Decisão inicial: revisão maior antes de tratar o texto como pronto para submissão.** Há um escopo empírico potencialmente defensável, mas a reivindicação de um seletor adaptativo validado ainda não está sustentada e existem versões incompatíveis dos resultados.

## Comparação bibliográfica rastreável

| Estudo e evidência primária local | O que já está estabelecido | Diferença possível neste trabalho | Exigência para sustentar a diferença |
|---|---|---|---|
| Zabraoui et al., *A comparative study of multi-algorithm optimization for inventory analytics in supply chains*, DOI [100154](https://doi.org/10.1016/j.sca.2025.100154). PDF `pdfs/2025/100154_A_comparative_study_of_multi_algorithm_optimization_for_inventory_analytics_in_supply.pdf`, pp.1–2,16–19; Algoritmo 1 e Tabela 13. | Benchmark multialgoritmo com Walmart M5; integração previsão/GA/RL; GA-DQN **e GA-PPO**; análise de custo/serviço e discussão de implantação seletiva de híbridos. | Avaliar a heterogeneidade das escolhas sob demanda varejista fortemente intermitente e quantificar se a seleção por perfis acrescenta utilidade operacional. | Não reivindicar novidade dos híbridos nem da modularidade. Mostrar ganho em relação à melhor política global e a regras de seleção simples sob informação disponível antes da avaliação. |
| Tadayonrad e Ndiaye, *A new key performance indicator model for demand forecasting in inventory management considering supply chain reliability and seasonality*, DOI [100026](https://doi.org/10.1016/j.sca.2023.100026). PDF `pdfs/2023/100026_A_new_key_performance_indicator_model_for_demand_forecasting_in_inventory_management_.pdf`, pp.1–2,12–13, Tabelas 11–12. | Critério de previsão ligado ao custo de excesso/ruptura, segurança considerando confiabilidade e sazonalidade; caso com 19.961 SKUs e 60 meses. | Escolha entre famílias de controle de inventário em vez de somente escolha de método preditivo e segurança. | Formular claramente a decisão escolhida e manter previsão como sinal auxiliar. Otimizar um KPI de custo ou usar dados reais, isoladamente, não é contribuição inédita. |
| Theunissen et al., *A topological analytics framework for preserving decision structure in supply chain inventory classification*, DOI [100229](https://doi.org/10.1016/j.sca.2026.100229). PDF `pdfs/2026/100229_A_topological_analytics_framework_for_preserving_decision_structure_in_supply_chain_i.pdf`, pp.1–2. | Classificação de inventário orientada à validade de critérios decisórios, distinguindo importância estatística de utilidade operacional; comparação de T-SAV com PCA/RFE/MI, bootstrap 10.000; conclusão explicitamente limitada ao caso. | Perfis de demanda podem ser julgados pelo desempenho de reposição obtido, e não apenas pela separação estatística. | Avaliar se PODs realmente mudam decisões e resultados. Não afirmar superar T-SAV: os problemas, critérios e dados não são equivalentes. |

As páginas são do PDF publicado. Resultados reportados pelos artigos são resultados dos respectivos autores, não verificações independentes feitas nesta revisão. O terceiro trabalho serve como posicionamento conceitual, não baseline obrigatório para uma decisão diferente.

## B1 — versões incompatíveis dos resultados impedem uma conclusão principal

Fontes locais conflitantes:

- `experiment/data/08_reporting/final_results_validation.md`, datado 2026-06-21: CTI puro EOQ 628,42 → perfil 589,62, redução 6,2%; NS 0,942 → 0,883; SA no perfil Sparse High Impact;12 políticas incluindo GA-PPO.
- `experiment/data/08_reporting/strategy/table_strategy_comparison.tex`: CTI **ajustado** 895,11 → 862,64, redução 3,6%; NS 0,942 → 0,938; redução do oráculo 50,7%. A legenda diz que o ajuste incorpora indisponibilidade/excesso, distinto do ajuste min-max/bullwhip descrito no relatório anterior.
- `docs/master_proposal/capitulos/resultados.tex`, discussão de PODs: EOQ dominante no perfil 116 séries e Fixed Interval no perfil 11 séries; narrativa e portfólio diferentes do relatório anterior.
- `docs/qualification_presentation/README.md` referencia essas fontes conjuntamente, portanto o README não resolve a divergência.

Correção: escolher um conjunto versionado de KPIs/configuração/código e regenerar todas as tabelas e interpretações. Identificar resultados históricos como tal. Não misturar custo puro, penalidade por serviço/excesso e critério min-max/bullwhip sob o nome genérico CTI ajustado. Critério de encerramento: tabela principal reproduzível de uma única fonte e manifesto de proveniência.

## B2 — seleção retrospectiva não prova recomendação para séries novas

A qualificação informa que o PSE não foi treinado/validado nacionalmente. É necessário auditar se existe qualquer avaliação de seletor em amostra independente. Escolher políticas por menor custo observado no próprio conjunto avaliado é análise retrospectiva; não mede generalização. O oráculo por série tampouco é estratégia implementável e não representa um limite matemático absoluto: é uma referência ex post dentro do portfólio e protocolo adotados.

Correção: no escopo atual, tratar como estudo de heterogeneidade e análise retrospectiva de seleção; colocar o seletor aprendido no trabalho futuro. Para reivindicar PSE operacional, treinar seleção/perfis em dados de treino, congelar decisões e avaliar em séries/tempo externos, com tuning separado. Critério: separação documentada e desempenho de teste acompanhado de incerteza.

## M1 — contribuição incremental deve exceder um benchmark maior

Adicionar políticas, usar varejo brasileiro, integrar scripts e executar em GPU melhora abrangência e reprodutibilidade, mas não estabelece sozinho contribuição metodológica. A pergunta defensável é se perfis de demanda permitem uma escolha de reposição útil além da política única quando custo, serviço e estabilidade são explicitados.

Exigir: melhor política global no mesmo objetivo; classificação macro ADI×CV²; política global SA quando relevante; regra simples por perfil; seleção aleatória como controle negativo se for reivindicado aprendizado; ablação das características/refinamento POD. Não exigir todos os controles para uma nota puramente retrospectiva, mas calibrar a força da conclusão ao protocolo efetivo.

## M2 — o ganho de custo pode ser apenas troca por menor serviço

Na versão histórica, 6,2% de redução acompanha queda de 5,9 pontos percentuais de serviço. O limiar 0,70 é uma preferência operacional, não uma garantia de superioridade. Viabilidade no serviço médio de um grupo não assegura viabilidade de cada série.

Exigir custo-serviço pareado, distribuição por série, taxa de séries abaixo do limiar e sensibilidade em limiares plausíveis. Para afirmar vantagem sem perda, usar comparações de serviço equivalente ou apresentar explicitamente a fronteira e o compromisso. Perfis n=18/n=11 são exploratórios; diferenças entre perfis precisam de incerteza e não apenas médias.

## M3 — alegações da revisão de literatura excedem a evidência lida

`correlatos.tex` argumenta que 38 ciclos são razão estrutural para degeneração do RL, que referências usam CV<1 e que o M5 foi avaliado como validação externa. Essas afirmações precisam ser conferidas no protocolo e artefatos, não aceitas a partir da narrativa. Um resultado ruim pode refletir implementação, recompensa, treinamento ou orçamento; episódios em simulação podem reutilizar trajetórias sem constituir novas observações reais. O teorema No Free Lunch não prova que seleção por perfil melhorará este dataset.

Correção: evitar causa universal para falhas de RL; separar futura avaliação M5 de experimentos concluídos; documentar definições de CV, demandas nãozero, orçamento e mudanças posteriores no código. Não usar frases de prioridade como primeiro framework sem pesquisa primária mais ampla.

## Escopo recomendado

Título provisório adequado: *Demand-profile heterogeneity and retrospective replenishment-policy selection in intermittent retail demand*. Contribuição: (1) protocolo comparável e transparente de políticas conhecidas; (2) caracterização da heterogeneidade dentro do regime Lumpy; (3) quantificação retrospectiva dos compromissos custo-serviço e dos limites da seleção por perfil. Esse escopo permite reportar resultados nulos ou modestos honestamente; não requer apresentar uma arquitetura aprendida como concluída.

Antes da submissão: resolver B1; alinhar título/resumo/conclusões a B2; demonstrar utilidade comparada ao melhor baseline coerente; quantificar incerteza/serviço por série; registrar disponibilidade de código/dados e limites de reprodução. Não há garantia de aceitação baseada no estilo dos artigos publicados.
