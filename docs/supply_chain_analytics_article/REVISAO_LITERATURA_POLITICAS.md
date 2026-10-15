# Revisão dirigida para selecionar políticas e algoritmos

Data da busca: **14/10/2026**, horário de São Paulo. Estado: seleção fundamentada para validação, sem retomada dos experimentos.

## Resultado da revisão

A literatura **não sustenta eliminar métodos apenas por correlação de custos ou suposta popularidade**. O critério mais defensável é a aderência ao modelo de estoque, a cobertura de mecanismos distintos, a qualidade dos comparadores e a possibilidade de testar a contribuição de cada componente. Não foi feita uma bibliometria capaz de estabelecer qual algoritmo é universalmente o mais popular em 2026.

O modelo do projeto é de estoque multiperíodo, revisão por ciclo, lead time positivo, vendas perdidas, custo fixo de pedido e custo de posse. O estado inclui pedidos em trânsito. Bahia e M5 são painéis de vendas observadas, não observações diretas de demanda perdida. Essas características orientaram o julgamento de aplicabilidade.

**Portfólio candidato para a comparação principal:** EOQ adaptado, (s,S), CappedBaseStock, GA, PPO e GA-DQN. **DQN isolado é controle obrigatório da ablação do híbrido**, mesmo fora da tabela principal. A formulação e a escala da ablação devem ser definidas antes dos resultados; no subconjunto usado para essa comparação, todos os seus métodos precisam usar as mesmas séries, origens e condições de teste.

Esse resultado corrige a proposta provisória de cinco métodos: restaura (s,S), desloca Newsvendor para comparação auxiliar e mantém GA-DQN como hipótese experimental. A redução passa de 12 para seis entradas principais, sem apresentar os métodos retirados como estatisticamente inferiores. O ganho do híbrido ainda precisa ser demonstrado nas nossas condições.

## Método e alcance

Revisão dirigida, com busca por título e problema em páginas de editoras, repositórios dos autores e arXiv; referências teóricas e empíricas foram encadeadas a partir de trabalhos de controle de estoques. Termos incluíram `lost-sales inventory`, `lead time`, `fixed ordering cost`, `capped base-stock`, `inventory PPO DQN`, `simulation optimization inventory genetic algorithm` e o título de Zabraoui.

Foram selecionadas **16 fontes**: um livro, estudos e revisões de estoques, dois fundamentos de algoritmos e preprints identificados como tais. Oito textos completos ficaram disponíveis localmente: sete novas cópias abertas e o PDF de Zabraoui já existente. Foi feita leitura dirigida de resumo, modelo, método e/ou seções pertinentes à decisão; não se afirma leitura integral de cada apêndice. Para outras fontes, a evidência usada está limitada a resumos e metadados de editoras ou repositórios. Não se trata de revisão sistemática exaustiva, contagem de todas as publicações ou meta-análise de desempenho.

Foram priorizados: aderência a vendas perdidas e lead time; explicitação de custos; presença de referências clássicas; comparação empírica com políticas fortes; transparência do desenho; e diferenças entre aprendizado, política de reposição e calibração de parâmetros. Resultados de jogos ou aplicações fora de estoques não foram tratados como evidência de superioridade no nosso domínio.

## Matriz de evidências

| ID e fonte primária | Evidência acessada | Contribuição à decisão e limite de aplicação |
|---|---|---|
| S01 — Axsäter, *Inventory Control*, 3ª ed., 2015. [Editora](https://link.springer.com/book/10.1007/978-3-319-15729-0) | Apresentação editorial e estrutura do livro | Organiza lote econômico, pontos de pedido, revisão periódica e demanda estocástica. Referência para distinguir famílias; não fornece um ranking empírico dos nossos métodos. |
| S02 — Zipkin, 2008, *Old and New Methods for Lost-Sales Inventory Systems*. [Artigo](https://pubsonline.informs.org/doi/10.1287/opre.1070.0471) | Resumo da editora | Mostra que heurísticas plausíveis em sistemas com backlog não são automaticamente boas em vendas perdidas. Favorece comparadores específicos do problema. |
| S03 — Xin, 2021, *Understanding the Performance of Capped Base-Stock Policies in Lost-Sales Inventory Models*. [Artigo](https://pubsonline.informs.org/doi/10.1287/opre.2020.2019) | Resumo e metadados da editora; publicação online em 2020 | Política de dois parâmetros que conecta base-stock e pedido constante, com análise assintótica e avaliação numérica. Fundamenta CappedBaseStock; não permite transferir garantias automaticamente às nossas hipóteses de custo e demanda. |
| S04 — Jalali e Van Nieuwenhuyse, 2015, *Simulation optimization in inventory replenishment: a classification*. [Repositório dos autores](https://documentserver.uhasselt.be/handle/1942/30466?mode=simple) | Resumo institucional e metadados | Revisão de aplicações de 1998–2013 destaca a presença de GA em otimização por simulação de estoques. Justifica um representante GA, sem provar que GA supera SA ou lidera a popularidade atual. |
| S05 — Ban e Rudin, 2019, *The Big Data Newsvendor*. [Artigo](https://pubsonline.informs.org/doi/10.1287/opre.2018.1757) | Texto completo do manuscrito dos autores; online em 2018 | Usa informação contextual para decisões de newsvendor. É contribuição distinta, não duplicação do modelo simples; nossa adaptação multiperíodo não é uma reprodução integral do trabalho. |
| S06 — Boute et al., 2022, *Deep reinforcement learning for inventory control: A roadmap*. [Artigo](https://www.sciencedirect.com/science/article/pii/S0377221721006111) | Texto completo da versão publicada em repositório institucional | Expõe decisões de modelagem, implementação e esforço de ajuste de DRL. Apoia exigir comparadores fortes, aderência do estado e transparência computacional. |
| S07 — Gijsbrechts et al., 2022, *Can Deep Reinforcement Learning Improve Inventory Management?* [Artigo](https://pubsonline.informs.org/doi/abs/10.1287/msom.2021.1064) | Resumo e metadados da editora | Avalia A3C em vendas perdidas, duas fontes e múltiplos elos; encontra desempenho comparável a heurísticas fortes após ajuste. Não é evidência específica de que PPO ou DQN dominam o problema. |
| S08 — Geevers, van Hezewijk e Mes, 2024, *Multi-echelon inventory optimization using deep reinforcement learning*. [Artigo](https://link.springer.com/article/10.1007/s10100-023-00872-2) | Texto completo aberto; online em 2023 | PPO em ações contínuas, sistemas multi-echelon e backlog. Sustenta sua presença na área, mas difere do nosso ambiente de ações discretas, um elo e vendas perdidas. |
| S09 — Dehaybe, Catanzaro e Chevalier, 2024, *Deep Reinforcement Learning for inventory optimization with non-stationary uncertain demand*. [Artigo](https://www.sciencedirect.com/science/article/pii/S0377221723007646) | Resumo e apresentação da editora | Examina DRL com previsão, não estacionariedade, custo fixo e lead time. Ressalta o papel de treinar para generalizar; não valida a necessidade de treinar uma rede nova por SKU. |
| S10 — Stranieri et al., 2025, *Classical and Deep Reinforcement Learning Inventory Control Policies for Pharmaceutical Supply Chains with Perishability and Non-Stationarity*. [Preprint](https://arxiv.org/abs/2501.10895) | Texto completo do preprint | Compara OUT, PIL e PPO em aplicação farmacêutica. PPO não supera as políticas clássicas em todos os experimentos; contexto e custo computacional importam. O status consultado é preprint. |
| S11 — Zabraoui et al., 2025, *A comparative study of multi-algorithm optimization for inventory analytics in supply chains*. [DOI](https://doi.org/10.1016/j.sca.2025.100154) | PDF completo já preservado, especialmente §§4.7–4.8 e Tabela 13 | Motiva GA-DQN como hipótese e comparação com componentes isolados. O horizonte e a implementação não coincidem automaticamente com os nossos. |
| S12 — van Jaarsveld e Arts, 2024, *Projected Inventory-Level Policies for Lost Sales Inventory Systems*. [Artigo](https://pubsonline.informs.org/doi/10.1287/opre.2021.0032) | Texto completo do manuscrito dos autores, versão arXiv v4 | PIL é uma referência adicional relevante para vendas perdidas e lead time, com hipóteses explícitas. Sua existência impede chamar o nosso conjunto reduzido de cobertura exaustiva do estado da arte. |
| S13 — Wang, Peng e Yang, 2022, *Solving Inventory Management Problems through Deep Reinforcement Learning*. [Artigo](https://journal.hep.com.cn/jossase/EN/10.1007/s11518-022-5544-6) | Resumo da editora e metadados Crossref | Estuda Double DQN em vendas perdidas, lead time e custos incluindo pedido fixo. Evidência de aderência de DQN ao domínio, contrária à sua exclusão por suposta irrelevância geral. |
| S14 — Schulman et al., 2017, *Proximal Policy Optimization Algorithms*. [Preprint original](https://arxiv.org/abs/1707.06347) | Texto completo | Define PPO como método de gradiente de política. Fundamenta o algoritmo, mas seus benchmarks originais não provam superioridade em estoques. |
| S15 — van Hasselt, Guez e Silver, 2016, *Deep Reinforcement Learning with Double Q-learning*. [Manuscrito AAAI](https://arxiv.org/abs/1509.06461) | Texto completo; preprint de 2015 | Fundamenta Double DQN, com desacoplamento de seleção e avaliação da ação. Trata mecanismo diferente de PPO; não é prova de vantagem no nosso painel. |
| S16 — Liu et al., 2024, *Reinforcement Learning with Intrinsically Motivated Feedback Graph for Lost-sales Inventory Control*. [Preprint](https://arxiv.org/abs/2406.18351) | Resumo dos autores | Discute eficiência amostral e observação incompleta da demanda em vendas perdidas. Reforça que volume de vendas observado não mede diretamente a demanda que foi perdida. |

## Decisões por família

### EOQ, (s,S) e CappedBaseStock não são o mesmo controlador

Na implementação atual, EOQ adaptado encomenda um lote fixo Q quando a posição de estoque cruza um ponto de pedido. (s,S) encomenda a diferença até S quando cruza s. CappedBaseStock encomenda até o alvo, sujeito a um teto de quantidade. Essas diferenças existem mesmo quando os custos obtidos são correlacionados.

S01–S03 sustentam distinguir o problema de controle e as hipóteses de cada família. Minha recomendação é usar EOQ como referência simples, (s,S) como comparação de reposição multiperíodo e CappedBaseStock como referência mais diretamente relacionada a vendas perdidas. Essa composição é uma decisão de desenho apoiada na literatura, não um teorema de que esses três métodos sejam os melhores.

Os resultados históricos ajudam a encontrar comparações úteis, mas não estabelecem equivalência: a reanálise encontrou custo e serviço iguais entre EOQ e sS em 91,9% das séries M5 antigas; nos 171 históricos com pelo menos três ciclos positivos no treino, essa igualdade desaparece. Portanto, cold start contaminava a evidência de redundância.

### Newsvendor e sua extensão contextual ficam como auxiliares

O problema clássico de período único e a adaptação implementada não coincidem com nosso estoque que transita entre vários ciclos, tem pedidos em trânsito e custo fixo de emissão. S05 confirma que acrescentar características contextuais é uma mudança substantiva, não mera renomeação.

O código do projeto usa uma heurística `mu + z*sigma` acionada por estoque, não a solução geral de um newsvendor contextual nem uma política multiperíodo comprovadamente ótima. Para reduzir o texto principal, essas versões passam ao material auxiliar. Isso não invalida suas vitórias retrospectivas nem justifica apagá-las.

### GA permanece; SA pode sair por escopo, não por inferioridade

O papel de GA é calibrar parâmetros de uma regra de reposição, enquanto PPO aprende uma decisão dependente do estado. Não devem ser apresentados como novas famílias físicas de estoque equivalentes às regras clássicas. S04 dá respaldo para escolher GA como representante consolidado de otimização por simulação. A seleção de apenas um otimizador é defensável para concentrar o artigo, com uma ressalva explícita: nossos resultados históricos incluem situações em que SA era vantajoso. Não concluiremos superioridade geral de GA sem testar essa proposição.

### PPO e DQN são distintos; DQN é necessário para avaliar o híbrido

S08 e S10 apoiam manter PPO como comparação de DRL. S13 é particularmente próximo dos nossos custos, lead time e vendas perdidas, e usa Double DQN. S14–S15 explicam que os mecanismos de aprendizado são diferentes. Assim, não existe fundamento para chamar DQN de duplicata de PPO ou afirmar que PPO é universalmente mais popular e adequado.

PPO fica no comparativo principal; DQN permanece no desenho de ablação GA / DQN / GA-DQN. Os resultados dessa ablação só podem ser comparados nas mesmas unidades e condições. Caso se use uma coorte menor para a ablação, sua seleção deve ocorrer por critérios anteriores ao teste e sua conclusão deve ficar restrita a essa coorte. A avaliação principal de todas as séries elegíveis continua sendo um objetivo separado.

## O que Zabraoui efetivamente acrescenta

Em S11, a Tabela 13 apresenta GA-DQN com custo 38.500, serviço 0,9400 e ruptura 0,0600; GA-PPO tem custo 40.851,70, serviço 0,9072 e ruptura 0,2589. A análise usa 365 períodos diários e combina previsão, busca genética e DRL. Isso motiva testar o híbrido. Não permite transportar os valores ou a significância reportada para nossos ciclos, custos e bases.

Nosso código precisa especificar como o GA orienta o DQN e distinguir demonstrações, gatilho de reposição e piso de quantidade. Não basta chamar toda combinação de reprodução exata do método publicado. VendorResponsive e FixedInterval podem sair da comparação principal por foco; retirar todos os métodos só porque aparecem em Zabraoui seria igualmente injustificado.

## Condições antes de validar o portfólio por novos resultados

As condições abaixo são conclusões da confrontação entre a literatura e o código do projeto, não resultados experimentais já obtidos:

1. **Correção do híbrido:** `prepopulate_from_ga` registra o índice discreto mais próximo de Q, mas executa Q contínuo. A transição precisa corresponder à ação efetivamente executada. As demonstrações também precisam ter tratamento de recompensa/penalidade terminal compatível com a definição de treinamento escolhida.
2. **Treino e execução:** explicitar se o gatilho GA e o piso de quantidade fazem parte do treinamento ou são impostos apenas na execução. Não atribuir todo efeito observado à inicialização se a política de decisão também mudou.
3. **Controles e orçamento:** comparar GA, DQN e GA-DQN de forma pareada; registrar busca GA, demonstrações, transições, atualizações e tempo total. Para afirmar ganho específico da inicialização, incluir controle sem demonstrações e com orçamento equivalente, e não apenas comparar um agente mais treinado com outro menos treinado.
4. **Comparadores calibrados:** ajustar parâmetros somente com treinamento/validação. Não usar versões deliberadamente fracas das políticas clássicas para favorecer redes. O z normal de 1,28 não garante fill rate de 90% em qualquer distribuição de demanda.
5. **Definições e hipóteses:** distinguir fill rate de probabilidade de não ruptura; custo por ciclo de custo por ano; demanda de vendas observadas; vendas perdidas de backlog; um elo de múltiplos elos. Não transportar garantias assintóticas dos artigos para dados e custos diferentes.
6. **Integridade temporal:** elegibilidade, escalas, perfis e limites de ação calculados antes da janela de avaliação; estoques/pipeline iniciais comuns; sementes de treinamento independentes; teste final preservado.
7. **Viabilidade computacional:** validar treinamento completo em lotes de redes independentes e cache de ajustes, com medição real. A presença de DRL na literatura não valida milhões de ajustes individuais sequenciais. O microbenchmark de atualizações já feito não é uma medição de toda a trajetória de RL.
8. **Conclusão condicional:** apresentar comparadores e hipóteses previamente definidos; retirar o que não ajuda a discussão não equivale a esconder resultados desfavoráveis. Matrizes completas e custos computacionais devem acompanhar o material suplementar.

PIL, em S12, permanece uma possível verificação adicional de robustez, condicionada à fidelidade da implementação e ao orçamento. Sua omissão deve ser reconhecida; não há justificativa para afirmar que a comparação reduzida cobre todas as referências fortes da literatura.

## Decisão e estado dos arquivos

O registro de políticas e as configurações foram alinhados a **seis candidatos principais**, com status pendente de validação de correção e desempenho computacional. O módulo registra separadamente os controles obrigatórios do híbrido. Com três sementes nos métodos estocásticos, são **12 ajustes por caso de série/origem**, em vez dos 22 do conjunto anterior. Isso é contagem de ajustes, não previsão de duração.

Foram corrigidas afirmações excessivas nas docstrings das políticas: não se afirma mais que as referências selecionadas sejam as mais recentes, que nenhum agente supere CappedBaseStock ou que uma constante de garantia possa ser aplicada diretamente ao nosso simulador. A alteração foi de documentação; não mudou a dinâmica de simulação ou os pesos de políticas existentes.

O executor atual percorre o portfólio principal; ainda não existe nele uma etapa automática completa de ablação pareada ou o treinamento vetorizado integral. Não se afirma que o desenho científico inteiro já esteja executável e validado. A fila continua parada, e não foram incorporados resultados de uma nova rodada ao artigo.

As fontes abertas estão em `docs/references/inventory_policy_selection/2026-10-14/`, com URLs e hashes no manifesto. O artigo de Zabraoui permanece em sua pasta original, sem duplicação. Textos extraídos para consulta ficam em `experiment/data/08_reporting/literature_review_2026-10-14/`.
