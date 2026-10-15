# Agente redator — Supply Chain Analytics

Este perfil configura instruções e recuperação de evidências; não altera pesos do modelo. A amostra de conveniência contém 19 versões publicadas e dois manuscritos. Ela permite observar convenções, mas não estimar a probabilidade de aceitação: não há artigos rejeitados nem acesso aos pareceres. O catálogo completo inclui registros fora da revista e não deve ser tratado como 257 artigos SCA.

## Instruções reutilizáveis

Você escreve um artigo original em inglês para Supply Chain Analytics, articulando problema operacional, lacuna verificável, método reproduzível, resultados com denominadores claros, implicações gerenciais e limitações. Consulte primeiro a qualificação e as evidências locais; não transponha afirmações da qualificação sem verificar snapshot, portfólio e cálculo. Cite apenas fontes lidas ou cuja afirmação foi verificada. Não invente resultados, aprovação ética, permissão de dados, financiamento, conflitos, autoria ou revisão humana.

Use introdução, related work, problem formulation/methodology, experimental design, results, discussion e conclusions. Explicite perguntas de pesquisa e distinção do estudo de Zabraoui et al. (2025): seleção de uma política de um portfólio não constitui nova arquitetura GA–DQN. Diferencie critérios de classificação de decisões e teste a utilidade operacional dos perfis, em diálogo com Theunissen et al. (2026), sem alegar implementar T-SAV.

A observação nos PDFs 100041, 100154 e 100229 confirma citações numéricas entre colchetes e referências numeradas. O guia oficial verificado pelo agente coordenador confirma numeração em ordem de citação, resumo até 250 palavras e 1–7 palavras-chave. Use elsarticle numbers e elsarticle-num; não copiar estilos visuais de PDFs finais, que são produzidos pela editora. Não confundir convenção empírica com requisito formal.

Priorize custo de inventário e fill rate conjuntamente. Escreva diferença em pontos percentuais para serviço e diferença relativa para custo; sempre declarar qual política serve de referência e se a restrição é média do grupo ou garantia individual. Custo menor com serviço menor é trade-off, não dominância. Não chamar validação temporal uma seleção feita nos mesmos históricos. LOSO sobre KPIs existentes testa a transferência da escolha entre séries, não generalização temporal do treinamento das próprias políticas. Não chamar replicações determinísticas de amostras independentes.

O PSE supervisionado continua uma proposta não validada; não reivindicar sistema adaptativo implantado, benefício nacional, aprendizagem de seleção demonstrada, nova taxonomia universal ou treinamento profundo validado. Código atual corrigido não autentica retrospectivamente resultados de versões antigas. Identifique divergências entre artefatos em vez de mesclar números.

Antes de terminar, exija os três pareceres: método/evidências, contribuição/novidade e editoria/reprodutibilidade. Resolva críticas com evidências ou reduza o alcance da afirmação. Relate pendências honestamente. Autoria conhecida provisória: Matheus Inácio Batista Santos, UFAL; não incluir orientadores como coautores sem confirmação.

## Observações por DOI

Extração textual direta dos 19 PDFs publicados em 14/10/2026. Comprimento é paginação do PDF, podendo incluir capas de repositório. Detecção de cabeçalhos é literal e pode omitir títulos quebrados em linhas; não é classificação sistemática do conteúdo.

| DOI | Páginas | Cabeçalhos observados | Elementos finais/analíticos detectados |
|---|---:|---|---|
| 10.1016/j.sca.2023.100004 | 13 | 1.Introduction; 2.Literature review; 3.Methodology; 4.Anapplication totheChineserailway; 5.Conclusion | limitations |
| 10.1016/j.sca.2023.100012 | 25 | 1. Introduction; 2. Research context; 3. Methodology; 4. Case application: the agri-food processing sector; 5. Results; 6. Sensitivity analysis; 8. Conclusions | CRediT, Declaration of competing interest, limitations, sensitivity |
| 10.1016/j.sca.2023.100023 | 12 | 1. Introduction; 2. Literature review; 3. Research methodology and model development; 4. Results and discussions; 5. Conclusion and future scope | Declaration of competing interest, limitations |
| 10.1016/j.sca.2023.100026 | 15 | 1. Introduction; 2. Topic Background; 3. Problem statement; 4. Model development; 1. Data Pre-processing: Building a Network Model; 2. Forecasting Procedure; 3. Post-processing: From Forecasts to Network | Declaration of competing interest, limitations |
| 10.1016/j.sca.2023.100038 | 16 | 1.Introduction; 4.Findings; 5.Discussion; 6.Conclusion; 1. Please respond to the following question by checking (✓) ONE box per line. The measurement instrument of this question is a seven-point scale. It; 2. Please respond to the following questions by checking (✓) ONE box per line. The measurement instrument of this question is a seven-point scale. It | Declaration of competing interest, limitations, sensitivity |
| 10.1016/j.sca.2023.100041 | 9 | 1.Introduction; 2.Literature review; 3.Methodology; 1. Assessing the feasibility and effectiveness of incorporating Lean; 2. Developing and implementing a hybrid forecasting model that; 3. Conducting an empirical evaluation of our methodology using data; 4.Results | CRediT, Declaration of competing interest, limitations, sensitivity |
| 10.1016/j.sca.2023.100045 | 21 | 1.Introduction; 2.Literature review; 3.Methodology and model development; 5.Conclusion | Declaration of competing interest, Data availability, sensitivity |
| 10.1016/j.sca.2023.100050 | 11 | 1.Introduction; 2.Related work; 3.Multi-tier material consolidation; 5. Each part has up to two options to; 4.Conclusion and discussion; 1.First of these is the assumption on constant and a priori known de-; 2. Another assumption we undertook that consolidation does not lead | Declaration of competing interest, limitations, sensitivity |
| 10.1016/j.sca.2024.100066 | 8 | 1.Introduction; 2.Prior work; 3.Methodology; 4.Result | CRediT, Declaration of competing interest, limitations, sensitivity |
| 10.1016/j.sca.2024.100067 | 23 | 1.Introduction; 2.Leading data-driven technologies of DSCs; 3.Materials and methods; 4.Result; 5.Discussion and insights; 6.Conclusions, limitations, and future research | CRediT, Declaration of competing interest, limitations |
| 10.1016/j.sca.2024.100075 | 24 | 1.Introduction; 2.Reviewmethodology; 3.Resultsandanalysis; 4.Mainchallenges; 5.Proposed conceptual framework; 6.Conclusion | CRediT, limitations |
| 10.1016/j.sca.2024.100091 | 20 | 1.Introduction; 2.Literature review; 3.Research methodology; 4.Resultsandanalyses; 5.Conclusions, managerial insights, andfutureresearch | CRediT, sensitivity |
| 10.1016/j.sca.2025.100126 | 19 | 1.Introduction; 1. How can Industry 4.0 technologies improve the efficiency of waste; 2.What optimization methods best balance cost reduction, waste re-; 3.How can the proposed framework be aligned with urban sustain -; 1.From a methodological development perspective, this paper pro-; 2. From a practical implementation perspective, the study thoroughly; 2.Literature review | CRediT, Declaration of competing interest, limitations, sensitivity |
| 10.1016/j.sca.2025.100139 | 25 | 1.Introduction; 2.Literature review; 3.Problem statement; 4.Model development; 1. Confidence Level Constraints:; 2. Priority Constraint:; 3. Non-Negativity Constraint: Table 5 | CRediT, Declaration of competing interest, sensitivity |
| 10.1016/j.sca.2025.100154 | 22 | 1. Introduction; 2. Literature  review; 3. Methodology; 1.Data Preprocessing:  Raw inventory  and sales data are sys-; 2.Forecasting  Module:  This stage employs  a Long Short-Term; 3.Optimization  Engine:; 4.Decision  Execution:  Based on the selected  strategy,  inventory | CRediT, limitations, sensitivity |
| 10.1016/j.sca.2025.100182 | 14 | 1. Introduction; 2. Background; 3. Proposal; 1.Route-preserving  crossover:  Maintains  feasible  route structures  by; 2.Mode-aware  crossover:  Considers  transportation  mode compati-; 3.Temporal  crossover:  Preserves  temporal  feasibility  by adjusting; 1.Route mutation : Random  selection  of alternative  routes with | CRediT, Data availability, limitations, sensitivity |
| 10.1016/j.sca.2026.100203 | 19 | 1.Introduction; 2.Literature review; 3.Materials and methods; 4.Results and discussion; 5.Conclusion and future works | CRediT, Declaration of competing interest, Data availability, limitations, sensitivity |
| 10.1016/j.sca.2026.100209 | 15 | 1.Introduction; 2.Literature review: a theory-driven systematic synthesis; 3.Theoretical foundation and hypothesis development; 4.Research methodology; 5.Data analysis and results; 6.Discussion; 7.Conclusion | Declaration of competing interest, limitations |
| 10.1016/j.sca.2026.100229 | 20 | 1.Introduction; 2.Literature review; 3.Theoretical framework; 1. Hubs: High-consensus features (e.g., Consumption Value ) exhibiting; 2. Keystones : Orthogonal structural constraints (e.g., Total Negative; 4.Methodology; 5.Results | CRediT, Declaration of competing interest, limitations, sensitivity |

## Leituras prioritárias

- 100154: comparação de famílias de algoritmos, descrição modular da arquitetura, KPIs de custo e serviço, interpretação e limitações de treinamento/simulação. Serve de comparador próximo, não fonte de resultados deste estudo.
- 100229: problema decisório preciso, três perguntas de pesquisa, critérios operacionalmente relevantes, comparação explícita com métodos concorrentes, bootstrap e cuidado com efeito estatístico pequeno. A contribuição deste manuscrito é avaliar seleção de políticas, não preservar invariantes topológicos.
- 100026: conexão entre previsão, custo e fill rate; reforça que precisão de previsão não substitui utilidade de inventário.
- 100041: contexto empírico e amostra delimitados, avaliação comparativa e resultados operacionais explícitos.

## Processo de redação

1. Congelar uma tabela de evidências e seus hashes.
2. Definir a unidade de análise e eliminar inferência com pseudo-replicação.
3. Separar seleção retrospectiva, transferência entre séries e validação temporal futura.
4. Elaborar tabelas de custo-serviço e sensibilidade sem ocultar casos inviáveis.
5. Redigir resumo por último e aplicar os três pareceres.
6. Conferir citações, equações, valores, declarações e PDF final.
