# Roadmap de fortalecimento científico do AIPE

Data: 16/10/2026. Documento de planejamento; não autoriza nem inicia novas execuções. Base: revisão de literatura dirigida, revisão do pipeline e fechamento do ciclo escritor–três revisores. Não representa nova busca bibliográfica.

## Objetivo e ponto de partida

Consolidar o AIPE como metodologia modular de apoio ao gerente de logística e estoques para reduzir desperdícios financeiros mediante seleção contextual de famílias de reposição e recomendações rastreáveis. O objeto é a decisão de reposição, não o desempenho gerencial de lojas. As duas abordagens de PSE — associação por perfil e seleção estimada — são instanciações a comparar. Previsão de consequências e recuo à referência são mecanismos candidatos dentro dessa metodologia, não sua definição inteira.

O artigo deve demonstrar uma contribuição delimitada; a dissertação deve explicar a metodologia, suas instanciações, desenvolvimento, limitações e evidências em maior amplitude. A qualidade editorial alcançada não resolve as pendências científicas.

Estado consultado nesta elaboração: Bahia concluída, 4.350/4.350 ajustes; último registro M5 em 16/10 às 10:02:48, horário de Brasília, com 28.458/79.890 ajustes (35,6%), 15 processos CPU. São registros de acompanhamento, não uma previsão de término. Brasil permanece fora do escopo. Os resultados harmonizados da Bahia ainda precisam ser reconciliados com todos os números, figuras e tabelas dos documentos.

## Sequência de ações

| Etapa | Prioridade | Ação e dependência | Entrega verificável | Critério de conclusão |
|---|---|---|---|---|
| R1. Contrato científico | Imediata | Fixar pergunta, hipóteses, população, referências, métricas e limites das alegações. Pode ocorrer durante o M5. | Protocolo versionado e mapa pergunta → comparação → medida → interpretação. | Cada alegação central tem teste definido e resultado capaz de contrariá-la; distinguir exploração de confirmação. |
| R2. Antecedentes próximos | Alta, em paralelo | Ler textos completos dos antecedentes de 2012 e 2021, aprofundar Francuz–Bányai e rastrear trabalhos próximos. | Matriz de objetivos, entradas, decisão, calibração, validação, dados e limitações, com localização da evidência. | Diferenças verificadas em texto completo; selecionar um comparador operacional compatível ou justificar precisamente a impossibilidade. |
| R3. Consolidar a matriz atual | Alta | Continuar M5 com checkpoints; auditar saídas harmonizadas da Bahia. Preservar código científico congelado da execução. | Matriz completa por base, relatório de integridade e manifesto de dados/configuração/código. | Sem casos ou sementes faltantes, duplicados ou incompatíveis; calibração e convenções identificadas; todos os resultados rastreáveis. |
| R4. Identificar mecanismos e limites | Alta, após R3 por base | Avaliar as nove estratégias existentes, ablações, oráculos, deteriorações e concentração dos ganhos. | Relatório pareado com incerteza, suporte e figuras; diagnóstico de ganho disponível e recuperado. | Explicar quando contextualizar ajuda, quando falha e quanto ganho é concentrado; não confundir sementes/origens com unidades independentes. |
| R5. Comparador e robustez dirigida | Alta, após R2 e diagnóstico R4 | Implementar o comparador escolhido; planejar pequeno conjunto de cenários para custo de ruptura, prazo de reposição e horizonte. | Protocolo e resultados de comparação justa; análise de sensibilidade com custo computacional registrado. | Comparabilidade declarada; direção e magnitude dos efeitos examinadas, incluindo resultados desfavoráveis; sem escolha de cenários pelo resultado. |
| R6. Confirmação independente | Essencial para alegação confirmatória | Definir reserva genuína antes de inspecioná-la; congelar método, margem, seleção de referência, métricas e análise. Planejar em R1, executar após escolhas de desenvolvimento. | Protocolo datado, reserva auditável e relatório confirmatório separado. | Nenhuma escolha ajustada ao teste reservado; evidência avaliada nos critérios prévios, mesmo se negativa ou inconclusiva. |
| R7. Integrar artigo e dissertação | Após estabilização das evidências | Atualizar resultados harmonizados e conclusão; distinguir resultados de desenvolvimento e confirmação, mantendo uma exposição integrada do experimento. | Artigo focado e dissertação completa, tabelas/figuras geradas da mesma fonte. | Toda alegação quantitativa corresponde à matriz correta; conclusão proporcional à evidência; AIPE permanece objeto principal. |
| R8. Revisão final e submissão | Final | Repetir ciclo escritor–três revisores sobre versão congelada; auditoria numérica, bibliográfica e visual, revisão humana e requisitos atuais da revista. | Respostas aos pareceres, PDFs verificados e pacote de submissão. | Nenhuma pendência crítica de validade ou rastreabilidade; limitações explícitas; aprovação autoral para submissão. |

R1 e R2 avançam enquanto R3 termina. R4 pode começar pela Bahia, identificando seu caráter exploratório. O trabalho de R4/R5 não deve modificar as dependências científicas da fila M5 em andamento.

## R1 — perguntas e regras de decisão

Atualização de uso em 16/10: o pesquisador informou reavaliação a cada ciclo na empresa-fonte. O protocolo deve distinguir essa cadência da seleção por bloco de sete ciclos da execução atual. Planejar avaliação sequencial com informação causal e continuidade de saldo/pipeline antes de alegar validação da reavaliação cíclica. Revisão da decisão não implica retreinamento completo a cada ciclo.

1. **Valor da seleção contextual:** em quais condições uma instanciação do AIPE melhora a decisão frente à família global selecionada apenas no treino/validação? Medir diferença pareada de custo, atendimento e proporção abaixo do alvo, na mesma população.
2. **Papel dos módulos:** perfil, seletor e recuo alteram a decisão e seu desempenho? Comparar instanciações com orçamento, entradas e capacidade controlados quando possível. Uma ablação de seletor não valida automaticamente a substituição de todos os módulos do AIPE.
3. **Utilidade da intervenção:** quando substituir a referência produz benefício e quando a manutenção dela é preferível? Medir cobertura de intervenção, frequência e magnitude das pioras e resultados nas intervenções, além do total.
4. **Limites do problema:** quanto da falha vem da seleção, do portfólio e das condições físicas? Distinguir oráculo retrospectivo do portfólio e planejamento com informação perfeita.
5. **Transferência e viabilidade:** os efeitos persistem entre lojas, períodos e as duas bases, e qual esforço de calibração/seleção exigem? A base brasileira sustenta relevância aplicada; M5 sustenta comparação em uma base pública conhecida. Ambas permanecem sob as limitações do simulador e dos custos adotados.

Definir antes da confirmação a métrica primária, a unidade de análise, a regra de avaliação do atendimento e o tratamento de múltiplas comparações. Conforme decisão do pesquisador em 16/10, não haverá percentual mínimo universal de economia: a relevância prática será julgada por humanos com contexto, escala, custos e resultados rastreáveis. Margens para uma eventual alegação formal de não inferioridade do atendimento precisam de justificativa anterior ao teste; sem elas, reportar diferenças e incerteza sem alegar equivalência. A entrega deve permitir reconstruir os agregados e consultar os resultados individuais, incluindo deteriorações.

## R3 e R4 — aproveitar o que já existe

- Reutilizar a matriz de controladores para comparar global, perfil, classe, consequências e recuo. Verificar primeiro as nove estratégias implementadas; só acrescentar outra se responder a pergunta não coberta.
- Separar família global estimada em validação da melhor família global retrospectiva. A primeira é referência executável; a segunda é diagnóstico.
- Reportar resultados por loja, perfil e origem, suporte de cada grupo e estabilidade das escolhas. No M5, as dez lojas limitam a precisão da inferência entre lojas.
- Usar diferenças pareadas e intervalos compatíveis com o agrupamento por loja e a repetição temporal. Agregar sementes conforme o protocolo; não multiplicar artificialmente a amostra.
- Avaliar concentração: contribuição das maiores séries para a economia e análise retirando a série de maior benefício. Essa retirada é diagnóstico de influência, não um novo resultado principal escolhido após observar os dados.
- Medir frequência de escolha ótima retrospectiva e fechamento da lacuna até o oráculo apenas em comparações com mesma população, objetivo e viabilidade. Denominador nulo ou quase nulo deve ser explicitamente tratado; não truncar resultados para favorecer a narrativa.
- Manter casos sem demanda de teste identificados, com atendimento indefinido, e relatar separadamente inviabilidade física, insuficiência do portfólio e erro de seleção.
- Integrar fronteira custo–atendimento, escolhas por perfil, distribuição de atendimento, mapa por perfil/limiar e caracterização da demanda; acrescentar intervenções/deteriorações e decomposição de falhas conforme a pergunta respondida.
- Registrar tempo e recursos da calibração, seleção, avaliação e inferência separadamente. Inferência rápida não prova economia computacional do pipeline inteiro.

## R5 — controlar a expansão experimental

O comparador deve responder à mesma decisão sob dados, informação disponível, custos, ambiente e orçamento documentados. Se o antecedente pressupuser outro regime de estoque, apresentar a adaptação e suas consequências; não rotulá-la como reprodução exata. Priorizar comparador que reutilize a matriz existente, se cientificamente adequado.

Nas sensibilidades, fixar antecipadamente uma pequena grade operacionalmente justificável e uma amostra com regra de seleção documentada. Mudanças em custos, prazo e horizonte podem exigir nova calibração e simulação: não reutilizar resultados incompatíveis como se fossem novos cenários. Ajustes que só afetam a camada de seleção podem reutilizar controladores quando suas dependências permanecerem válidas.

Não eliminar políticas usando o desempenho do teste que servirá de confirmação. Se houver poda, defini-la no desenvolvimento por redundância, contribuição e esforço; congelar o portfólio antes do teste reservado e conservar uma comparação com o portfólio completo para medir a perda causada pelo corte.

## R6 — proteger a confirmação

Auditar quais séries, períodos e resultados já influenciaram escolhas. As séries M5 fora do piloto não se tornam independentes apenas por receberem esse rótulo. Uma janela futura ou bloco ainda não consultado pode servir de reserva somente se houver história e volume adequados, separação temporal e ausência de uso nas decisões anteriores. Uma reserva pequena ou inadequada não deve ser chamada de confirmação.

Se não houver reserva genuína nas bases atuais, registrar essa limitação e definir nova janela/coorte ou outra fonte compatível. Não prometer validação independente com a matriz já examinada. Após o congelamento, qualquer ajuste provocado pelo teste transforma essa rodada em desenvolvimento e exige nova confirmação para a versão modificada.

## Decisões após os resultados

| Achado | Ação e consequência para a contribuição |
|---|---|
| Ganho estável e operacionalmente relevante, com deterioração dentro do limite prévio | Sustentar benefício nas condições avaliadas, com incerteza e escopo explícitos. |
| Ganho em poucos contextos ou poucas séries | Delimitar esses contextos, examinar suporte e influência; não generalizar o ganho médio. |
| Perfil ou seletor mais complexo não agrega valor | Preferir instanciação mais simples e discutir a ablação como resultado da metodologia. |
| Recuo raramente intervém | Examinar se há pouco ganho disponível ou pouca capacidade de identificá-lo; não confundir ausência de intervenção com sucesso. |
| Oráculo do portfólio também não atinge o alvo | Investigar candidatos e ambiente antes de ampliar o seletor. |
| Referência próxima do oráculo | Explicitar o baixo benefício potencial; não acrescentar algoritmos apenas para procurar uma vitória. |
| Resultado confirmatório negativo ou inconclusivo | Manter o resultado e restringir as alegações; não reabrir o teste para ajustar o método e reapresentá-lo como confirmação. |

## Ordem de trabalho recomendada

Primeiro, formalizar R1 e a auditoria da possível reserva de R6, enquanto continuam R2 e a execução R3. Depois, fechar o diagnóstico R4 da Bahia e, ao completar a matriz, do M5. Usar esse diagnóstico e os antecedentes para selecionar somente as comparações adicionais necessárias em R5. Congelar a versão candidata, executar R6 e então atualizar os dois documentos em R7/R8.

Não há necessidade demonstrada de iniciar Brasil, ampliar redes neurais, introduzir novos algoritmos ou repetir integralmente Bahia/M5 nesta etapa. O cronograma deve depender desses marcos e do tempo efetivamente medido da execução, sem prometer prazo com base apenas em divisão ideal de CPU.

## Fontes internas

- `docs/operations/aipe_pipeline_review_2026-10-15/PIPELINE_E_PLANO.md`: contrato e verificações do pipeline; seu estado operacional é histórico.
- `docs/supply_chain_analytics_article/REVISAO_DIFERENCIAL_AIPE_2026-10-15.md`: antecedentes e limitações da busca dirigida.
- `docs/supply_chain_analytics_article/reviews/writer_review_loop_2026-10-16/FECHAMENTO.md`: convergência editorial e pendências científicas.
- `experiment/data/08_reporting/research_v2/aipe_ba145_unified_2026-10-15/status.json` e `aipe_m5_2408_unified_2026-10-15/status.json`: registros operacionais consultados na elaboração.
