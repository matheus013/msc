# AIPE — proposta de protocolo científico

Versão 0.9, 16/10/2026. Estado: rascunho para discussão; não congelado.

Este documento descreve como formular o estudo sem usar os resultados para escolher a hipótese. Não é um pré-registro anterior aos experimentos existentes: houve exploração prévia. Uma avaliação confirmatória exigirá reserva auditada e congelamento prospectivo. Nenhum experimento é iniciado ou alterado por este documento.

Proveniência: a pergunta deriva da proposta do pesquisador de avaliar o AIPE como metodologia modular. O critério principal foi escolhido pelo pesquisador: **“Reduzir custo sem deterioração relevante do atendimento”**. Em seguida, o pesquisador esclareceu que a relevância prática depende de julgamento humano e contexto: uma economia de 1% pode ser muito valiosa em escala, e a metodologia deve fornecer rastreabilidade e todos os dados necessários à decisão. Não será imposto um percentual mínimo universal de economia. As demais especificações abaixo são propostas de desenho, não respostas atribuídas ao pesquisador; este arquivo não substitui um design brief concluído por diálogo.

## 1. Problema, objetivo e pergunta

Problema: uma família única de reposição pode responder de forma desigual a séries de demanda heterogêneas; selecionar famílias por contexto também pode introduzir erros e custo de implementação. A existência de heterogeneidade não garante que ela seja previsível nem que a contextualização supere uma referência competitiva.

Objetivo geral: propor e avaliar o AIPE como metodologia modular de apoio ao gerente de logística e estoques, que transforma informação disponível sobre a demanda e condições operacionais em recomendações rastreáveis de reposição, buscando reduzir desperdícios financeiros sem deterioração relevante do atendimento.

**Pergunta principal proposta:** como uma metodologia modular de seleção contextual de políticas de reposição pode auxiliar o gerente de logística e estoques a reduzir desperdícios financeiros, preservando o atendimento e oferecendo rastreabilidade das recomendações?

**Pergunta experimental associada:** uma instanciação do AIPE definida previamente reduz o custo de estoque frente à seleção de uma família global, sem deterioração relevante do atendimento, em decisões de reposição avaliadas fora dos dados usados para ajustar o seletor?

O objeto avaliado é a metodologia e a qualidade das decisões de reposição que ela apoia. Não se pretende avaliar ou ranquear o desempenho de lojas, equipes ou gerentes. Séries representam contextos de demanda; identificadores de loja permitem organizar dados, controlar dependências e testar transferência do seletor. Uma diferença de custo entre lojas não é uma medida de eficiência gerencial: pode refletir escala, mix de produtos e condições operacionais.

Uso pretendido: o gerente informa histórico e condições operacionais, recebe alternativas e recomendação, examina custos, atendimento, limitações e rastreabilidade e decide como utilizá-la. O experimento computacional avalia as consequências das escolhas sob condições declaradas. Compreensão, confiança, adoção e efetividade do apoio à decisão por gerentes reais exigem avaliação com usuários ou em campo; não serão consideradas demonstradas apenas pelo simulador.

### Escopo da recomendação aprovado pelo pesquisador

O pesquisador aprovou a proposta de o AIPE recomendar a família de política e seus parâmetros de reposição, apresentar a justificativa da escolha, alternativas e consequências esperadas e deixar a decisão de utilização ao gerente. Essa definição é requisito metodológico; sua aprovação não significa que todas essas saídas já estejam implementadas ou validadas.

Entrega por contexto de estoque:

- **Recomendação:** família de reposição e parâmetros calibrados, com unidades, condições de aplicação e horizonte de validade explicitados.
- **Fundamentação:** informação disponível na decisão, perfil/características pertinentes, critério de seleção e restrições consideradas. Não atribuir interpretação causal a uma importância preditiva.
- **Alternativas:** referência e candidatas comparáveis, com os motivos de preferência ou descarte e o suporte disponível para a comparação.
- **Consequências esperadas:** estimativas de custo e atendimento e suas limitações, identificadas como previsões ou resultados de validação. Resultados efetivamente observados no horizonte de teste devem aparecer separadamente e somente após sua realização; não usar o teste para justificar retrospectivamente a recomendação original.
- **Rastreabilidade:** ligação entre recomendação, dados admissíveis, configuração, versão dos métodos e registros da análise, incluindo eventual manutenção da referência por evidência insuficiente.

A aceitação, rejeição ou alteração pelo gerente pode ser registrada em uma futura aplicação para auditoria. Na avaliação computacional, simular o seguimento da recomendação mede suas consequências sob essa hipótese; não demonstra a conduta real do gerente nem o efeito de sua participação.

### Cadência operacional informada pelo pesquisador

O pesquisador informou que a empresa-fonte reavalia a gestão de estoques **a cada ciclo comercial** e propôs parametrizar a duração do ciclo e a cadência para ampliar a flexibilidade metodológica. Assim, a revisão a cada ciclo comercial é a configuração de referência dessa empresa, não uma frequência universal imposta pelo AIPE. Em cada instante de revisão configurado, usar apenas informações efetivamente disponíveis. Reavaliar não obriga a trocar de família, alterar parâmetros ou retreinar todos os modelos a cada ciclo.

### Configuração temporal parametrizável

Separar explicitamente:

| Parâmetro | Significado | Regra de coerência |
|---|---|---|
| Período básico de observação/ação | Resolução dos registros, atualização do estado e oportunidades de reposição representadas. | Deve ser suportado pelos dados e pelo ambiente; agregação não cria observações mais finas. |
| Ciclo de análise | Intervalo usado para agregar demanda e construir a representação contextual. | Registrar duração ou calendário comercial e limites dos intervalos; ciclo comercial pode ter duração variável. |
| Cadência de revisão do PSE | Intervalo entre oportunidades de rever a família recomendada. | Definir por períodos ou calendário; justificar qualquer revisão mais frequente que a chegada de nova informação. |
| Horizonte de planejamento | Extensão futura considerada na comparação das alternativas. | Pode abranger várias revisões; não se confunde com tempo de validade da recomendação até a revisão seguinte. |
| Janela histórica | Quantidade de passado usada na representação/calibração. | Explicitar duração física e regra de janela fixa ou expansiva, sem incluir futuro. |
| Frequência de atualização dos modelos | Quando recalibrar políticas e retreinar o seletor. | Separada da revisão do PSE, com regra congelada e custo computacional medido. |

Estoque e pipeline evoluem no período básico de ação, inclusive entre revisões. Se a informação só chega em ciclos agregados, o estudo não pode alegar decisões diárias validadas. Datas, calendário e latência devem ser preservados quando conhecidos; a duração aproximada de um ciclo comercial não substitui seu calendário real.

Alterar a duração dos períodos não equivale a trocar um número isolado: reagregar vendas, converter a incidência temporal dos custos e o prazo de entrega, rever características de demanda e recalibrar os modelos quando necessário. Uma taxa de manutenção por dia deve ser aplicada ao tempo correspondente, enquanto custo fixo por pedido não é multiplicado apenas porque a frequência de revisão mudou. Os R$ e as quantidades mantêm sua unidade física. Regimes de chegada que não se alinhem à grade temporal precisam de representação apropriada ou aproximação explícita.

A parametrização é propriedade do desenho pretendido. As configurações temporalmente distintas só serão consideradas validadas se efetivamente avaliadas; a execução atual não demonstra invariância ou superioridade sob qualquer cadência. Comparações entre cadências devem preservar calendário físico de acompanhamento, informação disponível e convenções de custo, reportando também esforço computacional e consequências das mudanças.

Proposta para o estudo dessa cadência: no início do ciclo t, incorporar os registros concluídos até t−1, verificar qualidade/atualidade dos dados e condições operacionais, apresentar a família e parametrização recomendadas para o ciclo t e registrar manutenção, mudança ou decisão do gerente. Confirmar a latência real dos registros antes de presumir que todo dado de t−1 está disponível. Separar a frequência de revisão da decisão da frequência de atualização do seletor e de recalibração dos controladores; estas últimas ainda precisam de regra explícita.

Em avaliação sequencial, preservar o estoque e os pedidos em trânsito que resultam das decisões anteriores, respeitando seu prazo de chegada. Não reinicializar o saldo nem apagar compromissos quando a recomendação mudar. Cada estratégia tem sua trajetória de estado; a igualdade entre estratégias deve ser nas condições iniciais e nos eventos exógenos, não na reposição artificial de um estado comum a cada revisão. Custos de mudança, quando pertinentes, precisam de modelagem ou limitação declarada.

**Diferença em relação à execução atual:** as origens existentes fazem uma escolha de família para um bloco de sete ciclos de teste; a política executa ações ao longo do bloco, mas o PSE não refaz a escolha de família a cada ciclo. Essa execução avalia seleção por bloco e continua válida nesse escopo; não comprova reavaliação cíclica do PSE. Um estudo com decisões revistas a cada ciclo exige protocolo sequencial próprio, sem modificar silenciosamente a fila congelada do M5. Sete ciclos podem continuar como horizonte de acompanhamento contendo sete revisões, em vez de intervalo entre revisões.

O [contrato do relatório gerencial](RELATORIO_GERENCIAL_AIPE.md) especifica os blocos da entrega e sua primeira exportação local da Bahia: 290 casos e nove estratégias, totalizando 2.610 registros. Recomendações e estimativas disponíveis na decisão estão separadas dos resultados de teste. A prévia preserva ausências de parâmetros, trajetórias e intervalos por decisão; não constitui ordem de compra ou entrega integral já validada. M5 permanece na execução original e não foi exportado como se estivesse concluído.

“Desperdício financeiro” é a motivação gerencial. Na avaliação, sua operacionalização será a parcela de custo evitável frente a referências comparáveis, respeitado o atendimento e os componentes de custo efetivamente modelados. Não classificar todo custo de estoque como desperdício, nem alegar redução de perdas físicas, vencimento ou obsolescência sem modelar e medir esses fenômenos.

O efeito esperado não é pressuposto. Pode haver ganho, ausência de ganho, deterioração ou resultado inconclusivo. O estudo também deve esclarecer em quais condições a seleção contextual agrega valor.

Escopo mantido: base brasileira da Bahia e coorte M5 já adotada, analisadas separadamente. As quantidades e datas finais deverão constar em manifesto. Não ampliar nem reduzir a população com base nas políticas vencedoras.

## 2. O que constitui o AIPE

### Contrato de entradas aprovado pelo pesquisador

Em 16/10/2026, o pesquisador aprovou o conjunto de entradas, sua obtenção e o tratamento de ausências discutidos a partir de documentação de sistemas de planejamento. Trata-se de requisito metodológico; a implementação e a cobertura das bases disponíveis ainda devem ser auditadas. Não se presume que Bahia e M5 contenham registros de estoque, custos reais ou pedidos em trânsito.

| Entrada | Origem preferencial | Necessidade e ausência |
|---|---|---|
| Produto, local de estoque, datas e unidades | Cadastro e movimentações | Identificação e referência temporal obrigatórias; exigir definição explícita da unidade de quantidade e do período. |
| Histórico de demanda ou vendas | Sistema transacional | Necessário à instanciação baseada em histórico. Identificar se representa demanda, vendas ou saídas; não converter lacunas em zeros automaticamente. Vendas limitadas por ruptura podem subestimar demanda. |
| Prazo de reposição | Pedidos e recebimentos ou informação gerencial | Estimar com dados disponíveis ou declarar cenário. Registrar unidade temporal, calendário, média/variabilidade quando aplicáveis e tratamento de entregas parciais. |
| Custos de manutenção, pedido e falta | Dados contábeis/operacionais e estimativas | Necessários para a comparação econômica que os utilize. Registrar unidade monetária ou abstrata, período de incidência e hipóteses; ausências podem ser tratadas por cenários declarados, sem alegar economia monetária observada. |
| Atendimento desejado | Gerente | Meta e definição da medida explícitas, por contexto quando necessário. Não inferir que a meta de outro sistema significa fill rate. |
| Estoque atual, compromissos e pedidos em trânsito | Sistema de estoque e compras | Necessários para recomendar a próxima compra na situação real. Para avaliar políticas, podem ser substituídos por condições iniciais e fluxos declarados de cenário. Ausência não significa saldo ou trânsito zero. |
| Restrições de reposição | Cadastro de fornecedores e regras operacionais | Lotes mínimos, múltiplos, calendários, capacidades e limites quando aplicáveis. Se ausentes, declarar quais restrições não foram representadas e limitar a recomendação. |
| Política atualmente utilizada e parâmetros | Sistema ou gerente | Referência operacional quando conhecida e reproduzível. Se indisponível, adotar referência metodológica identificada, sem apresentá-la como prática real da empresa. |

Cada entrada deve ter valor/unidade, escopo, data de referência, fonte, classificação de origem (**observada, informada, estimada ou assumida em cenário**) e registro de qualidade e transformações. Para estimativas, identificar método e histórico utilizado; para hipóteses, justificativa e responsável pela definição. Distinguir zero válido, ausência, não aplicável e valor desconhecido. Inputs incompatíveis ou insuficientes para uma saída devem impedir essa saída específica ou restringi-la explicitamente a um cenário, sem preenchimento silencioso.

O AIPE calcula as características estatísticas a partir do histórico admissível; o gerente não precisa fornecer ADI, variabilidade ou proporção de zeros já calculados. Um perfil fornecido continua admissível, com origem e disponibilidade temporal registradas, preservando as duas abordagens de PSE.

**Distinção de uso aprovada:** recomendar família e parâmetros pode ser avaliado sob condições declaradas; orientar quanto comprar agora exige o estado operacional atual. A validação computacional de políticas não será apresentada como validação de uma ordem de compra imediatamente executável.

Referências consultadas na discussão, em 16/10/2026:

- [Oracle/NetSuite — entradas e cálculos de otimização](https://docs.oracle.com/en/cloud/saas/netsuite/ns-online-help/article_0702045810.html): uso de históricos e distinção entre medidas de atendimento.
- [Microsoft Dynamics — propostas de estoque de segurança](https://learn.microsoft.com/en-us/dynamics365/supply-chain/master-planning/safety-stock-journal): revisão humana das propostas antes de aplicação.
- [Oracle — planejamento por políticas](https://docs.oracle.com/cd/E26401_01/doc.122/e48774/T473201T662268.htm): custos e condições de planejamento. Esses exemplos não constituem medição de popularidade nem evidência de novidade do AIPE.

### Cobertura verificada nas bases atuais

A [auditoria de entradas de 16/10/2026](input_audit_2026-10-16/RELATORIO.md) confrontou o contrato com arquivos locais e dependências congeladas. Bahia e M5 oferecem vendas observadas para simulação; custos, prazo de dois ciclos, meta de fill rate de 70% e estados iniciais são condições de cenário. Receita brasileira e preços M5 existem, mas não calibram os custos da execução. O estoque brasileiro encontrado é uma fotografia de 21/05/2025, posterior aos testes Bahia, em nível de franquia–material com vínculos múltiplos a revendedores; não pode ser incorporado retroativamente como saldo inicial real. As duas coortes são históricas e não equivalem a censos de Bahia ou FOODS.

Os painéis reconciliam com as quantidades brutas sob as transformações documentadas. Na Bahia, 2.697 de 5.510 células foram preenchidas com zero por ausência de agregado; isso exige hipótese explícita, não prova de demanda nula. Há rastreabilidade de código, dados e indicadores, mas a persistência canônica ainda não entrega parâmetros finais de todas as políticas, trajetórias e custos desagregados por caso. O contrato aprovado é mais amplo que a implementação demonstrada. Manter esses itens como requisitos de desenvolvimento, sem anunciar implantação ou economia monetária real.

### Sequência metodológica

Fluxo metodológico: preparação e delimitação da informação → representação da demanda → PSE → calibração/execução da política → avaliação e diagnóstico.

Duas abordagens do PSE:

- **Por perfil de entrada:** o perfil está disponível ao seletor como contexto. No desenho atual, quando derivado da série, deve ser calculado somente com o prefixo disponível. A associação perfil–família é estabelecida no desenvolvimento, nunca pelo teste-alvo.
- **Estimado:** um modelo aprende, com dados de desenvolvimento, a recomendar uma família para o contexto-alvo. Pode estimar diretamente uma classe ou as consequências das alternativas.

O recuo à referência é um mecanismo opcional de decisão. AIPE não se reduz ao recuo, a um classificador específico ou a uma política de estoque. A avaliação de algumas instanciações também não demonstra que qualquer substituição de qualquer módulo funciona.

## 3. Hipóteses e comparações

| Identificador | Hipótese proposta | Comparação que a testa | O que contrariaria a expectativa |
|---|---|---|---|
| H1, principal | A instanciação primária reduz custo sem deterioração relevante do atendimento. | AIPE primário versus família global escolhida em desenvolvimento. | Ausência de redução de custo, perda de atendimento incompatível com o contexto, ou incerteza que impeça concluir. A relevância prática é julgada separadamente, sem corte universal de economia. |
| H2, secundária | A representação por perfil pode modificar utilmente as escolhas em relação à seleção global. | PSE por perfil versus global; seletor estimado com/sem perfil, com capacidade controlada. | Decisões idênticas ou ausência de benefício consistente. |
| H3, secundária | Aprender consequências operacionais pode produzir escolhas melhores que aprender apenas a família vencedora. | Consequências versus classificação, com entradas e validação comparáveis. | Benefício ausente, instável ou explicado por diferenças de modelo/orçamento. |
| H4, secundária | O recuo reduz deteriorações em relação à versão sem recuo, preservando benefício útil. | Mesmo seletor com/sem recuo. | Pioras não diminuem ou a redução decorre apenas de eliminar toda intervenção. |

H2–H4 são mecanismos a investigar, não requisitos para declarar toda a metodologia válida. Se suas versões forem escolhidas após observar o teste, passam a exploratórias.

**Instanciação primária: TODO.** Duas opções admissíveis antes da confirmação: fixar uma versão específica; ou congelar uma regra que escolhe entre as versões apenas na validação interna. Na segunda opção, o objeto avaliado é o procedimento completo de escolha. Não chamar de primário o vencedor retrospectivo do teste externo. As demais versões permanecem secundárias.

## 4. Condições de comparação e informação

Referência: uma família global selecionada por partição e origem no desenvolvimento, sob a mesma regra nominal de custo e atendimento. Seus parâmetros podem ser calibrados por série no prefixo disponível; global não significa parâmetros iguais para todas as séries.

Para cada decisão, definir uma origem t e separar prefixo observado, validação disponível e horizonte futuro de avaliação. Características, perfis, normalização, hiperparâmetros, associações, margens, calibração e escolha da referência usam somente informação admissível até a decisão. O seletor externo não aprende com os resultados da loja-alvo. O controlador da série-alvo pode usar seu próprio passado, se essa informação estiver igualmente disponível a todas as alternativas.

Mesmos dados, custos, condições iniciais, dinâmica de estoque, horizonte, prazo, orçamento por família e sementes correspondentes em todas as estratégias de seleção. Estratégias que escolhem a mesma candidata devem receber os mesmos resultados daquela candidata. Acesso aos resultados de todas as candidatas no teste serve apenas para diagnóstico retrospectivo, nunca como entrada de seleção.

A comparação por perfil e a previsão individual podem aplicar a restrição de atendimento em níveis diferentes. É necessário explicitar esse escopo e harmonizá-lo ou tratar a diferença como parte da instanciação; sem esse controle, o efeito não pode ser atribuído apenas ao algoritmo de aprendizado.

## 5. Desfechos e critério principal

Definir caso como série–origem, agregando primeiro as sementes da família estocástica segundo regra congelada. Sementes não são novas séries. Séries pertencem a lojas, e origens da mesma série são dependentes.

**Custo primário proposto:** razão entre somas de custo AIPE e referência nos mesmos casos, menos um. Pesos iguais por caso dentro de cada base, sem selecionar a população pela viabilidade de uma das estratégias. Reportar também custo médio, diferenças por caso e resultado com lojas igualmente ponderadas como análise secundária. Não somar custos Bahia e M5 em um único efeito.

**Atendimento proposto:** fill rate por caso, na escala [0,1], com a mesma definição em todas as estratégias. Avaliar conjuntamente (a) diferença no atendimento médio por caso e (b) diferença na proporção de casos abaixo do alvo. Reportar também atendimento ponderado pelo volume, sem substituir silenciosamente a média por caso.

Demanda de teste zero: incluir no custo e reportar em grupo identificado; atendimento fica indefinido e sai somente do denominador das análises de atendimento, igualmente para ambas as estratégias. Não excluir do resultado principal casos em que o AIPE ou a referência falhem. Comparações apenas entre casos viáveis são análises condicionais secundárias.

Separar três dimensões na conclusão de cada base:

1. **Efeito observado:** apresentar economia absoluta e relativa, mudança no atendimento médio e na proporção abaixo do alvo, incluindo a distribuição por caso e loja. Uma redução pequena não é descartada por um corte percentual universal.
2. **Força da evidência:** reportar incerteza, estabilidade e sensibilidade. Para uma conclusão formal de redução de custo, a proposta é que o intervalo da variação relativa esteja inteiramente abaixo de zero. Para uma conclusão formal de não inferioridade do atendimento, ainda é preciso definir margens contextuais delta_F e delta_V antes da confirmação. Sem essas margens, descrever as diferenças e seus intervalos, sem anunciar não inferioridade demonstrada.
3. **Relevância prática:** o usuário julga o benefício considerando escala, horizonte, custos de implantação/operação, restrições de atendimento e distribuição de ganhos e perdas. Registrar a decisão e seus critérios, inclusive quando feita após ver os resultados, distinguindo-a da evidência confirmatória.

Não há g_min universal. delta_F e delta_V continuam TODO apenas se for adotado um teste formal de não inferioridade; são limites de perda aceitável no contexto, não percentuais mínimos de ganho. O nível nominal de atendimento também deve ser justificado e congelado; 70% é a configuração histórica, não uma exigência universal do problema.

O procedimento e nível dos intervalos, a regra de fronteira e o tratamento das múltiplas comparações também permanecem TODO. Custo menor à custa de deterioração relevante não satisfaz o critério escolhido. Ausência de significância para uma perda de atendimento não prova preservação do atendimento. Rastreabilidade é um requisito da entrega: torna o julgamento verificável, mas não transforma automaticamente um resultado desfavorável ou inconclusivo em benefício demonstrado.

### Entrega para julgamento humano

- Resultado por base, loja, série, origem, política, estratégia e semente, com identificadores que liguem cada agregado aos registros individuais.
- Histórico disponível na decisão, características/perfil, partição de treino/validação/teste, recomendação, referência e motivo de recuo quando houver.
- Custos desagregados, demanda, quantidade atendida, atendimento, violações e diferenças pareadas, com dicionário de dados, denominadores e regra de agregação.
- Configurações, versões do código e dos dados, sementes, manifesto de execução e procedimento para reproduzir as tabelas e figuras.
- Valores absolutos e relativos, incerteza, concentração dos ganhos, deteriorações e esforço computacional, além das médias.
- Exportações em formato aberto e relatório legível, incluindo falhas e resultados negativos. O julgamento não pode depender apenas de indicadores selecionados no painel.

Economia em reais somente quando os custos tiverem unidade monetária e calibração justificadas. Nas configurações de custos abstratos, reportar unidades de custo e identificar como ilustrativa qualquer conversão para uma escala financeira. Uma economia de 1% sobre R$ 1 milhão equivale aritmeticamente a R$ 10 mil, mas isso não é estimativa desta pesquisa. Extrapolações para novos períodos, lojas ou volume devem explicitar hipóteses e incerteza, sem anualização automática.

## 6. Validação e incerteza

Separar desenvolvimento, validação interna e avaliação externa temporal. No desenvolvimento, excluir a loja-alvo do ajuste do seletor e manter a ordem temporal das informações. Não usar resultados do horizonte externo para construir os rótulos ou consequências de treino daquele horizonte.

Para confirmação, auditar uma janela/coorte ainda não consultada ou usada para escolhas. Separar IDs que não pertenciam ao piloto não basta para provar independência. Se não houver reserva adequada, manter a análise como desenvolvimento e planejar nova evidência, sem renomeá-la como confirmação.

Proposta de incerteza: análise pareada que preserve conjuntamente os casos de cada loja e suas origens, sem reamostrar origens/sementes como observações independentes. Especificar se os intervalos são condicionais aos seletores ajustados ou abrangem também sua estimação. Dez lojas M5 limitam precisão e cobertura inferencial; aumentar séries dentro dessas lojas não elimina essa limitação. Método final, suficiência amostral e efeito detectável exigem planejamento antes da confirmação, podendo usar desenvolvimento identificado ou cenários simulados para estimar variabilidade.

Avaliar Bahia e M5 separadamente, sem escolher depois qual será a base principal. Proposta: H1 será testada em ambas, com uma conclusão por base; benefício em ambas permite alegar replicação nesses dois contextos, sem universalização. Se desejar uma conclusão conjunta formal, sua regra e multiplicidade devem ser definidas antes do teste.

## 7. Diagnósticos secundários

- Suporte por perfil, estabilidade entre origens e distribuição das escolhas.
- Frequência e magnitude das pioras, cobertura de intervenção e resultado nas intervenções.
- Concentração da economia nas maiores séries e diagnóstico de influência.
- Oráculo retrospectivo do portfólio: melhor alternativa disponível sob regra idêntica de objetivo/viabilidade. Planejamento com informação perfeita: limite retrospectivo sujeito às condições físicas e comprovação de otimalidade ou lacuna registrada.
- Frequência de atingir o ótimo do portfólio e fechamento da lacuna de custo, com denominadores positivos e populações comparáveis. Não interpretar razão de custos como percentual universal de desempenho ótimo.
- Falhas de seleção, insuficiência do portfólio e inviabilidade física, com critérios verificáveis e casos sem demanda separados.
- Tempo de calibração, treinamento do seletor, avaliação e inferência; custo computacional total e incremental. Não inferir economia total apenas pela inferência rápida.

## 8. Riscos e limites

| Risco | Sinal de alerta | Resposta planejada |
|---|---|---|
| Heterogeneidade sem ganho previsível | Oráculo próximo da referência ou vencedoras instáveis entre períodos. | Reportar limite de benefício e estabilidade; não ampliar algoritmos para buscar vitória. |
| Vazamento ou escolha orientada ao teste | Perfis de série completa, margens ou poda definidos com horizonte externo. | Auditoria de informação; reclassificar como desenvolvimento e reservar nova avaliação. |
| Ganho agregado oculta dano | Economia concentrada ou aumento de séries abaixo do alvo. | Critério conjunto e distribuição de efeitos, incluindo influência e suporte. |
| Diferenças de calibração explicam o efeito | Orçamentos, penalidades ou restrições incompatíveis. | Manifesto e comparação harmonizada; não atribuir o efeito apenas ao PSE. |
| Simulador ou amostra restringem aplicação | Custos não calibrados no campo, demanda observada censurada, poucas lojas. | Delimitar evidência computacional, sensibilidades planejadas e necessidade de validação operacional. |

Não se promete causalidade no campo, implantação validada, superioridade universal, garantia individual de serviço ou modularidade ilimitada. Preservar a referência pode ser uma decisão útil; um procedimento que nunca intervém não demonstra benefício da contextualização.

## 9. Pendências para congelamento

1. Confirmar a formulação da pergunta e o alcance por lojas/períodos. O critério principal já foi escolhido pelo pesquisador.
2. Escolher a instanciação primária ou a regra interna de escolha entre instanciações.
3. Justificar o alvo de atendimento e, se houver teste formal de não inferioridade, delta_F e delta_V. Não exigir percentual mínimo universal de economia; definir como registrar a avaliação humana de relevância e a escala/unidade dos custos.
4. Congelar manifesto de coortes, datas, elegibilidade, portfólio, orçamentos e objetivo/restrição de cada estratégia.
5. Auditar a reserva; fixar partições, horizonte e regra de atualização entre origens.
6. Fixar método de inferência, suficiência amostral e multiplicidade, incluindo as duas bases e hipóteses secundárias.
7. Identificar o comparador de literatura após leitura completa, com protocolo de adaptação se necessário.
8. Versionar protocolo, código e dados antes de abrir a reserva; registrar qualquer desvio posterior.

Até resolver esses pontos, o protocolo é uma proposta de estudo, não um plano confirmatório pronto para execução. Os resultados existentes continuam úteis como desenvolvimento, sem alterar sua proveniência.
