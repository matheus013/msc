# Revisão de contribuição — artigo em português com base brasileira e M5

Data: 2026-10-14. Fontes examinadas: `manuscript_ptbr.tex`, `references_ptbr.bib`, `experiment/data/08_reporting/sca_article/dual/dual_results.json`, leitura anterior dos KPIs/cenários/configuração M5 e pareceres de contribuição anteriores. Revisão independente; não alterei o manuscrito.

**Parecer: rascunho coerente de estudo comparativo retrospectivo, com revisão maior antes de submissão.** As duas bases estão salientes e preservam funções diferentes: dados proprietários brasileiros são o caso central, e vendas públicas M5 ampliam a comparação. O resultado negativo ou inconclusivo não é uma deficiência a ser corrigida pela busca de um efeito positivo. Permanecem questões de contribuição incremental, reproducibilidade da seleção e limites das matrizes. Não recomendo omitir o M5.

## Coerência da comparação entre as bases

- A coorte brasileira de 145 séries, 115 lojas e 9 produtos e a M5 de 2.408 séries, 10 lojas e 287 produtos correspondem ao JSON de análise. Os números não devem ser apresentados como resultados de populações representativas dos dois países.
- Os prefixos configurados de 17/19 ciclos não são confundidos com uma garantia de separação temporal em todo o pipeline. Suavização, filtros, características e estados iniciais potencialmente dependentes do histórico completo são reconhecidos.
- A condição M5 de 2.213 prefixos nulos,91,90%, e 2.237 prefixos com menos de três ciclos positivos é visível no resumo, dados e discussão. O texto evita atribuir causalmente todo o baixo atendimento a essa condição ou concluir falha universal de algoritmos.
- Os orçamentos diferentes de GA, SA e híbrido estão declarados como configuração examinada, sem falsa certificação da execução histórica. Assim, mudança de vencedor entre bases é exemplo de dependência do contexto, não efeito isolado da origem da demanda.
- Custos absolutos são interpretados somente dentro da base. O custo diferente entre tabelas não sustenta uma comparação monetária Brasil–Estados Unidos.
- Doze rótulos de política presentes na matriz não são confundidos com doze controladores reexecutados, nem com treinamento do PSE. Também é explícito que o benchmark M5 não passa pela etapa LSTM.

## Resultados e incerteza

O JSON sustenta a comparação principal: aumento de custo por perfil na base brasileira, redução pequena no M5, e intervalos condicionais de loja incluindo zero. O uso conjunto de custo, fill rate e violações individuais evita vender economia por serviço menor. A exposição de 39/145 e 274/2.408 séries abaixo do alvo é útil para a discussão operacional.

O agrupamento de loja é apropriado como correção parcial de dependência, mas dez clusters M5 oferecem pouca sustentação para inferência populacional. O texto explicita esse limite, a dependência por produto e a ausência de refitting dentro do bootstrap. Portanto, os intervalos devem permanecer descritivos/condicionais também nos highlights, resumo e comunicação do resultado. Não devem virar um teste forte de equivalência ou ausência de efeito útil.

O caso de 0,95 está coerente: a matriz M5 contém 40 escolhas de fallback de perfil, mas ambas as estratégias pontuam PPO nas 2.408 séries. A diferença exatamente zero deriva de políticas idênticas, inclusive quando a regra de viabilidade do grupo falha; não é evidência de certeza sobre demanda nova. Essa distinção é mantida no manuscrito.

## Correções concretas para o rascunho

### B1 — consolidar as referências duplicadas

Na versão examinada, a citação `Syntetos2005Categorization` usa a entrada com apenas Syntetos/Boylan, omitindo Croston. A mesma bibliografia contém `syntetos2005` com os três autores e mesmo DOI. Consolidar a entrada correta e a citação. O estudo é *On the categorization of demand patterns*, DOI 10.1057/palgrave.jors.2601841.

Também existem duas entradas de Rice 1976: a citada está como artigo sem DOI; a alternativa `rice1976` está como capítulo em *Advances in Computers*, com DOI 10.1016/S0065-2458(08)60520-3. Usar uma entrada coerente do capítulo. Não contar essas duplicatas como ampliação bibliográfica. Essas correções são locais e não requerem mudar os resultados.

### M1 — completar a especificação da camada de seleção

A prioridade dos perfis e os tipos de característica aparecem no texto, mas faltam os limiares e fórmulas necessários para reproduzir a atribuição. A definição de CV² deve explicitar que a implementação inclui zeros, diferente do tamanho das demandas positivas usado na classificação original. O alerta genérico nas limitações é honesto, mas a distinção merece definição operacional no método ou suplemento.

Acrescentar uma tabela ou referência precisa a um suplemento versionado com regras, limiares, desempates e tratamento de perfil ausente na exclusão de loja. O rascunho em inglês já tinha apêndice útil nesse sentido, mas não se deve copiar regras de seu diagnóstico temporal sem conferir que são as regras dos perfis arquivados usados nesta comparação. Hash de arquivo é proveniência, não substituto de especificação legível.

### M2 — calibrar “referência global forte” ao portfólio disponível

A referência global é competitiva dentro dos 12 rótulos armazenados, com a mesma regra de viabilidade; isso é superior a compará-la apenas com EOQ legado. Não está demonstrado que o portfólio contém uma referência de controle forte em sentido amplo, especialmente após encontrar o problema de unidade do EOQ. Sugestão concreta: usar “melhor alternativa global viável do portfólio analisado” onde “forte” puder sugerir um estado da arte certificado. A sensibilidade com EOQ corrigido do diagnóstico anterior é evidência útil da fragilidade do benchmark, mas não pode ser importada numericamente para as matrizes BA/M5.

### M3 — contribuição incremental precisa de posicionamento mais específico

Rice, Kotthoff e a revisão de RL melhoram o enquadramento em relação ao rascunho de quatro fontes. As fontes da revista também sustentam pertinência temática. Contudo, a contribuição não é uma nova arquitetura, uma nova taxonomia ou um PSE validado. O avanço plausível é a evidência comparativa e auditável de que esses perfis pouco acrescentam à seleção global nessas matrizes, incluindo partida sem histórico e efeitos do limiar.

Antes de submissão como artigo completo, situar esse resultado frente a estudos de escolha/segmentação de políticas de inventário, e não apenas a benchmarks de controladores. Não basta aumentar o número de referências. Mostrar concretamente qual conhecimento novo a presença das duas bases permite obter, sem atribuir a elas um experimento controlado de transferência. Uma contribuição de auditoria/reanálise pode ser legítima; precisa de objetivo e diferença científica reconhecíveis para além de falhas locais.

## Condições que não podem ser resolvidas só pela redação

A origem histórica dos controladores não é recuperada por um inventário estático de todos os arquivos atuais. Os casos de M5 com prefixo nulo, a seleção baseada no histórico completo e o pequeno número de lojas continuam impedindo alegações de aprendizado prospectivo. Uma ampliação científica pode alinhar orçamento, usar apenas informação disponível no passado e realizar janelas ou replicação externa; não existe exigência de que o resultado seja positivo. O manuscrito reconhece essas necessidades e deve continuar classificado como rascunho, sem garantia de aceitação.

## Verificação curta da condensação da dissertação

O relatório integrado `docs/master_proposal/revisao_concisao/comparison.json` registra 118 → 66 páginas, redução de 44,07%. Esse número é do comparativo de compilação do coordenador; minha inspeção de conteúdo concentra-se no capítulo de resultados e conclusões. A contagem aproximada do capítulo com meu critério `split()` é 6.020 → 3.382; o critério Unicode do relatório integrado produz outra contagem e não deve ser misturado.

A essência empírica foi preservada: piloto PB com 15 lojas/produto 48130 e seus limites; BA 145 séries/12 políticas; KPIs operacionais; perfis 116/18/11; estratégia por CTI ajustado e suas ressalvas; métricas históricas de previsão; M5 auditado com 2.408 séries e contrastes custo–serviço. A remoção de mapas ilustrativos, múltiplos gráficos sobre a mesma comparação, descrições de eixos e tabelas de significância incompatíveis diminuiu repetição. Nenhum ativo PDF foi apagado e o corte não deslocou volume novo para apêndices.

As conclusões atuais preservam formulação contextual, caso brasileiro e infraestrutura integrada, sem transformar achados descritivos em prova do PSE. O M5 já disponível aparece como complemento público que ainda precisa de auditoria/validação, coerente com a seção nova de resultados. A condensação preservou melhor a contribuição científica do que o texto longo anterior: suas limitações agora ficam próximas das alegações que restringem. Permanecem as pendências de snapshot e significância histórica explicitadas no relatório; concisão não as resolve nem deveria escondê-las.

## Adendo às correções finais

O coordenador consolidou Rice como capítulo com DOI e Syntetos/Boylan/Croston como referência de três autores: **B1 encerrado**. Também substituiu a qualificação ampla de referência forte por alternativa global viável do portfólio: **M2 encerrado**. A tabela de perfis agora explicita cortes, fórmulas e fallback: a exigência de especificação de **M1 foi atendida**. A tabela com as doze políticas de ambas as bases e custos normalizados internamente torna o contraste verificável sem comparar valores monetários absolutos. A condição de ganho em BA no limiar de 0,95 permanece acompanhada de suas restrições de atendimento/fallback, evitando promoção seletiva de um resultado favorável.

Essas correções resolvem os três ajustes concretos do rascunho, sem criar novos pedidos de revisão longa. Permanece a classificação de **versão de trabalho**, com contribuição comparativa incremental a posicionar e limites reais de protocolo: matrizes históricas, preparação potencialmente baseada no histórico completo, orçamentos diferentes, partida sem histórico positivo e apenas dez lojas M5. Nenhuma dessas ressalvas exige omitir o M5 ou obter resultado positivo.
