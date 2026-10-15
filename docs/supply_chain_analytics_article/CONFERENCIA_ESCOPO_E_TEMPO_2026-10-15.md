# Conferência antes de reiniciar os experimentos — 15/10/2026

**Atualização operacional mais recente:** o usuário cancelou o Brasil e solicitou que o M5 continue com 8 processos CPU e prioridade reduzida. A Bahia terminou com 4.350 treinos. O M5 foi interrompido com 4.439 checkpoints e retomado no mesmo banco, sem repetir os treinos concluídos; a mudança operacional foi registrada no manifesto. O Brasil foi removido do launcher e possui uma marca de cancelamento que bloqueia sua inicialização automática. Os tempos calculados abaixo para 16 processos são históricos e não constituem estimativa atual do M5 com 8 processos.

## Atualização após autorização de execução

Execução autorizada pelo usuário e iniciada em 15/10/2026, às 05h14 (America/Sao_Paulo), após 40 testes aprovados. As informações de pausa abaixo registram a conferência anterior à autorização.

- Bahia: `bahia145_cpu16_2026-10-15`, 145 IDs, duas origens, 4.350 treinos incluindo DQN de controle; 54 checkpoints compatíveis reaproveitados.
- M5: `m5_zabraoui2408_cpu16_2026-10-15`, 2.408 IDs de referência, elegibilidade 885/2.033/2.408, 79.890 treinos incluindo DQN de controle; 639 checkpoints compatíveis reaproveitados.
- Brasil: `br_all_eligible_cpu16_2026-10-15`, fila nacional independente posterior às duas coortes; 19.504.212 treinos principais antes de reaproveitamentos, sem repetir a ablação DQN de todas as séries nacionais. Já foram reaproveitados 7.440 checkpoints compatíveis da Bahia ampliada. Os resultados da Bahia145 concluída serão reaproveitados adicionalmente antes de iniciar a etapa nacional.

As filas executam em sequência, com no máximo 16 processos CPU no total, e possuem diretórios, bancos e manifestos próprios. Ordem e estado estão em `experiment/data/08_reporting/research_v2/scoped_execution_2026-10-15.json`.

O GA-DQN foi corrigido na variante `ga_gated_dqn_consistent_v2`: o índice da ação identifica uma quantidade nominal e a mesma transformação pelo gatilho/piso do GA vale nas demonstrações, no treinamento e na inferência; a recompensa terminal também é calculada da mesma forma. Não é alegada reprodução exata do algoritmo publicado. Os checkpoints antigos do híbrido não foram importados para essa variante, mas continuam preservados na rodada interrompida. Os outros métodos mantiveram a implementação e tiveram código, configuração e dados conferidos antes do reaproveitamento. O artigo histórico não foi atualizado com números dessas execuções ainda em curso.

**Registro da conferência anterior à autorização:** a fila ampliada foi interrompida e preservou 19.818 treinos. O pedido de todas as séries elegíveis foi aplicado indevidamente à Bahia e ao M5, embora se referisse ao Brasil completo. Os dados das bases não foram concatenados para treinar políticas; eram tarefas independentes intercaladas no mesmo executor. As novas filas devem manter diretórios, checkpoints, denominadores e relatórios separados.

## Coortes verificadas nos arquivos

- **Bahia:** os mesmos 145 IDs do experimento histórico, em 115 lojas e 9 itens. Todos os 145 estão presentes nos painéis brutos corrigidos e são elegíveis nas duas origens.
- **M5:** os mesmos 2.408 IDs da coorte histórica do projeto. A seção 3.7 de Zabraoui descreve alimentos de alta volatilidade, CV e histórico de ruptura, mas não publica IDs nem corte numérico de CV. FOODS + CV ≥ 1,5 é a adaptação local; não se pode afirmar identidade com a amostra publicada.
- **Brasil:** experimento nacional independente, com todas as séries elegíveis disponíveis. Há 26 UFs identificadas disponíveis, sem RO; algumas UFs não têm séries que atendam ao critério.

As coortes históricas foram selecionadas com filtros que consultaram a trajetória completa. Mesmo usando ajuste e perfis apenas no treino, a nova avaliação será **condicional à coorte retrospectiva**, e não uma validação integralmente prospectiva da seleção de séries.

## Origem temporal e quantidade de avaliações

Mantendo o protocolo de pesquisa atual: mínimo de três ciclos positivos no prefixo; sete ciclos de validação e sete de teste; perfis calculados no prefixo; mesmas condições iniciais entre políticas.

| Base | Ciclos de treino | Séries da coorte | Elegíveis nesta origem | Prefixo zerado | Apenas 1–2 ciclos positivos |
|---|---:|---:|---:|---:|---:|
| Bahia | 17 | 145 | 145 | 0 | 0 |
| Bahia | 23 | 145 | 145 | 0 | 0 |
| M5 | 50 | 2.408 | 885 | 1.433 | 90 |
| M5 | 64 | 2.408 | 2.033 | 214 | 161 |
| M5 | 78 | 2.408 | 2.408 | 0 | 0 |

São **290 casos série–origem na Bahia** e **5.326 no M5**. As origens repetem séries; não são novas séries independentes. No M5 não serão treinadas todas as 2.408 séries em todas as origens: o critério de informação passada exclui as ainda sem histórico suficiente. Prefixo sem vendas não comprova início operacional do produto nem ruptura.

O painel M5 corrigido tem 92 ciclos completos de 21 dias, sem a suavização IQR dos artefatos históricos. O histórico preservado tem 93 ciclos, incluindo um ciclo parcial. O artigo de Zabraoui trabalha em frequência diária e não fornece um protocolo que certifique identidade com essas adaptações. Os resultados antigos não serão substituídos ou renomeados como resultados do novo protocolo.

## Políticas, sementes e treinos

Portfólio principal: EOQCycle, sSCycle, CappedBaseStock, GA, PPO e GA-DQN. As três primeiras têm um ajuste por caso; as três últimas usam sementes 11, 29 e 47: **12 treinos por caso**, não seis.

| Política | Bahia | M5 |
|---|---:|---:|
| EOQCycle | 290 | 5.326 |
| sSCycle | 290 | 5.326 |
| CappedBaseStock | 290 | 5.326 |
| GA | 870 | 15.978 |
| PPO | 870 | 15.978 |
| GA-DQN | 870 | 15.978 |
| **Total principal** | **3.480** | **63.912** |

Foram encontrados **72 checkpoints da Bahia** e **852 do M5** nos IDs corretos e no protocolo atual. Assim, sem mudar os algoritmos, restariam **3.408** e **63.060** treinos. Os demais resultados da fila ampliada permanecem preservados e podem ser úteis ao experimento nacional; não pertencem ao denominador das coortes reduzidas.

Esse reaproveitamento é condicional à implementação: 18 dos checkpoints da Bahia e 213 do M5 são GA-DQN. Correções no híbrido exigem reavaliar sua compatibilidade; não será certificado reaproveitamento de uma versão numericamente diferente. Os checkpoints anteriores mantêm sua proveniência de CPU/GPU.

Para sustentar uma comparação do híbrido com os componentes, o **DQN isolado precisa de uma ablação adicional**: 870 treinos na Bahia e 15.978 no M5, antes de verificar reaproveitamentos. Com esse controle, os totais brutos seriam **4.350** e **79.890**. O DQN não deve ser contado simultaneamente como excluído e como controle já executado.

## Orçamento computacional

O código atual usa GA com população 100 e 50 gerações; DQN/PPO com pelo menos 20 mil transições, arredondadas para episódios completos; híbrido com 1.000 transições de demonstração antes do treino DQN. Esses valores são do protocolo de pesquisa atual, não uma reprodução dos orçamentos históricos ou de Zabraoui.

| Prefixo | Episódios RL | Transições RL efetivas |
|---|---:|---:|
| 17 | 1.177 | 20.009 |
| 23 | 870 | 20.010 |
| 50 | 400 | 20.000 |
| 64 | 313 | 20.032 |
| 78 | 257 | 20.046 |

Estimativa com **16 processos CPU no total**, executando as bases em sequência ou repartindo esses 16 processos. Dois experimentos simultâneos com 16 processos cada somariam 32 e não têm validação de desempenho.

| Escopo | Trabalho de treino / 16, estimativa central | Faixa prática para planejamento |
|---|---:|---:|
| Bahia, seis políticas principais | 1,20 h | **1,5–2 h** |
| M5, seis políticas principais | 19,48 h | **22–30 h** |
| Bahia incluindo DQN de controle | +0,68–0,91 h de treino | **2–3 h no total** |
| M5 incluindo DQN de controle | +12,48–16,64 h de treino | **36–48 h no total** |

As estimativas principais usam milhares de tempos por política observados na execução CPU16. A conta soma o tempo médio por treino multiplicado pelos treinos restantes e divide por 16. As faixas práticas incluem folga para avaliação, gravação, balanceamento, relatórios e variação entre origens. As medições principais disponíveis são das primeiras origens, não uma calibração completa do recorte corrigido. O custo do DQN vem de um benchmark CPU separado e foi colocado em faixa de 45–60 s por ajuste; não é uma medição do DQN na produção CPU16.

Não são prazos garantidos. Corrigir o híbrido pode alterar tempos e checkpoints reutilizáveis. Bahia e M5 compartilhando os mesmos 16 processos podem acabar em momentos diferentes; os tempos da tabela não significam que ambos recebem a capacidade total simultaneamente.

O **Brasil completo** tem 1.625.351 casos série–origem, correspondentes a **19.504.212 treinos principais antes de reaproveitamentos**. Uma extrapolação grosseira de 3 mil treinos/hora dá aproximadamente 271 dias. Esse número apenas evidencia a escala; não é um prazo confiável nem deve ser agregado ao prazo das coortes Bahia/M5.

## Pendências antes da próxima execução científica

A revisão encontrou três inconsistências no GA-DQN: demonstrações rotuladas com uma ação discreta, mas executadas com quantidade contínua; tratamento terminal diferente entre demonstrações e treino; e restrições de reposição aplicadas pelo wrapper na execução que não aparecem no treino da rede. É necessário corrigir ou formalizar explicitamente a variante antes de defender sua eficácia. O plano não deve ser anunciado como uma rodada já corrigida e pronta enquanto essas decisões estiverem pendentes.

Dados auditados, hashes de coortes e cálculos estão em `experiment/data/08_reporting/planning/scope_audit_2026-10-15/audit.json`. Nenhum experimento foi iniciado para produzir esta conferência.
