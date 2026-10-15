# Condensação e revisão científica — capítulo de resultados

Arquivo editado: `capitulos/resultados.tex`. Comparação com o backup `archive/condensation_2026-10-14/capitulos/resultados.tex`.

- Contagem final após incorporar o M5 solicitado (`texto.split()`): **6.020 → 3.382 palavras/tokens**, redução de **43,8%**. Essa contagem inclui comandos, legendas e células de tabelas; não é uma estimativa de palavras faladas ou de páginas compiladas. A primeira condensação, ainda sem M5, tinha 3.089 tokens.
- Figuras no capítulo: **10 → 1**. Permaneceram somente os pontos custo–serviço/Pareto. Nenhum PDF ou outro ativo de imagem foi apagado.
- Tabelas e inclusões de tabelas: **10 → 5**. Permaneceram KPIs das 12 políticas, vencedores dos três perfis, comparação de estratégias por CTI ajustado e métricas auxiliares de previsão; acrescentada síntese externa M5. As duas tabelas externas essenciais foram resumidas localmente, preservando seus números e labels.
- Nenhum material foi acrescentado aos apêndices para simular redução. A tabela completa do piloto já existente no apêndice continua referenciada.

## Cortes efetivos

Retiradas as inclusões de distribuição por filial, receita por segmento, lojas por segmento, mapa nacional, boxplot de CTI, KPIs por segmento, heatmap por POD, diferença crítica e figura dupla de previsão. Cortados comentários que descreviam eixos e células repetindo legendas ou tabelas. Eliminados quadro de regime→política sem suporte coerente e resumos de quantidade de testes significativos/tamanhos de efeito associados a versões conflitantes do portfólio.

O piloto foi condensado em resultados, estratos de volume e diagnóstico de treinamento. A expansão BA foi organizada em comparação agregada, perfis e seleção; validação, previsão e alcance têm seções curtas. A leitura deixa de alternar entre várias visualizações da mesma comparação.

## Evidências preservadas

Mantidos PB com 15 lojas/produto 48130, horizonte de 38 ciclos, contraste de até 48% de custo e 55% de ruptura e sua perda de serviço; BA com 145 séries e 12 políticas; cinco repetições descritas como nominais; KPIs históricos; perfis 116/18/11; CTI ajustado das estratégias 895,11→862,64, redução de 3,6%, serviço 0,942→0,938; oráculo ajustado 441,43/0,700/50,7%; métricas de previsão. Não substituí esses valores pelos experimentos de cinco/seis políticas do novo artigo.

Normalizada somente a pontuação decimal dos quatro custos escritos com dois pontos (`1.147.86`, etc.) na tabela KPI. O valor numérico não mudou. A indicação de empate de BE foi incorporada ao texto, dispensando a nota de rodapé.

## Correções de interpretação

- Ganho do piloto não é economia universal do AIPE; GA-DQN custa mais e atende menos que EOQ na média BA.
- Ausência de menor custo e maior serviço simultâneos não prova que os perfis sejam úteis nem que todo candidato pertença à fronteira eficiente.
- Vendor-Responsive supera o limiar médio de 0,70 na própria tabela; removida a afirmação contraditória de que seu NS global estava abaixo desse limiar.
- Cinco repetições com dispersão praticamente nula não constituem cinco evidências independentes. Testes históricos são condicionados à reconstrução de unidade pareada, dependência e versão dos dados.
- Não-significância da diferença em NS não comprova equivalência ou ausência de perda. Oráculo ex post não demonstra eficácia do PSE.
- ANN tem o menor sMAPE da tabela, enquanto LSTM tem menores MASE/RMSSE; removidas interpretações contrárias aos valores e a alegação de que apenas LSTM superava um na referência RMSSE.
- Falhas de PPO/DQN/SARSA no recorte não identificam incapacidade estrutural de RL nem validam mecanismos causais de inicialização sem ablação.

## Pendências científicas expostas, sem alterar resultados

O texto registra separadamente CTI bruto histórico 6,2%, CTI ajustado 3,6% e sensibilidade por BE. Explicita conflito de caracterização integral Lumpy versus percentuais de uma antiga tabela, portfólios diferentes em resumos estatísticos, narrativa de significância conflitante com tabela e necessidade de fixar snapshots. Também distingue parâmetro L=2 de prazo efetivo dependente da ordem de eventos. Não gerei novas estatísticas para preencher essas lacunas.

## Labels e integração

Preservados: `cap:results`, `sec:experimento1`, `tab:kpis_fase2`, `fig:pareto`, `sec:comparacao_pod`, `tab:dominancia_por_perfil`, `sec:custo_politica_unica_vs_perfil`, `tab:strategy_cost_comparison`, `sec:stats`, `tab:forecast_metrics`, `sec:threats`.

Removidos: `tab:dataset_stats`, `sec:dataset_distribuicao_geografica`, `sec:estratificada_segmento`, `tab:regime_policy_dominance`, `tab:wilcoxon_resumo`, `tab:cohens_d_resumo` e labels das figuras excluídas (`fig:distribuicao_lojas`, `fig:receita_segmento`, `fig:lojas_segmento`, `fig:nacional_mapa_lojas_receita`, `fig:boxplot_tic`, `fig:kpi_segmento`, `fig:heatmap_cti_pod`, `fig:critical_diff`, `fig:forecast_ranking`, `fig:forecast_boxplot`). As tabelas externas que deixam de ser incluídas continuam preservadas no acervo; seus labels não ficam disponíveis pela compilação deste capítulo.

Antes da edição dos demais agentes, havia referência externa a `tab:regime_policy_dominance` em `conclusoes.tex` e a `fig:forecast_boxplot` em `conceitos.tex`. Avisado o coordenador: a primeira deve apontar à tabela por perfis; a segunda deve ser retirada ou apontar à tabela de previsão. Não alterei esses arquivos, que pertencem a outros agentes. A compilação final integrada deve conferir os labels das tabelas externas retiradas e o fluxo das cinco tabelas remanescentes.

## Incorporação auditada da base M5

Em atendimento à nova orientação, incluída a seção `sec:resultados_m5` e a tabela `tab:m5_sintese`, sem trocar os snapshots PB/BA por M5 ou pelo artigo. Lidos diretamente `experiment/data/07_model_output/m5/kpis.parquet`, cenários e perfis correspondentes; consultados `conf/m5/parameters.yml`, catálogo isolado, loader M5, divisão temporal e montagem dos KPIs no código.

Verificados 28.896 pares série–política sem duplicatas de chave, 2.408 séries, dez lojas, 287 produtos FOODS, três estados, 93 ciclos por série; grupos 2.350 Lumpy / 33 Erratic / 24 Smooth / 1 Intermittent. A tabela calcula média aritmética por política e normaliza somente pelo CTI médio EOQ no próprio M5. CSV com os valores de precisão completa: `m5_resumo_auditado.csv`.

A configuração atual declara treino 19 e avaliação restante, filtros FOODS/CV e suavização IQR. O loader usa informação da série completa nessa suavização e o recorte/arquivo não permite certificar sozinho um snapshot completo de execução. Essas ressalvas impedem chamar o benchmark de réplica exata de Zabraoui, validação externa do PSE ou teste temporal inteiramente livre de informação futura. Só SA/PPO alcançam NS médio 0,70 no resumo. Custos absolutos M5 e BA não foram comparados. A seção final deixa de chamar M5 apenas de estudo planejado e passa a indicar consolidação/reprodução dos resultados existentes.

SHA256 do KPI M5 consultado: `50b901f536171a77b453b0deb454bf85b7b2688d6dbafca84a608fa6f1f669c7`.

## Emenda curta após auditoria do código

Acrescentada ressalva M5 de partida sem demanda: 2.213/2.408 séries (91,9%) têm demanda zero nos primeiros 19 ciclos; todas as 2.408 têm demanda positiva nos 74 ciclos restantes, verificado nos cenários. Registrados orçamentos diferentes M5/BA (GA500/50, SA100/500, híbrido30/200), impedindo atribuir o contraste exclusivamente à base. A tabela de previsão conserva seus números, mas a legenda e interpretação agora explicitam escalas MASE/RMSSE calculadas pelas diferenças da série de teste e exclusão de 0/0 no sMAPE. Removida a conclusão de superação de previsão ingênua convencional por indicadores abaixo de um. A revisão inteira cresceu apenas seis palavras/tokens líquidos e segue no alvo.
