# Auditoria científica do código e das duas bases

Data: 14/10/2026. **A existência do benchmark M5 é confirmada; o código e os artefatos não sustentam que PSE, pooling ou previsão tenham sido validados juntamente com esse benchmark.** O artigo em português pode integrar os resultados preservados das duas bases como comparação condicional de políticas e seletores retrospectivos, preservando diferenças de protocolo e limitações. Não deve apresentá-los como reprodução exata de Zabraoui ou validação prospectiva do AIPE completo.

## Cobertura e método de inspeção

`code_inventory_ptbr.json` inventaria **96 arquivos, 17.518 linhas: 76 fontes Python, 19 YAML e um arquivo YAML com extensão alternativa**. O escopo inclui todos os arquivos Python de `experiment/src` (incluindo o pacote `reporting`), scripts, testes, `run.py` e configurações. Cada arquivo foi lido integralmente pelo inventário, analisado estruturalmente quando Python e identificado por SHA256, definições e chamadas; as configurações estão transcritas no inventário. A inspeção científica manual acompanhou os caminhos de ingestão, preparação, perfil, calibração, otimização/RL, avaliação, estatística, PSE e relatórios. Isso não é execução de cada ramificação nem certificação de correção linha a linha. Bibliotecas externas, ambiente virtual, cache e dados brutos não são código do projeto e ficaram fora desse escopo.

Não alterei o código experimental nem reexecutei o treinamento neural completo nesta auditoria. Li os KPIs e os cenários preservados para verificar população, cobertura, dispersão e coerência com as configurações. Os scripts novos de reanálise são identificados separadamente dos geradores de resultados históricos.

## O que cada execução realmente contém

Fonte: `experiment/src/simulation/pipeline_registry.py` e os arquivos `pipeline.py`.

| Pipeline | Componentes efetivos | Não presumir |
|---|---|---|
| `__default__` | ingestão + previsão + perfil + políticas + estatística + PSE + relatórios | Que as previsões estejam conectadas às decisões do inventário |
| `benchmark_final` | retomada da base interna + perfil + políticas + estatística + relatório final | Que treine PSE ou execute previsão |
| `benchmark_m5 --env m5` | ingestão M5 + perfil + políticas + relatório final | PSE, previsão e validação estatística não fazem parte desse grafo |
| `benchmark_bot --env bot` | base interna multiestado + perfil + políticas + relatório final | Não é o recorte BA nem um novo tipo de dado |
| `profile_analysis` | recalcula perfil + analisa políticas/estratégias + treina PSE | Não reexecuta os controladores; consome matriz existente |
| scripts PB | preparo e execução independente com configuração própria | Não equivalem ao protocolo Kedro BA/M5 atual |
| `sca_validation`, `sca_dual_dataset` | seleção e reamostragem das matrizes arquivadas | Não treinam agentes nem removem informação futura dos resultados |
| `sca_temporal` | diagnóstico novo de heurísticas determinísticas com divisão temporal | Não valida controladores RL históricos nem o PSE supervisionado |

Os módulos de coleta bibliográfica, construção de PDFs, mapas, amostras e visualização foram inventariados, mas não produzem evidência experimental de desempenho de políticas. Mapas nacionais e listas de dados disponíveis não equivalem a validação nacional.

## Dados, amostra e horizontes

**Base real/BA:** a matriz atual tem 145 séries, 115 lojas, nove produtos e doze políticas, com 1.740 linhas completas. Os cenários têm 38 ciclos. A configuração-base usa prefixo de treino 17 e bloco de avaliação restante, compatível com 21 ciclos nos indicadores; o campo `n_periods=38` é metadado da série, não prova de avaliação em 38 ciclos. A matriz atual diverge da lista de políticas descrita no relatório de junho/qualificação.

**M5:** `data/07_model_output/m5/kpis.parquet` tem 2.408 séries, dez lojas e doze políticas, com 28.896 linhas e cobertura completa por série. Os cenários têm 223.944 linhas e 93 ciclos. A configuração M5 usa treino 19 e avaliação dos 74 restantes. O loader agrega 1.941 dias em blocos de 21; o último bloco contém nove dias, portanto não é um ciclo completo comparável aos demais. A agregação não autoriza comparar custos absolutos de 74 ciclos M5 com os de 21 ciclos BA.

`m5_loader.py` restringe FOODS na configuração atual, agrega por ciclo e trunca a cauda superior em Q3+1,5×IQR **usando a série inteira antes da separação temporal**. Isso altera a demanda usada para avaliar e incorpora informação futura na preparação. A seleção por CV em `data_ingestion/nodes.py::filter_by_parameters` é calculada por `item_id`, agrupando lojas/ciclos, não individualmente por série loja–produto. O limiar 1,5 não reproduz uma regra de ruptura histórica: vendas M5 não trazem uma indicação direta de stockout.

Na base real, `clean_sales_data` usa `drop_duplicates` pela chave armazém/produto/loja/ciclo antes da agregação. Se linhas da mesma chave forem transações complementares, manter a primeira perde demanda. O próprio comentário em `duckdb_loader.py` documenta valores diferentes entre duplicatas. A extensão dessa diferença precisa ser quantificada na auditoria de dados; não presumir que duplicidade técnica seja uma transação repetida. Os filtros de atividade e CV usam o horizonte completo e a preparação inclui zeros para ciclos ausentes, com possível censura por falta ou incompletude de registro.

## Algoritmos, ajuste e aleatoriedade

O núcleo efetivo está em `core/policies.py`, `policies_sota.py`, `policies_zabraoui.py`, `metaheuristics_torch.py`, `rl_torch.py` e nos ambientes escalar/batch. `policies.py` fornece regras clássicas e aliases para as implementações PyTorch. GA/SA procuram parâmetros de regra de limiar, não representam modelos independentes de cadeia. DQN é Double DQN dueling, PPO usa GAE com ações discretizadas e SARSA atual é Expected SARSA. Os comentários históricos e os nomes de arquivos não substituem essa identificação.

`inventory_simulation/nodes.py::_worker_generic` treina/otimiza uma vez por política e série, depois chama `InventoryEnv.run_policy`. Este repete a demanda fixa, reutiliza o mesmo controlador e não usa `base_seed` para novos treinamentos. O máximo `TIC_std` é essencialmente zero: BA zero e M5 5,82×10⁻¹¹, ruído aritmético. **Não existem cinco treinamentos independentes por célula.** As sementes 42 fixam geradores da busca ou inicialização dos agentes; `seed=ep` do ambiente não gera novas demandas. A avaliação das políticas neurais usa decisão sem exploração.

Há diferenças substanciais de orçamento entre bases, por mesclagem recursiva das configurações (`settings.py`):

| Parâmetro | BA/base | M5 |
|---|---:|---:|
| GA, gerações | 50 | 500 |
| SA, iterações | 500 | 100 |
| DQN/PPO, episódios | 1.000 | 1.000 |
| Híbrido, episódios RL | 200 | 30 |
| Treino, ciclos | 17 | 19 |

As famílias compartilham ambiente, mas **não orçamento computacional igual**. A população GA de 100 torna o orçamento M5 significativamente maior; comparações de escalabilidade exigem tempo efetivamente medido, não extrapolações em comentários. Versões recentes de operadores/penalidades não comprovam o código exato usado para produzir arquivos antigos.

O híbrido Kedro faz busca GA, pré-população de mil transições e novo ajuste RL. O script PB `_run_single`, por sua vez, treina DQN/PPO isolados e depois os encapsula com parâmetros GA; não chama a mesma pré-população. Assim, a explicação causal histórica de PB por warm-start do replay não está comprovada por esse script. Métodos com `demand_pool` existem nas classes, mas os workers do benchmark atual não fornecem esse argumento: capacidade de pooling no núcleo não significa que todas as execuções usem pooling.

## Dinâmica e vazamento residual

O ambiente pede antes da chegada, recebe a primeira posição da fila, acrescenta o novo pedido, atende demanda com vendas perdidas e cobra manutenção sobre estoque final. Fila L+1 implica chegada t+3 quando L=2. Estoque inicial gratuito, fila vazia e ausência de crédito terminal influenciam resultados em horizontes curtos. O armazém é um identificador; não há estoque compartilhado ou restrição de capacidade que configure otimização integrada de múltiplos escalões.

`_split_demand` separa um prefixo e um bloco final: o modo chamado `walkforward` na simulação **não refaz treinamento a cada origem**. `scale_parameters_per_store` usa média/desvio de `scenarios_meta` derivados da série completa para estoque inicial e limites de ação. `InventoryEnv` calcula normalizadores pela própria demanda de avaliação e usa média global na primeira previsão ingênua/estado inicial. Os perfis também são calculados no horizonte completo. Uma divisão nominal treino/teste não elimina esses canais de informação futura.

EOQ e `(s,S)` legados usam lote calculado com demanda acumulada no treino, apesar de manutenção por ciclo. O Jornaleiro é heurística normal com z fixo, não uma solução de razão crítica empiricamente calibrada. VendorResponsive e FixedInterval são interpretações locais de descrições pouco especificadas. Referências fortes da literatura não validam automaticamente a fidelidade de CappedBaseStock, PIL ou BigDataNewsvendor; devem-se confrontar hipóteses e formulações, inclusive fallback e regulação de viabilidade.

## Previsão e métricas

O grafo `inventory_simulation/pipeline.py` não recebe `forecast_predictions` nem os modelos de previsão. Seus workers não fornecem `forecast_series` ao ambiente. Logo, não atribuir o desempenho dos controladores a LSTM/XGBoost/Croston apenas porque o projeto contém uma etapa de previsão. O ambiente usa o sinal ingênuo quando esse argumento está ausente.

`demand_forecasting/nodes.py` oferece divisão única sequencial e janela expansível; o padrão é divisão única. `_metrics` calcula escala MASE/RMSSE por diferenças de `y_true` **do teste**, em vez da escala ingênua do treino. Métricas por um único ciclo têm denominador indefinido. O sMAPE descarta os pares (0,0), mudando o denominador da média. Esses detalhes impedem associar os artefatos sem ressalva às equações convencionais da dissertação.

`LSTMNumpy` preserva a implementação histórica, com aproximação de treinamento sem BPTT integral; `LSTMTorch` foi incorporada depois e usa autograd. Uma nova execução com backend PyTorch não comprova os antigos resultados de previsão. ANN e XGBoost têm suas próprias inicializações e versões. A auditoria atual não verifica todos os valores de métricas preexistentes contra uma nova execução.

## Perfil, PSE e inferência

Os perfis usam regras hierárquicas de ADI, CV² sobre todos os ciclos, sequências de zeros, burstiness, entropia, tendência e ACF. A coluna `dominant_policy` é um mapeamento fixo de perfil para controlador, não a política ótima aprendida. `generate_policy_labels` usa, separadamente, o menor custo por série sob NS mínimo. A seleção por perfil da reanálise usa a média do grupo; são objetos diferentes.

`train_policy_selector` realmente ajusta XGBoost/alternativas na base entregue. Entretanto, `_build_cv` usa **LeaveOneOut por série como padrão**, mesmo quando `cv_folds=5` aparece no YAML; classes com menos de dois exemplos são excluídas da CV, mas incluídas no ajuste final. O relatório por classe deriva de `predict(X)` depois de `fit(X,y)` em toda a amostra. Não é desempenho fora do ajuste. O desvio de acurácia entre folds LOO é dispersão de acertos binários, não cinco réplicas de treinamento. A matriz BA registra CV aproximada 0,4476; acurácia aparente de treino próxima de 0,986 não valida recomendações operacionais.

O diretório atual de KPIs M5 contém apenas os sete arquivos de benchmark por família/agregado, sem métricas/modelo/recomendações PSE materializados ali. O catálogo possui destinos para eles, mas um destino configurado não comprova execução. `benchmark_m5` não chama esse treinamento. A análise de redundância gera pares no relatório final, mas seu resultado não é entrada de `generate_policy_labels`; não afirmar agrupamento automático de classes já efetivado.

Wilcoxon aplica Holm no código estatístico principal, mas a tabela LaTeX genérica usa a coluna de significância bruta. Friedman/Nemenyi operam sobre séries completas e não tratam clusters loja/produto. Rankings ascendentes também exigem cuidado com NS, cuja direção favorável é crescente. Cohen usa desvio combinado de grupos, não covariância pareada. A reanálise nova usa bootstrap por loja, condicional a seletores ajustados; M5 tem somente dez clusters e dependência entre lojas por produto.

## Relatórios que podem alterar a leitura sem alterar os dados

Há pelo menos três objetos chamados custo ajustado: (i) CTI normalizado mais λ×BE normalizado; (ii) TIC mais déficit NS×10×custo-referência e penalidade de excesso de manutenção; (iii) aptidão com pesos de custo/serviço e termos de TR/BE no núcleo. Não unir tabelas desses objetivos como se fossem a mesma métrica.

`reporting/nodes.py::POLICY_ORDER` ainda privilegia o portfólio antigo: tabelas genéricas podem omitir os quatro candidatos adicionais. `profile_policy_analysis.py` inclui a lista ampliada, mas marca DQN/PPO como degenerados por constante histórica. Uma marca fixa não diagnostica a nova execução. Os desvios em tabelas globais são calculados entre séries, não entre treinamentos independentes.

Wrappers de perfil e estratégia passam DataFrames/out_dir ao pacote reporting, respeitando o ambiente. Já os wrappers de custo ajustado e apresentação chamam `run()` sem usar suas entradas; scripts podem reler caminhos padrão BA e produzir figuras da base errada ao executar sob M5. O catálogo isolado, sozinho, não elimina leituras/escritas internas hardcoded. Relatórios narrativos antigos, portanto, precisam ser confrontados com o manifesto de dados usado em cada tabela.

## Implicações para o artigo real + M5 em português

1. Definir a contribuição como **avaliação auditada da escolha de políticas em duas coortes**, não como PSE treinado e validado em duas bases.
2. Mostrar a matriz e o melhor global viável de cada base com custo e NS juntos. Na M5, GA–DQN tem serviço médio 0,4666, abaixo de 0,70; SA e PPO são os únicos candidatos com média global acima do limiar principal. Não usar o baixo custo dos inviáveis para alegar superioridade.
3. Separar horizontes, preparação, orçamento e população; normalizar custos relativos dentro de cada base quando comparar padrões. Não comparar diretamente valores monetários absolutos entre redes e durações diferentes.
4. Identificar a reanálise leave-one-store-out como transferência condicional da regra de seleção, com perfis/calibração históricos. Não descrevê-la como validação temporal integral.
5. Não importar as tabelas de junho ou imagens antigas para o texto novo sem reproduzir sua aritmética sobre as matrizes identificadas. Preservar resultados negativos e médias de serviço abaixo do alvo.
6. Tratar o diagnóstico temporal novo como análise suplementar de heurísticas, distinguindo portfólio original de cinco e emenda de seis candidatos. Seus resultados não regeneram RL/M5.
7. Para uma contribuição prospectiva forte, criar uma execução congelada que estime preparação/perfil/escalas somente no treino, selecione na validação, avalie no teste e treine métodos estocásticos com sementes independentes. Essa execução não foi realizada nesta auditoria.

**Decisão:** evidência suficiente para escrever um artigo exploratório rigorosamente delimitado com as duas bases; insuficiente para afirmar ganhos operacionais generalizáveis, reprodução exata da literatura ou validação do AIPE supervisionado. Nenhum resultado foi alterado para produzir uma narrativa favorável.
