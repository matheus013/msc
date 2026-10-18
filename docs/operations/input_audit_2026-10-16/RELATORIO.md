# Auditoria das entradas do AIPE — Bahia e M5

16/10/2026. Escopo: confrontar o contrato de entradas aprovado com arquivos locais, painéis congelados, código utilizado e persistência dos experimentos. Não foi feita nova análise de superioridade das políticas, treinamento, simulação ou alteração da fila em execução.

## Conclusão

As duas bases permitem avaliar a seleção de famílias de reposição sobre históricos reais de vendas, em um ambiente de estoque simulado com condições declaradas. Não permitem, na configuração atual, medir economia monetária efetivamente realizada pela empresa nem emitir uma ordem de compra baseada em seu estado operacional real. O caráter real da origem das vendas não torna custos, prazos e estoques simulados em observações operacionais.

Há dados brasileiros de estoque no acervo que ainda não são usados. Isso amplia as possibilidades de investigação futura, mas não resolve o estado inicial dos testes: o único dia disponível é 21/05/2025, posterior às janelas de teste da Bahia, e sua unidade é franquia–material, enquanto a demanda avaliada é revendedor–produto.

## Como foi verificado

- Inventário dos esquemas de todos os Parquet das seis fontes brasileiras e cabeçalhos dos oito CSV M5; sem exportar nomes de pessoas ou registros individuais.
- Leitura dos cinco painéis congelados e dos respectivos manifestos. Verificação de hashes, unicidade, valores finitos/não negativos, elegibilidade pelo prefixo e disponibilidade dos blocos.
- Verificação de 12 cópias congeladas de dependências científicas, seis por base, contra os hashes de seus planos. Todas também coincidem com os arquivos atuais selecionados.
- Reconciliação das quantidades do painel Bahia com as vendas brutas da coorte e do maior painel M5 com somas de 21 dias dos CSV originais. Diferença máxima zero em ambas. Nos painéis menores M5, foram verificados integridade e elegibilidade; a reconciliação completa de quantidades usou a coorte maior, que contém os mesmos IDs.
- Conferência dos hashes dos arquivos brutos consumidos de vendas Bahia e M5 contra os manifestos.
- Leitura de uma amostra de payload persistido de cada base, do código de gravação e dos cabeçalhos das saídas de seleção Bahia. A amostra não constitui auditoria exaustiva dos payloads.
- 22 verificações de integridade passaram. Não foram executados novos testes do pipeline nem atestada correção integral do experimento.

Script reproduzível: [audit_inputs.py](audit_inputs.py). Evidência agregada local: [evidence.json](../../../experiment/data/audits/input_contract_2026-10-16/evidence.json). O script usa um thread DuckDB, limite de memória de 768 MiB, SQLite em modo somente leitura e não importa o código científico para executá-lo. Essas restrições não limitam o consumo de outros aplicativos ou da fila existente.

## Matriz de disponibilidade e utilização

| Entrada do contrato | Bahia | M5 | Classificação e consequência |
|---|---|---|---|
| Produto e contexto de estoque | `produto_cod`, `revendedor_cod`, UF | `item_id`, `store_id`, `state_id` | Observados; local de demanda não equivale necessariamente ao nível onde o estoque físico é mantido. |
| Tempo e unidades | Data de venda e ciclo comercial; quantidade vendida | Vendas diárias, agregadas em 21 dias | Observados/transformados. A unidade comercial exata deve ser documentada; não presumir caixas ou equivalência entre bases. |
| Histórico de demanda | Quantidades de venda usadas como demanda do simulador | Quantidades de venda usadas como demanda do simulador | Observado como vendas; a equivalência com demanda não censurada é hipótese. Não há recuperação de procura perdida por falta. |
| Perfil/características | Calculados no prefixo | Calculados no prefixo | Estimados/derivados. Não são perfil fornecido pelo gerente; esse modo de entrada não está validado pela rodada atual. |
| Prazo de reposição | 2 ciclos, convenção `exact` | 2 ciclos, convenção `exact` | Assumido em cenário, fixo. Não estimado de pedidos/recebimentos. No M5 corresponde a 42 dias; no Brasil são dois ciclos comerciais, sem garantia de 42 dias exatos. |
| Custos de estoque, falta e pedido | h=1, p=5, K=50, c=0,5 | Mesmos valores | Assumidos em cenário. Não são reais ou dólares calibrados. h incide por unidade em saldo ao fim do ciclo; p por unidade não atendida; K por pedido positivo; c por unidade pedida. |
| Meta de atendimento | Fill rate 0,70 | Fill rate 0,70 | Assumida pelo protocolo; não coletada com o gerente. Configuração nominal de seleção e penalidade, sem garantia de cumprimento. |
| Saldo inicial | Derivado do prefixo e de aquecimento com EOQ | Mesmo procedimento | Estado sintético comum às candidatas; não saldo observado na empresa. |
| Pedidos em trânsito e compromissos | Pipeline gerado pela EOQ de referência | Mesmo procedimento | Simulados. Não histórico de pedidos reais; compromissos reais de clientes não são incorporados. |
| Restrições de fornecedores | Não consumidas | Não disponíveis no conjunto inspecionado | Não há ingestão de lotes mínimos, múltiplos, capacidade real ou calendário do fornecedor. Limites de busca/ação dos algoritmos não equivalem a restrições físicas observadas. |
| Política atual da empresa | Não identificada nos dados usados | Não identificada nos dados usados | Global é referência metodológica estimada na validação, não prática observada do gerente. |
| Receita/preço | Receita bruta e líquida disponíveis nas vendas, fora da configuração de custo | `sell_prices.csv` disponível, fora da configuração de custo | Disponível, não utilizado como custo de aquisição/manutenção/falta. Preço ou receita não substituem esses custos. |
| Perfil fornecido e avaliação humana | Não coletados nesta execução | Não coletados nesta execução | Requisitos metodológicos a avaliar; não há demonstração de compreensão, adoção ou impacto de uma decisão humana. |

Evidência de consumo: [configuração e estados iniciais](../../../experiment/src/simulation/research/protocol.py), [orçamentos de calibração](../../../experiment/src/simulation/research/production.py), [worker e campos devolvidos](../../../experiment/src/simulation/research/parallel.py), [dinâmica e componentes de custo](../../../experiment/src/simulation/core/inventory_env.py). As cópias congeladas verificadas estão nas pastas `code_snapshot` dos respectivos experimentos.

## Coortes e transformações

Bahia: 145 séries, 115 revendedores e nove produtos; 38 ciclos, de 202301 a 202504. Origens 17 e 23, com sete ciclos de validação e sete de teste. Os testes são 202408–202414 e 202414–202503; portanto há sobreposição no ciclo 202414. A coorte é histórica e fixa, não toda a população elegível da Bahia. O campo `sampled:false` isolado não autoriza chamá-la de censo.

Na coorte Bahia há 5.741 linhas de venda e 5.510 células série–ciclo. Dessas células, 2.697 (48,9%) não têm agregado observado e são preenchidas com zero. A reconciliação comprova a transformação aplicada; não comprova que toda ausência representa demanda nula. É necessário distinguir ausência de venda, falta de produto e falha de captura. Na coorte inspecionada, não houve quantidade negativa ou conversão inválida; o caminho de ingestão filtra esses casos quando presentes.

M5: coorte histórica de 2.408 séries, dez lojas e 287 itens no maior painel. Elegíveis por origem: 885 em 50, 2.033 em 64 e 2.408 em 78. Não são todas as 14.370 séries FOODS. A pertinência à coorte veio de um artefato histórico de perfis; o próprio preparador registra seleção retrospectiva e adaptação local do recorte Zabraoui. A exigência de três ciclos positivos no prefixo evita um filtro futuro adicional nessa etapa, mas não torna a seleção original prospectiva.

São 92 blocos completos de 21 dias, totalizando 1.932 dias; nove dias finais das 1.941 colunas de treino são descartados pela opção de blocos completos. Suavização IQR está desativada nesse caminho. Os rótulos YYYYCC são sintéticos, não correspondem diretamente a datas reais de calendário. O calendário local não contém coluna `d`: a auditoria usou a posição diária cronológica, conferida como consecutiva, para mapear `d_1` à primeira data.

| Origem M5 | Treino até | Validação real | Teste real |
|---|---|---|---|
| 50 | 13/12/2013 | 14/12/2013–09/05/2014 | 10/05/2014–03/10/2014 |
| 64 | 03/10/2014 | 04/10/2014–27/02/2015 | 28/02/2015–24/07/2015 |
| 78 | 24/07/2015 | 25/07/2015–18/12/2015 | 19/12/2015–13/05/2016 |

O teste dura sete ciclos, ou 147 dias no M5; não sete dias. As origens sucessivas reutilizam períodos anteriores como passado, sem constituir séries independentes. No Brasil, manter os códigos comerciais e consultar `venda_data` quando precisar de limites civis; não converter ciclo em mês.

Evidências: [leitura e transformação das fontes](../../../experiment/scripts/analysis/run_production_v2.py), [delimitação da coorte histórica](../../../experiment/scripts/analysis/run_scoped_production.py), [agregação M5](../../../experiment/src/simulation/pipelines/data_ingestion/m5_loader.py). O preparador genérico possui caminhos mais amplos; a execução canônica reutiliza os painéis congelados da coorte histórica.

## O estoque brasileiro encontrado

`data/source/estoque`: 16 arquivos, 136.789 registros, 51 franquias e 7.339 materiais no conjunto inteiro. Todos os registros têm a mesma data, 21/05/2025. Há `stk_total` e `stk_disponivel`; não foram encontrados saldos negativos ou ausentes no campo disponível. Isso não confirma a qualidade operacional dos saldos nem sua distribuição por Bahia.

`data/source/estoques_rev`: tabela de ligação entre revendedor e estoques, sem datas de vigência ou quantidades físicas em seu esquema. Os 145 pares revendedor–produto da coorte encontram ao menos um vínculo com franquia e material do estoque; todos os revendedores desses pares têm múltiplos códigos de estoque no mapeamento. Não há coincidência direta revendedor=franquia com material correspondente para nenhum dos 145 pares.

Essa cobertura de códigos é uma pista de integração, não prova de que cada saldo pertence ao revendedor ou estava disponível em todas as datas históricas. Antes de integrar: esclarecer o significado de `estoque_coluna`, a prioridade/validade dos vínculos, o nível físico da decisão e a correspondência produto–material. Não somar saldos de franquias nem replicá-los por revendedor sem regra operacional validada.

O cadastro de produtos também oferece `produto_cod_atual`, mas uma migração de código pode exigir validade temporal e correspondência de unidade. Sua existência não autoriza substituir códigos silenciosamente. Estoque de franquia compartilhado exigiria representar alocação e capacidade conjunta; o ambiente atual simula séries separadamente. Os comentários antigos sobre múltiplos níveis no ambiente não comprovam uma rede de estoques acoplada na execução.

## Rastreabilidade: o que existe e o que falta

Já existe: hashes de fontes, painéis e código, snapshot de dependências, identificação de tarefa e semente, configuração por ajuste, métricas de validação/teste e tempos. A seleção Bahia exporta recomendações antes da pontuação, referência, candidata, margem, intervenção e previsões de custo/atendimento nas estratégias por consequências. Os indicadores previstos não estão preenchidos uniformemente para todas as estratégias.

O pipeline persiste TIC, NS, TR, BE, FP, demanda, quantidade atendida e unidades comprometidas ao final, além de identificadores. O ambiente calcula custos desagregados e mantém trajetórias em memória, porém o worker descarta esses detalhes ao gravar o resultado. Saldo e pipeline efetivos do início de validação/teste são reconstruíveis pelo procedimento, mas não são gravados explicitamente como esses estados nos payloads amostrados.

Faltam na entrega canônica completa: parâmetros finais de todas as políticas, pesos dos agentes aprendidos e artefatos de seletores para aplicação, trajetórias por ciclo, custos desagregados por caso, origem/qualidade por entrada e um relatório de decisão uniforme para o gerente. Parâmetros GA usados para inicializar GA–DQN podem aparecer na configuração desse híbrido; isso não equivale à serialização de todas as políticas e redes. A existência de outros caminhos legados de persistência não supre a ausência no caminho canônico inspecionado.

Portanto, há boa rastreabilidade computacional parcial, mas ainda não a entrega integral do contrato recém-aprovado. Como o ajuste atual usa somente demanda/características e custos fixos, não demonstra que o seletor aprende a adaptar-se a diferentes custos ou prazos fornecidos pelo gerente. Essas variações precisam de validação própria.

## Capacidades sustentadas pelo desenho e limites

| Pode ser avaliado com a configuração atual | Exige evidência adicional |
|---|---|
| Selecionar famílias de reposição em históricos reais de vendas sob custos e dinâmica declarados. | Economia efetivamente realizada em reais, custos reais de implantação e impacto financeiro observado. |
| Comparar PSE por perfil derivado e estimado, com controle da informação de treino. | Perfil externo fornecido pelo gerente e seu tratamento quando divergir do perfil derivado. |
| Medir custo simulado, fill rate, deteriorações, referências e limites retrospectivos. | Pedidos executáveis com saldo real, compromissos, recebimentos e restrições de fornecedores. |
| Examinar transferência entre contextos amostrados e períodos, com as limitações da coorte. | Generalização para toda Bahia, todo Brasil, todas FOODS ou novas condições operacionais não testadas. |
| Comparar algumas instanciações de políticas e seletores. | Substituição arbitrária de qualquer módulo, demanda censurada corrigida, rede física com estoques compartilhados ou uso comprovado por gerentes. |

## Próximas ações, sem reiniciar a execução

1. Incorporar essas classificações no protocolo e preservar os custos e estados como cenários explícitos. A execução atual continua útil para esse escopo.
2. Especificar a saída mínima por decisão: recomendação, alternativas, estimativas admissíveis, hipóteses e vínculo ao registro experimental. Planejar exportação dos dados já persistidos sem retreinar controladores.
3. Separar enriquecimento reconstruível do que exige replay/novo ajuste. Custos desagregados e trajetórias ausentes não devem ser inventados por redistribuição do custo total; políticas não serializadas podem exigir novo ajuste determinístico ou retreinamento para recuperar ações. Registrar qualquer recuperação como execução adicional, sem substituir checkpoints silenciosamente.
4. Validar a semântica do estoque brasileiro e obter séries temporais de saldos/recebimentos e restrições, se o uso imediato de compra fizer parte da próxima validação. A fotografia de maio/2025 fica fora dos estados iniciais históricos atuais.
5. Auditar as ausências preenchidas por zero e registrar sua interpretação. Para novos dados, distinguir lacuna de captura de zero observado; qualquer mudança científica deve ser versionada fora da fila congelada.
6. Antes de habilitar receitas M5, adaptar e verificar a integração opcional: `_attach_revenue` espera uma coluna `d` ausente no calendário local. Esse caminho está desativado (`with_revenue:false`), portanto não afeta os ajustes em andamento; preço de venda tampouco substitui custos logísticos.

Não há, por esta auditoria, motivo para zerar M5 ou repetir integralmente Bahia. A necessidade de execuções adicionais depende da alegação pretendida e das saídas que forem priorizadas, não apenas da existência de mais dados no acervo.
