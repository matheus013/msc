# Diagnóstico de aceleração e poda — 14/10/2026

Este é o diagnóstico exploratório anterior à seleção. A recomendação atual de portfólio e suas ressalvas estão na [revisão dirigida de literatura](REVISAO_LITERATURA_POLITICAS.md) e no [registro do portfólio candidato](PORTFOLIO_REPRESENTATIVO.md).

A validação posterior do executor paralelo e do perfil para uso simultâneo da máquina está em [Desempenho de CPU e GPU](DESEMPENHO_CPU_GPU.md), incluindo a confirmação com orçamento completo de treinamento.

A fila censitária permanece interrompida por solicitação do usuário. Foram feitos somente diagnósticos curtos, em diretório separado, sem alterar o protocolo de produção, o portfólio ou os resultados antigos. Os testes de desempenho não são novas evidências de superioridade de políticas.

## O que os resultados antigos permitem cortar

A reanálise usa as matrizes preservadas: 145 séries da Bahia e 2.408 do M5, com 12 políticas. É uma triagem exploratória. O histórico tem problemas conhecidos de preparação, inicialização, seleção com informação futura e grande proporção de históricos sem demanda no treino M5. Esses resultados não provam que uma implementação corrigida continuará fraca.

**O candidato mais forte a sair do portfólio principal é DQN isolado.** Com meta de serviço de 70%, sua retirada aumenta o custo do melhor resultado retrospectivo por série em 0,105% na Bahia e 0,026% no M5, sem eliminar a viabilidade de nenhuma série nessas matrizes. Ele é o vencedor único em apenas uma série da Bahia e seis do M5 nessa meta. Para meta de 90%, a retirada elimina uma alternativa viável em uma série do M5; portanto, o efeito não é universalmente nulo. Essas comparações são oráculos retrospectivos, não estimativas de desempenho de um seletor implementável.

Nas 570 calibrações concluídas antes da interrupção, DQN isolado respondeu por 37,45% do tempo de treinamento registrado; DQN, PPO e GA-DQN juntos responderam por 96,41%. Essa composição parcial não representa uma previsão exata para todas as séries, mas identifica onde está o custo. Retirar heurísticas rápidas reduz o tamanho da tabela, porém quase não resolve o tempo de execução.

| Política | Evidência histórica | Recomendação preliminar |
|---|---|---|
| DQN | Pequena contribuição marginal ao oráculo; serviço médio de 22,5% na Bahia e 7,6% no M5; treinamento caro | Candidato a sair do portfólio principal após triagem corrigida; preservar como controle do híbrido |
| EOQ | Referência clássica; participa das escolhas de serviço elevado na Bahia | Manter |
| sS | Custos muito correlacionados com EOQ, mas existem vitórias locais | Candidato a simplificação editorial; confirmar redundância após correção temporal do EOQ |
| VendorResponsive | Escolhido em dois perfis da Bahia com meta de 70% | Manter |
| FixedInterval | Escolhido no perfil de tendência instável da Bahia com meta de 70% | Manter |
| CappedBaseStock | Escolhido no perfil de tendência instável do M5 com meta de 70% | Manter; média global esconde esse resultado |
| SA | Escolhido no maior perfil do M5 com meta de 70% | Manter |
| PPO | Necessário para metas altas de serviço no M5; forte contrapartida de custo | Manter para discutir a relação entre custo e serviço |
| GA-DQN | Escolhido no perfil de alto volume do M5 com meta de 70% | Manter na triagem; não descartar pela média global |
| GA | Dominado em médias na Bahia, mas possui 53 vitórias individuais únicas no M5 com meta de 70% | Não eliminar automaticamente; necessário também para decompor o híbrido |
| Newsvendor e BigDataNewsvendor | Possuem 19 e 12 vitórias individuais únicas na Bahia com meta de 70%; custo computacional baixo | Manter inicialmente na avaliação, condensar a apresentação se a validação não confirmar utilidade |

Uma poda direta para seis políticas (EOQ, VendorResponsive, FixedInterval, CappedBaseStock, SA e PPO) aumentou o custo do oráculo retrospectivo em **16,29% na Bahia e 11,86% no M5**, na meta de 70%. No M5, quatro séries que tinham alguma alternativa viável deixaram de ter alternativa nesse subconjunto. Logo, ainda não há suporte para chamar todas as políticas retiradas de irrelevantes. Adicionar o híbrido ao subconjunto reduz o aumento M5 para 5,32%, mas ainda perde uma série viável.

**Armadilha da redundância M5:** EOQ e sS têm custo e serviço iguais em 91,9% de todas as séries antigas. Porém, restringindo a históricos com pelo menos três ciclos positivos no treino, nenhuma das 171 séries tem esses dois KPIs exatamente iguais; a correlação de custos continua alta, em 0,991. A aparência de equivalência era fortemente influenciada pelo cold start. Correlação de custos também não prova igualdade das trajetórias de pedidos.

A poda editorial pode ocorrer antes da exclusão computacional: destacar os mecanismos de seis ou sete políticas no texto principal e guardar a matriz completa no suplemento. Se o artigo atribuir ganhos à combinação GA-DQN, as comparações com GA e DQN isolados continuam necessárias como ablações, em um desenho explicitamente definido. A seleção final de políticas deve ser congelada pela validação corrigida e depois avaliada no teste preservado.

## Onde o código está lento

1. `DoubleDQNAgent.train` faz uma chamada a `replay` por transição; aproximadamente 20 mil atualizações pequenas por ajuste. `replay` usa um `deque`, amostragem Python, listas e `np.stack`, recria cinco tensores e os transfere ao dispositivo em cada atualização.
2. A decisão DQN transfere um estado de seis números para a GPU, executa uma rede pequena e devolve o índice com `.item()`. A decisão PPO devolve índice, log-probabilidade e valor por `.item()` a cada ciclo.
3. `replay` também lê `loss.item()` duas vezes por atualização. As conversões CPU/GPU exigem sincronização; evitar essa rotina é uma recomendação explícita do [guia oficial de desempenho PyTorch](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).
4. O executor treina integralmente uma série antes da próxima. As redes não são atualizadas juntas; a GPU recebe milhares de tarefas pequenas em sequência.
5. O `BatchInventoryEnv` existente compartilha uma única demanda entre candidatos. Ele acelera GA/SA de uma série, mas não representa um conjunto de séries diferentes, cada qual com sua rede.
6. O híbrido executa novamente o GA, mesmo quando o ajuste GA correspondente já foi calculado para o mesmo treino e semente. Não há cache de ajustes para prefixos de treinamento iguais.

O perfil instrumentado do DQN CUDA, em um exemplo curto, atribuiu 4,27 dos 4,84 segundos de treinamento à chamada cumulativa de `replay`. Esses tempos têm overhead do perfilador e não devem ser usados como benchmark de produção.

## Medições nesta máquina

O diagnóstico usou os primeiros prefixos reais elegíveis de Bahia e M5, duas sementes, aproximadamente 2.048 transições por ajuste, aquecimento separado e uma thread PyTorch. Foram 32 ajustes curtos comparando dispositivo e cache do ambiente. Tempos abaixo são medianas das duas repetições, só do treinamento; não extrapolam automaticamente para 20 mil transições ou outras séries.

| Método / prefixo | CPU, segundos | CUDA, segundos | CUDA/CPU |
|---|---:|---:|---:|
| DQN / 17 ciclos | 2,92 | 9,02 | 3,10 |
| DQN / 50 ciclos | 2,84 | 9,14 | 3,22 |
| PPO / 17 ciclos | 1,67 | 6,65 | 3,99 |
| PPO / 50 ciclos | 1,01 | 4,29 | 4,26 |

**Para uma rede pequena por vez, a CPU foi mais rápida.** Selecionar CUDA automaticamente, sem considerar a forma de execução, foi inadequado para esse caminho. CPU/CUDA podem ter trajetórias diferentes por suas fontes de aleatoriedade e arredondamento; medir velocidade não prova igualdade estatística entre implementações.

Um protótipo que reutiliza o ambiente determinístico e pré-calcula as características exógenas do estado manteve igualdade exata de estados, recompensas e KPIs em uma trajetória de ações fixa para ambos os prefixos. O ganho foi pequeno no DQN e de aproximadamente 10% no PPO em CPU; não é a principal solução.

Também foi testada a atualização de **redes independentes**, empilhando parâmetros com `torch.func.stack_module_state`, `functional_call` e `vmap`, como no [exemplo oficial PyTorch](https://docs.pytorch.org/tutorials/intermediate/ensembling.html). Isso não transforma as redes em uma política compartilhada.

| Redes independentes | Tempo sequencial CUDA, 20 atualizações por rede | Tempo vetorizado | Ganho nas atualizações |
|---:|---:|---:|---:|
| 1 | 0,073 s | 0,107 s | 0,69× |
| 8 | 0,572 s | 0,113 s | 5,06× |
| 32 | 2,307 s | 0,096 s | 23,99× |
| 128 | 8,204 s | 0,109 s | 75,61× |

Cada rede manteve seus próprios parâmetros e momentos Adam. A perda foi somada entre redes, e o corte da norma do gradiente foi aplicado por rede. Foi verificada a proximidade dos pesos após uma atualização com a versão sequencial; a diferença máxima observada foi aproximadamente 2,2×10⁻⁶. Isso ainda não valida trajetórias completas de treinamento.

**O ganho de 75,61× é apenas das atualizações Double DQN:** o teste não inclui coleta de episódios, buffer de replay, exploração, atualização da rede alvo ou persistência dos resultados. Não é uma promessa de aceleração da execução inteira. Apenas aumentar o lote de uma única rede também não é equivalente a empilhar várias redes independentes.

## Reutilizar treinamento sem excluir nenhuma série

Prefixos de treinamento exatamente iguais e configurações, método e semente iguais geram o mesmo problema de ajuste. É possível ajustar uma vez e avaliar todas as séries correspondentes em suas próprias janelas de validação e teste. Isso preserva a população avaliada e não mistura históricos de séries diferentes em uma política compartilhada.

| Base / origem | Casos de série/origem | Prefixos de treinamento distintos |
|---|---:|---:|
| Brasil / 17 | 615.250 | 224.520 |
| Brasil / 23 | 1.010.101 | 461.262 |
| M5 / 50 | 11.780 | 11.780 |
| M5 / 64 | 13.231 | 13.231 |
| M5 / 78 | 14.192 | 14.189 |

No total, há uma redução potencial de aproximadamente **2,30× no número de ajustes**, antes de ponderar algoritmos, custo por origem e possibilidade de cache. A chave precisa incluir o vetor completo de treino, configuração completa, semente, método e versão de código; deve verificar igualdade dos dados em caso de chave coincidente. Validação, teste, estoque inicial e peso estatístico de cada série permanecem separados. Nenhuma transação ou série deve ser excluída para obter essa reutilização.

## Ordem recomendada para a implementação

1. Implementar cache de ajustes por prefixo/configuração/semente e reaproveitar parâmetros GA no híbrido. Manter atribuição do custo GA ao comparar esforço metodológico do híbrido.
2. Implementar treino DQN em lotes de redes independentes, inicialmente 32 ou 128, com ambiente de demandas `[rede, ciclo]`, replay pré-alocado, escalas e estoques por rede. Fazer toda a trajetória permanecer no dispositivo.
3. Preservar Adam, clipping, epsilon, alvo, amostragem do replay e orçamento de atualizações por rede. Não usar clipping global nem média de perdas entre redes. Definir RNG por tarefa e testes que detectem interferência entre redes e acesso a demanda futura.
4. Repetir o mesmo desenho para PPO, sem misturar vantagens e normalização de séries diferentes. Medir treinamento completo, memória e resultados pareados em pequenos blocos, antes de qualquer execução censitária.
5. Na validação corrigida, testar retirada do DQN do portfólio principal. Não eliminar políticas baratas só para acelerar. Se necessário, escolher orçamento de aprendizado com curvas de convergência em dados de calibração, sem consultar o teste final.
6. Congelar o novo protocolo e uma nova rodada. Os resultados interrompidos servem para auditoria e diagnóstico; uma implementação diferente não deve ser misturada silenciosamente à rodada anterior.

Treinar uma política compartilhada por perfil/UF é outra possibilidade, mas modifica a pergunta científica e exige tratamento das diferentes escalas, custos e partições de treino. Não é uma simples otimização do executor atual. Também não foi testada a vazão de múltiplos processos CPU, compilação ou CUDA Graphs; esses ganhos não devem ser inventados.

Antes de acelerar o híbrido, há ainda um ponto de correção: em `prepopulate_from_ga`, o código armazena o índice da ação discreta mais próxima de Q, mas executa Q contínuo no ambiente. Se Q difere do valor representado por esse índice, a transição fica associada a uma ação diferente da executada. As demonstrações também não recebem o mesmo ajuste terminal de serviço usado no treino DQN. Esses problemas devem ser separados das otimizações e testados; não foram corrigidos nesta análise.

## Artefatos para reprodução

- `experiment/scripts/analysis/benchmark_rl_acceleration.py`: medições CPU/CUDA, protótipo do ambiente e teste de atualizações independentes;
- `experiment/data/08_reporting/performance/analysis_2026-10-14/`: tempos, perfis, manifesto e contagem de prefixos;
- `experiment/scripts/analysis/assess_policy_pruning.py`: triagem histórica por média, perfil e sensibilidade do oráculo;
- `experiment/data/08_reporting/performance/historical_pruning_2026-10-14/`: tabelas e hashes das fontes.

Esses scripts não iniciam nem retomam a fila censitária.
