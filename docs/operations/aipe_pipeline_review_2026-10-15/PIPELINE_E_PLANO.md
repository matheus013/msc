# Revisão do pipeline AIPE e plano de execução — 15/10/2026

## Escopo e estado antes do início

Foi feita varredura estrutural dos 123 arquivos Python ativos de src, scripts e tests, com análise sintática e procura de corpos de funções exatamente duplicados. A revisão funcional concentrou-se no caminho científico: preparação e elegibilidade, configuração dos controladores, fila, reaproveitamento, seleção, avaliação e relatórios. Isso não é uma prova formal de correção de cada linha de todos os módulos legados.

Não havia processo Python de experimento ativo na inspeção inicial. Os planos foram preparados sem novos ajustes de estoque. A execução autorizada começa somente depois da apresentação deste relatório, pela Bahia e depois pelo M5; Brasil permanece excluído.

## Mudanças realizadas

1. Entrada atual `research-aipe`, com fases `plan`, `execute` e `evaluate`. Planejar não treina; avaliar não reajusta controladores. A execução continua de checkpoints transacionais.
2. Bahia e M5 permanecem em diretórios e filas diferentes. O launcher executa sequencialmente, com oito processos CPU, uma thread por processo e prioridade reduzida. Uma trava global impede duas filas canônicas concorrentes. Por padrão, o launcher apenas planeja; `-Execute` inicia.
3. Auditoria de reaproveitamento verifica integridade de snapshots, semântica das dependências científicas, configuração por ajuste, identidade da tarefa, painéis congelados e dados do piloto. Checkpoints de calibração antiga de CappedBaseStock são rejeitados isoladamente; as outras políticas compatíveis são preservadas.
4. A calibração atual da CappedBaseStock respeita pesos zero para ruptura/amplificação, como no piloto M5. A Bahia terá somente essa candidata reajustada. Não se alteraram custos, prazo, dados, orçamento de GA/RL, sementes nem a dinâmica do ambiente.
5. O mesmo avaliador passa a atender as duas bases, incluindo PSE preditivo na Bahia: global, por perfil, XGBoost por classe com/sem perfil, moda, floresta de classe de controle e previsão de consequências com/sem perfil e com recuo. São nove estratégias, não nove políticas de estoque.
6. O pré-processamento de características foi centralizado em uma função compartilhada por classificação e regressão. O replay reproduziu exatamente 450 recomendações do PSE anterior e 360 da comparação por consequências, sem novos ajustes de controladores.
7. A validação interna pode usar cinco grupos de lojas. A loja externa continua excluída, em cada origem. Isso evita a explosão do leave-two-stores-out na Bahia; muda o ajuste da margem e exige nova avaliação do seletor. O modo anterior continua disponível para reproduzir o piloto congelado.
8. A margem é escolhida em validação interna, incluindo a opção de manter sempre a referência. Serviço inteiramente indefinido não justifica intervenção. Suporte real dos treinos internos é registrado mesmo com tamanhos diferentes de lojas.
9. Matriz incompleta, duplicação de casos/sementes, custo inválido e serviço fora de [0,1] são recusados. As recomendações são gravadas antes da pontuação do teste.
10. Saídas incluem custo, serviço, casos viáveis, comparação pareada com a global, intervenções, novas falhas, métricas por loja e intervalos condicionais por bootstrap. As cinco visualizações anteriores continuam integradas.
11. Oráculo do portfólio, planejamento com informação perfeita, eficiência e decomposição de falhas integram a avaliação. Planos perfeitos possuem cache SQLite por dados/configuração, para retomar sem resolver novamente. Comparações teóricas são feitas por realização viável; sementes e origens não se tornam séries independentes.
12. No M5, o resumo separa as séries do piloto de desenvolvimento das demais. Essa separação de relatório não é, por si, garantia de confirmação independente; o protocolo e seu desenvolvimento continuam declarados.
13. Status distingue planejamento, controladores concluídos, avaliação concluída, interrupção e falha. O comando de status funciona antes de existir SQLite de execução. Corrigiu-se também o import da entrada canônica ao chamá-la fora do diretório do projeto.

## Limpeza e preservação

Foram limpas 17 declarações de importação em 13 arquivos. Imports futuros e símbolos exportados por módulos não foram removidos por uma regra cega. O piloto inicial `run_research_v2.py`, com 132 linhas não vazias e orçamento curto, saiu da árvore ativa e foi preservado em `docs/archive/pipeline_cleanup_2026-10-15/`. Seu comando deixou de ser oferecido como entrada corrente. O launcher antigo e o README extenso foram arquivados e substituídos por versões focadas no fluxo atual.

Não foram encontrados corpos de função exatamente duplicados com pelo menos 12 linhas na busca estrutural; isso não exclui duplicação conceitual. Os runners anteriores ainda usados como biblioteca de fila, por testes ou por reprodução histórica foram preservados. A nova orquestração e a avaliação acrescentam código necessário, portanto não se declara redução líquida do total de linhas. Nenhum PDF, dado bruto ou resultado concluído foi apagado.

## Pipeline atual

Coortes e painéis congelados → auditoria de checkpoints → ajustes faltantes → características do treino → seleção externa por loja com ajuste interno por grupos → recomendações fixadas → teste → comparação com referências → cinco visualizações → oráculos e causas das falhas → resultados para artigo e dissertação.

A metodologia é avaliada por diferentes instanciações de políticas e seletores, com referências competitivas. O ambiente e a representação principal permanecem fixos; não se alega substituição validada de qualquer módulo por qualquer método. Os seletores usam a mesma regra nominal de custo/serviço, mas associação agregada e restrição individual prevista conservam escopos diferentes. O relatório não afirma garantia individual de atendimento nem harmonização desse aspecto ainda não implementada.

## Decisão sobre Bahia

- 145 séries, 115 lojas e duas origens: 290 casos e 4.350 ajustes no total.
- 4.060 ajustes compatíveis preservados.
- Repetir somente 290 ajustes CappedBaseStock; não repetir GA, PPO, DQN ou GA–DQN.
- Reavaliar todos os seletores, incluindo os preditivos, e recalcular relatórios. Os números publicados permanecem identificados pela rodada anterior até essa avaliação concluir.
- Os ajustes restantes são baratos; preparação dos processos, PSE, figuras e oráculos dominam o tempo. Não foi medido ainda o tempo total desta nova avaliação Bahia.

## Decisão sobre M5

Coorte histórica de 2.408 identificadores, mantendo o recorte já adotado; não significa todas as 14.370 FOODS. Elegíveis por origem: 885 em 50, 2.033 em 64 e 2.408 em 78. São 5.326 casos e 79.890 ajustes.

| Política | Preservados | A executar |
|---|---:|---:|
| EOQCycle | 702 | 4.624 |
| sSCycle | 702 | 4.624 |
| CappedBaseStock | 90 | 5.236 |
| GA | 2.106 | 13.872 |
| PPO | 2.105 | 13.873 |
| GA–DQN | 2.101 | 13.877 |
| DQN | 2.100 | 13.878 |
| Total | 9.906 | 69.984 |

A contagem rejeita 612 checkpoints antigos de CappedBaseStock ainda não cobertos pelo piloto compatível e resolve sobreposições entre fontes. Não zera o M5 nem sobrescreve a fila anterior: a rodada unificada importa os checkpoints auditados e continua os faltantes.

A projeção linear pelas medianas históricas é aproximadamente 46,8 horas apenas para os ajustes restantes com oito processos, sem incluir análise e considerando divisão ideal da carga. O tempo real depende do uso do PC, das origens e da concorrência. Essa projeção não é uma garantia de término nem um novo benchmark.

## Verificação

67 testes passaram. Foram verificados ausência de sobreposição entre lojas de treino e alvos internos, recuo diante de serviço indefinido, rejeição de matrizes parciais, funcionamento da entrada única fora do diretório e preservação das recomendações anteriores no replay. A execução real ainda deve conferir a integração e o tempo total dos novos relatórios. Inventário, limpeza, duplicações e replay têm registros JSON neste diretório.
