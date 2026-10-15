# Validação de CPU, GPU e paralelismo — 14/10/2026

Perfil solicitado: rodar junto com os aplicativos do usuário. A fila censitária permanece parada; nenhum resultado científico do artigo foi substituído por estes testes.

## Escopo e controle

Uma série real da Bahia (origem 17) e uma série M5 (origem 50), DQN, PPO e GA-DQN. Triagem: quatro sementes e aproximadamente 2.048 transições por treino, 24 treinos por configuração. Confirmação: duas sementes e aproximadamente 20.000 transições por treino, 12 treinos por configuração. Episódios completos arredondam o orçamento para cima. O teste curto do perfil de uso cotidiano usa uma semente, seis treinos idênticos em cada configuração CPU/CUDA.

Cada processo limita Torch/BLAS a uma thread. O cronômetro do lote inclui treinamento, avaliação e comunicação com o processo principal; inicialização e aquecimento são mostrados separadamente. A RAM livre se refere à máquina inteira, incluindo outros aplicativos. Há uma repetição por configuração; a carga externa variou entre elas, portanto diferenças pequenas não definem um ótimo universal.

## Escala da CPU, prioridade normal

| Dispositivo | Processos | Treinos | Inicialização (s) | Lote após inicialização (s) | Treinos/min | RAM livre mínima (GB) |
|---|---:|---:|---:|---:|---:|---:|
| cpu | 1 | 24 | 3.74 | 98.20 | 14.66 | 7.63 |
| cpu | 2 | 24 | 4.79 | 47.99 | 30.01 | 7.30 |
| cpu | 4 | 24 | 5.59 | 27.85 | 51.71 | 6.43 |
| cpu | 8 | 24 | 6.53 | 17.80 | 80.90 | 5.15 |
| cpu | 12 | 24 | 7.75 | 15.02 | 95.86 | 3.71 |
| cpu | 16 | 24 | 12.43 | 13.52 | 106.51 | 2.66 |

## Orçamento completo de treinamento

As linhas de 1 e 12 processos usam prioridade normal. A linha de 8 processos usa prioridade reduzida, conforme o perfil solicitado; a diferença entre elas também inclui a variação da carga externa.

| Dispositivo | Processos | Treinos | Inicialização (s) | Lote após inicialização (s) | Treinos/min | RAM livre mínima (GB) |
|---|---:|---:|---:|---:|---:|---:|
| cpu | 1 | 12 | 4.51 | 319.49 | 2.25 | 9.19 |
| cpu | 8 | 12 | 4.32 | 79.97 | 9.00 | 13.88 |
| cpu | 12 | 12 | 5.49 | 47.36 | 15.20 | 12.60 |

As métricas de validação e teste permaneceram idênticas entre as execuções CPU, incluindo a confirmação com prioridade reduzida. Verificações e tempos individuais estão nos manifestos e JSONs das duas pastas de medição.

## Uso cotidiano: CPU e GPU com prioridade reduzida

A prioridade Windows foi conferida em todos os processos executores (BELOW_NORMAL_PRIORITY_CLASS). Isso favorece os aplicativos ativos na disputa por CPU, sem garantir limite de utilização ou de latência da interface. Não foram fechados aplicativos do usuário.

| Dispositivo | Processos | Treinos | Inicialização (s) | Lote após inicialização (s) | Treinos/min | RAM livre mínima (GB) |
|---|---:|---:|---:|---:|---:|---:|
| cpu | 1 | 6 | 3.18 | 16.54 | 21.76 | 16.59 |
| cpu | 8 | 6 | 4.54 | 5.34 | 67.42 | 14.09 |
| cuda | 1 | 6 | 3.76 | 49.45 | 7.28 | 15.36 |
| cuda | 2 | 6 | 3.99 | 33.10 | 10.88 | 14.40 |
| cuda | 4 | 6 | 4.61 | 24.85 | 14.49 | 11.79 |

Tempos médios apenas de treinamento, um processo em cada dispositivo, nos mesmos seis casos:

| Política | CPU (s) | GPU (s) | GPU/CPU |
|---|---:|---:|---:|
| DQN | 3.105 | 8.482 | 2.73 |
| PPO | 1.458 | 5.240 | 3.59 |
| GA-DQN | 3.657 | 10.906 | 2.98 |

O código faz inferências e atualizações pequenas, com conversões frequentes de arrays para tensores e extrações de escalares (`.item()`). Na GPU essas operações exigem transferências e sincronizações; o diagnóstico anterior de DQN já apontava o replay como gargalo. Os tempos medidos são compatíveis com esse custo predominando sobre o cálculo das redes pequenas. Processos CUDA adicionais também duplicam contextos e aumentam o consumo de memória; isso não equivale a treinar várias redes em um único lote vetorizado na GPU.

CPU e CUDA podem aprender políticas diferentes por usarem fluxos aleatórios e operações numéricas diferentes; igualdade de eficácia não é alegada. Os números curtos não são uma previsão de duração do censo.

## Configuração implementada

O executor de produção usa CPU, 8 processos e prioridade reduzida por padrão, com uma thread por processo e no máximo 16 tarefas pendentes. Há uma única conexão escritora SQLite, com transação por treino, sementes independentes, retomada apenas dos treinos não confirmados e registro do dispositivo/configuração no manifesto. Alterar código ou perfil exige outra identificação de rodada para não misturar resultados. A ablação DQN continua separada do portfólio principal.

O launcher aceita `-Workers`, `-Device` e `-Priority`; o script Python aceita `--workers`, `--device` e `--priority`. Preparar esses padrões não reiniciou a execução nacional. O perfil de 16 processos reduziu a folga de RAM na triagem e não foi adotado como padrão para uso simultâneo.

Trinta testes passaram, incluindo transações, retomada, correspondência dos resultados à tarefa e consumo limitado da fila. Os testes reais entre processos também verificaram a invariância das métricas com as mesmas sementes.

Fontes locais: `experiment/data/08_reporting/performance/parallel_validation_2026-10-14/` e `experiment/data/08_reporting/performance/background_validation_2026-10-14/`. Esta é uma medição de execução, não uma comparação final da qualidade das políticas.
