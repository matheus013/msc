# Diagnóstico das 19 novas violações de atendimento no M5

## Escopo e integridade

Análise descritiva posterior ao teste da implementação `Consequencia_recuo`, comparada à referência `Global`, nas mesmas 5.326 instâncias série–origem da avaliação consolidada. Não foram treinados modelos, executadas simulações, escolhidos novos limiares ou modificados arquivos canônicos. As recomendações foram conferidas contra o arquivo de previsões congelado; custos e atendimento foram reconstruídos a partir dos indicadores por política e semente. Os hashes das entradas permaneceram iguais ao término.

Os 19 casos foram selecionados porque a referência alcançou atendimento de pelo menos 70% e a recomendação ficou abaixo desse alvo. Esse recorte não representa todas as deteriorações: a avaliação completa tem 149 casos com atendimento menor. Das 679 intervenções, 422 partiram de uma referência que alcançava a meta; 19/422, ou 4,50%, produziram nova violação. A frequência observada não é uma estimativa calibrada de risco futuro. Houve também 43 recuperações, reportadas na avaliação completa.

## O que aconteceu

- Todos os 19 casos trocaram a referência por **sSCycle**, uma política determinística. Onze partiram de CappedBaseStock e oito de GA.
- São 19 séries distintas, distribuídas por oito lojas. Dezoito casos ocorreram na origem 78 e um na origem 64; nenhum na origem 50.
- A queda média de atendimento foi **6,51 pontos percentuais**, com mediana de 6,16 e intervalo observado de 2,62 a 13,40 pontos.
- Treze casos economizaram e seis aumentaram o custo. Nestes seis, a troca deteriorou simultaneamente custo e atendimento.
- A economia líquida dos 19 casos soma **1.272,43 unidades de cenário**, ou 1,57% da economia líquida agregada entre as estratégias. Essa soma sobre instâncias repetidas por origem não representa economia financeira realizada na rede.

As três sementes de GA são agregadas antes da classificação de viabilidade da referência. Dos 24 resultados individuais de GA nos oito casos correspondentes, vinte satisfazem o alvo e quatro não. A afirmação de que a referência era viável refere-se à média por caso, não a todas as realizações. A candidata sSCycle, determinística, fica abaixo do alvo nos 19 casos.

### Concentração exige denominadores

| Origem | Intervenções | Intervenções com referência viável | Novas violações |
|---|---:|---:|---:|
| 50 | 35 | 22 | 0 |
| 64 | 152 | 97 | 1 |
| 78 | 492 | 303 | 18 |

Na origem 78, a transição GA → sSCycle gerou oito novas violações entre 93 intervenções com referência viável (8,60%); CappedBaseStock → sSCycle gerou dez entre 205 (4,88%). São contrastes descritivos de grupos diferentes, sem identificação causal do efeito da família de referência.

Todos os 19 casos pertencem ao perfil `Sparse_High_Impact`, mas esse perfil também reúne 5.163 dos 5.326 casos da coorte. Portanto, sua presença não é evidência suficiente para criar uma restrição por perfil nem para eliminar sSCycle, que contribui para ganhos e recuperações em outros casos.

## O que era conhecido antes da decisão

A regra original admite uma troca quando prevê custo menor, atendimento da candidata pelo menos igual ao da referência e atendimento acima de 70%. Nos 19 casos, a vantagem prevista de atendimento foi positiva, porém pequena: média de **1,22 ponto**, mediana de 0,97 e intervalo de 0,27 a 2,50 pontos. O atendimento previsto da candidata excedeu o observado em **10,00 pontos**, em média; a superestimação da referência foi de 2,28 pontos. A previsão da diferença entre as políticas errou, em média, 7,73 pontos nesse recorte. Esse erro só é mensurável depois do teste.

Os casos com novas violações apresentaram menor folga prevista acima da meta: mediana de 6,32 pontos, contra 8,94 nas outras 403 intervenções cuja referência era viável. A vantagem prevista sobre a referência teve medianas de 0,97 e 1,59 ponto, respectivamente. Há ampla sobreposição entre os grupos; esses valores não definem um limiar validado.

No prefixo de demanda, a mediana da proporção de zeros foi 0,628 nos 19 casos e 0,560 nas outras intervenções comparáveis; o maior trecho sem demanda teve medianas de 45 e 34 ciclos. Tais características são observáveis antes da decisão. As diferenças são hipóteses para investigação, sem teste de significância ou seleção automática de regra neste relatório.

O desempenho da candidata na validação da própria série já ficava abaixo da meta em quatro casos e abaixo do atendimento da referência em sete. Porém, os resultados da loja externa foram deliberadamente excluídos do ajuste do seletor no protocolo leave-one-store-out. Usar esse histórico como veto pode ser operacionalmente plausível, mas muda o regime de informação e deve constituir uma variante explícita, comparada em condições iguais. Não seria uma simples correção do experimento atual.

## Próximo experimento recomendado

Manter as matrizes existentes e desenvolver uma regra que avalie a **incerteza da diferença de atendimento** entre candidata e referência, usando apenas resíduos de previsões fora da amostra nos grupos internos de treinamento. Examinar um limite conservador para essa diferença, em vez de tratar previsões pontuais como certezas. Poucos grupos, dependência e erros sistemáticos impedem prometer cobertura ou garantia individual sem validação específica.

Comparar a regra atual com essa variante e com um controle simples de folga de serviço. A margem de custo deve continuar separada: ela decide se a economia prevista justifica a troca, mas não mede confiança na preservação do atendimento. Limites e tolerâncias devem ser definidos no desenvolvimento; o teste de confirmação precisa usar períodos ou unidades ainda não empregados para escolher a regra.

Para a variante que utiliza histórico operacional da própria série, declarar a informação adicional, sua data de disponibilidade e o período efetivamente acessível a todas as estratégias. Separar esse teste do estudo de transferência para lojas excluídas.

A avaliação futura deve preservar os custos e o atendimento por caso, novas violações, recuperações, gravidade das perdas, frequência de intervenção e recuo. A tolerância à deterioração cabe ao contexto gerencial; o diagnóstico atual não determina um percentual universal aceitável.

Não se recomenda bloquear retrospectivamente estes 19 casos, escolher uma margem que os elimine nem apresentar o resultado de tal bloqueio como desempenho obtido pelo AIPE. Os resultados consolidados permanecem válidos no escopo declarado e inalterados.

## Artefatos e reprodução

- `nineteen_new_failures.csv`: todos os 19 casos, indicadores, previsões e características.
- `failure_policy_seed_metrics.csv`: resultados de cada política/semente nos casos afetados.
- `all_cases_diagnostic.csv`: 5.326 casos para rastrear os denominadores.
- `transition_denominators.csv`: contagens e resultados por transição e origem.
- `predecision_signal_descriptions.csv`: distribuições dos sinais disponíveis antes da decisão.
- `evidence.json`: resultados resumidos, hashes e limitações.
- `nineteen_failures.pdf/png`: visualização de atendimento e custo nos mesmos casos; verde indica economia e marrom, aumento de custo.

Reprodução, na raiz: `experiment/.venv/Scripts/python.exe experiment/scripts/analysis/diagnose_m5_service_deteriorations.py`.
