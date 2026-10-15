# Revisão e protocolo para refazer os experimentos

14/10/2026. **Recomendo uma nova rodada antes de sustentar ganhos do AIPE.** Há problemas de preparação e desenho que podem mudar os resultados mais do que trocar de algoritmo. A rodada anterior e os textos continuam preservados; o piloto novo não foi incorporado como evidência final ao artigo.

## Prioridades verificadas

| Prioridade | Problema e evidência | Consequência e ação |
|---|---|---|
| 1 | Limpeza da base real:442.330 linhas filtradas somam 748.555 unidades;272.779 linhas limpas somam 446.156. Existem 169.551 linhas adicionais na mesma chave de ciclo, mas nenhuma duplicata de linha completa. Em 26.449 grupos repetidos as datas diferem; em 8.751 as quantidades diferem. A coluna `id` tem a mesma cardinalidade das lojas. | O corte remove 40,4% da quantidade nessa etapa. Não usar loja-produto-ciclo nem `id` de loja como identificador de transação. Somar vendas complementares por ciclo; reconciliar identificadores transacionais antes de deduplicar. Recriar os cenários a partir do snapshot bruto, sem reutilizar o Parquet já reduzido. |
| 2 | Filtro CV, atividade, perfis, IQR M5, estoque inicial e normalizadores consultam o histórico completo. | Separar primeiro os períodos. Estimar seleção de coorte, transformação, características e escalas apenas no prefixo conhecido. Não suavizar demanda de teste como se os picos fossem ruído; se houver tratamento de qualidade, registrar uma regra treinada no passado e reportar também quantidades originais. |
| 3 | 2.213/2.408 séries M5 não têm demanda nos 19 ciclos usados no ajuste anterior. | Separar avaliação com histórico e partida sem histórico. Usar origens posteriores determinadas por calendário e elegibilidade pelo passado. A partida sem histórico precisa de estratégias de inicialização/pooling próprias, sem utilizar média futura. Não eliminar séries por desempenho de teste. |
| 4 | Cinco avaliações repetem o controlador já ajustado e a demanda fixa. | Ajustar GA/RL de novo para cada semente; registrar cada ajuste e avaliar nas mesmas demandas. Políticas determinísticas precisam de uma avaliação, não cinco pseudorreplicações. Sementes representam incerteza de treinamento; não multiplicam o número de lojas independentes. |
| 5 | EOQ mistura demanda acumulada com manutenção por ciclo; pipeline L+1 entrega em t+3 para L=2. | Comparadores com demanda e custo na mesma unidade. Definir a convenção de chegada e testá-la; a V2 usa entrega em t+L. Preservar uma opção legada para reproduzir os resultados existentes. |
| 6 | Seleção por perfil é majoritariamente retrospectiva; PSE usa validação por série, pode excluir classes raras e reporta ajuste na própria amostra. | Escolher políticas com validação e avaliar no teste futuro. Excluir todas as séries da loja-alvo na avaliação de transferência entre lojas. Comparar global, POD, características contínuas sem POD e modelo com POD; reportar custo e atendimento, não apenas acurácia. |
| 7 | Orçamentos GA, SA e híbrido diferem entre BA/M5; objetivos denominados CTI ajustado também diferem. | Fixar orçamento por método e objetivo antes da execução. Medir transições/avaliações, tempo e GPU; episódios iguais com horizontes diferentes não equivalem a computação igual. Tratar termos de estabilidade como análise secundária separada do custo bruto. |

Os números da primeira linha demonstram a diferença entre os snapshots, não certificam que toda linha removida seja uma transação válida. A reconciliação empresarial continua necessária; a evidência já invalida deduplicação automática pela chave do agregado.

## Correções implementadas e verificadas

O núcleo agora admite, de forma explícita:

- `causal_state=true` com `reference_mean/reference_std` estimados no treino. Falta desses valores gera erro; histórico recente e primeira previsão usam apenas observações passadas.
- `lead_time_convention=exact`, incluindo prazo zero, nas versões escalar e PyTorch. O padrão legado foi preservado.
- `initial_pipeline`, para comparar controladores com o mesmo estado conhecido de estoque e pedidos em trânsito.
- `eoq_time_basis=per_cycle` em EOQ e `(s,S)`. A variante antiga continua disponível.
- Estratégias de limpeza `none`, `exact_row` e `transaction_id`, além de `legacy_key_first`. Identificador transacional precisa ser fornecido; quantidades conflitantes não são descartadas silenciosamente.

Os parâmetros novos não são ativados automaticamente nos pipelines históricos. O piloto isolado `run_research_v2.py` os usa; uma reexecução com os comandos antigos continua dependendo das configurações legadas e deve ser evitada para a nova evidência.

Passaram **22 testes**, incluindo equivalência escalar/CPU/CUDA, perturbação do futuro sem mudança do estado presente, chegada no ciclo exato, preservação de vendas no mesmo ciclo e invariância do lote EOQ quando apenas se repete a mesma história.

## Piloto executado nas duas bases

Saídas locais: `experiment/data/08_reporting/research_v2/pilot_v2_verified_2026-10-14/`.

- BA parte de `sales_raw.parquet`, soma quantidades sem descartar linhas por chave de ciclo e não aplica o antigo filtro global CV/atividade. O snapshot ainda precisa ser reconciliado com a fonte transacional empresarial.
- M5 parte do CSV diário FOODS, sem IQR nem filtro CV futuro. Usa 92 blocos completos de 21 dias e exclui apenas o último bloco parcial de 9 dias.
- Duas origens BA: 17 e 23; três origens M5: 50, 64 e 78. Validação e teste têm sete ciclos cada.
- Elegibilidade pelo prefixo: ao menos três ciclos positivos. Casos com prefixo nulo ou insuficiente são contados separadamente, não avaliados como históricos maduros.
- Oito séries por origem, escolhidas por hash do identificador entre as elegíveis, sem consultar resultados futuros.
- Cinco heurísticas e GA. GA é reajustado para sementes 11, 29 e 47, com população 32 e 8 gerações no piloto.
- Todas as candidatas recebem o mesmo estado inicial de teste, obtido do histórico conhecido de uma referência fixa EOQ, incluindo pedidos em trânsito. Não se usa o estoque contrafactual produzido por cada candidata na validação para favorecê-la no teste.
- Seleção global e por perfil usa as validações das outras lojas. Serviço sem demanda é indefinido, com denominador e contagem reportados; não vira falsa violação ou sucesso de atendimento.
- O GA executou em CUDA. Foram 120 ajustes independentes em 40 casos série-origem; todos os 40 apresentaram alguma variação de custo entre sementes. Total 640 linhas de validação/teste; elas não são 640 observações varejistas independentes.
- Tempo total aproximado:73 segundos nesta máquina, incluindo carga e preparação. Não é estimativa do tempo da rodada completa.

Este piloto comprova execução e controles do protocolo. A amostra é pequena, os orçamentos são baixos e as coortes diferem das antigas; seus valores não sustentam alegações de ganho sobre os resultados da qualificação. DQN/PPO são opcionais nesse runner com apenas 10 episódios, suficientes para diagnóstico de execução, não para comparar desempenho publicado.

## Rodada científica proposta

1. **Reprodução controlada:** manter um recorte das séries originais para quantificar separadamente efeitos da agregação, escala EOQ, prazo e ausência de informação futura. Uma ablação por mudança evita atribuir todo o efeito ao AIPE.
2. **Avaliação com histórico:** amostragem por loja/produto definida no treino, múltiplas origens, pelo menos três sementes de treinamento e os mesmos protocolos de cada método nas duas bases. A amostra BA deve preservar o caso da empresa; M5 é a avaliação pública complementar. Amostra calculada antes do teste, com precisão desejada e custo de execução registrados.
3. **Partida sem histórico:** experimento separado com política adaptativa simples e pooling entre séries elegíveis apenas no passado. Não oferecer parâmetros derivados da demanda posterior. Comparar com global e com inicialização uniforme.
4. **Portfólio competitivo:** EOQ e `(s,S)` consistentes, revisão periódica, heurística responsiva, uma base-stock calibrada, GA/SA e DQN/PPO/híbrido após conferir suas hipóteses. Não aumentar portfólio para esconder comparadores fracos. GA e SA devem procurar a mesma regra; RL/híbrido precisam de ablação sob orçamento comparável.
5. **PSE:** treinamento com rótulos formados em validação interna, seleção de hiperparâmetros dentro do treino e avaliação externa por loja/período. Não usar o oráculo do teste para treinar. Comparar com/sem perfil e calcular arrependimento, custo, serviço ponderado e violações locais; documentar classes raras e cobertura avaliada.
6. **Robustez:** estoque inicial zero versus estado conhecido, crédito ou ausência de valor terminal, filas pendentes, custos de falta/manutenção/pedido e limiares de serviço. Os valores empresariais devem orientar o cenário principal; 0,70 e 0,90 permanecem sensibilidades, não metas comerciais confirmadas.
7. **Inferência:** unidades são lojas/produtos e origens, não sementes. M5 tem só dez lojas; reportar dependência por produto e limites dos intervalos. Ajustar seleção dentro da reamostragem quando o objetivo for inferir todo o procedimento, não apenas o contraste condicional. Não selecionar resultados por p-valor ou sinal favorável.

## Antes de incluir previsão ou ampliar a rodada

A previsão ainda não é entrada dos workers de inventário. Se for parte da contribuição, conectar previsões causais e comparar com sinal ingênuo sob o mesmo controlador; caso contrário, tratá-la como análise auxiliar. Corrigir MASE/RMSSE para escala do treino, preservar denominadores indefinidos e revisar o tratamento de pares zero no sMAPE antes de reportar novas métricas.

Conferir também taxonomia CV² sobre tamanhos positivos versus POD legado que usa todos os ciclos; a V2 ainda reutiliza as regras POD existentes como comparador. Avaliar uma versão correta da taxonomia e características contínuas em ablação, sem alterar regras depois de observar teste. Revisar limites fixos de busca, penalidades aproximadas de serviço, relatórios que omitem políticas e caminhos hardcoded antes de expandir para os doze controladores.

## Executar o piloto

```powershell
& experiment/.venv/Scripts/python.exe experiment/run.py research-v2 --max-series 8 --run nome_novo
```

Cada nome cria uma pasta nova e falha se ela já existir. Não usar `--max-series 0` para começar: a seleção pelo passado produz dezenas de milhares de séries BA e mais de 14 mil M5, com custo computacional muito diferente das 145/2.408 séries anteriores. A rodada completa deve ter amostragem e orçamento previamente definidos.
