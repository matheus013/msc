# Contrato do relatório de recomendação AIPE

16/10/2026. Primeira materialização do contrato de apoio ao gerente, após auditoria das entradas. Não seleciona uma instanciação primária com base no teste nem altera os experimentos em execução.

## Estrutura por decisão

1. **Contexto:** produto, local de demanda, instante de decisão e horizonte; histórico efetivamente admissível, perfil e origem dos dados.
2. **Recomendação:** família selecionada e referência metodológica, candidata examinada e motivo registrado para mudança/manutenção. Parâmetros finais devem constar quando disponíveis; sua ausência precisa impedir a apresentação como ordem executável.
3. **Consequências esperadas:** custo e fill rate da família efetivamente recomendada, origem da estimativa e limitações. Se houver recuo, apresentar estimativas da referência. Não substituir uma previsão ausente pelo resultado de teste.
4. **Alternativas:** resultados de validação por candidata, com sementes e denominadores declarados, identificados como históricos. Previsões de todas as alternativas para a série-alvo são outra saída, ainda não preservada integralmente.
5. **Hipóteses:** custo em unidades de cenário, prazo, meta, estado inicial sintético e restrições não representadas; distinguir observação, estimativa e hipótese.
6. **Avaliação posterior:** resultados simulados após o horizonte, diferenças pareadas e ocorrências de deterioração. Bloco separado da fundamentação disponível na decisão.
7. **Auditoria e decisão humana:** IDs, fontes e versões, arquivos abertos e status da avaliação do gerente. Ausência de avaliação humana permanece `not_collected`; nunca preencher aceitação automaticamente.

## Primeira exportação realizada

Fonte: execução Bahia `aipe_ba145_unified_2026-10-15`, concluída. Saída local: `experiment/data/decision_reports/aipe_ba145_unified_2026-10-15/v1/`.

São 290 casos série–origem e 2.610 registros de decisão: nove estratégias alternativas sobre os mesmos casos. Não são 2.610 séries nem 2.610 recomendações simultâneas de compra. A interface permite escolher estratégia e contexto, sem ranking de lojas e sem eleger uma estratégia vencedora por teste.

| Arquivo | Conteúdo e papel |
|---|---|
| `index.html` | Consulta local, filtros por estratégia e identificadores, recomendação e avaliação posterior separadas. Não exige internet e não grava parecer humano. |
| `decisions.json` | Registros estruturados de decisão, histórico de treino, hipóteses e campos ausentes explícitos. Sem indicadores de teste. |
| `recommendations.csv` | Visão tabular dos campos simples, com previsões somente quando persistidas. |
| `alternatives_validation.csv` | Médias de custo e fill rate na validação por caso/família, número de sementes e número com atendimento definido. Não são previsões do alvo. |
| `outcomes_retrospective.csv` | Resultados de teste, referência e diferenças pareadas. Custo negativo na diferença representa economia; diferença de atendimento está em fração. |
| `candidate_metrics_by_seed.csv` | Indicadores existentes por candidata/semente/bloco, sem agregação silenciosa. |
| `manifest.json` | Hashes dos arquivos de entrada, painéis e exportador, além dos arquivos de saída e limitações. |
| `README.md` | Dicionário mínimo, unidades, convenções e limitações. |

Scripts e dados permanecem separados: o exportador está em `experiment/scripts/operations/export_decision_report.py`; os registros com identificadores operacionais ficam na pasta local de dados. A exportação não publica arquivos nem envia dados ao GitHub.

Comando utilizado na raiz:

```powershell
experiment/.venv/Scripts/python.exe experiment/scripts/operations/export_decision_report.py --run experiment/data/08_reporting/research_v2/aipe_ba145_unified_2026-10-15 --output experiment/data/decision_reports/aipe_ba145_unified_2026-10-15/v1
```

O exportador exige execução concluída e pasta de saída nova dentro de `experiment/data/decision_reports`. Não grava no experimento de origem, não importa código de simulação e não treina. M5 em andamento é recusado por esse fluxo: terá sua própria exportação após concluir e validar sua avaliação. Cada nova versão de exportação usa outra pasta, preservando as anteriores.

## O que a prévia entrega e o que continua pendente

**Já materializado:** recomendação congelada, referência, perfil derivado, prefixo de treino, previsões registradas nas estratégias por consequências, margem/recuo, alternativas de validação, indicadores individuais e ligação por ID com a avaliação posterior.

**Justificativa parcial:** os motivos são descritivos da regra aplicada e das famílias escolhidas; não constituem explicação causal ou justificativa completa das características que levaram cada modelo à escolha. Estratégias de classe/perfil não têm uma previsão quantitativa de custo/atendimento preservada no mesmo formato; esses campos ficam ausentes. Não interpretar contagem de escolhas como confiança.

**Ausências preservadas:** parâmetros finais de todas as políticas, redes/seletores reutilizáveis, trajetórias por ciclo, custos desagregados por caso e intervalos de previsão por decisão. A política GA usada para inicializar GA–DQN não equivale ao agente final. A prévia não simula a completude desses campos.

**Limites de aplicação:** não há custos calibrados em reais, estado operacional atual validado, restrições reais de fornecimento ou comprovação de uso por gerentes. Não emitir compra a partir desta prévia. O status do M5 e seu número de processos não são alterados.

## Recuperação e próximos incrementos

| Incremento | Reutiliza o que existe? | Exige execução adicional? |
|---|---|---|
| Exportações e apresentação dos indicadores existentes | Sim | Apenas processamento de arquivos, já realizado na Bahia. |
| Campos de origem/qualidade e dicionário de dados | Em parte: auditoria de código e fontes | Documentação e validação sem treinamento; coleta humana quando a fonte não resolve a semântica. |
| Explicação por contexto e previsões completas das alternativas | Há dados de validação e código dos seletores | Pode exigir reexecutar a seleção com persistência mais rica; sem reajustar controladores se a matriz permanecer compatível. |
| Parâmetros/artefatos de políticas não preservados | Configuração, sementes e demanda disponíveis | Recuperação pode exigir novo ajuste/treinamento com equivalência verificada; nunca inventar parâmetros a partir dos KPIs. |
| Trajetórias e custos desagregados | Estado reconstruível e código disponíveis | Replay se a política exata for recuperada; caso contrário, ajuste adicional. Registrar diferenças, sem sobrescrever a matriz histórica. |
| Recomendação imediata de compra e julgamento humano | Parcial | Novos dados operacionais e avaliação com gerente; não decorre de um replay do simulador. |

Recomendação de sequência: revisar esta prévia como formato de decisão; completar origem/qualidade por entrada; definir o artefato mínimo de política e os detalhes de auditoria a persistir; planejar apenas as recuperações necessárias fora da fila científica congelada. A exportação não altera o objetivo da pesquisa, a definição das hipóteses ou a necessidade de confirmação independente.
