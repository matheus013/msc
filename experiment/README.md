# AIPE — projeto Python integrado

O pipeline de pesquisa atual usa uma entrada única, com fases separadas de planejamento, ajustes dos controladores e avaliação dos seletores. Bahia e M5 são execuções independentes. Brasil permanece cancelado. Os documentos estão em `../docs/` e os dados brutos em `../data/`.

## Pipeline atual

1. Conferir coorte, painéis congelados e configuração científica.
2. Auditar checkpoints por código, configuração, identidade e dados.
3. Reutilizar ajustes compatíveis e executar somente os faltantes.
4. Extrair características do prefixo de treino.
5. Comparar global, perfil, classe vencedora e consequências com/sem recuo.
6. Escolher margens em cinco grupos internos por loja; reservar a loja externa.
7. Fixar recomendações antes de pontuar o teste.
8. Gerar indicadores pareados, cinco análises visuais, oráculos e causas das falhas.

CPU, oito processos, uma thread por processo e prioridade reduzida. Parâmetros dos controladores permanecem os da produção: sete candidatas incluindo DQN como controle, três sementes nos estocásticos e 20 mil transições nos agentes. Os resultados são separados por base e origem; o piloto já observado é desenvolvimento.

## Planejamento e execução

Na raiz do repositório, com o ambiente existente:

    experiment/.venv/Scripts/python.exe experiment/run.py research-aipe plan --dataset BA --run aipe_ba145_unified_2026-10-15
    experiment/.venv/Scripts/python.exe experiment/run.py research-aipe plan --dataset M5 --run aipe_m5_2408_unified_2026-10-15

Planejar não treina nem inicia processos de simulação. O plano informa reaproveitamento e tarefas restantes. Depois da apresentação do plano:

    experiment/.venv/Scripts/python.exe experiment/run.py research-aipe execute --dataset BA --run aipe_ba145_unified_2026-10-15
    experiment/.venv/Scripts/python.exe experiment/run.py research-aipe execute --dataset M5 --run aipe_m5_2408_unified_2026-10-15

O launcher `scripts/operations/start_scoped_experiments.ps1` apenas planeja por padrão; `-Execute` executa Bahia e depois M5, em janelas ocultas, evitando duas filas concorrentes de oito processos. A fase `execute` retoma SQLite e chama a avaliação ao completar os controladores. A fase `evaluate` refaz apenas seleção e análise, sem novos ajustes de estoque. Não modificar o código de uma rodada iniciada; hashes são conferidos.

Para status:

    experiment/.venv/Scripts/python.exe experiment/run.py research-status --run aipe_m5_2408_unified_2026-10-15

## Módulos e contratos

- `simulation/research/protocol.py`: cenário de estoque, simulação e estado comum.
- `production.py`, `hybrid.py`, `parallel.py`: ajuste das políticas e fila CPU.
- `pse.py`: pré-processamento compartilhado, seleção e pontuação.
- `pse_consequences.py`: classe de controle, previsão de consequências e recuo.
- `evaluation.py`: matriz completa, seleção, comparações pareadas e oráculos.
- `visualizations.py`: fronteira, escolhas por perfil, distribuição do serviço, heatmap e demanda.
- `scripts/analysis/run_aipe_pipeline.py`: planejamento, reaproveitamento e execução por base.

Os runners anteriores de produção continuam como dependências de fila ou para reproduzir resultados congelados. Não são a entrada recomendada para novas execuções. O piloto inicial `run_research_v2.py` foi arquivado em `../docs/archive/pipeline_cleanup_2026-10-15/`; seu orçamento curto e sua mistura de escopos não definem o protocolo atual.

Os pipelines Kedro e as configurações `base`, `m5` e `bot` são interfaces históricas e auxiliares. Previsão e mapas continuam disponíveis, mas não integram automaticamente o experimento de seleção atual. A modularidade não implica que todo método existente tenha sido validado como opção intercambiável.

## Dados e ambiente

As fontes ficam em `../data/source/` e `../data/m5_walmart/`. Artefatos intermediários, bancos e resultados ficam em `experiment/data/`, fora do Git. PDFs de referência e versões documentais são preservados. As coortes atuais são 145 séries Bahia e a coorte histórica M5 de 2.408 IDs, com elegibilidade por origem; não se confunde essa coorte com todas as FOODS.

Dependências estão em `pyproject.toml`; `requirements.txt` referencia o projeto. Para um ambiente novo, usar `python -m venv .venv` e `pip install -e ".[test,documents]"`. O ambiente local já está disponível.

## Verificação

    experiment/.venv/Scripts/python.exe -m pytest experiment/tests -q

O inventário de código, a limpeza, os testes e a decisão de reaproveitamento estão em `../docs/operations/aipe_pipeline_review_2026-10-15/`. O README anterior, com os comandos e detalhes históricos, foi preservado no arquivo da revisão.

## Contexto adicional opcional

O pacote `src/aipe_context` acrescenta atributos opcionais ao PSE preditivo, com escolhas fixas/candidatas/excluídas, junção temporal e seleção por consequências operacionais. Está desligado por padrão e não é importado pela fila corrente. Sua avaliação reutiliza matrizes concluídas, sem ajustar controladores, e é bloqueada enquanto BA/M5 estiverem em execução. Configuração: `conf/aipe_context_extension_v1.json`; comandos: `scripts/operations/context_extension.py`. Contratos, limites e exemplos estão em [Módulo de contexto opcional](../docs/operations/MODULO_CONTEXTO_OPCIONAL_AIPE.md).
