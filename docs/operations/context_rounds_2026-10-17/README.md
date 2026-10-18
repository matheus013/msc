# Rodadas adicionais de contexto — 17/10/2026

Avaliação de desenvolvimento do módulo opcional de contexto do AIPE. Reutiliza os controladores e as matrizes canônicas: nenhum controlador é treinado novamente e nenhuma simulação de estoque é reiniciada. Os resultados do teste canônico já haviam sido observados; esta extensão não constitui confirmação independente.

## Desenho

- Mesmos casos, políticas, custos e ambiente do experimento de origem.
- Duas instâncias preditivas: classe da política e consequências de custo/atendimento. Os contrastes de contexto são feitos **dentro de cada instância**, contra seu próprio PSE básico.
- ExtraTrees com 32 árvores, profundidade quatro, mínimo três observações na folha, semente 20261017 e uma thread; mesmos parâmetros entre as variantes de contexto. Esta instância por classe não é o classificador XGBoost canônico.
- A instância por consequências desta rodada usa a configuração de 32 árvores do módulo adicional; não se confunde com variantes canônicas de 64 árvores ou com margens de intervenção exploradas anteriormente. A identificação do incremento de contexto depende do contraste com a básica **desta mesma rodada**.
- Cada origem é ajustada separadamente. A dobra externa exclui uma loja/revendedor e todos os seus produtos. A seleção de atributos usa três dobras internas agrupadas somente no treino externo.
- Limite de três atributos adicionais selecionados e 16 avaliações internas de subconjuntos. Critério: redução de custo com tolerâncias internas zero para queda do atendimento médio e aumento da proporção abaixo do alvo de 70%. As tolerâncias são critérios de desenvolvimento, não garantias empresariais.
- `PSE_base`: descritores básicos de demanda e perfil operacional.
- `PSE_context_fixed`: somente atributos adicionais fixados pelo gestor. **Não há atributos fixos nesta rodada; portanto esta saída coincide com `PSE_base` e não representa uma abordagem adicional distinta.**
- `PSE_context_all`: todos os atributos admissíveis desta rodada, sem seleção. Eles são tratados como obrigatórios exclusivamente para construir esse comparador. O limite desta variante comporta os quatro atributos do M5; os demais parâmetros permanecem iguais.
- `PSE_context_selected`: subconjunto escolhido por ganho operacional interno.
- `PSE_context_guarded`: utiliza a variante selecionada somente quando a validação interna favorece a ampliação em relação à básica.

Os modelos e as escolhas ficam em `folds/<identificador>/models.joblib` e `audit.json`. O teste só é lido para apurar consequências depois que as recomendações de todas as dobras e dos dois modelos foram gravadas em `predictions_frozen_before_test.csv`. Os resultados do teste não escolhem atributos nem alteram a ativação retroativamente.

## Bahia

Fonte completa: `aipe_ba145_unified_2026-10-15`, 4.350 ajustes já concluídos. São 290 casos, 145 séries, 115 revendedores, duas origens; 460 dobras considerando os dois modelos.

Contexto: idade conforme registrada, segmento comercial e filial. Não há atributos de renda nem interpretação do segmento como classe socioeconômica. Não se reconstruíram idades a partir de aniversários supostos.

A disponibilidade temporal depende da confirmação explícita do autor: **“sim são do mesmo periodo, eu coletei direto do banco”**. O protocolo registra `assumption_declared_snapshot_temporal_validity`: fotografia cadastral do mesmo período, sem histórico independente de alterações por campo. É uma premissa declarada e deve acompanhar qualquer interpretação dos resultados.

Não se inventaram datas civis. O contrato é ordinal: prefixos encerrados nos ciclos 202317/202406, validações encerradas em 202407/202413, decisões de teste em 202408/202414. Os atributos básicos continuam calculados no prefixo de calibração, por simetria com o protocolo canônico, embora os resultados de validação só estejam disponíveis na decisão posterior.

Pré-checagem: nenhuma ausência nos três atributos; idades finitas entre 23 e 70 anos.

- Preparado: `experiment/data/context_extension/BA/prepared_2026-10-17_cadastro_v2`.
- Estudo: `experiment/data/context_extension/BA/study_2026-10-17_cadastro_v1`.

## M5

Fonte com 79.890 controladores concluídos, 5.326 casos nas origens 50/64/78, dez lojas; 60 dobras considerando os dois modelos.

Contexto:

1. Departamento do produto, obtido da taxonomia invariável do benchmark e conferido por loja/produto/estado.
2. Preço unitário histórico médio nas semanas integralmente encerradas nos últimos 21 dias do prefixo de calibração. **É um atributo contextual; os custos da simulação continuam fixos e não passam a representar preços ou economia monetária efetiva.**
3. Fração de dias com indicação SNAP no estado correto, no último bloco completo de 21 dias da calibração.
4. Contagem de eventos no mesmo bloco passado. É constante dentro de cada origem; sua rejeição automática é esperada e será auditada. Não se fabricou variação com identificadores.

O alinhamento usa dias consecutivos a partir de `d_1 = 2011-01-29`, com 21 dias por ciclo. Não usa calendário futuro, preço realizado futuro nem o rótulo sintético YYYYCC como data civil.

| Origem | Janela passada do contexto | Primeiro dia do teste |
|---|---|---|
| 50 | 23/11/2013 a 13/12/2013 | 10/05/2014 |
| 64 | 13/09/2014 a 03/10/2014 | 28/02/2015 |
| 78 | 04/07/2015 a 24/07/2015 | 19/12/2015 |

O intervalo de validação separa o prefixo e o teste. Manter os atributos contextuais no prefixo, como os básicos, é uma escolha deliberada deste contraste. Os alvos de validação já precisam estar disponíveis quando se recomenda para o teste.

Pré-checagem: nenhuma ausência nos quatro atributos em qualquer origem ou treino externo. Departamento tem três categorias. SNAP oferece apenas três/dois/dois valores estaduais nas origens: as séries que compartilham o estado não são réplicas independentes desse contexto. A avaliação externa continua agrupada por loja, não valida generalização para estados novos.

O novo frontend `context_m5_completed_matrix.py` pode iniciar depois dos controladores, enquanto a avaliação canônica de seletores/oráculos ainda termina. Ele exige contagem 79.890 e verifica **exatamente 159.780 linhas caso × política × semente esperada × validação/teste**, sem duplicatas ou casos extras. Confere hashes de plano, painéis e matriz, captura o estado verdadeiro (`controllers_completed` ou `completed`) e não altera o status canônico. O status mutável é registrado como proveniência, sem tratá-lo como dependência imutável; as matrizes científicas permanecem protegidas por hash.

- Preparado: `experiment/data/context_extension/M5/prepared_2026-10-17_context_v1`.
- Estudo: `experiment/data/context_extension/M5/study_2026-10-17_context_v1`.

## Operação e retomada

As duas bases utilizam quatro processos cada, uma thread e prioridade reduzida: oito processos adicionais no total. A avaliação canônica preexistente é separada. Nenhum script da simulação foi editado.

Cada estudo possui `status.json`, `runner.stdout.log` e `runner.stderr.log`. Cada dobra possui confirmação e hashes; uma retomada reutiliza dobras completas, recusando checkpoints danificados ou mudança no protocolo/entradas. Os comandos abaixo são de **retomada explícita**, nunca devem criar uma segunda execução concorrente do mesmo estudo:

```powershell
& experiment/.venv/Scripts/python.exe experiment/scripts/operations/context_rounds.py run --prepared experiment/data/context_extension/BA/prepared_2026-10-17_cadastro_v2 --output experiment/data/context_extension/BA/study_2026-10-17_cadastro_v1 --workers 4
& experiment/.venv/Scripts/python.exe experiment/scripts/operations/context_m5_completed_matrix.py run --prepared experiment/data/context_extension/M5/prepared_2026-10-17_context_v1 --output experiment/data/context_extension/M5/study_2026-10-17_context_v1 --workers 4
```

Preparações executadas antes desses comandos:

```powershell
& experiment/.venv/Scripts/python.exe experiment/scripts/operations/context_rounds.py prepare --dataset BA --output experiment/data/context_extension/BA/prepared_2026-10-17_cadastro_v2
& experiment/.venv/Scripts/python.exe experiment/scripts/operations/context_m5_completed_matrix.py prepare --output experiment/data/context_extension/M5/prepared_2026-10-17_context_v1
```

A configuração padrão `experiment/conf/aipe_context_extension_v1.json` permanece desligada. As rodadas autorizadas são execuções explícitas com protocolos próprios congelados; não ativam automaticamente o módulo no pipeline canônico.

Processos efetivos no lançamento: Bahia PID **11412**, M5 PID **35696**, ambos `BelowNormal`. Os launchers do ambiente virtual são 32800/33816. O PID e o progresso atuais devem ser consultados em `status.json`; o número pode mudar em uma retomada. A estimativa inicial da Bahia, após as primeiras dobras, ficou em aproximadamente 20–35 minutos com quatro processos, sujeita ao custo dos subconjuntos e da instância por consequências. Nenhuma garantia de horário de conclusão foi imposta.

Testes sintéticos: `context_rounds_test.py` verifica duas instâncias, todos os cinco rótulos de saída, populações pareadas, serialização e retomada sem repetir dobras; `context_m5_completed_matrix_test.py` verifica recusa parcial, matriz completa, ausência de alteração do status verdadeiro, duplicatas e fonte alterada durante auditoria.

As saídas principais serão `summary.csv`, `case_metrics.csv` e `comparison.json`. Devem ser interpretadas como desenvolvimento, com distribuição dos efeitos e casos prejudicados, não apenas médias. Ausência de ganho é resultado possível; não será usada para alterar o protocolo com base no teste.
