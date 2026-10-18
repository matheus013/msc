# Piloto de proteção do atendimento no AIPE

## Escopo fixado antes da nova execução

Estudo de desenvolvimento com as lojas CA_1, CA_2, TX_1 e WI_1 do M5, nas origens 50, 64 e 78. Seleção operacional: duas primeiras lojas CA em ordem alfabética e primeira loja TX/WI; não houve ranqueamento de lojas pelo resultado. Os resultados canônicos já foram examinados, inclusive o diagnóstico das 19 novas violações. Portanto, a nova rodada não é confirmação independente, mesmo que não use o teste para calibrar parâmetros.

O ponto de recuperação anterior ao estudo foi publicado no GitHub, branch `enxugamento-texto`, commit `2a509f2bd058f799fee7ae998bbf86c982ac53ac`. As matrizes e os dados continuam locais/Drive. Nenhum controlador de estoque será treinado novamente.

## Comparações

1. **Atual:** reproduzir exatamente a candidata, a referência, a margem de custo e a decisão congeladas da avaliação consolidada.
2. **Folga simples:** conservar os critérios anteriores e exigir atendimento previsto de pelo menos 71% e vantagem prevista de serviço de pelo menos um ponto percentual. Esses valores são escolhas exploratórias explícitas, não tolerâncias aprovadas pelo gestor nem limiares escolhidos para eliminar as 19 falhas.
3. **Correção empírica de erro:** conservar os critérios anteriores e descontar erros de superestimação do atendimento absoluto e da diferença candidata–referência. Exigir atendimento corrigido de pelo menos 70% e diferença corrigida não negativa.

As duas alternativas apenas vetam trocas que a regra atual admitiria. Não mudam a candidata para outra família, não reativam uma troca rejeitada pela margem de custo e não recalibram políticas ou custos.

## Estimação dos erros

Em cada origem e loja externa, excluir integralmente essa loja dos dados de ajuste e das consequências usadas na calibração. Aplicar cinco divisões internas por grupos de lojas, usando o mesmo modelo canônico de consequências: ExtraTrees, 64 árvores, profundidade quatro, folha mínima três e semente 20261015.

Para cada política, obter resíduos de serviço em previsões internas fora da amostra. O erro absoluto é previsão menos atendimento observado na validação; o erro diferencial é a diferença prevista entre política e referência menos a diferença observada. A referência é a referência externa já congelada; os resíduos são calculados para todas as políticas antes de consultar a candidata do caso externo.

Usar o percentil empírico de 90% de cada erro, atribuindo a cada loja o mesmo peso total e truncando correções negativas em zero. Exigir pelo menos três lojas com resíduos definidos; sem suporte, conservar a referência. Esse desconto é uma aproximação exploratória: não produz garantia de cobertura, intervalo de confiança, não inferioridade ou proteção contra mudanças de distribuição. Os modelos internos usam menos lojas que o modelo externo original; essa diferença também limita a transferência da correção.

O teste não entra nos ajustes, quantis ou escolha das regras. A escolha de famílias de modelos e parâmetros aproveita conhecimento de desenvolvimento já existente; isso precisa permanecer declarado.

## Congelamento e avaliação

Salvar protocolo, hashes, lojas excluídas, resíduos internos, correções e previsões por caso antes de executar o comando separado de pontuação. A regra atual deve reproduzir políticas, custos e atendimento da matriz original no subconjunto. A origem dos resíduos e a exclusão das lojas são verificadas por asserções. Arquivos canônicos não são alterados.

Relatar custo médio, atendimento médio com denominador definido, intervenções, novas violações, recuperações, casos com maior custo e casos com menor serviço. Um recuo generalizado pode remover risco observado e simultaneamente remover valor incremental; ausência de trocas não demonstra superioridade preditiva. Poucas lojas e conhecimento prévio dos testes impedem uma conclusão confirmatória.

Após o piloto, examinar o compromisso entre economia e atendimento antes de ampliar o desenvolvimento para as demais lojas e Bahia. Uma confirmação posterior exige períodos ou unidades não utilizados para desenvolver as escolhas. A relevância e a tolerância gerencial não são definidas automaticamente por este piloto.

## Execução

Na raiz, executar primeiro os testes específicos e depois:

```powershell
& experiment/.venv/Scripts/python.exe experiment/scripts/analysis/run_service_guard_pilot.py run --workers 3
& experiment/.venv/Scripts/python.exe experiment/scripts/analysis/run_service_guard_pilot.py score
```

O primeiro comando usa três processos CPU, uma thread por processo e prioridade reduzida. Pode retomar dobras concluídas se código e protocolo permanecerem iguais. O segundo exige previsões congeladas e hashes compatíveis. Saídas: `experiment/data/service_guard_pilot/m5_four_stores_v1/`. Os quatro testes de contrato cobrem veto de custo, correções absoluta/diferencial, suporte e ponderação por loja.
