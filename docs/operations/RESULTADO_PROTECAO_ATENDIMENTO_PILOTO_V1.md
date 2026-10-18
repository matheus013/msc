# Resultado do piloto de proteção do atendimento

Execução de desenvolvimento concluída em 18/10/2026. Abrange 2.036 casos série–origem, com atendimento definido em 1.960, nas quatro lojas e três origens especificadas no protocolo. Não equivale ao M5 completo. Nenhum controlador de estoque foi treinado novamente; foram ajustados apenas modelos internos de consequências para estimar erros fora da amostra.

## Comparação principal

Variações de custo e atendimento são relativas à mesma referência global no subconjunto. O custo médio dessa referência é 707,69 unidades de cenário; seu atendimento médio é 67,20%.

| Regra | Custo médio | Economia relativa | Atendimento médio | Diferença de atendimento | Intervenções | Novas violações | Recuperações |
|---|---:|---:|---:|---:|---:|---:|---:|
| Atual | 700,16 | 1,064% | 67,428% | +0,228 p.p. | 266 | 10 | 11 |
| Folga simples | 703,70 | 0,565% | 67,443% | +0,243 p.p. | 181 | 6 | 8 |
| Correção empírica de erro | 707,69 | 0,000% | 67,201% | +0,000 p.p. | 0 | 0 | 0 |

A folga simples vetou 85 intervenções. Houve quatro novas violações a menos, mas também três recuperações a menos, e a economia foi aproximadamente metade da obtida pela regra atual. A regra simples apresentou 40 casos com atendimento menor que a referência, contra 70 na atual, e 65 casos com custo maior, contra 80. Essas contagens descrevem o subconjunto examinado e não estabelecem superioridade ou tolerância gerencial.

A proteção por erro conservou integralmente a referência. Sua ausência de novas violações não decorre de melhor escolha entre candidatas: ela não realizou nenhuma troca, não produziu recuperação nem economia incremental. O atendimento médio das três regras permanece abaixo de 70%; média e atendimento por caso são critérios distintos.

## Por que a correção bloqueou todas as trocas

Nas 266 intervenções admitidas pela regra atual, a vantagem prevista média de atendimento da candidata era de 1,77 ponto percentual. O desconto empírico para o erro dessa diferença apresentou mediana de 10,02 pontos, variando de 9,82 a 12,69 pontos. Nenhum caso manteve diferença corrigida não negativa.

O desconto para o erro absoluto de atendimento apresentou mediana de 23,03 pontos, variando de 22,29 a 26,32 pontos. Somente dois casos mantiveram atendimento corrigido acima da meta absoluta; nenhum passou simultaneamente pelos dois critérios. Esses descontos são quantis descritivos dos erros internos, não intervalos de confiança nem limites com cobertura garantida.

O resultado aponta uma incompatibilidade, nesta instanciação, entre vantagens previstas pequenas e erros internos grandes. Uma correção uniforme por par de políticas pode ser conservadora para contextos específicos; o piloto não permite concluir se seria possível estimar correções condicionais suficientemente confiáveis. Reduzir o percentil ou a folga até obter ganhos neste teste introduziria ajuste posterior aos resultados e não foi feito.

## Resultados por origem

| Origem | Casos | Trocas atual / simples / erro | Novas violações atual / simples / erro | Recuperações atual / simples / erro |
|---|---:|---:|---:|---:|
| 50 | 337 | 9 / 7 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 64 | 773 | 45 / 13 / 0 | 1 / 0 / 0 | 3 / 0 / 0 |
| 78 | 926 | 212 / 161 / 0 | 9 / 6 / 0 | 8 / 8 / 0 |

## Verificações realizadas

- Quatro testes específicos passaram: uma proteção não pode reativar troca vetada pelo custo; erros absoluto e diferencial podem vetar trocas; cada loja tem igual massa no quantil; ausência de suporte leva ao recuo.
- As doze dobras excluíram a loja externa dos ajustes e da calibração. Os grupos de ajuste e validação internos são disjuntos.
- Previsões foram gravadas e identificadas por hash antes de executar a pontuação externa em comando separado.
- A regra atual reproduziu exatamente as políticas e os indicadores canônicos no subconjunto.
- Hashes das entradas foram conferidos antes da pontuação. Matrizes, regras canônicas e resultados consolidados não foram alterados.

## Encaminhamento

Não substituir a regra canônica nem declarar benefício confirmado de nenhuma proteção. A folga simples oferece um compromisso observável, mas ainda preserva violações e sacrifica parte da economia. A correção empírica funcionou como mecanismo de abstenção total.

O próximo desenvolvimento deve examinar os erros internos por política, loja e contexto, buscando identificar se há suporte para uma correção menos uniforme. Uma nova formulação precisa ser fixada em protocolo separado e permanecer caracterizada como desenvolvimento nas matrizes já examinadas. A confirmação exige novos períodos ou unidades reservados. A tolerância a perdas de atendimento deve ser definida com o gestor, sem derivá-la automaticamente destes testes.

## Artefatos

- Protocolo: `docs/operations/PROTOCOLO_PROTECAO_ATENDIMENTO_PILOTO_V1.md`.
- Execução e testes: `experiment/scripts/analysis/run_service_guard_pilot.py` e `experiment/tests/test_service_guard_pilot.py`.
- Saídas locais: `experiment/data/service_guard_pilot/m5_four_stores_v1/`.
- Nessa pasta: `protocol.json`, auditorias de dobras, resíduos internos, previsões congeladas, `case_metrics.csv`, `summary.csv`, `summary.json` e `status.json`.

Os resultados deste relatório são exploratórios, com apenas quatro lojas e conhecimento prévio dos testes canônicos. Não foram adicionados ao artigo como confirmação independente.
