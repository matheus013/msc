# Revisor de método e evidência — rodada 2

**Decisão editorial: approve. Pronto cientificamente para submissão: não.** Na minha área, o artigo está suficientemente claro e correto para circular como manuscrito de trabalho. Não restam reparos editoriais obrigatórios identificados nesta verificação.

Versão lida: snapshot `round_02/source`, manuscrito SHA-256 real `7e6f1127d57348806b8339eab6d038818847d7306ce45d31d902e6e4ed22ad62`. Li meu parecer da rodada 1, a resposta do escritor e os trechos efetivamente alterados, sem consultar os pareceres de rodada 2 dos demais revisores. Conferi os 16 hashes de `frozen_code_checked.json`: todos correspondem aos arquivos históricos locais.

## Verificação dos reparos

| Achado | Situação | Evidência e critério satisfeito |
|---|---|---|
| MTH-01 | Resolvido | O texto agora diferencia objetivo de calibração, custo operacional reportado e viabilidade posterior na validação. Identifica peso 30, escala relativa ao custo para GA e escala intrínseca da demanda para RL/híbrido e Capped. A penalidade terminal consta no código congelado de RL e híbrido; riscos unitários da Bahia e zero no piloto M5 foram conferidos. Explicita que penalidade não garante atendimento. |
| MTH-02 | Resolvido | “Calibradas exclusivamente no prefixo de treino” elimina a leitura de ajuste no teste e conserva a explicação do esforço experimental. |
| MTH-03 | Resolvido | Define média aritmética por caso após agregar sementes, exclusão de atendimento indefinido e ausência de garantia individual; global é agora explicitamente comum à partição/origem, podendo variar entre elas. |

A formalização adicional corresponde às funções `_eval_static`, `_terminal_deficit_penalty`, `series_tic_ref`, `_risk_terms` e `_calibrate_batch` nos snapshots históricos, sem atribuir ao RL uma recompensa exclusivamente de custo. A declaração de que a calibração da Capped difere entre matrizes continua necessária e foi preservada.

A tabela de consequências mantém valores coerentes após normalização de porcentagens para frações. Permanecem explícitos os 90 casos de custo, 86 de atendimento, 47 viáveis e três mudanças. Nenhuma nova generalização ou garantia foi introduzida nos trechos inspecionados. O texto continua distinguindo média entre sementes de realização individual na comparação com informação perfeita, e incerteza condicional de confirmação independente.

## Pendências científicas preservadas

**MTH-04 — major, experiment, open:** confirmação independente da regra de recuo. O piloto orientou o desenvolvimento; validação aninhada interna não elimina essa reutilização. A evidência favorável permanece concentrada e exploratória. Não há reparo puramente textual que substitua avaliação reservada com protocolo fixado.

**MTH-05 — major, experiment, open:** robustez e calibração comparável. Harmonização e sensibilidades justificadas de custo, prazo e horizonte ainda são necessárias antes de conclusões amplas. Orçamento desigual e dependência da instanciação seguem reconhecidos. Nenhum resultado futuro foi antecipado.

O fechamento desta área é editorial. Não equivale a aprovação científica de submissão, revisão humana independente nem resolução de pendências experimentais ou autorais. Não executei nem modifiquei experimentos, manuscritos ou dissertação.
