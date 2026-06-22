# Validação — Comparação de Estratégias de Política de Inventário

Gerado em: 2026-06-21 21:23

## Fonte dos dados
- KPIs: `data/07_model_output/kpis.parquet`
- Perfis: `data/04_feature/demand_profiles.parquet`

## Cobertura
- Séries (loja, produto): **145** (Fase 2, BA)
- Políticas avaliadas: **12**
- Perfis operacionais: **3** de 5 definidos

## Checagem 1 — Quantidade de séries
  Esperado: 145 | Encontrado: 145 | OK

## Checagem 2 — Políticas
  Encontradas: ['DE', 'DQN', 'EOQ', 'GA', 'GA-DQN', 'GA-PPO', 'Newsvendor', 'PPO', 'PSO', 'SA', 'SARSA', 'sS']
  Esperadas:   ['DE', 'DQN', 'EOQ', 'GA', 'GA-DQN', 'GA-PPO', 'Newsvendor', 'PPO', 'PSO', 'SA', 'SARSA', 'sS']
  Match: OK

## Checagem 3 — Política única global (A1)
  Política dominante global: **EOQ**
  Políticas viáveis (NS >= 0.7): ['DE', 'EOQ', 'GA', 'GA-DQN', 'GA-PPO', 'PPO', 'PSO', 'SA', 'sS']

## Checagem 4 — Dominância por perfil (B)
  Sparse High Impact: SA | CTI=584.01 | NS=0.865 | status=normal
  Unstable Trend (*): EOQ | CTI=607.95 | NS=0.947 | status=normal
  High Vol. Seasonal (*): EOQ | CTI=618.79 | NS=0.97 | status=normal
  (*) n < 20: evidência exploratória

## Checagem 5 — Redução de CTI (fórmula verificada)
  redução (%) = 100 × (CTI_A1_total − CTI_B_total) / CTI_A1_total
  A2 (Política baseline (EOQ)…): CTI_total=91120.69 | red_pct_vs_A1=0.0%
  B (Seleção por perfil operacional…): CTI_total=85495.48 | red_pct_vs_A1=6.17%
  C (Oráculo por série (exploratório)…): CTI_total=50433.59 | red_pct_vs_A1=44.65%

## Checagem 6 — NS médio preservado
  A1: NS_medio=0.942
  A2: NS_medio=0.942
  B: NS_medio=0.883
  C: NS_medio=0.7

## Checagem 7 — Consistência com Tabela 5.2 (agregado global)
  Os valores de CTI aqui NÃO são idênticos à Tabela 5.2 (rodada anterior).
  Após regenerar kpis.parquet com a rodada final, reexecutar este script.
  EOQ CTI médio (kpis.parquet atual): 628.42
  EOQ CTI médio (Tabela 5.2 final):    628,42

## Limitações
- Perfis Low_Vol_Stable e Fast_Moving ausentes na Fase 2 (regime Lumpy, BA).
- Perfis com n < 20 séries: evidência exploratória.
- Oráculo por série (C) é limite superior exploratório, não estratégia operacional.
- Generalização para regimes não-Lumpy: objetivo da Fase 3.