# Validação — Comparação de Estratégias de Política de Inventário

Gerado em: 2026-08-18 12:04

## Fonte dos dados
- KPIs: `kpis.parquet` (caminho isolado por ambiente, ver conf/*/catalog.yml)
- Perfis: `demand_profiles.parquet` (idem)

## Cobertura
- Séries (loja, produto): **145**
- Políticas avaliadas: **18**
- Perfis operacionais: **3** de 5 definidos

## Checagem 1 — Quantidade de séries
  Esperado: 145 | Encontrado: 145 | OK

## Checagem 2 — Políticas
  Encontradas: ['BigDataNewsvendor', 'CappedBaseStock', 'DE', 'DQN', 'EOQ', 'FixedInterval', 'GA', 'GA-DQN', 'GA-PPO', 'MinMax', 'Newsvendor', 'PIL', 'PPO', 'PSO', 'SA', 'SARSA', 'VendorResponsive', 'sS']
  Esperadas:   ['BigDataNewsvendor', 'CappedBaseStock', 'DE', 'DQN', 'EOQ', 'FixedInterval', 'GA', 'GA-DQN', 'GA-PPO', 'MinMax', 'Newsvendor', 'PIL', 'PPO', 'PSO', 'SA', 'SARSA', 'VendorResponsive', 'sS']
  Match: OK

## Checagem 3 — Política única global (A1)
  Política dominante global: **VendorResponsive**
  Políticas viáveis (NS >= 0.7): ['BigDataNewsvendor', 'DE', 'EOQ', 'FixedInterval', 'GA', 'GA-DQN', 'GA-PPO', 'PSO', 'SA', 'VendorResponsive', 'sS']

## Checagem 4 — Dominância por perfil (B)
  High Vol. Seasonal (*): VendorResponsive | CTI=395.15 | NS=0.797 | status=normal
  Sparse High Impact: VendorResponsive | CTI=432.78 | NS=0.779 | status=normal
  Unstable Trend (*): FixedInterval | CTI=453.21 | NS=0.79 | status=normal
  (*) n < 20: evidência exploratória

## Checagem 5 — Redução de CTI (fórmula verificada)
  redução (%) = 100 × (CTI_A1_total − CTI_B_total) / CTI_A1_total
  A2 (Política baseline (EOQ)…): CTI_total=91120.69 | red_pct_vs_A1=-45.03%
  B (Seleção por perfil operacional…): CTI_total=62706.67 | red_pct_vs_A1=0.2%
  C (Oráculo por série (exploratório)…): CTI_total=52452.91 | red_pct_vs_A1=16.52%

## Checagem 6 — NS médio preservado
  A1: NS_medio=0.774
  A2: NS_medio=0.942
  B: NS_medio=0.782
  C: NS_medio=0.7

## Checagem 7 — Consistência com Tabela 5.2 (agregado global)
  Os valores de CTI aqui NÃO são idênticos à Tabela 5.2 (rodada anterior).
  Após regenerar kpis.parquet com a rodada final, reexecutar este script.
  EOQ CTI médio (kpis.parquet atual): 628.42
  EOQ CTI médio (Tabela 5.2 final):    628,42

## Limitações
- Perfis Low_Vol_Stable e Fast_Moving ausentes no Experimento 2 (regime Lumpy, BA).
- Perfis com n < 20 séries: evidência exploratória.
- Oráculo por série (C) é limite superior exploratório, não estratégia operacional.
- Generalização para regimes não-Lumpy: objetivo do Experimento 3.