# Validação — Comparação de Estratégias de Política de Inventário

Gerado em: 2026-08-18 19:41

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
  Política dominante global (por CTI ajustado): **EOQ**
  Políticas viáveis (NS >= 0.7): ['BigDataNewsvendor', 'DE', 'EOQ', 'FixedInterval', 'GA', 'GA-DQN', 'GA-PPO', 'PSO', 'SA', 'VendorResponsive', 'sS']

## Checagem 4 — Dominância por perfil (B)
  High Vol. Seasonal (*): FixedInterval | CTI=422.93 | CTI_ajustado=422.93 | NS=0.907 | status=normal
  Unstable Trend (*): EOQ | CTI=607.95 | CTI_ajustado=607.95 | NS=0.947 | status=normal
  Sparse High Impact: EOQ | CTI=632.51 | CTI_ajustado=686.35 | NS=0.939 | status=normal
  (*) n < 20: evidência exploratória

## Checagem 5 — Redução de CTI ajustado (fórmula verificada)
  redução (%) = 100 × (CTI_ajustado_A1_total − CTI_ajustado_B_total) / CTI_ajustado_A1_total
  CTI_ajustado = CTI + deficit_NS*penalty_weight*CTI_ref_serie(fixo) + excess_weight*excesso_holding (AJUSTES_INFRA item #33; corrige auto-referência do score antigo e incorpora estoque excessivo + indisponibilidade).
  A2 (Política baseline (EOQ)…): CTI_total=91120.69 | red_pct_vs_A1=0.0%
  B (Seleção por perfil operacional…): CTI_total=88966.2 | red_pct_vs_A1=2.21%
  C (Oráculo por série (exploratório)…): CTI_total=52452.91 | red_pct_vs_A1=46.13%

## Checagem 6 — NS médio preservado
  A1: NS_medio=0.942
  A2: NS_medio=0.942
  B: NS_medio=0.938
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