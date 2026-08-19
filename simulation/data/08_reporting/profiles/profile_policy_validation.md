# Validação — Avaliação por Perfil Operacional

Gerado em: 2026-08-18 19:40

## Fonte dos dados
- KPIs: `data\07_model_output\kpis.parquet`
- Perfis: `data\04_feature\demand_profiles.parquet`

## Granularidade
- Uma linha por (série loja-produto, política) em kpis.parquet
- Resultados agregados sobre replicações na geração de kpis.parquet

## Cobertura
- Séries (loja, produto): **145** (Experimento 2, BA)
- Políticas avaliadas: **18**
- Perfis operacionais presentes: **3** de 5 definidos

## Distribuição por perfil
- **High Vol. Seasonal**: 11 séries | dominante: Fixed Interval | CTI=422.9 | NS=0.91
- **Sparse High Impact**: 116 séries | dominante: EOQ | CTI=632.5 | NS=0.94
- **Unstable Trend**: 18 séries | dominante: EOQ | CTI=608.0 | NS=0.95

## Regra de dominância
- Políticas viáveis: NS médio >= 0.7
- Política dominante: menor CTI médio entre viáveis
- Fallback: maior NS médio quando nenhuma política é viável

## Consistência com Tabela 5.2 (agregado global)

| Política | CTI médio (kpis.parquet) |
|---|---|
| EOQ | 628.42 |
| (s,S) | 654.86 |
| Jornaleiro | 289.92 |
| PIL | 178.32 |
| Capped Base-Stock | 221.33 |
| Big Data Newsvendor | 582.11 |
| Min-Max | 261.94 |
| Fixed Interval | 457.81 |
| Vendor-Responsive | 433.31 |
| GA | 1269.59 |
| SA | 1165.35 |
| PSO | 1209.36 |
| DE | 1049.18 |
| DQN | 116.58 |
| PPO | 221.54 |
| SARSA | 358.06 |
| GA-DQN | 1183.85 |
| GA-PPO | 3189.22 |

## Limitações
- Análise concentrada no regime *Lumpy* (Experimentos 1 e 2).
- Perfis `Low_Vol_Stable` e `Fast_Moving` não têm séries no Experimento 2.
- Perfis com poucas séries (n < 20) devem ser interpretados de forma exploratória.
- Generalização para outros regimes é objetivo do Experimento 3.