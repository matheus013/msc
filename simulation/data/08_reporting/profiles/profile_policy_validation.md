# Validação — Avaliação por Perfil Operacional

Gerado em: 2026-06-21 21:23

## Fonte dos dados
- KPIs: `data\07_model_output\kpis.parquet`
- Perfis: `data\04_feature\demand_profiles.parquet`

## Granularidade
- Uma linha por (série loja-produto, política) em kpis.parquet
- Resultados agregados sobre replicações na geração de kpis.parquet

## Cobertura
- Séries (loja, produto): **145** (Fase 2, BA)
- Políticas avaliadas: **12**
- Perfis operacionais presentes: **3** de 5 definidos

## Distribuição por perfil
- **High Vol. Seasonal**: 11 séries | dominante: EOQ | CTI=618.8 | NS=0.97
- **Sparse High Impact**: 116 séries | dominante: SA | CTI=584.0 | NS=0.86
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
| GA | 1078.61 |
| SA | 635.26 |
| PSO | 1099.37 |
| DE | 1156.04 |
| DQN | 120.23 |
| PPO | 3871.75 |
| SARSA | 874.05 |
| GA-DQN | 1098.91 |
| GA-PPO | 2240.94 |

## Limitações
- Análise concentrada no regime *Lumpy* (Fases 1 e 2).
- Perfis `Low_Vol_Stable` e `Fast_Moving` não têm séries na Fase 2.
- Perfis com poucas séries (n < 20) devem ser interpretados de forma exploratória.
- Generalização para outros regimes é objetivo da Fase 3.