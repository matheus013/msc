# Relatorio de Recomendacao — Pares (Loja, Produto) para Dissertacao

**Objetivo**: Selecionar series temporais representativas de cada categoria
de demanda intermitente (Syntetos-Boylan 2005) para analise aprofundada na
dissertacao de mestrado.

## Metodologia de Selecao

| Criterio              | Peso | Justificativa |
|------------------------|------|---------------|
| MASE elevado (dificuldade) | 35% | Series desafiadoras sao mais informativas |
| Demanda media (volume) | 25% | Relevancia economica do par loja-produto |
| Riqueza de dados (n ciclos) | 20% | Mais ciclos = validacao mais confiavel |
| CV elevado (variabilidade) | 20% | Variabilidade e central no estudo |

**Modelo de referencia para metricas de forecast**: LSTM

## Grupos Syntetos-Boylan

| Grupo | ADI | CV² | Caracteristica |
|-------|-----|-----|----------------|
| Smooth | < 1.32 | < 0.49 | Demanda regular, facil de prever |
| Erratic | < 1.32 | >= 0.49 | Demanda frequente mas tamanho erratico |
| Intermittent | >= 1.32 | < 0.49 | Demanda infrequente, tamanhos estaveis |
| Lumpy | >= 1.32 | >= 0.49 | Demanda infrequente e erratica (mais dificil) |

## Series Recomendadas por Grupo

### Lumpy

| # | Estado | Loja | Produto | CV | ADI | Ciclos | Demanda Media | MASE | sMAPE | Score |
|---|--------|------|---------|-----|-----|--------|---------------|------|-------|-------|
| 1 | BA | 3522811 | 75792 | 3.89 | 1.73 | 38 | 9.2 | 0.494 | 122.6 | 0.527 |
| 2 | BA | 10533990 | 75792 | 1.07 | 1.81 | 38 | 0.8 | 1.872 | 191.9 | 0.465 |
| 3 | BA | 3522811 | 48062 | 2.72 | 1.81 | 38 | 3.0 | 1.009 | 162.4 | 0.410 |

## Sumario Executivo

- **Total de series avaliadas**: 145
- **Series recomendadas**: 3

### Por que estas series?

As series selecionadas maximizam simultaneamente:
1. **Dificuldade de previsao** (MASE > 1 indica que o modelo enfrenta mais
   incerteza que a previsao naive — ideal para demonstrar ganho da arquitetura proposta)
2. **Representatividade** de cada quadrante ADI-CV² para garantir que a
   dissertacao cobre toda a gama de comportamentos de demanda intermitente
3. **Relevancia economica** via volume de demanda (series com TIC mais alto
   mostram maior impacto de uma politica de estoque superior)

### Proximos passos

1. Validar as series recomendadas visualmente (`forecast/forecast_vs_actual.pdf`)
2. Confirmar que cada grupo tem pelo menos 3 ciclos de teste possiveis
3. Para a dissertacao, usar as series com MASE > 1 como caso principal e
   series com MASE < 1 como casos em que a previsao ja funciona bem

---
*Relatorio gerado automaticamente pelo pipeline Kedro (reporting pipeline)*