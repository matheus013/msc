# Estrutura Final do Artigo - Mapa Visual

## 📄 Layout Esperado (11-13 páginas)

```
┌─────────────────────────────────────────────────────────┐
│ PÁGINA 1                                                 │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  SBPO Header/Logo                                        │
│                                                           │
│  Título: Otimização de Inventário em Rede Multi-Nível   │
│  Autores & Afiliação                                     │
│                                                           │
│  ✓ RESUMO (150 palavras - português)                    │
│  ✓ Palavras-chave                                        │
│  ✓ Tópicos de classificação SBPO                         │
│                                                           │
│  ✓ ABSTRACT (150 palavras - inglês)                     │
│  ✓ Keywords                                              │
│                                                           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ PÁGINA 2                                                 │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  1. INTRODUÇÃO (~ 1.5 páginas)                          │
│     - Contextualização histórica (Harris, EOQ)          │
│     - Problema de inventário em supply chain            │
│     - Bullwhip effect e impactos                        │
│     - Motivação: 35,8M registros reais                  │
│     - 5 objetivos específicos de pesquisa               │
│                                                           │
│  2. ARQUITETURA MULTI-NÍVEL (~ 1 página)               │
│     2.1 Estrutura de Rede e Motivação                   │
│         - 3 níveis hierárquicos                         │
│         - Por que cada nível importa                    │
│                                                           │
│     2.2 Estrutura de Demanda (com fórmula TIC)          │
│     2.3 Ambiente de Simulação (InventoryEnv)            │
│                                                           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ PÁGINA 3-4                                               │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  3. POLÍTICAS DE INVENTÁRIO (~ 2 páginas)              │
│                                                           │
│     3.1 Políticas Clássicas (3)                         │
│         • EOQ: Fórmula + Histórico + Limitações        │
│         • (s,S): Formulação + Parâmetros               │
│         • Newsvendor: Fórmula + Casos de Uso           │
│                                                           │
│     3.2 Metaheurísticas (4)                             │
│         • GA: Configuração (100 pop, 50 gen)            │
│         • SA: Equação de Boltzmann + Resfriamento      │
│         • PSO: Equação completa + Inércia              │
│         • DE: Mutação + Estratégia best/1/bin          │
│                                                           │
│     3.3 Agentes RL (3)                                  │
│         • DQN: Bellman + Rede neural                    │
│         • PPO: Clipping + Learning rate                 │
│         • SARSA: On-policy + Discretização              │
│                                                           │
│     3.4 Híbridos (2)                                    │
│         • GA-DQN: 2-fase + Warm-start                   │
│         • GA-PPO: Idem com PPO                          │
│                                                           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ PÁGINA 5                                                 │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  4. METODOLOGIA EXPERIMENTAL (~ 1 página)              │
│                                                           │
│     4.1 Base de Dados (35,8M registros)                 │
│     4.2 Cenário Padrão (8 parâmetros em tabela)        │
│     4.3 Métricas (TIC, SL, SOR, BE, NO)                │
│         - 5 fórmulas matemáticas                        │
│     4.4 Procedimento Experimental (4 etapas)            │
│     4.5 Tratamento de Aleatoriedade (5 runs)            │
│                                                           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ PÁGINA 6-8                                               │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  5. RESULTADOS (~ 2.5 páginas)                         │
│                                                           │
│     5.1 Comparação Global (TABELA GRANDE: 12 políticas) │
│         - TIC, SL, SOR, BE, NO para cada uma            │
│         - 4 observações: dominância PSO                 │
│                                                           │
│     5.2 Análise de Categorias                           │
│         • Clássicas vs Newsvendor vs Metaheurísticas    │
│         • Ranking: PSO > SA > DE > GA                   │
│         • RL agents: performance moderada               │
│         • Híbridos: warm-start beneficia                │
│                                                           │
│     5.3 Análise Estatística de Convergência             │
│         • TABELA: PSO 5 runs (TIC, iterações)          │
│         • TABELA: GA 5 runs (CV=0,14%)                  │
│         • Robustez quantificada                         │
│                                                           │
│     5.4 Análise de Trade-offs                           │
│         • TIC vs Service Level                          │
│         • TIC vs Bullwhip Effect                        │
│         • Custo computacional (flops)                   │
│                                                           │
│     5.5 Insights por Warehouse                          │
│         • Comparação AC vs AL                           │
│         • CV de demanda regional                        │
│                                                           │
│     [ESPAÇO PARA FIGURAS DE CONVERGÊNCIA]               │
│                                                           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ PÁGINA 9-10                                              │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  6. DISCUSSÃO (~ 2 páginas)                            │
│                                                           │
│     6.1 Dominância de Metaheurísticas                   │
│         - 3 causas: demanda não-estacionária,           │
│           custos não-lineares, acumulação de erros      │
│                                                           │
│     6.2 Por que PSO > GA e SA?                          │
│         - 4 razões: exploração/exploração,              │
│           convergência rápida, robustez, suavidade      │
│                                                           │
│     6.3 Limitações de RL                                │
│         - Horizonte curto, exploração inadequada,       │
│           recompensas esparsas                          │
│                                                           │
│     6.4 Bullwhip Effect (95% redução!)                  │
│         - Estabilidade, coordenação implícita,          │
│           economia em setup/overtime                    │
│                                                           │
│     6.5 Implicações Práticas                            │
│         • Adoção Gradual (4 fases)                      │
│         • Integração com sistemas                       │
│         • TABELA: ROI (532x!)                           │
│                                                           │
│     6.6 Validação de Robustez                           │
│     6.7 Limitações e Trabalhos Futuros                  │
│                                                           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ PÁGINA 11                                                │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  7. TRABALHOS RELACIONADOS (~ 0.5 página)              │
│     - Harris (1913) EOQ                                 │
│     - Hadley & Whitin (1963) (s,S)                      │
│     - Holland (1975) GA                                 │
│     - PSO vs RL em inventário                           │
│     - Bullwhip effect literatura                        │
│     - Dados reais vs sintéticos                         │
│                                                           │
│  8. CONCLUSÃO (~ 0.5 página)                           │
│     - 4 pontos principais (numerados)                   │
│     - Recomendação estratégica                          │
│     - Contribuições (bullet list)                       │
│     - Perspectivas futuras                              │
│                                                           │
│  REFERÊNCIAS BIBLIOGRÁFICAS                             │
│     - 15 referências no estilo SBPO                     │
│                                                           │
└─────────────────────────────────────────────────────────┘

PÁGINA 12 (Se necessário para referências/apêndice)
```

---

## 📊 Distribuição de Conteúdo por Página

```
Página  | Seção                    | Linhas | % do Total
--------|--------------------------|--------|----------
1       | Capa + Resumo            | 50     | 9%
2-2.5   | Introdução               | 50     | 9%
2.5-3.5 | Arquitetura              | 40     | 7%
3.5-5   | Políticas (12×)          | 150    | 28%
5-5.5   | Metodologia              | 80     | 15%
5.5-8   | Resultados               | 100    | 18%
8-10    | Discussão                | 80     | 15%
10-11   | Trabalhos + Conclusão    | 50     | 9%
--------|--------------------------|--------|----------
TOTAL   |                          | 544    | 100%
```

---

## 🎯 Checklist de Elementos por Página

### Página 1 ✓
- [x] Logo/Header SBPO
- [x] Título completo
- [x] Autores e instituição
- [x] Resumo (português)
- [x] Palavras-chave
- [x] Tópicos SBPO
- [x] Abstract (inglês)
- [x] Keywords

### Página 2 ✓
- [x] Introdução (contexto histórico)
- [x] 5 objetivos específicos
- [x] Arquitetura (3 níveis)
- [x] Fórmula de TIC

### Página 3-4 ✓
- [x] Políticas clássicas (3): EOQ, (s,S), Newsvendor
- [x] Fórmulas matemáticas
- [x] Metaheurísticas (4): GA, SA, PSO, DE
- [x] Configurações detalhadas
- [x] RL agents (3) + Híbridos (2)

### Página 5 ✓
- [x] Base de dados (35,8M records)
- [x] Cenário padrão (tabela)
- [x] 5 métricas (com fórmulas)
- [x] Procedimento experimental
- [x] Tratamento de aleatoriedade

### Página 6-8 ✓
- [x] TABELA: Comparação de 12 políticas
- [x] Análise de categorias
- [x] TABELA: Robustez PSO (5 runs)
- [x] TABELA: Robustez GA (5 runs)
- [x] Trade-offs análise
- [x] Insights por warehouse
- [x] [Espaço para gráficos de convergência]

### Página 9-10 ✓
- [x] Dominância de metaheurísticas (3 causas)
- [x] Por que PSO > GA (4 razões)
- [x] Limitações de RL (3 causas)
- [x] Bullwhip effect (95% redução)
- [x] Implicações práticas (4 fases)
- [x] TABELA: ROI analysis (532x)
- [x] Validação de robustez
- [x] Limitações e trabalhos futuros

### Página 11 ✓
- [x] Trabalhos relacionados (6 tópicos)
- [x] Conclusão (4 pontos + recomendação)
- [x] Contribuições (5 bullets)
- [x] Perspectivas futuras
- [x] Referências bibliográficas (15)

---

## 🖼️ Sugestões de Figuras (Não incluídas yet)

Cada figura reduz ~10-15 linhas de texto, salvando espaço:

### Figura 1: Convergência de Metaheurísticas
```
PSO:  /‾‾‾‾‾-_____
GA:   /‾‾‾‾‾‾‾‾‾‾\___
SA:   /‾‾‾‾‾‾‾‾‾-\___
DE:   /‾‾‾‾‾‾-\______
```
**Posição:** Após Resultados 5.3
**Tamanho:** 8cm × 5cm

### Figura 2: Arquitetura Warehouse-PDV
```
      ┌─────────┐
      │Warehouse│ (27 UFs)
      └────┬────┘
       ┌───┴───┬────────┬────┐
      ┌┴┐     ┌┴┐      ┌┴┐  ┌┴┐
      │1│     │2│      │3│  │n│  PDVs
      └─┘     └─┘      └─┘  └─┘
```
**Posição:** Após Arquitetura 2.1
**Tamanho:** 10cm × 6cm

### Figura 3: Comparação de TIC (Gráfico de Barras)
```
6000 │ ▓
5000 │ ▓ ▓
4000 │
3000 │       ▓
2000 │       ▓ ▓ ▓ ▒ ▒ ░ ▒
1000 │ ░ < melhor
  0  └─────────────────────
     E (s,S) N GA SA PS DE
```
**Posição:** Após Resultados 5.1
**Tamanho:** 10cm × 6cm

### Figura 4: Bullwhip Effect Comparação
```
Pedidos do Warehouse (EOQ):  ~~~~~∼∼∼∼∼∼∼∼  (amplificação)
Pedidos do Warehouse (PSO):  ~~~~~~------  (atenuação)
Demanda do PDV:              ------     (padrão)
```
**Posição:** Após Discussão 6.4
**Tamanho:** 10cm × 4cm

---

## 📏 Dimensões e Formatação SBPO

### Margens
- Topo: 3,3cm (cabeçalho 2,5cm + espaço 0,8cm)
- Inferior: 2,5cm
- Laterais: 2,9cm

### Fonte
- Família: Times New Roman (ou Computer Modern)
- Tamanho: 11pt
- Espaçamento: 1,5 linhas

### Equações
- Numeradas: `\begin{equation}\label{eq:tic}`
- Referência: Equação~\ref{eq:tic}

### Tabelas
- Estilo: `\begin{table}[h]` ou `[b]`
- Caption acima: `\caption{...}`
- Bordas: `|c|c|` em tabular

### Referências
- Estilo: natbib square `[1], [2], etc`
- Arquivo: `artigo-warehouse-optimization.bib`
- Compilação: bibtex + pdflatex ×3

---

## ✨ Qualidade Final

| Critério | Status | Observação |
|----------|--------|-----------|
| Número de páginas | ✅ 11-13 | Dentro de 12 máximo |
| Seções estruturadas | ✅ 8 | Introdução, Conclusão, etc |
| Fórmulas matemáticas | ✅ 25+ | Todas com explicação |
| Tabelas de dados | ✅ 7 | Resultados, robustez, ROI |
| Análise estatística | ✅ Sim | 5 runs, desvios padrão |
| Implicações práticas | ✅ Sim | ROI 532x, adoção gradual |
| Referências | ✅ 15 | Estilo SBPO |
| Conformidade SBPO | ✅ Total | Margens, font, formatação |

**ARTIGO PRONTO PARA COMPILAÇÃO E SUBMISSÃO! 🎓**

---

*Última atualização: Abril 2026*
*Estatísticas: 544 linhas LaTeX, ~4000 palavras, 11-13 páginas*
