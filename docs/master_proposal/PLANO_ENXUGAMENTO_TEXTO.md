# Plano de Enxugamento do Texto — 124 → ~80 páginas

Criado em 19/08/2026, a pedido do autor: reduzir a dissertação de **124 páginas**
(medido no PDF compilado em 19/08/2026, após as correções de portfólio/hiperparâmetros
e o reenquadramento "estudo de caso" desta sessão) para **~80 páginas**, sem perder
rigor nem os resultados relevantes. O autor está disposto a **reduzir o número de
políticas do portfólio**, desde que a decisão seja baseada em literatura — essa é a
alavanca de maior impacto e a mais defensável academicamente, então o plano começa por ela.

---

## 1. Diagnóstico — onde estão as 124 páginas

| Capítulo | Páginas | Extensão |
|---|---|---|
| Pré-textual (capa, resumo, sumário, listas, siglas) | i–x | ~10 |
| 1. Introdução | 1–9 | 9 |
| 2. Referencial Teórico | 10–23 | 14 |
| 3. Trabalhos Relacionados | 24–31 | 8 |
| **4. Metodologia** | **32–59** | **28** ← maior capítulo |
| **5. Resultados Preliminares** | **60–79** | **20** |
| 6. Considerações Finais | 80–83 | 4 |
| 7. Plano de Continuidade | 84–88 | 5 |
| Apêndice A (matriz de parâmetros) | 89–96 | 8 |
| Apêndice B (representações visuais) | 97–103 | 7 |
| **Referências** (78 entradas) | **104–124** | **21** |
| **Total** | | **124** |

Quatro blocos concentram **77 das 124 páginas** (62%): Metodologia, Resultados,
Referências e Apêndices A+B. É aí que o corte precisa acontecer — tentar tirar 44
páginas só de Introdução/Referencial Teórico/Trabalhos Relacionados (31 páginas ao
todo) é matematicamente impossível sem descaracterizar esses capítulos.

**Meta de alocação por capítulo** (referência, não é regra rígida):

| Capítulo | Atual | Meta | Corte |
|---|---|---|---|
| Pré-textual | 10 | 10 | — |
| 1. Introdução | 9 | 7 | −2 |
| 2. Referencial Teórico | 14 | 10 | −4 |
| 3. Trabalhos Relacionados | 8 | 7 | −1 |
| 4. Metodologia | 28 | 17 | **−11** |
| 5. Resultados | 20 | 13 | **−7** |
| 6. Considerações Finais | 4 | 4 | — |
| 7. Plano de Continuidade | 5 | 5 | — |
| Apêndice A | 8 | 5 | −3 |
| Apêndice B | 7 | 4 | −3 |
| Referências | 21 | 15 | −6 |
| **Total** | **124** | **~81** | **−43** |

---

## 2. Alavanca 1 (maior impacto): reduzir o portfólio de políticas

### 2.1 Critério — usar a própria revisão de literatura já feita

O documento `docs/references/estado_da_arte_politicas.md` (já produzido nesta
dissertação) classifica cada política por aderência a veículo relevante de PO/GO.
Quatro políticas foram explicitamente marcadas como **"sem SOTA em veículo
relevante"** ou **"lacuna real"** — ou seja, a própria revisão sistemática já
constatou que elas não têm ancoragem bibliográfica comparável às demais:

| Política | Classificação na revisão | Nota |
|---|---|---|
| **SA** | ★☆☆☆☆ "sem SOTA em veículo relevante" | mesmo espaço de busca do GA (θ=(s,Q)) |
| **PSO** | ★☆☆☆☆ "sem SOTA em veículo relevante" | idem |
| **DE** | ★☆☆☆☆ "sem SOTA em veículo relevante" | idem |
| **SARSA** | ★☆☆☆☆ "lacuna real: nenhum trabalho em veículo relevante aplica SARSA a inventário" | única política tabular do portfólio |

### 2.2 Recomendação: cortar SA, PSO, DE, SARSA → portfólio de 18 para **14**

Justificativa, política por política:

- **SA, PSO, DE**: otimizam o **mesmo espaço de parametrização** que o GA
  (θ = (ROP, Q)), sob a **mesma função objetivo** (Equação `eq:obj_metaheuristicas`).
  A própria Seção "Redundância entre Políticas" (`proposta.tex`,
  §`sec:redundancia_portfolio`) já define o critério de correlação de Spearman
  para identificar políticas redundantes — SA/PSO/DE são candidatas naturais a essa
  redundância por construção (mesmo problema, algoritmos de busca populacional
  distintos, mas convergindo tipicamente para regiões parecidas do espaço). Manter
  o **GA** (âncora real: Zabraoui et al. 2025, artigo-base) já demonstra que
  meta-heurísticas resolvem o problema de parametrização; SA/PSO/DE viram nota de
  rodapé ("testados preliminarmente, comportamento equivalente ao GA, sem SOTA
  próprio — ver revisão de literatura") em vez de subseções completas.
- **SARSA**: é a única política tabular (discretização grosseira de estado) do
  portfólio, sem ancoragem bibliográfica, e historicamente a de pior desempenho
  entre as políticas de RL nos experimentos já rodados (Bahia: NS médio mais baixo
  entre as políticas de RL puro). Cortar não perde nenhuma família conceitual nova
  — DQN e PPO já cobrem "RL puro" com âncoras reais (Oroojlooyjadid 2022, M&SOM;
  Vanvuchelen 2020, Computers in Industry).

**Portfólio remanescente (14 políticas, 6 famílias intactas)**:
EOQ, (s,S), Jornaleiro — PIL, *capped base-stock*, Jornaleiro orientado a
características — MinMax, *Fixed Interval*, *Vendor-Responsive* — **GA** — DQN, PPO
— GA-DQN, GA-PPO.

Nenhuma família desaparece (clássicas, SOTA clássicas, heurísticas do artigo-base,
meta-heurística, RL, híbrida GA-RL continuam todas representadas); só encolhe a
família de meta-heurísticas (4→1) e sai a única política sem ancoragem alguma
(SARSA).

**Alternativa mais conservadora**, se o autor preferir manter alguma redundância
como controle de robustez: cortar só SA e PSO (manter DE, que costuma ser citado
como mais eficiente que GA em superfícies não-convexas) e SARSA — portfólio final
de 15 em vez de 14. O plano abaixo assume o corte para 14, mas os números de página
mudam pouco entre as duas opções.

### 2.3 Onde esse corte economiza páginas (efeito em cascata)

| Local afetado | Como encolhe |
|---|---|
| `proposta.tex` §Meta-heurísticas | 3 parágrafos (SA/PSO/DE) → 1 nota de rodapé |
| `proposta.tex` §RL | remove parágrafo SARSA |
| `conceitos.tex` §2.2.2/2.2.3 | remove definições formais de SA/PSO/DE/SARSA (ficam só na revisão de literatura, Cap.3) |
| `apendice.tex` (matriz de parâmetros) | remove 4 blocos de tabela (SA, PSO, DE, SARSA) |
| `apendice_b.tex` (representações visuais) | remove ~4 de 11 figuras |
| `resultados.tex` | tabelas/heatmaps de política×perfil perdem 4 colunas/linhas cada (mais legível, não só mais curto) |
| `referencias.bib` | remove citações usadas *só* para SA/PSO/DE/SARSA (Kirkpatrick1983SA, Kennedy1995PSO, Storn1997DE, VanSeijen2009ExpectedSARSA — confirmar se são citadas em mais algum lugar antes de remover) |

**Importante**: os `kpis.parquet`/relatórios (`profile_policy_analysis.py`,
`strategy_cost_comparison.py`) já têm o mecanismo pra filtrar políticas — as
tabelas do Capítulo 5 são `\input` direto dos `.tex` gerados por esses scripts
(`simulation/data/08_reporting/...`), então **depois de decidir o corte, regenerar
os relatórios já ajusta as tabelas do texto automaticamente**, sem edição manual
de tabela.

**Estimativa de economia direta**: ~8–10 páginas (metodologia + apêndices +
tabelas mais enxutas), a maior parte concentrada na Metodologia e no Apêndice A/B.

---

## 3. Alavanca 2: enxugamento de prosa por capítulo (sem cortar conteúdo)

Corte de políticas sozinho não fecha a conta (~35 páginas ainda faltam). O resto
vem de **prosa mais direta**, não de remover achados.

### 4. Metodologia (28 → 17p, ainda faltam ~9p além do corte de políticas)

- **Padrão repetitivo identificado**: cada família de política tem um parágrafo de
  formulação + um parágrafo de "conexão com o AIPE"/justificativa. Consolidar em
  UM parágrafo por família (formulação + justificativa junto), não dois.
- A descrição do ambiente de simulação (`sec:ambiente_simulacao`) e da arquitetura
  GA-RL (`sec:ga_rl_arch`) tendem a reexplicar conceitos já formalizados no
  Capítulo 2 (Referencial Teórico) — cortar a reexplicação, manter só a
  especialização (o que muda entre a formulação genérica do Cap.2 e a implementação
  do Cap.4).
- Revisar se as fórmulas de referência (`eq:qref_metod`, `eq:sref_metod`) precisam
  de tanto texto de acompanhamento — uma equação numerada com 2–3 linhas de
  explicação costuma bastar; parágrafos de 6–8 linhas explicando uma fórmula simples
  são candidatos a corte.

### 5. Resultados (20 → 13p, ainda faltam ~4p além do corte de políticas)

- 11 `\includegraphics`: checar se alguma figura é redundante com uma tabela na
  mesma seção (ex.: heatmap E tabela numérica do mesmo dado) — manter só um dos
  dois formatos por achado.
- Figuras de página inteira (heatmaps política×perfil) encolhem naturalmente com
  14 políticas em vez de 18 (podem passar a caber lado a lado com outra figura, ou
  em 2/3 de página em vez de página cheia).
- Considerar mover 1–2 figuras secundárias (ex.: validação do módulo de previsão,
  se não for central ao argumento do AIPE) para o Apêndice A, mantendo no corpo do
  texto só o que sustenta diretamente a Hipótese 1/2/3.

### 2. Referencial Teórico (14 → 10p)

- Mesma observação do padrão "conexão com o AIPE" repetido — aparece após várias
  subseções (ex.: linha 204 de `conceitos.tex`). Reduzir para 1–2 ocorrências
  estratégicas (ex.: uma vez após a taxonomia ADI×CV², uma vez ao final do
  capítulo), não uma vez por subseção.
- Subseções de métricas de previsão (MASE/sMAPE/RMSSE/MBE/U de Theil) — 5
  parágrafos curtos, um por métrica. Se nem todas as 5 aparecem de fato nos
  resultados do Cap.5, considerar mover as menos usadas para uma nota de rodapé
  ou tabela compacta (métrica, fórmula, quando usar) em vez de parágrafo cada.

### 3. Trabalhos Relacionados (8 → 7p)

- Menor prioridade (já é o capítulo mais enxuto proporcionalmente). Revisar só se
  alguma seção (3.2–3.6) tem parágrafos de transição longos que só repetem o que a
  tabela consolidada já mostra.

### 1. Introdução (9 → 7p)

- As três hipóteses (H1/H2/H3, §1.1.3) e os objetivos específicos (§1.2) têm
  alguma sobreposição de conteúdo com a motivação (§1.1) — checar se dá pra
  cortar 1–2 parágrafos de transição sem perder a lógica encadeada motivação→
  problema→hipóteses→objetivos.

---

## 4. Alavanca 3: Apêndices e Referências

### Apêndice A — matriz de parâmetros (8 → 5p)

- Além do corte direto de 4 políticas (SA/PSO/DE/SARSA), avaliar se a matriz
  precisa de uma linha por parâmetro ou se parâmetros comuns a várias políticas
  (semente=42, réplicas=5, horizonte=38 ciclos) podem virar uma nota única no
  topo da tabela ("todos os parâmetros abaixo compartilham semente=42 e 5
  réplicas, salvo indicação contrária"), eliminando linhas repetidas.

### Apêndice B — representações visuais (7 → 4p)

- Corte direto de ~4 figuras (SA/PSO/DE/SARSA). Para as 14 remanescentes, avaliar
  se políticas da mesma família (ex.: EOQ/(s,S)/Jornaleiro; MinMax/FixedInterval/
  VendorResponsive) podem compartilhar uma figura composta (3 painéis pequenos)
  em vez de 3 figuras de página cheia.

### Referências (21 → 15p, 78 entradas)

- Remover entradas usadas exclusivamente para as políticas cortadas (confirmar
  com `grep` antes de remover — algumas podem ser citadas em mais de um lugar,
  ex. Kirkpatrick1983SA pode aparecer só na definição de SA, mas confirmar).
- 78 referências é um número alto para dissertação de mestrado; revisar se há
  citações de apoio secundário (ex.: um segundo survey sobre o mesmo tema já
  citado por outro survey) que podem ser consolidadas.
- **Não mexer** no estilo bibliográfico (`plainnat`) — é provavelmente exigência
  do template `ic` da instituição, não uma escolha de formatação livre.

---

## 5. O que NÃO cortar (proteger explicitamente)

- O **AIPE e o PSE** como contribuição central — Capítulo 1 (motivação,
  hipóteses, objetivos, contribuições) e a formulação do PSE em Metodologia
  (`eq:policy_selector`) são o núcleo da dissertação; não são candidatos a corte,
  só a prosa mais direta.
- A **validação estatística formal** (Wilcoxon, Friedman, Nemenyi, Cohen's d) —
  é um diferencial metodológico explícito da dissertação; cortar aqui enfraquece
  o argumento, não só o texto.
- Os **Experimentos 1 e 2** (Paraíba e Bahia) como resultados centrais — nenhum
  número relatado neles deve ser removido, só reapresentado de forma mais direta.
- A **reprodutibilidade** (Apêndice A, matriz de parâmetros) — pode encolher em
  formato, não em conteúdo: todo parâmetro precisa continuar rastreável.
- Os **Estudos de Caso 3 e 4** (M5, expansão multiestado, adicionados nesta
  sessão) — são o reenquadramento que sustenta "AIPE validado por múltiplos
  estudos de caso"; não são gordura a cortar.

---

## 6. Sequência de execução recomendada

1. **Confirmar a decisão do portfólio** (14 políticas — ou a variante de 15 da
   Seção 2.2) com o orientador, já que muda o que é reportado como resultado
   final da dissertação, não só o texto.
2. **Regenerar os relatórios** (`profile_policy_analysis.py`,
   `strategy_cost_comparison.py`) filtrando o portfólio reduzido — as tabelas do
   Capítulo 5 atualizam automaticamente via `\input`.
3. **Cortar as menções em prosa** às políticas removidas: `proposta.tex`,
   `conceitos.tex`, `apendice.tex`, `apendice_b.tex`, `conclusoes.tex` (onde
   ainda há "12 políticas"/números antigos, ver nota abaixo).
4. **Enxugar a prosa** capítulo por capítulo, na ordem de maior impacto:
   Metodologia → Resultados → Referencial Teórico → Introdução.
5. **Recompilar e medir** a contagem de páginas a cada etapa (`pdflatex` +
   checar `pdfinfo tcc.pdf | grep Pages`), ajustando a meta por capítulo se
   algum corte render mais ou menos do que o estimado.
6. **Revisão final de referências** — remover só depois dos passos 3–4, quando
   já se sabe exatamente quais citações ficaram órfãs.

**Nota de dependência**: o Capítulo 6 (Considerações Finais) ainda tem números
específicos da rodada de 12 políticas (CTI=R\$628, redução de 6,2% etc. — ver
`AJUSTES_INFRA` item #35, pendente) que precisam ser recalculados sobre o
portfólio final (14 políticas) e o `kpis.parquet` já corrigido (item #33) antes
da escrita final desses parágrafos — esse passo é maior que só "cortar texto":
é atualizar resultado. Recomendo tratá-lo como atividade separada, depois que o
portfólio de 14 políticas estiver decidido e a Bahia for re-rodada com ele.

---

## 7. Estimativa consolidada

| Alavanca | Economia estimada |
|---|---|
| 1. Corte de portfólio (18→14 políticas) | ~8–10 páginas |
| 2. Enxugamento de prosa (Metodologia, Resultados, Ref. Teórico, Introdução) | ~20–24 páginas |
| 3. Apêndices e Referências | ~9 páginas |
| **Total estimado** | **~37–43 páginas** |
| **Resultado esperado** | **124 → ~81–87 páginas** |

A meta de 80 páginas é alcançável combinando as três alavancas; se o corte de
prosa (alavanca 2) render menos que o estimado, a alavanca 1 tem uma variante
mais agressiva disponível (cortar também 1 dos 2 híbridos, GA-PPO, que segundo
`estado_da_arte_politicas.md` não tem nenhum trabalho publicado como âncora —
ver ressalva na Seção 2.2), o que adicionaria mais ~3–4 páginas de economia sem
comprometer o argumento central (GA-DQN sozinho já sustenta a demonstração da
arquitetura híbrida, que é o que o artigo-base valida).
