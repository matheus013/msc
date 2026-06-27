# Apresentação de Qualificação de Mestrado

Apresentação Beamer (LaTeX) para o exame de qualificação do mestrado,
baseada na proposta de dissertação (`docs/master_proposal/`) e nos
artefatos do pipeline de simulação (`simulation/data/08_reporting/`).

## Estrutura

```
main.tex          # documento principal (tema, cores, \section{} por capítulo, \input das seções)
sections/
  00_titulo/                    # capa
  01_problema/                  # Capítulo: Problema
  02_introducao/                # Capítulo: Introdução
  03_trabalhos_relacionados/    # Capítulo: Principais Trabalhos Relacionados
  04_proposta/                  # Capítulo: Proposta (AIPE)
  05_metodologia/               # Capítulo: Metodologia
  06_resultados/                # Capítulo: Resultados
figures/          # figuras PDF copiadas do pipeline de simulação e do Capítulo 1
tables/           # fragmentos LaTeX de tabela copiados do pipeline (referência/rastreabilidade)
assets/           # reservado para logos/recursos visuais adicionais
Makefile          # compilação via latexmk + xelatex
```

Cada capítulo é aberto por um slide divisor automático (tema `metropolis`,
`sectionpage=progressbar`), gerado a partir do `\section{...}` correspondente
em `main.tex`. Um slide de Sumário (`\tableofcontents`) logo após a capa
lista os seis capítulos.

## Regra de imagens

Sempre que um slide exibe uma figura (PDF gerado pelo pipeline), o slide
contém **apenas a figura e uma legenda curta abaixo** — sem colunas de
texto, tabelas ou blocos ao lado. Qualquer leitura/interpretação da figura
fica nas notas de fala (`\note{...}`) ou em um slide de texto adjacente
(antes ou depois), nunca dividindo espaço com a imagem no mesmo slide.

## Compilação

Requer XeLaTeX (tema `metropolis` usa a fonte Fira Sans via `fontspec`) e
`latexmk`, disponíveis via MiKTeX/TeX Live.

```bash
make        # compila main.pdf
make clean  # remove artefatos de compilação
```

## Conteúdo e narrativa

36 frames físicos (30 slides numerados + 6 divisores de capítulo), ~28-29
minutos, organizados em 6 capítulos:

1. **Problema** — problema em uma frase, heterogeneidade da demanda,
   consequências de uma política única
2. **Introdução** — pergunta de pesquisa, objetivo da pesquisa, hipótese
   central, contribuição
3. **Principais Trabalhos Relacionados** — da literatura à lacuna, três
   lacunas convergentes, posicionamento da dissertação
4. **Proposta** — visão geral do AIPE, entradas do AIPE, formulação
   matemática do AIPE (variável de decisão, restrições, rótulo $y_i$)
5. **Metodologia** — síntese do pipeline, regimes de demanda na taxonomia
   de Syntetos-Boylan, PODs, ambiente de simulação, portfólio de
   políticas, do rótulo ao PSE
6. **Resultados** — Experimento 1 (PB), Experimento 2 (BA), distribuição
   ADI×CV², fronteira de Pareto, política única vs. seleção por perfil,
   dominância por perfil, limitações, plano de continuidade, fechamento

A narrativa segue: problema → pergunta → objetivo → lacuna → proposta
(com formulação matemática explícita) → metodologia → resultados →
limites. A formulação matemática do AIPE substitui o antigo slide "Do
benchmark ao rótulo" (a mesma equação não é repetida em dois slides
principais); o slide "Do rótulo ao PSE" foi mantido e ajustado para usar
a mesma notação ($\mathbf{x}_i$, $y_i$, $\hat{\pi}_i$).

O slide "Regimes de demanda na taxonomia de Syntetos-Boylan" foi movido
da Introdução para a Metodologia, imediatamente antes de "Perfis
Operacionais de Demanda (POD)", para ficar mais próximo dos gráficos
ADI×CV² que aplica.

A unidade de análise (série = par loja-produto) é explicitada na
Metodologia (slide "Síntese do pipeline metodológico") e no slide
"Perfis Operacionais de Demanda (POD)", que distingue série (unidade)
de POD (perfil operacional atribuído à série).

Quatro slides foram movidos para a seção de backup, comentada em
`main.tex` (disponíveis para perguntas da banca, sem ocupar o corpo
principal nem a contagem de slides):

- `b07_panorama_literatura.tex` — tabela densa de famílias de método;
- `b08_leitura_heterogeneidade.tex` — leitura detalhada da Fronteira de
  Pareto (mensagem-chave já incorporada ao slide principal);
- `b09_heatmap_cti_pod_fig.tex` — heatmap CTI/BE/NS por política e perfil;
- `b10_cti_ajustado_fig.tex` e `b11_cti_ajustado_texto.tex` — análise
  complementar de CTI ajustado por estabilidade (sensibilidade a
  $\lambda_{BE}$), preliminar e não substitui o critério principal.

Cada slide contém notas de fala em `\note{...}` (visíveis em modo
apresentador/instrutor, ex.: `pympress`, ou compilando com
`\setbeameroption{show notes}`).

## Origem dos números

Todos os números reportados vêm diretamente de:

- `simulation/data/08_reporting/final_results_validation.md`
- `docs/master_proposal/capitulos/correlatos.tex` (panorama da literatura
  e posicionamento — Capítulo 3 da apresentação)
- `docs/master_proposal/capitulos/resultados.tex`
- `docs/master_proposal/capitulos/conclusoes.tex`
- `docs/master_proposal/capitulos/plano_continuidade.tex`
- Fragmentos de tabela em `simulation/data/08_reporting/profiles/` e
  `simulation/data/08_reporting/strategy/`

Nenhum valor foi inventado ou extrapolado além do que está documentado
nessas fontes.

## Pendências conhecidas

- O PSE (Policy Selection Engine) ainda não foi treinado/validado em
  escala nacional — tratado explicitamente como trabalho futuro no
  capítulo de Metodologia e no plano de continuidade.
- Os perfis "Unstable Trend" (n=18) e "High Vol. Seasonal" (n=11) têm
  n < 20 séries e são tratados como evidência exploratória no capítulo
  de Resultados.
