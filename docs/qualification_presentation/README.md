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

38 frames físicos (32 slides numerados + 6 divisores de capítulo), ~30-31
minutos, organizados em 6 capítulos:

1. **Problema** — heterogeneidade da demanda, consequências operacionais
2. **Introdução** — pergunta de pesquisa, hipótese, contribuição
3. **Principais Trabalhos Relacionados** — da literatura à lacuna, três
   lacunas convergentes, posicionamento da dissertação
4. **Proposta** — visão geral do AIPE, entradas
5. **Metodologia** — caracterização/PODs, ambiente de simulação,
   portfólio de políticas, geração de rótulos e PSE
6. **Resultados** — Experimento 1 (PB), Experimento 2 (BA), regimes de
   demanda na taxonomia de Syntetos-Boylan, evidência de heterogeneidade,
   política única vs. seleção por perfil, CTI ajustado, limitações, plano
   de continuidade, fechamento

O antigo slide "Panorama da literatura" (tabela densa por família de
método) foi movido para a seção de backup (`b07_panorama_literatura.tex`),
disponível para perguntas da banca sem ocupar o corpo principal.

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
