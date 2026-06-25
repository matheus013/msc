# Apresentação de Qualificação de Mestrado

Apresentação Beamer (LaTeX) para o exame de qualificação do mestrado,
baseada na proposta de dissertação (`docs/master_proposal/`) e nos
artefatos do pipeline de simulação (`simulation/data/08_reporting/`).

## Estrutura

```
main.tex          # documento principal (tema, cores, \input das seções)
sections/         # um arquivo .tex por slide (s01_titulo.tex ... s23_fechamento.tex)
figures/          # figuras PDF copiadas do pipeline de simulação e do Capítulo 1
tables/           # fragmentos LaTeX de tabela copiados do pipeline (referência/rastreabilidade)
assets/           # reservado para logos/recursos visuais adicionais
Makefile          # compilação via latexmk + xelatex
```

## Compilação

Requer XeLaTeX (tema `metropolis` usa a fonte Fira Sans via `fontspec`) e
`latexmk`, disponíveis via MiKTeX/TeX Live.

```bash
make        # compila main.pdf
make clean  # remove artefatos de compilação
```

## Conteúdo e narrativa

23 slides (dentro do alvo de 20-24), ~30 minutos, seguindo a narrativa:
problema → lacuna → hipótese → proposta (AIPE) → método → evidência
(Experimento 1 PB, Experimento 2 BA) → continuidade.

Cada slide contém notas de fala em `\note{...}` (visíveis em modo
apresentador/instrutor, ex.: `pympress`, ou compilando com
`\setbeameroption{show notes}`).

## Origem dos números

Todos os números reportados vêm diretamente de:

- `simulation/data/08_reporting/final_results_validation.md`
- `docs/master_proposal/capitulos/resultados.tex`
- `docs/master_proposal/capitulos/conclusoes.tex`
- `docs/master_proposal/capitulos/plano_continuidade.tex`
- Fragmentos de tabela em `simulation/data/08_reporting/profiles/` e
  `simulation/data/08_reporting/strategy/`

Nenhum valor foi inventado ou extrapolado além do que está documentado
nessas fontes.

## Pendências conhecidas

- O PSE (Policy Selection Engine) ainda não foi treinado/validado em
  escala nacional — tratado explicitamente como trabalho futuro nos
  slides 14, 20 e 22.
- Os perfis "Unstable Trend" (n=18) e "High Vol. Seasonal" (n=11) têm
  n < 20 séries e são tratados como evidência exploratória (slide 19, 21).
