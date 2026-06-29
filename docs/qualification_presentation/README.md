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

37 frames físicos (31 slides numerados + 6 divisores de capítulo), ~30
minutos, organizados em narrativa top-down:

1. **Por que o problema importa?** — problema em uma frase,
   heterogeneidade da demanda, consequências de uma política única,
   pergunta, objetivo, hipótese e contribuição.
2. **Onde está a lacuna?** — da literatura à lacuna, três lacunas
   convergentes e posicionamento da dissertação.
3. **Qual é a proposta?** — visão geral do AIPE e formulação matemática
   como seleção de política por série loja-produto.
4. **Como o AIPE funciona?** — pipeline metodológico e zoom por módulo:
   dados preparados; características e sinais auxiliares; simulação;
   regimes de demanda; PODs; rótulo e PSE; portfólio de políticas.
5. **O que foi avaliado?** — Experimento 1 (PB), Experimento 2 (BA) e
   distribuição ADI×CV².
6. **O que os resultados mostram?** — fronteira de Pareto, política única
   vs. seleção por perfil e política viável de menor CTI por perfil.
7. **O que falta?** — limitações, plano de continuidade e fechamento.

A narrativa segue: problema real → decisão difícil → proposta AIPE →
formulação → pipeline → módulos → resultados → próximos passos.

O antigo slide "Entradas do AIPE" saiu do corpo principal e foi desdobrado
em dois módulos da metodologia: "Módulo 1 -- Dados preparados" e "Módulo 2
-- Características e sinais auxiliares". Assim, a previsão aparece apenas
como sinal auxiliar, não como mecanismo de escolha de política.

A simulação agora aparece antes dos PODs. Primeiro o deck explica o ambiente
controlado que avalia todas as políticas sob a mesma demanda, custos,
sementes e KPIs; depois apresenta Syntetos-Boylan como regime macro e os
PODs como refinamento operacional dentro desse regime.

O slide de POD foi reescrito como "PODs: refinamento operacional dentro dos
regimes" e explicita que, no recorte BA, todas as séries são Lumpy, mas são
separadas em perfis operacionais distintos para apoiar a seleção contextual.

O slide de rótulo e PSE foi unificado como "Do benchmark à recomendação":
a simulação gera o rótulo, o rótulo é a política viável de menor CTI sob
NS ≥ 0,70 e o PSE recomenda uma política e um ranking para novas séries.

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
