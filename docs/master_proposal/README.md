# Proposta de Dissertação - AIPE

Proposta de dissertação de mestrado de Matheus Inácio Batista Santos, UFAL, Instituto de Computação.

Título:

**Um Framework Adaptativo para Seleção de Políticas de Reposição em Regimes Operacionais Heterogêneos**

## Compilação

Use `latexmk` a partir desta pasta:

```bash
cd docs/master_proposal
latexmk tcc.tex
```

O PDF final é gerado em:

```text
build/tcc.pdf
```

A configuração está em `.latexmkrc`. Os arquivos auxiliares de compilação ficam em `build/`.

## Estrutura

```text
master_proposal/
|-- tcc.tex                   Documento raiz
|-- ic.cls                    Classe UFAL/IC
|-- lista_siglas_termos.tex   Lista de siglas e termos técnicos
|-- referencias.bib           Bibliografia BibTeX
|-- capitulos/
|   |-- introducao.tex
|   |-- conceitos.tex
|   |-- correlatos.tex
|   |-- proposta.tex
|   |-- resultados.tex
|   |-- conclusoes.tex
|   |-- apendice.tex
|   `-- apendice_b.tex
|-- figures/                  Figuras em LaTeX/TikZ
|-- img/                      Figuras em PDF
`-- build/                    PDF e artefatos de compilação
```

## Capítulos

| Arquivo | Conteúdo |
|---|---|
| `capitulos/introducao.tex` | Introdução |
| `capitulos/conceitos.tex` | Referencial Teórico |
| `capitulos/correlatos.tex` | Trabalhos Relacionados |
| `capitulos/proposta.tex` | Metodologia |
| `capitulos/resultados.tex` | Resultados Preliminares |
| `capitulos/conclusoes.tex` | Considerações Finais e Próximas Etapas |
| `capitulos/apendice.tex` | Matriz de Parâmetros Experimentais |
| `capitulos/apendice_b.tex` | Representações Visuais das Políticas Avaliadas |

## Arquivos externos usados no texto

O Capítulo 5 incorpora tabelas e figuras geradas pelo pipeline Kedro em:

```text
../../simulation/data/08_reporting/
```

Esses artefatos incluem mapas, gráficos comparativos, validação estatística, previsão e tabelas de estratégia por perfil.

## Cuidados

- Não editar resultados numéricos diretamente no texto sem verificar os artefatos geradores.
- Manter figuras da introdução em `img/` ou `figures/`, conforme o tipo de inclusão.
- Manter os artefatos de compilação dentro de `build/`.
- Recompilar com `latexmk tcc.tex` antes de enviar nova versão.
