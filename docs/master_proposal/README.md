# Proposta de Dissertação - AIPE

**Versão revisada em 15/10/2026: 62 páginas, redução de 47,5% em relação às 118 originais.** O capítulo de resultados organiza a mesma matriz pelas perguntas de pesquisa, com quatro figuras compartilhadas com o artigo e uma tabela de tempos por política. A dissertação mantém a formulação ampla do AIPE; o artigo tem escopo delimitado.

A Bahia tem 145 séries, duas origens e 4.350 ajustes completos. O M5 permanece em execução, sem conclusão quantitativa parcial. Textos e PDFs anteriores estão preservados nos arquivos de revisão; resultados de configurações diferentes não são combinados. Veja [PLANO_DE_EVIDENCIAS.md](../PLANO_DE_EVIDENCIAS.md).

Proposta de dissertação de mestrado de Matheus Inácio Batista Santos, UFAL, Instituto de Computação.

Título:

**Um Framework Adaptativo para Seleção de Políticas de Reposição em Regimes Operacionais Heterogêneos**

## Compilação

O comando abaixo, executado na raiz do repositório, usa `pdflatex` e `bibtex` diretamente; não exige Perl, ausente neste ambiente:

```bash
& experiment/.venv/Scripts/python.exe experiment/scripts/documents/build_dissertation.py
```

O PDF final é gerado em:

```text
build/tcc.pdf
```

A configuração está em `.latexmkrc`. Os arquivos auxiliares de compilação ficam em `build/`.

Depois de conferir o PDF, `--sync-pdf` atualiza a cópia canônica `tcc.pdf`. O comando não cria um arquivo histórico por si só; preserve a versão anterior antes de usá-lo. Nesta revisão, isso foi feito em `../scientific_writing_review_2026-10-15/before/`. `comparison.json` registra páginas e contagem aproximada das fontes, incluindo comandos, legendas e tabelas; a medição atual inclui o resumo inglês e a seção nova da Bahia.

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

O antigo Apêndice B permanece preservado nas fontes, mas suas imagens didáticas duplicavam o referencial/metodologia e não são incluídas na versão enxuta. Nenhuma figura PDF foi apagada. O apêndice ativo conserva parâmetros, rastreabilidade e a tabela histórica do piloto.

## Arquivos externos usados no texto

O Capítulo 5 incorpora tabelas e figuras geradas pelo pipeline Kedro em:

```text
../../experiment/data/08_reporting/
```

Esses artefatos incluem mapas, gráficos comparativos, validação estatística, previsão e tabelas de estratégia por perfil.

## Cuidados

- Não editar resultados numéricos diretamente no texto sem verificar os artefatos geradores.
- Manter figuras da introdução em `img/` ou `figures/`, conforme o tipo de inclusão.
- Manter os artefatos de compilação dentro de `build/`.
- Recompilar com `latexmk tcc.tex` antes de enviar nova versão.
