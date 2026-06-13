# Proposta de Dissertação — AIPE

**"Um Framework Adaptativo para Seleção de Políticas de Reposição em Regimes Operacionais Heterogêneos"**
Matheus Inácio Batista Santos — UFAL, Instituto de Computação

## Compilação

```bash
cd docs/master_proposal
latexmk -pdf -xelatex -outdir=build -interaction=nonstopmode tcc.tex
```

O PDF é gerado em `build/tcc.pdf`. Todos os artefatos intermediários (`.aux`, `.log`, `.bbl`, etc.) ficam em `build/` (gitignored).

Requer compilação com **XeLaTeX** (não pdfLaTeX).

## Estrutura

```
master_proposal/
├── tcc.tex                   ← documento raiz
├── ic.cls                    ← classe UFAL/IC
├── lista_siglas_termos.tex   ← lista de abreviações
├── referencias.bib           ← bibliografia BibTeX
├── capitulos/
│   ├── introducao.tex
│   ├── conceitos.tex
│   ├── correlatos.tex
│   ├── proposta.tex
│   ├── resultados.tex
│   ├── conclusoes.tex
│   └── apendice.tex
├── figures/                  ← figuras TikZ/PDF
└── build/                    ← artefatos (gitignored)
```

## Capítulos

| Arquivo | Capítulo |
|---|---|
| `introducao.tex` | 1 — Introdução |
| `conceitos.tex` | 2 — Referencial Teórico |
| `correlatos.tex` | 3 — Trabalhos Relacionados |
| `proposta.tex` | 4 — Metodologia |
| `resultados.tex` | 5 — Resultados Preliminares |
| `conclusoes.tex` | 6 — Considerações Finais e Próximas Etapas |
| `apendice.tex` | Apêndice — Matriz de Parâmetros Experimentais |
