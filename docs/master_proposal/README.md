# Dissertação de mestrado — AIPE

Versão de trabalho de 15/10/2026: **52 páginas**, ante 62 na revisão imediatamente anterior e 118 na base original. A redução total é de 55,9%. Textos, diagramas e PDFs anteriores foram preservados.

Título: **Um Framework Adaptativo para Seleção de Políticas de Reposição em Regimes Operacionais Heterogêneos**.

A dissertação mantém fundamentos, literatura, formulação, implementação e resultados do AIPE. O PSE é a camada de seleção, com abordagem por perfil e preditiva. Foram incorporados o piloto M5 concluído, seus diagnósticos, a extensão exploratória com recuo e os oráculos. O modelo preditivo não foi avaliado na Bahia; a conclusão não o apresenta como validado em ambas as bases.

## Compilação

Na raiz:

    & experiment/.venv/Scripts/python.exe experiment/scripts/documents/build_dissertation.py --sync-pdf

O comando usa pdflatex e bibtex, gera build/tcc.pdf e só atualiza tcc.pdf sem referências indefinidas ou caixas excedentes. A medição em revisao_concisao/comparison.json considera apenas fontes ativas e distingue contagem aproximada de palavras de contagem efetiva de páginas.

## Organização

- tcc.tex: documento raiz; ic.cls: formato institucional, identificado como versão de trabalho nos elementos pré-textuais.
- capitulos/: introdução, fundamentos, literatura, metodologia, resultados, conclusões e continuidade.
- bahia_20261015.tex: resultados completos Bahia.
- m5_pse_20261015.tex: resultados do piloto, idênticos ao trecho correspondente no artigo.
- figures/: diagrama AIPE e figuras quantitativas utilizadas.
- referencias.bib: bibliografia.

Rastreabilidade e parâmetros estão integrados à metodologia. Os arquivos de apêndices e figuras anteriores permanecem preservados, sem inclusão no PDF atual. Nenhuma fonte de conhecimento em PDF foi excluída.

A Bahia contém 145 séries, duas origens e 4.350 ajustes. O piloto M5 contém 30 séries, dez lojas, três origens e 1.350 ajustes. A rodada maior foi interrompida para construir o protocolo final. Brasil permanece cancelado. O recuo ao perfil é desenvolvimento no piloto, não confirmação independente.

A capa informa a data da versão; não estabelece data de defesa. Banca, elementos institucionais, declarações e revisão dos orientadores permanecem sob responsabilidade dos autores. Não se declara versão pronta para depósito.

A versão anterior está em docs/archive/aipe_revision_2026-10-15/. Veja também docs/PLANO_DE_EVIDENCIAS.md.
