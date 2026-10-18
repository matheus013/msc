# Dissertação de mestrado — AIPE

Versão de trabalho de 15/10/2026: **62 páginas**, ante 118 na base original; redução de 47,5%. A revisão metodológica amplia a versão anterior de 52 páginas para explicitar a proposta e integrar a comparação reduzida. Fontes e PDF anteriores foram preservados.

Título: **AIPE: Uma Metodologia para Seleção Contextual de Políticas de Reposição**.

O objeto principal é a metodologia AIPE, definida por entradas, cinco etapas, critérios e saídas. O PSE é seu componente de seleção: por perfil informado ou por estimação preditiva. Algoritmos e simuladores são meios de implementação; Bahia e M5 são aplicações experimentais.

O capítulo 4 define o procedimento geral. O capítulo 5 descreve sua instanciação experimental. Objetivos, resumo, literatura e conclusões seguem essa distinção. A comparação reduzida por consequências com recuo é exploratória: muda três de 90 decisões, com benefício concentrado, sem demonstrar superioridade generalizável. O preditivo não foi avaliado na Bahia.

## Compilação

Na raiz:

    & experiment/.venv/Scripts/python.exe experiment/scripts/documents/build_dissertation.py --sync-pdf

O comando usa pdflatex e bibtex, gera build/tcc.pdf e só atualiza tcc.pdf sem referências indefinidas ou caixas excedentes. A medição em revisao_concisao/comparison.json considera apenas fontes ativas e distingue contagem aproximada de palavras de contagem efetiva de páginas.

## Organização

- tcc.tex: documento raiz; ic.cls: formato institucional, identificado como versão de trabalho nos elementos pré-textuais.
- capitulos/: introdução, fundamentos, literatura, metodologia, resultados, conclusões e continuidade.
- instanciacao_aipe.tex: capítulo de desenho experimental, separado da metodologia geral.
- consequencias_20261015.tex: comparação reduzida entre implementações do seletor.
- bahia_20261015.tex: resultados completos Bahia.
- m5_pse_20261015.tex: resultados do piloto, idênticos ao trecho correspondente no artigo.
- figures/: diagrama AIPE e figuras quantitativas utilizadas.
- referencias.bib: bibliografia.

Rastreabilidade e parâmetros estão integrados à metodologia. Os arquivos de apêndices e figuras anteriores permanecem preservados, sem inclusão no PDF atual. Nenhuma fonte de conhecimento em PDF foi excluída.

A Bahia contém 145 séries, duas origens e 4.350 ajustes. O piloto M5 contém 30 séries, dez lojas, três origens e 1.350 ajustes. A rodada maior foi interrompida para construir o protocolo final. Brasil permanece cancelado. O recuo ao perfil é desenvolvimento no piloto, não confirmação independente.

A capa informa a data da versão; não estabelece data de defesa. Banca, elementos institucionais, declarações e revisão dos orientadores permanecem sob responsabilidade dos autores. Não se declara versão pronta para depósito.

A versão imediatamente anterior está em docs/archive/aipe_methodology_dissertation_2026-10-15/. A anterior a ela está em docs/archive/aipe_revision_2026-10-15/. Veja também docs/PLANO_DE_EVIDENCIAS.md.

A revisão de alinhamento ao pipeline de 15/10/2026 identifica a validação interna em cinco grupos na avaliação ampliada, mantendo os resultados anteriores vinculados aos protocolos que os produziram. Fontes e PDFs anteriores estão em docs/archive/pipeline_text_alignment_2026-10-15/.
