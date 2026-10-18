# Artigo AIPE para Supply Chain Analytics

Manuscrito de trabalho em português: **22 páginas em formato preprint, seis figuras vetoriais e 17 referências citadas**, sem apêndice. Nenhuma submissão foi realizada.

Título: **AIPE: uma metodologia modular para seleção contextual de políticas de reposição**.

O artigo apresenta o procedimento geral, os contratos dos módulos e as instanciações avaliadas. O PSE é componente do AIPE, com associação por perfil ou estimação por série. Os controladores e seletores exercitam a substituição de métodos nas interfaces testadas; não se presume benefício de qualquer combinação nem validação de todos os módulos em outros ambientes.

- manuscript_ptbr.tex / manuscript_ptbr.pdf: versão atual.
- aipe_bahia_resultados.tex: síntese das 145 séries e 4.350 ajustes Bahia.
- m5_pse_20261015.tex: piloto M5 completo, com 30 séries e 1.350 ajustes.
- aipe_consequencias_resultados.tex: variantes por consequências e recuo na mesma matriz M5.
- figures/: diagrama AIPE e cinco gráficos quantitativos.
- article_sources_ptbr_draft.zip: fontes e figuras efetivamente incluídas.
- reviews/aipe_revision_2026-10-15/: verificações estruturais e numéricas.

A comparação da Bahia oferece serviço adicional com custo pontualmente maior. No piloto M5, o preditivo por classe é inferior à global. Consequências com recuo à global reduzem custo em 1,56% e elevam serviço em 0,27 ponto, mas alteram apenas três decisões e concentram aproximadamente 92% da economia em uma série. As extensões com recuo permanecem exploratórias. Não se afirma vantagem geral, confirmação externa ou equivalência plena de calibração entre bases.

O piloto integra o desenvolvimento. A rodada maior M5 foi interrompida e Brasil permanece cancelado; nenhuma delas é apresentada como matriz completa. O PSE preditivo não foi avaliado na Bahia. Custos absolutos das duas bases não são reunidos.

Compilação e conferência, na raiz:

    & experiment/.venv/Scripts/python.exe experiment/scripts/literature/build_sca_article.py --ptbr
    & experiment/.venv/Scripts/python.exe experiment/scripts/documents/verify_article_package.py
    & experiment/.venv/Scripts/python.exe experiment/scripts/documents/audit_aipe_documents.py

As figuras M5 são geradas por build_aipe_results_figures.py, em experiment/scripts/documents/, sem ajustar políticas. O texto M5 é compartilhado, byte a byte, com a dissertação.

As versões anteriores, inclusive o manuscrito inglês manuscript.tex, estão preservadas. O texto português atual é a base da próxima tradução, não uma tradução já concluída. A revisão imediatamente anterior foi preservada em docs/archive/aipe_methodology_article_2026-10-15/. Versões anteriores estão em docs/archive/aipe_revision_2026-10-15/.

Antes da submissão: confirmação independente do protocolo final, harmonização das calibrações, revisão humana, autoria e declarações, tradução e conferência editorial. Dados e modelos permanecem fora do Git e seguem o backup autorizado.

A revisão de alinhamento ao pipeline de 15/10/2026 identifica a validação interna em cinco grupos na avaliação ampliada, mantendo os resultados anteriores vinculados aos protocolos que os produziram. Fontes e PDFs anteriores estão em docs/archive/pipeline_text_alignment_2026-10-15/.
