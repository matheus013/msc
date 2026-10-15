# Artigo AIPE para Supply Chain Analytics

Manuscrito de trabalho em português: **17 páginas, cinco figuras vetoriais e 13 referências citadas**, sem apêndice. Nenhuma submissão foi realizada.

O artigo apresenta o AIPE com duas abordagens do PSE: associação por perfil e previsão supervisionada da política por série. Examina seu valor frente à seleção global, o efeito do perfil explícito e a distância a oráculos retrospectivos.

- manuscript_ptbr.tex / manuscript_ptbr.pdf: versão atual.
- aipe_bahia_resultados.tex: síntese das 145 séries e 4.350 ajustes Bahia.
- m5_pse_20261015.tex: piloto M5 completo, com 30 séries e 1.350 ajustes.
- figures/: diagrama AIPE, duas figuras Bahia e duas figuras M5.
- article_sources_ptbr_draft.zip: fontes e figuras efetivamente incluídas.
- reviews/aipe_revision_2026-10-15/: verificações estruturais e numéricas.

A comparação da Bahia oferece serviço adicional com custo pontualmente maior. No piloto M5, o preditivo é inferior à global; o recuo ao perfil melhora esse preditivo, mas permanece exploratório. Não se afirma vantagem geral, confirmação externa ou equivalência plena de calibração entre bases.

O piloto integra o desenvolvimento. A rodada maior M5 foi interrompida e Brasil permanece cancelado; nenhuma delas é apresentada como matriz completa. O PSE preditivo não foi avaliado na Bahia. Custos absolutos das duas bases não são reunidos.

Compilação e conferência, na raiz:

    & experiment/.venv/Scripts/python.exe experiment/scripts/literature/build_sca_article.py --ptbr
    & experiment/.venv/Scripts/python.exe experiment/scripts/documents/verify_article_package.py
    & experiment/.venv/Scripts/python.exe experiment/scripts/documents/audit_aipe_documents.py

As figuras M5 são geradas por build_aipe_results_figures.py, em experiment/scripts/documents/, sem ajustar políticas. O texto M5 é compartilhado, byte a byte, com a dissertação.

As versões anteriores, inclusive o manuscrito inglês manuscript.tex, estão preservadas. O texto português atual é a base da próxima tradução, não uma tradução já concluída. A revisão anterior foi arquivada em docs/archive/aipe_revision_2026-10-15/.

Antes da submissão: confirmação independente do protocolo final, harmonização das calibrações, revisão humana, autoria e declarações, tradução e conferência editorial. Dados e modelos permanecem fora do Git e seguem o backup autorizado.
