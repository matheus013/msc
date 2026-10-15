# Figura conceitual do AIPE — versão 2

Nova figura, sem substituir a referência enviada pelo autor nem as figuras
existentes do manuscrito. A referência foi preservada byte a byte em
`aipe_conceito_referencia_original.png`.

Arquivos finais, nas variantes `ptbr` e `en`:

- `aipe_conceito_v2_IDIOMA.pdf`: figura vetorial para inclusão no artigo;
- `aipe_conceito_v2_IDIOMA.svg`: versão vetorial editável;
- `aipe_conceito_v2_IDIOMA.png`: imagem de alta resolução;
- `aipe_conceito_v2_IDIOMA_inserir.tex`: ambiente de figura com legenda;
- `aipe_conceito_v2_IDIOMA_preview.png`: conferência visual renderizada do PDF.

O fluxo distingue caracterização da demanda, seleção no AIPE e avaliação em
teste reservado. PSE por perfil e PSE preditivo são alternativas do mecanismo
de seleção, não fases sucessivas. O preditivo pode receber o perfil como
entrada adicional. As políticas A/B/C e as séries desenhadas são exemplos
ilustrativos; não estabelecem associações empíricas ou ganhos demonstrados.

A figura foi preparada para ocupar a largura total do texto. O manuscrito não
foi alterado nesta etapa: o trecho de inclusão fica pronto para a versão que
retomará o AIPE após consolidar o protocolo experimental. As legendas explicam
o caráter ilustrativo e a separação entre treino, validação e teste.

Reprodução:

```powershell
& experiment/.venv/Scripts/python.exe experiment/scripts/documents/build_aipe_concept_figure.py
```

A geração exige Matplotlib e `pdflatex`, usando o backend PGF para exportar PDFs
vetoriais com fontes incorporadas. Foram conferidas visualmente as renderizações
dos dois idiomas. Ambos os PDFs têm uma página, nenhum objeto de imagem raster,
texto dentro dos limites da página e fontes Type 1 incorporadas. O gerador e
seu SHA-256 ficam registrados em `aipe_conceito_v2_proveniencia.json`.
