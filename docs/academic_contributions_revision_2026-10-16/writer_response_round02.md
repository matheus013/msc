# Resposta do escritor — rodada 2

16/10/2026. Foram lidos os três pareceres independentes, todos com zero achados críticos ou maiores. As três sugestões menores foram incorporadas com alterações em três arquivos.

| Achado | Resposta | Local |
|---|---|---|
| MET-01 | As listas agora integram as duas abordagens do PSE com contratos de informação, referência e escopo da restrição de serviço **explícitos**. Foi removida a formulação que poderia sugerir critérios de serviço integralmente idênticos nas instâncias publicadas. | Introduções do artigo e da dissertação |
| EDC-01 | A abertura da seção de contribuições da dissertação foi reduzida a “A contribuição metodológica do AIPE se apoia em três eixos”. A formulação completa permanece na abertura e no objetivo geral. | `capitulos/introducao.tex` |
| EDC-02 | O abstract passou a usar “without practically important deterioration in service”. O texto mantém relevância contextual e não acrescenta margem, garantia de não inferioridade ou significância estatística. | `abstract_en.tex` |

`audit_writer.py` foi reexecutado. Os diffs e `writer_audit.json` refletem as fontes finais comparadas ao baseline original, que permaneceu intacto. Números científicos, citações, equações e figuras continuam preservados. O nome M5 acrescentado na introdução permanece a única diferença de dígitos quando o nome próprio é contado sem normalização.

Gate das linhas alteradas: zero em travessões, antíteses, floreios, voz passiva EN/PT, hype, palavras desnecessárias, qualificadores, demais categorias mecânicas e termos promocionais PT. Persiste apenas o falso positivo já justificado `utiliz`, em “A avaliação utiliza sete políticas”, português factual de passagem preexistente. Highlights permanecem com cinco linhas abaixo de 85 caracteres.

Fontes congeladas para releitura independente e compilação do coordenador. Nenhum PDF, apresentação, resultado ou código científico foi editado nesta rodada. Não compilei nem executei experimentos.
