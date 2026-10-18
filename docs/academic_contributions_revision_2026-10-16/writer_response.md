# Resposta do escritor — integração das contribuições

Data: 16/10/2026. Revisão focal autorizada pelo autor. Estado: fontes prontas para revisão independente e compilação pelo coordenador; este registro não certifica prontidão científica para submissão.

## Hierarquia e mapa de evidência

| Contribuição | Integração textual | Evidência que a sustenta | Limites preservados |
|---|---|---|---|
| Central: AIPE como metodologia modular e rastreável de apoio ao gerente na escolha de políticas | Artigo: resumo, lista na introdução, conclusão. Dissertação: resumo PT, abstract EN, introdução, proposta e conclusões | Cinco etapas, contratos de informação/saída, comparação com referência e interfaces implementadas | Objetivo de reduzir custos evitáveis sem deterioração relevante, sem garantia de economia. Persistência e entrega gerencial ainda parciais. Portabilidade integral não demonstrada. |
| Suporte: seleção contextual pelo PSE por perfil e por estimação | Artigo: introdução e distinção das variantes da estimação na metodologia. Dissertação: introdução, proposta e conclusões | Comparação global/perfil na Bahia; variantes por classe/consequências e recuo no piloto M5 | Perfil informado é opção metodológica; executado é derivado. Classe/consequências são variantes da estimação. Perfil não é fonte independente quando deriva dos mesmos descritores. |
| Suporte: diagnóstico das limitações | Artigo: lista da introdução e discussão. Dissertação: introdução, desenho e conclusões | Oráculos do portfólio e planejamento com informação perfeita, sob custos/estados/restrições do simulador | Categorias retrospectivas condicionais, sem inferência causal sobre rupturas na empresa. Demanda zero permanece separada. Ótimos não são controladores implantáveis. |
| Suporte: avaliação empírica complementar | Artigo: introdução, discussão e conclusão. Dissertação: introdução, desenho, conclusões e continuidade | Matrizes publicadas Bahia 145 séries e piloto M5 30 séries; dados empresariais observados e referência pública | Bases separadas, custos de cenário, possível censura de vendas, calibrações distintas, horizontes curtos. Execução ampliada não incorporada. Nenhum ganho financeiro real demonstrado. |

Os métodos constituintes seguem reconhecidos como conhecidos. O confronto empírico com integrações próximas da literatura continua pendente. A relevância permanece julgamento humano contextual, sem corte percentual universal. A família fixa por sete ciclos, a ausência de validação sequencial e a validação com gerentes continuam explícitas.

## Arquivos alterados

- `docs/supply_chain_analytics_article/manuscript_ptbr.tex`
- `docs/supply_chain_analytics_article/highlights_ptbr.txt`
- `docs/master_proposal/capitulos/introducao.tex`
- `docs/master_proposal/capitulos/resumo.tex`
- `docs/master_proposal/abstract_en.tex`
- `docs/master_proposal/capitulos/proposta.tex`
- `docs/master_proposal/instanciacao_aipe.tex`
- `docs/master_proposal/capitulos/conclusoes.tex`
- `docs/master_proposal/capitulos/plano_continuidade.tex`

Baseline com caminhos preservados e SHA-256: `docs/archive/contributions_revision_2026-10-16/baseline_manifest.json`. Diffs, hashes pós-escrita e contagens de caracteres: arquivos `.diff` e `writer_audit.json` neste diretório. `revise_contributions.py` documenta a aplicação inicial, mas não deve ser reexecutado: a trava protege o baseline já criado. Pequenas correções de voz passiva posteriores constam nos diffs finais.

## Auditoria executada

`audit_writer.py` aplica as expressões do gate mecânico de paper-writing às linhas adicionadas no diff, incluindo texto mantido em parágrafos modificados, e dois rastreadores de português. Não constitui auditoria nova das passagens intactas.

| Categoria | Achados finais | Tratamento |
|---|---:|---|
| M1 travessão, M2 antítese, M3/M4/M8/M9 floreios | 0 | Sem ocorrências nas linhas modificadas |
| M6, M10, M12–M15, M17–M18 e pares de precisão | 0 | Sem ocorrências |
| M11 voz passiva em inglês | 0 | Sem ocorrências |
| Voz passiva candidata em português | 0 | Três passagens herdadas em parágrafos alterados convertidas em forma ativa |
| M5/M16 palavras inglesas proibidas | 1 | Falso positivo `utiliz` em “A avaliação utiliza sete políticas”, português factual já existente; mantido |
| Adjetivos promocionais PT | 0 | Sem ocorrências |

Números (desconsiderando o dígito integrante do nome M5), citações, equações e ambientes de figura permanecem idênticos nas fontes comparadas. A introdução do artigo cita M5 uma vez adicional, única diferença na contagem bruta de dígitos. Os cinco highlights têm 75, 70, 66, 63 e 67 caracteres, respectivamente. Nenhum fragmento quantitativo, figura, bibliografia ou código científico foi editado. Nenhum experimento foi treinado, parado ou reconfigurado; não compilei PDFs.

Checklists aplicados: introdução, metodologia e avaliação. A hierarquia aparece explicitamente apenas nas introduções, com síntese nos abstracts/conclusões; o método e a discussão fornecem os vínculos sem repetir uma lista completa em cada seção. Não adotei a preferência genérica da skill por contribuição como resultado demonstrado quando isso inflaria a evidência: o autor adotou uma proposta metodológica e seu benefício operacional continua condicionado à instanciação. Nenhuma alegação bibliográfica nova foi introduzida.

## Encaminhamento

Revisores independentes devem conferir: contribuição e posicionamento; método/evidência e limites; clareza, paralelismo PT/EN e não duplicação. Qualquer achado maior desta integração exige correção antes do fechamento. Compilação, inspeção visual e testes do pacote ficam com o coordenador.
