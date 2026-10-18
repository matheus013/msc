# Revisão pelo escritor acadêmico — contrato metodológico

16/10/2026. Escopo: alinhar o artigo PTBR e a dissertação ao contrato discutido com o pesquisador. Esta entrega é textual; não compila PDFs, não altera resultados ou a execução científica, nem declara aprovação independente.

## Decisão editorial e fontes

Identidade mantida: AIPE como metodologia modular de apoio ao gerente de logística e estoques na redução de custos evitáveis, com atendimento e rastreabilidade. Objetivo pretendido e utilidade demonstrada permanecem separados. As discussões do protocolo v0.9, a auditoria das entradas e o contrato do relatório gerencial fundamentam o novo texto; documentação comercial não é apresentada como evidência de novidade ou popularidade.

Skills utilizadas: academic e academic-writing.md, research-paper-writing (guias de método, resumo e conclusão), paper-writing (perfil editorial, compressão, regras mecânicas/semânticas e checklists de desenho/avaliação), com academic-codex-bridge. Convenções específicas de sistemas/ML e preferências do autor original foram adaptadas ao PTBR e à revista-alvo; não foram introduzidos apêndices, slogans ou garantias empíricas para satisfazer templates.

## Arquivos alterados

- `docs/supply_chain_analytics_article/manuscript_ptbr.tex`: resumo, abertura, objeto, contrato de informação e configuração temporal, escopo efetivamente testado, interpretação gerencial, limitações e conclusão.
- `docs/supply_chain_analytics_article/highlights_ptbr.txt`: primeira linha sincronizada com apoio ao gerente.
- `docs/master_proposal/capitulos/introducao.tex`: pergunta, objetivo, unidade de decisão e distinção entre contexto e eficiência de gerentes/lojas.
- `docs/master_proposal/capitulos/proposta.tex`: contrato de entradas e saídas, seis dimensões temporais, julgamento contextual e evidência de benefício.
- `docs/master_proposal/instanciacao_aipe.tex`: zeros Bahia, inventário posterior aos testes, dados observados versus cenários, estado sequencial, limites da persistência e relatório parcial.
- `docs/master_proposal/capitulos/resumo.tex` e `abstract_en.tex`: enquadramento PT/EN equivalente, custos de cenário e família fixa por bloco.
- `docs/master_proposal/capitulos/conclusoes.tex`: alcance gerencial, rastreabilidade parcial, julgamento de relevância e validação sequencial futura.
- `docs/master_proposal/capitulos/plano_continuidade.tex`: completar artefatos e avaliar revisão sequencial sem alterar a fila atual.

## Minioutline e esboço reverso

1. **Problema e destinatário:** custos evitáveis e atendimento motivam apoiar o gerente; séries fornecem contextos, sem ranking gerencial.
2. **Contribuição metodológica:** cinco etapas e PSE modular ligam informação, evidência e recomendação.
3. **Contrato:** entradas têm origem e qualidade; entrega pretendida reúne família, parametrização, referência, alternativas e consequências para decisão humana.
4. **Temporalidade:** ação, agregação, revisão, planejamento, janela e atualização são dimensões distintas; reavaliar não obriga a trocar ou retreinar.
5. **Evidência atual:** vendas reais alimentam cenários de estoque; seleção fixa por sete ciclos é o objeto realmente testado. Números publicados mantidos.
6. **Interpretação e continuidade:** relevância humana contextual requer magnitude e incerteza; revisão cíclica, entrega integral e uso por gerentes exigem avaliação adicional.

No artigo, contratos ficam resumidos em uma subseção; a dissertação desenvolve responsabilidades, continuidade do estado e limites da cobertura de dados. A síntese de resultados e as seis figuras existentes foram preservadas. Não há promessa de ganho geral, economia realizada em reais, reserva confirmatória disponível ou implementação sequencial concluída.

## Mapa de afirmações e evidências

| Afirmação | Evidência | Situação |
|---|---|---|
| AIPE destina-se a apoiar o gerente e a decisão final é humana | Contrato aprovado, protocolo v0.9 | Requisito metodológico; eficácia de uso ainda não avaliada |
| Relevância prática não exige economia percentual universal | Decisão humana documentada no protocolo | Critério interpretativo, sem prova automática de benefício |
| Entradas devem distinguir observação, informação, estimação e cenário | Contrato e auditoria das fontes | Requisito especificado; persistência incompleta |
| Perfil fornecido e perfil derivado são admissíveis | Protocolo e interfaces propostas | Somente derivado exercitado nos resultados |
| Seleção atual fixa família em sete ciclos | Protocolo, desenho de origens e auditoria | Escopo empírico atual explicitado |
| Cadência por ciclo comercial corresponde à empresa-fonte | Informação de contexto fornecida pelo pesquisador | Relatada, sem inventar observação sistemática ou estudo com gerente |
| Parametrizar temporalidade exige compatibilizar informações e unidades | Contrato temporal e dinâmica declarada | Regra de especificação; configurações alternativas sem validação |
| Revisão sequencial deve preservar estado e trânsito | Requisito para representar decisões continuadas | Desenvolvimento futuro; fila não alterada |
| Zeros Bahia incluem ausência de agregado | Auditoria 2.697/5.510 na coorte canônica | Estatística de preparo; não prova demanda nula nem refere-se ao piloto M5 |
| Estoque de maio/2025 não é saldo inicial dos testes | Auditoria de data e vínculos | Limitação explicitada, sem integração retrospectiva |
| Custos atuais não estimam economia em reais | Configuração das matrizes e auditoria | Cenário, preservado |
| Relatório gerencial é parcial | Contrato/exportação Bahia e campos ausentes | Não executável como compra; parâmetros/traços pendentes |

## Autorrevisão e verificações

Evidências mecânicas reproduzíveis em `writer_checks.json`, geradas por `check_writer_scope.py`:

- Artigo: resumo com 197 palavras, inferior a 250.
- Oito fontes LaTeX: equações em display, citações, figuras incluídas, inputs e labels idênticos à baseline; balanço de chaves zero.
- Todos os sete fragmentos quantitativos protegidos conferidos são byte a byte iguais à baseline.
- Zero ocorrências nos padrões pesquisados de travessão Unicode, contraste decorativo, preenchimento retórico, adjetivos promocionais em inglês e voz passiva inglesa após ajuste do abstract. As negações factuais que delimitam escopo permanecem.
- Extensão textual cresceu para documentar requisitos antes ausentes: artigo de 35.368 para aproximadamente 38.700 caracteres; dissertação recebe detalhamento sobretudo no capítulo metodológico, sem novo apêndice ou fundamento tutorial. Paginação e equilíbrio visual serão avaliados pelo coordenador após compilação.

Perguntas de autorrevisão: **Contribuição** permanece metodologia e não algoritmo novo? Sim. **Clareza** separa contrato, implementação e resultados? Sim. **Força experimental** aumentou apenas por redação? Não; novos testes continuam pendentes. **Completude** reconhece dados e artefatos ausentes? Sim. **Desenho** distingue revisão de família das ações do controlador? Sim.

## Limites e passagem à revisão independente

Não se alterou inferência quantitativa, bibliografia, equações, matrizes, calibrações ou filas; não houve compilação por este agente. Não inferioridade formal continua dependente de margem e desenho próprios; não significância não foi tratada como preservação comprovada. Resultados conhecidos seguem desenvolvimento, sem rebatismo como pré-registro. A avaliação ampliada e a auditoria canônica não foram substituídas pelas antigas tabelas nem usadas para alterar seus números.

Compilação, inspeção visual, checagem do pacote e três pareceres independentes são responsabilidades da próxima etapa. Esta entrega permanece aberta a correções indicadas pelos revisores.
