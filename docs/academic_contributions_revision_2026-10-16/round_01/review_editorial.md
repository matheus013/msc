# Parecer editorial independente — integração das contribuições

**Decisão: aprovada neste escopo, com duas sugestões menores.** Zero achados críticos e zero maiores. A avaliação não confere prontidão para submissão ou originalidade demonstrada.

## Clareza e assimilação

A contribuição principal é reconhecível: metodologia AIPE modular e rastreável para apoiar o gerente na escolha de políticas. Seleção contextual, diagnóstico e avaliação empírica aparecem como seus três suportes. As introduções explicam a hierarquia; os resumos e conclusões sintetizam; o método e os resultados sustentam cada eixo sem reproduzir integralmente a lista em todas as seções.

A modularidade tem sentido operacional por contratos de entrada, saída e informação. O texto distingue proposta de execução e benefício: a troca de controladores e seletores foi exercitada, a rastreabilidade permanece parcial, a avaliação com usuários e a revisão sequencial continuam pendentes. Reduzir custos sem deterioração relevante é objetivo, sem garantia universal ou limiar imposto. Não identifiquei nova afirmação promocional ou promessa de superioridade.

As decomposições são compatíveis: três contribuições de suporte, cinco etapas do procedimento e seis dimensões temporais têm objetos distintos. No PSE, perfil e estimação são as duas abordagens; classe vencedora e consequências previstas são implementações da estimação. “Preditivo” permanece variante lexical claramente ligada a esta última.

Os resumos PT/EN preservam propósito, desenho, números, interpretação e limites. AIPE/PSE recebem expansão em cada resumo e no corpo. GA/DQN/PPO aparecem expandidos antes de uso argumentativo no artigo e na introdução da dissertação. Nenhuma sigla nova indefinida foi introduzida nesta integração. M5 funciona como nome do benchmark; a dissertação explicita sua origem na lista de siglas.

A leitura do artigo, incluindo seus três fragmentos de resultados, conserva o caminho problema → procedimento → comparação → diagnóstico → limites. A dissertação apresenta fundamentação mais ampla, sem converter essa amplitude em quatro contribuições algorítmicas independentes. Os antecedentes continuam reconhecidos e o confronto empírico com integrações próximas permanece pendente.

## Sugestões menores

- **EDC-01, MINOR — introducao.tex:67.** A formulação completa da contribuição e do objetivo operacional retoma a abertura e o objetivo geral na mesma introdução. A repetição é correta; uma retomada breve ajudaria a reduzir peso local. Sugestão: “A contribuição metodológica do AIPE se apoia em três eixos, detalhados a seguir.” Preservar a formulação integral na abertura e no objetivo geral. Não bloqueia a integração.
- **EDC-02, MINOR — abstract_en.tex:3.** “without relevant service deterioration” traduz fielmente o objetivo, mas “without practically important deterioration in service” é mais natural em inglês. Preserva relevância contextual sem afirmar não inferioridade estatística. Não bloqueia a integração.

## Gate mecânico independente

Reexecutei o scan nas linhas adicionadas/substituídas dos nove arquivos, extraindo as expressões por AST, sem executar o script de edição ou auditoria do escritor. As linhas modificadas incluem trechos preexistentes dos parágrafos alterados. O scan não estabelece ausência de todos os fenômenos em português; a leitura semântica complementa a contagem.

| Categoria | Ocorrências |
|---|---:|
| M1_em_dash | 0 |
| M11_passive_EN | 0 |
| M2_antithesis | 0 |
| M3_M4_M8_M9 | 0 |
| M6_openers | 0 |
| M5_M16_words | 1 |
| M10_hype | 0 |
| M17_fancy | 0 |
| M18_empty | 0 |
| M12_wordiness | 0 |
| M13_qualifiers | 0 |
| M14_adverbs | 0 |
| M15_exclamation | 0 |
| precision_pairs | 0 |
| PT_passive_candidates | 0 |
| PT_promotional | 0 |

O único hit de M5/M16 é `utiliz`, em “A avaliação utiliza sete políticas” (resumo.tex:1). Trata-se de verbo factual em português; o prefixo de regex inglesa não indica floreio. Mantido com justificativa. Não há voz passiva candidata nas linhas alteradas conforme as expressões configuradas; não se afirma detector linguístico completo.

As sínteses quantitativas em resumo e conclusão atendem às funções dessas seções; não aplico uma proibição literal de repetir qualquer número entre seções, que prejudicaria a comunicação científica. As negações que delimitam validação, implantação ou informação futura preservam fatos científicos e não são antíteses ornamentais.

## Escopo e rastreabilidade

O JSON acompanha SHA-256 de 20 fontes, incluindo os nove arquivos da integração e fontes de contexto. Fontes não foram editadas; não compilei nem executei experimentos. A inspeção visual do PDF, conferência de referências/pacote e auditoria de integridade científica ficam com o coordenador e o revisor de método. A aprovação editorial se limita à integração autorizada.

### Hashes do snapshot

| Fonte | SHA-256 |
|---|---|
| docs/supply_chain_analytics_article/manuscript_ptbr.tex | `a07435444d111a3bdf877e6c3ebd5bf8532628fccb1a9b14be97e50bf80378c8` |
| docs/supply_chain_analytics_article/highlights_ptbr.txt | `6218175d9bc2bb8179072a66a38a51fd94a31ca4c73acb6787592ce87f08974b` |
| docs/master_proposal/capitulos/introducao.tex | `60300e35844ce4ccb47bc05f1ee8924ca7612341d66a1a267d1063e8d25491f4` |
| docs/master_proposal/capitulos/resumo.tex | `d03ea020e48add0ee4c241afdf3d06f2a7f85bbbc317c275830adeea047f4df2` |
| docs/master_proposal/abstract_en.tex | `66e1024bc6fe9b3cb8f82f06748c93cfcdb0ca08c8e4e74aadb21ab68d40dee5` |
| docs/master_proposal/capitulos/proposta.tex | `6dda94accc531f65b12abaa10e20217bcecf4a3ebf6e393814bb35c33f3f8716` |
| docs/master_proposal/instanciacao_aipe.tex | `8c3f1a3da3c1b3734e5dec80313f8cfd95951673769c841ee74e6b57005fde3f` |
| docs/master_proposal/capitulos/conclusoes.tex | `2dad4485869322a71ccc6d10c192b15b8f4cfe2168d4e0321321f83499c38a9d` |
| docs/master_proposal/capitulos/plano_continuidade.tex | `2c2c50460106972267c0f299a8f2554e14be5a436e028c0d8cb04eea523985f3` |
| docs/supply_chain_analytics_article/aipe_bahia_resultados.tex | `7b9f9fe69b3c119b1dbae99ac2f61066ca3217678c0e5426faa863f3d64677cc` |
| docs/supply_chain_analytics_article/m5_pse_20261015.tex | `1b39ca833eb59bc998ea81a7e55c9953cf1ca1478291d1a506b2d482a0dff9b2` |
| docs/supply_chain_analytics_article/aipe_consequencias_resultados.tex | `e8f06ac6c936ac126b0549afe4f200c243227be01b3edfb5c2746582274e8585` |
| docs/master_proposal/tcc.tex | `7912edd25149d40b3c8ebbcfe089a7453531b594454015cb2b4073d066185768` |
| docs/master_proposal/lista_siglas_termos.tex | `f117bd30bc0cf4952fa61ab048584fc192d74462f2fa14e1b89dd1635f45469a` |
| docs/master_proposal/capitulos/conceitos.tex | `608d0c90a21d92bb2019ff0e83d81a0f1a449eed8561cbed2975e915275f824a` |
| docs/master_proposal/capitulos/correlatos.tex | `b1086932d172195c33c64c55f989da2ee611ec063ea7cf936eb67a35bf3312b4` |
| docs/master_proposal/capitulos/resultados.tex | `1057ef2cc5616775c4b6e5d4441626c08a2678e00654cf096a40d53048c21084` |
| docs/master_proposal/bahia_20261015.tex | `d8cdb749741414c6d458510d7f44d1e384593e66d72c9d9efc67fa5bb1e78d3f` |
| docs/master_proposal/m5_pse_20261015.tex | `1b39ca833eb59bc998ea81a7e55c9953cf1ca1478291d1a506b2d482a0dff9b2` |
| docs/master_proposal/consequencias_20261015.tex | `a82d92345279203cf548f7d7a2c61b71a97cdb75d4ef54188b0c2d3ea2dfa22e` |
