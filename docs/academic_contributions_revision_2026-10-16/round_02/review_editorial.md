# Fechamento editorial independente — rodada 2

**Aprovado para a integração autorizada: 0 críticos, 0 maiores, 0 menores.** EDC-01 e EDC-02 resolvidos. MET-01 não introduziu regressão editorial. Este parecer não certifica originalidade estabelecida, validade experimental integral ou prontidão para publicação.

## Releitura e regressão

A retomada curta na seção de contribuições da dissertação elimina peso local sem apagar a contribuição principal, que permanece na abertura e no objetivo geral. O abstract EN agora usa “without practically important deterioration in service”, equivalente à relevância prática contextual do resumo PT; não acrescenta margem ou alegação de não inferioridade estatística.

As duas listas passam a exigir contratos de informação, referência e escopo da restrição de serviço explícitos. A redação é fluente e não implica que as implementações publicadas tenham restrições integralmente iguais. Uma contribuição central e três suportes permanecem coerentes entre resumo do artigo, resumo PT, abstract EN, introduções e conclusões. Perfil e estimação são as duas abordagens; classe e consequências, variantes da estimação. Os três suportes não se confundem com cinco etapas ou seis dimensões temporais.

Limites de rastreabilidade parcial, caráter exploratório, diagnóstico retrospectivo, aplicações separadas e revisão sequencial pendente permanecem preservados. Nenhuma regressão de escopo ou nova sigla indefinida identificada.

## Gate mecânico final independente

Scan nas linhas alteradas dos nove arquivos contra o baseline original, sem executar scripts de escrita. As expressões foram extraídas por AST do script do gate. Contagens:

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

O único hit M5/M16 é o prefixo `utiliz` em “utiliza”, verbo factual em português. Falso positivo justificado; todas as demais categorias retornaram zero. O scan literal não é detector linguístico completo.

## Snapshot final

Três fontes mudaram desde a rodada 1: manuscrito PTBR do artigo, introdução da dissertação e abstract EN. Demais 17 fontes com hashes idênticos. Nenhuma fonte foi editada por este revisor; compilação e PDF ficam com o coordenador.

| Fonte | SHA-256 |
|---|---|
| docs/supply_chain_analytics_article/manuscript_ptbr.tex | `aeb42a46ba047cb9a92fdc6bf6d3fc8b9ed9e7c10f9b1e8f4192d554f1bde959` |
| docs/supply_chain_analytics_article/highlights_ptbr.txt | `6218175d9bc2bb8179072a66a38a51fd94a31ca4c73acb6787592ce87f08974b` |
| docs/master_proposal/capitulos/introducao.tex | `3293afc79d12a5d65ead9af6108f5a06418ba925db355e36ee546e209b0b6870` |
| docs/master_proposal/capitulos/resumo.tex | `d03ea020e48add0ee4c241afdf3d06f2a7f85bbbc317c275830adeea047f4df2` |
| docs/master_proposal/abstract_en.tex | `da79e8bae784548cc613e69cf96a8cda04dac4a622e1fe224d5b566f22ce92cd` |
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
