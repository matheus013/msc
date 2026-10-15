# Revisão de métodos — artigo em português com base real e M5

**Decisão: versão de trabalho científica defensável; revisão maior antes de submissão.** A redação distingue corretamente seleção contextual, heterogeneidade entre bases e validação do PSE. Os resultados centrais estão de acordo com as matrizes e com a reanálise dual. Não identifiquei fabricação de experimento, ganho de serviço não ocorrido ou atribuição de treinamento PSE ao benchmark M5.

Fontes examinadas: `manuscript_ptbr.tex`, `experiment/scripts/analysis/sca_dual_dataset.py`, `experiment/data/08_reporting/sca_article/dual/dual_results.json`, matrizes de KPIs/cenários BA e M5, configurações e código inventariado em `code_inventory_ptbr.json`. Os achados de código estão documentados integralmente em `code_audit_ptbr.md`. Não alterei o manuscrito, os dados ou controladores nesta revisão.

## Verificação de resultados

| Base e contraste, α=0,70 | Redução de custo por perfil | IC95% condicional | Diferença de fill rate |
|---|---:|---:|---:|
| BA, exclusão de loja | −0,575962% | [−1,442128%; 0,152586%] | +0,779142 p.p. |
| M5, exclusão de loja | +0,681055% | [−1,189760%; 2,401566%] | −0,121618 p.p. |

Conferem custos médios, serviço e frequências de violação do texto: BA global 433,305055 / 0,773816 / 40 violações, perfil 435,800728 / 0,781608 / 39; M5 global 19.500,540531 / 0,852133 / 270, perfil 19.367,731195 / 0,850916 / 274. Cobertura: 145×12 e 2.408×12 sem ausência de política por série. O artigo corretamente evita interpretar custos absolutos como economia comparável entre Brasil e Estados Unidos.

As escolhas também conferem: na BA, VendorResponsive para 130 e FixedInterval para 15; M5, SA para 2.350, VendorResponsive 19, GA–DQN 18, EOQ 11, FixedInterval nove e CappedBaseStock uma. O artigo distingue essa recomendação empírica da coluna de política fixa associada ao perfil no código.

O ganho insample BA de 0,195% é corretamente separado do custo maior após exclusão da loja. Os intervalos são os da nova reanálise dual; diferem ligeiramente dos primeiros relatórios pela ordem de amostragem, sem mudar a estimativa pontual. Não copiar os intervalos antigos em figuras novas.

As sensibilidades 0,80/0,85/0,90 conferem. No M5, em 0,90 e 0,95, global e perfil escolhem PPO em todas as séries; os contrastes e intervalos nulos decorrem de escolhas idênticas. O texto evita interpretar isso como certeza de generalização ou equivalência populacional.

## Separações metodológicas bem realizadas

1. **PSE:** a versão apresenta regras explícitas de seleção, não um classificador supervisionado validado. O pipeline M5 realmente não inclui PSE nem previsão, e o texto o reconhece.
2. **Orçamentos:** GA 50/500 gerações, SA 500/100 iterações e híbrido 200/30 episódios são identificados como valores configurados, sem alegar manifesto histórico completo. Não são custos computacionais iguais.
3. **Temporalidade:** prefixos 17/19 são identificados como cortes do código, sem garantia de que preparação, perfil e estado inicial foram restritos ao passado. A exclusão de loja é descrita como transferência condicional da escolha, não avaliação temporal independente.
4. **Repetições:** cinco avaliações nominais não são consideradas cinco treinamentos independentes. O código reutiliza o controlador numa demanda determinística; dispersão de custo próxima de zero confirma essa leitura.
5. **Serviço:** o texto define fill rate em unidades e diferencia média por série de demanda ponderada. Expõe violações locais e fallbacks, sem chamar uma escolha inviável de admissível.
6. **M5:** cold start de 91,90%, dez clusters e dependência por produto são tratados como limitações relevantes. Não foi atribuída causalmente toda a baixa qualidade de atendimento a essa condição.
7. **Algoritmos:** o texto não explica degeneração com hipóteses causais não testadas. EOQ legado, heurísticas aproximadas e diferenças de versão recebem ressalvas.
8. **Escopo:** o caso real permanece central. A base pública acrescenta uma comparação externa de artefatos, sem ser apresentada como reprodução exata de Zabraoui nem substituição de evidência regional.

## Correção de relato solicitada

Na BA, α=0,95 apresenta redução de custo **17,2460%**, IC95% [12,0906%; 22,9476%], acompanhada por redução de serviço **0,6404 ponto percentual**, IC95% [−1,0935; −0,2573] p.p., e 11 escolhas de fallback por perfil. O texto inspecionado menciona os fallbacks, mas não o maior contraste de custo. Solicitei incluir os três aspectos juntos na seção de sensibilidade, para evitar impressão de escolha seletiva de resultados. Esse achado não altera a conclusão no limiar principal e não prova viabilidade em todos os grupos.

## Detalhes ainda necessários para uma versão submetível

- Oferecer suplemento com fórmulas das heurísticas, limiares/precedência dos perfis, convenções de estatísticas e desempates; o artigo atual resume os critérios, mas não torna todos os parâmetros de classificação reproduzíveis por si. Se a contribuição é reanálise de matriz, publicar os scripts/tabelas permitidos e manifesto reduz essa lacuna, mesmo sem disponibilizar transações.
- Especificar que CV 1,5 do loader é filtro por produto agrupando lojas/ciclos, e que a winsorização IQR foi calculada no horizonte completo. “Filtro de irregularidade” genérico pode levar o leitor a supor um critério por série exclusivamente no treino. Informar também nove dias no último ciclo parcial M5.
- Declarar expressamente as durações efetivas compatíveis com avaliação (21 ciclos BA, 74 M5), distinguindo-as dos 38/93 disponíveis e mantendo a ressalva de configuração sem snapshot histórico certificado. Não dividir custos de forma retrospectiva e depois chamar a razão de preço econômico real.
- Verificar rastreabilidade dos cenários reais, sobretudo a deduplicação que mantém a primeira linha antes de somar por ciclo. Há risco de descartar transações complementares; a auditoria de dados deve definir se afeta a matriz usada. A atual redação de limitações sobre demanda censurada não cobre integralmente esse problema de preparo.
- Resolver estoque inicial, normalização e filtros com o passado em novas execuções antes de alegar valor preditivo. A análise atual é condicional e seus resultados negativos não requerem atribuição a uma causa isolada.
- Para evidência de inferência mais forte, avaliar múltiplas origens e produtos/lojas externos. Bootstrap com dez lojas, treino de dobras sobreposto e dependência por produto não justificam uma interpretação de p-valores convencionais ou representatividade de todo o M5.
- Conferir diferenças entre PB e Kedro: o script PB encapsula agentes isolados com parâmetros GA; o worker Kedro executa pré-população e refinamento próprios. O artigo atual não usa a alegação causal de warm-start PB e deve continuar assim.
- Autorizações dos dados, autoria, CRediT, financiamento, conflitos de interesse, revisão humana das fontes e idioma final permanecem pendências reais antes de envio.

## Julgamento das conclusões

**Sustentadas condicionalmente:** a política global viável difere entre coortes; perfis não mostram vantagem clara de custo no limiar principal; a restrição média não garante serviço individual; escolhas mudam com o limiar; comparações contra EOQ legado isolado podem confundir efeito de perfil e ponto de serviço.

**Não sustentadas:** seleção contextual inútil em geral; SA superior em qualquer M5; aprendizado híbrido inadequado em varejo; economia monetária realizada; PSE treinado e validado em duas bases; cinco sementes independentes por célula; ausência de vazamento temporal; reprodução exata da publicação de referência.

A conclusão da versão portuguesa inspecionada respeita essa fronteira. A auditoria de código completa amplia a compreensão dos experimentos, mas não equivale a regeneração dos controladores históricos. Os resultados devem permanecer associados às matrizes identificadas.

## Adendo da revisão consolidada

Reinspecionei a fonte portuguesa após as correções finais, sem novos experimentos ou alterações aos controladores. A solicitação central de relato está resolvida: a seção de sensibilidade apresenta a redução BA de 17,246% em α=0,95, intervalo [12,091%; 22,948%], queda de 0,640 ponto percentual de serviço e 11 fallbacks em conjunto. A redação não interpreta essa economia como atendimento garantido em todos os grupos.

Também estão resolvidas ou explicitamente delimitadas as seguintes solicitações:

- CV 1,5 é identificado como filtro por produto sobre lojas/ciclos, não por série individual.
- A preparação/suavização usa estatísticas do histórico completo, e essa informação futura é reconhecida; o texto não oferece garantia de independência temporal.
- O campo de duração disponível é distinguido da avaliação: cortes configurados deixam 21 ciclos BA e 74 M5, com necessidade de confirmação por manifesto histórico completo.
- A janela M5 inclui bloco final parcial. A duração de nove dias está documentada na auditoria de código; o manuscrito o identifica como parcial, sem comparar seus custos como uma unidade temporal/econômica idêntica às demais.
- O risco de deduplicar transações complementares da base real aparece nas limitações. Não há quantificação retrospectiva ou alegação de que o preparo já esteja cientificamente saneado.
- Regras e precedência dos perfis, variância incluindo zeros, burstiness, tendência, entropia, autocorrelação e classe residual estão explicitadas. Empates e perfil desconhecido usam regras estáveis/fallback global. Isso resolve a lacuna de especificação do seletor e da classificação, sem certificar implementação histórica de todos os controladores.
- A tabela do portfólio apresenta as doze políticas em ambas as coortes. Custos são normalizados pela melhor global viável a 0,70 dentro de cada base (VendorResponsive/SA). As razões e serviços estão consistentes com as médias auditadas; custo baixo dos inviáveis não é rotulado como superioridade sob a restrição.

O agente principal informa que a dissertação consolidada tem 66 páginas, frente a 118 anteriormente, e que a compilação não apresenta problemas de referências cruzadas. Esta revisão científica confirma as correções de fonte examinadas; não substitui a inspeção visual de cada página do PDF.

**Conclusão final da revisão:** não restam inconsistências numéricas centrais identificadas nem a omissão de sensibilidade solicitada. O texto é uma versão de trabalho confiável no alcance declarado. Mantêm-se limitações substantivas de matrizes históricas, falta de snapshot completo dos controladores, preparação retrospectiva, cold start e dez clusters M5, dependência por produto, ausência de avaliação temporal integral e PSE não validado. Essas limitações não foram resolvidas por edição textual e devem permanecer visíveis na versão destinada a revisão humana e posterior submissão.
