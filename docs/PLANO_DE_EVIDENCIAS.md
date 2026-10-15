# AIPE: perguntas e evidências

O artigo concentra a arquitetura e a avaliação operacional das duas abordagens do PSE. A dissertação mantém a formulação ampla e seus fundamentos. Não se combinam custos de bases ou calibrações diferentes como se fossem um mesmo tratamento.

| Pergunta | Bahia | Piloto M5 |
|---|---|---|
| PSE por perfil melhora a global? | 145 séries; serviço adicional com custo pontualmente maior | 30 séries; sem benefício agregado |
| O PSE preditivo melhora a escolha? | Não avaliado | Pior custo e serviço; recuo exploratório reduz a perda |
| Perfil explícito ajuda o classificador? | Não avaliado | 90/90 previsões idênticas; importância por ganho zero |
| Distância ao melhor disponível? | Portfólio e seleção temporal | Oráculo por caso e 90 ótimos com informação perfeita |
| Onde ocorrem as falhas? | Serviço e violações individuais | Seleção, portfólio, limite físico e demanda zero |
| Qual a incerteza? | Bootstrap condicional de 115 lojas | Bootstrap exploratório de dez lojas |

As matrizes analisadas estão completas: Bahia, 4.350 ajustes; piloto M5, 1.350. Isso não significa conclusão do M5 ampliado, que foi interrompido. Os 90 casos do piloto são 30 séries em três origens. Sementes não geram novas séries.

O piloto é desenvolvimento, pois motivou a regra de recuo. A confirmação exige séries reservadas, ajustes em validação interna por loja e harmonização da calibração da CappedBaseStock entre bases. Objetivos de serviço por série e por grupo precisam ser explicitados.

O artigo possui cinco figuras: conceito AIPE, fronteira Bahia, escolhas por perfil Bahia, comparação M5 e decomposição das falhas M5. A dissertação inclui também distribuição de serviço Bahia. Figuras usam resultados congelados, sem treinar políticas.

Não estão demonstrados: superioridade geral do AIPE, economia financeira real, efeito causal do híbrido com esforços iguais, implantação online, robustez a todos os custos/prazos e desempenho nacional. Esses pontos não são preenchidos por redação. Brasil permanece cancelado.

Auditorias: docs/supply_chain_analytics_article/reviews/aipe_revision_2026-10-15/. Textos e PDFs anteriores: docs/archive/aipe_revision_2026-10-15/.
