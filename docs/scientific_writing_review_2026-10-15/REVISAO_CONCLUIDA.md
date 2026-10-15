# Revisão do artigo e da dissertação — 15/10/2026

A revisão combina pesquisa de escrita, edição científica e integração autorizada da execução Bahia concluída hoje. Fontes e critérios estão em [PESQUISA_E_CRITERIOS.md](PESQUISA_E_CRITERIOS.md). As cópias anteriores dos dois documentos e respectivas fontes estão em `before/`.

## Mudanças de redação e estrutura

- Artigo: título e resumo alinhados ao desenho em duas etapas; pergunta e contribuição mais diretas; discussão relacionada à literatura; menos repetição de ressalvas e de detalhes de desenvolvimento. Resultados históricos continuam identificados como históricos.
- Dissertação: resumo quantitativo, abstract inglês correspondente, respostas às questões de pesquisa nas conclusões e plano de continuidade atualizado. A execução nacional cancelada deixou de ser apresentada como frente ativa. Foram preservados os fundamentos e os resultados PB, BA e M5.
- As duas versões apresentam o novo estudo Bahia com o mesmo conteúdo científico, em arquivos `bahia_20261015.tex` conferidos contra uma fonte comum. A matriz histórica de doze políticas não foi misturada com a nova de sete.
- Notação decimal em português, espaços entre palavras e números, nomes de políticas, defasagem da autocorrelação e identificação de custos do cenário foram revistos. Parâmetros de diagrama e dimensões LaTeX conservam ponto decimal por exigência da linguagem.
- A afirmação de ruptura 55% menor no piloto foi retirada: a tabela mostra 0,050 para ambas as políticas. A alegação absoluta de DQN sem pedidos foi substituída por frequência arredondada a zero, explicitando a incompatibilidade com seu BE positivo.
- Três referências da dissertação tiveram erros substanciais de metadados corrigidos em fonte primária; DOIs com sublinhado passaram a ser tratados corretamente tanto nos dados quanto na apresentação LaTeX. A lista contém 61 chaves citadas, sem entradas duplicadas ou chaves ausentes. O artigo contém dez chaves citadas, todas presentes.

## Integração da Bahia concluída

O [script de análise](analyze_bahia_completed.py) lê o CSV e o SQLite sem modificar a execução, exige estado concluído, verifica unicidade e cobertura de políticas/sementes/blocos e gera os derivados em `bahia_completed/`. Não houve novos treinamentos. A fonte contém 4.350 ajustes, 8.700 registros de bloco e 290 casos série–origem, correspondentes a 145 séries, 115 lojas e nove produtos.

As sementes são promediadas antes das agregações. O atendimento é definido em 283 casos; os sete sem demanda ficam separados, mantendo seus custos. As regras escolhem na validação das demais lojas e são avaliadas no teste da loja reservada. A nova análise relata:

- Custo global 161,3068 e por perfil 164,4135: aumento de 1,92597%.
- Atendimento médio global 0,736227 e por perfil 0,775097: aumento de 3,887 pontos percentuais.
- Atendimento ponderado 0,575037 e 0,639471; casos abaixo de 0,70: 119 e 100.
- Primeira origem: GA global; seleção por perfil muda parte dos casos para sSCycle/EOQCycle. Segunda origem: ambas usam sSCycle em todos os casos.
- O intervalo exploratório condicional para o aumento de custo é aproximadamente [−0,32%; 4,16%], com reamostragem pareada de lojas. Controladores e seletores não são reajustados nos sorteios.
- GA–DQN melhora custo e atendimento frente ao DQN, mas tem custo agregado maior e atendimento menor que GA. Os orçamentos totais diferem; isso não identifica causalmente um efeito da arquitetura.

Os testes têm um ciclo em comum, repetem as mesmas séries e preservam uma coorte escolhida retrospectivamente. Ajuste no passado não elimina essa seleção retrospectiva. A análise é exploratória, não uma aprovação de poda do portfólio. O M5 corrigido continua em execução e seus resultados parciais não foram incorporados.

## Entregas e verificação

- Artigo: `docs/supply_chain_analytics_article/manuscript_ptbr.pdf`, 22 páginas; fontes e arquivo ZIP atualizados, incluindo a seção Bahia.
- Dissertação: `docs/master_proposal/tcc.pdf`, 69 páginas; redução de 41,5% em relação às 118 páginas originais, mesmo com os novos resultados. A contagem aproximada de fontes, incluindo as novas inclusões, caiu 49,8%; essa medida inclui comandos e não equivale a palavras de prosa.
- Compilação dos dois documentos sem referências/citações indefinidas ou caixas excedendo margem. A classe institucional ainda emite avisos de pacotes legados, que não equivalem a falhas de compilação. A apresentação de tabelas, resultados e resumos foi conferida em renderizações dos PDFs; a rastreabilidade final está em `qa/final_checks.json`.
- Os dados e as rotinas de simulação permaneceram intactos. Apenas os construtores documentais foram ajustados para incluir a nova seção no ZIP e na medição da dissertação.

## Pendências científicas e institucionais

A revisão textual não valida o PSE nem completa o M5 corrigido. A auditoria bibliográfica automatizada é parcial, com limitações de acesso registradas. Autoria, financiamento, declarações, idioma da submissão, ficha catalográfica, banca e documentos de depósito continuam a depender dos responsáveis. O guia da revista retornou HTTP 403 nesta nova consulta; não se declarou reconfirmação integral das normas atuais. Nenhum documento foi submetido ou publicado.
