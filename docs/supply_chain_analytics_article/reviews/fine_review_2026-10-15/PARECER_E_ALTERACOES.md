# Revisão fina do artigo — 15/10/2026

**Avaliação interna:** o manuscrito apresenta uma pergunta delimitada e resultados auditáveis, mas ainda não deve ser tratado como pronto para submissão. A revisão foi aplicada às fontes e ao PDF. Não constitui parecer da revista nem previsão de aceitação.

## Contribuição e aderência

A pergunta é o valor incremental de escolher a família de política por perfil, comparado à escolha global. Ambas as estratégias continuam ajustando os parâmetros por série. A distinção foi explicitada: “global” não significa impor um único lote a toda a rede.

Há aderência temática à análise prescritiva de estoques, simulação e consequências operacionais no varejo, contempladas pelo [escopo oficial da revista](https://www.sciencedirect.com/journal/supply-chain-analytics/publish/guide-for-authors). Isso não demonstra novidade suficiente ou aceitação. O artigo foi separado do escopo amplo do AIPE e deixou de condicionar sua contribuição à validação de um seletor supervisionado que não participa desta comparação.

Foram incorporados antecedentes mais próximos: [Mohammaditabar et al., 2012](https://doi.org/10.1016/j.ijpe.2011.03.012), que integram classificação e escolha de políticas, e [Sheikh-Zadeh et al., 2021](https://doi.org/10.1016/j.omega.2020.102276), sobre classificação orientada ao desempenho. A formulação não é apresentada como primeira associação entre perfis e controle. [Xin, 2021](https://doi.org/10.1287/opre.2020.2019) fundamenta a família CappedBaseStock sem transferir suas garantias teóricas à implementação local.

## Conferências e alterações

| Ponto | Achado | Tratamento |
|---|---|---|
| Pergunta, resumo e conclusão | O texto ainda se estendia ao framework e sugeria combinação de bases já concluída | Recorte empírico Bahia explícito; M5 identificado como avaliação complementar em execução |
| Novidade | Faltavam antecedentes que já integram classificação e controle | Literatura próxima adicionada; contribuição delimitada à avaliação temporal das regras especificadas |
| Comparador global | Podia ser confundido com parâmetros iguais em todas as séries | Definido como família comum com calibração individual |
| Atendimento médio | A equação usava denominador de todos os casos | Subconjunto com demanda positiva definido; custo mantém todos os casos |
| Diferença de custo | Sinal da fórmula indicava redução, enquanto a narrativa reportava aumento | Definido aumento relativo, com sinal consistente em todo o texto |
| Controle DQN | Papel de controle não esclarecia sua elegibilidade à seleção | Explicitado que as sete candidatas estão disponíveis ao seletor |
| Simulação | Faltavam equação do custo, vendas perdidas e ordem dos eventos | Incluídos sistema, custo por componente, trânsito e ausência de crédito terminal |
| Políticas | Nomes sem descrição suficiente para identificar as implementações | Tabela de sete candidatas, regras analíticas, estado, orçamento e transformação do híbrido |
| Critério de serviço | Global restringe média geral; perfil restringe média por grupo | Explicitado que a comparação combina diferenciação e alocação da exigência; não isola o efeito informacional dos descritores |
| Calibração CappedBaseStock | Sua grade chama o custo com pesos unitários de ruptura e amplificação de pedidos; GA usa pesos zero nesses termos | Assimetria descrita no método e nas limitações; nenhum controlador foi alterado durante o M5 |
| Incerteza | Intervalo apenas para custo, apesar de o benefício alegado ser atendimento | Intervalo condicional do atendimento calculado e reportado |
| Sensibilidade | O texto destacava apenas alguns limiares | Tabela completa de 0,70 a 0,95, incluindo contingências e o contraste desfavorável ao perfil em 0,80 |
| Metas maiores | Podia parecer que cada meta tinha seu próprio treinamento | Esclarecido: seleção varia sobre candidatas calibradas com referência 0,70, sem retreinamento |
| Figuras | PNG de 300 dpi não é a melhor entrega para desenhos de linhas | PDFs vetoriais gerados via PGF/LaTeX, com fontes Type 1 incorporadas; PNGs mantidos para prévia |
| Disponibilidade | Texto não refletia o código já publicado | Repositório público indicado; restrição de dados e necessidade de versão arquivada descritas sem prometer acesso não autorizado |
| Linguagem | Afirmações podiam sugerir comprovação de custo maior ou causalidade do perfil | Separadas estimativa pontual, intervalo condicional, interpretação e generalização |

## Auditoria independente dos números

`experiment/scripts/documents/audit_article_evidence.py` reconstitui as escolhas filtrando explicitamente as demais lojas, independentemente da implementação otimizada por estatísticas suficientes. Confere os conjuntos de sementes e blocos, denominadores e razões de atendimento e compara cada escolha com os derivados reportados.

- 4.350 ajustes únicos; 8.700 registros de validação/teste.
- 2.900 escolhas conferidas: 290 casos × duas estratégias × cinco limiares. Todas coincidem em política, contingência, custo e atendimento.
- 290 casos série–origem de teste, 283 com demanda positiva e sete sem denominador de atendimento. Não há denominadores nulos na validação.
- Aumento relativo de custo: **1,92597%**; intervalo condicional de 95% **[−0,32199%; 4,16102%]**.
- Diferença de atendimento: **3,88697 pontos percentuais**; intervalo condicional de 95% **[2,78417; 5,07002] pontos percentuais**.
- Bootstrap de 5.000 sorteios das 115 lojas, preservando todas as séries e origens e recalculando denominadores. Controladores e escolhas permanecem fixos. Produtos compartilhados entre lojas continuam uma limitação.

O relatório `numerical_audit.json` contém proveniência por hash. Os intervalos não equivalem à incerteza de novos treinamentos, novos períodos ou novas realizações da demanda. Não foi estimado efeito causal da representação.

## Requisitos editoriais conferidos

O [guia oficial](https://www.sciencedirect.com/journal/supply-chain-analytics/publish/guide-for-authors) foi acessado no navegador em 15/10/2026, após bloqueio da leitura HTTP. Resumo: até 250 palavras; palavras-chave: 1–7 em inglês; destaques opcionais: 3–5 itens, até 85 caracteres; referências numéricas em ordem de citação; figuras separadas e legendadas; fontes editáveis; política de dados Option B. O resumo revisado tem 192 palavras e os quatro destaques têm 70, 67, 67 e 64 caracteres. A versão permanece em português por decisão do projeto; a preparação editorial em inglês ainda é necessária.

## O que permanece antes da submissão

O pacote revisado tem 19 páginas, quatro figuras vetoriais e doze referências citadas. A compilação e a conferência do arquivo de fontes passaram sem referências indefinidas ou caixas excedidas; fontes Type 1 estão incorporadas às figuras. As páginas foram inspecionadas em contato visual, com conferência ampliada de métodos, tabelas e gráficos. Esses controles verificam apresentação e integridade do artefato, sem substituir os pontos científicos abaixo.

1. **Consolidar o M5 para o escopo pretendido pelos autores.** A Bahia responde à pergunta nesse recorte; não é substituto de evidência externa. Completar a matriz, conferir sua cobertura e aplicar as mesmas análises por base, sem concatenar custos absolutos.
2. **Decidir a força da afirmação pretendida.** Comparar duas regras operacionais é sustentado pelo desenho. Afirmar que os descritores, por si, causam melhora exige comparação com restrições equivalentes e ablações. O texto atual adota a afirmação mais estreita.
3. **Se o ranking de famílias for uma contribuição central, harmonizar calibração e esforço.** A assimetria de CappedBaseStock e o esforço extra do híbrido impedem essa leitura causal. A revisão documenta a diferença; não a resolve por redação nem altera resultados já obtidos.
4. **Avaliar robustez econômica e temporal compatível com a conclusão final.** Há um único cenário de custos e duas janelas curtas com um ciclo compartilhado. A sensibilidade do limiar não substitui sensibilidade a custos, prazo ou inicialização. Não foram executados treinamentos adicionais nesta revisão.
5. **Fechar os itens dos autores.** Confirmar autoria e correspondência, endereço institucional, CRediT, financiamento e interesses; revisar humanamente todo o conteúdo, completar a divulgação de assistência por IA e preparar versão em inglês. Não foram inventadas declarações pessoais.
6. **Arquivar uma versão reproduzível.** O GitHub contém código; uma versão fixa do pacote, com identificador persistente e derivados de divulgação autorizada, ainda deve ser preparada. Dados restritos podem ser justificados conforme o guia; não se presume autorização de redistribuição.

Os pontos 2–4 não significam executar todo o plano da dissertação. Servem para ajustar a conclusão à evidência e definir quais extensões são necessárias para o artigo escolhido. O manuscrito não precisa contemplar todos os módulos do trabalho.
