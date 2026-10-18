# Esboço reverso dos textos revistos

Primeiras frases por seção; conferidas em conjunto com os contratos e o mapa de evidências.

## supply_chain_analytics_article/manuscript_ptbr.tex

### Introdução
- A escolha da família de política constitui uma decisão distinta da calibração de seus parâmetros.
- O desafio é transformar evidência comparativa em uma recomendação justificável.
- Propomos o \textit{Adaptive Inventory Policy Engine} (AIPE), uma metodologia modular para organizar essa decisão.
- A contribuição metodológica compreende a articulação dessas etapas, a decisão explícita de intervenção e o diagnóstico das perdas.
- As aplicações usam vendas de uma rede varejista brasileira e um piloto da base pública Walmart M5.
### Trabalhos relacionados
- Na literatura da revista, Tadayonrad e Ndiaye \cite{sca_f81ad2b3e023} aproximam previsão e consequências operacionais.
- Zabraoui et al.\ \cite{sca_27e994c54848} comparam algoritmos de inventário e combinações de busca genética e aprendizado por reforço com dados M5.
- Políticas limitadas de estoque-base oferecem outra referência para vendas perdidas \cite{Xin2021CappedBaseStock}.
- Francuz e Bányai \cite{FrancuzBanyai2026Selection} integram escolha de família e parâmetros em otimização multiobjetivo, usando aprendizado para aproximar o simulador.
### AIPE: metodologia modular
- O procedimento tem cinco etapas: (1) especificar unidade, ambiente, objetivo e referência; (2) preparar históricos e representar o contexto; (3) calibrar candidatas e construir evidência de validação; (4) recomendar pelo PSE e aplicar a regra de intervenção; (5) avaliar em dados reservados e diagnosticar perdas.
### Substituição dos métodos e decisão de intervenção
- Seja $\Pi$ o portfólio, $x_i$ a descrição do treino e $g_i$ o perfil.
- A avaliação exercita a substituição de controladores e seletores, mantendo a representação de contexto e o simulador.
- A implementação preserva esses contratos ao ajustar imputação e codificação dentro de cada partição de treinamento e exigir uma matriz completa de políticas, sementes e blocos.
### Custo, serviço e escolha
- Na validação $A$, a regra global minimiza $\bar C_{\pi,A}$ entre candidatas com $\bar F_{\pi,A}\geq\alpha$, com $\alpha=0{,}70$.
- O rótulo supervisionado aplica a restrição à própria série de validação: menor custo viável ou, sem viável, maior serviço e menor custo.
### Características e perfis
- Os perfis usam regras ordenadas e fixas: esparso (ADI $\geq1{,}32$, CV$^2\geq0{,}49$, sequência de zeros $\geq5$); volatilidade/sazonalidade (autocorrelação $\geq0{,}25$ ou burstiness $\geq0{,}30$); tendência instável (tendência absoluta $\geq0{,}05$ ou entropia $\geq2$); baixo volume estável (média $\leq2$ e CV$^2<0{,}49$); e venda frequente (ADI $<1{,}32$).
- Burstiness compara o desvio e a média dos intervalos positivos pela razão de sua diferença e soma, com regularização numérica.
### Desenho experimental
- O M5 é uma fonte pública de vendas associada a uma competição de previsão \cite{Makridakis2022M5Accuracy}.
- A elegibilidade exige três ciclos positivos no prefixo.
### Simulador, portfólio e calibração
- Validação e teste começam em estados comuns às candidatas, produzidos por uma referência de lote econômico de compra (\textit{economic order quantity}, EOQ): seis ciclos passados aquecem o sistema, e sua trajetória na validação determina estoque e pipeline do teste.
- O portfólio contém EOQCycle, sSCycle, CappedBaseStock, GA, otimização proximal de políticas (\textit{proximal policy optimization}, PPO), GA--DQN e DQN.
- Os quatro métodos estocásticos usam sementes 11, 29 e 47; as três referências determinísticas têm um ajuste.
### Avaliação e desenvolvimento do piloto
- No M5, XGBoost tem 200 árvores, profundidade três, taxa 0,05, sem subamostragem e com semente fixa.
- Uma implementação com recuo usa o PSE por perfil se o maior escore de classe for inferior a 0,60 ou a classe prevista tiver menos de três exemplos de treino.
- A comparação reduzida adicional reutiliza a matriz M5 e substitui a implementação do PSE.
- A variante com recuo à global exige serviço previsto $\geq0{,}70$, não inferior ao previsto da referência, e economia acima de uma margem em $\{0;0{,}02;0{,}05;0{,}10;\infty\}$.
- Na Bahia, os testes compartilham um ciclo; no M5, ocupam os ciclos 58--64, 72--78 e 86--92, sem sobreposição.
### Avaliação unificada e alcance dos resultados
- A calibração unificada da CappedBaseStock usa pesos zero para ruptura e amplificação em ambas as bases.
### Oráculos e informação perfeita
- No M5, um programa linear inteiro misto (\textit{mixed-integer linear programming}, MILP) conhece toda a demanda de teste e otimiza pedidos sob os mesmos custos, prazo e condições iniciais.
- A eficiência de custo é $100C^*/C_{\mathrm{PSE}}$, apenas quando referência e recomendação satisfazem o mesmo critério de serviço; ambos os custos zero equivalem a 100\%.
### Discussão
- Na Bahia, a seleção por perfil acrescenta atendimento com custo pontualmente maior; na segunda origem, coincide com a escolha global.
- O piloto M5 evidencia a dificuldade de aprender a identidade da melhor família com pouco suporte.
- O recuo à global restringe a intervenção e concentra a investigação nas decisões em que o seletor prevê uma vantagem.
- Os oráculos mostram que corrigir o seletor é apenas uma das intervenções possíveis.
- O suporte do contexto também limita a transferência.
### Limitações e próximos testes
- Não se trata de experimento causal de implantação.
- As extensões com recuo foram desenvolvidas no piloto; suas métricas não confirmam independentemente a melhoria.
### Conclusão
- A avaliação delimita onde essas escolhas ajudam.
- A confirmação requer calibrações harmonizadas e avaliação da regra de intervenção em dados reservados, excluídos do desenvolvimento.
### \section*{Disponibilidade de dados e código}
### \section*{Autoria, financiamento e interesses}
### \section*{Uso de inteligência artificial na preparação do manuscrito}

## master_proposal/capitulos/introducao.tex

### Introdução{cap:intro}INTRODUÇÃO
- Uma rede varejista precisa decidir quando e quanto repor em cada loja.
- Esta dissertação propõe o \textit{Adaptive Inventory Policy Engine} (AIPE), uma metodologia para selecionar políticas de reposição conforme o contexto de cada série loja--produto.
### Motivação
### Reposição em redes varejistas
- Diferenças de consumo, sazonalidade e condições logísticas dificultam o ajuste manual de estoques em redes com muitas lojas.
- A instabilidade dos pedidos acrescenta outra preocupação.
- Na demanda intermitente, intervalos sem venda alternam com ocorrências positivas; quando estas variam muito de tamanho, o padrão é chamado de \textit{lumpy} \citep{Croston1972ForecastingIntermittentDemand,Syntetos2005Categorization,Boylan2008Intermittent}.
- Essas diferenças motivam comparar regras, mas não demonstram que um seletor por perfil será melhor que uma política única bem calibrada.
### Da calibração à seleção de políticas
- Políticas clássicas, meta-heurísticas e aprendizado por reforço oferecem formas distintas de decidir reposição.
- O AIPE organiza essa escolha em cinco etapas: especificar a decisão, representar o contexto, construir evidência comparável, recomendar pelo PSE e avaliar as consequências.
- A previsão fornece informação auxiliar: sua precisão não determina automaticamente quando pedir nem garante menor custo de inventário.
- A Figura~\ref{fig:intro_problem_selection} sintetiza a proposta: partir de séries heterogêneas, comparar um portfólio e avaliar a seleção contextual.
### Problema e questões de pesquisa
- A unidade de análise é a série de vendas de um produto em uma loja, agregada por ciclo comercial.
- A questão principal é: \emph{como estruturar uma metodologia de seleção contextual de políticas de reposição que produza recomendações verificáveis e permita avaliar quando substituir uma referência?} A investigação se desdobra em três perguntas:
- A primeira pergunta trata da formulação metodológica; a segunda, de sua utilidade operacional; a terceira, de seus limites.
- As hipóteses empíricas são que existe heterogeneidade operacional aproveitável, que o contexto disponível antes da decisão captura parte dela e que uma regra de intervenção pode reduzir escolhas desfavoráveis.
### Objetivos
### \subsection*{Objetivo geral}
- Propor, implementar e avaliar o AIPE como metodologia de seleção contextual de políticas de reposição para séries loja--produto, explicitando o procedimento de recomendação, suas condições de aplicação e seu benefício incremental em custo e serviço frente a referências competitivas.
### \subsection*{Objetivos específicos}
- Os estudos da Paraíba e da qualificação motivaram o desenvolvimento.
### Contribuições e alcance da evidência
- A contribuição principal é formular a seleção contextual como um procedimento com compromissos verificáveis.
- A contribuição empírica é delimitar a utilidade dessas escolhas em duas aplicações complementares.
- GA, DQN, PPO e suas combinações são métodos conhecidos.
- O procedimento permite avaliar separadamente controladores e seletores.
- As três sementes das políticas estocásticas descrevem variação do treinamento, sem representar novas lojas ou demandas.
### Escopo e delimitações
- A pesquisa trata cada série loja-produto como uma decisão de reposição.
- O simulador avalia políticas sobre históricos preparados.
- Os dados proprietários limitam a divulgação pública.
- O Capítulo~\ref{cap:tec} apresenta os fundamentos; o Capítulo~\ref{cap:rel}, a literatura; e o Capítulo~\ref{cap:metod}, a metodologia AIPE.

## master_proposal/capitulos/proposta.tex

### AIPE: Metodologia Proposta{cap:metod}METODOLOGIA AIPE
- O \textit{Adaptive Inventory Policy Engine} (AIPE) é a metodologia proposta nesta dissertação para organizar a seleção contextual de políticas de reposição.
- A metodologia define entradas, etapas, critérios e saídas.
### Objeto, entradas e saídas
- Para a unidade $i$ e o instante $o$, a informação $H_{i,o}$ reúne somente observações disponíveis até a escolha.
- A saída compreende a família recomendada, os parâmetros de aplicação e um registro que identifica contexto, referência, suporte da escolha, restrição de serviço e versão do procedimento.
### Princípios e contribuição investigada
- A contribuição investigada é a articulação desses princípios em um procedimento de reposição que vincula duas abordagens do PSE à decisão de intervir e ao diagnóstico das perdas.
- A aplicação requer históricos com cobertura identificável, candidatas executáveis sob convenções comuns e uma definição de serviço.
### Procedimento metodológico
### Especificar a decisão
- A referência recebe o mesmo acesso ao passado.
### Representar o contexto
- A representação é examinada por seu apoio à decisão.
### Construir evidência comparável
- Preserva-se a matriz de consequências, em vez de guardar somente a vencedora.
### Recomendar pelo PSE
- O preditivo pode aprender a identidade da vencedora ou estimar suas consequências.
- Uma regra de intervenção $a_i$ determina se a candidata substitui a referência: \[  p_i^{\mathrm{AIPE}}=  \begin{cases}\widehat p_i,&a_i=1,\\p_i^{\mathrm{ref}},&a_i=0.\end{cases} \] A regra pode permitir a escolha direta ou exigir benefício previsto e suporte.
### Avaliar e diagnosticar
- O diagnóstico distingue erro de escolha com candidata viável, insuficiência do portfólio apesar de possibilidade física e inviabilidade nas condições do sistema.
### Metodologia e instanciação
- Trocar base, modelo ou limiar não cria uma nova metodologia: essas escolhas instanciam o procedimento.
- Na implementação atual, políticas e seletores seguem interfaces verificadas no fluxo de pesquisa.
### Critérios de avaliação da contribuição
- Essas dimensões não são intercambiáveis.
- No uso pretendido, o AIPE reaproveita evidências de outras unidades para recomendar uma família antes de comparar todas novamente.

## master_proposal/capitulos/resultados.tex

### Resultados e Discussão{cap:results}RESULTADOS E DISCUSSÃO
- Este capítulo avalia as instanciações da metodologia AIPE, distinguindo qualidade dos controladores, utilidade da seleção e limites da recomendação.
- Os números a seguir pertencem às matrizes e protocolos identificados no Capítulo~\ref{cap:experimental}.
### Esforço computacional das políticas
- Tempo de ajuste é uma dimensão adicional da comparação, apresentada na Tabela~\ref{tab:fit_time}.
- O tempo deve ser interpretado junto ao desempenho: maior esforço não comprova melhor recomendação.
### Síntese das respostas e limites
- O classificador reproduz o treino, mas tem pouco suporte por classe e rótulos que mudam entre períodos e sementes.
- O oráculo do portfólio e o planejamento com informação perfeita mostram limites diferentes.
- A concentração dos perfis existe também fora da amostra: na primeira origem M5, 826/885 séries elegíveis são esparsas, 57 têm tendência e duas são do grupo de volatilidade/sazonalidade.
- As evidências são condicionais a coortes retrospectivas, sete ciclos de teste, vendas possivelmente censuradas, custos de cenário e estados iniciais de referência.
- Quanto à primeira questão de pesquisa, as aplicações mostram a execução das etapas e dos diagnósticos definidos pelo AIPE.

## master_proposal/capitulos/conclusoes.tex

### Conclusões{cap:conclusion}CONCLUSÕES
- Esta dissertação propõe o AIPE como metodologia para selecionar políticas de reposição conforme o contexto da demanda.
### Respostas às questões de pesquisa
- A segunda questão trata de sua utilidade.
- A extensão por consequências com recuo apresenta custo médio 1,56\% menor e atendimento 0,27 ponto maior que a global.
- A terceira questão trata dos limites.
### Contribuição e alcance
- O diagnóstico distingue perdas com causas diferentes.
- A implementação e os registros computacionais sustentam a rastreabilidade; os resultados brasileiros e públicos avaliam o procedimento em contextos delimitados.
### Continuidade
