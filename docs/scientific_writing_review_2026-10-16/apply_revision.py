"""Scoped editorial revision. Frozen result fragments and scientific code are untouched."""
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
ARTICLE=ROOT/'docs/supply_chain_analytics_article/manuscript_ptbr.tex'
DISS=ROOT/'docs/master_proposal'


def section(text,start,end,replacement):
    assert text.count(start)==1,start
    a=text.index(start);b=text.index(end,a+len(start))
    return text[:a]+replacement.strip()+'\n\n'+text[b:]


s=ARTICLE.read_text(encoding='utf-8')
s=section(s,r'\begin{abstract}',r'\end{abstract}',r'''
\begin{abstract}
A heterogeneidade da demanda motiva selecionar políticas de reposição por contexto, mas a diversidade de vencedoras não assegura recomendações melhores. Propomos o Adaptive Inventory Policy Engine (AIPE), uma metodologia modular que vincula representação do contexto, evidência operacional, seleção e avaliação. Seu Policy Selection Engine (PSE) associa perfis a famílias de políticas ou estima recomendações por série, com uma regra explícita para substituir ou conservar a referência. Avaliamos sete candidatas em duas aplicações separadas. Na Bahia, 145 séries em duas origens totalizam 4.350 ajustes; a seleção por perfil aumenta o atendimento em 3,89 pontos percentuais e o custo em 1,93\% frente à referência global. No piloto Walmart M5, 30 séries de dez lojas em três origens totalizam 1.350 ajustes. O preditivo por classe apresenta custo 13,9\% maior e atendimento 4,11 pontos inferior. Uma extensão exploratória por consequências previstas, com recuo à referência, reduz custo em 1,56\% e aumenta atendimento em 0,27 ponto; apenas três de 90 decisões mudam, e uma série concentra cerca de 92\% da economia. A contribuição é um procedimento que torna verificáveis a recomendação, sua utilidade incremental e as causas das perdas. Os resultados delimitam as instanciações avaliadas; calibrações específicas e desenvolvimento no piloto exigem confirmação independente.
''')
s=section(s,r'\section{Introdução}',r'\section{Trabalhos relacionados}',r'''
\section{Introdução}
Selecionar uma política de reposição para cada combinação de loja e produto exige conciliar custo e atendimento sob demandas heterogêneas. Vendas frequentes, intervalos sem demanda e picos ocasionais podem favorecer regras diferentes. A vantagem de contextualizar a escolha, porém, depende de reconhecer essas diferenças antes da decisão e convertê-las em resultados operacionais posteriores \cite{Axsater2006InventoryControl}.

A escolha da família de política constitui uma decisão distinta da calibração de seus parâmetros. Uma referência global pode usar a mesma família em toda a rede e ainda ajustar lotes e limiares ao histórico de cada série. A seleção de algoritmos fornece uma formulação para ir além dessa referência: características da instância orientam a escolha de um método de um portfólio \cite{rice1976,Kotthoff2014AlgorithmSelection}. No inventário, essa escolha precisa considerar tanto o custo quanto a viabilidade do atendimento.

O desafio é transformar evidência comparativa em uma recomendação justificável. Identificar a melhor candidata após observar o teste informa o potencial do portfólio, mas não demonstra que um seletor consiga antecipá-la. Além disso, uma recomendação pode falhar porque escolheu mal, porque nenhuma candidata atende ao alvo ou porque o sistema não permite atendê-lo nas condições iniciais. Cada situação exige uma intervenção diferente.

Propomos o \textit{Adaptive Inventory Policy Engine} (AIPE), uma metodologia modular para organizar essa decisão. O procedimento especifica o problema, representa o contexto, constrói uma matriz comparável de consequências, recomenda pelo \textit{Policy Selection Engine} (PSE) e avalia a escolha em dados reservados. O PSE admite associação por perfil ou estimação por série. Uma regra de intervenção explicita quando a candidata substitui a referência. Contratos de entrada, saída e informação permitem substituir métodos e reavaliar suas consequências mantendo o problema de decisão definido.

A contribuição metodológica compreende a articulação dessas etapas, a decisão explícita de intervenção e o diagnóstico das perdas. Os controladores e seletores são instanciações usadas para examinar o procedimento. A avaliação distingue sua execução modular de sua utilidade: a primeira requer interfaces compatíveis; a segunda requer benefício em custo e serviço frente à referência. Os antecedentes de classificação de inventário e seleção de algoritmos delimitam o posicionamento apresentado na seção seguinte.

As aplicações usam vendas de uma rede varejista brasileira e um piloto da base pública Walmart M5. A Bahia oferece heterogeneidade empresarial observada; o M5 permite examinar seletores preditivos em um referencial público. As consequências são simuladas, e as bases mantêm custos, populações e calibrações identificados separadamente. A análise inclui ganhos, deteriorações e concentração dos benefícios. As tabelas reportam matrizes auditadas; a avaliação ampliada em andamento permanece fora das conclusões quantitativas.
''')
s=s.replace('O objeto é escolher uma família para a série loja--produto a partir da informação disponível antes da decisão. As entradas são históricos e contexto, portfólio, ambiente de estoque, objetivo operacional e desenho de avaliação. A saída é uma recomendação acompanhada de referência, parâmetros de aplicação, suporte e identificação da configuração. Ausência de candidata viável deve ser registrada; uma contingência não certifica atendimento do alvo.',
'O AIPE organiza a escolha de uma família de reposição para a série loja--produto a partir da informação disponível antes da decisão. A aplicação começa pela definição de históricos e contexto, portfólio, ambiente de estoque, objetivo operacional e desenho de avaliação. Ao final, produz uma recomendação acompanhada da referência, dos parâmetros de aplicação, do suporte da escolha e da identificação da configuração. Quando nenhuma candidata é viável, o registro conserva essa condição mesmo que uma contingência permita emitir a recomendação.')
s=s.replace('As sete candidatas verificam diferentes implementações do módulo de evidência; os seletores verificam alternativas do PSE. Os experimentos mantêm a mesma representação de contexto e o mesmo simulador: não validam a substituição de todos os módulos nem toda combinação possível. No uso pretendido, evidências anteriores orientam a família a calibrar. No teste experimental, todas as candidatas são ajustadas para avaliar escolhas e oráculos; não se demonstrou economia computacional de implantação. A adaptação refere-se à escolha por contexto e à reavaliação documentada, sem troca online de controlador já validada.',
'A avaliação exercita a substituição de controladores e seletores, mantendo a representação de contexto e o simulador. Todas as candidatas são ajustadas nas unidades de teste para medir as consequências das recomendações e construir os oráculos. Essa despesa experimental difere do uso pretendido, em que evidências anteriores orientariam a família a calibrar; eventual economia computacional de implantação requer avaliação própria. A adaptação aqui estudada é a escolha por contexto e sua reavaliação documentada, com alcance restrito às interfaces exercitadas.')
s=s.replace('Na implementação atual, o contrato de representação é compartilhado pelos seletores: imputação e codificação categórica são ajustadas dentro do fold. O avaliador exige a matriz completa de políticas, sementes e blocos, fixa recomendações antes da pontuação e registra referência e intervenções. A fila separa planejamento, ajuste dos controladores e avaliação dos seletores; resultados compatíveis podem ser reutilizados após auditoria. Essas interfaces exercitadas sustentam a substituição de métodos na seleção e no portfólio, sem constituir uma interface de programação genérica validada para qualquer ambiente.',
'A implementação preserva esses contratos ao ajustar imputação e codificação dentro de cada partição de treinamento e exigir uma matriz completa de políticas, sementes e blocos. Recomendações, referências e intervenções são registradas antes da pontuação do teste. O reaproveitamento de ajustes depende de auditoria da equivalência de dados, configuração e código.')
s=s.replace(r'\section{Desenho experimental}',r'''\section{Desenho experimental}
A avaliação examina três questões: a seleção por perfil melhora o compromisso entre custo e atendimento; a estimação por série acrescenta valor à referência; e as perdas observadas podem ser atribuídas à seleção, ao portfólio ou à viabilidade física? A Bahia investiga a primeira questão; o piloto M5 reúne as comparações preditivas e os diagnósticos. A comparação por consequências reutiliza sua matriz para examinar outra implementação do PSE.''')
s=s.replace('O pipeline atual foi organizado para avaliar as mesmas variantes do PSE separadamente na Bahia e no recorte M5, preservando as matrizes já concluídas.',
'O protocolo ampliado avalia as mesmas variantes do PSE separadamente na Bahia e no recorte M5, com identificação das matrizes e dos procedimentos que geraram cada resultado.')
s=section(s,r'\section{Discussão}',r'\section{Limitações e próximos testes}',r'''
\section{Discussão}
Os resultados distinguem a viabilidade de executar o AIPE do benefício de contextualizar a reposição. Controladores e seletores distintos operam sob os contratos definidos, exercitando a modularidade do procedimento. Seu valor operacional depende de uma segunda verificação: a recomendação precisa melhorar o compromisso entre custo e atendimento frente a uma referência competitiva. A execução de alternativas, por si só, não estabelece esse benefício.

Na Bahia, a seleção por perfil acrescenta atendimento com custo pontualmente maior; na segunda origem, coincide com a escolha global. Esse comportamento mostra que a utilidade da segmentação depende da janela e do critério operacional. A base brasileira fortalece a relevância aplicada da avaliação ao preservar heterogeneidade observada em vendas empresariais, enquanto a simulação e a possível censura por ruptura delimitam a interpretação econômica.

O piloto M5 evidencia a dificuldade de aprender a identidade da melhor família com pouco suporte. O XGBoost perde para a referência em custo e atendimento. A previsão de consequências preserva a magnitude operacional dos erros de escolha, mas sua implementação direta também perde serviço; o controle por classe com outra floresta altera os resultados. Portanto, a comparação não isola todo o efeito do alvo de aprendizagem.

O recuo à global restringe a intervenção e concentra a investigação nas decisões em que o seletor prevê uma vantagem. A variante por consequências conserva a referência em 87/90 casos, melhora pontualmente custo e serviço e concentra cerca de 92\% da economia em uma série. Esse padrão sustenta a investigação de intervenções seletivas, com análise de concentração e deteriorações. Seu desenvolvimento no mesmo piloto impede interpretá-lo como confirmação independente da melhoria.

Os oráculos mostram que corrigir o seletor é apenas uma das intervenções possíveis. Havendo candidata viável, a escolha pode ser aprimorada; sem candidata viável, deve-se examinar o portfólio e sua calibração; diante de inviabilidade física, é necessário rever as condições do sistema. A distinção orienta qual componente do AIPE merece revisão e evita atribuir toda ruptura ao modelo preditivo.

O suporte do contexto também limita a transferência. Na primeira origem M5, 826/885 séries elegíveis são esparsas, 57 têm tendência e duas pertencem ao grupo volatilidade/sazonalidade. Mais séries não asseguram diversidade de perfis. A ampliação precisa preservar a definição do serviço: meta média, atendimento ponderado e garantia por série respondem a objetivos distintos. Alterar um módulo exige manter esse significado e conferir novamente as consequências.
''')
s=section(s,r'\section{Conclusão}',r'\section*{Disponibilidade de dados e código}',r'''
\section{Conclusão}
O AIPE propõe um procedimento modular para escolher políticas de reposição conforme o contexto da série loja--produto. Sua contribuição articula evidência comparável, duas abordagens do PSE, uma decisão explícita de intervenção e diagnóstico das perdas. Os métodos utilizados preenchem essas responsabilidades e permitem avaliar escolhas alternativas sem redefinir todo o problema.

A avaliação delimita onde essas escolhas ajudam. Na Bahia, o perfil acrescenta 3,89 pontos de atendimento com aumento de custo de 1,93\%. No piloto M5, a estimação por classe perde para a global; consequências com recuo reduzem custo em 1,56\% e acrescentam 0,27 ponto de atendimento, com apenas três intervenções e benefício concentrado. Assim, a evidência sustenta a execução e a utilidade diagnóstica do procedimento, enquanto seu benefício operacional permanece condicionado à instanciação.

A confirmação requer calibrações harmonizadas e avaliação da regra de intervenção em dados reservados ao desenvolvimento. O critério decisivo será a capacidade de reconhecer quando uma recomendação contextual compensa e quando preservar a referência, considerando serviço, custo e deteriorações.
''')
# Correct the direction of reservation: held-out data are excluded FROM development.
s=s.replace('em dados reservados ao desenvolvimento','em dados reservados, excluídos do desenvolvimento')
ARTICLE.write_text(s,encoding='utf-8')

p=DISS/'capitulos/introducao.tex';s=p.read_text(encoding='utf-8')
s=s.replace('O AIPE organiza essa escolha em quatro etapas: caracterizar a demanda, comparar políticas em um simulador comum, identificar alternativas viáveis segundo custo e serviço e investigar se essas diferenças orientam recomendações para novas séries. A formulação se aproxima da seleção de algoritmos, na qual características de uma instância ajudam a escolher um método',
'O AIPE organiza essa escolha em cinco etapas: especificar a decisão, representar o contexto, construir evidência comparável, recomendar pelo PSE e avaliar as consequências. A separação permite identificar quais informações sustentam a escolha e qual evidência seria necessária para substituí-la. A formulação se aproxima da seleção de algoritmos, na qual características de uma instância ajudam a escolher um método')
s=s.replace('Antes desses ajustes, permanece a escolha de qual família utilizar','Além de calibrar cada candidata, é necessário decidir qual família utilizar')
s=s.replace('O procedimento precisa registrar esses resultados sem transformar reprodução computacional em prova de utilidade.',
'O procedimento registra tanto melhorias quanto deteriorações, permitindo avaliar separadamente a execução da metodologia e seu benefício operacional.')
s=s.replace('A rodada maior do M5 foi interrompida durante a construção do protocolo; sua continuidade foi organizada no pipeline unificado e ainda não integra os resultados desta versão.',
'A avaliação ampliada do M5 está em andamento sob o protocolo unificado e permanece fora das conclusões quantitativas desta versão.')
s=s.replace('A contribuição principal é o procedimento AIPE, que conecta contexto, evidência operacional e recomendação em cinco etapas explícitas. Ele acomoda duas abordagens do PSE e torna a substituição de uma referência uma decisão avaliável, em vez de assumir que toda contextualização é vantajosa. O diagnóstico distingue erros do seletor de limitações que exigem outras intervenções.',
'A contribuição principal é formular a seleção contextual como um procedimento com compromissos verificáveis. O AIPE encadeia cinco etapas, admite associação por perfil e estimação por série e exige uma referência competitiva para julgar a intervenção. O diagnóstico liga cada perda à decisão que pode corrigi-la: melhorar a escolha, ampliar o portfólio ou rever as condições operacionais. A metodologia torna essas decisões explícitas e passíveis de avaliação.')
s=s.replace('A implementação oferece os meios computacionais para executar e auditar esse procedimento. A contribuição empírica consiste em avaliar suas instanciações em vendas de uma rede varejista brasileira e no piloto público, incluindo resultados negativos e concentração dos benefícios. A aplicação empresarial e o referencial público são complementares; usar vendas reais não equivale a observar os ganhos de uma implantação. Código, matrizes de resultados e oráculos são instrumentos de sustentação da contribuição metodológica, não substitutos dela.',
'A contribuição empírica é delimitar a utilidade dessas escolhas em duas aplicações complementares. As vendas de uma rede varejista brasileira preservam heterogeneidade empresarial observada; o piloto público M5 permite examinar implementações preditivas e suas falhas. Código, matrizes e oráculos sustentam a rastreabilidade da avaliação. Como as consequências são simuladas, o uso de dados reais oferece relevância aplicada, mas não comprova ganho financeiro de implantação.')
p.write_text(s,encoding='utf-8')

p=DISS/'capitulos/proposta.tex';s=p.read_text(encoding='utf-8')
s=s.replace('A contribuição investigada é sua articulação em um procedimento de seleção para reposição que explicita duas abordagens do PSE, a decisão de intervir e o diagnóstico das perdas. Não se reivindica a invenção de seleção contextual, rejeição de recomendações ou controle de estoques. O valor da articulação deve ser examinado por sua coerência, utilidade operacional e transferência entre aplicações; a integração de módulos de software, sozinha, não demonstra contribuição científica.',
'A contribuição investigada é a articulação desses princípios em um procedimento de reposição que vincula duas abordagens do PSE à decisão de intervir e ao diagnóstico das perdas. A associação entre contexto e escolha possui antecedentes; a avaliação do AIPE examina como essa articulação define responsabilidades, permite comparar implementações e orienta a revisão dos componentes. Coerência do procedimento, utilidade operacional e transferência entre aplicações constituem critérios distintos de avaliação.')
s=s.replace('Definem-se unidade, horizonte, dinâmica de estoque e instante da recomendação. A exigência de serviço informa se vale para cada série, perfil ou conjunto.',
'A especificação fixa o significado de uma recomendação e do benefício esperado. Definem-se unidade, horizonte, dinâmica de estoque e instante da escolha. A exigência de serviço informa se vale para cada série, perfil ou conjunto.')
s=s.replace('Preparam-se os históricos e calculam-se características ou perfis anteriores à escolha.',
'A representação disponibiliza ao seletor o contexto que poderá ser observado no uso pretendido. Preparam-se os históricos e calculam-se características ou perfis anteriores à escolha.')
s=s.replace('Cada família recebe parâmetros ajustados com os dados admitidos para calibração.',
'A evidência precisa distinguir diferenças entre políticas de diferenças nas condições de avaliação. Cada família recebe parâmetros ajustados com os dados admitidos para calibração.')
s=s.replace('O \\textit{Policy Selection Engine} (PSE) é o componente de seleção do AIPE. Duas abordagens respondem à mesma decisão:',
'O \\textit{Policy Selection Engine} (PSE) converte contexto e evidência de validação em uma recomendação. Duas abordagens respondem à mesma decisão:')
s=s.replace('Depois de fixar recomendações, comparam-se custo, serviço, deteriorações e intervenções em dados que não determinaram o ajuste.',
'A avaliação mede se a recomendação acrescenta valor à referência. Depois de fixar as escolhas, comparam-se custo, serviço, deteriorações e intervenções em dados que não determinaram o ajuste.')
s=s.replace('A arquitetura não fica dispensada de evidência por receber o nome de metodologia.',
'A avaliação, portanto, precisa vincular cada conclusão ao critério que a sustenta.')
p.write_text(s,encoding='utf-8')

p=DISS/'capitulos/conceitos.tex';s=p.read_text(encoding='utf-8').replace('A descrição operacional dos experimentos fica no Capítulo~\\ref{cap:metod}.','A descrição operacional dos experimentos fica no Capítulo~\\ref{cap:experimental}.');p.write_text(s,encoding='utf-8')

p=DISS/'capitulos/resultados.tex';s=p.read_text(encoding='utf-8')
s=s.replace('A heterogeneidade produz escolhas diferentes, mas sua utilidade depende de período e comparador.',
'A segunda questão de pesquisa encontra respostas diferentes nas duas aplicações: contextualizar a escolha nem sempre melhora o compromisso entre custo e serviço.')
s=s.replace('Isso responde parcialmente à pergunta de aprendizagem: a implementação testada não converteu suas características em vantagem operacional.',
'O contraste localiza a dificuldade na conversão de contexto em decisão útil: o preditivo por classe falha nesse objetivo, enquanto a extensão conservadora oferece uma hipótese de melhoria ainda exploratória.')
s=s.replace('A aplicação torna verificáveis as etapas e os diagnósticos da metodologia. A contribuição empírica é delimitar os resultados de suas implementações. O AIPE não recebe confirmação de vantagem geral; oferece um desenho para identificar quando a seleção agrega valor e qual componente precisa melhorar. A avaliação seguinte deve reservar séries não usadas no desenvolvimento e fixar escolhas antes de examinar seus testes.',
'Quanto à primeira questão de pesquisa, as aplicações mostram a execução das etapas e dos diagnósticos definidos pelo AIPE. Quanto à terceira, distinguem dificuldades de aprendizagem, cobertura do portfólio e viabilidade física. Essa separação é o resultado metodológico da análise: identifica qual decisão deve ser revista quando a recomendação falha. A confirmação de benefício operacional requer séries excluídas do desenvolvimento e escolhas fixadas antes de examinar seus testes.')
p.write_text(s,encoding='utf-8')

p=DISS/'capitulos/conclusoes.tex';s=p.read_text(encoding='utf-8')
s=s.replace('A dissertação deve ser julgada pela formulação para o problema, pela articulação justificada de suas decisões e pela evidência oferecida, não pelo nome da arquitetura.',
'Seu alcance decorre da formulação para o problema, da justificativa das decisões e da evidência disponível para cada implementação.')
s=s.replace('A constatação de uma limitação deve orientar a revisão do componente correspondente.',
'Cada diagnóstico orienta a revisão do componente correspondente.')
s=s.replace('Uma avaliação desfavorável precisa orientar os limites de uso da metodologia, sem ser ocultada por uma nova configuração escolhida retrospectivamente.',
'O fechamento da avaliação deve informar em quais condições recomendar a intervenção, quais deteriorações ela produz e quando conservar a referência.')
p.write_text(s,encoding='utf-8')

p=DISS/'capitulos/correlatos.tex';s=p.read_text(encoding='utf-8')
s=s.replace('Esses antecedentes impedem apresentar a associação genérica entre perfil e política como uma ideia inédita. A revisão dirigida desta etapa reconfirmou seus metadados, sem sustentar uma comparação detalhada de seus algoritmos por leitura integral.',
'Esses antecedentes situam a associação entre perfil e política como uma linha estabelecida. A comparação aqui é conceitual: a revisão dirigida verificou os metadados, mas não reproduziu os métodos nem realizou uma comparação algorítmica sustentada pela leitura integral desses estudos.')
s=s.replace('Esses elementos isolados não distinguem o AIPE.',
'O posicionamento do AIPE depende, portanto, de explicitar a decisão de recomendação e a evidência usada para avaliá-la, além de reunir métodos em uma arquitetura.')
p.write_text(s,encoding='utf-8')

p=DISS/'capitulos/plano_continuidade.tex';s=p.read_text(encoding='utf-8')
s=s.replace('Não há execução nacional ou expansão automática autorizada como requisito de conclusão desta análise.',
'A eficiência computacional deve ser relatada como propriedade da execução, separada da qualidade das recomendações.')
s=s.replace('O material SBPO permanece arquivado. ','')
p.write_text(s,encoding='utf-8')

p=DISS/'tcc.tex';s=p.read_text(encoding='utf-8').replace(r'\dataMesAno{15 de outubro}{2026}',r'\dataMesAno{16 de outubro}{2026}');p.write_text(s,encoding='utf-8')
print('Revised article and dissertation narrative; result fragments unchanged.')
