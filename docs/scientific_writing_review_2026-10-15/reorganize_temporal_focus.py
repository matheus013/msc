"""One-time structural revision, preserving the preceding manuscripts."""
from pathlib import Path
import shutil
import re

ROOT=Path(__file__).resolve().parents[2]
ARTICLE=ROOT/'docs/supply_chain_analytics_article'
THESIS=ROOT/'docs/master_proposal'
ARCHIVE=Path(__file__).resolve().parent/'before_temporal_reorganization'
if ARCHIVE.exists():raise SystemExit('Archive already exists; do not apply this one-time transformation twice.')
ARCHIVE.mkdir()
for folder in [ARTICLE,THESIS]:
    dest=ARCHIVE/folder.name;dest.mkdir()
    for name in ['manuscript_ptbr.tex','manuscript_ptbr.pdf','tcc.tex','tcc.pdf','bahia_20261015.tex']:
        if (folder/name).exists():shutil.copy2(folder/name,dest/name)
    if folder==THESIS:shutil.copytree(folder/'capitulos',dest/'capitulos')

old=(ARTICLE/'manuscript_ptbr.tex').read_text(encoding='utf-8')
shared=(ARTICLE/'bahia_20261015.tex').read_text(encoding='utf-8')
cut=shared.index('\\subsection{Custo e atendimento no teste}')
protocol=shared[:cut]
protocol=protocol.replace('\\section{Bahia: avaliação temporal com o protocolo corrigido}','\\section{Dados e protocolo temporal da Bahia}')
protocol=protocol.replace('\\label{sec:ba2026}','\\label{sec:ba2026_methods}')
protocol=protocol.replace('Uma nova execução, concluída em 15 de outubro de 2026, avalia as mesmas 145 séries da Bahia, em 115 lojas e nove produtos. Esta etapa acrescenta teste temporal à investigação da seleção de políticas. Seus resultados são apresentados separadamente das matrizes históricas: demanda preparada, portfólio, convenção de entrega e horizonte de avaliação foram alterados. A diferença entre custos antigos e novos não estima o efeito isolado de uma correção.',
    'A avaliação principal utiliza 145 séries loja-produto da Bahia, em 115 lojas e nove produtos, com 38 ciclos comerciais disponíveis. O objetivo é comparar uma política global com a escolha por perfil em períodos posteriores aos usados no ajuste e na seleção. Os identificadores da coorte provêm do estudo histórico; a preparação e o protocolo desta execução são descritos a seguir. Os resultados históricos da Bahia e do M5 têm função complementar e não são concatenados à avaliação temporal.')
protocol=protocol.replace('A variante corrigida mantém','A variante avaliada mantém')
for folder in [ARTICLE,THESIS]:(folder/'protocolo_bahia_20261015.tex').write_text(protocol,encoding='utf-8')

results=shared[cut:]
interval_start=results.index('Uma reamostragem exploratória pareada')
interval_end=results.index('\n\n',interval_start)
interval=results[interval_start:interval_end]
results=results[:interval_start]+results[interval_end+2:]
results=results.replace('\\subsection{Alcance da nova evidência}','\\subsection{Perfis e variação entre origens}')
last=results.index('A nova execução controla o uso do passado')
results=results[:last]
results='\\section{Resultados da avaliação temporal}\n\\label{sec:ba2026}\n\n'+results
for folder in [ARTICLE,THESIS]:(folder/'bahia_20261015.tex').write_text(results,encoding='utf-8')

def section(title,next_title):
    return old[old.index('\\section{'+title+'}'):old.index('\\section{'+next_title+'}')]
front=old[:old.index('\\section{Introdução}')]
front=front.replace('Seleção de políticas de reposição sob restrições de serviço: evidências históricas e avaliação temporal no varejo',
    'Seleção de políticas de reposição por perfil de demanda: avaliação temporal em uma rede varejista')
abstract=r'''A heterogeneidade da demanda motiva selecionar políticas de reposição por perfil, mas essa complexidade só se justifica se melhorar a decisão em períodos posteriores. Este estudo compara seleção por perfil e política global em 145 séries loja-produto de uma rede varejista brasileira, distribuídas em 115 lojas e nove produtos. Avaliamos seis políticas principais e DQN como controle adicional, em duas origens temporais, com blocos separados de treino, validação e teste. Os métodos estocásticos usam três sementes, totalizando 4.350 ajustes. Em cada origem, a escolha minimiza custo entre candidatas que atingem atendimento médio mínimo de 0,70 na validação das demais lojas; o desempenho é medido no teste da loja reservada. A seleção por perfil aumenta o custo médio em 1,93\% e o atendimento em 3,89 pontos percentuais. Os casos abaixo da meta diminuem de 119 para 100 entre 283 casos série--origem com demanda positiva; sete casos têm atendimento indefinido. O intervalo condicional de 95\% para o aumento de custo é de $[-0{,}32;4{,}16]\%$. A diferença concentra-se na primeira origem; na segunda, ambos os seletores escolhem a mesma política. Os resultados mostram serviço adicional com custo maior, sem economia geral da seleção contextual. A coorte retrospectiva e duas janelas curtas parcialmente sobrepostas limitam a generalização. Matrizes históricas da Bahia e de 2.408 séries M5 complementam o contexto, com protocolos distintos da avaliação principal.'''
front=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda m:'\\begin{abstract}\n'+abstract+'\n\\end{abstract}',front,flags=re.S)
intro=section('Introdução','Trabalhos relacionados e posicionamento')
start=intro.index('A pesquisa parte dos experimentos')
intro=intro[:start]+r'''A pesquisa parte dos experimentos da qualificação de mestrado com dados de uma rede brasileira. O caso central deste artigo é a avaliação temporal de 145 séries da Bahia. Os estudos históricos da Bahia e do M5 são preservados no Apêndice~\ref{app:historical}, como antecedentes com protocolos próprios. O M5 oferece uma origem pública adicional, mas sua nova execução ainda não integra a evidência principal.

Investigamos se a seleção por perfil oferece um compromisso entre custo e atendimento mais favorável que uma política global escolhida com a mesma informação passada. A análise distingue o desempenho de cada candidata, o resultado das estratégias de seleção e sua variação entre períodos. Sua contribuição é avaliar o ganho incremental da escolha contextual com comparação entre lojas e teste temporal, incluindo diferenças de serviço, casos sem demanda e situações sem alternativa viável.

'''
related=section('Trabalhos relacionados e posicionamento','Dados e ambiente de reposição')
related=related.replace('O benchmark M5 utilizado aqui não inclui, em seu grafo de execução, a etapa de previsão por redes de memória de longo e curto prazo (LSTM); não atribuímos seus resultados à aprendizagem de previsão.',
    'Na avaliação principal, previsão especializada não é uma intervenção experimental; a escolha é julgada pelas decisões e pelo serviço obtido.')
selection=section('Seleção e protocolo de comparação','Resultados históricos')
selection=selection[:selection.index('\\subsection{Transferência entre lojas e incerteza condicional}')]
selection=selection.replace('Aqui, $n$ é o número de ciclos do cenário','Aqui, $n$ é o número de ciclos do prefixo de treino')
selection=selection.replace('Os perfis de baixo volume estável e venda frequente não aparecem nas matrizes de perfil analisadas.','Os perfis de baixo volume estável e venda frequente não aparecem nos prefixos avaliados da Bahia.')
start=selection.index('O código calcula essas características sobre as séries de cenário completas.')
selection=selection[:start]+r'''Na avaliação temporal, essas características são calculadas exclusivamente no prefixo de treino. Os nomes dos perfis identificam regras locais; não validam sazonalidade ou impacto econômico. A eventual associação de um nome de perfil a uma política no código não determina a escolha: custo e serviço na validação definem a recomendação.

\subsection{Avaliação e incerteza}
As regras global e por perfil usam a validação das demais lojas e são aplicadas ao teste da loja reservada. Cada política é ajustada apenas no treino da respectiva série; o seletor usa resultados de validação, sem consultar o teste para escolher. As sementes são promediadas dentro de cada caso série--origem antes das comparações. Custos incluem todos os casos; médias de atendimento excluem os denominadores nulos, contabilizados separadamente. Origens são medidas repetidas, com um ciclo de teste compartilhado.

'''+interval+'\n\n'

hist_data=section('Dados e ambiente de reposição','Seleção e protocolo de comparação')
table_cohort=re.search(r'\\begin\{table\}.*?\\end\{table\}',hist_data,re.S)[0]
hist_results=section('Resultados históricos','Discussão e implicações operacionais')
hist_tables=re.findall(r'\\begin\{table\}.*?\\end\{table\}',hist_results,re.S)
hist_figs=re.findall(r'\\begin\{figure\}.*?\\end\{figure\}',hist_results,re.S)
historical=r'''\section{Resultados históricos e contexto M5}
\label{app:historical}

Este apêndice preserva a reanálise das matrizes anteriores. Seu objetivo é documentar os antecedentes da avaliação temporal, não acrescentar observações ao teste principal. As mesmas 145 séries da Bahia e 2.408 séries de alimentos do M5 foram avaliadas com doze políticas. Características, preparação, episódios e orçamentos diferem do protocolo principal. Assim, a diferença entre custos históricos e atuais não estima o efeito isolado de uma mudança de método.

O M5 histórico agrega vendas em 93 blocos de 21 dias, incluindo um bloco final parcial. O filtro local FOODS com CV de 1,5 por produto, sobre lojas e ciclos, e a suavização superior por intervalo interquartil usam a trajetória completa. O recorte foi inspirado em Zabraoui et al.\ \cite{sca_27e994c54848}, que não publicam identificadores e corte numérico suficientes para certificar uma amostra idêntica. A nova execução M5 utiliza protocolo próprio e permanece em andamento.

'''+table_cohort+r'''

Os prefixos configurados têm 17 ciclos na Bahia e 19 no M5. No M5, 2.213 séries (91,90\%) não registram demanda no prefixo e todas têm demanda posteriormente. Isso não comprova lançamento dos produtos ou indisponibilidade real de histórico. As diferenças de inicialização, orçamento e frequência limitam a comparação entre bases. Na dinâmica histórica, $L=2$ corresponde a chegada em $t+3$; na avaliação principal, o prazo efetivo é de dois ciclos. A heurística EOQ histórica também usa uma escala de lote distinta da EOQ por ciclo.

As matrizes registram cinco avaliações nominais por série e política, com desvios de custo praticamente nulos. Essas avaliações não são cinco treinamentos independentes. O portfólio completo da Tabela~\ref{tab:portfolio_dual} mostra que baixo custo pode acompanhar atendimento insuficiente; seus denominadores relativos são específicos de cada base.

'''+hist_tables[0]+r'''

\subsection{Seleção retrospectiva entre lojas}
Na reanálise, a política global e as escolhas por perfil são determinadas excluindo a loja avaliada, mas consultando as saídas históricas já armazenadas. A exclusão da loja não elimina informação futura incorporada aos perfis ou à geração das matrizes. Esse desenho difere da escolha na validação e avaliação em teste posterior empregadas no corpo principal.

'''+hist_tables[1]+r'''

Em 0,70, os perfis aumentam o custo em 0,576\% na Bahia e reduzem em 0,681\% no M5, com diferenças de atendimento de 0,779 e $-0{,}122$ ponto percentual. Os intervalos condicionais de redução de custo são $[-1{,}442;0{,}153]\%$ e $[-1{,}190;2{,}402]\%$, respectivamente. Usam 5.000 reamostragens pareadas de lojas, sem reajustar escolhas; no M5, existem apenas dez lojas. Não demonstram equivalência nem vantagem geral dos perfis. A Tabela~\ref{tab:strategies} informa também as violações individuais, que não desaparecem quando a média do grupo é viável.

A sensibilidade histórica das Figuras~\ref{fig:dualmain} e~\ref{fig:dualsensitivity} ilustra mudanças de vencedor com o serviço exigido. No M5, ambas as estratégias convergem para PPO em 0,90 e 0,95. Na Bahia, o aparente ganho de custo em 0,95 acompanha menor atendimento e escolhas de contingência em grupos sem alternativa viável. Esses diagnósticos motivam a separação temporal adotada no estudo principal; não determinam um limiar comercial ótimo.

'''+ '\n'.join(hist_figs)+'\n\\FloatBarrier\n'
(ARTICLE/'historical_ptbr.tex').write_text(historical,encoding='utf-8')

discussion=r'''\section{Discussão e implicações operacionais}
A avaliação temporal responde à pergunta central com um compromisso: os perfis aumentam o atendimento, mas também o custo. A melhora concentra-se na primeira origem, enquanto a segunda reproduz integralmente a escolha global. Não há evidência de economia geral que justifique, por si só, a complexidade adicional da seleção contextual.

Essa interpretação é diferente de comparar apenas as políticas individuais. GA--DQN melhora custo e atendimento frente ao DQN, mas não frente ao GA no conjunto dos testes. Mesmo que um controlador fosse superior em média, ainda seria necessário demonstrar que as características permitem identificar quando recomendá-lo. A distinção entre qualidade dos candidatos e utilidade do seletor aproxima o problema da seleção de algoritmos \cite{rice1976,Kotthoff2014AlgorithmSelection}.

A comparação conceitual com Zabraoui et al.\ \cite{sca_27e994c54848} situa o portfólio e a arquitetura híbrida na literatura. Recortes, implementações e orçamentos distintos impedem confirmar ou refutar seu ranking com os valores deste estudo. O foco em consequências operacionais também se relaciona à avaliação de custo e serviço discutida por Tadayonrad e Ndiaye \cite{sca_f81ad2b3e023}. Precisão de previsão ou identidade do algoritmo não substituem essa comparação.

A agregação temporal exige atenção. Na primeira origem, ambas as estratégias ficam abaixo de 0,70 no teste; reunir as origens produz médias acima desse valor. O atendimento ponderado permanece inferior à média por série, indicando que o serviço não se distribui uniformemente pelo volume de demanda. Uma decisão empresarial precisa definir se a meta se aplica à média das séries, às unidades demandadas ou a itens críticos individualmente.

Os resultados históricos do Apêndice~\ref{app:historical} oferecem antecedentes, mas não estimam o ganho do protocolo temporal. A troca de VendorResponsive por outros vencedores acompanha mudanças de portfólio, preparação e horizonte. Seu valor é mostrar por que a seleção precisa de avaliação própria, e não servir de referência numérica direta para quantificar melhora entre versões.

\section{Limitações e continuidade}
O ajuste e os perfis usam apenas o passado da origem, mas a coorte mantém identificadores escolhidos retrospectivamente. Os testes cobrem sete ciclos e compartilham um ciclo entre origens. Séries da mesma loja e produtos presentes em várias lojas podem ser dependentes. A reamostragem por loja preserva parte dessa estrutura, sem representar novos treinamentos ou novas realizações da demanda.

As vendas podem omitir demanda censurada por falta de estoque. Custos são unidades do cenário e não demonstram economia monetária realizada. O estado inicial comum é produzido por uma política de referência; o teste não mede a continuação do estoque particular de cada candidata. Estoque terminal sem crédito, horizonte curto e valores de custos podem alterar o compromisso observado. Essas condições precisam acompanhar a interpretação dos resultados.

A nova execução M5 deverá ampliar a comparação sob o protocolo temporal, mantendo separadas as bases. A continuidade inclui mais períodos, menor sobreposição dos testes, sensibilidade econômica e restrições de serviço por série. O PSE supervisionado permanece uma extensão a validar frente às regras explícitas de seleção e à melhor referência global.

\section{Conclusão}
Em 145 séries da Bahia, a seleção por perfil acrescenta 1,93\% ao custo e 3,89 pontos percentuais ao atendimento médio, reduzindo as violações de 119 para 100 entre 283 casos definidos. O efeito varia entre períodos e desaparece como diferenciação de escolha na segunda origem. A evidência sustenta avaliar o valor do serviço adicional, sem afirmar economia geral ou superioridade do seletor.

A contribuição é separar desempenho dos controladores, escolha da política e avaliação temporal da recomendação. O resultado permanece condicional à coorte, às janelas e ao cenário. A consolidação do M5 e a validação do PSE são necessárias antes de generalizar a recomendação para outros contextos.

'''
declarations=old[old.index('\\section*{Disponibilidade de dados e código}'):old.index('\\bibliographystyle')]
new=front+intro+related+'\\input{protocolo_bahia_20261015}\n\n'+selection+'\\input{bahia_20261015}\n\n'+discussion+declarations+'\n\\appendix\n\\input{historical_ptbr}\n\\bibliographystyle{elsarticle-num}\n\\bibliography{references_ptbr}\n\\end{document}\n'
(ARTICLE/'manuscript_ptbr.tex').write_text(new,encoding='utf-8')

# Thesis: report the latest evidence first; retain the historical research path.
p=THESIS/'capitulos/resultados.tex';s=p.read_text(encoding='utf-8')
s=s.replace('\\input{bahia_20261015}\n','')
scope=s.index('\\section{Escopo dos Experimentos}')
tail=s.index('\\section{Alcance e Continuidade}')
history=s[scope:tail]
history=history.replace('\\subsection{','\\subsubsection{').replace('\\section{','\\subsection{')
intro_thesis=r'''\mychapter{Resultados e Discussão}{cap:results}\lhead{RESULTADOS E DISCUSSÃO}

A avaliação temporal da Bahia é a evidência principal deste capítulo. Ela compara decisões escolhidas na validação e observadas em teste posterior, sob o protocolo da Seção~\ref{sec:ba2026_methods}. Os estudos históricos da Paraíba, Bahia e M5 aparecem depois, como antecedentes com objetivos e configurações próprios. As etapas não são somadas nem usadas para estimar diretamente uma melhora entre versões.

\input{bahia_20261015}

\section{Estudos históricos e antecedentes da avaliação temporal}
\label{sec:historical_studies}

Os resultados seguintes preservam o percurso da qualificação e a comparação histórica M5. CTI bruto, CTI ajustado por penalidades e objetivo normalizado com BE têm funções distintas. Seus valores documentam o diagnóstico que motivou a avaliação temporal; não substituem seus blocos de validação e teste.

'''
p.write_text(intro_thesis+history+s[tail:],encoding='utf-8')
p=THESIS/'capitulos/proposta.tex';s=p.read_text(encoding='utf-8')
start=s.index('O AIPE liga');end=s.index('\\section{Visão Geral')
s=s[:start]+r'''O AIPE organiza a caracterização da demanda, a comparação de políticas e a avaliação de sua escolha. O protocolo temporal da Bahia é apresentado primeiro, por fundamentar os resultados principais. As seções seguintes desenvolvem a formulação geral, identificam configurações históricas e descrevem o PSE proposto. A distinção entre essas funções evita tratar parâmetros de estudos anteriores como configuração da avaliação atual.

\input{protocolo_bahia_20261015}

'''+s[end:]
s=s.replace('\\section{Instanciação Experimental do Problema de Inventário}','\\section{Formulação e instanciação dos estudos históricos}')
p.write_text(s,encoding='utf-8')
for folder in [ARTICLE,THESIS]:
    paths=list((folder/'capitulos').glob('*.tex')) if folder==THESIS else []
    paths += [p for p in folder.glob('*.tex') if p.name in {'manuscript_ptbr.tex','tcc.tex','abstract_en.tex','bahia_20261015.tex','protocolo_bahia_20261015.tex','historical_ptbr.tex'}]
    for p in paths:
        if p.name=='apendice_b.tex':continue
        s=p.read_text(encoding='utf-8')
        for a,b in [('protocolo corrigido','protocolo temporal'),('protocolo corrigida','protocolo temporal'),('execução corrigida','nova execução'),('execuções corrigidas','novas execuções'),('M5 corrigido','M5 sob o protocolo temporal'),('Bahia corrigida','avaliação temporal da Bahia'),('agregação corrigida','agregação adotada'),('EOQ com unidades corrigidas','EOQ com unidades consistentes')]:s=s.replace(a,b)
        p.write_text(s,encoding='utf-8')
shutil.copy2(ROOT/'experiment/data/08_reporting/comparison/pareto_frontier.pdf',THESIS/'figures/pareto_frontier_historical.pdf')
p=THESIS/'capitulos/resultados.tex';s=p.read_text(encoding='utf-8').replace('../../experiment/data/08_reporting/comparison/pareto_frontier','figures/pareto_frontier_historical');p.write_text(s,encoding='utf-8')
print('Main evidence: temporal Bahia; historical results retained; neutral terminology applied.')
