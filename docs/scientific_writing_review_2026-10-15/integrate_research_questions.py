"""Replace article's version-based narrative with a focused research design."""
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/supply_chain_analytics_article'
archive=Path(__file__).parent/'before_question_focus'
archive.mkdir(exist_ok=True)
for name in ['manuscript_ptbr.tex','manuscript_ptbr.pdf','bahia_20261015.tex','protocolo_bahia_20261015.tex']:
    if not (archive/name).exists(): shutil.copy2(DOC/name,archive/name)

p=DOC/'manuscript_ptbr.tex'
s=p.read_text(encoding='utf-8')
s=s.replace(' Matrizes históricas da Bahia e de 2.408 séries M5 complementam o contexto, com protocolos distintos da avaliação principal.','')
start=s.index('A pesquisa parte dos experimentos da qualificação')
end=s.index('\\section{Trabalhos relacionados',start)
s=s[:start]+r'''O artigo concentra-se em uma pergunta: selecionar políticas por perfil de demanda melhora o compromisso entre custo e atendimento frente à seleção de uma política global? As duas estratégias usam a mesma informação passada e o mesmo portfólio. A avaliação combina dados de uma rede brasileira e um conjunto público de varejo, mantendo as bases separadas. Neste estágio, a matriz da Bahia está completa e a execução M5 está em andamento; apenas resultados completos sustentam conclusões quantitativas.

Três análises sustentam essa pergunta: o compromisso custo--serviço das candidatas; a diferença entre seleção global e por perfil no teste; e a estabilidade das escolhas e do atendimento entre origens temporais. A comparação de controladores serve para interpretar a seleção, sem constituir uma segunda proposta de algoritmo. O estudo não pretende abranger todos os componentes do AIPE nem validar seu seletor supervisionado.

'''+s[end:]
s=s.replace('Corrigir um controlador atual não valida retrospectivamente uma saída armazenada de outra versão.','A interpretação exige identificar o controlador, seu orçamento e as condições comuns de avaliação.')
s=s.replace('A comparação adicional examina retrospectivamente se o mesmo procedimento de seleção apresenta comportamento semelhante em outro conjunto, mantendo claras as restrições da transformação e do recorte.','O conjunto público permite investigar a mesma pergunta de seleção em outro contexto, com protocolo explicitado por base e sem comparar custos absolutos entre bases.')
start=s.index('Os resultados históricos do Apêndice')
end=s.index('\\section{Limitações',start)
s=s[:start]+s[end:]
s=s.replace('A nova execução M5 deverá ampliar a comparação sob o protocolo temporal, mantendo separadas as bases.','A execução M5 deverá completar a avaliação da mesma pergunta em uma base pública, mantendo separadas as estimativas por conjunto.')
s=s.replace('\\appendix\n\\input{historical_ptbr}\n','')
p.write_text(s,encoding='utf-8')

p=DOC/'protocolo_bahia_20261015.tex'
s=p.read_text(encoding='utf-8')
s=s.replace('A avaliação principal utiliza','A avaliação utiliza')
s=s.replace('Os identificadores da coorte provêm do estudo histórico; a preparação e o protocolo desta execução são descritos a seguir. Os resultados históricos da Bahia e do M5 têm função complementar e não são concatenados à avaliação temporal.','A coorte foi definida retrospectivamente; os ajustes e a seleção, porém, utilizam apenas a informação disponível antes do teste. Essa distinção limita a generalização para séries ainda não observadas.')
s=s.replace('A coorte mantém os identificadores históricos;','A coorte tem identificadores fixos;')
s=s.replace('Cinquenta e quatro ajustes compatíveis foram reaproveitados com proveniência registrada. ','')
s+=r'''
\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/fluxo_experimental.png}
\caption{Fluxo da avaliação e separação da informação. Cada candidata é ajustada no treino da própria série. A estratégia de seleção usa a validação das demais lojas e é avaliada no teste da loja reservada. Os dois testes compartilham um ciclo; as origens são medidas repetidas.}
\label{fig:experimental_flow}
\end{figure}

O fluxo da Figura~\ref{fig:experimental_flow} mantém separadas duas decisões: ajustar o controlador e escolher qual controlador recomendar. Para cada caso, o atendimento é $F=\sum_t v_t/\sum_t d_t$, em que $v_t$ é a demanda atendida e $d_t$ a demanda observada. Quando o denominador é zero, $F$ permanece indefinido. A média por caso atribui peso igual às séries com demanda; o atendimento ponderado usa a razão entre todas as unidades atendidas e demandadas. Ambos acompanham o custo total, para evitar que baixo atendimento seja interpretado como eficiência econômica.
'''
p.write_text(s,encoding='utf-8')

p=DOC/'bahia_20261015.tex'
s=p.read_text(encoding='utf-8').replace('\\section{Resultados da avaliação temporal}','\\section{Resultados: valor da seleção por perfil}')
s=s.replace('\\subsection{Custo e atendimento no teste}','\\subsection{Quais candidatas oferecem compromissos competitivos?}')
s=s.replace('\\subsection{Seleção global e por perfil}','\\subsection{O perfil melhora a decisão frente à seleção global?}')
s=s.replace('\\subsection{Perfis e variação entre origens}','\\subsection{As escolhas e o atendimento são estáveis entre períodos?}')
marker='\\subsection{O perfil melhora a decisão frente à seleção global?}'
figure=r'''\begin{figure}[htbp]
\centering\includegraphics[width=\linewidth]{figures/custo_servico.png}
\caption{Compromisso custo--serviço das sete candidatas na Bahia. Médias de teste em 290 casos série--origem para custo e 283 com demanda positiva para atendimento. A fronteira é descritiva: não utiliza o teste para escolher a recomendação nem representa dominância estatística ou por série.}
\label{fig:cost_service}
\end{figure}

A Figura~\ref{fig:cost_service} distingue economia aparente por baixo atendimento de um compromisso competitivo. A restrição usada pelo seletor continua sendo avaliada na validação, sem recorrer a essa fronteira de teste.

'''
s=s.replace(marker,figure+marker)
marker='\\subsection{As escolhas e o atendimento são estáveis entre períodos?}'
s=s.replace(marker,r'''\begin{figure}[htbp]
\centering\includegraphics[width=\linewidth]{figures/distribuicao_atendimento.png}
\caption{Distribuição do atendimento das estratégias no teste, por origem. Cada caixa resume séries com demanda positiva; a linha tracejada indica 0,70. Há 144 casos por estratégia na primeira origem e 139 na segunda. As mesmas séries participam das duas estratégias; observações e origens não são independentes.}
\label{fig:service_distribution}
\end{figure}

A Figura~\ref{fig:service_distribution} mostra por que a comparação não se esgota na média: a primeira janela concentra violações de serviço, enquanto a segunda tem distribuição e escolha coincidentes entre estratégias.

'''+marker)
s=s.replace('Os perfis calculados no prefixo diferem dos rótulos do histórico completo.','Os perfis são recalculados no prefixo disponível em cada origem.')
s+=r'''
\begin{figure}[htbp]
\centering\includegraphics[width=\linewidth]{figures/escolhas_perfil.png}
\caption{Políticas escolhidas por perfil na validação das demais lojas, com meta 0,70. Cada painel contém 145 escolhas; os números indicam séries. A primeira origem diferencia grupos, enquanto a segunda recomenda sSCycle em todos eles. Nomes abreviados dos perfis correspondem às regras operacionais descritas no método.}
\label{fig:profile_choices}
\end{figure}

A Figura~\ref{fig:profile_choices} localiza a diferenciação que produz o serviço adicional: na primeira origem, a regra por perfil substitui GA por sSCycle ou EOQCycle em parte das séries. Na segunda, identificar perfis não muda a política recomendada. Assim, a existência de grupos distintos não implica valor incremental da seleção em todo período.
\FloatBarrier
'''
p.write_text(s,encoding='utf-8')

# Keep the dissertation's broader scope, while applying the same evidence hierarchy.
thesis=ROOT/'docs/master_proposal'
for name in ['bahia_20261015.tex','protocolo_bahia_20261015.tex']:
    shutil.copy2(DOC/name,thesis/name)
for image in (DOC/'figures').glob('*.png'):
    shutil.copy2(image,thesis/'figures'/image.name)
print('Focused article sources updated; shared evaluation and figures synced to dissertation.')
