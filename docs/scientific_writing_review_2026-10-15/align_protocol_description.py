"""Align active dissertation methods with the evaluated experiment, not prototypes."""
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/master_proposal'
archive=Path(__file__).parent/'before_question_focus/dissertation'
for name in ['capitulos/apendice.tex']:
    target=archive/Path(name).name
    if not target.exists(): shutil.copy2(DOC/name,target)
p=DOC/'capitulos/proposta.tex'
s=p.read_text(encoding='utf-8')
def paragraph(start,new):
    global s
    a=s.index(start); b=s.find('\n\n',a)
    if b<0:b=len(s)
    s=s[:a]+new+s[b:]
paragraph('Na Bahia, os artefatos de perfil',
    'Na Bahia, o prefixo de 17 ciclos apresenta 65 séries Sparse High Impact e 80 Unstable Trend. No prefixo de 23 ciclos, são 60, 69 e 16 High Volatility Seasonal. A caracterização é recalculada apenas no treino; diferenças entre origens refletem tanto a janela quanto as regras, inclusive a indisponibilidade da autocorrelação na primeira origem. Os limiares são heurísticos e não foram aprendidos em uma população independente.')
paragraph('A comparação emprega MASE',
    'Uma avaliação específica de previsão deverá empregar MASE e RMSSE escalados pela referência ingênua do treino e explicitar os casos de sMAPE com denominador nulo. Essas métricas não foram usadas para escolher as políticas reportadas no capítulo de resultados. Se o erro de previsão integrar o PSE, deverá ser produzido sem observar o bloco usado para avaliar a recomendação.')
s=s.replace(r'Q_{\mathrm{ref}}&=\sqrt{2\hat\mu T_{\mathrm{cal}}c_o^{\mathrm{fix}}/h}',r'Q_{\mathrm{ref}}&=\sqrt{2\hat\mu c_o^{\mathrm{fix}}/h}')
paragraph('$T_{\\mathrm{cal}}$ é a quantidade',
    'A média e a dispersão são estimadas no prefixo de treino, com demanda e manutenção expressas por ciclo. A primeira expressão define a inicialização de referência anterior ao aquecimento; não substitui os estados comuns usados na validação e no teste, obtidos pela trajetória EOQ descrita na Seção~\\ref{sec:ba2026_methods}.')
a=s.index('O portfólio inicial contém');b=s.index('\\subsection{Regras Analíticas',a)
s=s[:a]+r'''O portfólio avaliado contém EOQCycle, sSCycle, CappedBaseStock, GA, PPO e GA--DQN; DQN é o controle adicional da hibridização. A composição busca referências analíticas, busca de parâmetros e aprendizado, sem acumular algoritmos de comportamento semelhante. A arquitetura híbrida é motivada por \citet{Zabraoui2025GaDQN}; sua inclusão não pressupõe superioridade.

CappedBaseStock é motivada por \citet{Xin2021CappedBaseStock} e pelas comparações discutidas por \citet{Temizoz2025DeepControlledLearning}. Outros módulos existentes no projeto, como Jornaleiro, SA, PSO ou DE, não integram a matriz deste experimento. Sua disponibilidade no código não equivale a evidência reportada. A retirada de candidatos é uma decisão de escopo e redundância, sem afirmar ausência de literatura ou inferioridade universal.

'''+s[b:]
a=s.index('EOQ legado pede');b=s.index('\\subsection{Busca de Parâmetros}',a)
s=s[:a]+r'''EOQCycle pede um lote $Q_{\mathrm{ref}}$ quando a posição de estoque cai até $s_{\mathrm{ref}}$. sSCycle compartilha o limiar e repõe até $S=s_{\mathrm{ref}}+Q_{\mathrm{ref}}$. Média, dispersão e escalas são calculadas no treino, usando a mesma unidade temporal dos custos. O nome sSCycle identifica essa instanciação, sem alegar otimização exata dos parâmetros de uma política $(s,S)$.

CappedBaseStock acrescenta uma regra de reposição até um alvo com limite de quantidade. Sua interpretação depende das hipóteses e da calibração implementadas, além do rótulo bibliográfico. As três referências determinísticas recebem um ajuste por série e origem e são avaliadas nas mesmas condições de demanda e estoque inicial das candidatas aprendidas.

'''+s[b:]
s=s.replace('GA e SA ajustam','GA ajusta')
paragraph('A configuração de referência histórica BA do GA',
    'O GA emprega 100 indivíduos e 50 gerações em cada ajuste, com sementes 11, 29 e 47. Operadores e a função de aptidão são identificados pela configuração e pelo código arquivados. O artigo-base motiva a busca evolutiva~\\citep{Zabraoui2025GaDQN}, mas a implementação local e seu orçamento devem ser explicitados, sem pressupor reprodução exata.')
paragraph('Os parâmetros de referência histórica incluem',
    'DQN e PPO utilizam duas camadas de 64 unidades e 20 ações discretas. O orçamento mínimo é de 20 mil transições, completando o último episódio: 20.009 e 20.010 transições nas origens Bahia de 17 e 23 ciclos. O número de episódios deriva desse orçamento e do tamanho do treino. As sementes são 11, 29 e 47. Os valores efetivamente usados, incluindo limites de ação e atualização, acompanham o manifesto e a configuração de reprodução.')
paragraph('GA--DQN utiliza a parametrização',
    'GA--DQN usa a solução evolutiva para gerar mil transições de demonstração e depois refina o agente com o orçamento de transições do DQN. Mantém gatilho e piso de pedido entre demonstração, treinamento e inferência. A busca genética e as demonstrações representam esforço adicional: comparar o híbrido ao DQN não isola o efeito da arquitetura sob computação equivalente. A combinação é motivada por \\citet{Zabraoui2025GaDQN} e não constitui a novidade principal da seleção contextual.')
paragraph('Correções de operadores, penalidades',
    'Episódios repetidos na demanda observada são interações sintéticas, não novas amostras de vendas. Sementes distintas medem parte da variabilidade de ajuste, sem substituir avaliação em outras lojas ou períodos. Explicar o efeito isolado de operadores, demonstrações ou gatilhos exige ablações controladas.')
paragraph('O simulador recebe demanda fixa',
    'O simulador recebe a trajetória de demanda, o estado inicial comum, custos e uma política. O estado pode incluir pedidos em trânsito produzidos pelo aquecimento. A política decide com informação passada, e o recebimento segue a convenção exata de prazo. A demanda é atendida com o estoque disponível; faltas são perdidas, e manutenção incide sobre o saldo após demanda. Custos fixos e variáveis do pedido são cobrados na emissão, sem crédito por estoque ou pedidos em trânsito ao final.')
paragraph('Na instanciação histórica, a fila contém',
    'O prazo efetivo é de dois ciclos: o pedido emitido em $t$ torna-se disponível em $t+2$ segundo a ordem de eventos da execução. Essa convenção é compartilhada entre candidatas e deve acompanhar qualquer reprodução. O horizonte curto, o estado inicial e a ausência de crédito terminal limitam a interpretação.')
a=s.index('A documentação inicial menciona');b=s.index('O treinamento de controladores',a)
s=s[:a]+'Na Bahia, os prefixos de treino têm 17 e 23 ciclos, seguidos por sete de validação e sete de teste. As duas origens compartilham um ciclo de teste. Características, escalas e parâmetros usam somente o prefixo; a escolha usa a validação das demais lojas. O teste da loja reservada não participa dessas decisões.\n\n'+s[b:]
paragraph('O segundo experimento registra cinco',
    'As três sementes estocásticas são promediadas dentro de cada caso série--origem e política antes da comparação. A reamostragem pareada preserva os produtos e origens de cada loja. Os intervalos são condicionais aos ajustes e escolhas fixos, sem retreinamento nos sorteios; por isso não representam toda a incerteza da investigação.')
paragraph('O piloto da Paraíba fornece',
    'A Bahia reúne 145 séries em 115 lojas e nove produtos, com duas origens e 4.350 ajustes completos. O M5 utiliza uma coorte de 2.408 identificadores, com elegibilidade própria em cada prefixo, e permanece em execução. As bases são analisadas separadamente sob perguntas e critérios comuns. A expansão nacional foi retirada da execução ativa.')
paragraph('O Apêndice~\\ref{app:reproducibility} concentra',
    'O Apêndice~\\ref{app:reproducibility} concentra os parâmetros da avaliação e sua rastreabilidade. Dados, fontes de conhecimento e versões de desenvolvimento são preservados no projeto, sem compor matrizes paralelas na síntese dos resultados.')
paragraph('Os resultados históricos do M5',
    'O Capítulo~\\ref{cap:results} responde às perguntas com a cobertura disponível, distinguindo evidência completa de análises ainda pendentes. A execução M5 só sustentará conclusões após conferência de sua matriz.')
s=s.replace('A nova execução da Bahia utiliza a agregação adotada','A avaliação da Bahia utiliza a agregação').replace('na preparação histórica','na preparação').replace('Mantêm-se os rótulos já arquivados para rastrear os resultados, mas sem apresentar equivalência exata entre as definições.','Os perfis são recalculados no treino, sem pressupor equivalência exata entre as definições.')
p.write_text(s,encoding='utf-8')
(DOC/'capitulos/apendice.tex').write_text(r'''\appendix
\chapter{Parâmetros e rastreabilidade da avaliação}
\label{app:reproducibility}
\lhead{APÊNDICE A}

Este apêndice sintetiza os parâmetros da matriz avaliada. Detalhes de módulos propostos, como previsão especializada e PSE supervisionado, não são apresentados como configurações de um resultado já demonstrado.

\begin{table}[htbp]\centering\small
\caption{Configuração da avaliação Bahia.}
\begin{tabular}{@{}p{3cm}p{11.4cm}@{}}\toprule
Componente & Configuração\\\midrule
Coorte & 145 séries, 115 lojas, nove produtos; 38 ciclos disponíveis.\\
Janelas & Treino de 17 ou 23 ciclos; validação e teste de sete ciclos cada; um ciclo de teste compartilhado.\\
Prazo & Dois ciclos efetivos, com convenção exata de entrega.\\
Custos & Manutenção 1/unidade/ciclo; falta 5/unidade; emissão 50/pedido; variável 0,5/unidade.\\
Estado inicial & Aquecimento de seis ciclos com EOQ de referência; validação da referência define o estado comum do teste.\\
Portfólio & EOQCycle, sSCycle, CappedBaseStock, GA, PPO, GA--DQN; DQN como controle.\\
Sementes & 11, 29 e 47 nos quatro métodos estocásticos; um ajuste nas três referências determinísticas.\\
GA & População 100; 50 gerações.\\
Redes & Duas camadas de 64 unidades; 20 ações discretas; escalas estimadas no treino.\\
Orçamento RL & Mínimo de 20 mil transições por ajuste, completando o episódio final.\\
Híbrido & Busca genética e mil demonstrações antes do refinamento DQN; esforço adicional não equiparado ao controle.\\
Seleção & Validação das demais lojas; menor custo entre candidatas com atendimento médio de pelo menos 0,70. Sem viável, maior serviço com contingência explícita.\\
Métricas & Custo em 290 casos; atendimento em 283 casos com demanda; sete denominadores nulos contabilizados separadamente.\\
Incerteza & 5.000 reamostragens pareadas por loja; todas as origens preservadas; ajustes e escolhas fixos.\\
\bottomrule\end{tabular}\end{table}

\section{Artefatos para reprodução}

O diretório da rodada é \nolinkurl{experiment/data/08_reporting/research_v2/bahia145_cpu16_2026-10-15}. Manifesto, banco de ajustes, KPIs e estado de conclusão permitem conferir cobertura e proveniência. A configuração e os arquivos de código registrados pela execução identificam a implementação efetivamente utilizada. Os dados ficam fora do Git; o backup armazena suas partes e manifesto com hashes para verificação.

As figuras são produzidas por \nolinkurl{experiment/scripts/documents/article_research_figures.py}. A auditoria das escolhas e os derivados de escrita ficam em \nolinkurl{docs/scientific_writing_review_2026-10-15/}. A versão de desenvolvimento dos documentos é preservada nos arquivos de revisão, sem substituir os dados da matriz reportada.

O M5 tem diretório próprio e permanece em execução. Sua cobertura, parâmetros específicos de elegibilidade e resultados deverão ser conferidos antes da consolidação das conclusões. Compartilhar uma fonte de dados com outro artigo não demonstra reprodução exata de seu recorte ou de seus algoritmos.
''',encoding='utf-8')
print('Methods and reproducibility appendix aligned to evaluated protocol.')
