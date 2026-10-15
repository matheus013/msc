"""Editorial transformations and documented corrections; no experiment mutation."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/master_proposal'

def replace(text, old, new):
    if old not in text:
        if new in text:
            return text
        raise ValueError('Editorial target missing: '+old[:100])
    return text.replace(old,new)

path = DOC/'capitulos/resultados.tex'
s = path.read_text(encoding='utf-8')
s = replace(s,
'Este capítulo reúne os experimentos da qualificação — piloto com 15 lojas da Paraíba e expansão para 145 séries da Bahia — e uma avaliação externa com 2.408 séries M5.',
'Este capítulo reúne o piloto com 15 lojas da Paraíba, a etapa histórica com 145 séries da Bahia, a comparação histórica com 2.408 séries M5 e a nova execução temporal da Bahia.')
s = replace(s,
'A apresentação mantém os resultados históricos da qualificação e distingue seus critérios de comparação. CTI bruto, CTI acrescido de penalidades por indisponibilidade e excesso, e objetivo normalizado com penalização de BE são medidas diferentes. Os experimentos complementares desenvolvidos para o novo artigo não substituem esses resultados: utilizam outro portfólio e outro protocolo. A síntese abaixo privilegia as evidências decisivas e explicita as pendências de consistência que impedem interpretar todo o acervo como um único experimento reproduzido.',
'As etapas respondem a perguntas complementares. Os estudos históricos descrevem o portfólio e a seleção retrospectiva; a nova execução avalia escolhas em blocos posteriores. CTI bruto, CTI ajustado por penalidades e objetivo normalizado com BE permanecem identificados, pois produzem ordenações diferentes. A Seção~\\ref{sec:ba2026} apresenta o protocolo corrigido, sem substituir os valores históricos.')
s = replace(s,
'A cobertura geográfica efetivamente avaliada restringe-se a esses estados.',
'A cobertura brasileira efetivamente avaliada restringe-se a esses estados.')
s = replace(s,
'O piloto foi usado para verificar o ambiente e identificar diferenças entre as políticas em séries de alta intermitência, com CV entre 2,5 e 5,4. A tabela completa permanece no Apêndice~\\ref{app:reproducibility}, Tabela~\\ref{tab:global_prelim}; os resultados não foram transferidos para novos anexos nesta revisão. Nesse recorte, o GA-DQN apresentou CTI até 48\\% menor que $(s,S)$, ruptura 55\\% menor e BE de 5,0, contra 119,4. O NS foi 82,1\\%, frente a 84,3\\%: houve pequena perda de serviço, e não melhoria simultânea de todas as dimensões.',
'O piloto verificou o ambiente em séries de alta intermitência, com CV entre 2,5 e 5,4. A Tabela~\\ref{tab:global_prelim}, no Apêndice~\\ref{app:reproducibility}, apresenta os indicadores completos. GA--DQN teve custo médio 48\\% menor que $(s,S)$ e BE de 5,0, contra 119,4. O atendimento foi menor: 82,1\\%, frente a 84,3\\%. A taxa de ruptura arredondada é 0,050 para ambas as políticas; a tabela não sustenta a redução de 55\\% anteriormente atribuída ao híbrido.')
s = replace(s,'$^\\dagger$DQN sem pedidos.','$^\\dagger$Frequência de pedidos do DQN arredondada a zero.')
s = replace(s,
'O menor CTI pertence ao DQN, que apresenta FP igual a zero e NS de 0,23.',
'O menor CTI pertence ao DQN, com frequência de pedidos reportada como 0,00 e NS de 0,23. O BE de 1,87 é incompatível com uma sequência de pedidos rigorosamente nula; os valores agregados não permitem concluir ausência absoluta de pedidos.')
s = replace(s,
'A configuração atual agrega vendas diárias em blocos de 21 dias, seleciona \\textit{FOODS} com filtro de CV e aplica truncamento superior por $1{,}5\\times\\mathrm{IQR}$. Declara separação de 19 ciclos de treino e restante de avaliação.',
'A configuração histórica M5 agrega vendas diárias em blocos de 21 dias, seleciona \\textit{FOODS} com filtro de CV por produto sobre lojas e ciclos e prevê truncamento superior por $1{,}5\\times\\mathrm{IQR}$. O horizonte de 93 ciclos inclui um bloco final parcial. O prefixo configurado tem 19 ciclos, seguido pelo bloco de avaliação.')
s = replace(s,
'predomina partida sem histórico positivo.',
'predomina ausência de vendas registradas no prefixo. Isso não comprova lançamento de produto ou início real de operação.')
s = replace(s,
'\\section{Previsão como Sinal Auxiliar}',
'\\input{bahia_20261015}\n\n\\section{Previsão como Sinal Auxiliar}')
s = replace(s,
'A consolidação do M5 já disponível e a expansão multiestado, discutidas no Capítulo~\\ref{cap:plano_continuidade}, exigem protocolo próprio e critérios de validação definidos antes da análise de teste.',
'A nova avaliação da Bahia acrescenta teste temporal, mas mantém a coorte retrospectiva e duas janelas parcialmente sobrepostas. A conclusão do M5 corrigido e a validação do seletor, discutidas no Capítulo~\\ref{cap:plano_continuidade}, exigem critérios definidos antes de examinar o teste.')
path.write_text(s,encoding='utf-8')

path = DOC/'capitulos/proposta.tex'
s = path.read_text(encoding='utf-8')
s = replace(s,'A sazonalidade é resumida pela autocorrelação de lag 17, correspondente a um ano comercial.',
'A dependência anual é resumida pela autocorrelação na defasagem 17, correspondente a um ano comercial.')
s = replace(s,'Autocorrelação de lag 17','Autocorrelação na defasagem 17')
s = replace(s,
'O preparo consolida transações por armazém, loja, produto e ciclo. São removidos valores negativos ou registros incompatíveis com o recorte, conforme a configuração da execução. Quantidades repetidas entre arquivos precisam ser agregadas, não mantidas como observações distintas.',
'O preparo consolida transações por armazém, loja, produto e ciclo. Valores negativos e registros incompatíveis seguem os filtros da versão executada. Transações complementares da mesma chave são somadas; repetições técnicas do mesmo evento exigem identificação antes da remoção. A nova execução da Bahia utiliza a agregação corrigida descrita na Seção~\\ref{sec:ba2026}.')
s = replace(s,
'A expansão multiestado e novas avaliações temporais permanecem no plano de continuidade.',
'A nova avaliação temporal da Bahia está concluída; o M5 corrigido permanece em execução. A expansão nacional foi retirada da execução ativa.')
path.write_text(s,encoding='utf-8')

(DOC/'capitulos/conclusoes.tex').write_text(r'''\mychapter{Conclusões}{cap:conclusion}\lhead{CONCLUSÕES}

Esta dissertação investiga se características da demanda ajudam a escolher políticas de reposição com melhor compromisso entre custo e atendimento. O AIPE organiza essa avaliação em preparação, caracterização e comparação de candidatos. Os resultados disponíveis sustentam examinar a utilidade incremental da escolha por perfil; ainda não demonstram economia geral nem validam o PSE supervisionado.

\section{Respostas às questões de pesquisa}

O desempenho relativo das políticas varia entre recortes e períodos. O resultado favorável de GA--DQN no piloto da Paraíba não se repete como superioridade agregada na Bahia. No M5 histórico, apenas SA e PPO superam o serviço médio de 0,70. Essas diferenças descrevem implementações e protocolos específicos, sem identificar um controlador universal.

A nova execução da Bahia torna mais precisa a resposta sobre os perfis. Os 4.350 ajustes cobrem as 145 séries em duas origens, com escolha na validação das demais lojas e avaliação no teste da loja reservada. Em conjunto, selecionar por perfil aumenta o custo em 1,93\% e o atendimento médio em 3,89 pontos percentuais. Os casos abaixo da meta passam de 119 para 100 entre os 283 com demanda positiva no teste. O resultado representa serviço adicional com custo maior. Na segunda origem, as estratégias fazem escolhas idênticas.

Essas observações ainda não respondem integralmente às perguntas sobre aprendizagem e generalização. A escolha explícita por perfil foi avaliada temporalmente, mas o PSE supervisionado requer treinamento e teste próprios. A coorte continua selecionada retrospectivamente; as duas janelas são curtas, compartilham um ciclo e repetem as mesmas séries. A transferência para outros produtos, períodos e fontes de dados permanece aberta.

\section{Contribuições e limites}

A contribuição metodológica é distinguir três problemas: calibrar uma política, escolher uma candidata do portfólio e avaliar a recomendação em dados posteriores. A contribuição empírica é examinar essa distinção em dados reais do varejo brasileiro e contrastá-la com resultados históricos públicos do M5. A infraestrutura integrada permite rastrear dados, configurações, sementes e saídas da nova execução.

Os métodos do portfólio e a arquitetura GA--DQN já existem na literatura. Seu uso não constitui novidade algorítmica. Na nova Bahia, o híbrido melhora custo e atendimento frente ao DQN, mas não frente ao GA na média dos testes. Como também consome busca genética e demonstrações, a comparação não isola um ganho causal da combinação. A avaliação M5 corrigida deverá complementar essa leitura antes de uma decisão definitiva sobre o portfólio.

As matrizes históricas preservam valor exploratório, com limites de rastreabilidade, uso do histórico completo e repetições determinísticas. A nova execução restringe o ajuste ao passado e documenta sementes independentes de treinamento, mas essas sementes não são novas lojas nem novas realizações da demanda. Custos simulados, vendas possivelmente censuradas e estados iniciais de referência delimitam a interpretação operacional.

\section{Implicações e continuidade}

A definição do serviço antecede a escolha do controlador. Média por série, atendimento ponderado pela demanda e garantia individual produzem exigências distintas. Na nova Bahia, as médias das estratégias superam 0,70 quando as origens são reunidas, mas o atendimento ponderado permanece abaixo desse alvo e há numerosas violações individuais. A média global, portanto, é insuficiente para julgar a adequação empresarial.

O avanço seguinte é consolidar o M5 corrigido, examinar estabilidade temporal e definir o valor econômico do atendimento adicional antes de treinar o PSE. O plano do Capítulo~\ref{cap:plano_continuidade} organiza essas dependências. A conclusão atual é delimitada: perfis podem alterar o compromisso entre custo e serviço, mas seu benefício precisa ser demonstrado frente a uma política global competitiva em períodos reservados.
''',encoding='utf-8')

(DOC/'capitulos/plano_continuidade.tex').write_text(r'''\mychapter{Plano de Continuidade}{cap:plano_continuidade}\lhead{PLANO DE CONTINUIDADE}

A continuidade parte de uma avaliação temporal da Bahia já concluída e de uma reexecução M5 em andamento. O objetivo é consolidar evidência comparável para a seleção de políticas e, então, avaliar o PSE supervisionado. A expansão nacional foi cancelada na execução atual e não constitui requisito para concluir as etapas seguintes.

\section{Consolidação dos experimentos}

A nova Bahia dispõe de manifesto, 4.350 ajustes completos e métricas de validação e teste. A análise deverá examinar a variação entre origens, os casos sem demanda e o compromisso entre custo e atendimento. Os resultados históricos da Paraíba, Bahia e M5 permanecem identificados pelos respectivos protocolos; correções do código não os transformam em saídas da versão atual.

\subsection{M5: avaliação pública com protocolo corrigido}
\label{sec:estudos_caso_adicionais}

A execução M5 utiliza a coorte histórica de 2.408 identificadores, sujeita à elegibilidade no prefixo de cada origem. Isso preserva uma comparação condicional à coorte, sem tornar prospectiva sua seleção inicial. O recorte local é inspirado em \citet{Zabraoui2025GaDQN}; não reproduz uma amostra publicada com identificadores e filtros numéricos integralmente especificados.

A comparação com a Bahia deverá usar métricas e regras de escolha comuns, mantendo separadas as escalas de custo e as propriedades da demanda. A execução corrigida M5 ainda não sustenta conclusões neste documento. O resultado histórico continua relevante como diagnóstico de um cenário em que 91,9\% das séries não tinham vendas no prefixo configurado.

\section{Validação da seleção}

O próximo desenho deverá ampliar os períodos e reduzir a sobreposição dos testes. Características, normalização, parâmetros e critérios de inclusão devem depender do passado disponível. A seleção usará validação, e o teste ficará reservado à avaliação. O comparador principal será a melhor política global viável escolhida sob a mesma informação.

O PSE receberá características anteriores à decisão e rótulos produzidos apenas nos dados de ajuste. Sua avaliação incluirá custo, atendimento, déficit de serviço e arrependimento relativo à referência. Acurácia e F1 serão diagnósticos auxiliares: rótulos próximos em desempenho não têm o mesmo significado econômico que erros com grande perda de serviço. Modelos supervisionados serão comparados com política global, regras por perfil e modelos simples, respeitando tempo e agrupamento por loja.

\subsection{Calibração agrupada por perfil}
\label{sec:robustez_pooling}

Calibrar parâmetros em grupos é uma hipótese distinta de escolher políticas por perfil. Uma eventual comparação entre ajuste individual e agrupado deverá manter cortes, custos e orçamento comuns. Benefícios e inversões entre períodos serão apresentados conjuntamente, sem interpretar um recorte favorável como robustez geral.

\section{Sensibilidade e eficiência}

As análises deverão variar custos, prazo, estoque inicial, tratamento terminal e exigência de serviço. Média por série, atendimento ponderado e restrições individuais serão comparados como problemas de decisão distintos. A escolha de um peso para custo e serviço deverá anteceder o teste, evitando ajustar o objetivo para favorecer um resultado observado.

A revisão do portfólio combinará diversidade das regras, custo computacional e evidência nas duas bases. Métodos semelhantes não serão considerados equivalentes apenas por correlação de custo. A eventual retirada de candidatos será definida nos dados de desenvolvimento e documentada antes da avaliação final. GA--DQN continuará sendo comparado com GA e DQN, com o esforço total da combinação informado.

CPU e GPU serão escolhidas por medições do tempo total e do uso de recursos. A execução M5 atual usa oito processos CPU para coexistir com o uso normal da máquina. Essa configuração operacional não altera sementes, cortes ou candidatos, nem é um resultado de qualidade das políticas.

\section{Consolidação do texto e critérios de conclusão}

O artigo para \textit{Supply Chain Analytics} e a dissertação compartilham resultados conferidos, mas têm funções diferentes: o artigo concentra a avaliação incremental da seleção; a dissertação explica fundamentos, percurso e limites. O material SBPO permanece arquivado. Autoria, declarações, disponibilidade de dados e adequação editorial serão consolidadas antes de submissão ou depósito.

\begin{table}[htbp]
\centering\small
\caption{Etapas de continuidade e critérios de conclusão, sem datas ainda não acordadas.}
\label{tab:cronograma}
\begin{tabular}{@{}p{4.1cm}p{8.9cm}@{}}\toprule
Etapa & Critério de conclusão\\\midrule
Consolidação Bahia & Resultados temporais, falhas de serviço e limites documentados\\
Conclusão M5 & Cobertura planejada, matriz completa e comparação por origem\\
Seleção e sensibilidade & Critérios de decisão definidos e teste preservado\\
Avaliação do PSE & Comparação fora da amostra com referências competitivas\\
Redação e defesa & Afirmações, contribuições e pendências coerentes com a evidência\\
\bottomrule\end{tabular}
\end{table}
''',encoding='utf-8')

# Presentation fixes: never change values, identifiers, bibliography or plot files.
decimal = re.compile(r'(?<![\w.])(0\.\d+|\d+\.\d{1,2})(?![\d.])')
math_span = re.compile(r'(?<!\\)\$(?:\\.|[^$])*?(?<!\\)\$|\\\[.*?\\\]|\\begin\{(equation\*?|align\*?)\}.*?\\end\{\1\}', re.S)
for path in (DOC/'capitulos').glob('*.tex'):
    if path.name == 'apendice_b.tex':
        continue
    text = path.read_text(encoding='utf-8')
    protected = []
    def protect(match):
        protected.append(match[0])
        return 'EDITORIALCODE'+chr(65+len(protected)-1)+'END'
    # Dot decimals are required in TeX dimensions, TikZ coordinates and options.
    for pattern in [r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}',
                    r'\\includegraphics[^\n]*',r'\\begin\{tabular\}[^\n]*',
                    r'\\renewcommand\{\\arraystretch\}[^\n]*']:
        text = re.sub(pattern,protect,text,flags=re.S if 'tikzpicture' in pattern else 0)
    segments=[];last=0
    for match in math_span.finditer(text):
        segments.append(decimal.sub(lambda m:m[0].replace('.',','),text[last:match.start()]))
        segments.append(decimal.sub(lambda m:m[0].replace('.','{,}'),match[0]))
        last=match.end()
    segments.append(decimal.sub(lambda m:m[0].replace('.',','),text[last:]))
    text=''.join(segments)
    for index, code in enumerate(protected):
        text=text.replace('EDITORIALCODE'+chr(65+index)+'END',code)
    if path.name == 'apendice.tex':
        pairs={'BA:145':'BA: 145','séries,115':'séries, 115',' e9 ':' e 9 ',
        'de38':'de 38','2.408séries':'2.408 séries','lojas,287':'lojas, 287',' e93 ':' e 93 ',
        'aproximadamente21':'aproximadamente 21','Manutenção1':'Manutenção 1','falta5':'falta 5',
        'fixo50':'fixo 50','variável0,5':'variável 0,5','Referência42':'Referência 42',
        'População100':'População 100','cruzamento0,8':'cruzamento 0,8','torneio3':'torneio 3',
        'inicial0,05':'inicial 0,05','até20':'até 20','inicial1.000':'inicial 1.000',
        'resfriamento0,95':'resfriamento 0,95','lote64':'lote 64','memória10.000':'memória 10.000',
        'cada10':'cada 10','taxa0,001':'taxa 0,001','referência:1.000':'referência: 1.000',
        'recorte0,2':'recorte 0,2','taxa0,0003':'taxa 0,0003','com1.000':'com 1.000',
        'e200':'e 200','entrada6':'entrada 6','LSTM:64':'LSTM: 64','unidades,50':'unidades, 50',
        'ANN:camadas':'ANN: camadas','máximo200':'máximo 200','validação0,1':'validação 0,1',
        'Previsão:200':'Previsão: 200','taxa0,05':'taxa 0,05','subamostragem0,8':'subamostragem 0,8',
        'Suavização0,1':'Suavização 0,1','ordem$':'ordem $','CV$^2$0,49':'CV$^2$ 0,49',
        'silêncio5':'silêncio 5','burstiness0,30':'burstiness 0,30','autocorrelação0,25':'autocorrelação 0,25',
        'tendência0,05':'tendência 0,05','entropia2':'entropia 2','XGBoost:300':'XGBoost: 300',
        'colunas0,8':'colunas 0,8','de15':'de 15','de48':'de 48','frente a$':'frente a $',
        'NS0,821':'NS 0,821','contra0,843':'contra 0,843','ouM5':'ou M5','PB:15':'PB: 15'}
        for a,b in pairs.items():text=text.replace(a,b)
    text=text.replace('GA-DQN','GA--DQN').replace('GA-PPO','GA--PPO')
    path.write_text(text,encoding='utf-8')
