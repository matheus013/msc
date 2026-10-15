"""Keep a single experiment narrative, preserving prior sources outside the text."""
from pathlib import Path
import shutil
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/master_proposal'
archive=Path(__file__).parent/'before_question_focus/dissertation'
archive.mkdir(exist_ok=True,parents=True)
names=['capitulos/resultados.tex','capitulos/conclusoes.tex','capitulos/resumo.tex','abstract_en.tex','capitulos/proposta.tex','capitulos/plano_continuidade.tex']
for name in names:
    target=archive/Path(name).name
    if not target.exists(): shutil.copy2(DOC/name,target)

run=ROOT/'experiment/data/08_reporting/research_v2/bahia145_cpu16_2026-10-15'
raw=pd.read_csv(run/'kpis.csv')
fits=raw.drop_duplicates(['dataset','origin','warehouse','store_id','item_id','policy','seed'])
# A reused fit carries an earlier measured duration; exclude it from this run's timing sample.
fresh=fits[fits.reused_from.isna() | fits.reused_from.eq('')]
timing=fresh.groupby('policy').fit_seconds.agg(['count','median','mean'])
table=r'''\section{Esforço computacional das políticas}
\label{sec:computational_results}

Tempo de ajuste é uma dimensão adicional da comparação, apresentada na Tabela~\ref{tab:fit_time}. Cada ajuste é contado uma vez, sem duplicar seus registros de validação e teste. Ajustes reaproveitados são excluídos desta amostra de tempos. O registro mede a duração do ajuste no ambiente de execução, sob concorrência e carga variável; não equivale a um teste controlado de velocidade entre processadores nem ao tempo total decorrido da rodada.

\begin{table}[htbp]\centering\small
\caption{Duração registrada dos ajustes executados na rodada Bahia. Tempos em segundos; médias e medianas por ajuste, sem os 54 reaproveitados.}
\label{tab:fit_time}
\begin{tabular}{@{}lrrr@{}}\toprule
Política & Ajustes & Mediana (s) & Média (s)\\\midrule
'''
for policy,row in timing.iterrows():
    table+=f"{policy.replace('-', '--')} & {int(row['count'])} & {row['median']:.2f} & {row['mean']:.2f}".replace('.',',')+r'\\'+'\n'
table+=r'''\bottomrule\end{tabular}\end{table}

O tempo deve ser interpretado junto ao desempenho: maior esforço não comprova melhor recomendação. GA--DQN reúne busca, demonstrações e aprendizado, enquanto as referências analíticas têm ajuste simples. Uma comparação causal de eficiência exigiria controlar hardware, carga, orçamento e repetições. Os tempos registrados permitem descrever a execução, mas não substituem esse desenho.
'''
(DOC/'capitulos/resultados.tex').write_text(r'''\mychapter{Resultados e Discussão}{cap:results}\lhead{RESULTADOS E DISCUSSÃO}

Este capítulo organiza as evidências pelas perguntas de pesquisa. A mesma matriz permite examinar a qualidade das candidatas, o valor da escolha contextual, sua estabilidade e o esforço computacional. Bahia e M5 são bases separadas da avaliação, não versões concorrentes do experimento. A Bahia está completa; o M5 permanece em execução e não recebe conclusões quantitativas parciais.

\input{bahia_20261015}

'''+table+r'''
\section{Generalização e componentes ainda não avaliados}
\label{sec:resultados_m5}

O M5 amplia a investigação para uma fonte pública de vendas. A execução utiliza 2.408 identificadores de alimentos, com elegibilidade determinada no prefixo de cada origem. A conclusão deverá apresentar custo, atendimento, escolhas e incerteza por base, sem comparar custos absolutos entre conjuntos. Enquanto a matriz não estiver completa, não é possível afirmar que o benefício observado na Bahia se mantém no M5.

A previsão e o PSE supervisionado integram a proposta ampla do AIPE, mas não são intervenções validadas pela comparação entre seleção global e por perfil. Sua eficácia exige desenho próprio, com características e rótulos produzidos antes do teste, referências competitivas e métricas operacionais. Resultados de desenvolvimento ficam preservados nos arquivos de pesquisa e não funcionam como evidência adicional desta matriz.

\section{Alcance e limitações}
\label{sec:threats}

A escolha por perfil melhora o atendimento agregado na Bahia com custo maior; o efeito concentra-se em uma origem. Isso responde à pergunta sobre valor incremental sob as condições avaliadas, sem demonstrar uma vantagem universal. A restrição média não assegura atendimento individual ou ponderado pelo volume. As duas janelas compartilham um ciclo e repetem séries, e a coorte foi definida retrospectivamente.

As vendas podem omitir demanda censurada por falta de estoque. Custos são unidades de cenário, e o estado inicial comum depende de uma política de referência. Prazo, custos relativos, crédito terminal e extensão do horizonte podem alterar os resultados. Não foram avaliadas todas essas sensibilidades. O intervalo por loja é condicional aos controladores e escolhas fixos; não mede incerteza de novos treinamentos ou novas realizações da demanda.

O fechamento da investigação depende da consolidação M5 e das análises previstas no Capítulo~\ref{cap:plano_continuidade}. Questões que exigem outros desenhos permanecem explicitamente abertas, sem preencher lacunas com valores de configurações diferentes.
''',encoding='utf-8')

p=DOC/'capitulos/conclusoes.tex'
s=p.read_text(encoding='utf-8')
start=s.index('O desempenho relativo das políticas')
end=s.index('Os 4.350 ajustes',start)
s=s[:start]+'O desempenho das candidatas e o benefício de selecionar entre elas são questões distintas. '+s[end:]
s=s.replace('e contrastá-la com resultados históricos públicos do M5','e preparar sua avaliação complementar no conjunto público M5')
s=s.replace('Na nova Bahia','Na Bahia').replace('saídas da nova execução','saídas da avaliação').replace('da nova execução','da avaliação')
start=s.index('As matrizes históricas preservam')
end=s.index('Custos simulados',start)
s=s[:start]+'O ajuste utiliza informação passada e sementes independentes de treinamento, que não equivalem a novas lojas ou novas realizações de demanda. '+s[end:]
p.write_text(s,encoding='utf-8')

p=DOC/'capitulos/resumo.tex'
s=p.read_text(encoding='utf-8')
start=s.index('A pesquisa compreende um piloto')
end=s.index('avalia a coorte de 145 séries',start)
s=s[:start]+'A avaliação brasileira '+s[end:]
s=s.replace('a nova execução do M5 constitui uma etapa distinta ainda em andamento','a avaliação complementar no M5 permanece em andamento')
p.write_text(s,encoding='utf-8')

p=DOC/'capitulos/proposta.tex'
s=p.read_text(encoding='utf-8')
start=s.index('O AIPE organiza')
end=s.index('\\input{protocolo_bahia',start)
s=s[:start]+'O AIPE organiza a caracterização da demanda, a comparação de políticas e a avaliação de sua escolha. O protocolo a seguir define a evidência experimental. A formulação geral e o PSE proposto ampliam o escopo da dissertação, com seus componentes ainda não avaliados identificados explicitamente.\n\n'+s[end:]
s=s.replace('\\section{Formulação e instanciação dos estudos históricos}','\\section{Formulação do problema e componentes do framework}')
p.write_text(s,encoding='utf-8')

p=DOC/'capitulos/plano_continuidade.tex'
s=p.read_text(encoding='utf-8').replace('reexecução M5','avaliação M5').replace('A nova Bahia','A Bahia').replace('A nova execução M5','A avaliação M5')
s=s.replace(' Os resultados históricos da Paraíba, Bahia e M5 permanecem identificados pelos respectivos protocolos; correções do código não os transformam em saídas da versão atual.',' As análises complementam a mesma matriz experimental, com cobertura e proveniência verificadas.')
s=s.replace(' O resultado histórico continua relevante como diagnóstico de um cenário em que 91,9\\% das séries não tinham vendas no prefixo configurado.','')
p.write_text(s,encoding='utf-8')
print('Dissertation results unified by question; previous text archived; timing table derived from unique fresh fits.')
