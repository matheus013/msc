import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = json.loads((ROOT/'docs/operations/m5_final_2026-10-17/evidence.json').read_text(encoding='utf-8'))


def paragraph(text, start, replacement):
    begin = text.index(start)
    end = text.find('\n\n', begin)
    return text[:begin] + replacement + text[end:]


p = ROOT/'docs/supply_chain_analytics_article/manuscript_ptbr.tex'
t = p.read_text(encoding='utf-8')
t = t.replace('O M5 amplia a avaliação para diferentes origens e contextos de demanda, com transferência do seletor entre lojas.', 'No M5, prever a classe vencedora aumenta custo e reduz atendimento; prever consequências com recuo à referência reduz o custo médio em 1,96\\% e acrescenta 0,32 ponto percentual de atendimento, sem eliminar deteriorações individuais.')
t = t.replace('O M5 amplia a cobertura de séries e origens para examinar a transferência do seletor entre lojas. As análises distinguem executar o procedimento, melhorar uma recomendação e demonstrar benefício de uso gerencial.', 'No M5, prever consequências com recuo reduz custo e aumenta atendimento médio, enquanto a classificação da família vencedora deteriora essas duas dimensões. As análises distinguem executar o procedimento, melhorar uma recomendação e demonstrar benefício de uso gerencial.')
t = t.replace('O acerto exige serviço mínimo e empate numérico no menor custo; concordância nominal é informada separadamente.', 'O acerto exige serviço mínimo e empate numérico no menor custo; não exige identidade nominal da política.')
begin = t.index('\\input{aipe_bahia_resultados}')
end = t.index('\\section{Discussão}', begin)
t = t[:begin] + r'''\input{resultados_unificados_20261017}
\subsection{Consequências da seleção e da intervenção}
Na Bahia, o perfil acrescenta 3,89 pontos percentuais de atendimento com custo 1,93\% maior que a global (Tabela~\ref{tab:unificado_ba}). Recupera 19 casos abaixo da meta, sem criar novas violações, mas não entrega redução de custo. O preditivo por classe custa 3,26\% mais e perde 3,58 pontos de atendimento. A previsão direta de consequências reduz custo em 1,40\%, com perda de 0,86 ponto; seu recuo conserva a global nos 290 casos. As diferenças mostram que contextualização, previsão e intervenção têm consequências distintas.

No M5, o perfil permanece próximo da global, com custo 0,17\% maior e acréscimo de 0,22 ponto de atendimento (Tabela~\ref{tab:unificado_m5}). O preditivo por classe custa 4,09\% mais e perde 2,90 pontos; a floresta de classe também apresenta maior custo e menor serviço. A previsão direta de consequências reduz custo em 2,44\%, mas perde 0,18 ponto de atendimento. Com recuo, reduz custo em 1,96\% e aumenta atendimento em 0,32 ponto. O atendimento médio resultante é 68,59\%, abaixo de 70\%: a melhora frente à referência não equivale ao cumprimento do alvo.

O recuo modifica 679 das 5.326 escolhas M5, frente a 3.629 da previsão direta. A diferença média de custo é $-15{,}25$ unidades do cenário, com intervalo condicional de 95\% $[-20{,}79;-10{,}01]$; a de serviço é $0{,}32$ ponto, com intervalo $[0{,}23;0{,}43]$ (Figura~\ref{fig:unificado_diferencas}). Há economia em 560 casos, mas também 97 casos com custo maior e 149 com atendimento menor. São 19 novas violações do alvo e 43 recuperações, elevando os casos viáveis de 2.448 para 2.472. A maior economia individual corresponde a 1,84\% das economias positivas. Esses resultados caracterizam benefício médio acompanhado de deteriorações, sem garantia por série.

\subsection{Distância às referências retrospectivas}
A previsão por consequências com recuo iguala o menor custo viável do portfólio em 417 dos 2.472 casos M5 comparáveis, com eficiência média condicional de 69,77\%. Os conjuntos comparáveis dependem da recomendação satisfazer o alvo; suas médias não constituem comparação no mesmo subconjunto. Das 5.326 instâncias com informação perfeita, todas têm otimalidade comprovada. Na variante com recuo, 3.005 realizações de treinamento permitem comparar custo sob serviço compatível: 122 igualam o ótimo determinístico e a eficiência média é 47,56\%. Realizações e casos são denominadores diferentes; sementes não ampliam a população independente. Essas distâncias delimitam oportunidades condicionais, sem estimar economia disponível a uma política que desconhece a demanda futura.

Na decomposição M5 do recuo, 2.472 casos atendem ao alvo, 649 têm alternativa viável não escolhida, 1.586 não têm candidata viável apesar de possibilidade física, 431 apresentam restrição física inicial e 188 não têm demanda (Figura~\ref{fig:unificado_falhas}). Na Bahia, o recuo conserva as 32 falhas de seleção da referência; outras 64 falhas decorrem da cobertura do portfólio e 23 das condições iniciais. A Figura~\ref{fig:unificado_distribuicao} complementa as médias com a dispersão do atendimento individual.

''' + t[end:]
t = paragraph(t, 'A coorte M5 amplia o suporte', 'A coorte M5 amplia o suporte disponível para aprender recomendações, mas mantém somente dez lojas e forte concentração em um perfil. A previsão de consequências com recuo melhora custo e atendimento médios, enquanto os classificadores avaliados deterioram essas dimensões. Aumento do número de séries, diversidade do contexto e escolha do alvo de aprendizagem são questões diferentes. O controle por floresta de classe torna a comparação mais informativa, mas não constitui ablação de todas as escolhas de modelo e regularização.')
t = paragraph(t, 'A regra de intervenção torna explícito', 'A regra de intervenção torna explícito quanto o PSE modifica a referência e com que consequência. Na Bahia, a variante por consequências com recuo reproduz a global nos 290 casos; no M5, intervém em 679 casos e melhora as médias, acompanhada de novas violações individuais. Conservar a referência é uma saída admitida quando a validação interna não justifica substituí-la, sem demonstrar valor adicional do preditivo. A intervenção seletiva apresenta utilidade na instanciação M5 avaliada, sem garantir preservação futura do atendimento.')
t = paragraph(t, 'A avaliação delimita onde essas escolhas ajudam.', 'A avaliação delimita onde essas escolhas ajudam. Na Bahia, o perfil acrescenta 3,89 pontos percentuais de atendimento com aumento de custo de 1,93\\%, enquanto o recuo por consequências conserva a referência. No M5, a estimação por consequências com recuo reduz custo em 1,96\\% e acrescenta 0,32 ponto de atendimento; a classificação da família vencedora deteriora ambos. Os ganhos médios coexistem com novas falhas individuais e não atingem atendimento médio de 70\\%. A evidência sustenta a execução, a utilidade diagnóstica e um benefício operacional condicionado à instanciação, sem demonstrar vantagem universal ou economia financeira realizada.')
p.write_text(t,encoding='utf-8')

p = ROOT/'docs/master_proposal/capitulos/resultados.tex'
t = p.read_text(encoding='utf-8')
t = t.replace('\\input{bahia_20261015}\n\\input{m5_pse_20261015}\n\\input{consequencias_20261015}', '\\input{resultados_unificados_20261017}\n\\input{diagnosticos_unificados_20261017}')
start = t.index('\\section{Esforço computacional das políticas}')
end = t.index('\\section{Síntese das respostas e limites}', start)
rows=[]
for ds,label in [('BA','Bahia'),('M5','M5')]:
    for row in EVIDENCE['datasets'][ds]['fit_times_nonreused']:
        def fmt(x):
            return f'{x:.5f}'.replace('.',',') if x < .01 else f'{x:.2f}'.replace('.',',')
        rows.append(f"{label} & {row['policy'].replace('GA-DQN','GA--DQN')} & {row['count']:,}".replace(',', '.') + f" & {fmt(row['median'])} & {fmt(row['mean'])}\\\\")
timing = r'''\section{Esforço computacional das políticas}
\label{sec:computational_results}
A Tabela~\ref{tab:fit_time} descreve os ajustes efetivamente executados nas filas consolidadas. Cada ajuste é contado uma vez, sem duplicar validação e teste. Os ajustes reaproveitados são excluídos. Na Bahia, apenas a CappedBaseStock foi reajustada nessa fila; os demais controladores foram incorporados após auditoria. No M5, a amostra contém ajustes novos das sete políticas. As contagens descrevem esforço, sem ampliar a população de séries.

\begin{table}[htbp]\centering\small
\caption{Duração registrada dos ajustes não reaproveitados. Tempos em segundos; execução sob concorrência e carga variável.}\label{tab:fit_time}
\begin{tabular}{@{}llrrr@{}}\toprule
Base & Política & Ajustes & Mediana (s) & Média (s)\\\midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}\end{table}

As referências analíticas têm ajuste simples; GA--DQN inclui busca, demonstrações e treinamento da rede. Maior esforço não comprova recomendação melhor. Esses registros não constituem teste controlado de velocidade entre processadores e não equivalem ao tempo total decorrido. Comparar causalmente eficiência computacional exigiria controlar hardware, carga, orçamento e repetições.

'''
t = t[:start] + timing + t[end:]
t = paragraph(t, 'A contribuição metodológica está', 'No M5, a seleção por perfil permanece próxima da referência, enquanto o preditivo por classe aumenta custo em 4,09\\% e reduz atendimento em 2,90 pontos percentuais. A previsão direta de consequências reduz custo e perde atendimento; sua variante com recuo reduz custo em 1,96\\% e acrescenta 0,32 ponto. A comparação distingue aprender uma vencedora, estimar consequências e decidir quando intervir. Os ganhos médios coexistem com 19 novas violações e 43 recuperações do alvo: o número de casos viáveis aumenta, sem garantia individual.\n\nA contribuição metodológica está em explicitar essas escolhas, submetê-las à mesma comparação operacional e localizar limitações. Conservar a referência é uma saída admitida; não demonstra que o preditivo acrescentou valor. Economia com atendimento menor também não satisfaz automaticamente o objetivo do gerente. Magnitude, incerteza, deteriorações e suporte das recomendações precisam acompanhar as médias.')
t = paragraph(t, 'O oráculo do portfólio e o planejamento', 'O oráculo do portfólio e o planejamento com informação perfeita delimitam problemas diferentes. No M5, o recuo iguala o custo viável do portfólio em 417 dos 2.472 casos comparáveis, com eficiência média condicional de 69,77\\%. Frente ao ótimo determinístico, são 122 acertos em 3.005 realizações de treinamento comparáveis, com eficiência média de 47,56\\%. Casos, realizações e subconjuntos condicionais não são intercambiáveis. As Tabelas~\\ref{tab:unificado_oraculo_m5} e~\\ref{tab:unificado_teorico_m5} preservam os denominadores de cada implementação.')
t = t.replace('O módulo opcional de atributos adicionais permanece separado desta consolidação.', 'O módulo opcional de atributos adicionais é avaliado separadamente desta matriz de controladores e seletores.')
p.write_text(t,encoding='utf-8')

p = ROOT/'docs/master_proposal/capitulos/conclusoes.tex'
t = p.read_text(encoding='utf-8')
t = paragraph(t, 'A avaliação M5 amplia o exame', 'No M5, a previsão de consequências com recuo reduz o custo médio em 1,96\\% e acrescenta 0,32 ponto percentual de atendimento frente à global. A regra modifica 679 dos 5.326 casos, com 19 novas violações e 43 recuperações do alvo. Os intervalos condicionais favorecem essas diferenças médias na coorte avaliada, mas a existência de perdas individuais impede alegar preservação geral do serviço. O atendimento médio é 68,59\\%, abaixo do alvo de 70\\%. Já a classificação da família vencedora aumenta custo e reduz atendimento. Assim, a avaliação sustenta utilidade de uma instanciação específica, sem demonstrar superioridade de toda abordagem contextual.')
t = paragraph(t, 'O diagnóstico retrospectivo distingue limitações', 'O diagnóstico retrospectivo distingue limitações sob as hipóteses do simulador, sem estabelecer causas das rupturas observadas na empresa. No M5, o recuo mantém 649 falhas com alternativa viável, 1.586 casos sem candidata viável apesar de possibilidade física e 431 com restrição física inicial; outros 188 não têm demanda. Melhorar o seletor não resolve toda falta. A eficiência média é 69,77\\% frente ao portfólio nos 2.472 casos comparáveis e 47,56\\% frente ao planejamento perfeito nas 3.005 realizações de treinamento comparáveis. Os denominadores distintos delimitam cada medida, sem converter essa distância em economia disponível a uma política sem informação futura.')
p.write_text(t,encoding='utf-8')

p=ROOT/'docs/master_proposal/capitulos/resumo.tex'
t=p.read_text(encoding='utf-8')
t=t.replace('Oráculos distinguem erros de escolha, insuficiência do portfólio e inviabilidade física.', 'No M5, a previsão por consequências com recuo reduz o custo médio em 1,96\\% e acrescenta 0,32 ponto percentual de atendimento, embora produza deteriorações individuais e mantenha atendimento médio abaixo de 70\\%. Oráculos distinguem erros de escolha, insuficiência do portfólio e inviabilidade física.')
p.write_text(t,encoding='utf-8')
p=ROOT/'docs/master_proposal/abstract_en.tex'
t=p.read_text(encoding='utf-8')
t=t.replace('Oracles distinguish selection errors, portfolio gaps, and physical infeasibility.', 'In M5, prediction of consequences with fallback reduces mean cost by 1.96\\% and adds 0.32 percentage points of service, although individual deterioration occurs and mean service remains below 70\\%. Oracles distinguish selection errors, portfolio gaps, and physical infeasibility.')
p.write_text(t,encoding='utf-8')
print('Sínteses finais M5, novos inputs, diagnósticos e tempos incorporados.')
