"""Apply the evidence-based fine review; keep a pre-review source snapshot."""
from pathlib import Path
import json
import re
import shutil

ROOT=Path(__file__).resolve().parents[4]
DOC=ROOT/'docs/supply_chain_analytics_article'
REVIEW=Path(__file__).resolve().parent
if '\\label{tab:implemented_policies}' in (DOC/'protocolo_bahia_20261015.tex').read_text(encoding='utf-8'):
    raise SystemExit('One-time review already applied; edit current sources directly.')
archive=REVIEW/'before'
archive.mkdir(exist_ok=True)
for name in ['manuscript_ptbr.tex','manuscript_ptbr.pdf','protocolo_bahia_20261015.tex',
             'bahia_20261015.tex','references_ptbr.bib']:
    if not (archive/name).exists(): shutil.copy2(DOC/name,archive/name)

p=DOC/'manuscript_ptbr.tex'
s=p.read_text(encoding='utf-8')
abstract=r'''A escolha de políticas por perfil de demanda acrescenta complexidade à reposição, mas seu valor depende do desempenho em períodos posteriores. Comparamos essa estratégia com a seleção de uma política global em 145 séries loja--produto de uma rede varejista brasileira. Sete candidatas são avaliadas em duas origens temporais, com treino, validação e teste separados e três sementes nos métodos estocásticos, totalizando 4.350 ajustes. Cada seletor minimiza custo entre candidatas com atendimento médio de pelo menos 0,70 na validação das demais lojas; a avaliação usa o teste da loja reservada. A seleção por perfil apresenta custo médio 1,93\% maior e atendimento 3,89 pontos percentuais superior. Os intervalos condicionais de 95\% são, respectivamente, $[-0{,}32;4{,}16]\%$ e $[2{,}78;5{,}07]$ pontos percentuais. Os casos abaixo da meta diminuem de 119 para 100 entre 283 casos série--origem com demanda positiva; sete casos têm atendimento indefinido. A diferença concentra-se na primeira origem; na segunda, os seletores fazem escolhas idênticas. O resultado explicita um compromisso entre custo e serviço, sem estabelecer economia geral. A avaliação combina diferenciação de políticas e restrição por grupo, sem isolar o efeito da representação. Coorte retrospectiva, janelas curtas parcialmente sobrepostas e inferência condicional limitam a generalização.'''
s=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _: '\\begin{abstract}\n'+abstract+'\n\\end{abstract}',s,flags=re.S)
s=re.sub(r'\\begin\{keyword\}.*?\\end\{keyword\}',lambda _:r'''\begin{keyword}
Inventory control \sep Policy selection \sep Demand heterogeneity \sep Service level \sep Retail
\end{keyword}''',s,flags=re.S)
a=s.index('Essa pergunta pode ser formulada');b=s.index('Trabalhos publicados',a)
s=s[:a]+r'''Formulamos a escolha como um problema de seleção de algoritmos: características de uma instância ajudam a escolher um método de um conjunto candidato \cite{rice1976,Kotthoff2014AlgorithmSelection}. Em inventário, a instância é uma série loja--produto, e a recomendação depende do custo e do serviço exigido. A referência global aplica a mesma família de política a todas as séries, com parâmetros ajustados individualmente. O seletor por perfil pode mudar essa família entre grupos. Portanto, a comparação mede o valor da diferenciação da escolha, mantendo a calibração por série em ambas as estratégias.

'''+s[b:]
s=s.replace('A avaliação combina dados de uma rede brasileira e um conjunto público de varejo, mantendo as bases separadas. Neste estágio, a matriz da Bahia está completa e a execução M5 está em andamento; apenas resultados completos sustentam conclusões quantitativas.','A evidência reportada corresponde à matriz completa da Bahia. O conjunto público M5 integra a avaliação complementar em execução e não fornece resultados quantitativos neste manuscrito.')
s=s.replace('O estudo não pretende abranger todos os componentes do AIPE nem validar seu seletor supervisionado.','A contribuição é uma avaliação empírica do valor incremental dessa escolha em teste temporal, com perfis calculados no treino, exclusão da loja da seleção e análise da insuficiência de atendimento individual.')
a=s.index('Tadayonrad e Ndiaye');b=s.index('Zabraoui et al.',a)
s=s[:a]+r'''Integrar classificação e controle já é um objetivo estabelecido. Mohammaditabar et al.\ \cite{Mohammaditabar2012Integrated} propõem escolher grupos e políticas conjuntamente, com solução por recozimento simulado. Sheikh-Zadeh et al.\ \cite{SheikhZadeh2021PBIC} desenvolvem classificação orientada ao desempenho para um sistema de múltiplos itens e escalões. Esses trabalhos impedem tratar a simples associação entre perfil e política como novidade. Nosso recorte é diferente: regras de perfil fixas e seleção de candidatas já ajustadas são confrontadas com uma referência global, usando validação em outras lojas e teste temporal na loja reservada. Não implementamos os métodos desses autores como comparadores e não alegamos superioridade sobre eles.

Tadayonrad e Ndiaye \cite{sca_f81ad2b3e023} aproximam a avaliação de previsão dos custos e do serviço de inventário. Essa ligação sustenta o uso de consequências operacionais como critério: neste estudo, precisão preditiva não substitui custo e atendimento. A política de estoque limitado CappedBaseStock acrescenta uma referência motivada pela literatura de vendas perdidas \cite{Xin2021CappedBaseStock}; propriedades teóricas dessa família não são garantias para nossa calibração e cenário.

'''+s[b:]
a=s.index('O M5 documenta uma competição');b=s.index('\\input{protocolo_bahia',a)
s=s[:a]+s[b:]
a=s.index('Para um conjunto $A$');b=s.index('Seja $\\Pi$',a)
s=s[:a]+r'''Para um conjunto $A$ de casos série--origem usados na escolha, seja $A^+$ o subconjunto com demanda positiva no bloco correspondente. Definimos
\begin{equation}
 \bar C_{\pi,A}=\frac{1}{|A|}\sum_{i\in A}C_{i\pi},\qquad
 \bar F_{\pi,A}=\frac{1}{|A^+|}\sum_{i\in A^+}F_{i\pi}.
\end{equation}
Todos os casos entram no custo; somente os casos com atendimento definido entram em sua média. Na validação da Bahia não há denominadores nulos; no teste, eles são contabilizados separadamente. Cada caso definido recebe o mesmo peso, sem ponderação pelo volume demandado. '''+s[b:]
s=s.replace('Um perfil ausente nas lojas de escolha utiliza a regra global.','Um perfil sem informação válida de atendimento nas lojas de escolha utiliza a regra global e recebe marcação de contingência.')
s=s.replace('A seleção por perfil não precisa apresentar menor custo mesmo na avaliação retrospectiva.','A seleção por perfil não precisa apresentar menor custo mesmo na avaliação retrospectiva.')
needle='Uma política global viável pode ser inviável em um grupo; a escolha por perfil não precisa apresentar menor custo mesmo na avaliação retrospectiva.'
s=s.replace(needle,needle+' A comparação avalia duas estratégias operacionais com diferentes formas de distribuir a exigência de serviço. Ela não isola o efeito dos descritores de demanda de um efeito de impor a restrição em cada grupo.')
s=s.replace('herdado da investigação original','fixado no cenário experimental')
s=s.replace('Avaliamos também 0,80, 0,85, 0,90 e 0,95, sem escolher a conclusão pelo limiar mais favorável.','Avaliamos também 0,80, 0,85, 0,90 e 0,95 na etapa de seleção, sem retreinar o portfólio para cada meta. A calibração das candidatas mantém referência de 0,70; a sensibilidade descreve escolhas entre essas candidatas, não a melhor política possível para cada limiar.')
s=s.replace('A diferença relativa de custo é','O aumento relativo de custo é')
s=s.replace(r'R=100\frac{\bar C_{\mathrm{global}}-\bar C_{\mathrm{perfil}}}{\bar C_{\mathrm{global}}}.',r'\Delta_C=100\frac{\bar C_{\mathrm{perfil}}-\bar C_{\mathrm{global}}}{\bar C_{\mathrm{global}}}.')
s=s.replace('$R>0$ indica redução; $R<0$, aumento.','$\\Delta_C>0$ indica aumento; $\\Delta_C<0$, redução.')
s=s.replace('Os nomes dos perfis identificam regras locais; não validam sazonalidade ou impacto econômico. ','')
s=s.replace('A incerteza de custo é descrita','A incerteza das diferenças de custo e atendimento é descrita')
s=s.replace('A dependência entre lojas que compartilham produtos permanece uma limitação.','Para atendimento, as somas e os denominadores positivos são recalculados em cada sorteio. A dependência entre lojas que compartilham produtos permanece uma limitação.')
needle='Não há evidência de economia geral que justifique, por si só, a complexidade adicional da seleção contextual.'
s=s.replace(needle,'O aumento de custo é uma estimativa pontual cujo intervalo condicional inclui zero; isso não comprova igualdade de custos. O intervalo do atendimento permanece positivo sob a reamostragem adotada, sem estabelecer que o efeito se repetirá em novas trajetórias.')
needle='Uma decisão empresarial precisa definir se a meta se aplica à média das séries, às unidades demandadas ou a itens críticos individualmente.'
s=s.replace(needle,needle+' Para o gestor, o resultado oferece uma decisão concreta: avaliar se o serviço adicional compensa o custo e o uso de regras diferenciadas. Essa avaliação exige calibrar custos e criticidade comercial; o limiar de 0,70 do cenário não deve ser convertido em meta operacional por conveniência.')
needle='Essas condições precisam acompanhar a interpretação dos resultados.'
s=s.replace(needle,needle+' Além disso, a calibração de CappedBaseStock incorpora penalidades de estabilidade que não são usadas no GA, e as candidatas não têm esforço computacional total equivalente. A matriz permite comparar o portfólio especificado, sem atribuir os contrastes apenas à família algorítmica.')
s=s.replace('A execução M5 deverá completar a avaliação da mesma pergunta em uma base pública, mantendo separadas as estimativas por conjunto.','O M5 é um conjunto público de vendas de varejo associado a uma competição de previsão \\cite{Makridakis2022M5Accuracy}. Sua execução de reposição deverá avaliar a mesma pergunta, com preparação e elegibilidade documentadas e estimativas separadas por base. A execução em andamento ainda não demonstra generalização.')
s=s.replace('O PSE supervisionado permanece uma extensão a validar frente às regras explícitas de seleção e à melhor referência global.','Um seletor supervisionado é uma extensão possível, mas sua validação não é necessária para interpretar a comparação entre as duas regras explícitas aqui avaliadas.')
s=s.replace('A consolidação do M5 e a validação do PSE são necessárias antes de generalizar a recomendação para outros contextos.','A avaliação complementar no M5 e em outras janelas é necessária para examinar a extensão desse resultado a outros contextos. Diferenciar grupos só terá utilidade operacional se o benefício se mantiver frente a uma referência global competitiva e justificar seu custo.')
a=s.index('Os dados transacionais brasileiros');b=s.index('\\section*{Autoria',a)
s=s[:a]+r'''Os dados transacionais brasileiros são proprietários e não são redistribuídos. Código, protocolo, scripts de análise e fontes do manuscrito estão disponíveis em \url{https://github.com/matheus013/msc}. Dados, bancos de execução e modelos não integram a versão atual desse repositório. Uma versão arquivada e identificável do pacote de reprodução, bem como os derivados cuja divulgação seja autorizada, deverão acompanhar a submissão. Não se promete acesso aos registros restritos mediante solicitação.
'''+s[b:]
p.write_text(s,encoding='utf-8')

# The protocol describes the actual controllers, including a calibration asymmetry.
p=DOC/'protocolo_bahia_20261015.tex'
s=p.read_text(encoding='utf-8')
s=s.replace('O limiar principal continua em 0,70.','O limiar principal é 0,70.')
s=s.replace('A avaliação utiliza 145 séries loja-produto da Bahia, em 115 lojas e nove produtos, com 38 ciclos comerciais disponíveis.','A avaliação utiliza 145 séries loja--produto da Bahia, em 115 lojas e nove produtos, com 38 ciclos comerciais disponíveis, identificados de 202301 a 202504. Os ciclos são unidades comerciais da fonte; não representam observações diárias.')
marker='\\begin{figure}[htbp]'
i=s.index(marker)
extra=r'''\subsection{Sistema de estoque, métricas e candidatas}

Cada série é simulada separadamente em um sistema de vendas perdidas: demanda não atendida no ciclo não se transforma em pedido pendente. Não há estoque compartilhado entre lojas, capacidade conjunta ou transbordo. A política decide com o estado anterior ao recebimento do ciclo; a entrega respeita o prazo exato de dois ciclos. O recebimento precede a demanda, e manutenção incide sobre o saldo após o atendimento.

Para o teste de sete ciclos, o custo é
\begin{equation}
C=\sum_t\left[hI_t+p(d_t-v_t)+K\mathbf{1}(q_t>0)+cq_t\right],
\end{equation}
em que $I_t$ é o estoque após a demanda, $d_t$ a demanda observada, $v_t$ a quantidade atendida e $q_t$ o pedido emitido. Os coeficientes são $h=1$, $p=5$, $K=50$ e $c=0{,}5$. Pedidos em trânsito não atendem demanda antes da chegada. O custo reportado não contém as penalidades auxiliares usadas no treino.

\begin{table}[htbp]\centering\small
\caption{Candidatas e decisões implementadas. Parâmetros e escalas usam somente o treino da própria série.}
\label{tab:implemented_policies}
\begin{tabular}{@{}p{2.7cm}p{8.1cm}@{}}\toprule
Candidata & Regra e ajuste\\\midrule
EOQCycle & Lote fixo ao atingir o ponto de pedido; lote EOQ por ciclo e proteção com $z=1{,}28$.\\
sSCycle & Mesmo ponto de pedido, com reposição até o ponto acrescido do lote EOQ; parâmetros analíticos, sem otimização exata de $(s,S)$.\\
CappedBaseStock & Reposição até um alvo com teto de quantidade; busca determinística em grade de $20\times12$ pares.\\
GA & Algoritmo genético ajusta ponto de pedido, lote e proteção adicional; população 100 e 50 gerações.\\
DQN & Rede Q profunda, variante Double DQN com arquitetura dueling, para escolher entre 20 quantidades discretas.\\
PPO & Otimização proximal de políticas, com 20 ações discretas e estimação generalizada de vantagem.\\
GA--DQN & DQN orientado pelo gatilho genético e por demonstrações; não é reprodução exata de Zabraoui et al.\\
\bottomrule\end{tabular}
\end{table}

A Tabela~\ref{tab:implemented_policies} identifica as implementações que compõem o portfólio. Nas referências analíticas, $Q=\max(1,\sqrt{2\mu K/h})$ e $r=2\mu+1{,}28\sigma\sqrt{2}$, em que $\mu$ e $\sigma$ são média e desvio-padrão populacional do treino. EOQCycle pede $Q$ quando a posição de estoque não excede $r$; sSCycle repõe até $r+Q$. CappedBaseStock usa $q_t=\min(u,\max(0,S-\mathrm{IP}_t))$, com alvo $S$, teto $u$ e posição de estoque $\mathrm{IP}_t$.

DQN e PPO usam duas camadas de 64 unidades. O estado tem seis componentes: estoque, previsão ingênua, pedidos em trânsito, média e dispersão dos últimos seis ciclos e posição no horizonte. As escalas vêm do treino e o estado usa apenas observações passadas. No híbrido, a quantidade é zero acima do gatilho genético; abaixo dele, é o máximo entre a ação da rede e metade do lote genético. Essa transformação é mantida em demonstração, treino e aplicação.

Os objetivos de calibração não são idênticos. GA minimiza custo acrescido de penalidade pelo déficit de atendimento frente a 0,70. Os agentes usam recompensa de custo negativo e penalidade terminal pelo mesmo déficit. A escala da penalidade é $30\max(5\sum_t d_t,1)$. CappedBaseStock acrescenta termos de frequência de ruptura e amplificação da variância dos pedidos, com peso unitário, que não são usados pelo GA. Essa diferença precisa acompanhar a leitura do ranking: a avaliação compara as candidatas configuradas, sem isolar a qualidade de suas famílias ou igualar todos os esforços de ajuste.

'''
s=s[:i]+extra+s[i:]
p.write_text(s,encoding='utf-8')

p=DOC/'bahia_20261015.tex'
s=p.read_text(encoding='utf-8')
s=s.replace('Não há redução de custo, mas há melhora de serviço, concentrada na primeira origem.','A estimativa pontual combina custo maior e melhor serviço, com a diferença concentrada na primeira origem.')
s=s.replace('O intervalo inclui zero e não representa incerteza de novos treinamentos ou de novas trajetórias de demanda; dependência entre lojas que compartilham produtos também permanece possível.','Para a diferença de atendimento, o mesmo procedimento produz $[2{,}78;5{,}07]$ pontos percentuais. O intervalo de custo inclui zero; o de atendimento permanece positivo. Ambos são condicionais e não representam incerteza de novos treinamentos ou de novas trajetórias de demanda; dependência entre lojas que compartilham produtos também permanece possível.')
a=s.index('A sensibilidade reforça');b=s.index('\\begin{figure}',a)
s=s[:a]+r'''\subsection{Sensibilidade ao limiar de escolha}

A Tabela~\ref{tab:threshold_sensitivity} apresenta todos os limiares avaliados. Em 0,80, a estratégia por perfil tem atendimento médio ligeiramente menor e custo maior que a global. Em 0,85, ambas selecionam sSCycle nos 290 casos, embora a frequência de contingência seja diferente: uma escolha pode cumprir o critério de seu grupo e não cumprir o critério agregado, ou o contrário. Em 0,90 e 0,95, ambas usam contingências em todos os casos. Portanto, a diferenciação contextual não preserva a vantagem de atendimento em todas as metas.

\begin{table}[htbp]\centering\small
\caption{Sensibilidade da seleção, sem retreinamento. Custo em 290 casos; atendimento médio em 283. Contingências indicam ausência de candidata viável na validação e não cumprimento de serviço no teste. G: global; P: por perfil.}
\label{tab:threshold_sensitivity}
\begin{tabular}{@{}rrrrrr@{}}\toprule
Meta & Custo G & Custo P & Atend. G & Atend. P & Conting. G/P\\\midrule
0,70 &161,31&164,41&0,7362&0,7751&0/0\\
0,80 &164,33&164,53&0,8037&0,8025&145/223\\
0,85 &164,23&164,23&0,8037&0,8037&290/225\\
0,90 &164,23&164,23&0,8037&0,8037&290/290\\
0,95 &164,23&164,23&0,8037&0,8037&290/290\\
\bottomrule\end{tabular}
\end{table}

As candidatas continuam calibradas com referência de 0,70. A inexistência de uma opção viável em uma meta maior não demonstra impossibilidade de atendê-la com outro ajuste ou portfólio. A sensibilidade examina a regra de seleção sobre a matriz disponível.

'''+s[b:]
s=s.replace('Contingências indicam ausência de candidata viável na validação e não cumprimento de serviço no teste.','Contingências indicam ausência de candidata viável na validação; não informam cumprimento de serviço no teste.')
p.write_text(s,encoding='utf-8')

p=DOC/'references_ptbr.bib'
s=p.read_text(encoding='utf-8')
extra=r'''
@article{Mohammaditabar2012Integrated,
 author={Mohammaditabar, Davood and Ghodsypour, Seyed Hassan and O'Brien, Chris},
 title={Inventory control system design by integrating inventory classification and policy selection},
 journal={International Journal of Production Economics}, year={2012},
 volume={140}, number={2}, pages={655--659}, doi={10.1016/j.ijpe.2011.03.012}
}
@article{SheikhZadeh2021PBIC,
 author={Sheikh-Zadeh, Alireza and Rossetti, Manuel D. and Scott, Marc A.},
 title={Performance-based inventory classification methods for large-scale multi-echelon replenishment systems},
 journal={Omega}, year={2021}, volume={101}, pages={102276}, doi={10.1016/j.omega.2020.102276}
}
@article{Xin2021CappedBaseStock,
 author={Xin, Linwei},
 title={Technical Note---Understanding the Performance of Capped Base-Stock Policies in Lost-Sales Inventory Models},
 journal={Operations Research}, year={2021},volume={69},number={1},pages={61--70},
 doi={10.1287/opre.2020.2019}
}
'''
s+=extra
p.write_text(s,encoding='utf-8')
print('Fine review applied; seven controllers specified; service CI and full threshold sensitivity included.')
