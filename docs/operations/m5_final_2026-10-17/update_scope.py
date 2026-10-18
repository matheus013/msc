from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def swap(text, old, new):
    if old not in text:
        raise ValueError('Passagem não encontrada: ' + old[:100])
    return text.replace(old, new)


def paragraph(text, start, replacement):
    begin = text.index(start)
    end = text.find('\n\n', begin)
    return text[:begin] + replacement + text[end:]


p = ROOT / 'docs/supply_chain_analytics_article/manuscript_ptbr.tex'
t = p.read_text(encoding='utf-8')
t = swap(t, 'Na Bahia, empate no custo viável prefere maior serviço e depois identificador; no M5, prefere identificador. Contingências empatadas em serviço preferem menor custo. Essas convenções são preservadas por base e registradas no código.', 'Empates no custo viável são resolvidos pelo identificador da política. Contingências empatadas em serviço preferem menor custo e depois identificador. A mesma convenção é aplicada às duas bases.')
t = swap(t, 'A Bahia investiga a primeira pergunta; o piloto M5 permite examinar as três, incluindo a substituição do preditivo por classe por um seletor de consequências na mesma matriz de controladores.', 'As duas bases permitem examinar as três perguntas, incluindo a substituição do preditivo por classe por um seletor de consequências na mesma matriz de controladores.')
t = paragraph(t, 'O M5 é uma fonte pública', 'O M5 é uma fonte pública de vendas associada a uma competição de previsão \\cite{Makridakis2022M5Accuracy}. A avaliação utiliza uma coorte histórica local de 2.408 identificadores loja--produto da categoria de alimentos (\\texttt{FOODS}), nas dez lojas de Califórnia, Texas e Wisconsin. Os códigos CA, TX e WI identificam esses estados; por exemplo, CA\\_2 identifica uma loja da Califórnia. Usamos ciclos completos de 21 dias, sem remover picos por regras baseadas no intervalo interquartil. A elegibilidade é reavaliada no prefixo de cada origem: 885 séries aos 50 ciclos, 2.033 aos 64 e 2.408 aos 78.')
t = swap(t, 'Dimensão & Bahia & Piloto M5', 'Dimensão & Bahia & M5')
t = swap(t, 'Séries / lojas & 145 / 115 & 30 / 10', 'Séries / lojas & 145 / 115 & 2.408 / 10')
t = swap(t, 'Casos série--origem & 290 & 90', 'Casos série--origem & 290 & 5.326')
t = swap(t, 'Ajustes de políticas & 4.350 & 1.350', 'Ajustes de políticas & 4.350 & 79.890')
t = swap(t, 'Testes sem demanda & 7 & 4', 'Testes sem demanda & 7 & 188')
t = swap(t, 'Seletores & Global e por perfil & Global, por perfil e preditivo', 'Seletores & Perfil e estimação & Perfil e estimação')
t = paragraph(t, 'A elegibilidade exige três ciclos positivos', 'A elegibilidade exige três ciclos positivos no prefixo, dentro dos identificadores históricos disponíveis. O recorte não representa todas as séries FOODS do M5. É inspirado no contexto de Zabraoui et al., sem afirmar identidade com sua amostra. As 5.326 observações série--origem repetem identificadores: custo usa todos os casos e atendimento médio os 5.138 com demanda positiva no teste. Há 69, 106 e 13 casos sem demanda nas respectivas origens. Um piloto anterior de 30 séries orientou o desenvolvimento dos seletores; a ampliação não constitui, por si, confirmação independente.')
t = paragraph(t, 'A CappedBaseStock minimiza custo', 'A CappedBaseStock minimiza custo acrescido da penalidade de atendimento com escala fixa. Os pesos adicionais de ruptura e amplificação de pedidos são zero nas duas bases. Penalidades orientam a calibração sem garantir o alvo; a seleção aplica depois a viabilidade observada na validação. A harmonização conserva a definição do objetivo entre bases, sem tornar suas demandas e condições equivalentes.')
t = swap(t, '\\subsection{Avaliação e desenvolvimento do piloto}', '\\subsection{Avaliação dos seletores}')
t = paragraph(t, 'No M5, XGBoost tem', 'O preditivo por classe utiliza XGBoost com 200 árvores, profundidade três, taxa 0,05, sem subamostragem e com semente fixa. Imputação, codificação categórica e definição das classes são ajustadas dentro de cada partição de treinamento. Classes raras não são removidas; treinos de classe única usam classificador constante. A ablação sem perfil mantém os demais componentes. O tamanho de treino depende da origem e da loja excluída; não se usam origens posteriores para recomendar em uma origem anterior.')
t = paragraph(t, 'Uma implementação com recuo usa', 'O piloto anterior foi usado para desenvolver alternativas de aprendizagem e recuo. Na avaliação consolidada, as implementações são aplicadas às matrizes completas das duas bases. As recomendações são gravadas antes da pontuação no teste, mas esse procedimento não elimina o uso prévio do piloto no desenvolvimento \\cite{CawleyTalbot2010}.')
t = swap(t, 'A comparação reduzida adicional reutiliza a matriz M5 e substitui a implementação do PSE.', 'A comparação por consequências reutiliza cada matriz e substitui a implementação do PSE, sem repetir ajustes de controladores.')
t = paragraph(t, 'A variante com recuo à global exige', 'A variante com recuo à global exige serviço previsto $\\geq0{,}70$, não inferior ao previsto da referência, e economia acima de uma margem em $\\{0;0{,}02;0{,}05;0{,}10;\\infty\\}$. A margem minimiza custo em validação interna, sujeita a serviço médio não inferior à referência. Em cada origem, uma loja externa fica reservada e as demais são divididas em cinco grupos internos de lojas; cada grupo recebe previsões produzidas sem suas observações. A margem é fixada antes da recomendação externa. Um classificador ExtraTrees com a mesma regularização controla a aprendizagem por rótulo, e uma ablação retira perfil do seletor por consequências sem recuo. A comparação permanece uma avaliação de desenvolvimento, sem garantia de atendimento futuro.')
t = paragraph(t, 'Na Bahia, os testes compartilham', 'Na Bahia, os testes compartilham um ciclo; no M5, ocupam os ciclos 58--64, 72--78 e 86--92, sem sobreposição. Origens repetem séries. Intervalos exploratórios usam bootstrap pareado por loja, preservando séries e origens, com 5.000 sorteios em cada base. Seletores e controladores permanecem fixos; a incerteza é condicional, sem novos treinamentos. O M5 mantém somente dez lojas, limitando a precisão dos intervalos apesar do maior número de séries.')
t = swap(t, 'Coortes históricas, elegibilidade na primeira origem e horizontes de sete ciclos', 'Coortes históricas, elegibilidade por origem e horizontes de sete ciclos')
t = swap(t, 'A calibração distinta da CappedBaseStock e os esforços desiguais limitam comparações entre algoritmos e bases.', 'Os esforços desiguais de ajuste limitam comparações entre algoritmos; a harmonização da CappedBaseStock não elimina diferenças de demanda, escala ou cobertura entre as bases.')
t = paragraph(t, 'Uma avaliação ampliada, ainda fora dos resultados', 'O módulo opcional de contexto permitirá avaliar atributos adicionais admissíveis, com escolhas explícitas do gestor, seleção interna e comparação com o PSE básico. Essa extensão não integra as matrizes aqui analisadas. Sua avaliação requer disponibilidade histórica comprovada e separação entre seleção de atributos e pontuação externa; o cadastro atual não pode ser tratado automaticamente como informação disponível em decisões passadas.')
p.write_text(t, encoding='utf-8')

p = ROOT / 'docs/master_proposal/instanciacao_aipe.tex'
t = p.read_text(encoding='utf-8')
t = paragraph(t, 'O piloto M5 utiliza', 'O M5 utiliza a coorte histórica local de 2.408 identificadores loja--produto da categoria de alimentos (\\texttt{FOODS}), nas dez lojas de três estados. Usam-se ciclos completos de 21 dias, sem corte de picos pelo intervalo interquartil (\\textit{interquartile range}, IQR). A exigência de três ciclos positivos no prefixo admite 885 séries na origem 50, 2.033 na origem 64 e 2.408 na origem 78. O recorte não representa todas as séries FOODS. Um piloto de 30 séries orientou o desenvolvimento; sua ampliação não estabelece confirmação independente.')
t = swap(t, 'Dimensão & Bahia & Piloto M5', 'Dimensão & Bahia & M5')
t = swap(t, 'Séries / lojas & 145 / 115 & 30 / 10', 'Séries / lojas & 145 / 115 & 2.408 / 10')
t = swap(t, 'Casos série--origem & 290 & 90', 'Casos série--origem & 290 & 5.326')
t = swap(t, 'Ajustes de políticas & 4.350 & 1.350', 'Ajustes de políticas & 4.350 & 79.890')
t = swap(t, 'Testes com demanda positiva & 283 & 86', 'Testes com demanda positiva & 283 & 5.138')
t = swap(t, 'nem caracterizar isoladamente o piloto M5', 'nem caracterizar a base M5')
t = swap(t, 'Penalização não garante viabilidade; o serviço observado deve ser conferido. Na Bahia, a calibração da CappedBaseStock também utiliza ruptura e amplificação dos pedidos com pesos unitários; no M5, esses pesos são zero. As matrizes representam instanciações específicas, sem equivalência plena entre objetivos de treino.', 'Penalização não garante viabilidade; o serviço observado deve ser conferido. A CappedBaseStock utiliza pesos zero para ruptura e amplificação dos pedidos nas duas bases. A harmonização do objetivo permite comparar as instanciações sob a mesma definição de calibração, preservando diferenças de demanda e escala.')
t = swap(t, 'O piloto utiliza XGBoost', 'O preditivo por classe utiliza XGBoost')
t = swap(t, 'Cada treinamento tem 27 exemplos.', 'A quantidade de exemplos de treinamento depende da origem e da loja excluída.')
t = paragraph(t, 'A ablação retira somente a entrada de perfil.', 'A ablação retira somente a entrada de perfil. O piloto anterior motivou alternativas de aprendizagem e intervenção; a avaliação consolidada aplica os seletores às matrizes completas e mantém a distinção entre desenvolvimento e confirmação independente \\citep{CawleyTalbot2010}.')
t = swap(t, '\\subsection{Comparação reduzida por consequências}', '\\subsection{Comparação por consequências}')
t = swap(t, 'A extensão usa a mesma matriz M5, sem repetir ajustes de controladores.', 'A comparação usa a mesma matriz de cada base, sem repetir ajustes de controladores.')
t = paragraph(t, 'A variante com recuo usa a global', 'A variante com recuo usa a global como referência e exige serviço previsto de pelo menos 0,70, serviço previsto não inferior ao da referência e economia acima de uma margem. A margem pertence a $\\{0;0{,}02;0{,}05;0{,}10;\\infty\\}$, sendo $\\infty$ o recuo permanente. Ela minimiza custo na validação interna sujeita a serviço médio não inferior à referência. Cada origem mantém uma loja externa reservada e divide as demais em cinco grupos de lojas. Cada grupo interno recebe previsões de um modelo que exclui suas observações. Depois da escolha da margem, o modelo externo é ajustado em todas as lojas admitidas e recomenda para a loja reservada.')
t = swap(t, 'Ainda assim, o piloto já motivou o desenho: a análise é exploratória.', 'O piloto orientou o desenho; a análise consolidada não é apresentada como confirmação independente.')
start = t.index('\\subsection{Instanciação unificada em consolidação}')
end = t.index('\\subsection{Aplicação e reavaliação}', start)
t = t[:start] + '\\subsection{Execução e compatibilidade das matrizes}\nA execução separa planejamento, ajuste dos controladores e avaliação dos seletores. A preparação audita painéis, identidades, código e configurações antes de aproveitar checkpoints. A avaliação recusa matrizes incompletas e usa o mesmo pré-processamento para modelos por classe e por consequências, ajustado somente no conjunto admitido em cada partição. As duas bases incluem global, perfil, classe vencedora com e sem perfil, moda dos rótulos, floresta de classe de controle e previsão de consequências com e sem perfil e com recuo. Oráculos, intervenções e deteriorações completam a avaliação.\n\nA Bahia reúne 4.350 ajustes e o M5, 79.890. Ajustes previamente compatíveis foram reaproveitados após auditoria, sem serem contados como novas observações independentes. A avaliação brasileira preserva sua relevância aplicada, mas continua sendo simulação com custos de cenário e possível censura das vendas. As bases fornecem contextos complementares, sem combinar custos entre elas.\n\n' + t[end:]
t = paragraph(t, 'Bootstrap pareado preserva produtos', 'Bootstrap pareado preserva produtos e origens de cada loja, com 5.000 sorteios em cada base. Ajustes e escolhas permanecem fixos; os intervalos são condicionais. Dependência entre lojas que compartilham produtos e a quantidade de somente dez lojas do M5 limitam a inferência, mesmo com milhares de casos série--origem.')
t = paragraph(t, 'Manifestos, snapshots, sementes', 'Manifestos, snapshots, sementes e bancos transacionais identificam cada matriz. Os resultados derivam de \\nolinkurl{aipe_ba145_unified_2026-10-15} e \\nolinkurl{aipe_m5_2408_unified_2026-10-15}, em \\nolinkurl{experiment/data/08_reporting/research_v2}. As recomendações são gravadas antes da pontuação externa. Dados não integram o Git e possuem backup separado; código e fontes documentais estão versionados.')
t = swap(t, 'não completa esses artefatos nem atualiza as tabelas históricas desta versão.', 'não completa esses artefatos.')
t = paragraph(t, 'Os estudos brasileiros anteriores', 'Os estudos brasileiros anteriores, incluindo a qualificação, motivam a investigação. A evidência quantitativa deste capítulo utiliza somente as matrizes consolidadas descritas, sem misturar configurações. A expansão nacional está cancelada. O módulo opcional de atributos adicionais é uma extensão implementada cuja avaliação própria não integra estes resultados; sua ativação depende da disponibilidade histórica comprovada dos atributos.')
p.write_text(t, encoding='utf-8')

p = ROOT / 'docs/master_proposal/capitulos/introducao.tex'
t = p.read_text(encoding='utf-8')
t = paragraph(t, 'Os estudos da Paraíba e da qualificação', 'Os estudos da Paraíba e da qualificação motivaram o desenvolvimento. A evidência quantitativa consolidada compreende 145 séries da Bahia em duas origens e uma coorte histórica de 2.408 identificadores do M5 nas dez lojas, com elegibilidade por origem. São 290 e 5.326 casos série--origem, respectivamente. O capítulo de resultados separa as bases e suas condições. O piloto anterior orientou o desenvolvimento, sem constituir confirmação independente. A avaliação nacional permanece fora do escopo experimental desta versão.')
t = paragraph(t, 'O procedimento permite avaliar separadamente', 'O procedimento permite avaliar separadamente controladores e seletores. Em ambas as bases, referências globais e por perfil são confrontadas com implementações preditivas por classe e por consequências, incluindo uma regra de recuo à referência. A comparação verifica a utilidade de cada instanciação, sem pressupor vantagem do modelo supervisionado. O piloto anterior integra o desenvolvimento do protocolo.')
p.write_text(t,encoding='utf-8')
p = ROOT / 'docs/master_proposal/capitulos/proposta.tex'
t = p.read_text(encoding='utf-8').replace('na Bahia e no piloto M5.', 'na Bahia e no M5.')
p.write_text(t,encoding='utf-8')

p = ROOT / 'docs/master_proposal/capitulos/resultados.tex'
t = p.read_text(encoding='utf-8')
t = paragraph(t, 'Este capítulo avalia as instanciações', 'Este capítulo avalia as instanciações da metodologia AIPE, distinguindo qualidade dos controladores, utilidade da seleção e limites da recomendação. A Bahia reúne 145 séries em duas origens, com 290 casos; a coorte histórica do M5 reúne 2.408 identificadores, com 5.326 casos nas três origens. As bases são analisadas separadamente sob a mesma definição de calibração e avaliação. Um piloto anterior orientou o desenvolvimento, sem tornar a ampliação uma confirmação independente.')
t = paragraph(t, 'Os números a seguir pertencem', 'Os números a seguir pertencem às matrizes completas e aos protocolos identificados no Capítulo~\\ref{cap:experimental}. A comparação usa validação externa por loja e cinco grupos internos na escolha da margem de recuo. Custo inclui todos os casos; atendimento usa os 283 casos brasileiros e os 5.138 casos M5 com demanda positiva no teste. O módulo opcional de atributos adicionais não integra estes resultados.')
p.write_text(t,encoding='utf-8')
print('Protocolo e escopo atualizados em cinco arquivos; sínteses numéricas aguardam auditoria.')
