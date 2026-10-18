from pathlib import Path
import hashlib, json, shutil

root = Path(__file__).resolve().parents[2]
archive = root / 'docs/archive/contributions_revision_2026-10-16'
records = Path(__file__).resolve().parent
paths = [
 'docs/supply_chain_analytics_article/manuscript_ptbr.tex',
 'docs/supply_chain_analytics_article/highlights_ptbr.txt',
 'docs/master_proposal/capitulos/introducao.tex',
 'docs/master_proposal/capitulos/resumo.tex',
 'docs/master_proposal/abstract_en.tex',
 'docs/master_proposal/capitulos/proposta.tex',
 'docs/master_proposal/instanciacao_aipe.tex',
 'docs/master_proposal/capitulos/conclusoes.tex',
 'docs/master_proposal/capitulos/plano_continuidade.tex',
]
manifest = []
for rel in paths:
    src = root / rel
    dst = archive / rel
    if dst.exists():
        raise RuntimeError(f'Baseline exists: {dst}')
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest.append({'path':rel,'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'bytes':src.stat().st_size})
(archive / 'baseline_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

def edit(rel, old, new):
    path = root / rel
    text = path.read_text(encoding='utf-8')
    if text.count(old) != 1:
        raise RuntimeError(f'{rel}: expected one match, got {text.count(old)}: {old[:90]}')
    path.write_text(text.replace(old,new),encoding='utf-8',newline='\n')

a = paths[0]
edit(a,'uma metodologia modular de apoio à decisão que vincula contexto, evidência operacional, seleção e avaliação.', 'uma metodologia modular e rastreável de apoio ao gerente na escolha de políticas de reposição, buscando reduzir custos evitáveis sem deterioração relevante do atendimento.')
edit(a,'A metodologia permite distinguir ganhos da seleção, limitações do portfólio e restrições físicas do atendimento.', 'A contribuição central articula contexto, evidência, recomendação e avaliação; seleção por perfil ou estimação, diagnóstico retrospectivo e aplicações complementares sustentam essa proposta.')
edit(a,'A contribuição do AIPE é formular essa sequência como um procedimento verificável: a evidência sustenta a recomendação, a regra de intervenção define sua aplicação e o diagnóstico identifica onde aperfeiçoá-la. Os controladores e seletores concretizam responsabilidades substituíveis dentro desse procedimento. Avaliamos tanto a compatibilidade entre essas implementações quanto sua utilidade incremental frente à referência global. A integração entre classificação e inventário possui antecedentes; o posicionamento do trabalho depende da articulação proposta e das decisões que ela permite examinar.',r'''A contribuição principal é o AIPE como metodologia modular e rastreável de apoio ao gerente na escolha de políticas de reposição. O objetivo operacional é reduzir custos evitáveis sem deterioração relevante do atendimento. Três contribuições sustentam essa formulação:
\begin{enumerate}
 \item \textbf{Seleção contextual.} Integrar o PSE por perfil e por estimação sob contratos de informação, critérios operacionais e referência comuns.
 \item \textbf{Diagnóstico das limitações.} Distinguir falhas de seleção, insuficiência do portfólio e inviabilidade física do atendimento por limites retrospectivos condicionais ao cenário.
 \item \textbf{Avaliação empírica.} Examinar a utilidade e as limitações do procedimento em vendas reais brasileiras e no referencial público M5, analisados separadamente.
\end{enumerate}
Essas contribuições se apoiam em métodos conhecidos. A integração entre classificação e inventário possui antecedentes; a comparação empírica com as integrações mais próximas permanece necessária para avaliar o diferencial do AIPE.''')
edit(a,'O preditivo recomenda', 'O preditivo recomenda') if False else None
edit(a,'Ele pode aprender um rótulo ou estimar consequências por candidata.', 'Aprender um rótulo ou estimar consequências por candidata são variantes da abordagem por estimação, não abordagens adicionais do PSE.')
edit(a,'Os oráculos mostram que corrigir o seletor é apenas uma das intervenções possíveis.', 'O diagnóstico retrospectivo distingue falhas de seleção, insuficiência do portfólio e inviabilidade física, sob as hipóteses do simulador. Essa classificação identifica oportunidades de revisão; não estabelece causas de rupturas observadas na empresa.')
edit(a,'Essa estrutura permite localizar tanto o benefício de uma implementação quanto o componente a revisar quando o atendimento falha.', 'A seleção por perfil e por estimação, o diagnóstico retrospectivo e as aplicações complementares sustentam essa contribuição central. Os resultados exercitam a substituição de controladores e seletores; a rastreabilidade completa da entrega gerencial permanece pendente.')

i=paths[2]
edit(i,'uma metodologia modular de apoio ao gerente de logística e estoques para reduzir custos evitáveis sem deterioração relevante do atendimento.', 'uma metodologia modular e rastreável de apoio ao gerente de logística e estoques na escolha de políticas de reposição, buscando reduzir custos evitáveis sem deterioração relevante do atendimento.')
old='A contribuição principal é formular a seleção contextual como um procedimento com compromissos verificáveis. O AIPE encadeia cinco etapas, admite associação por perfil e estimação por série e exige uma referência competitiva para julgar a intervenção. O diagnóstico liga cada perda à decisão que pode corrigi-la: melhorar a escolha, ampliar o portfólio ou rever as condições operacionais. A metodologia torna essas decisões explícitas e passíveis de avaliação.\n\nA contribuição empírica é delimitar a utilidade dessas escolhas em duas aplicações complementares. As vendas de uma rede varejista brasileira preservam heterogeneidade empresarial observada; o piloto público M5 permite examinar implementações preditivas e suas falhas. Código, matrizes e oráculos sustentam a rastreabilidade da avaliação. Como as consequências são simuladas, o uso de dados reais oferece relevância aplicada, mas não comprova ganho financeiro de implantação.'
new=r'''A contribuição principal é propor o AIPE como metodologia modular e rastreável de apoio ao gerente na escolha de políticas de reposição. Seus contratos ligam contexto, evidência, recomendação e avaliação, buscando reduzir custos evitáveis sem deterioração relevante do atendimento. Essa formulação se desdobra em três contribuições de suporte:
\begin{enumerate}
  \item \textbf{Seleção contextual:} integrar as abordagens do PSE por perfil e por estimação sob os mesmos contratos de informação, critérios operacionais e referência de comparação.
  \item \textbf{Diagnóstico das limitações:} distinguir falha de seleção, insuficiência do portfólio e inviabilidade física do atendimento, orientando quais componentes ou condições revisar.
  \item \textbf{Avaliação empírica:} examinar utilidade, condições de uso e limitações em vendas reais da Bahia e no referencial público M5, com análises separadas.
\end{enumerate}

A implementação exercita substituições no portfólio e no PSE; a portabilidade dos demais módulos e a rastreabilidade completa da entrega gerencial permanecem pendentes. O diagnóstico é retrospectivo e condicional ao simulador, sem identificar causas de rupturas na empresa. As bases complementares sustentam a avaliação do procedimento, mas as consequências simuladas não comprovam ganho financeiro de implantação. A comparação empírica com integrações próximas da literatura também permanece necessária para delimitar o diferencial do AIPE.'''
edit(i,old,new)

r=paths[3]
edit(r,'uma metodologia modular de apoio ao gerente de logística e estoques para reduzir custos evitáveis preservando o atendimento.', 'uma metodologia modular e rastreável de apoio ao gerente de logística e estoques na escolha de políticas de reposição, buscando reduzir custos evitáveis sem deterioração relevante do atendimento.')
edit(r,'A contribuição é um procedimento que vincula cada recomendação à evidência para subsidiar a decisão humana.', 'A contribuição central é o procedimento de apoio à decisão, sustentado pela integração das duas abordagens do PSE, pelo diagnóstico das limitações e pela avaliação em contextos complementares.')
e=paths[4]
edit(e,'a modular methodology supporting logistics and inventory managers in reducing avoidable costs while preserving service.', 'a modular, traceable methodology supporting logistics and inventory managers in selecting replenishment policies, with the aim of reducing avoidable costs without relevant service deterioration.')
edit(e,'The contribution is a procedure linking each recommendation to evidence to support human decisions.', 'The central contribution is the decision-support procedure, supported by the integration of both PSE approaches, the diagnosis of limitations, and evaluation in complementary contexts.')

p=paths[5]
edit(p,'A contribuição investigada é a articulação desses princípios em um procedimento de reposição que vincula duas abordagens do PSE à decisão de intervir e ao diagnóstico das perdas.', 'A contribuição central é a metodologia modular e rastreável de apoio à escolha de políticas. A integração das duas abordagens do PSE, o diagnóstico das limitações e a avaliação em contextos complementares sustentam essa proposta.')

x=paths[6]
edit(x,'O protocolo separa o ajuste dos controladores, a construção do seletor e a avaliação das recomendações.', 'O protocolo separa o ajuste dos controladores, a construção do seletor e a avaliação das recomendações. A comparação entre abordagens do PSE examina a seleção contextual; os oráculos delimitam falhas de seleção, insuficiência do portfólio e inviabilidade física. As duas aplicações fornecem evidência sobre condições de uso, sem validar toda configuração admitida pela metodologia.')

c=paths[7]
edit(c,'A contribuição principal é o procedimento AIPE e sua instanciação verificável. Ele não pressupõe que a escolha contextual seja sempre preferível: permite examinar quando intervir e quando conservar a referência. O mecanismo de recuo, isoladamente, possui antecedentes; o uso de M5 ou GA--DQN também não constitui novidade própria. Seu alcance decorre da formulação para o problema, da justificativa das decisões e da evidência disponível para cada implementação.', 'A contribuição principal é o AIPE como metodologia modular e rastreável de apoio ao gerente na escolha de políticas de reposição. As duas abordagens do PSE concretizam a seleção contextual sob contratos comuns; o diagnóstico identifica limitações da escolha, do portfólio e das condições físicas; Bahia e M5 delimitam sua utilidade em contextos complementares. O procedimento permite examinar quando intervir e quando conservar a referência. Seus métodos constituintes possuem antecedentes, e o diferencial frente às integrações próximas ainda requer confronto empírico.')
edit(c,'O diagnóstico distingue perdas com causas diferentes.', 'O diagnóstico retrospectivo distingue limitações sob as hipóteses do simulador, sem estabelecer causas de rupturas observadas na empresa.')

q=paths[8]
edit(q,'A metodologia AIPE está formulada e possui duas aplicações experimentais.', 'A metodologia AIPE está formulada e possui duas aplicações experimentais. Seu fechamento deve sustentar a contribuição central de apoio modular e rastreável à escolha de políticas, ligando a evidência sobre seleção contextual, diagnóstico e condições de uso.')
edit(q,'O artigo concentra a arquitetura e sua avaliação operacional;', 'A comparação com as integrações mais próximas da literatura deverá explicitar o que o AIPE acrescenta em formulação, responsabilidades dos módulos e evidência, sem atribuir originalidade aos métodos constituintes. O artigo concentra a metodologia e sua avaliação operacional;')

(root/paths[1]).write_text('AIPE propõe apoio modular e rastreável à escolha de políticas de reposição.\nO PSE integra seleção por perfil e por estimação sob contratos comuns.\nO diagnóstico distingue seleção, portfólio e inviabilidade física.\nVendas reais da Bahia e o piloto M5 delimitam condições de uso.\nMétodos conhecidos sustentam a proposta; ganhos exigem confirmação.\n',encoding='utf-8')
print(json.dumps({'baseline_files':len(paths),'modified_files':paths},ensure_ascii=False,indent=2))
