from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
paths = [
    'docs/supply_chain_analytics_article/manuscript_ptbr.tex',
    'docs/supply_chain_analytics_article/aipe_bahia_resultados.tex',
    'docs/supply_chain_analytics_article/m5_pse_20261015.tex',
    'docs/supply_chain_analytics_article/aipe_consequencias_resultados.tex',
    'docs/supply_chain_analytics_article/highlights_ptbr.txt',
    'docs/master_proposal/capitulos/introducao.tex',
    'docs/master_proposal/capitulos/proposta.tex',
    'docs/master_proposal/capitulos/conclusoes.tex',
    'docs/master_proposal/capitulos/resumo.tex',
    'docs/master_proposal/abstract_en.tex',
    'docs/master_proposal/instanciacao_aipe.tex',
    'docs/master_proposal/capitulos/plano_continuidade.tex',
    'docs/supply_chain_analytics_article/REVISAO_DIFERENCIAL_AIPE_2026-10-15.md',
    'docs/academic_contract_revision_2026-10-16/FECHAMENTO.md',
    'docs/academic_contributions_revision_2026-10-16/CONTRATO.md',
    'docs/academic_contributions_revision_2026-10-16/writer_response.md',
]
sources = [{'path': p, 'sha256': hashlib.sha256((root / p).read_bytes()).hexdigest()} for p in paths]
review = {
    'reviewer': '/root/contributions_review_position',
    'lens': 'Contribuição científica e posicionamento; revisão independente, sem autoria das fontes',
    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    'verdict': 'APPROVED_EDITORIAL_SCOPE',
    'submission_ready': False,
    'counts': {'critical': 0, 'major': 0, 'minor': 0},
    'findings': [],
    'scope': 'Integração da hierarquia de contribuições adotada pelo autor; não é nova revisão bibliográfica ou certificação científica integral.',
    'checks': [
        {'id': 'C1', 'status': 'pass', 'locations': ['docs/supply_chain_analytics_article/manuscript_ptbr.tex:35', 'docs/master_proposal/capitulos/introducao.tex:67'], 'result': 'Uma contribuição principal e três suportes explícitos, sem tratar algoritmos constituintes como quatro novidades independentes.'},
        {'id': 'C2', 'status': 'pass', 'locations': ['docs/supply_chain_analytics_article/manuscript_ptbr.tex:41', 'docs/supply_chain_analytics_article/manuscript_ptbr.tex:54', 'docs/master_proposal/capitulos/conclusoes.tex:15'], 'result': 'Antecedentes reconhecidos; confronto empírico próximo ainda necessário. Nenhuma prioridade universal ou superioridade nova.'},
        {'id': 'C3', 'status': 'pass', 'locations': ['docs/supply_chain_analytics_article/manuscript_ptbr.tex:81', 'docs/supply_chain_analytics_article/manuscript_ptbr.tex:205', 'docs/master_proposal/capitulos/introducao.tex:74', 'docs/master_proposal/capitulos/conclusoes.tex:19'], 'result': 'Rastreabilidade é compromisso metodológico; persistência e entrega gerencial atuais explicitamente parciais.'},
        {'id': 'C4', 'status': 'pass', 'locations': ['docs/supply_chain_analytics_article/manuscript_ptbr.tex:139', 'docs/master_proposal/capitulos/introducao.tex:63', 'docs/master_proposal/instanciacao_aipe.tex'], 'result': 'BA 145 e piloto M5 30 separados; população histórica e simulação delimitadas; expansão em curso fora dos números publicados.'},
        {'id': 'C5', 'status': 'pass', 'locations': ['docs/supply_chain_analytics_article/manuscript_ptbr.tex:181', 'docs/master_proposal/capitulos/proposta.tex:112', 'docs/master_proposal/abstract_en.tex:3'], 'result': 'Redução de custos sem deterioração relevante é objetivo, não garantia ou não inferioridade demonstrada. PT/EN preservam conteúdo e grau de evidência.'},
        {'id': 'C6', 'status': 'pass', 'locations': ['docs/supply_chain_analytics_article/m5_pse_20261015.tex', 'docs/master_proposal/capitulos/proposta.tex'], 'result': 'Diagnóstico condicional ao cenário, oráculos retrospectivos e demanda zero separados; nenhuma alegação causal sobre rupturas reais.'},
    ],
    'declared_scientific_gaps_not_editorial_defects': [
        'Confronto com integrações próximas e leitura integral dos antecedentes cujos detalhes não foram verificados.',
        'Confirmação independente após o desenvolvimento no piloto; ganhos concentrados e intervalos que alcançam zero permanecem declarados.',
        'Harmonização da calibração, sensibilidades e análise dos diferentes níveis de restrição de atendimento.',
        'Completar artefatos de rastreabilidade, validar portabilidade além dos controladores/seletores exercitados e avaliar uso com gerentes.',
        'Revisão sequencial e custos financeiros reais não avaliados; aplicação atual fixa família por sete ciclos.',
    ],
    'limitations_of_review': [
        'Fontes locais e relatório bibliográfico existente; sem pesquisa web e sem nova verificação dos textos originais citados.',
        'Não reproduz simulações nem certifica integridade numérica: revisa a correspondência entre a contribuição declarada e evidência relatada.',
        'Não certifica layout dos PDFs ou requisitos atuais da revista; esses controles cabem ao coordenador.',
        'O parecer é assistência editorial interna por agente e não avaliação humana, aceite ou previsão de publicação.',
    ],
    'sources': sources,
}
(out / 'review_contribution.json').write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
lines = [
    '# Parecer independente — contribuição e posicionamento', '',
    '16/10/2026. Revisor: `/root/contributions_review_position`. **Aprovado no escopo editorial: 0 críticos, 0 maiores e 0 menores identificados. Prontidão para submissão: não certificada.**', '',
    'A integração adota a contribuição principal do AIPE como metodologia modular e rastreável de apoio ao gerente, com seleção contextual, diagnóstico e aplicações complementares como sustentação. Não encontrei inconsistência que exija reparo nesta integração. Não escrevi ou alterei as fontes revisadas.', '',
    '## Verificações e mapa de evidência', '',
]
for c in review['checks']:
    lines += [f"- **{c['id']}** — {c['result']} Localização: " + '; '.join(f'`{p}`' for p in c['locations']) + '.', '']
lines += ['## Achados editoriais', '', 'Nenhum achado crítico, maior ou menor nesta lente. Não proponho alterações cosméticas para preencher um parecer. A revisão considerou continuidade entre resumo, introdução, formulação, discussão, limites e conclusões; números desfavoráveis e limites das aplicações permanecem explícitos.', '', '## Lacunas científicas já declaradas', '']
lines += ['- ' + x for x in review['declared_scientific_gaps_not_editorial_defects']]
lines += ['', 'Essas pendências limitam a contribuição demonstrada e a submissão, mas não constituem defeitos novos da incorporação editorial autorizada. Uma metodologia pode ser proposta e executada sem que todas as suas implementações ofereçam melhoria operacional. O texto preserva essa distinção.', '', '## Limites do parecer', '']
lines += ['- ' + x for x in review['limitations_of_review']]
lines += ['', '## Fontes vinculadas por SHA-256', '', '| Arquivo | SHA-256 |', '|---|---|']
lines += [f"| `{s['path']}` | `{s['sha256']}` |" for s in sources]
(out / 'review_contribution.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({'verdict': review['verdict'], 'counts': review['counts'], 'sources': len(sources)}, ensure_ascii=False))
