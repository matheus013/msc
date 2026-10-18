from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

base = Path(__file__).resolve().parent
root = base.parents[1]
prior_path = base / 'round_02/review_contribution.json'
prior = json.loads(prior_path.read_text(encoding='utf-8'))
sources = []
changed = []
for item in prior['sources']:
    digest = hashlib.sha256((root / item['path']).read_bytes()).hexdigest()
    sources.append({'path': item['path'], 'sha256': digest})
    if digest != item['sha256']:
        changed.append(item['path'])
assert changed == ['docs/master_proposal/capitulos/introducao.tex'], changed
result = {
    'reviewer': '/root/contributions_review_position',
    'round': 3,
    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    'verdict': 'APPROVED_EDITORIAL_SCOPE',
    'counts': {'critical': 0, 'major': 0, 'minor': 0},
    'findings': [],
    'submission_ready': False,
    'scope': 'Regressão focal após compressão: introdução completa, diff e resposta do escritor da rodada 3.',
    'checked': [
        'Contribuição central metodológica, gerente destinatário, redução de custos evitáveis e preservação contextual do atendimento mantidos.',
        'Seleção contextual, diagnóstico e aplicações complementares continuam três suportes explícitos da proposta.',
        'Antecedentes e confronto próximo pendente permanecem; métodos constituintes não recebem novidade algorítmica.',
        'Objetivos e hipóteses mantêm caráter investigativo; resultado favorável não virou condição presumida do método.',
        'Modularidade exercitada, rastreabilidade parcial, diagnóstico retrospectivo, simulação, desenvolvimento no piloto e família fixa em sete ciclos continuam delimitados.',
        'A redução elimina repetição sem apagar cinco etapas, duas abordagens, bases separadas ou diferença entre execução e benefício.',
        'Reparo final: a frase “A avaliação investiga quando a seleção contextual oferece vantagem operacional” explicita o referente e conserva caráter investigativo, sem alegar vantagem demonstrada.',
        'Fechamento: removido somente o parágrafo de navegação “Os capítulos apresentam...”, redundante com o sumário; a remoção não altera contribuição, objetivos, evidência ou limites científicos.',
        'Somente introducao.tex mudou entre as 16 fontes vinculadas na rodada 2.',
    ],
    'prior_review': {'path': 'docs/academic_contributions_revision_2026-10-16/round_02/review_contribution.json', 'sha256': hashlib.sha256(prior_path.read_bytes()).hexdigest()},
    'sources': sources,
    'limitations': 'Parecer anterior permanece aplicável às fontes intactas; sem nova revisão de literatura, simulação ou inspeção PDF. Aprovação editorial focal não certifica submissão.',
}
out = base / 'round_03'
out.mkdir(exist_ok=True)
(out / 'review_contribution.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
lines = ['# Parecer de regressão — contribuição, rodada 3', '', '**Aprovado no escopo editorial: 0 críticos, 0 maiores e 0 menores.**', '', 'Releitura completa da introdução final, do diff e da resposta do escritor. A compressão preserva a hierarquia, objetivos e alcance das contribuições. Não editei fontes.', '']
lines += ['- ' + check for check in result['checked']]
lines += ['', 'O parecer da rodada 2 permanece válido para as demais fontes, cujos hashes foram reconferidos. As lacunas científicas já declaradas continuam abertas; esta aprovação não certifica prontidão para submissão nem paginação.', '', f"Parecer anterior: `{result['prior_review']['path']}`, SHA-256 `{result['prior_review']['sha256']}`.", '', '## Hashes finais', '', '| Arquivo | SHA-256 |', '|---|---|']
lines += [f"| `{item['path']}` | `{item['sha256']}` |" for item in sources]
(out / 'review_contribution.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({'verdict': result['verdict'], 'counts': result['counts'], 'changed': changed}, ensure_ascii=False))
