from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

base = Path(__file__).resolve().parent
root = base.parents[1]
previous = json.loads((base / 'review_contribution.json').read_text(encoding='utf-8'))
current = []
changes = []
for source in previous['sources']:
    path = source['path']
    digest = hashlib.sha256((root / path).read_bytes()).hexdigest()
    current.append({'path': path, 'sha256': digest})
    if digest != source['sha256']:
        changes.append(path)
expected = {
    'docs/supply_chain_analytics_article/manuscript_ptbr.tex',
    'docs/master_proposal/capitulos/introducao.tex',
    'docs/master_proposal/abstract_en.tex',
}
assert set(changes) == expected, changes
review = {
    'reviewer': '/root/contributions_review_position',
    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    'round': 2,
    'verdict': 'APPROVED_EDITORIAL_SCOPE',
    'counts': {'critical': 0, 'major': 0, 'minor': 0},
    'findings': [],
    'submission_ready': False,
    'checks': [
        'MET-01: escopo da restrição de serviço explicitado, preservando diferenças entre instâncias e a integração das duas abordagens.',
        'EDC-01: retomada curta não apaga a contribuição central, expressa na abertura, objetivo geral e conclusão.',
        'EDC-02: practically important deterioration in service preserva relevância prática contextual, sem criar margem ou garantia de não inferioridade.',
        'Somente as três fontes declaradas mudaram em relação aos hashes do parecer inicial; demais treze fontes permanecem idênticas.',
        'A hierarquia de uma contribuição metodológica e três suportes, os antecedentes e o alcance condicionado da evidência permanecem preservados.',
    ],
    'changed_since_round01': changes,
    'scientific_gaps': previous['declared_scientific_gaps_not_editorial_defects'],
    'limitations': 'Releitura incremental das três fontes alteradas e resposta do escritor; parecer inicial permanece aplicável às fontes intactas. Não certifica submissão, simulação, literatura nova ou layout PDF.',
    'sources': current,
}
out = base / 'round_02'
out.mkdir(exist_ok=True)
(out / 'review_contribution.json').write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
text = ['# Parecer final — contribuição e posicionamento, rodada 2', '',
        '**Aprovado no escopo editorial: 0 críticos, 0 maiores e 0 menores.** Prontidão científica para submissão continua não certificada.', '',
        'Releitura independente e incremental dos três arquivos alterados, à luz de `writer_response_round02.md`. O parecer inicial permanece aplicável aos demais arquivos, cujos hashes não mudaram.', '',
        '## Verificações', '']
text += ['- ' + item for item in review['checks']]
text += ['', 'Nenhum reparo adicional é necessário nesta lente. Não alterei fontes nem executei experimentos. As lacunas científicas anteriormente declaradas continuam abertas e distintas da aprovação editorial da integração.', '', '## Hashes finais', '', '| Arquivo | SHA-256 |', '|---|---|']
text += [f"| `{item['path']}` | `{item['sha256']}` |" for item in current]
(out / 'review_contribution.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
print(json.dumps({'verdict': review['verdict'], 'counts': review['counts'], 'changed': changes}, ensure_ascii=False))
