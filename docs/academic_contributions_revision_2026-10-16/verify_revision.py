"""Document-only integrity and style checks; never fits or simulates policies."""
from pathlib import Path
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'docs/archive/contributions_revision_2026-10-16'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_paths():
    for name in ('master_proposal', 'supply_chain_analytics_article'):
        folder = ROOT / 'docs' / name
        yield from folder.glob('*.bib')
        for stem in ('bahia_20261015', 'm5_pse_20261015', 'consequencias_20261015'):
            yield folder / (stem + '.tex')
        yield from (folder / 'figures').rglob('*.*')


parser = argparse.ArgumentParser()
parser.add_argument('--capture-protected', action='store_true')
args = parser.parse_args()
if args.capture_protected:
    target = OUT / 'protected_before.json'
    if target.exists():
        raise SystemExit('Existing protected baseline must not be overwritten.')
    target.write_text(json.dumps({str(p.relative_to(ROOT)): sha(p) for p in protected_paths() if p.is_file()}, indent=2), encoding='utf-8')
    print('Protected document artifacts captured.')
else:
    baseline = json.loads((BASE / 'baseline_manifest.json').read_text(encoding='utf-8'))
    patterns = {
        'unicode_dash': r'[—–]',
        'decorative_contrast': r'não (?:só|apenas|somente)[^.!?]*?mas',
        'filler': r'é importante (?:notar|destacar)|vale (?:ressaltar|destacar)|it is worth noting|moreover|furthermore',
        'english_hype': r'\b(?:novel|significant|substantial|impressive|promising|comprehensive|powerful|seamless)\b',
        'english_passive_candidate': r'\b(?:is|are|was|were|be|been|being)\s+(?:[a-z]+ed|done|made|shown|given|taken|held|built|chosen|known|found|seen|used|based)\b',
    }
    files = {}
    for row in baseline:
        rel = row['path']
        before_path, after_path = BASE / rel, ROOT / rel
        assert sha(before_path) == row['sha256'], f'Baseline drift: {rel}'
        before, after = before_path.read_text(encoding='utf-8'), after_path.read_text(encoding='utf-8')
        math = lambda s: re.findall(r'\\\[.*?\\\]|\\begin\{equation\}.*?\\end\{equation\}', s, re.S)
        commands = lambda s, cmd: re.findall(r'\\' + cmd + r'(?:\[[^]]*\])?\{[^}]*\}', s)
        checks = {'display_math_unchanged': math(before) == math(after)}
        for cmd in ('cite', 'citep', 'citet', 'input', 'includegraphics', 'label'):
            checks[cmd + '_unchanged'] = commands(before, cmd) == commands(after, cmd)
        checks['brace_balance'] = after.count('{') - after.count('}')
        hits = {name: [{'line': after[:m.start()].count('\n') + 1, 'match': m.group()} for m in re.finditer(pattern, after, re.I)] for name, pattern in patterns.items()}
        files[rel] = {'changed': sha(after_path) != row['sha256'], 'sha256': sha(after_path), 'chars_before': len(before), 'chars_after': len(after), **checks, 'style_hits': hits}
        assert all(value for name, value in checks.items() if name != 'brace_balance'), (rel, checks)
        assert checks['brace_balance'] == 0, rel
    protected = json.loads((OUT / 'protected_before.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT / rel) == digest for rel, digest in protected.items()), 'Protected artifacts changed'
    plan = json.loads((ROOT / 'experiment/data/08_reporting/research_v2/aipe_m5_2408_unified_2026-10-15/plan.json').read_text(encoding='utf-8'))
    code = plan['code_sha256']
    assert all(sha(ROOT / rel) == digest for rel, digest in code.items()), 'Frozen scientific code changed'
    report = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'files': files, 'protected_artifacts_unchanged': len(protected), 'scientific_code_matching_frozen_plan': len(code), 'training_or_simulation_started': False}
    (OUT / 'integrity_checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'changed_sources': [r for r, v in files.items() if v['changed']], 'protected_artifacts_unchanged': len(protected), 'scientific_code_matching_frozen_plan': len(code), 'style_hits': {name: sum(len(v['style_hits'][name]) for v in files.values()) for name in patterns}}, ensure_ascii=False, indent=2))
