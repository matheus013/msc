"""Fetch the current Elsevier-maintained class from CTAN; retain provenance."""
from pathlib import Path
import hashlib
import json
import zipfile
import io
import requests

root = Path(__file__).resolve().parents[3] / 'docs/supply_chain_analytics_article'
target = root / 'template'
target.mkdir(exist_ok=True)
url = 'https://mirrors.ctan.org/macros/latex/contrib/elsarticle.zip'
response = requests.get(url, timeout=90)
response.raise_for_status()
archive = response.content
(target / 'elsarticle.zip').write_bytes(archive)
with zipfile.ZipFile(io.BytesIO(archive)) as z:
    for member in z.infolist():
        path = (target / member.filename).resolve()
        if not path.is_relative_to(target.resolve()):
            raise ValueError('Unsafe archive member')
    z.extractall(target)
(target / 'provenance.json').write_text(json.dumps({
    'source': url, 'resolved_url': response.url,
    'package_page': 'https://ctan.org/pkg/elsarticle',
    'verified_version': '3.5', 'version_date': '2026-01-09',
    'retrieved': '2026-10-14', 'sha256': hashlib.sha256(archive).hexdigest(),
    'journal_guide': 'https://www.sciencedirect.com/journal/supply-chain-analytics/publish/guide-for-authors',
    'note': 'Generic Elsevier submission class; journal requirements checked separately.'
}, indent=2), encoding='utf-8')
print('Template downloaded and archive paths checked:', target)
