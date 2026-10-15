"""Fetch explicitly identified open research copies for the selection review."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import requests
from PyPDF2 import PdfReader

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'docs/references/inventory_policy_selection/2026-10-14'
TEXT=ROOT/'experiment/data/08_reporting/literature_review_2026-10-14'
SOURCES=[
 ('boute2022','https://pure.tue.nl/ws/files/194325798/1_s2.0_S0377221721006111_main.pdf'),
 ('banrudin2019','https://lbsresearch.london.edu/id/eprint/966/1/BanRudin_BDNV_FINAL-onlineversion.pdf'),
 ('jaarsveldarts2024','https://arxiv.org/pdf/2101.07519'),
 ('stranieri2025','https://arxiv.org/pdf/2501.10895'),
 ('schulman2017','https://arxiv.org/pdf/1707.06347'),
 ('vanhasselt2016','https://arxiv.org/pdf/1509.06461'),
 ('geevers2024','https://link.springer.com/content/pdf/10.1007/s10100-023-00872-2.pdf'),
]


def collect(item):
    key,url=item;path=OUT/(key+'.pdf')
    row={'key':key,'source_url':url}
    try:
        if not path.exists():
            response=requests.get(url,timeout=(10,30))
            response.raise_for_status()
            if not response.content.lstrip().startswith(b'%PDF'):raise ValueError('Not a PDF response')
            path.write_bytes(response.content)
        reader=PdfReader(path)
        text='\n\n'.join(f'PAGE {i+1}\n'+(page.extract_text() or '') for i,page in enumerate(reader.pages))
        (TEXT/(key+'.txt')).write_text(text,encoding='utf-8')
        row.update(status='downloaded_open_copy',pages=len(reader.pages),
                   local_path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    except Exception as exc:row.update(status='unavailable',error=str(exc))
    return row


def main():
    OUT.mkdir(parents=True,exist_ok=True);TEXT.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(collect,SOURCES))
    original=ROOT/'docs/references/supply_chain_analytics/pdfs/2025/100154_A_comparative_study_of_multi_algorithm_optimization_for_inventory_analytics_in_supply.pdf'
    reader=PdfReader(original)
    (TEXT/'zabraoui2025.txt').write_text('\n\n'.join(f'PAGE {i+1}\n'+(p.extract_text() or '') for i,p in enumerate(reader.pages)),encoding='utf-8')
    rows.append({'key':'zabraoui2025','status':'existing_local_copy_read','pages':len(reader.pages),
                 'source_url':'https://doi.org/10.1016/j.sca.2025.100154',
                 'local_path':str(original.relative_to(ROOT)),'sha256':hashlib.sha256(original.read_bytes()).hexdigest()})
    (OUT/'download_manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    metadata=requests.get('https://api.crossref.org/works/10.1007/s11518-022-5544-6',timeout=20).json()['message']
    (TEXT/'inventory_dqn_crossref.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps([{'key':r['key'],'status':r['status'],'pages':r.get('pages')} for r in rows],indent=2))
    print('Inventory DQN authors:',metadata.get('author'))


if __name__=='__main__':main()
