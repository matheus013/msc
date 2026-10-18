"""Targeted Crossref discovery; metadata is not a substitute for full-text review."""
from datetime import datetime,timezone
from pathlib import Path
import json
import re
import requests
import time
import hashlib

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiment/data/literature/aipe_opportunities_2026-10-15'
QUERIES=[
    'inventory classification policy selection',
    'performance based inventory classification policy',
    'inventory policy selection machine learning',
    'conformal inventory control',
    'safe reinforcement learning contextual information inventory',
    'algorithm selection cost sensitive regret abstention',
    'Smart Predict then Optimize',
    'Conformal Risk Control',
]


def search(query):
    cache=OUT/(hashlib.sha256(query.encode()).hexdigest()[:12]+'.json')
    if cache.exists():
        return json.loads(cache.read_text(encoding='utf-8'))
    response=requests.get('https://api.crossref.org/works',params={
        'query.bibliographic':query,'rows':15,'filter':'until-pub-date:2026-10-15'},
        headers={'User-Agent':'AIPE-LiteratureReview/1.0 (academic bibliographic discovery)'},timeout=30)
    response.raise_for_status()
    result={'query':query,'response':response.json()}
    cache.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    results=[]
    errors=[]
    for query in QUERIES:
        try:
            results.append(search(query))
        except requests.RequestException as exc:
            errors.append({'query':query,'error':str(exc)})
            if getattr(exc.response,'status_code',None)==429:
                break  # Preserve progress and let a later invocation resume.
        time.sleep(2)
    (OUT/'search_errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
    (OUT/'crossref_searches.json').write_text(json.dumps({'retrieved_utc':datetime.now(timezone.utc).isoformat(),'cutoff':'2026-10-15','searches':results},indent=2),encoding='utf-8')
    unique={}
    for result in results:
        print('\nQUERY:',result['query'])
        for item in result['response']['message']['items']:
            doi=item.get('DOI','');title=' '.join(item.get('title',[]))
            unique[doi]={'doi':doi,'title':title,'year':item.get('published',{}).get('date-parts'),
                'url':item.get('URL'),'abstract':re.sub('<[^>]+>',' ',item.get('abstract','')),
                'links':item.get('link',[]),'type':item.get('type')}
        for item in result['response']['message']['items'][:8]:print(item.get('title',[''])[0],item.get('DOI'))
    (OUT/'discovered.json').write_text(json.dumps(list(unique.values()),indent=2),encoding='utf-8')
    print('\nUNIQUE RECORDS',len(unique))
