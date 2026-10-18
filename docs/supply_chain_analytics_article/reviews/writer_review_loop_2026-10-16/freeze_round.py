"""Freeze the actual article reviewed in each writer/reviewer iteration."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil

BASE=Path(__file__).resolve().parent
ARTICLE=BASE.parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(number):
    destination=BASE/f'round_{number:02d}'/'source'
    if destination.exists():raise ValueError('A frozen review source cannot be overwritten')
    original=BASE/'round_00/source'
    protected=['aipe_bahia_resultados.tex','m5_pse_20261015.tex',
               'aipe_consequencias_resultados.tex','references_ptbr.bib']
    for name in protected:
        if sha(original/name)!=sha(ARTICLE/name):
            if name!='aipe_consequencias_resultados.tex':
                raise ValueError('Frozen evidence changed: '+name)
            expected=(original/name).read_text(encoding='utf-8')
            expected=expected.replace('Custo & Serviço & Casos','Custo & Atendimento & Casos')
            for percent,fraction in [('68,97','0,6897'),('69,37','0,6937'),
                                     ('69,25','0,6925'),('71,72','0,7172')]:
                expected=expected.replace('& '+percent+r'\% &','& '+fraction+' &')
            if expected!=(ARTICLE/name).read_text(encoding='utf-8'):
                raise ValueError('Change exceeds approved table presentation: '+name)
    display=re.compile(r'\\\[.*?\\\]|\\begin\{(equation\*?|align\*?)\}.*?\\end\{\1\}',re.S)
    before=(original/'manuscript_ptbr.tex').read_text(encoding='utf-8')
    after=(ARTICLE/'manuscript_ptbr.tex').read_text(encoding='utf-8')
    if [m.group() for m in display.finditer(before)]!=[m.group() for m in display.finditer(after)]:
        raise ValueError('Equations changed; coordinator must review before freezing')
    files={'manuscript_ptbr.pdf','references_ptbr.bib','highlights_ptbr.txt',
           'elsarticle.cls','elsarticle-num.bst','manuscript_ptbr.bbl'}
    pending=['manuscript_ptbr.tex'];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name);files.add(name)
        text=(ARTICLE/name).read_text(encoding='utf-8')
        for child in re.findall(r'\\input\{([^}]+)\}',text):
            pending.append(child if child.endswith('.tex') else child+'.tex')
        files.update(re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text))
    for name in sorted(files):
        source=ARTICLE/name;target=destination/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    manifest={name:sha(destination/name) for name in sorted(files)}
    (destination.parent/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'round':number,'source':str(destination),'manuscript_sha256':manifest['manuscript_ptbr.tex']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('round',type=int)
    freeze(parser.parse_args().round)
