"""Compile dissertation without requiring Perl/latexmk; measure condensation."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from PyPDF2 import PdfReader

ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'docs/master_proposal'


def run(command,cwd,env=None):
    p=subprocess.run(command,cwd=cwd,env=env,capture_output=True,text=True,errors='replace')
    if p.returncode:
        print((p.stdout+p.stderr)[-5000:])
        raise SystemExit(p.returncode)


def count(path):
    s=re.sub(r'(?m)(?<!\\)%.*$','',path.read_text(encoding='utf-8'))
    return len(re.findall(r'\b[^\W\d_]+(?:[-–][^\W\d_]+)*\b',s))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sync-pdf',action='store_true');args=parser.parse_args()
    build=DOC/'build';(build/'capitulos').mkdir(parents=True,exist_ok=True)
    run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build','tcc.tex'],DOC)
    env=os.environ.copy();env['BIBINPUTS']=str(DOC)+';'+env.get('BIBINPUTS','')
    run(['bibtex','tcc'],build,env)
    for _ in range(3):run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build','tcc.tex'],DOC)
    log=(build/'tcc.log').read_text(errors='replace')
    issues=[line for line in log.splitlines() if any(x in line for x in ['undefined','Overfull','LaTeX Error','multiply defined'])]
    if issues:
        print('Diagnostics requiring review:\n'+'\n'.join(issues))
    baseline=json.loads((DOC/'archive/condensation_2026-10-14/baseline_counts.json').read_text())
    active=re.findall(r'\\(?:include|input)\{capitulos/([^}]+)\}',re.sub(r'(?m)%.*$','',(DOC/'tcc.tex').read_text()))
    rows={name+'.tex':{'before':baseline[name+'.tex']['words_approx'],'after':count(DOC/'capitulos'/f'{name}.tex')} for name in active}
    supplemental=['abstract_en.tex','instanciacao_aipe.tex']
    pending=[DOC/'capitulos'/f'{name}.tex' for name in active]+[DOC/'instanciacao_aipe.tex']
    seen=set()
    while pending:
        source=pending.pop()
        if source in seen:continue
        seen.add(source)
        clean=re.sub(r'(?m)(?<!\\)%.*$','',source.read_text(encoding='utf-8'))
        for child in re.findall(r'\\(?:input|include)\{([^}]+)\}',clean):
            name=child if child.endswith('.tex') else child+'.tex'
            supplemental.append(name);pending.append(DOC/name)
    for name in dict.fromkeys(supplemental):
        if (DOC/name).exists():
            rows[name]={'before':0,'after':count(DOC/name)}
    before=sum(v['words_approx'] for v in baseline.values());after=sum(v['after'] for v in rows.values())
    pages=len(PdfReader(build/'tcc.pdf').pages)
    payload={'before_pages':118,'after_pages':pages,'page_reduction_percent':100*(118-pages)/118,
       'before_approx_source_words':before,'after_approx_source_words':after,
       'source_word_reduction_percent':100*(before-after)/before,
       'word_count_note':'Approximate Unicode-word count including TeX commands; active chapters, English abstract and recursively active auxiliary TeX only. Inactive protocol/archive files excluded. Page counts are the primary condensation measure.',
       'chapters':rows,'diagnostics':issues,'pdf_sha256':hashlib.sha256((build/'tcc.pdf').read_bytes()).hexdigest()}
    target=DOC/'revisao_concisao';target.mkdir(exist_ok=True)
    (target/'comparison.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(payload,indent=2))
    if args.sync_pdf:
        if issues:raise SystemExit('Resolve diagnostics before syncing the canonical PDF.')
        shutil.copy2(build/'tcc.pdf',DOC/'tcc.pdf')


if __name__=='__main__':main()
