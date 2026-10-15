"""Compile the journal draft and flatten source files for Editorial Manager.

Run from any folder: python experiment/scripts/literature/build_sca_article.py
Use --reanalyse to regenerate local analyses and data plots first.
The archive is a review draft; this command never submits or uploads it.
"""
from pathlib import Path
import argparse
import json
import hashlib
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'docs/supply_chain_analytics_article'


def run(command):
    result=subprocess.run(command,cwd=DOC,capture_output=True,text=True,errors='replace')
    if result.returncode:
        print((result.stdout+result.stderr)[-6000:])
        raise SystemExit(result.returncode)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reanalyse',action='store_true')
    parser.add_argument('--ptbr',action='store_true',help='Build the new Portuguese manuscript preserving the English version.')
    args=parser.parse_args()
    if args.reanalyse:
        for name in (['sca_dual_dataset.py','sca_dual_figures.py'] if args.ptbr else ['sca_validation.py','sca_temporal.py','sca_figures.py']):
            run([sys.executable,str(ROOT/'experiment/scripts/analysis'/name)])
    for exe in ['pdflatex','bibtex']:
        if not shutil.which(exe):
            raise SystemExit(f'Missing TeX command: {exe}')
    stem='manuscript_ptbr' if args.ptbr else 'manuscript'
    run(['pdflatex','-interaction=nonstopmode','-halt-on-error',stem+'.tex'])
    run(['bibtex',stem])
    for _ in range(2):
        run(['pdflatex','-interaction=nonstopmode','-halt-on-error',stem+'.tex'])
    log=(DOC/(stem+'.log')).read_text(errors='replace')
    bad=[line for line in log.splitlines() if any(v in line for v in ['undefined','Overfull','LaTeX Error'])]
    if bad:
        raise SystemExit('Review compilation diagnostics:\n'+'\n'.join(bad))
    files=[stem+'.tex',stem+'.bbl','references_ptbr.bib' if args.ptbr else 'references.bib','elsarticle.cls','elsarticle-num.bst']
    files+=['highlights_ptbr.txt'] if args.ptbr else ['Figure_1.png','Figure_2.png','highlights.txt']
    if args.ptbr:
        files.extend(['bahia_20261015.tex','protocolo_bahia_20261015.tex'])
        files.extend('figures/'+name+'.png' for name in [
            'fluxo_experimental','custo_servico','escolhas_perfil','distribuicao_atendimento'])
    zip_name='article_sources_ptbr_draft.zip' if args.ptbr else 'article_sources_draft.zip'
    with zipfile.ZipFile(DOC/zip_name,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(DOC/name,name)
    manifest={name:hashlib.sha256((DOC/name).read_bytes()).hexdigest()
              for name in files+[stem+'.pdf',zip_name]}
    (DOC/('build_manifest_ptbr.json' if args.ptbr else 'build_manifest.json')).write_text(json.dumps({'status':'research draft; not submitted',
        'elsarticle':'3.5 2026-01-09','sha256':manifest},indent=2),encoding='utf-8')
    print(f'Compiled {stem}.pdf; flattened review archive; no undefined references or overfull boxes.')


if __name__=='__main__':
    main()
