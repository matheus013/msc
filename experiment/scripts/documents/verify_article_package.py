"""Verify references, portable TeX inputs, figures and hashes after compilation."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'docs/supply_chain_analytics_article'
main=(DOC/'manuscript_ptbr.tex').read_text(encoding='utf-8')
assert '\\appendix' not in main
sources={'manuscript_ptbr.tex':main}
for name in re.findall(r'\\input\{([^}]+)\}',main):
    sources[name+'.tex']=(DOC/(name+'.tex')).read_text(encoding='utf-8')
all_text='\n'.join(sources.values())
bib=(DOC/'references_ptbr.bib').read_text(encoding='utf-8')
keys=re.findall(r'@\w+\{([^,]+),',bib)
assert len(keys)==len(set(keys)), 'Duplicate bibliography keys'
cited={key.strip() for block in re.findall(r'\\cite\w*\{([^}]+)\}',all_text) for key in block.split(',')}
assert cited<=set(keys),'Missing bibliography entries'
figures=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',all_text)
assert len(figures)>=5 and all(f.endswith('.pdf') for f in figures)
fonts={}
for name in figures:
    assert (DOC/name).exists()
    text=subprocess.check_output(['pdffonts',str(DOC/name)],text=True)
    assert 'Type 3' not in text, name+' uses bitmap fonts'
    assert 'Type 1' in text
    fonts[name]=text.splitlines()[2:]
abstract=main.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
assert len(abstract.split())<=250
highlights=(DOC/'highlights_ptbr.txt').read_text(encoding='utf-8-sig').splitlines()
assert 3<=len(highlights)<=5 and all(len(x)<=85 for x in highlights)
assert len(main.split(r'\begin{keyword}')[1].split(r'\end{keyword}')[0].split(r'\sep')) in range(1,8)
log=(DOC/'manuscript_ptbr.log').read_text(errors='replace')
assert not any(x in log for x in ['undefined','Overfull','LaTeX Error'])
manifest=json.loads((DOC/'build_manifest_ptbr.json').read_text())
for name,expected in manifest['sha256'].items():
    assert hashlib.sha256((DOC/name).read_bytes()).hexdigest()==expected,name
with zipfile.ZipFile(DOC/'article_sources_ptbr_draft.zip') as archive:
    for name in list(sources)+figures+['references_ptbr.bib','manuscript_ptbr.bbl']:
        assert archive.read(name)==(DOC/name).read_bytes(),name
pdfinfo=subprocess.check_output(['pdfinfo',str(DOC/'manuscript_ptbr.pdf')],text=True)
report={'status':'passed','pages':int(re.search(r'Pages:\s+(\d+)',pdfinfo)[1]),
        'abstract_whitespace_words':len(abstract.split()),'cited_references':len(cited),
        'highlight_characters':[len(x) for x in highlights],'figures':figures,
        'figure_fonts':fonts,'portable_source_archive':True,
        'no_undefined_references_or_overfull_boxes':True,
        'manuscript_sha256':hashlib.sha256((DOC/'manuscript_ptbr.pdf').read_bytes()).hexdigest(),
        'note':'Structural and artifact checks; scientific audit is recorded separately.'}
target=DOC/'reviews/aipe_revision_2026-10-15/package_checks.json'
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='figure_fonts'},indent=2))
