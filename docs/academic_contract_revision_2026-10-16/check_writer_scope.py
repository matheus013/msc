from pathlib import Path
import re, json, hashlib

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'docs/archive/methodology_contract_revision_2026-10-16'
files=['docs/supply_chain_analytics_article/manuscript_ptbr.tex','docs/master_proposal/capitulos/proposta.tex','docs/master_proposal/capitulos/introducao.tex','docs/master_proposal/instanciacao_aipe.tex','docs/master_proposal/capitulos/resumo.tex','docs/master_proposal/abstract_en.tex','docs/master_proposal/capitulos/conclusoes.tex','docs/master_proposal/capitulos/plano_continuidade.tex']
report={}
for f in files:
    t=(ROOT/f).read_text(encoding='utf-8')
    b=(BASE/f).read_text(encoding='utf-8')
    math=lambda s: re.findall(r'\\\[.*?\\\]|\\begin\{equation\}.*?\\end\{equation\}',s,re.S)
    cmds=lambda s,name: re.findall(r'\\'+name+r'(?:\[[^]]*\])?\{[^}]*\}',s)
    report[f]={
      'chars_before':len(b),'chars_after':len(t),
      'display_math_unchanged':math(t)==math(b),
      'citations_unchanged':cmds(t,'cite')==cmds(b,'cite') and cmds(t,'citep')==cmds(b,'citep') and cmds(t,'citet')==cmds(b,'citet'),
      'figures_unchanged':cmds(t,'includegraphics')==cmds(b,'includegraphics'),
      'inputs_unchanged':cmds(t,'input')==cmds(b,'input'),
      'labels_unchanged':cmds(t,'label')==cmds(b,'label'),
      'brace_balance':t.count('{')-t.count('}'),
      'style_hits':{k:len(re.findall(p,t,re.I)) for k,p in {'unicode_dash':r'[—–]','decorative_contrast':r'não (?:só|apenas|somente).*?mas','filler':r'é importante (?:notar|destacar)|vale (?:ressaltar|destacar)|it is worth noting|moreover|furthermore','english_hype':r'\b(?:novel|significant|substantial|impressive|promising|comprehensive|powerful|seamless)\b','english_passive':r'\b(?:is|are|was|were|be|been|being)\s+(?:[a-z]+ed|done|made|shown|given|taken|held|built|chosen|known|found|seen|used|based)\b'}.items()},
    }
article=(ROOT/files[0]).read_text(encoding='utf-8')
report['article_abstract_words']=len(re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',article,re.S)[1].split())
report['protected_fragments']={}
for root in ['docs/master_proposal','docs/supply_chain_analytics_article']:
    for p in (ROOT/root).glob('*.tex'):
        if p.name in ['bahia_20261015.tex','m5_pse_20261015.tex','consequencias_20261015.tex'] or ('aipe_' in p.name and 'resultados' in p.name):
            q=BASE/p.relative_to(ROOT)
            report['protected_fragments'][str(p.relative_to(ROOT))]=q.exists() and p.read_bytes()==q.read_bytes()
out=Path(__file__).with_name('writer_checks.json')
out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
