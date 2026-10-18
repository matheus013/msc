from pathlib import Path
import json, re, hashlib, difflib
root = Path(__file__).resolve().parents[2]
record = Path(__file__).resolve().parent
base = root/'docs/archive/contributions_revision_2026-10-16'
entries=json.loads((base/'baseline_manifest.json').read_text(encoding='utf-8'))
patterns={
 'M1_em_dash':r'---|—|–| -- ',
 'M11_passive_EN':r'\b(?:is|are|was|were|be|been|being)\s+(?:[a-z]+ed|done|made|shown|given|taken|held|built|drawn|chosen|written|known|found|seen|set|put|sent|kept|met|run|used|based)\b',
 'M2_antithesis':r',? not [0-9A-Za-z\\]|not only .* but|rather than|less .* than|is the point|whatever it is|means nothing|more than just|not in competition with|on one hand|on the other hand',
 'M3_M4_M8_M9':r'in effect|in a sense|at (?:its|the) (?:heart|core)|in essence|\btruly\b|\bgenuinely\b|\bindeed\b|\bin fact\b|precisely because|a testament to|the kind of .* that|exactly the kind|is the point|sets? .* apart|no (?:predecessor|one) .* (?:made|posed)|the key (?:insight|idea) is|the machine that|draws? .* power from|under the hood|where .* meets',
 'M6_openers':r'^(?:Moreover|Furthermore|Additionally|Notably|Importantly|Indeed|Ultimately|Crucially|In turn|That said)',
 'M5_M16_words':r'\bnovel\b|\bsignificant\b|\bsubstantial\b|\bimpressive\b|\bpromising\b|\bcomprehensive\b|\brobust\b|\bpowerful\b|\bseamless|\bcrucial\b|\bparadigm\b|\bleverag|\butiliz|\bfinaliz|[a-z]+-oriented\b|\bfactor\b|\bfeature[ds]?\b|\bmeaningful\b|\binsightful\b|\bprestigious\b|\bpossess|\bcontact(?:s|ed|ing)?\b|\bcurrently\b|\bimpact(?:s|ed|ing)?\b',
 'M10_hype':r'promises to|stands? to|is poised to|opens the door to|is set to|has the potential to|keeps .* from|stands? in the way|\bunlocks?\b',
 'M17_fancy':r'\b(?:pit(?:s|ted|ting)?|dispatch(?:es|ed|ing)?|chip(?:s|ped|ping)? (?:away )?at|marshal(?:s|led|ling)?|orchestrat(?:e|es|ed|ing)|wrangl(?:e|es|ed|ing)|harness(?:es|ed|ing)?|forge[sd]?|weav(?:e|es|ed|ing)|delv(?:e|es|ed|ing) into|usher(?:s|ed)? in|grappl(?:e|es|ed|ing) with|anew|afresh)\b',
 'M18_empty':r'In this (?:paper|section), we',
 'M12_wordiness':r'the fact that|the question (?:as to |of )?whether|as to whether|in order to|there is no doubt but|the reason .* is because|owing to the fact that|in a [a-z]+ manner|is a (?:subject|man|woman) (?:that|who)|in the last analysis|along these lines|in terms of|one of the most',
 'M13_qualifiers':r'\b(?:rather|very|pretty|little|quite|somewhat|fairly|certainly)\b',
 'M14_adverbs':r'\b(?:thusly|muchly|overly|firstly|secondly|thirdly)\b|[a-z]+wise\b',
 'M15_exclamation':r'!',
 'precision_pairs':r'\bcomprised of\b|\bdata is\b|different than|\bvery unique\b|\bdue to\b|\bless (?:than )?[0-9]',
 'PT_passive_candidates':r'\b(?:é|são|foi|foram|ser|será|serão)\s+\w+(?:ado|ada|ados|adas|ido|ida|idos|idas)\b',
 'PT_promotional':r'\b(?:inovador|inédito|revolucionário|robusto|abrangente)\b',
}
report={'files':[],'changed_line_gate_counts':{},'changed_line_gate_hits':{},'integrity':{}}
alladded=[]
for entry in entries:
 rel=entry['path']; old=(base/rel).read_text(encoding='utf-8');new=(root/rel).read_text(encoding='utf-8')
 diff=list(difflib.unified_diff(old.splitlines(),new.splitlines(),fromfile=rel+' baseline',tofile=rel+' revised'))
 alladded.extend(line[1:] for line in diff if line.startswith('+') and not line.startswith('+++'))
 report['files'].append({'path':rel,'before_chars':len(old),'after_chars':len(new),'sha256':hashlib.sha256((root/rel).read_bytes()).hexdigest()})
 (record/(Path(rel).name+'.diff')).write_text('\n'.join(diff)+'\n',encoding='utf-8')
 if rel.endswith('.tex'):
  for kind, pat in {'numerals':r'\d+(?:[.,]\d+)*','citations':r'\\cite\w*\{[^}]+\}','equations':r'\\begin\{(?:equation|align)\}.*?\\end\{(?:equation|align)\}|\\\[.*?\\\]','figures':r'\\begin\{figure\}.*?\\end\{figure\}'}.items():
   report['integrity'][rel+':'+kind]=re.findall(pat,old.replace('M5','Mfive'),re.S)==re.findall(pat,new.replace('M5','Mfive'),re.S)
text='\n'.join(alladded)
for cat,pat in patterns.items():
 hits=re.findall(pat,text,re.I|re.M)
 report['changed_line_gate_counts'][cat]=len(hits)
 if hits:report['changed_line_gate_hits'][cat]=hits
report['highlights_lengths']=[len(line) for line in (root/entries[1]['path']).read_text(encoding='utf-8').splitlines()]
(record/'writer_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='files'},ensure_ascii=False,indent=2))
