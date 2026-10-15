"""Editable vector concept figure; illustrative series, not experiment results.

Creates a new bilingual figure without replacing manuscript figures or sources.
PDF export uses the repository's PGF/LaTeX workflow for embedded vector fonts.
"""
from pathlib import Path
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle,Polygon,Circle
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'docs/supply_chain_analytics_article/figures'
NAVY='#163957';BLUE='#287DA6';TEAL='#197E74';ORANGE='#C57430';INK='#25384A';GRAY='#647789'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12.5,'figure.facecolor':'white','svg.fonttype':'none',
    'pgf.texsystem':'pdflatex','pgf.rcfonts':False,
    'pgf.preamble':r'\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}'})

TEXT={
'ptbr':{
    'title':'AIPE: seleção contextual de políticas de reposição',
    'subtitle':'Da heterogeneidade da demanda à decisão por série loja–produto',
    'heads':['Caracterizar a demanda','Selecionar com o AIPE','Recomendar e avaliar'],
    'series':['Loja A · produto 1','Loja B · produto 1','Loja C · produto 2'],
    'patterns':['Regular','Intermitente','Com picos'],
    'features':'Volume · variabilidade\nintermitência · tendência',
    'features_label':'Características e perfil observado',
    'portfolio_title':'Portfólio de políticas candidatas',
    'portfolio_body':'Ajuste no treino · avaliação na validação',
    'alternatives':'Duas abordagens do PSE',
    'profile':'PSE por perfil','predictive':'PSE preditivo','or':'ou',
    'p_input':'Perfil informado','p_method':'Associação\nperfil → política',
    'm_input':'Características da série\n+ perfil opcional','m_method':'Modelo estima\na política',
    'decision':'Política recomendada\npara cada série loja–produto',
    'pretest':'Seleção antes de observar o teste',
    'action':'Quando e quanto repor',
    'outputs':['Série A → política A','Série B → política C','Série C → política B'],
    'test':'Avaliação em teste reservado',
    'output_label':'Exemplos de recomendações',
    'metrics':'Custo · atendimento\nestabilidade das escolhas',
    'reference':'Comparação com a seleção global',
    'bottom':'A mesma política pode ter desempenhos distintos entre séries.\nO AIPE permite avaliar quando a seleção contextual agrega valor.',
    'caption':r'Visão conceitual do Adaptive Inventory Policy Engine (AIPE). Históricos heterogêneos são caracterizados por série loja--produto. O Policy Selection Engine (PSE) pode associar o perfil informado a uma política ou estimar a política a partir das características da série, com perfil opcional. As candidatas são ajustadas no treino e a seleção utiliza evidências de validação; custo, atendimento e estabilidade são avaliados no teste reservado, em comparação com a seleção global. Séries, políticas e associações são ilustrativas; a figura não representa resultados experimentais nem pressupõe superioridade da contextualização.'
},
'en':{
    'title':'AIPE: context-aware replenishment policy selection',
    'subtitle':'From heterogeneous demand to decisions for each store–item series',
    'heads':['Characterize demand','Select with AIPE','Recommend and test'],
    'series':['Store A · item 1','Store B · item 1','Store C · item 2'],
    'patterns':['Regular','Intermittent','Spikes'],
    'features':'Volume · variability\nintermittency · trend',
    'features_label':'Demand features and observed profile',
    'portfolio_title':'Portfolio of candidate policies',
    'portfolio_body':'Fit on training · assess on validation',
    'alternatives':'Two PSE approaches',
    'profile':'Profile-based PSE','predictive':'Predictive PSE','or':'or',
    'p_input':'Given profile','p_method':'Profile-to-policy\nassociation',
    'm_input':'Series features\n+ optional profile','m_method':'Model predicts\nthe policy',
    'decision':'Recommended policy\nfor each store–item series',
    'pretest':'Selection before observing the test set',
    'action':'When and how much to order',
    'outputs':['Series A → policy A','Series B → policy C','Series C → policy B'],
    'test':'Held-out test evaluation',
    'output_label':'Illustrative recommendations',
    'metrics':'Cost · fill rate\nselection stability',
    'reference':'Compare against global selection',
    'bottom':'The same policy can perform differently across series.\nAIPE enables evaluation of when contextual selection adds value.',
    'caption':r'Conceptual overview of the Adaptive Inventory Policy Engine (AIPE). Heterogeneous histories are characterized at the store--item level. The Policy Selection Engine (PSE) either maps a given profile to a policy or predicts a policy from series features, optionally including the profile. Candidate policies are fitted on training data and selection uses validation evidence; cost, fill rate and selection stability are evaluated on held-out test data against global selection. Series, policies and mappings are illustrative; the figure neither presents experimental results nor assumes that contextual selection is superior.'
}}


def build(language):
    t=TEXT[language];fig=plt.figure(figsize=(12,6.7));ax=fig.add_axes([0,0,1,1])
    ax.set(xlim=(0,1200),ylim=(0,670));ax.axis('off')
    def box(x,y,w,h,face='white',edge='#CFDBE4',lw=1.1,r=8):
        patch=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',facecolor=face,edgecolor=edge,linewidth=lw)
        ax.add_patch(patch);return patch
    def text(x,y,label,size=12.5,color=INK,weight='normal',ha='center',va='center'):
        label=label.replace('→',r'$\rightarrow$')
        return ax.text(x,y,label,fontsize=size,color=color,weight=weight,ha=ha,va=va,linespacing=1.32)
    def arrow(x1,y1,x2,y2,color=BLUE,lw=1.5):
        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops={'arrowstyle':'-|>','color':color,'lw':lw,'shrinkA':2,'shrinkB':2,'mutation_scale':12})
    text(600,641,t['title'],22,NAVY,'bold')
    text(600,610,t['subtitle'],13.5,GRAY)
    panels=[(18,310),(352,482),(858,324)]
    for number,(x,w) in enumerate(panels,1):
        box(x,104,w,480,face='#FBFCFD',edge='#C9D7E2')
        ax.add_patch(Circle((x+26,557),12,color=NAVY))
        text(x+26,557,str(number),12.5,'white','bold')
        text(x+49,557,t['heads'][number-1],14.8,NAVY,'bold',ha='left')
        ax.plot([x+18,x+w-18],[533,533],color='#DCE5EB',lw=.8)
    regular=np.array([.46,.52,.5,.48,.58,.51,.47,.54,.49,.56,.52,.46,.57,.51,.49,.55,.5,.53,.47,.56,.51,.48,.54,.5])
    intermittent=np.array([0,0,.7,0,0,0,0,.3,0,0,0,.55,0,0,0,0,0,.42,0,0,0,0,.72,0])
    spikes=np.array([.12,.11,.13,.1,.12,.14,.11,1.,.13,.11,.12,.13,.15,.12,.1,.12,.14,.11,.13,.9,.13,.12,.11,.12])
    for i,(values,color) in enumerate(zip([regular,intermittent,spikes],[BLUE,TEAL,ORANGE])):
        y=399-i*112
        box(34,y,278,101,edge='#D5E1E9')
        text(49,y+80,t['series'][i],10.8,GRAY,ha='left')
        text(296,y+80,t['patterns'][i],12.5,color,'bold',ha='right')
        # Small store icon; all geometry is editable vector artwork.
        sx=49;sy=y+26
        ax.add_patch(Rectangle((sx,sy),34,27,facecolor='#F0F4F7',edgecolor=NAVY,lw=1.1))
        ax.add_patch(Polygon([(sx-3,sy+27),(sx+17,sy+40),(sx+37,sy+27)],facecolor='#D6E3ED',edgecolor=NAVY,lw=1.1))
        ax.add_patch(Rectangle((sx+6,sy),9,17,facecolor='white',edgecolor=NAVY,lw=.8))
        ax.add_patch(Rectangle((sx+21,sy+9),8,8,facecolor=color,edgecolor='none'))
        xs=np.linspace(105,295,len(values));ax.bar(xs,values*41,width=4.7,bottom=y+20,color=color,alpha=.92)
        ax.plot([101,302],[y+19,y+19],color='#B9C9D5',lw=.8)
    text(173,130,t['features'],11.7,GRAY)
    # AIPE: common evidence, two alternative selection mechanisms, one output.
    box(370,494,446,29,face='#EAF2F7',edge='none')
    text(593,508.5,t['features_label'],12,NAVY)
    box(370,431,446,49,edge='#AFC5D7')
    text(593,463,t['portfolio_title'],13,NAVY,'bold')
    text(593,445,t['portfolio_body'],11.3,GRAY)
    text(593,410,t['alternatives'],11.5,GRAY)
    branches=[(370,TEAL,t['profile'],t['p_input'],t['p_method']),
              (609,BLUE,t['predictive'],t['m_input'],t['m_method'])]
    for x,color,title,inputs,method in branches:
        box(x,244,207,150,face='white',edge=color,lw=1.35)
        box(x,362,207,32,face='#EAF5F2' if color==TEAL else '#EAF2F9',edge='none',r=7)
        text(x+103.5,378,title,13,color,'bold')
        text(x+103.5,333,inputs,11.5)
        arrow(x+103.5,308,x+103.5,292,color,lw=1.2)
        text(x+103.5,269,method,12,color,'bold')
        arrow(x+103.5,244,x+103.5,220,color,lw=1.25)
    text(593,322,t['or'],11,GRAY)
    ax.plot([473.5,712.5],[220,220],color=BLUE,lw=1.3)
    arrow(593,220,593,209)
    box(402,155,382,53,face=NAVY,edge=NAVY)
    text(593,181.5,t['decision'],13,'white','bold')
    text(593,126,t['pretest'],10.7,GRAY)
    # A connected path across panels keeps the concept readable at article width.
    arrow(329,508,367,508,NAVY,lw=1.6)
    arrow(836,508,874,508,NAVY,lw=1.6)
    text(1020,516,t['output_label'],11,GRAY)
    for i,(label,color) in enumerate(zip(t['outputs'],[BLUE,ORANGE,TEAL])):
        y=464-i*48
        box(876,y,288,37,face='white',edge='#D2DEE7')
        ax.add_patch(Rectangle((876,y+7),4,23,facecolor=color,edgecolor='none'))
        text(1020,y+18.5,label,12.6,color,'bold')
    arrow(1020,365,1020,340,NAVY,lw=1.2)
    text(1020,323,t['action'],12.3,NAVY,'bold')
    ax.plot([880,1160],[299,299],color='#CDDCE6',lw=1,ls=(0,(3,3)))
    text(1020,277,t['test'],12.5,NAVY,'bold')
    text(1020,241,t['metrics'],12.5)
    box(876,158,288,38,face='#EAF2F7',edge='none')
    text(1020,177,t['reference'],11.4,NAVY)
    text(1020,126,'Adaptive Inventory Policy Engine',9.8,GRAY)
    box(18,15,1164,68,face='#EEF4F8',edge='#C9D7E2')
    text(600,49,t['bottom'],13,NAVY)
    name='aipe_conceito_v2_'+language
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/(name+'.png'),dpi=250,facecolor='white')
    fig.savefig(OUT/(name+'.svg'),facecolor='white')
    svg=OUT/(name+'.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    fig.savefig(OUT/(name+'.pdf'),backend='pgf',facecolor='white')
    plt.close(fig)
    snippet='\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/'+name+'.pdf}\n\\caption{'+t['caption']+'}\n\\label{fig:aipe-conceito-v2}\n\\end{figure}\n'
    (OUT/(name+'_inserir.tex')).write_text(snippet,encoding='utf-8')
    return name


if __name__=='__main__':
    names=[build(language) for language in ['ptbr','en']]
    manifest={'generator':str(Path(__file__).relative_to(ROOT)),
        'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'kind':'Conceptual vector illustration; no measured results',
        'original_preserved':'aipe_conceito_referencia_original.png','outputs':names,
        'manuscript_modified':False}
    (OUT/'aipe_conceito_v2_proveniencia.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(manifest,indent=2,ensure_ascii=False))
