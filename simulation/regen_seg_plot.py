"""Regenera apenas nacional_distribuicao_segmento.pdf com dados ja agregados."""
import glob, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})

SOURCE = Path('../data/source/vendas')
OUT = Path('data/08_reporting/maps')
dpi = 150

SEG_ORDER = ['Platina', 'Ouro', 'Rubi', 'Diamante GB', 'Esmeralda GB', 'Prata', 'Bronze']
SEG_COLORS = {
    'Platina':      '#1A237E', 'Ouro':         '#F9A825',
    'Rubi':         '#B71C1C', 'Diamante GB':  '#00838F',
    'Esmeralda GB': '#2E7D32', 'Prata':        '#616161',
    'Bronze':       '#6D4C41',
}

print("Agregando segmentos de todos os estados...")
chunks = []
for path in sorted(glob.glob(str(SOURCE / 'uf=*' / '*.parquet'))):
    uf = path.replace('\\', '/').split('uf=')[1].split('/')[0]
    if uf == '__HIVE_DEFAULT_PARTITION__':
        continue
    df = pd.read_parquet(path, columns=['revendedor_cod', 'venda_vlr_receita_liquida', 'segmento'])
    df = df[df['segmento'] != 'Revendedor']
    grp = (df.groupby('segmento')
             .agg(n_lojas=('revendedor_cod', 'nunique'),
                  receita=('venda_vlr_receita_liquida', 'sum'))
             .reset_index())
    grp['uf'] = uf
    chunks.append(grp)
    print(f"  {uf}: ok")

seg_df = pd.concat(chunks, ignore_index=True)
seg_nacional = (seg_df.groupby('segmento')[['n_lojas', 'receita']]
                .sum()
                .reindex([s for s in SEG_ORDER if s in seg_df['segmento'].unique()]))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pizza: lojas por segmento
axes[0].pie(
    seg_nacional['n_lojas'].values,
    labels=seg_nacional.index,
    colors=[SEG_COLORS.get(s, '#999') for s in seg_nacional.index],
    autopct='%1.1f%%', startangle=90, pctdistance=0.80,
    wedgeprops=dict(edgecolor='white', linewidth=1.5),
)
axes[0].set_title(
    f'Lojas por Segmento\n({seg_nacional["n_lojas"].sum():,} lojas — Nacional)',
    fontsize=10,
)

# Barras horizontais: receita + numero de lojas
seg_plot = seg_nacional.sort_values('receita', ascending=True)
bars = axes[1].barh(
    seg_plot.index,
    seg_plot['receita'].values / 1e6,
    color=[SEG_COLORS.get(s, '#999') for s in seg_plot.index],
    edgecolor='white', height=0.6,
)
for bar, rev, n_lj in zip(bars,
                           seg_plot['receita'].values / 1e6,
                           seg_plot['n_lojas'].values):
    axes[1].text(
        bar.get_width() + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f'R${rev:.0f}M  ({int(n_lj):,} lojas)',
        va='center', fontsize=8.5, fontweight='bold',
    )

axes[1].set_xlabel('Receita Liquida Total (R$ milhoes)')
axes[1].set_title('Receita Total por Segmento — Nacional', fontsize=10)
axes[1].set_xlim(right=seg_plot['receita'].max() / 1e6 * 1.50)
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'R${x:.0f}M'))

fig.suptitle('Distribuicao Nacional por Segmento de Loja\n(excluindo categoria Revendedor)',
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT / 'nacional_distribuicao_segmento.pdf', dpi=dpi, bbox_inches='tight')
plt.close(fig)
print(f'\nOK -> {OUT}/nacional_distribuicao_segmento.pdf')
