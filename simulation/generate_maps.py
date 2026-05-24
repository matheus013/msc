"""Script temporário: gera visualizações de distribuição de lojas e vendas."""
import sys; sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})

meta = pd.read_parquet('data/03_primary/scenarios_meta.parquet')
kpis = pd.read_parquet('data/07_model_output/kpis.parquet')

out = Path('data/08_reporting/maps')
out.mkdir(parents=True, exist_ok=True)
dpi = 150

SEG_ORDER = ['Platina', 'Ouro', 'Rubi', 'Diamante GB', 'Esmeralda GB', 'Prata']
SEG_COLORS = {
    'Platina': '#1A237E', 'Ouro': '#F9A825', 'Rubi': '#B71C1C',
    'Diamante GB': '#00838F', 'Esmeralda GB': '#2E7D32', 'Prata': '#616161'
}
FIL_ORDER = ['Calcada', 'Cajazeiras', 'Lauro']
FIL_COLORS = {'Calcada': '#1565C0', 'Cajazeiras': '#E65100', 'Lauro': '#2E7D32'}

series_meta = meta[['warehouse', 'store_id', 'item_id', 'filial', 'segmento',
                     'mu_revenue', 'mu', 'cv']].drop_duplicates()

# Normalize filial names (remove encoding issues)
series_meta = series_meta.copy()
series_meta['filial_clean'] = series_meta['filial'].str.strip()

# ── 1. Distribuição de lojas: filial x segmento ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

fil_store_counts = series_meta.groupby('filial_clean')['store_id'].nunique().sort_values(ascending=False)
fil_colors_list = ['#1565C0', '#E65100', '#2E7D32']

axes[0].pie(fil_store_counts.values,
            labels=fil_store_counts.index,
            colors=fil_colors_list[:len(fil_store_counts)],
            autopct='%1.0f%%', startangle=90, pctdistance=0.82,
            wedgeprops=dict(edgecolor='white', linewidth=1.5))
axes[0].set_title('Lojas por Filial\n(Bahia — 145 series)', fontsize=10)

cross = pd.crosstab(series_meta['filial_clean'], series_meta['segmento'])
cross = cross[[c for c in SEG_ORDER if c in cross.columns]]
cross.plot(kind='bar', stacked=True, ax=axes[1],
           color=[SEG_COLORS.get(c, '#999') for c in cross.columns],
           edgecolor='white', linewidth=0.5)
axes[1].set_title('Series por Filial e Segmento', fontsize=10)
axes[1].set_xlabel('Filial')
axes[1].set_ylabel('Numero de series')
axes[1].tick_params(axis='x', rotation=15)
axes[1].legend(title='Segmento', fontsize=7, title_fontsize=8, loc='upper right')

fig.suptitle('Distribuicao de Lojas por Filial e Segmento (BA)', fontsize=11)
fig.tight_layout()
fig.savefig(out / 'distribuicao_lojas_segmento.pdf', dpi=dpi, bbox_inches='tight')
plt.close(fig)
print('OK distribuicao_lojas_segmento.pdf')

# ── 2. Receita esperada por filial e segmento ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

filiais = series_meta['filial_clean'].unique()
data_box = [series_meta[series_meta['filial_clean'] == f]['mu_revenue'].values for f in filiais]
bp = axes[0].boxplot(data_box, labels=filiais, patch_artist=True)
for patch, color in zip(bp['boxes'], fil_colors_list[:len(filiais)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
axes[0].set_title('Receita Esperada por Ciclo (R$)\npor Filial', fontsize=10)
axes[0].set_ylabel('Receita media por ciclo (R$)')
axes[0].tick_params(axis='x', rotation=10)

seg_rev = (series_meta.groupby('segmento')['mu_revenue']
           .mean().reindex([s for s in SEG_ORDER if s in series_meta['segmento'].unique()]))
bars = axes[1].barh(seg_rev.index, seg_rev.values,
                    color=[SEG_COLORS.get(s, '#999') for s in seg_rev.index],
                    edgecolor='white', height=0.6)
axes[1].set_xlabel('Receita media por ciclo (R$)')
axes[1].set_title('Receita Media por Ciclo\npor Segmento de Loja', fontsize=10)
for bar, val in zip(bars, seg_rev.values):
    axes[1].text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'R${val:.0f}', va='center', fontsize=8)

fig.suptitle('Distribuicao de Receita por Segmento e Filial (BA)', fontsize=11)
fig.tight_layout()
fig.savefig(out / 'distribuicao_receita_segmento.pdf', dpi=dpi, bbox_inches='tight')
plt.close(fig)
print('OK distribuicao_receita_segmento.pdf')

# ── 3. Perfil de demanda mu x CV por filial ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for fil, color in zip(filiais, fil_colors_list):
    sub = series_meta[series_meta['filial_clean'] == fil]
    ax.scatter(sub['mu'], sub['cv'], label=f'{fil} ({len(sub)} series)',
               color=color, alpha=0.65, s=45, edgecolors='white', linewidths=0.4)

ax.axhline(y=0.49**0.5, color='gray', ls='--', lw=0.8, label='CV = 0,70 (limiar SB)')
ax.set_xlabel('Demanda Media por Ciclo (mu)', fontsize=9)
ax.set_ylabel('Coeficiente de Variacao (CV)', fontsize=9)
ax.set_title('Perfil de Demanda por Filial\n(mu x CV - todas as series BA)', fontsize=10)
ax.legend(fontsize=8)
ax.set_xscale('log')
fig.tight_layout()
fig.savefig(out / 'perfil_demanda_filial.pdf', dpi=dpi, bbox_inches='tight')
plt.close(fig)
print('OK perfil_demanda_filial.pdf')

# ── 4. KPI medio por segmento (top 4 politicas) ───────────────────────────────
TOP_POLS = ['EOQ', 'Newsvendor', 'PSO', 'GA-DQN']
kpi_seg = (kpis[kpis['policy'].isin(TOP_POLS)]
           .groupby(['segmento', 'policy'])[['TIC', 'NS', 'TR']]
           .mean().reset_index())

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
kpis_show = [('TIC', 'CTI (R$)'), ('NS', 'Nivel de Servico'), ('TR', 'Taxa de Ruptura')]
for ax, (col, label) in zip(axes, kpis_show):
    pivot = kpi_seg.pivot(index='segmento', columns='policy', values=col)
    pivot = pivot.reindex([s for s in SEG_ORDER if s in pivot.index])
    pivot = pivot[[p for p in TOP_POLS if p in pivot.columns]]
    pivot.plot(kind='bar', ax=ax, edgecolor='white', linewidth=0.4)
    ax.set_title(label, fontsize=9)
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=30, labelsize=7)
    ax.legend(fontsize=7, title='Politica', title_fontsize=7)
    ax.set_ylabel(label)

fig.suptitle('KPIs por Segmento de Loja - 4 Politicas Selecionadas (BA)', fontsize=10)
fig.tight_layout()
fig.savefig(out / 'kpi_por_segmento.pdf', dpi=dpi, bbox_inches='tight')
plt.close(fig)
print('OK kpi_por_segmento.pdf')

# ── 5. Heatmap receita filial x SKU ───────────────────────────────────────────
pivot_rev = series_meta.pivot_table(values='mu_revenue',
                                     index='filial_clean', columns='item_id',
                                     aggfunc='mean')
fig, ax = plt.subplots(figsize=(max(8, len(pivot_rev.columns) * 0.55 + 2), 4))
im = ax.imshow(pivot_rev.values, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(pivot_rev.columns)))
ax.set_xticklabels(pivot_rev.columns, rotation=60, ha='right', fontsize=7)
ax.set_yticks(range(len(pivot_rev.index)))
ax.set_yticklabels(pivot_rev.index, fontsize=9)
fig.colorbar(im, ax=ax, label='Receita media por ciclo (R$)', shrink=0.8)
ax.set_title('Receita Media por Ciclo - Filial x SKU (BA)', fontsize=10)
for i in range(len(pivot_rev.index)):
    for j in range(len(pivot_rev.columns)):
        val = pivot_rev.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=5, color='black')
fig.tight_layout()
fig.savefig(out / 'heatmap_receita_filial_sku.pdf', dpi=dpi, bbox_inches='tight')
plt.close(fig)
print('OK heatmap_receita_filial_sku.pdf')

# ── 6. Distribuicao do numero de lojas por segmento (barras + contagem) ───────
fig, ax = plt.subplots(figsize=(8, 5))
seg_store = series_meta.groupby('segmento')['store_id'].nunique().reindex(
    [s for s in SEG_ORDER if s in series_meta['segmento'].unique()])
bars = ax.bar(seg_store.index,
              seg_store.values,
              color=[SEG_COLORS.get(s, '#999') for s in seg_store.index],
              edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, seg_store.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xlabel('Segmento')
ax.set_ylabel('Numero de lojas')
ax.set_title('Numero de Lojas por Segmento de Loja (BA)\nPiramide de Fidelidade', fontsize=10)
ax.tick_params(axis='x', rotation=15)
fig.tight_layout()
fig.savefig(out / 'lojas_por_segmento.pdf', dpi=dpi, bbox_inches='tight')
plt.close(fig)
print('OK lojas_por_segmento.pdf')

print('\nTodos os plots gerados em data/08_reporting/maps/')
