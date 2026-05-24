"""Mapas e distribuicoes usando todos os dados filtrados (23.645 lojas, 87 SKUs)."""
import sys; sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})

df = pd.read_parquet('data/02_intermediate/sales_filtered.parquet')
print(f"Dados: {df.shape[0]:,} registros | {df['store_id'].nunique():,} lojas | "
      f"{df['item_id'].nunique()} SKUs | {df[['store_id','item_id']].drop_duplicates().shape[0]:,} series")

out = Path('data/08_reporting/maps')
out.mkdir(parents=True, exist_ok=True)
dpi = 150

# Normaliza nomes de filial
df['filial'] = df['filial'].str.strip()

SEG_ORDER = ['Platina', 'Ouro', 'Rubi', 'Diamante GB', 'Esmeralda GB', 'Prata', 'Bronze', 'Revendedor']
SEG_COLORS = {
    'Platina':      '#1A237E',
    'Ouro':         '#F9A825',
    'Rubi':         '#B71C1C',
    'Diamante GB':  '#00838F',
    'Esmeralda GB': '#2E7D32',
    'Prata':        '#616161',
    'Bronze':       '#6D4C41',
    'Revendedor':   '#9E9E9E',
}
FIL_COLORS = {
    'Calcada':    '#1565C0',
    'Cajazeiras': '#E65100',
    'Lauro':      '#2E7D32',
    'Aracaju':    '#7B1FA2',
}

# Nivel de loja: agrega por ciclo bimestral
store_level = (df.groupby(['warehouse', 'store_id', 'filial', 'segmento'])
               .agg(
                   n_skus=('item_id', 'nunique'),
                   n_cycles=('venda_ciclo', 'nunique'),
                   total_revenue=('revenue', 'sum'),
                   total_demand=('demand', 'sum'),
                   mean_revenue_cycle=('revenue', 'mean'),
               ).reset_index())

fil_order = store_level.groupby('filial')['store_id'].count().sort_values(ascending=False).index.tolist()
fil_colors_list = [FIL_COLORS.get(f.strip(), '#999999') for f in fil_order]

# ── 1. Lojas e series por filial + segmento ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1a. Pizza: lojas por filial
fil_store = store_level.groupby('filial')['store_id'].nunique().reindex(fil_order)
axes[0].pie(fil_store.values, labels=fil_store.index,
            colors=fil_colors_list, autopct='%1.0f%%',
            startangle=90, pctdistance=0.80,
            wedgeprops=dict(edgecolor='white', linewidth=1.5))
axes[0].set_title(f'Lojas por Filial\n({fil_store.sum():,} lojas total)', fontsize=10)

# 1b. Barras: lojas por segmento
seg_store = store_level.groupby('segmento')['store_id'].nunique()
seg_store = seg_store.reindex([s for s in SEG_ORDER if s in seg_store.index])
bars = axes[1].bar(seg_store.index, seg_store.values,
                   color=[SEG_COLORS.get(s, '#999') for s in seg_store.index],
                   edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, seg_store.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                 f'{val:,}', ha='center', va='bottom', fontsize=7, fontweight='bold')
axes[1].set_title('Lojas por Segmento', fontsize=10)
axes[1].set_ylabel('Numero de lojas')
axes[1].tick_params(axis='x', rotation=25, labelsize=7)
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

# 1c. Barras empilhadas: segmento por filial
cross = pd.crosstab(store_level['filial'], store_level['segmento'])
cross = cross.reindex(fil_order)
cross = cross[[c for c in SEG_ORDER if c in cross.columns]]
cross.plot(kind='bar', stacked=True, ax=axes[2],
           color=[SEG_COLORS.get(c, '#999') for c in cross.columns],
           edgecolor='white', linewidth=0.4)
axes[2].set_title('Composicao de Segmentos por Filial', fontsize=10)
axes[2].set_xlabel('')
axes[2].set_ylabel('Numero de lojas')
axes[2].tick_params(axis='x', rotation=15, labelsize=8)
axes[2].legend(title='Segmento', fontsize=6, title_fontsize=7,
               loc='upper right', ncol=2)
axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

fig.suptitle('Distribuicao de Lojas — Dataset Completo (BA, 23.645 lojas)', fontsize=11)
fig.tight_layout()
fig.savefig(out / 'full_distribuicao_lojas.pdf', dpi=dpi, bbox_inches='tight')
plt.close(fig)
print('OK full_distribuicao_lojas.pdf')

# ── 2. Receita e demanda por filial e segmento ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 2a. Receita total por filial (barras horizontais)
fil_rev = (store_level.groupby('filial')['total_revenue'].sum()
           .reindex(fil_order) / 1e6)
bars = axes[0].barh(fil_rev.index, fil_rev.values,
                    color=fil_colors_list, edgecolor='white', height=0.55)
for bar, val in zip(bars, fil_rev.values):
    axes[0].text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'R${val:.1f}M', va='center', fontsize=9, fontweight='bold')
axes[0].set_xlabel('Receita Liquida Total (R$ milhoes)')
axes[0].set_title('Receita Total por Filial\n(Dataset Completo)', fontsize=10)

# 2b. Receita media por ciclo por segmento (violin)
segs_present = [s for s in SEG_ORDER if s in store_level['segmento'].values]
data_violin = [store_level[store_level['segmento'] == s]['mean_revenue_cycle'].values
               for s in segs_present]
vp = axes[1].violinplot(data_violin, positions=range(len(segs_present)),
                        showmedians=True, showextrema=True)
for pc, seg in zip(vp['bodies'], segs_present):
    pc.set_facecolor(SEG_COLORS.get(seg, '#999'))
    pc.set_alpha(0.75)
axes[1].set_xticks(range(len(segs_present)))
axes[1].set_xticklabels(segs_present, rotation=25, ha='right', fontsize=7)
axes[1].set_ylabel('Receita media por ciclo (R$)')
axes[1].set_title('Distribuicao de Receita Media\npor Segmento', fontsize=10)

fig.suptitle('Distribuicao de Receita — Dataset Completo (BA)', fontsize=11)
fig.tight_layout()
fig.savefig(out / 'full_distribuicao_receita.pdf', dpi=dpi, bbox_inches='tight')
plt.close(fig)
print('OK full_distribuicao_receita.pdf')

# ── 3. SKUs por loja por segmento ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
seg_sku = store_level.groupby('segmento')['n_skus'].agg(['mean', 'median', 'max'])
seg_sku = seg_sku.reindex([s for s in SEG_ORDER if s in seg_sku.index])
x = np.arange(len(seg_sku))
w = 0.28
ax.bar(x - w, seg_sku['mean'], width=w, label='Media',
       color='#1565C0', alpha=0.85, edgecolor='white')
ax.bar(x, seg_sku['median'], width=w, label='Mediana',
       color='#43A047', alpha=0.85, edgecolor='white')
ax.bar(x + w, seg_sku['max'], width=w, label='Maximo',
       color='#E53935', alpha=0.85, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(seg_sku.index, rotation=20, ha='right', fontsize=8)
ax.set_ylabel('Numero de SKUs por loja')
ax.set_title('SKUs por Loja por Segmento — Dataset Completo (23.645 lojas)', fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / 'full_skus_por_loja_segmento.pdf', dpi=dpi, bbox_inches='tight')
plt.close(fig)
print('OK full_skus_por_loja_segmento.pdf')

# ── 4. Mapa Brasil (geobr) com BA destacado e metricas reais ──────────────────
try:
    import geopandas as gpd
    import geobr
    states = geobr.read_state(year=2020)
    state_col = 'abbrev_state'

    store_stats = (store_level.groupby('warehouse')
                   .agg(n_lojas=('store_id', 'nunique'),
                        n_skus_total=('n_skus', 'sum'),
                        receita_total=('total_revenue', 'sum'))
                   .reset_index())
    store_stats['receita_M'] = store_stats['receita_total'] / 1e6

    merged = states.merge(store_stats, left_on=state_col, right_on='warehouse', how='left')

    fig, ax = plt.subplots(figsize=(9, 11))
    ax.set_facecolor('#E8F4F8')
    states.plot(ax=ax, color='#ECECEC', edgecolor='white', linewidth=0.5)
    has_data = merged[merged['n_lojas'].notna()]
    if not has_data.empty:
        has_data.plot(ax=ax, column='n_lojas', cmap='Blues',
                      legend=True,
                      legend_kwds={'label': 'Numero de Lojas', 'shrink': 0.5,
                                   'orientation': 'vertical'},
                      linewidth=0.5, edgecolor='white')
    for _, row in merged.iterrows():
        try:
            centroid = row.geometry.centroid
            abbrev = str(row.get(state_col, ''))
            n = row.get('n_lojas', None)
            rev = row.get('receita_M', None)
            if n is not None and not (isinstance(n, float) and np.isnan(n)):
                label = f'{abbrev}\n{int(n):,} lojas\nR${rev:.1f}M'
                ax.annotate(label, (centroid.x, centroid.y),
                            ha='center', va='center', fontsize=7,
                            fontweight='bold', color='white',
                            bbox=dict(boxstyle='round,pad=0.2', fc='#1565C0', alpha=0.7))
            else:
                ax.annotate(abbrev, (centroid.x, centroid.y),
                            ha='center', va='center', fontsize=6, color='#888')
        except Exception:
            continue
    ax.set_title('Distribuicao de Lojas por Estado\n'
                 f'Dataset Completo: {store_stats["n_lojas"].sum():,.0f} lojas | '
                 f'R${store_stats["receita_total"].sum()/1e6:.1f}M receita',
                 fontsize=10)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(out / 'full_brazil_store_map.pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print('OK full_brazil_store_map.pdf')
except Exception as e:
    print(f'WARN brazil map: {e}')

# ── 5. Evolucao mensal da receita por filial (serie temporal) ─────────────────
df['mes'] = df['venda_data'].dt.to_period('M')
ts = (df.groupby(['filial', 'mes'])['revenue']
      .sum().reset_index())
ts['mes_dt'] = ts['mes'].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(12, 5))
for fil, color in zip(fil_order, fil_colors_list):
    sub = ts[ts['filial'] == fil].sort_values('mes_dt')
    if sub.empty:
        continue
    ax.plot(sub['mes_dt'], sub['revenue'] / 1e3, label=fil, color=color, lw=1.8)
    ax.fill_between(sub['mes_dt'], sub['revenue'] / 1e3, alpha=0.08, color=color)
ax.set_xlabel('Mes')
ax.set_ylabel('Receita Liquida (R$ mil)')
ax.set_title('Evolucao Mensal da Receita por Filial — Dataset Completo (BA)', fontsize=10)
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b/%y'))
ax.tick_params(axis='x', rotation=30)
fig.tight_layout()
fig.savefig(out / 'full_evolucao_receita_temporal.pdf', dpi=dpi, bbox_inches='tight')
plt.close(fig)
print('OK full_evolucao_receita_temporal.pdf')

# ── 6. Cobertura de SKUs por filial (heatmap normalizado) ────────────────────
sku_fil = pd.crosstab(df['filial'], df['item_id'])
sku_fil_norm = sku_fil.div(sku_fil.max())

fig, ax = plt.subplots(figsize=(max(10, len(sku_fil.columns)*0.4 + 2), 3.5))
im = ax.imshow(sku_fil_norm.values, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(sku_fil.columns)))
ax.set_xticklabels(sku_fil.columns, rotation=60, ha='right', fontsize=6)
ax.set_yticks(range(len(sku_fil.index)))
ax.set_yticklabels(sku_fil.index, fontsize=9)
fig.colorbar(im, ax=ax, label='Volume de vendas (normalizado)', shrink=0.8)
ax.set_title('Cobertura de SKUs por Filial (volume normalizado) — Dataset Completo', fontsize=10)
fig.tight_layout()
fig.savefig(out / 'full_cobertura_sku_filial.pdf', dpi=dpi, bbox_inches='tight')
plt.close(fig)
print('OK full_cobertura_sku_filial.pdf')

print(f'\nTotal: {len(list(out.glob("full_*.pdf")))} arquivos gerados em {out}/')
