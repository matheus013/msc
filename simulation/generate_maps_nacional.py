"""Mapas nacionais usando todos os estados — lê e agrega estado por estado."""
import sys; sys.path.insert(0, 'src')
import glob, os
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

SOURCE = Path('../data/source/vendas')
OUT = Path('data/08_reporting/maps')
OUT.mkdir(parents=True, exist_ok=True)
dpi = 150

COLS = ['revendedor_cod', 'produto_cod', 'venda_qtd',
        'venda_vlr_receita_liquida', 'segmento', 'filial', 'status', 'venda_ciclo']
USABLE_STATUS = {'ativo', 'Ativo', 'ATIVO', 'A', 'a'}

SEG_ORDER = ['Platina', 'Ouro', 'Rubi', 'Diamante GB', 'Esmeralda GB',
             'Prata', 'Bronze', 'Revendedor']
SEG_COLORS = {
    'Platina':      '#1A237E', 'Ouro':         '#F9A825',
    'Rubi':         '#B71C1C', 'Diamante GB':  '#00838F',
    'Esmeralda GB': '#2E7D32', 'Prata':        '#616161',
    'Bronze':       '#6D4C41', 'Revendedor':   '#9E9E9E',
}

print("Lendo e agregando todos os estados...")
state_chunks = []
seg_chunks = []
filial_chunks = []

paths = sorted(glob.glob(str(SOURCE / 'uf=*' / '*.parquet')))
states_with_data = []

for path in paths:
    uf = path.replace('\\', '/').split('uf=')[1].split('/')[0]
    if uf == '__HIVE_DEFAULT_PARTITION__':
        continue
    try:
        raw_df = pd.read_parquet(path)
    except Exception:
        continue

    available_cols = [c for c in COLS if c in raw_df.columns]
    if not available_cols:
        continue
    df = raw_df[available_cols].copy()

    # Filtra status ativo se disponivel
    if 'status' in df.columns:
        df = df[df['status'].str.lower().str.strip().isin(['ativo', 'a'])]

    if df.empty:
        continue

    n_lojas = df['revendedor_cod'].nunique()
    n_skus = df['produto_cod'].nunique() if 'produto_cod' in df.columns else 0
    n_series = df[['revendedor_cod', 'produto_cod']].drop_duplicates().shape[0] if 'produto_cod' in df.columns else 0
    receita = df['venda_vlr_receita_liquida'].sum() if 'venda_vlr_receita_liquida' in df.columns else 0
    qtd = df['venda_qtd'].sum() if 'venda_qtd' in df.columns else 0

    state_chunks.append({
        'uf': uf, 'n_lojas': n_lojas, 'n_skus': n_skus,
        'n_series': n_series, 'receita_total': receita, 'qtd_total': qtd,
    })
    states_with_data.append(uf)

    # Segmento
    if 'segmento' in df.columns:
        seg_grp = (df.groupby('segmento')
                   .agg(n_lojas=('revendedor_cod', 'nunique'),
                        receita=('venda_vlr_receita_liquida', 'sum'))
                   .reset_index())
        seg_grp['uf'] = uf
        seg_chunks.append(seg_grp)

    # Filial
    if 'filial' in df.columns:
        fil_grp = (df.groupby('filial')
                   .agg(n_lojas=('revendedor_cod', 'nunique'),
                        receita=('venda_vlr_receita_liquida', 'sum'))
                   .reset_index())
        fil_grp['uf'] = uf
        filial_chunks.append(fil_grp)

    print(f"  {uf}: {n_lojas:,} lojas | {n_skus} SKUs | R${receita/1e6:.1f}M")

state_df = pd.DataFrame(state_chunks)
seg_df = pd.concat(seg_chunks, ignore_index=True) if seg_chunks else pd.DataFrame()
fil_df = pd.concat(filial_chunks, ignore_index=True) if filial_chunks else pd.DataFrame()

if state_df.empty:
    print("WARN: Nenhum dado valido encontrado para gerar mapas nacionais.")
    sys.exit(0)

total_lojas = state_df['n_lojas'].sum()
total_receita = state_df['receita_total'].sum()
print(f"\nTotal: {total_lojas:,} lojas | R${total_receita/1e6:.1f}M receita")

# ── 1. Mapa do Brasil: lojas por estado (choropleth) ─────────────────────────
try:
    import geobr
    states_gdf = geobr.read_state(year=2020)
    state_col = 'abbrev_state'
    merged = states_gdf.merge(state_df, left_on=state_col, right_on='uf', how='left')

    fig, axes = plt.subplots(1, 2, figsize=(18, 11))

    for ax, col, label, cmap in [
        (axes[0], 'n_lojas', 'Número de Lojas Ativas', 'Blues'),
        (axes[1], 'receita_total', 'Receita Líquida Total (R$)', 'YlOrRd'),
    ]:
        ax.set_facecolor('#E8F4F8')
        states_gdf.plot(ax=ax, color='#ECECEC', edgecolor='white', linewidth=0.5)
        has_data = merged[merged[col].notna() & (merged[col] > 0)]
        if not has_data.empty:
            has_data.plot(ax=ax, column=col, cmap=cmap, legend=True,
                          legend_kwds={'label': label, 'shrink': 0.55,
                                       'orientation': 'vertical', 'format': '%.0f'},
                          linewidth=0.4, edgecolor='white')
        for _, row in merged.iterrows():
            try:
                centroid = row.geometry.centroid
                abbrev = str(row.get(state_col, ''))
                val = row.get(col, None)
                if val is not None and not (isinstance(val, float) and np.isnan(val)) and val > 100:
                    if col == 'n_lojas':
                        txt = f"{abbrev}\n{int(val):,}"
                    else:
                        txt = f"{abbrev}\nR${val/1e6:.0f}M"
                    ax.annotate(txt, (centroid.x, centroid.y),
                                ha='center', va='center', fontsize=6.5, fontweight='bold',
                                color='white',
                                bbox=dict(boxstyle='round,pad=0.15', fc='#333', alpha=0.55))
                else:
                    ax.annotate(abbrev, (centroid.x, centroid.y),
                                ha='center', va='center', fontsize=5.5, color='#666')
            except Exception:
                continue
        ax.set_title(label, fontsize=11)
        ax.axis('off')

    fig.suptitle(f'Rede Varejista — Cobertura Nacional\n'
                 f'{total_lojas:,} lojas ativas | R${total_receita/1e6:.0f}M receita total | '
                 f'{len(states_with_data)} estados',
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / 'nacional_mapa_lojas_receita.pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print('OK nacional_mapa_lojas_receita.pdf')
except Exception as e:
    print(f'WARN geobr: {e}')

# ── 2. Ranking de estados por lojas e receita ─────────────────────────────────
state_plot = state_df[state_df['n_lojas'] >= 100].sort_values('n_lojas', ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

colors_lojas = plt.cm.Blues(np.linspace(0.4, 0.9, len(state_plot)))
bars = axes[0].barh(state_plot['uf'], state_plot['n_lojas'],
                    color=colors_lojas, edgecolor='white', height=0.7)
for bar, val in zip(bars, state_plot['n_lojas']):
    axes[0].text(val + 20, bar.get_y() + bar.get_height() / 2,
                 f'{int(val):,}', va='center', fontsize=7.5)
axes[0].set_xlabel('Número de Lojas Ativas')
axes[0].set_title('Lojas Ativas por Estado', fontsize=10)
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

state_plot_rev = state_df[state_df['receita_total'] >= 1e6].sort_values('receita_total', ascending=True)
colors_rev = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(state_plot_rev)))
bars2 = axes[1].barh(state_plot_rev['uf'], state_plot_rev['receita_total'] / 1e6,
                     color=colors_rev, edgecolor='white', height=0.7)
for bar, val in zip(bars2, state_plot_rev['receita_total'] / 1e6):
    axes[1].text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'R${val:.0f}M', va='center', fontsize=7.5)
axes[1].set_xlabel('Receita Líquida Total (R$ milhões)')
axes[1].set_title('Receita Total por Estado', fontsize=10)

fig.suptitle('Ranking de Estados — Dataset Completo', fontsize=11)
fig.tight_layout()
fig.savefig(OUT / 'nacional_ranking_estados.pdf', dpi=dpi, bbox_inches='tight')
plt.close(fig)
print('OK nacional_ranking_estados.pdf')

# ── 3. Distribuicao de segmentos: nacional (sem Revendedor) ──────────────────
if not seg_df.empty:
    SEG_ORDER_CLEAN = [s for s in SEG_ORDER if s != 'Revendedor']
    seg_nacional = (seg_df[seg_df['segmento'] != 'Revendedor']
                    .groupby('segmento')[['n_lojas', 'receita']]
                    .sum()
                    .reindex([s for s in SEG_ORDER_CLEAN if s in seg_df['segmento'].unique()]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pizza: lojas por segmento
    axes[0].pie(seg_nacional['n_lojas'].values,
                labels=seg_nacional.index,
                colors=[SEG_COLORS.get(s, '#999') for s in seg_nacional.index],
                autopct='%1.1f%%', startangle=90, pctdistance=0.80,
                wedgeprops=dict(edgecolor='white', linewidth=1.5))
    axes[0].set_title(f'Lojas por Segmento\n({seg_nacional["n_lojas"].sum():,} lojas — Nacional)',
                      fontsize=10)

    # Barras horizontais: receita + número de lojas por segmento
    seg_plot = seg_nacional.sort_values('receita', ascending=True)
    bar_colors = [SEG_COLORS.get(s, '#999') for s in seg_plot.index]
    bars = axes[1].barh(seg_plot.index,
                        seg_plot['receita'].values / 1e6,
                        color=bar_colors, edgecolor='white', height=0.6)
    for bar, rev, n_lj in zip(bars, seg_plot['receita'].values / 1e6, seg_plot['n_lojas'].values):
        axes[1].text(bar.get_width() + 1,
                     bar.get_y() + bar.get_height() / 2,
                     f'R${rev:.0f}M  ({int(n_lj):,} lojas)',
                     va='center', fontsize=8.5, fontweight='bold')
    axes[1].set_xlabel('Receita Líquida Total (R$ milhões)')
    axes[1].set_title('Receita Total por Segmento — Nacional', fontsize=10)
    # margem direita para os labels
    axes[1].set_xlim(right=seg_plot['receita'].max() / 1e6 * 1.45)
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'R${x:.0f}M'))

    fig.suptitle('Distribuição Nacional por Segmento de Loja\n(excluindo categoria Revendedor)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / 'nacional_distribuicao_segmento.pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print('OK nacional_distribuicao_segmento.pdf')

# ── 4. Mapa de receita por segmento (top 5 estados por receita) ───────────────
if not seg_df.empty:
    top5 = state_df.nlargest(5, 'receita_total')['uf'].tolist()
    seg_top5 = seg_df[seg_df['uf'].isin(top5)]
    pivot = seg_top5.pivot_table(values='receita', index='uf', columns='segmento', aggfunc='sum')
    pivot = pivot[[c for c in SEG_ORDER if c in pivot.columns]]
    pivot = pivot.loc[top5]

    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.div(1e6).plot(kind='bar', stacked=True, ax=ax,
                        color=[SEG_COLORS.get(c, '#999') for c in pivot.columns],
                        edgecolor='white', linewidth=0.4)
    ax.set_xlabel('Estado')
    ax.set_ylabel('Receita (R$ milhões)')
    ax.set_title('Composição de Receita por Segmento — Top 5 Estados', fontsize=10)
    ax.tick_params(axis='x', rotation=15)
    ax.legend(title='Segmento', fontsize=7, title_fontsize=8, loc='upper right')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'R${x:.0f}M'))
    fig.tight_layout()
    fig.savefig(OUT / 'nacional_receita_segmento_top5.pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print('OK nacional_receita_segmento_top5.pdf')

print(f'\nTotal: {len(list(OUT.glob("nacional_*.pdf")))} mapas nacionais gerados em {OUT}/')
