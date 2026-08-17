"""Gera graficos a partir dos resultados ja calculados em outputs/pb/pb_kpis.csv"""
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs/pb"
df = pd.read_csv(os.path.join(OUT, "pb_kpis.csv"))
os.makedirs(OUT, exist_ok=True)

lojas    = df["loja"].unique()
policies = df["policy"].unique()
n_pol    = len(policies)
n_loja   = len(lojas)

COLORS = ["#1565C0","#2E7D32","#F57C00","#C62828","#6A1B9A","#00838F",
          "#558B2F","#4E342E","#AD1457","#0277BD","#37474F","#E65100"]
pol_color = {p: COLORS[i % len(COLORS)] for i, p in enumerate(policies)}

# ─── 1. Barras horizontais: media SL, TIC, BW por politica ─────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
metrics = [("ServiceLevel","Service Level (media)"),
           ("TIC","Custo Total Medio (TIC, R$)"),
           ("BullwhipEffect","Bullwhip Effect (media)")]
for ax, (m, lbl) in zip(axes, metrics):
    ascending = (m != "ServiceLevel")
    grp = (df.groupby("policy")[m]
           .agg(["mean","std"])
           .sort_values("mean", ascending=ascending))
    colors = [pol_color[p] for p in grp.index]
    ax.barh(grp.index, grp["mean"], xerr=grp["std"],
            color=colors, alpha=0.8, error_kw={"elinewidth":1,"capsize":3})
    ax.set_title(lbl, fontsize=10)
    ax.set_xlabel("Media entre lojas")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25, axis="x")
fig.suptitle("Comparativo de Politicas — UF PB | Produto 48130 | 15 Lojas",
             fontsize=11, fontweight="bold")
fig.tight_layout()
p = os.path.join(OUT, "pb_barras.png")
fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"  [Plot] {p}")

# ─── 2. Heatmap SL por loja x politica ─────────────────────────
pivot_sl = df.pivot_table(index="loja", columns="policy",
                          values="ServiceLevel", aggfunc="mean")
# Ordenar colunas por SL medio decrescente
col_order = df.groupby("policy")["ServiceLevel"].mean().sort_values(ascending=False).index
pivot_sl  = pivot_sl[col_order]
# Ordenar lojas por demanda media decrescente
loja_order = df.groupby("loja")["dem_media"].first().sort_values(ascending=False).index
pivot_sl   = pivot_sl.loc[loja_order]

fig, ax = plt.subplots(figsize=(max(14, n_pol * 1.3), max(5, n_loja * 0.7 + 1.5)))
im = ax.imshow(pivot_sl.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(n_pol))
ax.set_xticklabels(pivot_sl.columns, rotation=30, ha="right", fontsize=8)
ax.set_yticks(range(n_loja))
# Label y: loja + mu
labels_y = []
for loja in pivot_sl.index:
    mu = df[df["loja"]==loja]["dem_media"].iloc[0]
    labels_y.append(f"Loja {loja}  (mu={mu:.0f})")
ax.set_yticklabels(labels_y, fontsize=7.5)
for i in range(n_loja):
    for j in range(n_pol):
        v = pivot_sl.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                color="black" if 0.25 < v < 0.75 else "white")
plt.colorbar(im, ax=ax, label="Service Level")
ax.set_title("Service Level por Loja x Politica\nUF PB | Produto 48130 | (verde=otimo, vermelho=ruim)",
             pad=12, fontsize=10)
fig.tight_layout()
p = os.path.join(OUT, "pb_heatmap_sl.png")
fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"  [Plot] {p}")

# ─── 3. Box plots SL por politica (ordenados por mediana) ───────
grp_order = (df.groupby("policy")["ServiceLevel"]
             .median().sort_values(ascending=False).index)
data_box = [df[df["policy"] == p]["ServiceLevel"].values for p in grp_order]
colors_box = [pol_color[p] for p in grp_order]

fig, ax = plt.subplots(figsize=(14, 5))
bp = ax.boxplot(data_box, labels=grp_order, patch_artist=True,
                medianprops={"color":"black","lw":1.5})
for patch, col in zip(bp["boxes"], colors_box):
    patch.set_facecolor(col); patch.set_alpha(0.75)
ax.set_title("Distribuicao do Service Level por Politica — 15 Lojas\nUF PB | Produto 48130")
ax.set_ylabel("Service Level"); ax.tick_params(axis="x", rotation=30)
ax.grid(True, alpha=0.3, axis="y"); ax.set_ylim(-0.05, 1.05)
fig.tight_layout()
p = os.path.join(OUT, "pb_boxplot_sl.png")
fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"  [Plot] {p}")

# ─── 4. Scatter: demanda_media x SL (melhor politica por loja) ──
best_per_loja = (df.sort_values("ServiceLevel", ascending=False)
                 .groupby("loja").first().reset_index())
fig, ax = plt.subplots(figsize=(10, 6))
for _, row in best_per_loja.iterrows():
    ax.scatter(row["dem_media"], row["ServiceLevel"],
               color=pol_color.get(row["policy"], "#333"),
               s=90, zorder=3, edgecolors="white", linewidths=0.5)
    ax.annotate(f"{row['policy']}\nL{row['loja']}",
                (row["dem_media"], row["ServiceLevel"]),
                fontsize=6, xytext=(5, 3), textcoords="offset points")
from matplotlib.patches import Patch
pol_presentes = best_per_loja["policy"].unique()
handles = [Patch(color=pol_color[p], label=p) for p in col_order if p in pol_presentes]
ax.legend(handles=handles, fontsize=7, loc="lower right", ncol=2)
ax.set_xlabel("Demanda media por ciclo (unidades/ciclo)", fontsize=9)
ax.set_ylabel("Melhor Service Level alcancado", fontsize=9)
ax.set_title("Melhor politica por loja x escala de demanda\nUF PB | Produto 48130", fontsize=10)
ax.grid(True, alpha=0.3); ax.set_xscale("log")
fig.tight_layout()
p = os.path.join(OUT, "pb_scatter_melhor.png")
fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"  [Plot] {p}")

# ─── 5. Tabela resumo final ──────────────────────────────────────
sumario = (df.groupby("policy")
           .agg(SL_media=("ServiceLevel","mean"),
                SL_std=("ServiceLevel","std"),
                TIC_media=("TIC","mean"),
                BW_media=("BullwhipEffect","mean"),
                SR_media=("StockoutRate","mean"))
           .sort_values("SL_media", ascending=False)
           .reset_index())
sumario.columns = ["Politica","SL medio","SL std","TIC medio","BW medio","SR medio"]

fig, ax = plt.subplots(figsize=(14, max(3, len(sumario)*0.65 + 1.5)))
ax.axis("off")
cols = list(sumario.columns)
data = []
for _, row in sumario.iterrows():
    data.append([row["Politica"],
                 f"{row['SL medio']:.3f}",
                 f"±{row['SL std']:.3f}",
                 f"{row['TIC medio']:,.0f}",
                 f"{row['BW medio']:.2f}",
                 f"{row['SR medio']:.3f}"])
tbl = ax.table(cellText=data, colLabels=cols, cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.1, 1.7)
for j in range(len(cols)):
    tbl[0, j].set_facecolor("#1565C0")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(data)+1):
    bg = "#F5F5F5" if i % 2 == 0 else "white"
    for j in range(len(cols)):
        tbl[i, j].set_facecolor(bg)
fig.suptitle("Tabela Comparativa — 12 Politicas | UF PB | Produto 48130 | 15 Lojas",
             fontsize=11, fontweight="bold")
fig.tight_layout()
p = os.path.join(OUT, "pb_tabela_resumo.png")
fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"  [Plot] {p}")

print("\nTodos os graficos gerados em outputs/pb/")
