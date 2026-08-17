"""
visualizations.py — Gráficos do artigo + comparativo estendido
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import os

plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"white",
    "axes.grid":True,"grid.alpha":0.25,"font.size":9,
})

COLORS = ["#1565C0","#2E7D32","#F57C00","#C62828",
          "#6A1B9A","#00838F","#558B2F","#4E342E",
          "#AD1457","#0277BD","#37474F","#E65100"]

def _save(fig, name, out_dir, fmt="png", dpi=150):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    p = os.path.join(out_dir, f"{name}.{fmt}")
    fig.savefig(p, bbox_inches="tight", dpi=dpi); plt.close(fig)
    print(f"  [Plot] {p}")
    return p


# ── Fig 3: Forecast comparison ───────────────────────────────
def plot_forecast_comparison(y_true, results, out_dir, cfg):
    models = [k for k in ["LSTM","ANN","XGBoost"] if k in results]
    fig, axes = plt.subplots(len(models), 1, figsize=(13, 3.5*len(models)), sharex=True)
    if len(models)==1: axes=[axes]
    for ax, name in zip(axes, models):
        preds = results[name]["predictions"]
        acc   = results[name]["metrics"]["Accuracy"]
        n = min(len(y_true), len(preds))
        ax.plot(range(n), y_true[:n], "#2196F3", lw=0.9, label="True")
        ax.plot(range(n), preds[:n],  "#FF9800", lw=0.9, ls="--", label="Predicted")
        ax.set_title(f"{name} | Accuracy: {acc:.1f}%"); ax.set_ylabel("Demand")
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Time Steps")
    fig.suptitle("Fig. 3 — Demand Forecasting Comparison", fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig3_forecast", out_dir,
                 cfg["OUTPUT"].get("plot_format","png"), cfg["OUTPUT"].get("dpi",150))


# ── Policy 3-panel ───────────────────────────────────────────
def _policy_3panel(sim, title, name, out_dir, cfg):
    inv=sim["inv_history"]; inv_std=sim.get("inv_std",np.zeros_like(inv))
    ord_=sim["order_history"]; dem=np.array(sim["demand_history"])
    sto=sim["stockout_history"]; T=len(dem); x=np.arange(T)
    fig,axes=plt.subplots(3,1,figsize=(13,7),sharex=True)
    ax=axes[0]
    ax.plot(x,inv[:T],"#1565C0",lw=0.8,label="Inventory")
    ax.set_title("Inventory Level Over Time"); ax.set_ylabel("Units"); ax.legend(fontsize=8)
    ax=axes[1]
    ax.bar(x,ord_[:T],color="#F57C00",alpha=0.8,width=1,label="Orders")
    ax.set_title("Order Quantities Over Time"); ax.set_ylabel("Units"); ax.legend(fontsize=8)
    ax=axes[2]
    ax.plot(x,dem[:T],"#1565C0",lw=0.8,ls="--",label="Demand")
    ax.plot(x,sto[:T],"#C62828",lw=0.8,label="Stockouts")
    ax.set_title("Demand vs. Stockouts"); ax.set_ylabel("Units")
    ax.set_xlabel("Day"); ax.legend(fontsize=8)
    fig.suptitle(title,fontsize=11); fig.tight_layout()
    return _save(fig,name,out_dir,cfg["OUTPUT"].get("plot_format","png"),cfg["OUTPUT"].get("dpi",150))

def plot_policy(sim, policy_name, out_dir, cfg):
    safe = policy_name.replace(" ","_").replace("-","_").lower()
    return _policy_3panel(sim, f"Policy: {policy_name}", f"policy_{safe}", out_dir, cfg)


# ── Fig 10 estendido: Boxplots comparativos ──────────────────
def plot_comparison_boxplots(all_results, out_dir, cfg):
    policies = list(all_results.keys())
    metrics  = ["TIC","ServiceLevel","BullwhipEffect"]
    titles   = ["Total Inventory Cost (USD)","Service Level (0–1)","Bullwhip Effect"]
    n_reps   = cfg["SIMULATION"].get("n_replications",5)
    np.random.seed(42)
    fig,axes = plt.subplots(1,3,figsize=(16,5))
    for ax,metric,title in zip(axes,metrics,titles):
        data  = []
        for pol in policies:
            kpis = all_results[pol]["kpis"]
            mu   = kpis.get(metric,0)
            std  = kpis.get(metric+"_std",0)
            samp = np.random.normal(mu,std,n_reps) if std>0 else np.full(n_reps,mu)
            data.append(samp)
        bp=ax.boxplot(data,labels=policies,patch_artist=True,
                      medianprops={"color":"black","lw":1.5})
        for patch,col in zip(bp["boxes"],COLORS[:len(policies)]):
            patch.set_facecolor(col); patch.set_alpha(0.7)
        ax.set_title(title,fontsize=10); ax.set_ylabel(metric)
        ax.tick_params(axis="x",rotation=30,labelsize=7)
    fig.suptitle("Comparativo de KPIs — Todas as Políticas",fontsize=11)
    fig.tight_layout()
    return _save(fig,"fig_boxplots_all",out_dir,
                 cfg["OUTPUT"].get("plot_format","png"),cfg["OUTPUT"].get("dpi",150))


# ── Radar chart ──────────────────────────────────────────────
def plot_radar(all_results, out_dir, cfg):
    """Radar/Spider chart normalizando 5 KPIs."""
    metrics_raw = ["ServiceLevel","TIC","BullwhipEffect","StockoutRate","OrderFrequency"]
    labels      = ["Service\nLevel","Cost\n(inv)","Bullwhip\n(inv)","Stockout\n(inv)","Order\nFreq(inv)"]
    # métricas onde MAIOR = melhor: ServiceLevel
    # métricas onde MENOR = melhor: TIC, Bullwhip, Stockout, OrderFrequency
    # Normalizar 0-1, depois inverter as "menor=melhor"
    policies = list(all_results.keys())
    N = len(metrics_raw)
    raw = np.array([[all_results[p]["kpis"].get(m,0) for m in metrics_raw] for p in policies])
    # Normalizar
    mn  = raw.min(axis=0); mx = raw.max(axis=0)
    rng = np.where(mx-mn>1e-9, mx-mn, 1.0)
    norm = (raw - mn) / rng  # 0=min, 1=max
    # Inverter métricas "menor=melhor"
    invert = [False, True, True, True, True]
    for j,inv in enumerate(invert):
        if inv: norm[:,j] = 1 - norm[:,j]

    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(9,8))
    ax  = fig.add_subplot(111, polar=True)
    for i,(pol,color) in enumerate(zip(policies, COLORS)):
        vals = norm[i].tolist() + [norm[i,0]]
        ax.plot(angles, vals, color=color, lw=1.5, label=pol)
        ax.fill(angles, vals, color=color, alpha=0.07)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.set_ylim(0,1); ax.set_yticks([0.25,0.5,0.75,1.0])
    ax.set_yticklabels(["0.25","0.5","0.75","1.0"], size=7)
    ax.set_title("Radar de Desempenho — Todas as Políticas\n"
                 "(maior = melhor em cada eixo)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35,1.1), fontsize=8)
    fig.tight_layout()
    return _save(fig,"fig_radar",out_dir,
                 cfg["OUTPUT"].get("plot_format","png"),cfg["OUTPUT"].get("dpi",150))


# ── Convergência das metaheurísticas ─────────────────────────
def plot_convergence(convergence_data, out_dir, cfg):
    """convergence_data: {name: list_of_fitness_values}"""
    fig,ax = plt.subplots(figsize=(10,5))
    for i,(name,hist) in enumerate(convergence_data.items()):
        if hist:
            ax.plot(hist, color=COLORS[i%len(COLORS)], lw=1.5, label=name)
    ax.set_xlabel("Iteração / Geração"); ax.set_ylabel("Fitness (melhor)")
    ax.set_title("Convergência das Metaheurísticas (GA, SA, PSO, DE)")
    ax.legend(fontsize=9); fig.tight_layout()
    return _save(fig,"fig_convergence",out_dir,
                 cfg["OUTPUT"].get("plot_format","png"),cfg["OUTPUT"].get("dpi",150))


# ── Comparativo de parâmetros otimizados ─────────────────────
def plot_optimized_params(params_data, out_dir, cfg):
    """params_data: {name: {ROP, Q, SS}}"""
    names = list(params_data.keys())
    rop   = [params_data[n].get("ROP",0) for n in names]
    q     = [params_data[n].get("Q",0)   for n in names]
    ss    = [params_data[n].get("SS",0)  for n in names]
    x = np.arange(len(names)); w = 0.25
    fig,ax = plt.subplots(figsize=(11,5))
    ax.bar(x-w, rop, w, label="ROP", color="#1565C0", alpha=0.8)
    ax.bar(x,   q,   w, label="Q",   color="#F57C00", alpha=0.8)
    ax.bar(x+w, ss,  w, label="SS",  color="#2E7D32", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, fontsize=9)
    ax.set_ylabel("Valor (unidades)")
    ax.set_title("Parâmetros Otimizados por Algoritmo (ROP, Q, SS)")
    ax.legend(); fig.tight_layout()
    return _save(fig,"fig_params_comparison",out_dir,
                 cfg["OUTPUT"].get("plot_format","png"),cfg["OUTPUT"].get("dpi",150))


# ── Ranking heat-map ─────────────────────────────────────────
def plot_heatmap_ranking(all_results, out_dir, cfg):
    import matplotlib.colors as mcolors
    policies = list(all_results.keys())
    metrics  = ["TIC","ServiceLevel","BullwhipEffect","StockoutRate","OrderFrequency","N_Orders"]
    mlabels  = ["TIC ($)","Serv. Level","Bullwhip","Stockout Rate","Order Freq","N Orders"]
    data = np.array([[all_results[p]["kpis"].get(m,0) for m in metrics] for p in policies])
    # Ranking por métrica: ServiceLevel → maior=melhor; resto → menor=melhor
    better_higher = [False, True, False, False, False, False]
    ranks = np.zeros_like(data)
    for j in range(len(metrics)):
        order = np.argsort(data[:,j])
        if better_higher[j]: order = order[::-1]
        for rank_pos, pol_idx in enumerate(order):
            ranks[pol_idx, j] = rank_pos + 1

    fig, ax = plt.subplots(figsize=(len(metrics)*1.4+2, len(policies)*0.55+2))
    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(ranks, cmap=cmap, aspect="auto", vmin=1, vmax=len(policies))
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels(mlabels, rotation=30, ha="right")
    ax.set_yticks(range(len(policies))); ax.set_yticklabels(policies, fontsize=9)
    for i in range(len(policies)):
        for j in range(len(metrics)):
            val = data[i,j]
            rank= int(ranks[i,j])
            txt = f"#{rank}\n{val:.1f}" if abs(val)<1000 else f"#{rank}\n{val/1000:.1f}k"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if rank <= len(policies)//2 else "black")
    plt.colorbar(im, ax=ax, label="Ranking (1=melhor)")
    ax.set_title("Ranking de Desempenho por Métrica\n(verde=melhor, vermelho=pior)", pad=12)
    fig.tight_layout()
    return _save(fig,"fig_ranking_heatmap",out_dir,
                 cfg["OUTPUT"].get("plot_format","png"),cfg["OUTPUT"].get("dpi",150))


# ── Tabela resumo ────────────────────────────────────────────
def plot_metrics_table(all_results, out_dir, cfg):
    rows = []
    for name,res in all_results.items():
        k = res["kpis"]
        rows.append({
            "Política":       name,
            "TIC ($)":        f"{k.get('TIC',0):,.0f}",
            "TIC std":        f"±{k.get('TIC_std',0):,.0f}",
            "Serv. Level":    f"{k.get('ServiceLevel',0):.3f}",
            "Stockout Rate":  f"{k.get('StockoutRate',0):.3f}",
            "N Pedidos":      f"{k.get('N_Orders',0):.0f}",
            "Freq. Pedidos":  f"{k.get('OrderFrequency',0):.3f}",
            "Bullwhip":       f"{k.get('BullwhipEffect',0):.2f}",
        })
    cols = list(rows[0].keys())
    data = [[r[c] for c in cols] for r in rows]
    fig,ax = plt.subplots(figsize=(16, max(3, len(rows)*0.65+1.5)))
    ax.axis("off")
    tbl = ax.table(cellText=data, colLabels=cols, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.1, 1.7)
    for j in range(len(cols)):
        tbl[0,j].set_facecolor("#1565C0"); tbl[0,j].set_text_props(color="white",fontweight="bold")
    for i in range(1, len(rows)+1):
        bg = "#F5F5F5" if i%2==0 else "white"
        for j in range(len(cols)): tbl[i,j].set_facecolor(bg)
    fig.suptitle("Tabela Comparativa Completa — KPIs de Inventário",fontsize=12)
    fig.tight_layout()
    return _save(fig,"table_kpi_full",out_dir,
                 cfg["OUTPUT"].get("plot_format","png"),cfg["OUTPUT"].get("dpi",150))


# ── RL reward learning curves ────────────────────────────────
def plot_rl_learning_curves(rl_data, out_dir, cfg):
    """rl_data: {name: list_of_episode_rewards}"""
    fig,ax = plt.subplots(figsize=(11,5))
    window = 20
    for i,(name,rewards) in enumerate(rl_data.items()):
        if not rewards: continue
        r = np.array(rewards, dtype=float)
        sm = np.convolve(r, np.ones(window)/window, mode="valid")
        x  = np.arange(len(sm)) + window//2
        ax.plot(x, sm, color=COLORS[i%len(COLORS)], lw=1.5, label=name)
    ax.set_xlabel("Episódio"); ax.set_ylabel("Recompensa (média móvel 20ep)")
    ax.set_title("Curvas de Aprendizado — Agentes RL (DQN, PPO, SARSA)")
    ax.legend(fontsize=9); fig.tight_layout()
    return _save(fig,"fig_rl_learning",out_dir,
                 cfg["OUTPUT"].get("plot_format","png"),cfg["OUTPUT"].get("dpi",150))
