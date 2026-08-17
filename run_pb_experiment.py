"""
run_pb_experiment.py
Experimento multi-nivel: UF=PB, produto 48130
  Para cada loja (revendedora): executa 12 politicas de inventario
  Arquitetura: estoque (UF) -> loja -> produto

Uso:
    python run_pb_experiment.py
    python run_pb_experiment.py --config config.yaml --out outputs/pb
"""
import sys, os, time, argparse, warnings
import numpy as np
import pandas as pd
import yaml
import copy
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _scale_cfg(cfg_base, demand, mu, std, uf, loja, produto):
    """Cria config adaptado para a demanda de cada loja."""
    cfg = copy.deepcopy(cfg_base)
    lt  = cfg["SIMULATION"].get("lead_time", 2)
    z   = cfg.get("HEURISTIC", {}).get("z_score", 1.645)
    h   = cfg["COST"].get("holding_cost_per_unit", 1.0)
    K   = cfg["COST"].get("ordering_cost_per_order", 50.0)

    # Inventario inicial = 2 × demanda_media × lead_time (mínimo 100)
    cfg["SIMULATION"]["initial_inventory"] = max(100.0, 2.0 * mu * lt)

    # EOQ e estoque de segurança de referência
    q_eoq  = float(np.sqrt(2 * mu * max(len(demand), 1) * K / max(h, 1e-6)))
    ss_ref = z * std * np.sqrt(lt)
    rop_ref = mu * lt + ss_ref

    # Espaço de busca escalado (4× referência)
    cfg["GENETIC_ALGORITHM"]["search_space"] = {
        "ROP": [0,   max(200, int(rop_ref * 4))],
        "Q":   [1,   max(200, int(q_eoq   * 4))],
        "SS":  [0,   max(100, int(ss_ref  * 4))],
    }

    # max_order_qty para RL = 4 × demanda_media (mínimo 200)
    mq = max(200, int(mu * 4))
    for sec in ("DQN", "PPO", "SARSA"):
        if sec in cfg:
            cfg[sec]["max_order_qty"] = mq

    return cfg


def _run_single(demand, cfg, uf, loja, produto, seed):
    """Roda forecasting + 12 politicas para uma serie de demanda."""
    from modules.data_loader import train_test_split, build_features
    from modules.forecasting import train_all_forecasters
    from modules.inventory_env import InventoryEnv
    from modules.policies_extended import (
        EOQPolicy, SsPolicyClass, NewsvendorPolicy,
        GAPolicyOptimizer, SimulatedAnnealingPolicy, PSOPolicy, DEPolicy,
        DQNPolicy, PPOPolicy, SARSAPolicy, HybridGADQN, HybridGAPPO,
    )

    np.random.seed(seed)

    # Split treino/teste sobre a demanda da loja (vetor puro)
    n = len(demand)
    split = int(n * cfg["DATA"].get("train_ratio", 0.7))
    train_d = demand[:split]
    test_d  = demand[split:]

    # Serie de simulacao: usa teste se tiver >= 12 periodos, senao serie completa
    sim_demand = test_d if len(test_d) >= 12 else demand

    # Features para forecasting
    lb = cfg["FORECASTING"].get("lookback", 6)
    if len(train_d) <= lb:
        lb = max(2, len(train_d) - 2)
    X_tr, y_tr = build_features(train_d, lb)
    X_te, y_te = build_features(np.concatenate([train_d[-lb:], test_d]), lb)

    results = {}

    # --- Forecasting ---
    fc_results = {}
    if len(X_tr) >= 5:
        fc_results = train_all_forecasters(X_tr, y_tr, X_te, y_te,
                                           cfg["FORECASTING"])

    # --- Ambiente padronizado ---
    n_reps = cfg["SIMULATION"].get("n_replications", 5)

    def run(name, policy_fn):
        env = InventoryEnv(sim_demand, cfg, seed=seed,
                           warehouse=uf, store_id=str(loja), product_id=str(produto))
        res = env.run_policy(policy_fn, n_reps=n_reps, base_seed=seed)
        results[name] = res
        return res

    # Classicas
    run("EOQ",       EOQPolicy(train_d, cfg))
    run("(s,S)",     SsPolicyClass(train_d, cfg))
    run("Newsvendor", NewsvendorPolicy(train_d, cfg))

    # Metaheuristicas
    ga  = GAPolicyOptimizer(train_d, cfg); ga_res  = ga.optimize(verbose=False)
    sa  = SimulatedAnnealingPolicy(train_d, cfg); sa.optimize(verbose=False)
    pso = PSOPolicy(train_d, cfg); pso.optimize(verbose=False)
    de  = DEPolicy(train_d, cfg); de.optimize(verbose=False)
    run("GA",  ga.make_policy())
    run("SA",  sa.make_policy())
    run("PSO", pso.make_policy())
    run("DE",  de.make_policy())

    # RL
    S = InventoryEnv.STATE_DIM
    dqn   = DQNPolicy(S, cfg);   dqn.train(train_d, cfg, verbose=False)
    ppo   = PPOPolicy(S, cfg);   ppo.train(train_d, cfg, verbose=False)
    sarsa = SARSAPolicy(S, cfg); sarsa.train(train_d, cfg, verbose=False)
    run("DQN",   dqn.make_policy())
    run("PPO",   ppo.make_policy())
    run("SARSA", sarsa.make_policy())

    # Hibridos
    run("GA-DQN", HybridGADQN(ga_res, dqn).make_policy())
    run("GA-PPO", HybridGAPPO(ga_res, ppo).make_policy())

    return results, fc_results, ga_res


def main(config_path="config.yaml", out_dir="outputs/pb"):
    t0 = time.time()
    cfg_base = _load_cfg(config_path)
    os.makedirs(out_dir, exist_ok=True)
    seed = cfg_base["SIMULATION"].get("random_seed", 42)

    print("=" * 65)
    print("  Experimento Multi-Nivel — UF=PB")
    print("  Arquitetura: estoque (UF) -> loja -> produto")
    print("=" * 65)

    # Carrega cenarios preparados
    cen_path  = "data/pb_cenarios.csv"
    meta_path = "data/pb_cenarios_meta.csv"
    if not os.path.exists(cen_path):
        print("[ERRO] Execute primeiro: python prepare_pb_data.py")
        sys.exit(1)

    df_cen  = pd.read_csv(cen_path)
    df_meta = pd.read_csv(meta_path)

    lojas   = df_meta["loja"].tolist()
    produto = int(df_meta["produto"].iloc[0])
    uf      = df_meta["uf"].iloc[0]

    print(f"\n  UF      : {uf}")
    print(f"  Produto : {produto}")
    print(f"  Lojas   : {len(lojas)}")
    print(f"  Ciclos  : {df_cen['venda_ciclo'].nunique()}")
    print()

    all_rows = []  # resultados finais de KPI

    for i, loja in enumerate(lojas, 1):
        meta = df_meta[df_meta["loja"] == loja].iloc[0]
        mu   = float(meta["dem_media"])
        std  = float(meta["dem_std"])
        demand = (df_cen[df_cen["loja"] == loja]
                  .sort_values("venda_ciclo")["demanda"].values.astype(float))

        print(f"\n[{i}/{len(lojas)}] Loja {loja} | "
              f"mu={mu:.1f} std={std:.1f} cv={meta['cv']:.1f} "
              f"n_p={meta['n_periodos']}")

        # Config adaptado para esta loja
        cfg = _scale_cfg(cfg_base, demand, mu, std, uf, loja, produto)

        try:
            results, fc_results, ga_res = _run_single(
                demand, cfg, uf, loja, produto, seed)

            for policy, res in results.items():
                k = res["kpis"]
                all_rows.append({
                    "uf":           uf,
                    "loja":         loja,
                    "produto":      produto,
                    "dem_media":    round(mu, 1),
                    "dem_std":      round(std, 1),
                    "cv":           round(float(meta["cv"]), 2),
                    "n_periodos":   int(meta["n_periodos"]),
                    "policy":       policy,
                    "TIC":          round(k["TIC"], 2),
                    "ServiceLevel": round(k["ServiceLevel"], 4),
                    "StockoutRate": round(k["StockoutRate"], 4),
                    "BullwhipEffect": round(k["BullwhipEffect"], 4),
                    "N_Orders":     round(k["N_Orders"], 1),
                    "OrderFrequency": round(k["OrderFrequency"], 4),
                })

            # Ranking local
            best = max(results, key=lambda x: results[x]["kpis"]["ServiceLevel"])
            print(f"  -> Melhor SL: {best} "
                  f"(SL={results[best]['kpis']['ServiceLevel']:.3f} "
                  f"TIC={results[best]['kpis']['TIC']:,.0f})")

        except Exception as e:
            print(f"  [ERRO] Loja {loja}: {e}")
            continue

    if not all_rows:
        print("\n[ERRO] Nenhum resultado gerado.")
        return

    df_res = pd.DataFrame(all_rows)
    out_csv = os.path.join(out_dir, "pb_kpis.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\n[CSV] {out_csv} ({len(df_res)} linhas)")

    # Sumário por política (media entre lojas)
    print("\n" + "=" * 80)
    print("  RANKING MEDIO (media entre lojas) — Service Level")
    print("=" * 80)
    sumario = (df_res.groupby("policy")
               .agg(SL_media=("ServiceLevel","mean"),
                    SL_std=("ServiceLevel","std"),
                    TIC_media=("TIC","mean"),
                    TIC_std=("TIC","std"),
                    BW_media=("BullwhipEffect","mean"),
                    SR_media=("StockoutRate","mean"),
                    NOrders=("N_Orders","mean"))
               .sort_values("SL_media", ascending=False))
    print(f"  {'Politica':<12} {'SL (media)':>10} {'SL(std)':>8} {'TIC(media)':>12} "
          f"{'TIC(std)':>10} {'Stockout':>9} {'N_Ped':>7} {'BW':>7}")
    print("  " + "-" * 78)
    for pol, row in sumario.iterrows():
        print(f"  {pol:<12} {row['SL_media']:>10.3f} {row['SL_std']:>8.3f} "
              f"{row['TIC_media']:>12,.0f} {row['TIC_std']:>10,.0f} "
              f"{row['SR_media']:>9.3f} {row['NOrders']:>7.1f} {row['BW_media']:>7.2f}")

    # Ranking por TIC (menor = melhor)
    print("\n  RANKING MEDIO — Custo Total (TIC, menor = melhor)")
    print("  " + "-" * 78)
    for pol, row in sumario.sort_values("TIC_media").iterrows():
        print(f"  {pol:<12} TIC={row['TIC_media']:>12,.0f}  "
              f"SL={row['SL_media']:.3f}  BW={row['BW_media']:.2f}")

    # Frequencia de vitoria
    best_sl_cnt = (df_res.sort_values("ServiceLevel", ascending=False)
                   .groupby("loja")["policy"].first().value_counts())
    best_tic_cnt = (df_res.sort_values("TIC", ascending=True)
                    .groupby("loja")["policy"].first().value_counts())
    print("\n  FREQUENCIA DE VITORIA POR POLITICA")
    print(f"  {'Politica':<12} {'Vitorias SL':>12} {'Vitorias TIC':>14}")
    print("  " + "-" * 40)
    for pol in sumario.index:
        v_sl  = best_sl_cnt.get(pol, 0)
        v_tic = best_tic_cnt.get(pol, 0)
        print(f"  {pol:<12} {v_sl:>12}    {v_tic:>12}")

    # Gera visualizacoes
    _plot_results(df_res, out_dir)

    elapsed = time.time() - t0
    print(f"\n  Concluido em {elapsed/60:.1f} min | Resultados: {os.path.abspath(out_dir)}/")
    print("=" * 65)


def _plot_results(df_res, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    lojas   = df_res["loja"].unique()
    policies = df_res["policy"].unique()
    n_pol   = len(policies)

    COLORS = ["#1565C0","#2E7D32","#F57C00","#C62828","#6A1B9A","#00838F",
              "#558B2F","#4E342E","#AD1457","#0277BD","#37474F","#E65100"]

    n_lojas = df_res["loja"].nunique()

    # 1. Medias por politica (barras sem IC)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [("ServiceLevel", "Service Level"), ("TIC", "Custo Total (TIC)"),
               ("BullwhipEffect", "Bullwhip Effect")]
    for ax, (metric, label) in zip(axes, metrics):
        grp = (df_res.groupby("policy")[metric]
               .mean()
               .sort_values(ascending=(metric != "ServiceLevel")))
        bars = ax.barh(grp.index, grp.values, color=COLORS[:len(grp)], alpha=0.85)
        for bar, val in zip(bars, grp.values):
            fmt = f"{val:.3f}" if metric != "TIC" else f"{val:,.0f}"
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    fmt, va="center", fontsize=7)
        ax.set_title(label); ax.set_xlabel("Media entre lojas")
        ax.invert_yaxis()
        ax.set_xlim(right=ax.get_xlim()[1] * 1.18)
    fig.suptitle(f"Comparativo de Politicas — UF PB, Produto 48130\n(media entre {n_lojas} lojas)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pb_politicas_barras.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {out_dir}/pb_politicas_barras.pdf")

    # 2. Heatmap SL por loja x politica
    pivot_sl = df_res.pivot_table(index="loja", columns="policy",
                                  values="ServiceLevel", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(14, n_pol * 1.2), max(5, len(lojas) * 0.6 + 1.5)))
    im = ax.imshow(pivot_sl.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n_pol)); ax.set_xticklabels(pivot_sl.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(lojas)))
    # Legenda: loja + demanda media
    labels_y = []
    for loja in pivot_sl.index:
        mu = df_res[df_res["loja"] == loja]["dem_media"].iloc[0]
        labels_y.append(f"Loja {loja}\n(mu={mu:.0f})")
    ax.set_yticklabels(labels_y, fontsize=7)
    for i in range(len(pivot_sl)):
        for j in range(n_pol):
            v = pivot_sl.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="black" if 0.3 < v < 0.7 else "white")
    plt.colorbar(im, ax=ax, label="Service Level")
    ax.set_title(f"Service Level por Loja x Politica — UF PB | Produto 48130\n"
                 f"(verde=bom, vermelho=ruim)", pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pb_heatmap_sl.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {out_dir}/pb_heatmap_sl.pdf")

    # 3. Scatter: demanda_media x SL por politica (melhor politica por loja)
    best_per_loja = (df_res.sort_values("ServiceLevel", ascending=False)
                    .groupby("loja").first().reset_index())
    fig, ax = plt.subplots(figsize=(10, 6))
    pol_colors = {p: COLORS[i % len(COLORS)] for i, p in enumerate(policies)}
    for _, row in best_per_loja.iterrows():
        ax.scatter(row["dem_media"], row["ServiceLevel"],
                   color=pol_colors[row["policy"]], s=80, zorder=3)
        ax.annotate(f"L{row['loja']}\n{row['policy']}",
                    (row["dem_media"], row["ServiceLevel"]),
                    fontsize=6.5, ha="left", xytext=(4, 2), textcoords="offset points")
    # Legenda de politicas
    from matplotlib.patches import Patch
    handles = [Patch(color=pol_colors[p], label=p) for p in policies if p in best_per_loja["policy"].values]
    ax.legend(handles=handles, fontsize=7, loc="lower right", ncol=2)
    ax.set_xlabel("Demanda media por ciclo (unidades)"); ax.set_ylabel("Melhor Service Level")
    ax.set_title("Melhor politica por loja vs. escala de demanda\nUF PB | Produto 48130")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pb_scatter_demanda_sl.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {out_dir}/pb_scatter_demanda_sl.pdf")

    # 4. Box plots SL por politica
    fig, ax = plt.subplots(figsize=(14, 5))
    data_box = [df_res[df_res["policy"] == p]["ServiceLevel"].values for p in policies]
    bp = ax.boxplot(data_box, labels=policies, patch_artist=True,
                    medianprops={"color": "black", "lw": 1.5})
    for patch, col in zip(bp["boxes"], COLORS):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax.set_title(f"Distribuicao do Service Level por Politica — {n_lojas} Lojas\nUF PB | Produto 48130")
    ax.set_ylabel("Service Level"); ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pb_boxplot_sl.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {out_dir}/pb_boxplot_sl.pdf")

    # 5. Trade-off SL x TIC (custo-servico): scatter por politica
    sumario_tradeoff = (df_res.groupby("policy")
                        .agg(SL=("ServiceLevel", "mean"),
                             TIC=("TIC", "mean"),
                             BW=("BullwhipEffect", "mean"),
                             SR=("StockoutRate", "mean"))
                        .reset_index())
    fig, ax = plt.subplots(figsize=(10, 7))
    bw_min = sumario_tradeoff["BW"].min()
    bw_max = sumario_tradeoff["BW"].max()
    bw_rng = max(bw_max - bw_min, 1e-6)
    for i, row in sumario_tradeoff.iterrows():
        size = 80 + 300 * (row["BW"] - bw_min) / bw_rng
        ax.scatter(row["TIC"], row["SL"], s=size,
                   color=COLORS[i % len(COLORS)], alpha=0.85,
                   edgecolors="white", linewidths=0.6, zorder=3)
        ax.annotate(row["policy"], (row["TIC"], row["SL"]),
                    fontsize=8, ha="left", xytext=(6, 3), textcoords="offset points")
    ax.set_xlabel("Custo Total Medio (TIC)", fontsize=10)
    ax.set_ylabel("Service Level Medio", fontsize=10)
    ax.set_title("Trade-off Custo x Servico por Politica\n"
                 f"UF PB | Produto 48130 | {n_lojas} lojas\n"
                 "(tamanho da bolha = Bullwhip Effect medio)", fontsize=10)
    ax.grid(True, alpha=0.3)
    # Quadrantes referencia
    sl_med = sumario_tradeoff["SL"].mean()
    tic_med = sumario_tradeoff["TIC"].mean()
    ax.axhline(sl_med, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(tic_med, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.text(tic_med * 0.97, sl_med * 1.001, "alto SL\nbaixo custo", fontsize=7,
            color="green", ha="right", va="bottom")
    ax.text(tic_med * 1.03, sl_med * 0.999, "baixo SL\nalto custo", fontsize=7,
            color="red", ha="left", va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pb_tradeoff_sl_tic.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {out_dir}/pb_tradeoff_sl_tic.pdf")

    # 6. Frequencia de vitoria por politica (quantas lojas cada politica vence)
    best_sl = (df_res.sort_values("ServiceLevel", ascending=False)
               .groupby("loja")["policy"].first())
    best_tic = (df_res.sort_values("TIC", ascending=True)
                .groupby("loja")["policy"].first())
    win_sl  = best_sl.value_counts().reindex(policies, fill_value=0)
    win_tic = best_tic.value_counts().reindex(policies, fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, (wins, criterion) in zip(axes, [(win_sl, "Melhor Service Level"),
                                             (win_tic, "Menor TIC (custo)")]):
        order = wins.sort_values(ascending=False)
        bars = ax.bar(range(len(order)), order.values,
                      color=[COLORS[list(policies).index(p) % len(COLORS)] for p in order.index],
                      alpha=0.85)
        for bar, val in zip(bars, order.values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order.index, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Numero de lojas vencidas")
        ax.set_title(f"Vitoria por Politica — {criterion}\n({n_lojas} lojas, UF PB | Produto 48130)")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, n_lojas * 1.2 + 1)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pb_vitoria_politica.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {out_dir}/pb_vitoria_politica.pdf")

    # 7. CV da demanda vs SL (por loja) — qual politica vence em demandas lumpier
    pol_colors = {p: COLORS[i % len(COLORS)] for i, p in enumerate(policies)}
    best_sl_df = (df_res.sort_values("ServiceLevel", ascending=False)
                  .groupby("loja").first().reset_index())
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (xcol, xlabel) in zip(axes, [("cv", "CV da demanda (std/media)"),
                                           ("dem_media", "Demanda media por ciclo")]):
        for _, row in best_sl_df.iterrows():
            ax.scatter(row[xcol], row["ServiceLevel"],
                       color=pol_colors.get(row["policy"], "gray"),
                       s=70, zorder=3, edgecolors="white", linewidths=0.5)
            ax.annotate(f"L{row['loja']}", (row[xcol], row["ServiceLevel"]),
                        fontsize=6, xytext=(3, 2), textcoords="offset points")
        from matplotlib.patches import Patch
        handles = [Patch(color=pol_colors[p], label=p)
                   for p in policies if p in best_sl_df["policy"].values]
        ax.legend(handles=handles, fontsize=7, ncol=2, loc="lower right")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Melhor Service Level alcancado", fontsize=9)
        ax.set_title(f"Perfil de demanda vs SL por loja\n(cor = politica vencedora)")
        ax.grid(True, alpha=0.3)
        # Linha de tendencia simples
        x_vals = best_sl_df[xcol].values
        y_vals = best_sl_df["ServiceLevel"].values
        if len(x_vals) > 2:
            z = np.polyfit(x_vals, y_vals, 1)
            p_line = np.poly1d(z)
            x_sorted = np.sort(x_vals)
            ax.plot(x_sorted, p_line(x_sorted), "k--", lw=1, alpha=0.5,
                    label=f"Tendencia (r={np.corrcoef(x_vals, y_vals)[0,1]:.2f})")
            ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.suptitle(f"Perfil de Demanda vs Desempenho — UF PB | Produto 48130", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pb_demanda_vs_sl.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {out_dir}/pb_demanda_vs_sl.pdf")

    # 8. Violin plots: distribuicao de TIC e BullwhipEffect por politica
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (metric, label) in zip(axes, [("TIC", "Custo Total (TIC)"),
                                            ("BullwhipEffect", "Bullwhip Effect")]):
        order_by = (df_res.groupby("policy")[metric].median()
                    .sort_values(ascending=True).index.tolist())
        data_v = [df_res[df_res["policy"] == p][metric].values for p in order_by]
        parts = ax.violinplot(data_v, positions=range(len(order_by)),
                              showmedians=True, showextrema=True)
        for i, (pc, col) in enumerate(zip(parts["bodies"],
                                           [COLORS[list(policies).index(p) % len(COLORS)]
                                            for p in order_by])):
            pc.set_facecolor(col); pc.set_alpha(0.7)
        parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.5)
        ax.set_xticks(range(len(order_by)))
        ax.set_xticklabels(order_by, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(label); ax.set_title(f"Distribuicao de {label} por Politica")
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(f"Violin Plots — {n_lojas} Lojas | UF PB | Produto 48130", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pb_violin_tic_bw.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {out_dir}/pb_violin_tic_bw.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="outputs/pb")
    args = ap.parse_args()
    main(args.config, args.out)
