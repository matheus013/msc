"""
final_report/nodes.py — Análises e tabelas da versão final da dissertação.

Substitui os scripts avulsos que existiam na raiz do projeto
(`run_m5_full_benchmark.py`, `run_ba_final_benchmark.py`,
`compare_proposta_vs_final.py`), trazendo tudo para o grafo do Kedro: as
entradas passam a vir do catálogo, as saídas são versionadas como datasets e a
execução é rastreável por `kedro run --pipeline final_report`.

Nós:
  compute_policy_redundancy   pares de políticas redundantes para os rótulos do PSE
  stratify_by_volume          desempenho por estrato de volume de demanda
  compare_with_proposal       confronto entre os KPIs da proposta e os da versão final
  build_final_latex_tables    tabelas LaTeX prontas para \\input{} na dissertação
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

KPI_COLS = ["TIC", "NS", "TR", "BE", "FP"]


def _ensure_family(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante a coluna `policy_family`.

    Artefatos gerados antes da reestruturacao do portfolio nao a possuem; o
    confronto com os numeros da proposta precisa continuar funcionando sobre
    eles.
    """
    if "policy_family" in df.columns:
        return df
    from simulation.pipelines.inventory_simulation.nodes import POLICY_FAMILY
    out = df.copy()
    out["policy_family"] = out["policy"].map(POLICY_FAMILY).fillna("other")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Redundância entre políticas
# ═══════════════════════════════════════════════════════════════════════════

def compute_policy_redundancy(kpis: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Identifica pares de políticas cujo desempenho é indistinguível.

    Motivação metodológica: o PSE aprende a partir de rótulos de política
    dominante. Quando duas políticas produzem CTI praticamente idêntico, o
    rótulo atribuído à série reflete ruído numérico em vez de vantagem
    operacional, o que degrada diretamente a aprendibilidade postulada pela
    Hipótese 2. Pares redundantes devem ser agrupados em uma única classe no
    conjunto de rótulos, permanecendo reportados individualmente no benchmark.

    Critério (configurável em `params:final_report`):
        correlacao de Spearman do CTI > rho_min  E
        diferenca mediana relativa de CTI < dif_max
    """
    cfg = params or {}
    rho_min = float(cfg.get("redundancy_rho_min", 0.98))
    dif_max = float(cfg.get("redundancy_dif_max", 0.05))

    piv = kpis.pivot_table(index=["warehouse", "store_id", "item_id"],
                           columns="policy", values="TIC")
    cor = piv.corr(method="spearman")
    cols = list(piv.columns)

    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            rho = float(cor.loc[a, b])
            denom = piv[[a, b]].max(axis=1)
            dif = float(((piv[a] - piv[b]).abs() / denom.where(denom != 0)).median())
            rows.append({
                "policy_a": a, "policy_b": b,
                "spearman_rho": round(rho, 4),
                "dif_mediana_pct": round(dif * 100, 2),
                "redundante": bool(rho > rho_min and dif < dif_max),
            })

    out = pd.DataFrame(rows).sort_values("spearman_rho", ascending=False)
    n_red = int(out["redundante"].sum())
    log.info("Redundancia: %d pares avaliados, %d redundantes (rho>%.2f e dif<%.0f%%)",
             len(out), n_red, rho_min, dif_max * 100)
    if n_red:
        for _, r in out[out["redundante"]].iterrows():
            log.info("  %s ~ %s  rho=%.4f  dif=%.2f%%",
                     r["policy_a"], r["policy_b"], r["spearman_rho"],
                     r["dif_mediana_pct"])
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Estratificação por volume
# ═══════════════════════════════════════════════════════════════════════════

def stratify_by_volume(kpis: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Desempenho por estrato de volume de demanda média.

    Reproduz a estratificação da Seção 5.2 da dissertação. É a análise que
    revela por que o custo de pedido fixo torna a reposição antieconômica em
    séries de baixo giro, independentemente da política escolhida.
    """
    cfg = params or {}
    hi = float(cfg.get("volume_high", 50.0))
    lo = float(cfg.get("volume_low", 10.0))

    def stratum(mu):
        if mu > hi:
            return f"1_alto (mu>{hi:.0f})"
        if mu > lo:
            return f"2_medio ({lo:.0f}<mu<={hi:.0f})"
        return f"3_baixo (mu<={lo:.0f})"

    df = _ensure_family(kpis)
    df["volume"] = df["mu"].apply(stratum)
    out = (df.groupby(["volume", "policy_family", "policy"])[KPI_COLS]
           .mean()
           .join(df.groupby(["volume", "policy_family", "policy"]).size().rename("n"))
           .round(3)
           .reset_index())
    log.info("Estratificacao por volume: %d combinacoes", len(out))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Comparação com os resultados da proposta
# ═══════════════════════════════════════════════════════════════════════════

def compare_with_proposal(kpis: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Confronta os KPIs da versão final com os reportados na proposta.

    Os números da proposta foram gerados por uma implementação com dois
    defeitos corrigidos desde então: o agente rotulado PPO não otimizava o
    objetivo do PPO (a atualização do ator somava a vantagem aos logits e
    treinava por erro quadrático), e a aptidão das meta-heurísticas usava soma
    ponderada com pesos fixos, cujo ponto de operação depende da escala do CTI.
    Esta comparação quantifica o efeito de cada correção, em vez de deixá-lo
    como afirmação narrativa.

    O caminho do artefato antigo vem de `params:final_report.baseline_kpis_path`.
    Se não existir, devolve DataFrame vazio e registra aviso — a ausência do
    baseline não deve derrubar o pipeline.
    """
    cfg = params or {}
    path = cfg.get("baseline_kpis_path")
    # DataFrame vazio "com coluna" (nao pd.DataFrame() puro): sem nenhuma
    # coluna, o catalogo (CSVDataset) grava um arquivo sem cabecalho e a
    # releitura subsequente por outro no do pipeline falha com
    # EmptyDataError. `len(...)` continua 0 para quem consome a jusante.
    empty = pd.DataFrame(columns=["policy"])
    if not path:
        log.warning("baseline_kpis_path nao configurado; comparacao ignorada")
        return empty

    p = Path(path)
    if not p.exists():
        log.warning("Baseline da proposta nao encontrado em %s; comparacao ignorada", p)
        return empty

    old = pd.read_parquet(p)
    a = old.groupby("policy")[KPI_COLS].mean()
    b = kpis.groupby("policy")[KPI_COLS].mean()
    comuns = sorted(set(a.index) & set(b.index))
    if not comuns:
        log.warning("Nenhuma politica em comum entre proposta e versao final")
        return empty

    rows = []
    for pol in comuns:
        row = {"policy": pol}
        for k in KPI_COLS:
            va, vb = float(a.loc[pol, k]), float(b.loc[pol, k])
            row[f"{k}_proposta"] = round(va, 3)
            row[f"{k}_final"] = round(vb, 3)
            row[f"{k}_delta_pct"] = round((vb - va) / max(abs(va), 1e-9) * 100, 1)
        rows.append(row)

    novas = sorted(set(b.index) - set(a.index))
    out = pd.DataFrame(rows)
    log.info("Comparacao proposta x final: %d politicas em comum, %d novas (%s)",
             len(comuns), len(novas), ", ".join(novas) if novas else "-")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Tabelas LaTeX
# ═══════════════════════════════════════════════════════════════════════════

def _num(v, nd=1):
    return "--" if pd.isna(v) else f"{v:,.{nd}f}".replace(",", ".")


def build_final_latex_tables(kpis: pd.DataFrame,
                             redundancy: pd.DataFrame,
                             comparison: pd.DataFrame,
                             params: dict) -> dict:
    """
    Gera as tabelas LaTeX da versão final, prontas para \\input{}.

    Retorna dict {nome_do_arquivo: conteudo}, materializado pelo catálogo como
    PartitionedDataset de arquivos .tex.
    """
    cfg = params or {}
    alpha = float(cfg.get("alpha_min", 0.70))
    out: dict[str, str] = {}

    # ── 1. benchmark completo do portfólio ───────────────────────────────
    k = _ensure_family(kpis)
    agg = (k.groupby(["policy_family", "policy"])[KPI_COLS].mean()
           .join(k.groupby(["policy_family", "policy"]).size().rename("n"))
           .sort_values("TIC"))
    linhas = [
        f"{pol} & {fam.replace('_', ' ')} & {_num(r['TIC'])} & {r['NS']:.3f} & "
        f"{r['TR']:.3f} & {_num(r['BE'], 2)} & {r['FP']:.2f} \\\\"
        for (fam, pol), r in agg.iterrows()
    ]
    out["tab_benchmark_portfolio"] = f"""% gerado pelo pipeline final_report
\\begin{{table}}[htb]
\\centering\\footnotesize
\\caption{{Desempenho medio das politicas do portfolio, ordenado por CTI.}}
\\label{{tab:benchmark_portfolio}}
\\begin{{tabular}}{{@{{}}llrrrrr@{{}}}}
\\toprule
Politica & Familia & CTI & NS & TR & BE & FP \\\\
\\midrule
{chr(10).join(linhas)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    # ── 2. redundância ───────────────────────────────────────────────────
    red = redundancy[redundancy["redundante"]] if len(redundancy) else redundancy
    if len(red):
        lr = [f"{r['policy_a']} & {r['policy_b']} & {r['spearman_rho']:.4f} & "
              f"{r['dif_mediana_pct']:.2f}\\% \\\\" for _, r in red.iterrows()]
        corpo = chr(10).join(lr)
    else:
        corpo = "\\multicolumn{4}{c}{Nenhum par redundante sob o criterio adotado} \\\\"
    out["tab_redundancia"] = f"""% gerado pelo pipeline final_report
\\begin{{table}}[htb]
\\centering\\footnotesize
\\caption{{Pares de politicas redundantes: correlacao de Spearman do CTI e
diferenca mediana relativa de custo. Pares redundantes sao agrupados em uma
unica classe no conjunto de rotulos do PSE, permanecendo reportados
individualmente no \\textit{{benchmark}}.}}
\\label{{tab:redundancia}}
\\begin{{tabular}}{{@{{}}llrr@{{}}}}
\\toprule
Politica A & Politica B & $\\rho$ Spearman & Dif. mediana de CTI \\\\
\\midrule
{corpo}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    # ── 3. proposta vs versão final ──────────────────────────────────────
    if len(comparison):
        lc = [f"{r['policy']} & {_num(r['TIC_proposta'])} & {_num(r['TIC_final'])} & "
              f"{r['TIC_delta_pct']:+.1f}\\% & {r['NS_proposta']:.3f} & "
              f"{r['NS_final']:.3f} & {r['FP_proposta']:.2f} & {r['FP_final']:.2f} \\\\"
              for _, r in comparison.iterrows()]
        out["tab_proposta_vs_final"] = f"""% gerado pelo pipeline final_report
\\begin{{table}}[htb]
\\centering\\footnotesize\\setlength{{\\tabcolsep}}{{4pt}}
\\caption{{Comparacao entre os resultados reportados na proposta e os da versao
final, para as politicas presentes nas duas execucoes. A reexecucao corrige o
gradiente do agente PPO e a formulacao da aptidao das meta-heuristicas.}}
\\label{{tab:proposta_vs_final}}
\\begin{{tabular}}{{@{{}}lrrrrrrr@{{}}}}
\\toprule
& \\multicolumn{{3}}{{c}}{{CTI}} & \\multicolumn{{2}}{{c}}{{NS}} & \\multicolumn{{2}}{{c}}{{FP}} \\\\
\\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-6}} \\cmidrule(lr){{7-8}}
Politica & Proposta & Final & $\\Delta$ & Proposta & Final & Proposta & Final \\\\
\\midrule
{chr(10).join(lc)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    # ── 4. política dominante por série ──────────────────────────────────
    ok = kpis[kpis["NS"] >= alpha]
    if len(ok):
        win = ok.loc[ok.groupby(["warehouse", "store_id", "item_id"])["TIC"].idxmin()]
        vc = win["policy"].value_counts()
        total = kpis.groupby(["warehouse", "store_id", "item_id"]).ngroups
        ld = [f"{p} & {n} & {n/len(win)*100:.1f}\\% \\\\" for p, n in vc.items()]
        out["tab_politica_dominante"] = f"""% gerado pelo pipeline final_report
\\begin{{table}}[htb]
\\centering\\footnotesize
\\caption{{Frequencia com que cada politica e dominante, definida como o menor
CTI entre as que atingem NS $\\geq$ {alpha:.2f}. {len(win)} de {total} series
possuem ao menos uma politica viavel.}}
\\label{{tab:politica_dominante}}
\\begin{{tabular}}{{@{{}}lrr@{{}}}}
\\toprule
Politica & Series & Participacao \\\\
\\midrule
{chr(10).join(ld)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    log.info("Tabelas LaTeX geradas: %s", ", ".join(sorted(out)))
    return {k: v for k, v in out.items()}
