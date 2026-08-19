"""
kpi_por_segmento.py — CTI, NS e TR por segmento comercial para políticas
representativas do portfólio final (12 políticas).

Regenera a Figura kpi_por_segmento (resultados.tex, Seção "Análise
Estratificada por Filial e Segmento"), que estava desatualizada desde
26/06 (script original perdido, figura ainda citava PSO, política
removida do portfólio final -- ver PLANO_ENXUGAMENTO_TEXTO.md).

Políticas representativas (uma por família ainda presente no portfólio
final, substituindo o antigo EOQ/Jornaleiro/PSO/GA-DQN):
  EOQ (clássica), Newsvendor (clássica), SA (meta-heurística), GA-DQN (híbrida).

Uso:
    python simulation/src/reporting/kpi_por_segmento.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[2]
KPI_PATH = REPO_ROOT / "data" / "07_model_output" / "kpis.parquet"
OUT_PATH = REPO_ROOT / "data" / "08_reporting" / "maps" / "kpi_por_segmento.pdf"

POLICIES = ["EOQ", "Newsvendor", "SA", "GA-DQN"]
POLICY_DISPLAY = {"EOQ": "EOQ", "Newsvendor": "Jornaleiro", "SA": "SA", "GA-DQN": "GA-DQN"}
METRICS = ["TIC", "NS", "TR"]
METRIC_LABEL = {"TIC": "CTI médio (R\\$)", "NS": "NS médio", "TR": "TR médio"}
SEGMENT_ORDER = ["Platina", "Ouro", "Rubi", "Diamante GB", "Esmeralda GB", "Prata"]


def run(kpi_path: Path = KPI_PATH, out_path: Path = OUT_PATH) -> pd.DataFrame:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Lendo %s ...", kpi_path)
    df = pd.read_parquet(kpi_path)
    df = df[df["policy"].isin(POLICIES)].copy()

    segs = [s for s in SEGMENT_ORDER if s in df["segmento"].unique()]
    agg = df.groupby(["segmento", "policy"])[METRICS].mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    colors = {"EOQ": "#5B7183", "Newsvendor": "#237E72", "SA": "#6C4C96", "GA-DQN": "#3D4291"}
    width = 0.2
    x = range(len(segs))

    for ax, metric in zip(axes, METRICS):
        for i, pol in enumerate(POLICIES):
            sub = agg[agg["policy"] == pol].set_index("segmento").reindex(segs)
            offset = (i - (len(POLICIES) - 1) / 2) * width
            ax.bar([xi + offset for xi in x], sub[metric].to_numpy(),
                   width=width, label=POLICY_DISPLAY[pol], color=colors[pol])
        ax.set_xticks(list(x))
        ax.set_xticklabels(segs, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(METRIC_LABEL[metric], fontsize=9)
        ax.set_title(metric, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("CTI, NS e TR por segmento de loja (EOQ, Jornaleiro, SA, GA-DQN)", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Figura salva: %s", out_path)
    return agg


if __name__ == "__main__":
    run()
