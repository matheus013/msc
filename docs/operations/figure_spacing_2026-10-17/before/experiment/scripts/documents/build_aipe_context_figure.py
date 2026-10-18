"""AIPE v3: editable vector diagram with an optional context extension.

Preserves all v2 artwork. Illustrates the method, not experimental results.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/supply_chain_analytics_article/figures"
NAME = "aipe_conceito_v3_contexto_ptbr"
NAVY, BLUE, TEAL, AMBER = "#163957", "#287DA6", "#197E74", "#A56327"
INK, GRAY, BORDER = "#25384A", "#586C7E", "#C9D7E2"


def build():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "svg.fonttype": "none",
        "figure.facecolor": "white", "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}",
    })
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1200), ylim=(0, 800))
    ax.axis("off")

    def box(x, y, w, h, face="white", edge=BORDER, lw=1.1, dashed=False):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
                          facecolor=face, edgecolor=edge, linewidth=lw,
                          linestyle=(0, (5, 3)) if dashed else "solid")
        ax.add_patch(p)
        return p

    def text(x, y, label, size=12.5, color=INK, bold=False, ha="center"):
        return ax.text(x, y, label, fontsize=size, color=color,
                       weight="bold" if bold else "normal", ha=ha, va="center", linespacing=1.35)

    def arrow(x1, y1, x2, y2, color=NAVY, dashed=False):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.45,
                                "shrinkA": 2, "shrinkB": 2, "mutation_scale": 11,
                                "linestyle": (0, (4, 3)) if dashed else "solid"})

    def heading(x, w, n, label):
        ax.add_patch(Circle((x + 25, 665), 12, color=NAVY))
        text(x + 25, 665, str(n), 12.5, "white", True)
        text(x + 48, 665, label, 15, NAVY, True, "left")
        ax.plot([x + 17, x + w - 17], [640, 640], color=BORDER, lw=.85)

    text(600, 776, "AIPE: seleção contextual para apoiar a gestão de estoques", 21, NAVY, True)
    box(18, 715, 1164, 35, face="#EEF4F8", edge="none")
    text(600, 732.5, "Especificar decisão, ciclos, portfólio, ambiente e critérios de avaliação", 13.4, NAVY)
    arrow(600, 714, 600, 697)

    box(18, 347, 304, 347, face="#FBFCFD")
    box(344, 347, 482, 347, face="#FBFCFD")
    box(848, 92, 334, 602, face="#FBFCFD")
    heading(18, 304, 1, "Caracterizar séries")
    heading(344, 482, 2, "Evidência e seleção")
    heading(848, 334, 3, "Apoiar o gestor")

    text(170, 615, "Históricos por série loja–produto", 11, GRAY)
    patterns = [
        ("Regular", [4, 5, 4, 5, 6, 5, 4, 5, 5, 4, 6, 5, 5, 4, 5, 5], BLUE),
        ("Intermitente", [0, 0, 5, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0], TEAL),
        ("Com picos", [1, 1, 1, 1, 7, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1, 1], AMBER),
    ]
    for i, (label, values, color) in enumerate(patterns):
        y = 536 - i * 70
        box(34, y, 272, 61)
        text(45, y + 44, label, 11.9, color, True, "left")
        xs = np.linspace(49, 293, len(values))
        ax.bar(xs, np.asarray(values) * 3.8, width=7, bottom=y + 10, color=color)
        ax.plot([43, 298], [y + 9, y + 9], color=BORDER, lw=.8)
    text(170, 369, "Volume · variabilidade · intervalos", 10.9, GRAY)

    box(362, 563, 446, 61, edge="#ADC4D4")
    text(585, 604, "Portfólio e política de referência", 13.4, NAVY, True)
    text(585, 581, "Ajuste no treino · consequências na validação", 11.3, GRAY)
    arrow(323, 593, 360, 593)
    for x, title, body, color in [
        (362, "PSE por perfil", "Perfil informado\nAssociação à política", TEAL),
        (601, "PSE preditivo", "Características da série\nEstimar classe ou\nconsequências", BLUE),
    ]:
        box(x, 429, 207, 112, edge=color, lw=1.3)
        box(x, 506, 207, 35, face="#EAF5F2" if color == TEAL else "#EAF2F9", edge="none")
        text(x + 103.5, 523.5, title, 13.5, color, True)
        text(x + 103.5, 470, body, 11)
        arrow(x + 103.5, 562, x + 103.5, 543, color)
        arrow(x + 103.5, 428, x + 103.5, 414, color)
    text(585, 485, "ou", 10.7, GRAY)
    ax.plot([465.5, 704.5], [414, 414], color=NAVY, lw=1.1)
    arrow(585, 414, 585, 402)
    box(379, 356, 412, 46, face=NAVY, edge=NAVY)
    text(585, 379, "Regra de intervenção\nCandidata ou política de referência", 12, "white", True)

    # Optional extension: its gate concerns the representation, not policy intervention.
    box(18, 92, 808, 235, face="#FFFBF5", edge=AMBER, dashed=True)
    text(36, 306, "Contexto adicional · módulo opcional", 14.6, AMBER, True, "left")
    text(805, 306, "Benefício ainda não avaliado", 10.6, GRAY, ha="right")
    cards = [(36, 208), (261, 233), (511, 297)]
    for x, w in cards:
        box(x, 183, w, 96, edge="#DECDB8")
    text(140, 258, "Dados adicionais", 12.6, AMBER, True)
    text(140, 223, "Cadastro · calendário\nConhecidos na decisão", 11)
    text(377.5, 258, "Escolha do gestor", 12.6, AMBER, True)
    text(377.5, 223, "Fixos / candidatos\nExcluir demais atributos", 11)
    text(659.5, 264, "Seleção dos candidatos", 12.6, AMBER, True)
    text(659.5, 217, "Reduzir custo sob\nrestrições de atendimento\nValidação interna", 11)
    arrow(245, 232, 260, 232, AMBER)
    arrow(495, 232, 510, 232, AMBER)
    text(36, 163, "Fixos preservados na variante ampliada.", 11, AMBER, ha="left")
    arrow(659.5, 182, 659.5, 146, AMBER)
    box(36, 105, 772, 39, face="#F5E9D9", edge="none")
    text(422, 124.5, "Comparar com o PSE básico; sem ganho interno, manter a versão básica.", 11.9, AMBER, True)
    # A gutter route feeds only the predictive branch; no path to the profile branch.
    ax.plot([808, 817, 817], [124.5, 124.5, 373], color=AMBER, lw=1.4, ls=(0, (4, 3)))
    # A small gap marks a crossing, not a junction with the policy-decision path.
    ax.plot([817, 817], [385, 461], color=AMBER, lw=1.4, ls=(0, (4, 3)))
    arrow(817, 461, 810, 461, AMBER, dashed=True)
    text(741, 338, "Contexto admitido", 10.6, AMBER)

    # One-way decision/evaluation path: no test information feeds feature selection.
    ax.plot([792, 836, 836], [379, 379, 584], color=NAVY, lw=1.35)
    arrow(836, 584, 864, 584)
    box(866, 528, 298, 95, face="#EAF2F7", edge="none")
    text(1015, 601, "Recomendação rastreável", 13.4, NAVY, True)
    text(1015, 559, "Política e parâmetros\nReferência e alternativas\nConsequências estimadas", 11)
    arrow(1015, 527, 1015, 517)
    box(866, 447, 298, 68)
    text(1015, 491, "Gestor decide a aplicação", 12.7, NAVY, True)
    text(1015, 467, "Entradas e escolha documentadas", 11.1, GRAY)
    arrow(1015, 446, 1015, 414)
    ax.plot([867, 1163], [428, 428], color=BORDER, lw=.9, ls=(0, (3, 3)))
    text(1015, 393, "Avaliar em teste reservado", 13.2, NAVY, True)
    text(1015, 343, "Custo e atendimento\nCasos prejudicados\nIncerteza e estabilidade", 12.2)
    arrow(1015, 303, 1015, 281)
    box(866, 181, 298, 98, face="#F0F5F8", edge="none")
    text(1015, 256, "Diagnóstico retrospectivo", 12.5, NAVY, True)
    text(1015, 218, "Seleção · portfólio\nLimites físicos do atendimento", 11.7)
    text(1015, 135, "Resultados posteriores não orientam\na decisão que está sendo avaliada.", 10.8, GRAY)

    box(18, 18, 1164, 53, face="#EEF4F8", edge="none")
    text(600, 44.5, "Objetivo: reduzir custos evitáveis sem deterioração relevante do atendimento.", 14, NAVY, True)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        kwargs = {"backend": "pgf"} if ext == "pdf" else {"dpi": 240} if ext == "png" else {}
        fig.savefig(OUT / f"{NAME}.{ext}", facecolor="white", **kwargs)
    plt.close(fig)
    manifest = {
        "kind": "Conceptual vector diagram; optional extension has no empirical results yet",
        "generator": str(Path(__file__).relative_to(ROOT)),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "preserved": ["aipe_conceito_v2_ptbr.pdf", "aipe_conceito_v2_ptbr.svg", "aipe_conceito_v2_ptbr.png"],
        "outputs_sha256": {f"{NAME}.{ext}": hashlib.sha256((OUT / f"{NAME}.{ext}").read_bytes()).hexdigest()
                           for ext in ("pdf", "svg", "png")},
    }
    (OUT / f"{NAME}_proveniencia.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build()
