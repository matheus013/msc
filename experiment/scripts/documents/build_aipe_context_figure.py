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
    fig = plt.figure(figsize=(12.8, 8.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1280), ylim=(0, 860))
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

    arrow_lengths = []

    def arrow(x1, y1, x2, y2, color=NAVY, dashed=False):
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < 28:
            raise ValueError("Arrow requires at least 28 layout units of clearance")
        arrow_lengths.append(length)
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.45,
                                "shrinkA": 2, "shrinkB": 2, "mutation_scale": 11,
                                "linestyle": (0, (4, 3)) if dashed else "solid"})

    def heading(x, w, n, label):
        ax.add_patch(Circle((x + 25, 701), 12, color=NAVY))
        text(x + 25, 701, str(n), 12.5, "white", True)
        text(x + 48, 701, label, 15, NAVY, True, "left")
        ax.plot([x + 17, x + w - 17], [676, 676], color=BORDER, lw=.85)

    text(640, 836, "AIPE: seleção contextual para apoiar a gestão de estoques", 21, NAVY, True)
    box(18, 772, 1244, 35, face="#EEF4F8", edge="none")
    text(640, 789.5, "Especificar decisão, ciclos, portfólio, ambiente e critérios de avaliação", 13.4, NAVY)
    arrow(569, 768, 569, 734)

    box(18, 304, 280, 426, face="#FBFCFD")
    box(338, 304, 462, 426, face="#FBFCFD")
    box(880, 73, 382, 657, face="#FBFCFD")
    heading(18, 280, 1, "Caracterizar séries")
    heading(338, 462, 2, "Evidência e seleção")
    heading(880, 382, 3, "Apoiar o gestor")

    text(158, 653, "Históricos por série loja–produto", 10.6, GRAY)
    patterns = [
        ("Regular", [4, 5, 4, 5, 6, 5, 4, 5, 5, 4, 6, 5, 5, 4, 5, 5], BLUE),
        ("Intermitente", [0, 0, 5, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0], TEAL),
        ("Com picos", [1, 1, 1, 1, 7, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1, 1], AMBER),
    ]
    for i, (label, values, color) in enumerate(patterns):
        y = 574 - i * 80
        box(34, y, 248, 69)
        text(45, y + 51, label, 11.9, color, True, "left")
        xs = np.linspace(49, 269, len(values))
        ax.bar(xs, np.asarray(values) * 4.2, width=7, bottom=y + 11, color=color)
        ax.plot([43, 274], [y + 10, y + 10], color=BORDER, lw=.8)
    text(158, 353, "Volume · variabilidade\nIntervalos entre demandas", 11.2, GRAY)

    box(356, 596, 426, 61, edge="#ADC4D4")
    text(569, 637, "Portfólio e política de referência", 13.4, NAVY, True)
    text(569, 614, "Ajuste no treino · consequências na validação", 11.1, GRAY)
    arrow(302, 626, 352, 626)
    for x, title, body, color in [
        (356, "PSE por perfil", "Perfil informado\nAssociação à política", TEAL),
        (581, "PSE preditivo", "Características da série\nEstimar classe ou\nconsequências", BLUE),
    ]:
        box(x, 433, 201, 125, edge=color, lw=1.3)
        box(x, 523, 201, 35, face="#EAF5F2" if color == TEAL else "#EAF2F9", edge="none")
        text(x + 100.5, 540.5, title, 13.5, color, True)
        text(x + 100.5, 480, body, 11)
        arrow(x + 100.5, 592, x + 100.5, 562, color)
        arrow(x + 100.5, 431, x + 100.5, 401, color)
    text(569, 490, "ou", 10.3, GRAY)
    ax.plot([456.5, 681.5], [399, 399], color=NAVY, lw=1.1)
    arrow(569, 397, 569, 367)
    box(356, 317, 426, 48, face=NAVY, edge=NAVY)
    text(569, 341, "Regra de intervenção\nCandidata ou política de referência", 12, "white", True)

    # Optional extension: its gate concerns the representation, not policy intervention.
    box(18, 73, 782, 207, face="#FFFBF5", edge=AMBER, dashed=True)
    text(36, 260, "Contexto adicional · módulo opcional", 14, AMBER, True, "left")
    text(782, 260, "Avaliação por aplicação", 10, GRAY, ha="right")
    cards = [(36, 190), (268, 214), (524, 258)]
    for x, w in cards:
        box(x, 146, w, 90, edge="#DECDB8")
    text(131, 216, "Dados adicionais", 12.6, AMBER, True)
    text(131, 181, "Cadastro · calendário\nConhecidos na decisão", 10.6)
    text(375, 216, "Escolha do gestor", 12.6, AMBER, True)
    text(375, 181, "Fixos / candidatos\nExcluir demais atributos", 10.8)
    text(653, 217, "Seleção dos candidatos", 12.3, AMBER, True)
    text(653, 175, "Reduzir custo sob\nrestrições de atendimento\nValidação interna", 10.5)
    arrow(228, 191, 266, 191, AMBER)
    arrow(484, 191, 522, 191, AMBER)
    text(36, 130, "Fixos preservados na variante ampliada.", 10.6, AMBER, ha="left")
    arrow(653, 145, 653, 117, AMBER)
    box(36, 85, 746, 31, face="#F5E9D9", edge="none")
    text(409, 100.5, "Comparar com o PSE básico; sem ganho interno, manter a versão básica.", 11.4, AMBER, True)
    # A gutter route feeds only the predictive branch; no path to the profile branch.
    ax.plot([782, 822, 822], [100.5, 100.5, 333], color=AMBER, lw=1.4, ls=(0, (4, 3)))
    # A small gap marks a crossing, not a junction with the policy-decision path.
    ax.plot([822, 822], [349, 480], color=AMBER, lw=1.4, ls=(0, (4, 3)))
    arrow(822, 480, 784, 480, AMBER, dashed=True)
    text(778, 292, "Contexto admitido", 10, AMBER, ha="right")

    # One-way decision/evaluation path: no test information feeds feature selection.
    ax.plot([783, 854, 854], [341, 341, 598], color=NAVY, lw=1.35)
    arrow(854, 598, 896, 598)
    box(898, 539, 346, 118, face="#EAF2F7", edge="none")
    text(1071, 635, "Recomendação rastreável", 13.4, NAVY, True)
    text(1071, 591, "Política e parâmetros\nReferência e alternativas\nConsequências estimadas", 11)
    arrow(1071, 535, 1071, 505)
    box(898, 424, 346, 77)
    text(1071, 474, "Gestor decide a aplicação", 12.7, NAVY, True)
    text(1071, 445, "Entradas e escolha documentadas", 11.1, GRAY)
    arrow(1071, 420, 1071, 380)
    ax.plot([899, 1243], [391, 391], color=BORDER, lw=.9, ls=(0, (3, 3)))
    text(1071, 354, "Avaliar em teste reservado", 13.2, NAVY, True)
    text(1071, 301, "Custo e atendimento\nCasos prejudicados\nIncerteza e estabilidade", 12.2)
    arrow(1071, 259, 1071, 227)
    box(898, 137, 346, 86, face="#F0F5F8", edge="none")
    text(1071, 201, "Diagnóstico retrospectivo", 12.5, NAVY, True)
    text(1071, 165, "Seleção · portfólio\nLimites físicos do atendimento", 11.7)
    text(1071, 102, "Resultados posteriores não orientam\na decisão que está sendo avaliada.", 10.8, GRAY)

    box(18, 12, 1244, 43, face="#EEF4F8", edge="none")
    text(640, 33.5, "Objetivo: reduzir custos evitáveis sem deterioração relevante do atendimento.", 14, NAVY, True)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        kwargs = {"backend": "pgf"} if ext == "pdf" else {"dpi": 240} if ext == "png" else {}
        fig.savefig(OUT / f"{NAME}.{ext}", facecolor="white", **kwargs)
    plt.close(fig)
    manifest = {
        "kind": "Conceptual vector diagram; optional extension evaluated separately per application",
        "generator": str(Path(__file__).relative_to(ROOT)),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "minimum_arrow_layout_length": min(arrow_lengths),
        "arrow_count": len(arrow_lengths),
        "parallel_routes_separation_layout_units": 32,
        "preserved": ["aipe_conceito_v2_ptbr.pdf", "aipe_conceito_v2_ptbr.svg", "aipe_conceito_v2_ptbr.png"],
        "outputs_sha256": {f"{NAME}.{ext}": hashlib.sha256((OUT / f"{NAME}.{ext}").read_bytes()).hexdigest()
                           for ext in ("pdf", "svg", "png")},
    }
    (OUT / f"{NAME}_proveniencia.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build()
