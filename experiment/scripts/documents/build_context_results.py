"""Audit completed optional-context studies and publish aggregate evidence.

No training, simulation, manuscript editing, or modification of frozen inputs.
Bootstrap intervals are conditional on frozen fits and resample whole stores,
retaining all products and origins of each store together.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import os
import shutil
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/operations/context_rounds_2026-10-17"
ARTICLE = ROOT / "docs/supply_chain_analytics_article"
THESIS = ROOT / "docs/master_proposal"
CASE = ["origin", "warehouse", "store_id", "item_id"]
BASE = ROOT / "experiment/data/context_extension"
RUNS = {"BA": (BASE / "BA/prepared_2026-10-17_cadastro_v2", BASE / "BA/study_2026-10-17_cadastro_v1"),
        "M5": (BASE / "M5/prepared_2026-10-17_context_v1", BASE / "M5/study_2026-10-17_context_v1")}
SOURCES = {}
STRATEGIES = ["PSE_base", "PSE_context_fixed", "PSE_context_all", "PSE_context_selected", "PSE_context_guarded"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    SOURCES[str(Path(path).relative_to(ROOT))] = sha(path)


def js(path):
    record(path)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read(path):
    record(path)
    return pd.read_csv(path, dtype={c: str for c in CASE[1:]})


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def paired(candidate, baseline, draws=5000):
    p = candidate.merge(baseline[CASE + ["TIC", "NS", "policy"]], on=CASE,
                        suffixes=("", "_base"), validate="one_to_one")
    assert len(p) == len(candidate) == len(baseline)
    assert p.NS.isna().eq(p.NS_base.isna()).all()
    p["dc"] = p.TIC - p.TIC_base
    p["ds"] = p.NS - p.NS_base
    p["defined"] = p.NS.notna().astype(int)
    g = p.groupby(["warehouse", "store_id"]).agg(cost=("TIC", "sum"), base=("TIC_base", "sum"),
        service_delta=("ds", "sum"), defined=("defined", "sum"))
    rng = np.random.default_rng(20261017)
    indices = rng.integers(0, len(g), (draws, len(g)))
    cost = g.cost.to_numpy()[indices].sum(axis=1)
    base = g.base.to_numpy()[indices].sum(axis=1)
    defined = g.defined.to_numpy()[indices].sum(axis=1)
    delta = g.service_delta.to_numpy()[indices].sum(axis=1)
    cost_samples = 100 * (cost[base > 0] / base[base > 0] - 1)
    service_samples = 100 * delta[defined > 0] / defined[defined > 0]
    ci = lambda x: [float(np.quantile(x, .025)), float(np.quantile(x, .975))] if len(x) else [None, None]
    savings = (-p.dc).clip(lower=0)
    return dict(cases=len(p), defined_service=int(p.defined.sum()), clusters=len(g),
        mean_cost=float(p.TIC.mean()), baseline_mean_cost=float(p.TIC_base.mean()),
        mean_service=float(p.NS.mean()), baseline_mean_service=float(p.NS_base.mean()),
        cost_change_percent=float(100 * (p.TIC.sum() / p.TIC_base.sum() - 1)),
        service_change_pp=float(100 * p.ds.mean()),
        cost_change_percent_ci95=ci(cost_samples), service_change_pp_ci95=ci(service_samples),
        paired_cost_difference_mean=float(p.dc.mean()),
        interventions=int(p.policy.ne(p.policy_base).sum()),
        lower_service_cases=int(p.ds.lt(-1e-9).sum()), higher_cost_cases=int(p.dc.gt(1e-6).sum()),
        new_service_failures=int((p.NS_base.ge(.7 - 1e-9) & p.NS.lt(.7 - 1e-9)).sum()),
        recovered_service_failures=int((p.NS_base.lt(.7 - 1e-9) & p.NS.ge(.7 - 1e-9)).sum()),
        largest_positive_saving_share=float(savings.max() / savings.sum()) if savings.sum() > 0 else None,
        bootstrap=dict(draws=draws, seed=20261017, unit="warehouse,store_id", all_origins_retained=True,
            conditional_on_frozen_models=True, refits=False,
            limitation="Descriptive conditional percentile intervals; overlapping fitted training sets and model-selection uncertainty are not reproduced. M5 has only ten store clusters."))


def audit(dataset):
    prepared, run = RUNS[dataset]
    status = js(run / "status.json")
    protocol = js(run / "protocol_before_predictions.json")
    assert protocol == js(prepared / "context_protocol.json")
    manifest = js(prepared / "manifest.json")
    assert status["state"] == "completed" and status["completed_folds"] == status["total_folds"]
    assert status["controller_fits"] == 0
    for name, digest in manifest["file_sha256"].items():
        assert sha(prepared / name) == digest, name
    for path, digest in manifest["source_sha256"].items():
        assert sha(path) == digest, path
    for field in ("source_context_sha256", "context_code_sha256"):
        for path, digest in protocol[field].items():
            assert sha(path) == digest, path
    assert sha(prepared / "manifest.json") == protocol["prepared_manifest_sha256"]
    assert sha(prepared / "context_features.csv") == protocol["context_features_sha256"]
    predictions = read(run / "predictions_frozen_before_test.csv")
    assert sha(run / "predictions_frozen_before_test.csv") == status["predictions_sha256"]
    features = read(prepared / "context_features.csv")
    wanted_cases = set(features[CASE].itertuples(index=False, name=None))
    assert len(wanted_cases) == len(features)
    all_predictions, audits = [], []
    frequency, gates = [], []
    active = [s["name"] for s in protocol["feature_contracts"] if s["mode"] != "excluded"]
    expected_tasks = {(model, int(o), str(w), str(s)) for model in protocol["models"]
        for o, w, s in features[["origin", "warehouse", "store_id"]].drop_duplicates().itertuples(index=False, name=None)}
    seen_tasks = set()
    for directory in sorted((run / "folds").iterdir()):
        done = js(directory / "completed.json")
        for name, digest in done["files"].items():
            assert sha(directory / name) == digest, str(directory / name)
        task = tuple(done["task"])
        assert task in expected_tasks and task not in seen_tasks
        seen_tasks.add(task)
        a = js(directory / "audit.json")
        assert task == (a["model"], a["origin"], a["warehouse"], a["held_store"])
        fold = read(directory / "predictions.csv")
        assert fold.model.eq(task[0]).all() and fold.origin.eq(task[1]).all()
        assert fold.warehouse.eq(task[2]).all() and fold.store_id.eq(task[3]).all()
        held = features[features.origin.eq(task[1]) & features.warehouse.eq(task[2]) & features.store_id.eq(task[3])]
        held_cases = set(held[CASE].itertuples(index=False, name=None))
        assert set(fold.strategy) == set(STRATEGIES)
        for strategy, rows in fold.groupby("strategy"):
            assert set(rows[CASE].itertuples(index=False, name=None)) == held_cases
            assert len(rows) == len(held_cases)
        base = fold[fold.strategy.eq("PSE_base")].sort_values(CASE)
        fixed = fold[fold.strategy.eq("PSE_context_fixed")].sort_values(CASE)
        assert base.policy.to_list() == fixed.policy.to_list()
        assert a["selected"]["fixed_features"] == []
        assert set(a["all"]["fixed_features"]) == set(active)
        reference = "PSE_context_selected" if a["selected"]["context_recommended"] else "PSE_base"
        guarded = fold[fold.strategy.eq("PSE_context_guarded")].sort_values(CASE)
        expected = fold[fold.strategy.eq(reference)].sort_values(CASE)
        assert guarded.policy.to_list() == expected.policy.to_list()
        gate = dict(dataset=dataset, model=a["model"], origin=a["origin"], warehouse=a["warehouse"],
            store_id=a["held_store"], cases=len(held), context_recommended=a["selected"]["context_recommended"],
            selected_count=len(a["selected"]["selected_features"]), evaluations=a["selected"]["evaluations"],
            stop_reason=a["selected"]["stop_reason"])
        gates.append(gate)
        for name in active:
            q = a["selected"]["quality"][name]
            frequency.append(gate | dict(feature=name, selected=name in a["selected"]["selected_features"],
                admissible_to_search=q["reason"] is None, quality_reason=q["reason"]))
        all_predictions.append(fold)
        audits.append(a)
    assert seen_tasks == expected_tasks and len(seen_tasks) == status["total_folds"]
    reconstructed_predictions = pd.concat(all_predictions, ignore_index=True).sort_values(["model", "strategy", *CASE]).reset_index(drop=True)
    pd.testing.assert_frame_equal(predictions, reconstructed_predictions, check_dtype=False)
    test = read(prepared / "test.csv")
    assert test.block.eq("test").all()
    assert not test.duplicated(CASE + ["policy", "seed"]).any()
    means = test.groupby(CASE + ["policy"], as_index=False)[["TIC", "NS"]].mean()
    metrics = predictions.merge(means, on=CASE + ["policy"], validate="many_to_one")
    stored = read(run / "case_metrics.csv").sort_values(["model", "strategy", *CASE]).reset_index(drop=True)
    metrics = metrics.sort_values(["model", "strategy", *CASE]).reset_index(drop=True)
    assert len(metrics) == len(predictions)
    for col in ("TIC", "NS"):
        np.testing.assert_allclose(metrics[col], stored[col], rtol=1e-12, atol=1e-10, equal_nan=True)
    assert metrics[CASE + ["model", "strategy", "policy"]].equals(stored[CASE + ["model", "strategy", "policy"]])
    summary = read(run / "summary.csv").set_index(["model", "strategy"])
    original_comparison = js(run / "comparison.json")
    results, origins = {}, {}
    for model, group in metrics.groupby("model"):
        selected_policies = group[group.strategy.eq("PSE_context_selected")].sort_values(CASE).policy.to_list()
        guarded_policies = group[group.strategy.eq("PSE_context_guarded")].sort_values(CASE).policy.to_list()
        assert selected_policies == guarded_policies, "Current narrative requires selected/guarded equality"
        results[model] = {}
        baseline = group[group.strategy.eq("PSE_base")]
        for strategy, rows in group.groupby("strategy"):
            assert set(rows[CASE].itertuples(index=False, name=None)) == wanted_cases
            np.testing.assert_allclose([rows.TIC.sum(), rows.TIC.mean(), rows.NS.mean()],
                summary.loc[(model, strategy), ["total_cost", "mean_cost", "mean_service"]].to_numpy(float), rtol=1e-12)
            result = paired(rows, baseline)
            np.testing.assert_allclose(result["cost_change_percent"],
                -100 * original_comparison[model][strategy]["relative_cost_improvement"], atol=1e-10)
            np.testing.assert_allclose(result["service_change_pp"],
                100 * original_comparison[model][strategy]["service_difference"], atol=1e-10)
            results[model][strategy] = result
        origins[model] = {int(o): {strategy: paired(rows, g[g.strategy.eq("PSE_base")])
            for strategy, rows in g.groupby("strategy")} for o, g in group.groupby("origin")}
    frequency = pd.DataFrame(frequency)
    gates = pd.DataFrame(gates)
    frequencies = frequency.groupby(["dataset", "model", "origin", "feature"], as_index=False).agg(
        folds=("selected", "size"), selected_folds=("selected", "sum"), eligible_folds=("admissible_to_search", "sum"))
    frequencies["selected_fraction"] = frequencies.selected_folds / frequencies.folds
    gate_summary = gates.groupby(["dataset", "model", "origin"], as_index=False).agg(
        folds=("context_recommended", "size"), recommended_folds=("context_recommended", "sum"),
        mean_selected_features=("selected_count", "mean"), cases=("cases", "sum"))
    gate_summary["recommended_fraction"] = gate_summary.recommended_folds / gate_summary.folds
    frequencies.to_csv(OUT / f"{dataset}_selection_frequency.csv", index=False)
    gate_summary.to_csv(OUT / f"{dataset}_gate_rates.csv", index=False)
    gates.to_csv(OUT / f"{dataset}_fold_gates.csv", index=False)
    return dict(status=status, protocol=protocol, contrasts=results, by_origin=origins,
        selection_frequency=frequencies.to_dict("records"), gate_rates=gate_summary.to_dict("records"),
        verified=dict(all_frozen_fold_hashes=True, all_source_and_context_hashes=True,
            predictions_reconstructed_from_folds=True, test_metrics_independently_reconstructed=True,
            summary_and_comparison_match=True, fixed_equals_base=True,
            guarded_respects_internal_gate=True, controller_fits=0))


def number(x, digits=3, signed=False):
    fmt = f"{{:{'+' if signed else ''}.{digits}f}}"
    return fmt.format(x).replace(".", ",")


def interval_text(result, key):
    low, high = result[key]
    return f"[{number(low)}; {number(high)}]"


def tex(evidence):
    lines = [r"% Gerado por build_context_results.py; dados de desenvolvimento, não confirmação independente.",
        r"\begin{table}[htbp]", r"\centering\small",
        r"\caption{Incremento do contexto adicional em relação ao PSE básico da mesma instância. Variação negativa de custo indica economia; atendimento em pontos percentuais. Nesta rodada, selecionada e protegida coincidem em todas as recomendações.}",
        r"\label{tab:contexto_adicional_20261017}",
        r"\begin{tabular}{llrr}", r"\toprule", r"Base & Instância & $\Delta$ custo (\%) & $\Delta$ atendimento (p.p.) \\", r"\midrule"]
    for ds in ("BA", "M5"):
        for model, label in (("class", "Classe"), ("consequences", "Consequências")):
            r = evidence[ds]["contrasts"][model]["PSE_context_guarded"]
            lines.append(f"{'Bahia' if ds == 'BA' else 'M5'} & {label} & {number(r['cost_change_percent'], signed=True)} & {number(r['service_change_pp'], signed=True)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    ba = evidence["BA"]["contrasts"]
    m5 = evidence["M5"]["contrasts"]
    lines += [
        r"O módulo adicional foi avaliado reutilizando as matrizes de políticas, sem novos ajustes dos controladores. Foram comparados o PSE básico, todos os atributos admissíveis, a seleção interna de atributos e sua ativação protegida, separadamente nas instâncias por classe e por consequências. Os contrastes usam florestas de 32 árvores com os mesmos parâmetros entre variantes; portanto, suas referências básicas não se confundem com o classificador XGBoost nem com as variantes canônicas de 64 árvores. Não houve atributos adicionais impostos pelo gestor: a variante de atributos fixos coincide com a básica.",
        r"Na Bahia, idade, segmento comercial e filial foram incorporados sob a premissa declarada pelo autor de que o cadastro foi extraído diretamente do banco no mesmo período. Essa confirmação não substitui um histórico de alterações por campo. No M5, foram usados departamento do produto, preço histórico, indicação estadual do Supplemental Nutrition Assistance Program (SNAP), programa norte-americano de assistência alimentar, e eventos do último bloco completo de 21 dias do prefixo. Nenhuma informação futura foi usada; o preço funciona como contexto e não altera os custos unitários do ambiente.",
        f"A redução de custo por classe na Bahia ({number(-ba['class']['PSE_context_guarded']['cost_change_percent'])}\\%) veio acompanhada de queda de {number(-ba['class']['PSE_context_guarded']['service_change_pp'])} ponto percentual no atendimento. Assim, o ganho de custo não satisfaz isoladamente a finalidade gerencial do AIPE. A instância por consequências manteve exatamente as decisões básicas. No M5, os efeitos selecionados foram pequenos: {number(-m5['class']['PSE_context_guarded']['cost_change_percent'])}\\% de economia e {number(m5['class']['PSE_context_guarded']['service_change_pp'])} ponto percentual adicional por classe; por consequências, {number(-m5['consequences']['PSE_context_guarded']['cost_change_percent'])}\\% de economia e variação de {number(m5['consequences']['PSE_context_guarded']['service_change_pp'])} ponto percentual.",
        f"O bootstrap pareado de 5.000 reamostragens por loja/revendedor, preservando produtos e origens no mesmo grupo, produziu para a variação de custo os intervalos condicionais de 95\\% {interval_text(ba['class']['PSE_context_guarded'], 'cost_change_percent_ci95')} na Bahia por classe e {interval_text(m5['class']['PSE_context_guarded'], 'cost_change_percent_ci95')} no M5 por classe. Para o atendimento, os respectivos intervalos foram {interval_text(ba['class']['PSE_context_guarded'], 'service_change_pp_ci95')} e {interval_text(m5['class']['PSE_context_guarded'], 'service_change_pp_ci95')} pontos percentuais. Esses intervalos descrevem os modelos congelados, sem refazer a seleção ou o ajuste; no M5 há somente dez lojas.",
        f"Usar todos os atributos também não assegurou vantagem da seleção. No M5 por classe, todos os atributos reduziram o custo em {number(-m5['class']['PSE_context_all']['cost_change_percent'])}\\%, ante {number(-m5['class']['PSE_context_guarded']['cost_change_percent'])}\\% com seleção, mas com menor incremento de atendimento ({number(m5['class']['PSE_context_all']['service_change_pp'])} ante {number(m5['class']['PSE_context_guarded']['service_change_pp'])} ponto percentual). Na Bahia por classe, todos os atributos elevaram o custo em {number(ba['class']['PSE_context_all']['cost_change_percent'])}\\% e reduziram o atendimento. Os contrastes sustentam a necessidade de avaliar informação adicional pelo efeito operacional, sem estabelecer superioridade geral da seleção de atributos ou garantia de atendimento da regra de ativação.",
        r"A rodada constitui desenvolvimento: os resultados canônicos já eram conhecidos, as partições de treino dos seletores se sobrepõem e não se realizou confirmação independente. A ampliação permanece opcional; eventuais economias devem ser examinadas com as deteriorações por caso e a relevância definida pelo gestor.",
        r"A Figura~\ref{fig:contexto_incremental_20261017} apresenta os contrastes de custo e atendimento e sua incerteza condicional.",
        r"\begin{figure}[htbp]", r"\centering",
        r"\includegraphics[width=\linewidth]{figures/aipe_contexto_incremental_20261017.pdf}",
        r"\caption{Efeito incremental do contexto sobre custo (esquerda) e atendimento (direita), perante o PSE básico da mesma instância. Pontos: variações observadas; barras: intervalos condicionais de 95\% de 5.000 reamostragens pareadas por loja/revendedor, mantendo produtos e origens agrupados. Custo negativo indica economia. Seleção e ativação protegida coincidiram nesta rodada. O bootstrap não refaz os ajustes; o M5 possui dez lojas.}",
        r"\label{fig:contexto_incremental_20261017}", r"\end{figure}"]
    text = "\n\n".join(lines) + "\n"
    for dest in (ARTICLE, THESIS):
        (dest / "context_results_20261017.tex").write_text(text, encoding="utf-8")


def figure(evidence):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                        "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    order = [(ds, model) for ds in ("BA", "M5") for model in ("class", "consequences")]
    labels = [f"{'Bahia' if d == 'BA' else 'M5'} · {'classe' if m == 'class' else 'consequências'}" for d, m in order]
    colors = {"PSE_context_all": "#9b6a2d", "PSE_context_guarded": "#177b8b"}
    for i, (key, ci_key, title, xlabel) in enumerate([
            ("cost_change_percent", "cost_change_percent_ci95", "Custo", "Variação em relação à básica (%)"),
            ("service_change_pp", "service_change_pp_ci95", "Atendimento", "Variação em relação à básica (p.p.)")]):
        ax = axes[i]
        for strategy, offset, name in (("PSE_context_all", -.12, "Todos os atributos"),
                                      ("PSE_context_guarded", .12, "Selecionada / protegida")):
            for row, (ds, model) in enumerate(order):
                r = evidence[ds]["contrasts"][model][strategy]
                low, high = r[ci_key]
                point = r[key]
                ax.errorbar(point, row + offset, xerr=[[max(0, point-low)], [max(0, high-point)]],
                    fmt="o", markersize=4, capsize=3, color=colors[strategy],
                    label=name if row == 0 else None)
        ax.axvline(0, color="#657489", linewidth=.9, linestyle="--")
        ax.set_yticks(range(4), labels if i == 0 else [])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", alpha=.15)
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.56, -.01))
    fig.subplots_adjust(left=.23, right=.98, bottom=.23, top=.89, wspace=.18)
    for ext in ("pdf", "png"):
        path = ARTICLE / "figures" / f"aipe_contexto_incremental_20261017.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        shutil.copy2(path, THESIS / "figures" / path.name)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = {ds: audit(ds) for ds in RUNS}
    record(Path(__file__))
    payload = dict(studies=evidence, source_sha256=SOURCES,
        interpretation="Matched contrasts within each new context learner; development only, no guarantees or independent confirmation.")
    (OUT / "evidence.json").write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    rows = [dict(dataset=ds, model=model, strategy=strategy, **{k: v for k, v in result.items()
            if not isinstance(v, (dict, list))}) for ds, study in evidence.items()
            for model, variants in study["contrasts"].items() for strategy, result in variants.items()]
    pd.DataFrame(rows).to_csv(OUT / "paired_summary.csv", index=False)
    tex(evidence)
    figure(evidence)
    (OUT / "FECHAMENTO.md").write_text(
        "# Fechamento das rodadas adicionais\n\n"
        "Bahia: 460/460 dobras. M5: 60/60 dobras. Zero novos ajustes de controladores.\n\n"
        "Auditoria independente confirmou hashes de entradas e modelos, recomendações de todas as dobras, composição dos casos, regra interna de ativação e reconstrução das consequências por médias das sementes do teste. Resumos e comparações coincidem com os publicados pelos runners.\n\n"
        "`evidence.json` contém contrastes pareados por instância, por origem, intervalos condicionais por loja (5.000 reamostragens), frequência de atributos e ativação interna. Há apenas dez lojas no M5; os intervalos não refazem ajuste/seleção nem representam confirmação independente.\n\n"
        "A economia por classe da Bahia acompanha deterioração do atendimento. Consequências na Bahia não alteram a referência. No M5, as variações são pequenas e o uso de todos os atributos apresenta maior economia que a seleção, sem assegurar melhor compromisso com o atendimento. Não há evidência para declarar superioridade geral do módulo ou da seleção de atributos.\n\n"
        "Gerados os inputs compartilhados `context_results_20261017.tex` e a figura `aipe_contexto_incremental_20261017.pdf/png` no artigo e na dissertação, sem editar seus arquivos principais.\n",
        encoding="utf-8")
    print("Context studies audited and aggregate artifacts generated; no fits or manuscript edits.")


if __name__ == "__main__":
    main()
