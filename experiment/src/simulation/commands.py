"""Integra pipelines Kedro, preparacao de dados e utilitarios existentes."""
import argparse
import os
import runpy
import subprocess
import sys

from simulation.paths import EXPERIMENT_ROOT

SCRIPTS = {
    "research-v2": "analysis/run_research_v2.py",
    "research-production": "analysis/run_production_v2.py",
    "research-status": "analysis/research_status.py",
    "research-visualizations": "analysis/generate_research_visualizations.py",
    "historical-visualizations": "analysis/generate_historical_visualizations.py",
    "collect-sca": "literature/collect_sca.py",
    "prepare-pb": "pb/prepare_pb_data.py",
    "pb": "pb/run_pb_experiment.py",
    "plot-pb": "pb/plot_pb_results.py",
    "check-pb": "data_preparation/check_vendas_pb.py",
    "sample-data": "data_preparation/generate_sample_data.py",
    "prepare-json": "data_preparation/transform_data_simple.py",
    "analyze-pdf": "documents/analyze_pdf.py",
    "maps": "reporting/generate_maps.py",
    "maps-full": "reporting/generate_maps_full.py",
    "maps-national": "reporting/generate_maps_nacional.py",
    "segment-plot": "reporting/regen_seg_plot.py",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Experimentos integrados AIPE")
    parser.add_argument("command", choices=["run", "prepare-sales", "device", *SCRIPTS])
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    previous_cwd, previous_argv = os.getcwd(), sys.argv
    try:
        os.chdir(EXPERIMENT_ROOT)
        if options.command == "run":
            env = {**os.environ, "KEDRO_DISABLE_TELEMETRY": "1"}
            return subprocess.call([sys.executable, "-m", "kedro", "run", *options.args], env=env)
        if options.command == "device":
            from simulation.core.device import describe, get_device
            print(describe())
            print(f"Dispositivo selecionado: {get_device()}")
            return 0
        if options.command == "prepare-sales":
            from simulation.preprocessing.sales import main as prepare_sales
            return prepare_sales(options.args)
        script = EXPERIMENT_ROOT / "scripts" / SCRIPTS[options.command]
        sys.argv = [str(script), *options.args]
        runpy.run_path(str(script), run_name="__main__")
        return 0
    finally:
        os.chdir(previous_cwd)
        sys.argv = previous_argv
