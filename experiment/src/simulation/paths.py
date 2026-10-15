"""Caminhos comuns: fontes brutas na raiz; derivados no projeto Python."""
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parent
RAW_DATA_ROOT = REPOSITORY_ROOT / "data"
ARTIFACTS_ROOT = EXPERIMENT_ROOT / "data"
DOCS_ROOT = REPOSITORY_ROOT / "docs"
