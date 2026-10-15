"""Entrada unica, utilizavel da raiz ou de experiment/."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from simulation.commands import main

if __name__ == "__main__":
    raise SystemExit(main())
