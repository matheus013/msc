"""Rebuild and synchronize the article after the scoped Figure 1 revision."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs/supply_chain_analytics_article"

for command in (["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "manuscript_ptbr.tex"],
                ["bibtex", "manuscript_ptbr"],
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "manuscript_ptbr.tex"],
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "manuscript_ptbr.tex"]):
    result = subprocess.run(command, cwd=DOC, capture_output=True, text=True, errors="replace")
    if result.returncode:
        raise RuntimeError((result.stdout + result.stderr)[-5000:])
log = (DOC / "manuscript_ptbr.log").read_text(errors="replace")
issues = [line for line in log.splitlines() if any(w in line for w in ("undefined", "Overfull", "LaTeX Error"))]
if issues:
    raise RuntimeError("\n".join(issues))

main = (DOC / "manuscript_ptbr.tex").read_text(encoding="utf-8")
sources = ["manuscript_ptbr.tex"] + [n + ".tex" for n in re.findall(r"\\input\{([^}]+)\}", main)]
all_text = "\n".join((DOC / n).read_text(encoding="utf-8") for n in sources)
figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", all_text)
files = list(dict.fromkeys(sources + figures + ["references_ptbr.bib", "manuscript_ptbr.bbl",
                                               "elsarticle.cls", "elsarticle-num.bst", "highlights_ptbr.txt"]))
with zipfile.ZipFile(DOC / "article_sources_ptbr_draft.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
    for name in files:
        z.write(DOC / name, name)
manifest = json.loads((DOC / "build_manifest_ptbr.json").read_text(encoding="utf-8"))
manifest["sha256"] = {name: hashlib.sha256((DOC / name).read_bytes()).hexdigest()
                      for name in files + ["manuscript_ptbr.pdf", "article_sources_ptbr_draft.zip"]}
(DOC / "build_manifest_ptbr.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"compiled": True, "diagnostics": issues, "source_files": len(files), "figures": figures}, indent=2))
