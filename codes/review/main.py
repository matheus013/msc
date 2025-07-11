import requests, xml.etree.ElementTree as ET, os, json, time
from pathlib import Path

HEADERS = {"User-Agent": "multi-echelon-review/1.0 (mailto:seu_email@dominio.com)"}
SAVE_DIR = Path("pdfs")
SAVE_DIR.mkdir(exist_ok=True)
EMAIL = "seu_email@dominio.com"          # exigido pelo Unpaywall

# ---------- BUSCA ----------
def search_arxiv(q, n=25):
    url = f"http://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results={n}"
    root = ET.fromstring(requests.get(url, headers=HEADERS).text)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    results = []
    for e in root.findall("a:entry", ns):
        pdf_url = next((l.attrib["href"] for l in e.findall("a:link", ns)
                        if l.attrib.get("type") == "application/pdf"), None)
        results.append({
            "title": e.find("a:title", ns).text.strip(),
            "pdf": pdf_url,
            "doi": None,
            "source": "arXiv"
        })
    return results

def search_semantic(q, n=25):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": q, "limit": n,
              "fields": "title,year,openAccessPdf,url,externalIds"}
    data = requests.get(url, params=params, headers=HEADERS).json()
    res = []
    for p in data.get("data", []):
        res.append({
            "title": p["title"],
            "pdf": (p.get("openAccessPdf") or {}).get("url"),
            "doi": (p.get("externalIds") or {}).get("DOI"),
            "source": "SemanticScholar"
        })
    return res

# ---------- PDF OA via DOI ----------
def unpaywall_pdf(doi):
    url = f"https://api.unpaywall.org/v2/{doi}"
    r = requests.get(url, params={"email": EMAIL}, headers=HEADERS)
    if r.status_code != 200: return None
    data = r.json()
    return (data.get("best_oa_location") or {}).get("url_for_pdf")

# ---------- DOWNLOAD ----------
def download_pdf(url, title):
    if not url: return False
    fn = SAVE_DIR / (title[:120].replace("/", "_") + ".pdf")
    try:
        with requests.get(url, stream=True, headers=HEADERS, timeout=20) as r:
            r.raise_for_status()
            with open(fn, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"✓ {fn.name}")
        return True
    except Exception as e:
        print(f"✗ {title[:60]}...  ({e})")
        return False

# ---------- PIPELINE PRINCIPAL ----------
def main():
    queries = ["multi-echelon inventory", "multi echelon supply chain"]
    seen_titles, collected = set(), []
    for q in queries:
        collected += search_arxiv(q, 30)
        time.sleep(3)                            # respeitar rate-limit
        collected += search_semantic(q, 30)

    for art in collected:
        if art["title"] in seen_titles:
            continue
        seen_titles.add(art["title"])
        if not art["pdf"] and art["doi"]:
            art["pdf"] = unpaywall_pdf(art["doi"])
        if art["pdf"]:
            download_pdf(art["pdf"], art["title"])
    # salva metadados
    with open("articles.json", "w", encoding="utf-8") as f:
        json.dump(collected, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
