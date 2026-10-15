"""Coleta retomavel de PDFs abertos da Supply Chain Analytics (ISSN 2949-8635)."""
import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import re
import threading
import unicodedata
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from PyPDF2 import PdfReader

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT.parent / "docs/references/supply_chain_analytics"
META = CORPUS / "metadata"
BLOCKED = {"www.sciencedirect.com", "linkinghub.elsevier.com", "api.elsevier.com"}
LOCKS = {}
LOCK = threading.Lock()


def doi_key(value):
    return re.sub(r"^https?://(?:dx\.)?doi.org/", "", value or "").lower()


def slug(value):
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")[:85]


def catalog():
    records = {}
    for item in json.loads((META / "crossref_raw.json").read_text(encoding="utf-8"))["message"]["items"]:
        doi = doi_key(item["DOI"])
        resource = item.get("resource", {}).get("primary", {}).get("URL", "")
        pii = re.search(r"S29498635\w{8}", resource)
        records[doi] = dict(doi=doi, title=item.get("title", [""])[0],
                            year=item.get("published", {}).get("date-parts", [[0]])[0][0],
                            volume=item.get("volume", ""), type=item.get("type", ""),
                            authors=[" ".join([a.get("given", ""), a.get("family", "")]).strip() for a in item.get("author", [])],
                            pii=pii.group() if pii else "", publisher_url=resource,
                            licenses=[l.get("URL") for l in item.get("license", [])],
                            sources=["Crossref"], candidates=[], status="pending", attempts=[])
    for file in sorted(META.glob("openalex_page*.json")):
        for work in json.loads(file.read_text(encoding="utf-8"))["results"]:
            doi = doi_key(work.get("doi"))
            if not doi:
                continue
            record = records.setdefault(doi, dict(doi=doi, title=work["display_name"],
                year=work["publication_year"], volume=work.get("biblio", {}).get("volume", ""),
                type=work.get("type", ""), authors=[a["author"]["display_name"] for a in work.get("authorships", [])],
                pii="", publisher_url=work.get("primary_location", {}).get("landing_page_url"),
                licenses=[], sources=[], candidates=[], status="pending", attempts=[]))
            record["sources"].append("OpenAlex")
            for location in work.get("locations", []):
                for kind in ("pdf_url", "landing_page_url"):
                    url = location.get(kind)
                    if not url or urlparse(url).netloc in BLOCKED or any(x in url for x in ["doi.org/", "doaj.org/", "scopus.com/", "orcid.org/", '/view/author/']):
                        continue
                    record["candidates"].append(dict(url=url, kind=kind, version=location.get("version", "unknown")))
    for record in records.values():
        seen = set()
        record["candidates"] = [c for c in sorted(record["candidates"], key=lambda c: c["kind"] != "pdf_url")
                                 if not (c["url"] in seen or seen.add(c["url"]))]
        record["sources"] = sorted(set(record["sources"]))
    for file in META.glob("doaj_page*.json"):
        for item in json.loads(file.read_text(encoding="utf-8"))["results"]:
            identifiers = item["bibjson"].get("identifier", [])
            doi = next((doi_key(i["id"]) for i in identifiers if i["type"] == "doi"), "")
            if doi in records:
                records[doi]["sources"] = sorted(set(records[doi]["sources"] + ["DOAJ"]))
    return records


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.pdfs = []

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        if tag == "meta" and attr.get("name", "").lower() == "citation_pdf_url":
            self.pdfs.append(attr.get("content", ""))
        if tag == "a":
            href = attr.get("href", "")
            if ".pdf" in href.lower() or "/bitstream" in href or "/download" in href or href.endswith("/document"):
                self.pdfs.append(href)


def pdf_details(content, record):
    if not content.lstrip().startswith(b"%PDF"):
        raise ValueError("Response is not a PDF")
    reader = PdfReader(io.BytesIO(content), strict=False)
    if not reader.pages:
        raise ValueError("PDF has no pages")
    first = reader.pages[0].extract_text() or ""
    words = lambda text: set(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKD", text.lower())))
    expected = words(record["title"])
    score = len(expected & words(first)) / max(1, len(expected))
    text = first + "\n" + "\n".join((page.extract_text() or "") for page in list(reader.pages)[1:3])
    compact = lambda value: re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", value.lower()))
    exact_title = compact(record["title"]) in compact(first)
    if (score < 0.80 and not exact_title) or (len(expected) < 6 and record["doi"] not in re.sub(r"\s+", "", text.lower())):
        raise ValueError(f"Article identity not confirmed (title overlap {score:.2f})")
    return dict(pages=len(reader.pages), sha256=hashlib.sha256(content).hexdigest(),
                bytes=len(content), title_overlap=round(score, 3), exact_title=exact_title)


def request(url):
    host = urlparse(url).netloc
    if host in BLOCKED:
        raise ValueError("Publisher automation requires access permissions")
    with LOCK:
        host_lock = LOCKS.setdefault(host, threading.Lock())
    with host_lock:
        response = requests.get(url, timeout=(15, 40), headers={"User-Agent": "AIPE-literature-collection/1.0 (academic research)"})
    response.raise_for_status()
    return response


def download(record, extra):
    number = record["doi"].rsplit(".", 1)[-1]
    target = CORPUS / "pdfs" / str(record["year"]) / f"{number}_{slug(record['title'])}.pdf"
    if target.exists():
        record.update(pdf_details(target.read_bytes(), record), status="downloaded", file=str(target.relative_to(CORPUS)))
        return record
    attempted = {a["url"] for a in record.get("attempts", [])}
    candidates = [c for c in extra.get(record["doi"], []) + record["candidates"] if c["url"] not in attempted]
    for candidate in candidates:
        try:
            response = request(candidate["url"])
            urls = []
            if not response.content.lstrip().startswith(b"%PDF"):
                parser = Links()
                parser.feed(response.text)
                urls = list(dict.fromkeys(urljoin(response.url, u) for u in parser.pdfs if u))[:10]
                record["attempts"].append(dict(url=candidate["url"], result="landing page", links=len(urls)))
            else:
                urls = [response.url]
            for url in urls:
                try:
                    pdf_response = response if url == response.url and response.content.lstrip().startswith(b"%PDF") else request(url)
                    details = pdf_details(pdf_response.content, record)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(pdf_response.content)
                    record.update(details, status="downloaded", file=str(target.relative_to(CORPUS)),
                                  downloaded_from=pdf_response.url, version=candidate.get("version", "unknown"))
                    print(f"DOWNLOADED {record['doi']} ({details['pages']} pages)", flush=True)
                    return record
                except Exception as error:
                    record["attempts"].append(dict(url=url, result=str(error)[:240]))
        except Exception as error:
            record["attempts"].append(dict(url=candidate["url"], result=str(error)[:240]))
    record["status"] = "unavailable" if record["candidates"] or extra.get(record["doi"]) else "publisher_access_required"
    return record


def save(records):
    items = sorted(records.values(), key=lambda r: (str(r["year"]), r["doi"]))
    (CORPUS / "catalog.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["doi", "title", "year", "volume", "type", "status", "file", "pages", "version", "downloaded_from", "publisher_url"]
    with (CORPUS / "catalog.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(items)
    print(json.dumps(Counter(r["status"] for r in items)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retry-failed", action="store_true", help="Repetir URLs que falharam em coleta anterior")
    options = parser.parse_args()
    records = catalog()
    if (CORPUS / "catalog.json").exists():
        for previous in json.loads((CORPUS / "catalog.json").read_text(encoding="utf-8")):
            if previous["doi"] in records:
                sources = sorted(set(records[previous["doi"]]["sources"] + previous.get("sources", [])))
                records[previous["doi"]].update(previous)
                records[previous["doi"]]["sources"] = sources
    if options.retry_failed:
        for record in records.values():
            if record["status"] != "downloaded":
                record["attempts"] = []
    extra = json.loads((META / "extra_sources.json").read_text(encoding="utf-8")) if (META / "extra_sources.json").exists() else {}
    save(records)
    with concurrent.futures.ThreadPoolExecutor(max_workers=options.workers) as executor:
        futures = {executor.submit(download, record, extra): doi for doi, record in records.items()}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            records[futures[future]] = future.result()
            if index % 25 == 0:
                save(records)
    save(records)
    print("Complete", flush=True)


if __name__ == "__main__":
    main()
