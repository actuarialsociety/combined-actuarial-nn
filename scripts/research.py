#!/usr/bin/env python3
"""Local research library: every paper this repo cites, downloaded and full-text indexed.

    python scripts/research.py fetch            # find citations, download open-access PDFs
    python scripts/research.py index            # extract text, build the search index
    python scripts/research.py search "seed variance" [-n 10]
    python scripts/research.py status           # what is downloaded, what is missing

Citations are collected from the lessons, README, `Open Actuarial Data.md` and
`conf/base/parameters_benchmarks.yml` (every DOI), plus `EXTRA` below (book chapters and
monographs without a DOI). Open-access copies are found with OpenAlex (no account or email
needed). Paywalled papers are listed in `status`: put their PDF in
`data/12_research/manual/` named after the DOI with `/` replaced by `_`
(`10.1080_03461238.2022.2081816.pdf`), then run `index`.

Everything lives in `data/12_research/` (git-ignored like the rest of `data/`): the papers
are copyrighted, so they are never committed. Each page of each paper is one row of an
SQLite FTS5 index, so a search answers with paper and page, and `text/<doi>.txt` holds the
full text (pages separated by form feeds) for reading or grep.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "12_research"
PDFS, MANUAL, TEXT = OUT / "pdfs", OUT / "manual", OUT / "text"
MANIFEST, INDEX = OUT / "manifest.csv", OUT / "index.sqlite"
SOURCES = [
    *sorted((ROOT / "lessons").glob("*.md")),
    ROOT / "README.md",
    ROOT / "Open Actuarial Data.md",
    ROOT / "conf" / "base" / "parameters_benchmarks.yml",
]
DOI = re.compile(r"(?:doi\.org/|doi:\s*)(10\.\d{4,9}/[^\s)\"'>,\]]+)", re.I)
# cited without a DOI: (key, title, pdf url)
EXTRA = [
    (
        "aitools-ch8",
        "Richman, Scognamiglio & Wüthrich (2026), AI Tools for Actuaries, Ch. 8: mortality "
        "forecasting with CNNs, RNNs and Transformers",
        "https://aitools4actuaries.com/Ch8E/mortality_forecasting.pdf",
    ),
    (
        "arxiv-1604.06737",
        "Guo & Berkhahn (2016), Entity embeddings of categorical variables",
        "https://arxiv.org/pdf/1604.06737",
    ),
    (
        "meyers-2015-cas-monograph-1",
        "Meyers (2015), Stochastic Loss Reserving Using Bayesian MCMC Models, CAS Monograph 1",
        "https://www.casact.org/sites/default/files/2021-02/01-meyers.pdf",
    ),
]
HEADERS = {"User-Agent": "deep-actuarial-science-research/1.0 (+open-access fetcher)"}
FIELDS = ["key", "doi", "title", "year", "oa_status", "pdf_url", "path", "pages", "status"]


def key_of(doi: str) -> str:
    return doi.replace("/", "_")


def citations() -> list[str]:
    found = set()
    for f in SOURCES:
        if f.exists():
            for m in DOI.finditer(f.read_text(encoding="utf-8")):
                found.add(m.group(1).rstrip(".;"))
    return sorted(found, key=str.lower)


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def openalex(doi: str) -> dict:
    url = "https://api.openalex.org/works/doi:" + urllib.parse.quote(doi)
    url += "?select=title,publication_year,open_access,best_oa_location,locations"
    try:
        return json.loads(_get(url, 30))
    except Exception as e:  # noqa: BLE001 - one bad DOI must not stop the others
        return {"error": str(e)}


def pdf_urls(work: dict) -> list[str]:
    urls = []
    best = work.get("best_oa_location") or {}
    for loc in [best, *(work.get("locations") or [])]:
        u = (loc or {}).get("pdf_url")
        if u and u not in urls:
            urls.append(u)
    oa = (work.get("open_access") or {}).get("oa_url")
    if oa and oa not in urls:
        urls.append(oa)
    return urls


CITATION_PDF = re.compile(
    r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)', re.I
)


def landing_pdfs(work: dict) -> list[str]:
    """PDF links that repository landing pages declare (`citation_pdf_url`, the Google Scholar
    tag that ETH Research Collection, EconStor, DOAJ hosts and most journals publish)."""
    urls = []
    for loc in work.get("locations") or []:
        page = (loc or {}).get("landing_page_url") or ""
        if not page or "doi.org/10.2139" in page:  # SSRN pages refuse automated clients
            continue
        try:
            html = _get(page, 30).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            continue
        urls += [urllib.parse.urljoin(page, u) for u in CITATION_PDF.findall(html)]
    return urls


def arxiv_pdf(title: str) -> list[str]:
    """The arXiv preprint with this exact title, if there is one (arXiv API)."""
    if not title:
        return []
    words = re.sub(r"[^\w\s]", " ", title).split()
    q = urllib.parse.quote(" AND ".join(f"ti:{w}" for w in words[:12]))
    try:
        xml = _get(f"https://export.arxiv.org/api/query?search_query={q}&max_results=5", 30)
    except Exception:  # noqa: BLE001
        return []
    want = " ".join(w.lower() for w in words)
    out = []
    for entry in xml.decode("utf-8", "replace").split("<entry>")[1:]:
        t = re.search(r"<title>(.*?)</title>", entry, re.S)
        link = re.search(r'href="(https?://arxiv\.org/pdf/[^"]+)"', entry)
        got = " ".join(re.sub(r"[^\w\s]", " ", t.group(1) if t else "").lower().split())
        if link and got == want:
            out.append(link.group(1))
    return out


def download(urls: list[str], dest: Path) -> str | None:
    """First URL that returns a real PDF; the file is written only if it starts with %PDF."""
    for u in urls:
        try:
            data = _get(u)
        except Exception:  # noqa: BLE001
            continue
        if data[:5] == b"%PDF-":
            dest.write_bytes(data)
            return u
    return None


def load_manifest() -> dict[str, dict]:
    if not MANIFEST.exists():
        return {}
    with MANIFEST.open() as f:
        return {r["key"]: r for r in csv.DictReader(f)}


def save_manifest(rows: dict[str, dict]) -> None:
    with MANIFEST.open("w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for k in sorted(rows):
            w.writerow({c: rows[k].get(c, "") for c in FIELDS})


def fetch(force: bool = False) -> None:
    for d in (PDFS, MANUAL, TEXT):
        d.mkdir(parents=True, exist_ok=True)
    rows = load_manifest()
    todo = [(key_of(d), d, None, None) for d in citations()]
    todo += [(k, "", title, url) for k, title, url in EXTRA]
    for key, doi, title, url in todo:
        dest = PDFS / f"{key}.pdf"
        row = rows.get(key, {"key": key, "doi": doi})
        if dest.exists() and not force:
            print(f"have     {key}")
            continue
        if doi:
            work = openalex(doi)
            row.update(
                title=work.get("title") or row.get("title", ""),
                year=work.get("publication_year") or "",
                oa_status=(work.get("open_access") or {}).get("oa_status", ""),
            )
            urls = pdf_urls(work)
            if doi.lower().startswith("10.2139/ssrn."):  # SSRN serves the PDF to a direct link
                sid = doi.split(".")[-1]
                urls.append(f"https://papers.ssrn.com/sol3/Delivery.cfm?abstractid={sid}&type=2")
            time.sleep(0.2)  # be polite to the API
        else:
            row.update(title=title, oa_status="direct")
            urls = [url]
        got = download(urls, dest)
        if not got and doi:  # repository copies, then an arXiv preprint
            work_urls = landing_pdfs(work) + arxiv_pdf(row.get("title", ""))
            got = download([u for u in work_urls if u not in urls], dest)
            urls += work_urls
        row.update(pdf_url=got or (urls[0] if urls else ""), path=str(dest.relative_to(ROOT)))
        row["status"] = "downloaded" if got else ("paywalled" if not urls else "no pdf")
        if not got:
            row["path"] = ""
        rows[key] = row
        print(f"{row['status']:<10} {key}  {row.get('title', '')[:70]}")
    save_manifest(rows)


def pages(pdf: Path) -> list[str]:
    """Text of each page: pdftotext (poppler) if installed, else pypdf."""
    if shutil.which("pdftotext"):
        r = subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=False
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.split("\f")
    from pypdf import PdfReader

    return [p.extract_text() or "" for p in PdfReader(str(pdf)).pages]


def index() -> None:
    rows = load_manifest()
    for pdf in sorted(MANUAL.glob("*.pdf")):  # PDFs you added by hand
        key = pdf.stem
        row = rows.setdefault(key, {"key": key, "doi": key.replace("_", "/", 1)})
        row.update(path=str(pdf.relative_to(ROOT)), status="manual")
    if INDEX.exists():
        INDEX.unlink()
    db = sqlite3.connect(INDEX)
    db.execute("CREATE VIRTUAL TABLE pages USING fts5(key UNINDEXED, page UNINDEXED, title, text)")
    n_pages = 0
    for key, row in sorted(rows.items()):
        if not row.get("path"):
            continue
        pdf = ROOT / row["path"]
        try:
            text = pages(pdf)
        except Exception as e:  # noqa: BLE001 - a damaged PDF must not stop the index
            print(f"skip {key}: {e}")
            continue
        (TEXT / f"{key}.txt").write_text("\f".join(text), encoding="utf-8")
        db.executemany(
            "INSERT INTO pages VALUES (?, ?, ?, ?)",
            [(key, i + 1, row.get("title", ""), t) for i, t in enumerate(text) if t.strip()],
        )
        row["pages"] = len(text)
        n_pages += len(text)
    db.commit()
    db.close()
    save_manifest(rows)
    print(f"indexed {sum(1 for r in rows.values() if r.get('pages'))} papers, {n_pages} pages")


def search(query: str, n: int = 10) -> None:
    if not INDEX.exists():
        sys.exit("no index yet: run `research.py fetch`, then `research.py index`")
    db = sqlite3.connect(INDEX)
    hits = db.execute(
        "SELECT key, page, title, snippet(pages, 3, '[', ']', ' … ', 24) FROM pages "
        "WHERE pages MATCH ? ORDER BY rank LIMIT ?",
        (query, n),
    ).fetchall()
    for key, page, title, snip in hits:
        print(f"{title[:80]}\n  {key}  p. {page}\n  {' '.join(snip.split())}\n")
    if not hits:
        print('no matches (FTS5 syntax: words, "exact phrase", a OR b, prefix*)')


def status() -> None:
    rows = load_manifest()
    for state in ("downloaded", "manual", "no pdf", "paywalled"):
        group = [r for r in rows.values() if r.get("status") == state]
        if group:
            print(f"\n{state} ({len(group)})")
            for r in sorted(group, key=lambda r: r["key"]):
                print(f"  {r['key']:<40} {r.get('title', '')[:70]}")
    missing = [r for r in rows.values() if r.get("status") in ("no pdf", "paywalled")]
    if missing:
        print(f"\nAdd missing PDFs to {MANUAL.relative_to(ROOT)}/<key>.pdf, then run `index`.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--force", action="store_true", help="download again")
    sub.add_parser("index")
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-n", type=int, default=10)
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.force)
    elif a.cmd == "index":
        index()
    elif a.cmd == "search":
        search(a.query, a.n)
    else:
        status()


if __name__ == "__main__":
    main()
