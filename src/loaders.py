import json
import re
import time
import urllib.request
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import tiktoken
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

TOKENIZER = tiktoken.get_encoding("cl100k_base")

CORPUS_SOURCES = {
    "fastapi": {
        "sitemap": "https://fastapi.tiangolo.com/sitemap.xml",
        "base_url": "https://fastapi.tiangolo.com/",
        "raw_dir": RAW_DIR / "fastapi",
        "processed_file": PROCESSED_DIR / "fastapi.jsonl",
    },
    "pytorch": {
        "sitemap": "https://pytorch.org/tutorials/sitemap.xml",
        "base_url": "https://docs.pytorch.org/tutorials/",
        "raw_dir": RAW_DIR / "pytorch",
        "processed_file": PROCESSED_DIR / "pytorch.jsonl",
    },
}

HEADERS = {"User-Agent": "RAGLab-Corpus-Collector/1.0 (academic project)"}

MAX_PAGES_PER_SOURCE = 80


def _slug(url: str) -> str:
    path = urlparse(url).path.strip("/").replace("/", "__") or "index"
    path = re.sub(r"[^\w\-]", "_", path)
    return path[:120]


def _token_count(text: str) -> int:
    return len(TOKENIZER.encode(text, disallowed_special=()))


def _word_count(text: str) -> int:
    return len(text.split())


def _get_sitemap_urls(sitemap_url: str, base_url: str) -> list[str]:
    try:
        resp = requests.get(sitemap_url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml-xml")
        urls = [loc.get_text(strip=True) for loc in soup.find_all("loc")]
        filtered = [u for u in urls if u.startswith(base_url)]
        return filtered
    except Exception as e:
        print(f"  [warn] sitemap fetch failed ({e}), falling back to base URL only")
        return [base_url]


def _extract_page(url: str, raw_path: Path) -> dict | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        html = resp.text
        raw_path.write_text(html, encoding="utf-8")
    except Exception as e:
        print(f"  [skip] {url} — {e}")
        return None

    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "head",
                     "aside", ".sidebar", ".toc", ".announcement"]):
        tag.decompose()

    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find(class_=re.compile(r"(content|doc|page)", re.I))
        or soup.body
    )
    if main is None:
        return None

    sections = []
    current_title = "Introduction"
    current_lines = []

    for el in main.find_all(["h1", "h2", "h3", "p", "li", "pre", "code", "blockquote"]):
        if el.name in ("h1", "h2", "h3"):
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append({"section_title": current_title, "text": body})
            current_title = el.get_text(" ", strip=True)
            current_lines = []
        else:
            text = el.get_text(" ", strip=True)
            if text:
                current_lines.append(text)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append({"section_title": current_title, "text": body})

    return sections if sections else None


def collect_source(source_name: str, max_pages: int = MAX_PAGES_PER_SOURCE) -> list[dict]:
    cfg = CORPUS_SOURCES[source_name]
    raw_dir: Path = cfg["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{source_name}] Fetching sitemap...")
    urls = _get_sitemap_urls(cfg["sitemap"], cfg["base_url"])
    urls = urls[:max_pages]
    print(f"[{source_name}] {len(urls)} pages to fetch (capped at {max_pages})")

    passages = []
    for i, url in enumerate(urls, 1):
        slug = _slug(url)
        raw_path = raw_dir / f"{slug}.html"

        if raw_path.exists():
            html = raw_path.read_text(encoding="utf-8")
            sections = _extract_page_from_html(html, url)
        else:
            print(f"  [{i}/{len(urls)}] {url}")
            sections = _extract_page(url, raw_path)
            time.sleep(0.3)

        if not sections:
            continue

        for sec in sections:
            text = sec["text"]
            if len(text.strip()) < 40:
                continue
            passages.append({
                "source": source_name,
                "url": url,
                "file": raw_path.name,
                "section": sec["section_title"],
                "text": text,
                "word_count": _word_count(text),
                "token_count": _token_count(text),
            })

    return passages


def _extract_page_from_html(html: str, url: str) -> list | None:
    import tempfile, os
    tmp = Path(tempfile.mktemp(suffix=".html"))
    tmp.write_text(html, encoding="utf-8")
    result = _extract_page(url, tmp)
    tmp.unlink(missing_ok=True)
    return result


def save_processed(passages: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  Saved {len(passages)} passages -> {out_path.name}")


def compute_stats(passages: list[dict], source_name: str) -> dict:
    sections = {p["section"] for p in passages}
    files = {p["file"] for p in passages}
    total_words = sum(p["word_count"] for p in passages)
    total_tokens = sum(p["token_count"] for p in passages)
    avg_tokens = total_tokens / len(passages) if passages else 0

    stats = {
        "source": source_name,
        "num_pages": len(files),
        "num_passages": len(passages),
        "num_sections": len(sections),
        "total_words": total_words,
        "total_tokens": total_tokens,
        "avg_tokens_per_passage": round(avg_tokens, 1),
    }
    return stats


def load_processed(source_name: str) -> list[dict]:
    path = CORPUS_SOURCES[source_name]["processed_file"]
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    passages = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                passages.append(json.loads(line))
    return passages


def load_all_passages() -> list[dict]:
    all_passages = []
    for source_name in CORPUS_SOURCES:
        try:
            passages = load_processed(source_name)
            all_passages.extend(passages)
        except FileNotFoundError:
            print(f"[warn] {source_name} not yet processed — run collect_corpus.py first")
    return all_passages


if __name__ == "__main__":
    all_passages = []
    all_stats = []

    for source_name in CORPUS_SOURCES:
        passages = collect_source(source_name)
        save_processed(passages, CORPUS_SOURCES[source_name]["processed_file"])
        stats = compute_stats(passages, source_name)
        all_stats.append(stats)
        all_passages.extend(passages)
        print(f"\n[{source_name}] stats: {stats}")

    combined_stats = {
        "sources": all_stats,
        "combined": {
            "total_pages": sum(s["num_pages"] for s in all_stats),
            "total_passages": sum(s["num_passages"] for s in all_stats),
            "total_sections": sum(s["num_sections"] for s in all_stats),
            "total_words": sum(s["total_words"] for s in all_stats),
            "total_tokens": sum(s["total_tokens"] for s in all_stats),
        },
    }

    stats_path = ROOT / "data" / "corpus_stats.json"
    stats_path.write_text(json.dumps(combined_stats, indent=2), encoding="utf-8")
    print(f"\nCorpus stats saved → {stats_path}")
    print(json.dumps(combined_stats["combined"], indent=2))
